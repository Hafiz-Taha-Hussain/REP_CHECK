"""
FastAPI wrapper for the squat form checker.

Assumes rep_counter.py, compute_angles.py, thresholds.py, and this file
all live in the same folder.

Design decisions (see project chat for full reasoning):
- Job-ID / polling, not a synchronous response - rep_counter.run() can
  take well over the ~30s timeout most hosting proxies allow.
- Concurrency capped at exactly 1 - a single background worker thread
  consumes a bounded queue, so only one video is processed at a time.
  Free-tier hosts give you a shared single vCPU anyway.
- Bounded queue (MAX_QUEUE_SIZE) - new submissions are rejected with
  503 once full, instead of accepting unlimited jobs and OOM-ing later.
- TTL eviction - finished job metadata AND the video/report/annotated-
  video files on disk are deleted after JOB_TTL_SECONDS.
- rep_counter.run() is called directly (imported, not subprocess) -
  it's a real function that returns the report dict, so there's no
  reason to pay for a second process + a second MediaPipe model load.
  It's still called from a background worker THREAD, not inline in the
  request handler, because it's a long-running blocking call and must
  not block the event loop that serves /status and /analyze.
- rep_counter.run() writes its annotated video with OpenCV's `mp4v`
  fourcc, which browsers' <video> decoders generally can't play
  (they expect H.264/VP9/AV1). So after run() succeeds, the worker
  shells out to `ffmpeg` to transcode that file to browser-safe H.264
  before marking the job done. This REQUIRES ffmpeg to be installed
  and on PATH -- locally on Windows (e.g. via `winget install ffmpeg`,
  confirm with `ffmpeg -version`), and in the Dockerfile later
  (`apt-get install -y ffmpeg`).
- Errors are split into two layers: the real exception (with full
  traceback, including server file paths) is always logged server-side
  via `logger.exception(...)`; only a short, curated, non-leaky message
  (_user_facing_error) is ever stored in job["error"] and shown to the
  client. A stuck pipeline call is soft-timed-out after
  PROCESSING_TIMEOUT_SECONDS so a hung job can't block the single worker
  forever -- see _run_pipeline_with_timeout for the real limitation this
  doesn't solve (the abandoned thread itself can't be forcibly killed).
  A RequestValidationError handler flattens FastAPI/Pydantic's default
  error shape to a plain string, so every error path -- ours and the
  framework's -- returns {"detail": "<string>"} consistently.
"""

import logging
import os
import subprocess
import threading
import time
import uuid
import queue
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from rep_counter import run as run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("repcheck")

# ---- Config ----
MAX_QUEUE_SIZE = 10
JOB_TTL_SECONDS = 3600  # 1 hour
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")

# If a single job's pipeline call runs longer than this, stop waiting on it
# and report a timeout instead of leaving the job (and the one worker
# behind it) stuck in "processing" forever. See the comment above
# _run_pipeline_with_timeout for the real limitation this doesn't solve.
PROCESSING_TIMEOUT_SECONDS = 420  # 7 min

# "ffmpeg" alone works once it's on PATH (true inside the Docker image,
# where apt-get installs it there). Locally on Windows, if winget didn't
# register it on PATH, set FFMPEG_BIN to the full ffmpeg.exe path instead
# of fighting PATH further:
#   $env:FFMPEG_BIN = "C:\Users\<you>\AppData\Local\Microsoft\WinGet\Packages\...\ffmpeg.exe"
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
# ffprobe ships in the same bin/ folder as ffmpeg -- same override pattern.
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")

# Measured locally: BOTH pose-extraction memory (python.exe) and transcode
# memory (ffmpeg) scale with FRAME RESOLUTION, not upload file size -- a
# 15.6MB 4K clip hit far higher peaks (and took 180s vs 55s for a similar
# 1080p clip) than a 5.4MB 1080p clip. Resolution drives per-frame buffer
# size in both MediaPipe/OpenCV and libx264's encoder; file size is a poor
# proxy for either. Checked as total pixel count, not a single width or
# height number -- a portrait phone clip and a landscape clip at "the
# same" resolution report width/height swapped, so checking only one
# dimension is orientation-dependent and unreliable.
MAX_PROCESSING_PIXELS = 1920 * 1080  # ~1080p budget, either orientation
# Within an accepted upload, still downscale down to this width before
# pose extraction -- single source of truth shared by downscale_if_needed
# and transcode_for_web, so an accepted-but-heavy 1080p clip still gets
# the memory benefit even though it's under the reject ceiling above.
MAX_PROCESSING_WIDTH = 1280

# Resolution is now separately capped (MAX_PROCESSING_PIXELS, checked at
# upload time below) -- so this is no longer standing in as a proxy for
# resolution. What's left for it to guard against is mostly clip DURATION
# (a long, low-res clip could still be slow/memory-heavy over many
# frames) and just sane upload sizes generally. 10MB comfortably covers a
# normal 1080p squat clip (~30-60s) at reasonable bitrate.
MAX_VIDEO_SIZE_MB = 10
MAX_VIDEO_SIZE_BYTES = MAX_VIDEO_SIZE_MB * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"video/mp4"}
ALLOWED_EXTENSIONS = {".mp4"}

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


class ViewOption(str, Enum):
    front = "front"
    side = "side"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # job_store always starts empty on process boot (it's in-memory only),
    # so anything already sitting in uploads/ or outputs/ at this point is
    # guaranteed orphaned from a previous run -- crash, redeploy, or a dev
    # --reload restart. evict_expired_jobs() can't clean these because it
    # only acts on entries still in job_store; wipe both folders once here
    # instead of letting them accumulate across restarts forever.
    for d in (UPLOAD_DIR, OUTPUT_DIR):
        for f in d.iterdir():
            if f.is_file():
                f.unlink(missing_ok=True)
    yield


app = FastAPI(title="Squat Form Checker", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # FastAPI's default shape for this is a JSON array of {loc, msg, type}
    # objects, not a plain string -- every other error path in this app
    # returns {"detail": "<plain string>"}, and the frontend's error
    # handling assumes that shape. Flatten to match, so a bad request
    # (e.g. an invalid `view` sent directly to the API, bypassing our own
    # UI's toggle) renders as a real message instead of "[object Object]".
    messages = []
    for err in exc.errors():
        loc = " -> ".join(str(x) for x in err["loc"] if x != "body")
        messages.append(f"{loc}: {err['msg']}" if loc else err["msg"])
    return JSONResponse(status_code=422, content={"detail": "; ".join(messages)})


@app.get("/")
async def index():
    return FileResponse(Path("index.html"))

# ---- Job store ----
# job_id -> {status, created_at, view, video_path, report_path,
#            output_video_path, report, error, pending_delete?}
# video_path is cleared to None once a job finishes successfully (the
# original upload is redundant once the annotated output exists).
# pending_delete is set by request_job_deletion() when a delete comes in
# for a still-processing job; the worker checks it and cleans up itself
# once done, instead of the request racing an in-progress file write.
job_store: dict[str, dict] = {}
job_lock = threading.Lock()  # protects job_store from the worker thread
work_queue: "queue.Queue[str]" = queue.Queue(maxsize=MAX_QUEUE_SIZE)


def probe_video_resolution(path: Path) -> tuple[int, int]:
    """(width, height) of the video's first stream, via ffprobe (ships
    alongside ffmpeg -- same FFMPEG_BIN/FFPROBE_BIN override pattern)."""
    cmd = [
        FFPROBE_BIN, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Could not run ffprobe at '{FFPROBE_BIN}' -- required to check "
            "video resolution before processing."
        ) from e
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"ffprobe could not read video info: {result.stderr[-500:]}")
    width_str, height_str = result.stdout.strip().split("x")
    return int(width_str), int(height_str)


def downscale_if_needed(input_path: Path, output_path: Path) -> Path:
    """If input_path is wider than MAX_PROCESSING_WIDTH, downscale it once
    up front, BEFORE pose extraction ever sees it -- run_pipeline() reads
    the original upload directly, so capping only the transcode's output
    (below) doesn't help here. Uploads are already capped at
    MAX_PROCESSING_PIXELS by /analyze, so this only ever has modest work
    to do (e.g. an accepted 1080p clip shrinking to MAX_PROCESSING_WIDTH),
    never a drastic 4K-scale jump.

    Returns input_path unchanged if no downscale was needed (skips a
    pointless re-encode for the common case), or output_path if it did.
    """
    width, _height = probe_video_resolution(input_path)
    if width <= MAX_PROCESSING_WIDTH:
        return input_path

    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(input_path),
        "-vf", f"scale='min({MAX_PROCESSING_WIDTH},iw)':-2",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-threads", "1",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=420)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Could not run ffmpeg at '{FFMPEG_BIN}' -- required to "
            "downscale this video before processing."
        ) from e
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg pre-downscale failed: {result.stderr[-1000:]}")
    return output_path


def transcode_for_web(input_path: Path, output_path: Path) -> None:
    """Re-encode input_path (whatever OpenCV produced) to browser-playable
    H.264/yuv420p, with +faststart so the browser can start playing before
    the whole file has downloaded. Raises RuntimeError with a clear message
    if ffmpeg isn't installed or the transcode fails -- this is surfaced to
    the client as the job's error, not swallowed.

    The -vf scale here is now mostly defense-in-depth (a no-op in the
    common case) since downscale_if_needed above already bounds the input
    run_pipeline() worked from -- kept anyway, cheap when it's a no-op,
    and it's still the thing bounding libx264's lookahead buffer directly.
      - preset veryfast -- shallower encoder lookahead than the default
        "medium" preset, trading a bit of compression efficiency (doesn't
        matter for a short demo clip) for meaningfully less RAM.
      - threads 1 -- confirmed necessary by testing, not preemptive: on
        Render's free tier (0.1 CPU -- a HARD quota, not a soft share),
        libx264's default multi-threaded encoding has no real parallelism
        to exploit and just adds scheduling overhead fighting over that
        tiny slice. Single-threaded avoids that overhead entirely. This
        does NOT fix the underlying compute ceiling -- 0.1 CPU is still
        0.1 CPU -- it only removes self-inflicted inefficiency on top of it.
    """
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(input_path),
        "-vf", f"scale='min({MAX_PROCESSING_WIDTH},iw)':-2",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-threads", "1",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=420)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Could not run ffmpeg at '{FFMPEG_BIN}' -- required to produce "
            "a browser-playable video. Check it's installed and either on "
            "PATH or pointed to via the FFMPEG_BIN environment variable."
        ) from e
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg transcode failed: {result.stderr[-1000:]}")


def _run_pipeline_with_timeout(video_path, report_path, view, output_video_path):
    """Runs run_pipeline() with a hard wait limit. If it doesn't finish in
    PROCESSING_TIMEOUT_SECONDS, this function returns control (raising
    TimeoutError) so the job can be marked failed and the worker can move
    on to the next queued job -- instead of blocking forever.

    Known, deliberate limitation: Python threads can't be forcibly killed.
    On timeout the abandoned thread keeps running in the background,
    consuming CPU/RAM, until the pipeline call naturally finishes or
    errors on its own. This is a soft timeout (stop waiting, free up the
    worker and give the user an answer) -- not a hard kill. A true hard
    kill needs process-level isolation (multiprocessing.Process +
    .terminate(), a real task queue, etc.), which is more moving parts
    than this project's scale currently justifies. Worth revisiting if
    stuck jobs turn out to be a real, recurring problem in practice.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        run_pipeline, video_path,
        report_path=report_path, view=view, output_video_path=output_video_path,
    )
    try:
        return future.result(timeout=PROCESSING_TIMEOUT_SECONDS)
    except FutureTimeoutError:
        raise TimeoutError(
            f"Processing took longer than {PROCESSING_TIMEOUT_SECONDS}s and "
            "was abandoned. Try a shorter clip."
        )
    finally:
        executor.shutdown(wait=False)  # never block here waiting on a runaway thread


def _user_facing_error(exc: Exception) -> str:
    """Map an internal exception to a short, honest, non-leaky message for
    the client. The real exception (with traceback) is always logged
    server-side separately -- this is only about what the user sees."""
    if isinstance(exc, TimeoutError):
        return str(exc)  # already written to be user-facing, no internal detail
    if isinstance(exc, (IOError, OSError)):
        return (
            "Couldn't open that video. It may be corrupted or use an "
            "unsupported codec -- try re-exporting it as a standard H.264 MP4."
        )
    if isinstance(exc, RuntimeError) and ("ffmpeg" in str(exc).lower() or "ffprobe" in str(exc).lower()):
        return "Couldn't prepare that video for analysis. Try a different clip."
    return "Something went wrong while processing your video. Try a different clip."


def _delete_job_files(job: dict) -> None:
    for key in ("video_path", "output_video_path", "report_path"):
        p = job.get(key)
        if p and Path(p).exists():
            Path(p).unlink(missing_ok=True)


def evict_expired_jobs() -> None:
    """Delete job metadata + associated files older than JOB_TTL_SECONDS.
    Backstop for anything the explicit-delete path below doesn't catch:
    a crashed tab, a forgotten browser window that never refreshes, etc."""
    now = time.time()
    with job_lock:
        expired = [
            jid for jid, job in job_store.items()
            if now - job["created_at"] > JOB_TTL_SECONDS
        ]
        removed = [job_store.pop(jid) for jid in expired]
    for job in removed:
        _delete_job_files(job)


def request_job_deletion(job_id: str) -> str:
    """Called from DELETE /jobs/{id} -- either the explicit 'check another
    clip' button, or a best-effort call fired as the page unloads/refreshes.
    Returns what happened:
      'deleted'   - job was queued/done/error, safe to remove right away
      'deferred'  - job is still processing; the worker is actively reading/
                    writing its files, so deletion is flagged and carried
                    out by the worker itself the moment it finishes, rather
                    than racing a file that's mid-read/write
      'not_found' - already gone (expired, already deleted, or never existed)
    """
    with job_lock:
        job = job_store.get(job_id)
        if job is None:
            return "not_found"
        if job["status"] == "processing":
            job["pending_delete"] = True
            return "deferred"
        job_store.pop(job_id, None)
    _delete_job_files(job)
    return "deleted"


def worker_loop() -> None:
    """Single long-running worker -> concurrency of exactly 1."""
    while True:
        job_id = work_queue.get()
        with job_lock:
            job = job_store.get(job_id)
            if job is None:  # evicted before it got picked up
                work_queue.task_done()
                continue
            job["status"] = "processing"
            video_path = job["video_path"]
            view = job["view"]

        report_path = OUTPUT_DIR / f"{job_id}_report.json"
        downscaled_path = OUTPUT_DIR / f"{job_id}_source.mp4"   # only created if the upload needs resizing
        raw_video_path = OUTPUT_DIR / f"{job_id}_raw.mp4"       # OpenCV's output, not browser-safe
        web_video_path = OUTPUT_DIR / f"{job_id}_web.mp4"       # transcoded, served to clients

        try:
            pipeline_input = downscale_if_needed(Path(video_path), downscaled_path)

            report = _run_pipeline_with_timeout(
                str(pipeline_input),
                report_path=str(report_path),
                view=view,
                output_video_path=str(raw_video_path),
            )
            # Recorded the moment the file actually exists on disk, not only
            # once every later step (transcode) also succeeds -- otherwise a
            # transcode failure here would leave this report.json orphaned:
            # written to disk, but untracked by any cleanup path (TTL sweep,
            # explicit delete) until the next process restart wipes it.
            with job_lock:
                job["report_path"] = str(report_path)

            transcode_for_web(raw_video_path, web_video_path)
            raw_video_path.unlink(missing_ok=True)  # keep only the web-playable copy
            if pipeline_input != Path(video_path):
                pipeline_input.unlink(missing_ok=True)  # drop the intermediate downscaled copy

            with job_lock:
                job["status"] = "done"
                job["report"] = report
                job["output_video_path"] = str(web_video_path)
                # The original upload is fully redundant once we have the
                # annotated output -- drop it now instead of waiting for TTL.
                Path(video_path).unlink(missing_ok=True)
                job["video_path"] = None
        except Exception as e:  # noqa: BLE001 - surface any pipeline/transcode failure to the client
            # Full detail (path, traceback) goes to the server log only --
            # never straight to the client, which is both unhelpful and a
            # minor info leak (server file paths in a raw exception string).
            logger.exception(f"Job {job_id} failed")
            with job_lock:
                job["status"] = "error"
                job["error"] = _user_facing_error(e)
                # Deliberately KEPT on error (unlike the success path above)
                # -- useful for debugging what the pipeline choked on. It
                # still gets cleaned up normally, just later: by TTL, or by
                # the pending_delete check right below if the user has
                # already moved on.
            raw_video_path.unlink(missing_ok=True)
            downscaled_path.unlink(missing_ok=True)  # harmless no-op if it was never created
        finally:
            with job_lock:
                still_marked = job_store.get(job_id) is job
                pending_delete = still_marked and job.pop("pending_delete", False)
                if pending_delete:
                    job_store.pop(job_id, None)
            if pending_delete:
                _delete_job_files(job)
            work_queue.task_done()


# Start the single worker thread once, when the app process starts.
threading.Thread(target=worker_loop, daemon=True).start()


# ---- Routes ----

@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    view: ViewOption = Form(ViewOption.front),
):
    # --- Validate file type before touching disk ---
    # Checked two ways deliberately: extension is what the filename claims,
    # content_type is what the browser/client declared in the request --
    # neither alone is trustworthy (a renamed file, or a client that sends
    # a generic content_type), so both have to agree.
    ext = Path(video.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS or video.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            415,
            f"Only .mp4 video files are accepted "
            f"(got filename='{video.filename}', content_type='{video.content_type}').",
        )

    evict_expired_jobs()

    if work_queue.full():
        raise HTTPException(
            503, "Server is busy processing other videos. Try again shortly."
        )

    job_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{job_id}.mp4"

    # --- Enforce the size cap while streaming, not after buffering the
    # whole upload --- rejecting only once the full file is already on
    # disk (or in memory) defeats the point of having a cap at all.
    size_so_far = 0
    try:
        with video_path.open("wb") as f:
            while chunk := await video.read(1024 * 1024):
                size_so_far += len(chunk)
                if size_so_far > MAX_VIDEO_SIZE_BYTES:
                    raise HTTPException(
                        413,
                        f"Video exceeds the {MAX_VIDEO_SIZE_MB}MB limit "
                        f"(stopped after {size_so_far // (1024 * 1024)}MB).",
                    )
                f.write(chunk)
    except HTTPException:
        video_path.unlink(missing_ok=True)  # don't leave a partial file behind
        raise

    # --- Reject absurdly high-resolution uploads outright ---
    # Rather than trying to gracefully accommodate any resolution: decoding
    # cost at the pre-downscale step is unavoidable regardless of how much
    # we shrink the output afterward (you can't scale a frame you haven't
    # decoded yet), and a genuine 4K clip measured 3x slower to process
    # (180s vs 55s) than 1080p, tying up the single worker -- and the queue
    # behind it -- for that whole time. A squat clip has no real reason to
    # be above 1080p.
    try:
        width, height = probe_video_resolution(video_path)
    except RuntimeError:
        video_path.unlink(missing_ok=True)
        raise HTTPException(
            422,
            "Couldn't read that video's resolution -- it may be corrupted "
            "or not a valid video file.",
        )
    if width * height > MAX_PROCESSING_PIXELS:
        video_path.unlink(missing_ok=True)
        raise HTTPException(
            422,
            f"That video's resolution ({width}x{height}) is above the "
            "1080p limit this app supports. Re-export it at 1080p or "
            "lower and try again.",
        )

    with job_lock:
        job_store[job_id] = {
            "status": "queued",
            "created_at": time.time(),
            "view": view.value,
            "video_path": str(video_path),
            "report_path": None,
            "output_video_path": None,
            "report": None,
            "error": None,
        }

    work_queue.put(job_id)
    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
async def status(job_id: str):
    evict_expired_jobs()
    with job_lock:
        job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found (completed jobs expire after "
                                  f"{JOB_TTL_SECONDS}s)")
    return {"job_id": job_id, "status": job["status"], "error": job["error"]}


@app.get("/result/{job_id}")
async def result(job_id: str):
    with job_lock:
        job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found (may have expired)")
    if job["status"] != "done":
        raise HTTPException(409, f"job not finished yet (status: {job['status']})")
    # run_pipeline() already returns the report dict, so serve it straight
    # from memory rather than re-reading the file we also happen to have.
    return job["report"]


@app.get("/result/{job_id}/video")
async def result_video(job_id: str):
    with job_lock:
        job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found (may have expired)")
    if job["status"] != "done":
        raise HTTPException(409, f"job not finished yet (status: {job['status']})")
    return FileResponse(Path(job["output_video_path"]), media_type="video/mp4")


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Called explicitly ('check another clip') or as a best-effort
    fire-on-unload request when the page refreshes/closes. Idempotent --
    calling it on an already-gone job just returns 'not_found', not an
    error, since that's an entirely normal case for a beacon-style call."""
    outcome = request_job_deletion(job_id)
    return {"job_id": job_id, "outcome": outcome}