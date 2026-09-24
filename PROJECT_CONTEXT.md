# Gym Exercise Form Checker — Project Context (Phase 1 & 2 Complete)

## What this project is

A CV/MLOps portfolio project built to fill a specific CV gap: existing work
(FlyRank ML internship capstone) demonstrates modeling skill but nothing on
shipping/deployment. This project's differentiator is **the deployment
pipeline**, not novel CV algorithms — the pose-estimation-plus-angle-rules
approach is a well-known pattern; what's uncommon at this career stage is
doing it with evidence-based calibration, honest documentation of what
doesn't work, and a real deployed endpoint.

**Product scope:** upload a squat video → get back an annotated video
(skeleton overlay + live verdict HUD) + a JSON report (rep count, per-rep
pass/fail per check). Deliberately NOT live webcam streaming — that was
scoped out early as unnecessary deployment complexity for this stage.

**Tech stack:** Python, OpenCV, MediaPipe (Tasks API), FastAPI, Docker,
deployed on Render.

**Original 5-week plan:** Weeks 1-4 = local pipeline (pose extraction →
angle math/rules → rep counting → output generation). Week 5 = deployment
(FastAPI, Docker, CI, hosting, monitoring, README). **Both phases are now
complete.** This doc covers everything built and learned in Phase 1, plus a
closing section on Phase 2 (deployment) — see the bottom of this file.

---

## Pipeline architecture (current, working)

Single entry point: **`rep_counter.py`** does pose extraction, smoothing,
rep detection, scoring, and (optionally) annotated video output — all in
one pass over the video. No separate extraction step needed for normal use.

```
python rep_counter.py <video.mp4> --view {front|side} \
    --report output\report.json \
    --output-video output\annotated.mp4
```

### File-by-file

- **`rep_counter.py`** — Main pipeline. Contains:
  - `RepCounter` class: hysteresis state machine on knee angle
    (STANDING_ANGLE=155°, IN_SQUAT_ANGLE=140°) that detects each rep as it
    completes and scores it via `_score_rep()`, which is view-aware (reads
    `thresholds.VIEWS[view]` to decide which checks run).
  - Skeleton drawing (`draw_skeleton`) + HUD overlay (`draw_hud`) for
    annotated video output — rep count, view, last rep's verdict, failed
    check names, color-coded pass(green)/fail(red).
  - Uses MediaPipe Tasks API (`mp.tasks.vision.PoseLandmarker`), **`heavy`**
    model variant (upgraded from `lite` → `full` → `heavy` over the
    session as tracking-quality issues came up).
  - CLI flags: `video_path` (positional), `--report`, `--view`
    (front/side, default front), `--output-video`.

- **`compute_angles.py`** — Library of angle-math functions, imported by
  everything else, not run directly:
  - `knee_angle(lm, side)` — hip-knee-ankle angle (depth signal).
  - `knee_alignment(lm, side)` — knee x-position relative to ankle x,
    normalized by hip width (valgus/cave signal). Has a guard:
    `MIN_HIP_WIDTH = 0.05` — returns `None` if hip width is too small to
    trust the ratio (near-zero denominator). **Known limitation:** this
    guard is an absolute threshold, not scale-invariant — it can
    incorrectly reject frames where hips are confidently tracked but
    naturally appear narrow due to camera distance/framing. Not yet fixed
    (flagged as a real, understood bug — see Open Issues below).
  - `back_angle(lm, side)` — shoulder-hip line angle from vertical (lean
    signal).
  - `shin_angle(lm, side)` / `relative_lean(lm, side)` — an attempted
    fix for back_angle (see "Things that were tried and failed" below).
    Still in the file but **not used** by the live pipeline.
  - `visible(lm, name)` — visibility gate helper, `VISIBILITY_THRESHOLD =
    0.5`.
  - `LandmarkSmoother` class — rolling 5-frame average on x/y/z per
    landmark, applied before any angle math. Visibility is deliberately
    NOT smoothed (it's a confidence signal, not a spatial coordinate;
    smoothing it would hide real detection dropouts).

- **`thresholds.py`** — All calibrated thresholds, organized per camera
  view. This file has extensive inline comments documenting the evidence
  behind (or against) each threshold — read it directly for full
  reasoning. Structure:
  ```python
  DEPTH_THRESHOLD_DEG = 65          # both views (see Phase 2 note below)
  BACK_ANGLE_MAX_DEG_SIDE = 75      # side view only, provisional
  KNEE_ALIGNMENT_MIN = -1.85        # front view only, provisional
  KNEE_ALIGNMENT_MAX = 0.85
  VIEWS = {
      "front": {"depth_enabled": True, "back_angle_enabled": False,
                 "knee_alignment_enabled": True},
      "side":  {"depth_enabled": True, "back_angle_enabled": True,
                 "knee_alignment_enabled": False},
  }
  ```

- **`extract_landmarks.py`** — Standalone diagnostic tool. Extracts raw
  landmarks from a video to a `landmarks.json` file. Not part of the live
  pipeline anymore (rep_counter.py does extraction inline), but useful for
  manual debugging.

- **`rep_windows.py`** — Standalone diagnostic tool. Takes a
  `landmarks.json`, isolates just the real rep frames (cuts standing/setup
  frames), prints angle stats limited to that window. Used historically to
  calibrate thresholds properly instead of guessing across whole clips.
  Its core logic is now duplicated inside `RepCounter` for live use.

- **`check_landmarks.py`** — Standalone diagnostic tool. Sanity-checks a
  `landmarks.json` for visibility/detection quality per joint.

- **`batch_process.py`** — Batch-processes a folder of videos into one CSV
  of per-rep metrics (depth, back_angle, alignment, relative_lean), with
  good/bad label inferred from folder path. Used to test calibration
  against the external Mendeley multi-subject dataset (see below). Uses a
  **fresh PoseLandmarker instance per video** (required — MediaPipe VIDEO
  mode needs strictly increasing timestamps per landmarker instance;
  reusing one across videos crashes on video #2 since its timestamp resets
  to 0).

---

## Personal dataset (Weeks 1-2)

6 self-recorded squat clips, front-facing camera:
- **Clips 1, 3, 4:** clean tracking (3 and 4 needed reshoots — camera
  framing/occlusion issues on first attempts).
- **Clip 5:** mildly degraded but usable.
- **Clip 2:** kept **deliberately** as a low-visibility/unreliable-tracking
  test case (left knee/ankle visibility genuinely low for real portions of
  the clip) — used throughout to validate that the pipeline correctly
  reports `insufficient_data` rather than guessing, when tracking is
  actually unreliable.
- **Clip 6:** a deliberate "bad knee alignment" test clip, recorded later.
  Result was inconclusive — landed inside the "good" band rather than
  outside it — but the test was confounded (depth didn't match the good
  clips, left/right knee angles diverged sharply suggesting asymmetric
  execution). Not treated as disproof of the alignment metric, but also
  never resolved. Open issue.

---

## External dataset (used for real calibration testing)

**Source:** Mendeley "A Multi-View Raw Video Dataset of Seven Fitness
Exercises with Good/Bad Form Labels" — 26 subjects, 7 exercises,
good/bad labeled, 3 camera views (front/side/diagonal). DOI:
10.17632/kgbb3yn47p.2. CC BY 4.0 licensed — usable in a public repo with
attribution (added to `README.md`).

**Important limitation discovered by direct visual inspection:** this
dataset's good/bad label is a **flat binary tag that mixes multiple
unrelated error types together**. Manually inspected 3 "bad" squat clips
frame-by-frame and found each demonstrated a *different* mistake:
- subject_001: partial depth + visible knee cave (valgus)
- subject_005: minimal depth, upright torso — a depth-only problem
- subject_020: deep squat but knees splayed outward + arms crossed
  (wrong stance/arm position, not valgus)

**Consequence:** you cannot calibrate a single metric's threshold by
comparing it against the aggregate "bad" label — averaging across bad reps
mixes in examples that aren't bad for that metric's reason at all, which
washes out any real signal. This was confirmed empirically: none of
depth/back_angle/alignment showed real good-vs-bad separation when tested
against the aggregate label, even though depth separation exists when
tested against the (curated) personal dataset. **This is a structural
dataset limitation, not evidence the metrics are broken.**

What the dataset WAS useful for: checking the natural **variance among
confirmed-good reps** across many different real subjects — this is what
actually drove the back-angle and alignment threshold revisions below.

---

## Calibration decisions and the evidence behind each

### Depth (knee angle) — ENABLED, both views, threshold = 65° (originally 60°)
- Calibrated from personal clips: good clips bottomed at 18-35°, bad
  clips only reached 82-88°. Standard fitness guidance (~90° = parallel)
  does NOT separate these correctly (82.7 < 90), so the threshold is fit
  to the actual data gap, not textbook guidance.
- Held up reasonably against the 25-subject dataset too (weak but real
  signal — the metric most robust to camera-view/subject variance,
  because leg flexion produces a large, robust angular change that
  dominates noise better than lean or alignment do).
- **Phase 2 update:** manually loosened from 60 to 65° on request. Still
  sits comfortably inside the validated good/bad gap above (good ≤35°,
  bad ≥82°), so this remains a safe change within the evidence already
  gathered — but 65° specifically, unlike 60°, isn't itself backed by an
  independent data point. See `thresholds.py` for the live note.

### Back angle / lean — DISABLED for front view, ENABLED (provisional) for side view
- **Front view, original threshold (35°) FAILED in production.** A stock
  footage clip (`squat_test.mp4`) with genuinely good, deep squat form
  (visually confirmed by inspecting the actual video frame) was
  incorrectly flagged FAIL — because deep squats without external
  counterweight require forward lean to balance, which is correct
  technique, not bad form.
- **Root cause investigation, in order tried:**
  1. Tested against the 25-subject dataset: confirmed-GOOD front-view reps
     ranged 8-90° (median 36°, 90th percentile 73°, 95th percentile 85°)
     — far too much natural inter-subject variance for any flat threshold.
  2. Hypothesized a depth-conditional threshold (deeper squat → more
     expected lean, matching real coaching guidance). Tested: correlation
     between depth and back-angle in the good-rep data was only **-0.12**
     — too weak to support this. Abandoned.
  3. Hypothesized `relative_lean` (back_angle − shin_angle, a real
     biomechanical coaching cue: torso should track shin angle) would be
     more stable, since it's build-independent. **Tested and it made
     things WORSE**, not better: coefficient of variation went from 0.434
     (raw back_angle) to 0.631 (relative_lean) — subtracting two noisy
     angle estimates compounds their noise rather than cancelling it, when
     the two aren't correlated in the right way. Abandoned. (Code for this
     — `shin_angle`, `relative_lean` — is still in `compute_angles.py` but
     unused by the pipeline.)
  4. **Identified the actual root cause: camera plane mismatch.** Forward
     lean is a sagittal-plane motion. A front-facing camera can't observe
     it directly — only indirectly via foreshortening of the shoulder-hip
     vertical distance, which is inherently noisy and confounded by any
     body rotation. This explains why front-view lean measurement failed
     regardless of formula.
  5. **Tested side view directly** (re-ran `batch_process.py` against the
     dataset's side-view good/bad folders): coefficient of variation
     dropped to ~0.26-0.29 — genuinely tighter/more reliable, confirming
     side view is the correct plane for this measurement.
  6. **However, side-view back_angle still didn't separate good/bad**
     (good mean 56.6°, bad mean 52.3° — still overlapping, still backwards
     from expectation) — but this is explained by the same aggregate-label
     mixing problem described above (the "bad" side clips likely have the
     same mixed-error-type issue as the front clips), not a new failure of
     the metric itself.
- **Current state:** front view — `not_scored` (correctly identified as
  unmeasurable from this camera plane, not just under-calibrated).
  Side view — enabled with threshold = 75° (documented in `thresholds.py`
  as the 95th percentile of confirmed-GOOD side-view reps). This is
  explicitly a **loose outlier-flag**, not a validated pass/fail boundary.
  No true bad-lean-specific labeled data exists to validate it further.
- **For context on how much error is normal in this problem class:**
  published sports-science research on the analogous 2D knee-valgus
  metric (Frontal Plane Projection Angle) reports 95% limits of agreement
  of -30° to +17° against real 3D motion capture — i.e., even
  peer-reviewed methodology accepts this much error as an inherent limit
  of single-camera 2D measurement. Not a bar this project needs to beat.

### Knee alignment (valgus) — ENABLED (provisional) for front view, DISABLED for side view
- Calibrated from personal good clips (1, 3) only — their observed
  combined range (-1.8 to +0.8) became the "acceptable" band, widened
  slightly to -1.85/+0.85 after finding a rounding-boundary bug (the
  original threshold used console-rounded display values instead of exact
  computed values, causing clip 3 — one of the clips that DEFINED the
  threshold — to incorrectly fail against its own calibration data; fixed
  by using exact unrounded extremes).
- **Never validated against a true bad-alignment example.** One deliberate
  test clip (clip 6, described above) was inconclusive due to confounded
  variables. This remains an open issue — the check works and has real
  signal (it did catch one deliberately-caved-knee example's outlier
  reading; it also caught real, previously-unnoticed valgus in a "good"
  personal clip — see below), but its precision/recall against genuine
  bad examples is unverified.
- **A genuinely useful finding along the way:** frame-by-frame tracing of
  clip 1 (a personal "good form" clip) showed a real, sustained,
  high-confidence (visibility 0.97-0.99) right-knee valgus pattern that
  tracked the squat descent/ascent precisely — i.e., the metric correctly
  caught a real, subtle form issue in footage the person hadn't noticed
  themselves. This is evidence the underlying signal is real, even though
  the threshold isn't fully validated.
- Disabled for side view: knee alignment is a frontal-plane motion, not
  visible from the side, and structurally the `MIN_HIP_WIDTH` guard would
  reject nearly every side-view frame anyway (hips nearly stack in x from
  a side profile) — so there's nothing to score.

---

## Model and tracking-quality history

- Started with MediaPipe's legacy `mp.solutions.pose` API. **This API was
  removed in current MediaPipe releases** (confirmed via GitHub issue
  research) — had to rewrite extraction using the current Tasks API
  (`mp.tasks.vision.PoseLandmarker`), including auto-downloading the
  `.task` model file and manually implementing skeleton drawing (the old
  `mp.solutions.drawing_utils` is also gone).
- Model tier progression: `lite` (initial, fastest/least accurate) → `full`
  (mid-session, after user reported visible landmark jitter/overlap) →
  `heavy` (most accurate tier, current). Each swap is a one-line change
  (`MODEL_PATH`/`MODEL_URL` constants) — same Tasks API, no other code
  changes needed.
- Added `LandmarkSmoother` (rolling 5-frame average) at the same time as
  the model upgrade, hypothesizing that ratio/difference-based metrics
  (knee_alignment divides by hip-width; relative_lean subtracts two
  angles) are more sensitive to raw jitter than single-angle metrics like
  depth — consistent with which metrics actually struggled most this
  session.

---

## Things that were tried and explicitly failed (don't retry without new evidence)

1. **`relative_lean` (back_angle − shin_angle)** as a fix for front-view
   lean measurement. Made variance worse (0.434 → 0.631 CV), not better.
   Root cause: subtracting two noisy angle estimates compounds noise.
2. **Depth-conditional back-angle threshold** (deeper squat → allow more
   lean). Correlation between depth and lean in real good-rep data was
   only -0.12 — too weak to build a rule on.
3. **Trusting the external dataset's aggregate "bad" label** for
   per-metric threshold calibration. Structurally invalid — visually
   confirmed the label mixes unrelated error types per clip.
4. **A flat, single back-angle threshold from only 2 personal clips.**
   Looked clean on 2 examples, failed hard on real stock footage and on
   the 25-subject dataset. The lesson generalizes: don't trust a threshold
   from a 2-example calibration set, even if it looks clean, without
   testing against independent data.

---

## Open issues (known, unresolved, real)

1. **`MIN_HIP_WIDTH` guard is scale-dependent, not scale-invariant.**
   Found by tracing clip 2: a stretch of frames had genuinely
   high-confidence hip tracking (visibility 1.0) but hip-width fell below
   the 0.05 absolute cutoff simply because of camera distance/framing, not
   any real occlusion or unreliable tracking. Correct fix: normalize
   hip-width against another body measurement from the same frame (e.g.
   shoulder width) instead of an absolute cutoff, making the guard
   invariant to camera distance/zoom. **Not yet implemented.**
2. **Knee alignment threshold not validated against a true bad-alignment
   example** (see above — clip 6 test was confounded).
3. **Side-view back-angle threshold is an outlier-flag, not a tested
   boundary** — same root limitation as #2 (no dataset with per-error-type
   labels exists).
4. **`BACK_ANGLE_MAX_DEG_SIDE` documentation discrepancy** — this doc
   previously stated the calibrated value as 78°; the actual value in
   `thresholds.py` is 75°. Noted, not yet resolved which is correct.
5. **View selection is a manual CLI/API flag** (`--view front|side` /
   the frontend's toggle), not auto-detected from the video.

---

## Explored but deliberately not pursued

- **Multi-view fusion** (combining front+side+diagonal simultaneously):
  rejected because the actual product only ever receives one arbitrary
  camera angle per upload — a fusion model trained on synchronized
  multi-camera data wouldn't transfer to the real single-video use case.
- **Heavier pose models (YOLO-pose, ViTPose, ViTPose+ via cloud API):**
  researched as options if `heavy` MediaPipe proves insufficient. Not
  pursued past MediaPipe `heavy` in Phase 1 — no evidence model precision
  was the bottleneck for any issue at that stage. Revisited briefly in
  Phase 2 as a *speed* lever (not accuracy) given free-tier CPU
  constraints — see below.
- **Reference-comparison approach** (Dynamic Time Warping against a
  known-good reference rep): identified as a more scalable v2
  architecture than absolute thresholds. Explicitly deferred as future
  work.

---

## Phase 2 — Deployment

Full detail lives in `main.py`'s inline comments and `README.md`; this is
the condensed evidence trail, in the same spirit as Phase 1 above.

**Architecture:** FastAPI wrapper around `rep_counter.py`'s `run()`,
called by direct import (not subprocess — confirmed safe once the actual
function signature was known, avoiding a redundant MediaPipe model reload
per job). Job-ID/polling pattern rather than a synchronous request, since
processing can exceed typical HTTP proxy timeouts. Concurrency capped at
exactly 1 via a bounded queue and a single worker thread.

**Video codec fix:** OpenCV's `mp4v`-fourcc output isn't browser-playable
(most browsers only decode H.264/VP9/AV1 in an mp4 container). Added an
`ffmpeg` transcode pass after pipeline processing.

**Memory findings (measured, not assumed):** peak memory during both pose
extraction and `ffmpeg` transcoding scales with video **resolution**, not
upload file size — a small, highly-compressed 4K clip peaked far higher
and took ~3x longer to process than a larger, lower-resolution clip.
Resolution is now capped at upload time (checked as total pixel count, not
a single width/height value, so it's correct for portrait phone clips too)
and internally downscaled before processing.

**Container dependency gap, found only by testing in Docker:** MediaPipe's
native library requires `libEGL.so.1`/GLES at runtime even for pure CPU
inference (an optional GPU-delegate code path in the same compiled
binary). Present via GPU drivers on Windows/most desktops; absent from a
minimal Debian slim base — only surfaced when actually testing in a
container matching the deployment target, not on the local dev machine.

**Hosting decision:** targeted Render's free tier after checking current
terms rather than assuming prior knowledge held — Hugging Face Spaces no
longer offers free CPU Basic Docker/Gradio Spaces to new accounts (July
2026), and Fly.io no longer has a real free tier. Render's free tier is
**512MB RAM, 0.1 CPU** (a hard quota, not a soft share) — this CPU figure
was initially misstated as 0.5 CPU during planning and corrected only
after real-world processing times (measured on the actual deployment, not
locally) were dramatically slower than local testing predicted. `ffmpeg`
is pinned to a single thread as a direct consequence — with no real
parallelism available in a 10%-of-one-core quota, multi-threaded encoding
adds scheduling overhead rather than helping.

**Cleanup design:** layered rather than relying on any single mechanism —
explicit delete on user action, best-effort delete via `fetch(keepalive)`
on page unload, a 1-hour TTL sweep as backstop, and a full wipe of
orphaned files on process restart (since the in-memory job store can't
know about files left over from a previous process).

**Known, accepted limitation:** the processing timeout (7 minutes per
stage, raised from an initial 5 minutes that proved too tight for the
real free-tier CPU constraint) is soft, not a hard kill — Python threads
can't be forcibly terminated, so a genuinely hung job keeps running in the
background even after the queue moves on to the next job. Documented in
`main.py` rather than solved with heavier infrastructure (a real task
queue, process-level isolation) that this project's scale doesn't justify.
