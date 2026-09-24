# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System-level libraries OpenCV/MediaPipe need on Debian that aren't in
# the base slim image, plus ffmpeg/ffprobe -- main.py shells out to both
# (transcode_for_web, downscale_if_needed, probe_video_resolution).
# libsm6/libxext6 are common extra requirements for opencv-python-headless
# on a minimal Debian base.
# libegl1/libgles2: confirmed necessary by testing, not preemptive --
# MediaPipe's native library links against libEGL.so.1 at load time even
# for pure CPU inference (the same compiled binary supports an optional
# GPU delegate path). Present via GPU drivers on Windows/most desktops,
# but genuinely absent from a minimal Debian base -- this is exactly the
# kind of "works on my machine" gap Docker exists to catch, confirmed by
# an OSError: libEGL.so.1: cannot open shared object file at runtime
# without these two packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libegl1 \
    libgles2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copied and installed BEFORE the rest of the app code, as its own layer:
# Docker caches layers, so this only re-runs (the slow part -- installing
# mediapipe/opencv) when requirements.txt itself changes, not on every
# code edit. Editing main.py alone won't trigger a full dependency
# reinstall on the next build.
COPY requirements.txt .
# --root-user-action=ignore: silences pip's "running as root" warning --
# expected and harmless inside an isolated single-purpose container image
# (no multi-user permission conflict to actually cause), not something
# that needs the fuller fix (a dedicated non-root USER) at this project's
# scale.
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# Now the actual app: main.py, rep_counter.py, compute_angles.py,
# thresholds.py, index.html -- everything expected to live in one folder.
COPY . .

# Pre-download the MediaPipe model at BUILD time, not first-request time.
# Without this, ensure_model() in rep_counter.py downloads it on-demand --
# fine for a long-lived local process, but on Render's free tier the
# container's disk is wiped every time it spins down after 15min idle, so
# the model would otherwise be RE-downloaded on every cold start, adding
# real latency (and a runtime dependency on storage.googleapis.com being
# reachable) to the first request after any idle period. Baking it into
# the image layer removes both problems.
RUN python -c "from rep_counter import ensure_model; ensure_model()"

EXPOSE 8000

# JSON-array form calling `sh -c` explicitly (satisfies Docker's
# JSONArgsRecommended check), WITH `exec` -- this is the part that
# actually fixes the underlying issue, not just the lint warning. A
# shell is still needed to expand ${PORT:-8000} (Render injects the real
# port to listen on via $PORT at runtime, not always 8000). Without
# `exec`, sh stays alive as PID 1 and uvicorn runs as ITS child, so a
# shutdown signal (SIGTERM, e.g. during a Render redeploy) goes to the
# shell first, not guaranteed to reach uvicorn. `exec` replaces the shell
# process with uvicorn (the exec syscall, not a subprocess) -- uvicorn
# becomes PID 1 itself, so it receives signals directly.
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]