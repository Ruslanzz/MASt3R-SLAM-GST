#!/usr/bin/env bash
# MASt3R-SLAM from a v4l2 camera (monocular).
#
#   pipelines/v4l2_camera.sh [/dev/videoN]
#
# Prints "mast3r-slam-pose" element messages (-m) and, on Ctrl-C, sends EOS (-e)
# so the trajectory/point-cloud are written to logs/live.{txt,ply}.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/../setup_env.sh" >/dev/null

DEVICE="${1:-/dev/video0}"
CONFIG="${CONFIG:-config/base.yaml}"

echo "[pipeline] v4l2 device=${DEVICE} config=${CONFIG}"

# --- USB camera on a dGPU host -------------------------------------------------
gst-launch-1.0 -m -e \
    v4l2src device="${DEVICE}" io-mode=2 ! \
    videoconvert ! video/x-raw,format=RGBA ! \
    queue max-size-buffers=4 leaky=downstream ! \
    mast3rslam config="${CONFIG}" save-dir=logs sequence-name=live \
               backend=torch device=cuda:0 ! \
    fakesink sync=false

# --- Jetson CSI camera (Argus) — uncomment instead of the block above ----------
# gst-launch-1.0 -m -e \
#     nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1' ! \
#     nvvideoconvert ! video/x-raw,format=RGBA ! \
#     mast3rslam config="${CONFIG}" save-dir=logs sequence-name=live ! \
#     fakesink sync=false
