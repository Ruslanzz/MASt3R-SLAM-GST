#!/usr/bin/env bash
# MASt3R-SLAM from a video file, using NVIDIA hardware decode (nvv4l2decoder).
#
#   pipelines/video_file.sh /path/to/video.mp4 [config.yaml] [calib.yaml]
#
# Writes logs/<basename>.{txt,ply} + keyframes on EOS.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/../setup_env.sh" >/dev/null

VIDEO="${1:?usage: video_file.sh <video.mp4> [config.yaml] [calib.yaml]}"
CONFIG="${2:-config/base.yaml}"
CALIB="${3:-}"
SEQ="$(basename "${VIDEO}")"; SEQ="${SEQ%.*}"

CALIB_PROP=""
[ -n "${CALIB}" ] && CALIB_PROP="calib=${CALIB}"

echo "[pipeline] file=${VIDEO} config=${CONFIG} seq=${SEQ} ${CALIB_PROP}"

# Container-agnostic demux via decodebin3; nvvideoconvert copies NVMM -> system.
gst-launch-1.0 -m -e \
    filesrc location="${VIDEO}" ! decodebin3 ! \
    nvvideoconvert ! video/x-raw,format=RGBA ! \
    mast3rslam config="${CONFIG}" ${CALIB_PROP} \
               save-dir=logs sequence-name="${SEQ}" \
               backend=torch device=cuda:0 ! \
    fakesink sync=false

# --- Explicit H.264 hw-decode variant (if decodebin3 picks sw decode) ----------
# gst-launch-1.0 -m -e \
#     filesrc location="${VIDEO}" ! qtdemux ! h264parse ! nvv4l2decoder ! \
#     nvvideoconvert ! video/x-raw,format=RGBA ! \
#     mast3rslam config="${CONFIG}" ${CALIB_PROP} sequence-name="${SEQ}" ! \
#     fakesink sync=false
