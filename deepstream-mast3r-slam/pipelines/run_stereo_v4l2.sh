#!/usr/bin/env bash
# Stereo-hybrid MASt3R-SLAM: mono tracking on the LEFT camera + metric scale
# from the stereo baseline (DESIGN-STEREO.md). Trajectory comes out in METERS.
#
#   pipelines/run_stereo_v4l2.sh [/dev/videoL] [/dev/videoR] [baseline_m]
#
# sink_0 -> source-id 0 (left), sink_1 -> source-id 1 (right).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"

LEFT="${1:-/dev/video0}"
RIGHT="${2:-/dev/video1}"
BASELINE="${3:-0.12}"
W="${MUX_W:-512}"
H="${MUX_H:-384}"
ENC_CFG="${ROOT}/configs/config_infer_mast3r_encoder_stereo.txt"
DEC_ENGINE="${DEC_ENGINE:-${ROOT}/../checkpoints/mast3r_decoder.engine}"

echo "[stereo] left=${LEFT} right=${RIGHT} baseline=${BASELINE}m"

gst-launch-1.0 -e \
    v4l2src device="${LEFT}" ! videoconvert ! nvvideoconvert ! \
    "video/x-raw(memory:NVMM),format=RGBA" ! m.sink_0 \
    v4l2src device="${RIGHT}" ! videoconvert ! nvvideoconvert ! \
    "video/x-raw(memory:NVMM),format=RGBA" ! m.sink_1 \
    nvstreammux name=m batch-size=2 width="${W}" height="${H}" \
        live-source=1 batched-push-timeout=40000 ! \
    nvinfer config-file-path="${ENC_CFG}" ! \
    nvdsmast3rslam infer-gie-id=1 decoder-engine="${DEC_ENGINE}" \
        stereo-mode=true baseline="${BASELINE}" \
        left-source-id=0 right-source-id=1 loop-closure=true \
        save-dir="${ROOT}/../logs" sequence-name=stereo ! \
    nvvideoconvert ! fakesink sync=false

# --- two synchronized video files instead of cameras -------------------------
# gst-launch-1.0 -e \
#     filesrc location=left.mp4  ! decodebin3 ! nvvideoconvert ! \
#       'video/x-raw(memory:NVMM),format=RGBA' ! m.sink_0 \
#     filesrc location=right.mp4 ! decodebin3 ! nvvideoconvert ! \
#       'video/x-raw(memory:NVMM),format=RGBA' ! m.sink_1 \
#     nvstreammux name=m batch-size=2 width=512 height=384 ! \
#     nvinfer config-file-path=".../config_infer_mast3r_encoder_stereo.txt" ! \
#     nvdsmast3rslam stereo-mode=true baseline=0.12 decoder-engine=... ! \
#     nvvideoconvert ! fakesink sync=false
