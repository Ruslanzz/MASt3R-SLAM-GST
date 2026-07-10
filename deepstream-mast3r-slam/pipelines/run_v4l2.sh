#!/usr/bin/env bash
# DeepStream MASt3R-SLAM from a v4l2 camera.
#   pipelines/run_v4l2.sh [/dev/videoN]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"

DEVICE="${1:-/dev/video0}"
W="${MUX_W:-512}"
H="${MUX_H:-384}"
ENC_CFG="${ROOT}/configs/config_infer_mast3r_encoder.txt"
DEC_ENGINE="${DEC_ENGINE:-${ROOT}/../checkpoints/mast3r_decoder.engine}"

gst-launch-1.0 -e \
    v4l2src device="${DEVICE}" ! videoconvert ! nvvideoconvert ! \
    "video/x-raw(memory:NVMM),format=RGBA" ! m.sink_0 \
    nvstreammux name=m batch-size=1 width="${W}" height="${H}" \
        live-source=1 ! \
    nvinfer config-file-path="${ENC_CFG}" ! \
    nvdsmast3rslam infer-gie-id=1 decoder-engine="${DEC_ENGINE}" \
        config="${ROOT}/../config/base.yaml" save-dir="${ROOT}/../logs" \
        sequence-name=live ! \
    nvvideoconvert ! fakesink sync=false

# Jetson CSI (Argus) variant:
#   nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1280,height=720' ! \
#   nvvideoconvert ! 'video/x-raw(memory:NVMM),format=RGBA' ! m.sink_0 nvstreammux ...
