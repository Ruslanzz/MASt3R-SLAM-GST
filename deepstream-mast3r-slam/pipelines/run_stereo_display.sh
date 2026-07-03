#!/usr/bin/env bash
# Stereo-hybrid SLAM with the overlay rendered into a WINDOW opened by the
# pipeline (left|right tiled side by side, HUD/minimap on the left view).
#
#   pipelines/run_stereo_display.sh [/dev/videoL] [/dev/videoR] [baseline_m]
#   SINK=egl ... for nveglglessink (full NVIDIA display stack required)
#
# X11 must reach the container (see run_file_display.sh header).
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

case "${SINK:-auto}" in
  egl)  SINK_ELEM="nveglglessink" ;;
  xv)   SINK_ELEM="xvimagesink" ;;
  *)    SINK_ELEM="autovideosink" ;;
esac
echo "[display] left=${LEFT} right=${RIGHT} baseline=${BASELINE}m sink=${SINK_ELEM}"

gst-launch-1.0 -e \
    v4l2src device="${LEFT}" ! videoconvert ! nvvideoconvert ! \
    "video/x-raw(memory:NVMM),format=RGBA" ! m.sink_0 \
    v4l2src device="${RIGHT}" ! videoconvert ! nvvideoconvert ! \
    "video/x-raw(memory:NVMM),format=RGBA" ! m.sink_1 \
    nvstreammux name=m batch-size=2 width="${W}" height="${H}" \
        live-source=1 batched-push-timeout=40000 ! \
    nvinfer config-file-path="${ENC_CFG}" ! \
    nvdsmast3rslam infer-gie-id=1 decoder-engine="${DEC_ENGINE}" \
        stereo-mode=true baseline="${BASELINE}" loop-closure=true \
        save-dir="${ROOT}/../logs" sequence-name=stereo ! \
    nvdsmast3rviz overlay=true ! \
    nvmultistreamtiler rows=1 columns=2 width=1024 height=384 ! \
    nvvideoconvert ! "video/x-raw(memory:NVMM),format=RGBA" ! \
    nvdsosd ! nvvideoconvert ! videoconvert ! \
    ${SINK_ELEM} sync=false
