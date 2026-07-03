#!/usr/bin/env bash
# Mono SLAM from a video file with the HUD/trajectory overlay rendered into a
# WINDOW opened by the pipeline itself (no UDP streaming).
#
#   pipelines/run_file_display.sh /path/to/video.mp4 [seq-name]
#   SINK=egl pipelines/run_file_display.sh ...   # nveglglessink (full NVIDIA
#                                                #  display stack required)
#
# X11 must reach the container; start it with:
#   xhost +local:            # once, on the host
#   docker run ... -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix ...
#
# Default sink is autovideosink (xv/ximagesink): the frame is copied to system
# memory and drawn by X — works on Optimus laptops / hosts without the NVIDIA
# modeset module, and the copy cost is negligible at 512x384 @ 1-2 FPS.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"

VIDEO="${1:?usage: run_file_display.sh <video> [seq]}"
SEQ="${2:-$(basename "${VIDEO%.*}")}"
W="${MUX_W:-512}"
H="${MUX_H:-384}"
ENC_CFG="${ROOT}/configs/config_infer_mast3r_encoder.txt"
DEC_ENGINE="${DEC_ENGINE:-${ROOT}/../checkpoints/mast3r_decoder.engine}"

case "${SINK:-auto}" in
  egl)  SINK_ELEM="nveglglessink" ;;
  xv)   SINK_ELEM="xvimagesink" ;;
  *)    SINK_ELEM="autovideosink" ;;
esac
echo "[display] video=${VIDEO} seq=${SEQ} sink=${SINK_ELEM} (DISPLAY=${DISPLAY:-<unset>})"

gst-launch-1.0 -e \
    filesrc location="${VIDEO}" ! decodebin3 ! nvvideoconvert ! \
    "video/x-raw(memory:NVMM),format=RGBA" ! m.sink_0 \
    nvstreammux name=m batch-size=1 width="${W}" height="${H}" \
        live-source=0 ! \
    nvinfer config-file-path="${ENC_CFG}" ! \
    nvdsmast3rslam infer-gie-id=1 decoder-engine="${DEC_ENGINE}" \
        save-dir="${ROOT}/../logs" sequence-name="${SEQ}" ! \
    nvdsmast3rviz overlay=true ! \
    nvvideoconvert ! "video/x-raw(memory:NVMM),format=RGBA" ! \
    nvdsosd ! nvvideoconvert ! videoconvert ! \
    ${SINK_ELEM} sync=false
