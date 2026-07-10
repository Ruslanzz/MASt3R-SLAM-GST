#!/usr/bin/env bash
# DeepStream MASt3R-SLAM from a network stream (RTP/H264 over UDP, or RTSP).
#   pipelines/run_udp.sh udp  [port]
#   pipelines/run_udp.sh rtsp rtsp://host/stream
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"

MODE="${1:-udp}"
W="${MUX_W:-512}"
H="${MUX_H:-384}"
ENC_CFG="${ROOT}/configs/config_infer_mast3r_encoder.txt"
DEC_ENGINE="${DEC_ENGINE:-${ROOT}/../checkpoints/mast3r_decoder.engine}"
COMMON_TAIL=(nvinfer config-file-path="${ENC_CFG}" !
    nvdsmast3rslam infer-gie-id=1 decoder-engine="${DEC_ENGINE}"
        config="${ROOT}/../config/base.yaml" save-dir="${ROOT}/../logs"
        sequence-name=net !
    nvvideoconvert ! fakesink sync=false)

if [ "${MODE}" = "udp" ]; then
  PORT="${2:-5000}"
  gst-launch-1.0 -e \
      udpsrc port="${PORT}" \
          caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! \
      rtpjitterbuffer latency=100 ! rtph264depay ! h264parse ! nvv4l2decoder ! \
      nvvideoconvert ! "video/x-raw(memory:NVMM),format=RGBA" ! m.sink_0 \
      nvstreammux name=m batch-size=1 width="${W}" height="${H}" live-source=1 ! \
      "${COMMON_TAIL[@]}"
elif [ "${MODE}" = "rtsp" ]; then
  URL="${2:?usage: run_udp.sh rtsp <url>}"
  gst-launch-1.0 -e \
      rtspsrc location="${URL}" latency=100 ! rtph264depay ! h264parse ! \
      nvv4l2decoder ! nvvideoconvert ! "video/x-raw(memory:NVMM),format=RGBA" ! m.sink_0 \
      nvstreammux name=m batch-size=1 width="${W}" height="${H}" live-source=1 ! \
      "${COMMON_TAIL[@]}"
else
  echo "unknown mode '${MODE}' (use udp|rtsp)" >&2; exit 1
fi
