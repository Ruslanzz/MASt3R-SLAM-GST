#!/usr/bin/env bash
# Stereo-hybrid SLAM + live visualization: HUD/trajectory overlay burned into
# the video and streamed as H.264/RTP over UDP (works on headless hosts), and
# optionally ROS 2 publishing (requires image built with WITH_ROS2=1).
#
#   pipelines/run_stereo_viz.sh [/dev/videoL] [/dev/videoR] [baseline_m]
#   VIEW_HOST=192.168.1.50 VIEW_PORT=5600 ROS=true pipelines/run_stereo_viz.sh ...
#
# Watch on the laptop:
#   gst-launch-1.0 udpsrc port=5600 \
#     caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! \
#     rtpjitterbuffer ! rtph264depay ! avdec_h264 ! autovideosink sync=false
# (or QGroundControl: UDP video, port 5600)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"

LEFT="${1:-/dev/video0}"
RIGHT="${2:-/dev/video1}"
BASELINE="${3:-0.12}"
VIEW_HOST="${VIEW_HOST:-127.0.0.1}"
VIEW_PORT="${VIEW_PORT:-5600}"
ROS="${ROS:-false}"
W="${MUX_W:-512}"
H="${MUX_H:-384}"
ENC_CFG="${ROOT}/configs/config_infer_mast3r_encoder_stereo.txt"
DEC_ENGINE="${DEC_ENGINE:-${ROOT}/../checkpoints/mast3r_decoder.engine}"

echo "[viz] stream -> udp://${VIEW_HOST}:${VIEW_PORT}  ros=${ROS}"

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
    nvdsmast3rviz overlay=true ros-enable="${ROS}" ! \
    nvmultistreamtiler rows=1 columns=2 width=1024 height=384 ! \
    nvvideoconvert ! "video/x-raw(memory:NVMM),format=RGBA" ! \
    nvdsosd ! nvvideoconvert ! \
    nvv4l2h264enc bitrate=4000000 insert-sps-pps=1 idrinterval=30 ! \
    h264parse ! rtph264pay config-interval=1 pt=96 ! \
    udpsink host="${VIEW_HOST}" port="${VIEW_PORT}" sync=false
