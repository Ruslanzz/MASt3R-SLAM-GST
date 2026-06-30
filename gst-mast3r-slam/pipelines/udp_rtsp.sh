#!/usr/bin/env bash
# MASt3R-SLAM from a network stream (UDP/RTP H.264 or RTSP).
#
#   pipelines/udp_rtsp.sh udp  [port]            # raw RTP/H264 over UDP
#   pipelines/udp_rtsp.sh rtsp rtsp://host/stream # RTSP
#
# Writes logs/net.{txt,ply} + keyframes on EOS.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/../setup_env.sh" >/dev/null

MODE="${1:-udp}"
CONFIG="${CONFIG:-config/base.yaml}"

case "${MODE}" in
  udp)
    PORT="${2:-5000}"
    echo "[pipeline] UDP RTP/H264 port=${PORT}"
    gst-launch-1.0 -m -e \
        udpsrc port="${PORT}" \
            caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! \
        rtpjitterbuffer latency=100 ! rtph264depay ! h264parse ! \
        nvv4l2decoder ! nvvideoconvert ! video/x-raw,format=RGBA ! \
        queue max-size-buffers=4 leaky=downstream ! \
        mast3rslam config="${CONFIG}" save-dir=logs sequence-name=net \
                   backend=torch device=cuda:0 ! \
        fakesink sync=false
    ;;
  rtsp)
    URL="${2:?usage: udp_rtsp.sh rtsp <rtsp-url>}"
    echo "[pipeline] RTSP url=${URL}"
    gst-launch-1.0 -m -e \
        rtspsrc location="${URL}" latency=100 ! rtph264depay ! h264parse ! \
        nvv4l2decoder ! nvvideoconvert ! video/x-raw,format=RGBA ! \
        queue max-size-buffers=4 leaky=downstream ! \
        mast3rslam config="${CONFIG}" save-dir=logs sequence-name=net \
                   backend=torch device=cuda:0 ! \
        fakesink sync=false
    ;;
  *)
    echo "unknown mode '${MODE}' (use 'udp' or 'rtsp')" >&2
    exit 1
    ;;
esac

# Test sender for the UDP mode (run on another host/terminal):
#   gst-launch-1.0 videotestsrc ! x264enc tune=zerolatency ! rtph264pay ! \
#       udpsink host=127.0.0.1 port=5000
