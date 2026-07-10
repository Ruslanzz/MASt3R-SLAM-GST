#!/usr/bin/env bash
# Universal MASt3R-SLAM launcher: ONE script for a mono or a stereo camera,
# any source type, optional visualization. The element runs stereo-mode=auto,
# so the same pipeline code handles both setups: one frame per batch -> mono,
# left+right in the batch -> stereo with metric scale from the baseline.
#
#   MONO:    pipelines/run_slam.sh <input> [seq]
#   STEREO:  pipelines/run_slam.sh <left> <right> [baseline_m] [seq]
#
# <input>/<left>/<right> is any of:
#   video file        /data/clip.mp4
#   V4L2 camera       /dev/video0
#   RTSP stream       rtsp://host/stream
#   RTP/H264 over UDP udp://:5000        (listen port)
#
# Visualization (VIZ env, default none):
#   VIZ=none    headless, fakesink (trajectory/.ply still saved to logs/)
#   VIZ=window  HUD/trajectory overlay in an X11 window (VISUALIZATION.md 1b;
#               SINK=egl|xv overrides the default autovideosink)
#   VIZ=udp     overlay streamed as H264/RTP (VIEW_HOST/VIEW_PORT, def. 5600)
#   VIZ=rviz    3D map + trajectory in RViz launched NEXT TO the pipeline in
#               this container (image built WITH_ROS2=1; X11 like VIZ=window;
#               RVIZ_CFG overrides configs/mast3r_slam.rviz); implies ROS=true
#   ROS=true    publish odom/path/map/TF to ROS 2 (image built WITH_ROS2=1)
#
# Examples:
#   bash pipelines/run_slam.sh /data/clip.mp4
#   bash pipelines/run_slam.sh /dev/video0 walk1
#   bash pipelines/run_slam.sh /dev/video0 /dev/video1 0.12
#   VIZ=window bash pipelines/run_slam.sh /data/left.mp4 /data/right.mp4 0.12
#   VIZ=udp VIEW_HOST=192.168.1.50 ROS=true \
#       bash pipelines/run_slam.sh rtsp://cam/l rtsp://cam/r 0.20 offroad
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"

usage="usage: run_slam.sh <input> [seq]  |  run_slam.sh <left> <right> [baseline_m] [seq]"

# A CLI arg is a video source (not a sequence name) if it is a device node,
# a URI, or an existing file.
is_source() {
  case "$1" in
    /dev/*|*://*) return 0 ;;
    *) [ -f "$1" ] ;;
  esac
}

INPUT="${1:?${usage}}"
STEREO=0
if [ "$#" -ge 2 ] && is_source "$2"; then
  STEREO=1
  LEFT="$1"; RIGHT="$2"
  BASELINE="${3:-0.12}"
  SEQ="${4:-stereo}"
else
  SEQ="${2:-$(basename "${INPUT%.*}")}"
fi
SEQ="${SEQ//[^A-Za-z0-9._-]/_}"   # seq becomes a file name (logs/<seq>.txt)

W="${MUX_W:-512}"
H="${MUX_H:-384}"
DEC_ENGINE="${DEC_ENGINE:-${ROOT}/../checkpoints/mast3r_decoder.engine}"
if [ "${STEREO}" = 1 ]; then
  ENC_CFG="${ROOT}/configs/config_infer_mast3r_encoder_stereo.txt"
else
  ENC_CFG="${ROOT}/configs/config_infer_mast3r_encoder.txt"
fi
VIZ="${VIZ:-none}"
ROS="${ROS:-false}"

# ---------------------------------------------------------------- source bins
LIVE=0
PIPE=()
src_branch() {  # $1 = input, $2 = nvstreammux pad (m.sink_0 / m.sink_1)
  local in="$1" pad="$2"
  case "${in}" in
    /dev/*)
      LIVE=1
      PIPE+=(v4l2src device="${in}" ! videoconvert ! nvvideoconvert !
             "video/x-raw(memory:NVMM),format=RGBA" ! "${pad}") ;;
    rtsp://*)
      LIVE=1
      PIPE+=(rtspsrc location="${in}" latency=100 ! rtph264depay ! h264parse !
             nvv4l2decoder ! nvvideoconvert !
             "video/x-raw(memory:NVMM),format=RGBA" ! "${pad}") ;;
    udp://*)
      LIVE=1
      local port="${in#udp://}"; port="${port#:}"; port="${port%%/*}"
      PIPE+=(udpsrc port="${port}"
             caps="application/x-rtp,media=video,encoding-name=H264,payload=96" !
             rtpjitterbuffer latency=100 ! rtph264depay ! h264parse !
             nvv4l2decoder ! nvvideoconvert !
             "video/x-raw(memory:NVMM),format=RGBA" ! "${pad}") ;;
    *)
      [ -f "${in}" ] || { echo "input not found: ${in}" >&2; exit 1; }
      PIPE+=(filesrc location="${in}" ! decodebin3 ! nvvideoconvert !
             "video/x-raw(memory:NVMM),format=RGBA" ! "${pad}") ;;
  esac
}

if [ "${STEREO}" = 1 ]; then
  src_branch "${LEFT}" m.sink_0     # source-id 0 = left (drives SLAM)
  src_branch "${RIGHT}" m.sink_1    # source-id 1 = right (metric scale)
  BATCH=2
  SLAM_ARGS=(baseline="${BASELINE}")
  echo "[slam] STEREO left=${LEFT} right=${RIGHT} baseline=${BASELINE}m seq=${SEQ} viz=${VIZ}"
else
  src_branch "${INPUT}" m.sink_0
  BATCH=1
  SLAM_ARGS=()
  echo "[slam] MONO input=${INPUT} seq=${SEQ} viz=${VIZ}"
fi

MUX=(nvstreammux name=m batch-size="${BATCH}" width="${W}" height="${H}"
     live-source="${LIVE}")
[ "${LIVE}" = 1 ] && MUX+=(batched-push-timeout=40000)

CORE=(nvinfer config-file-path="${ENC_CFG}" !
      nvdsmast3rslam infer-gie-id=1 decoder-engine="${DEC_ENGINE}"
          save-dir="${ROOT}/../logs" sequence-name="${SEQ}")
[ "${#SLAM_ARGS[@]}" -gt 0 ] && CORE+=("${SLAM_ARGS[@]}")

# ------------------------------------------------------------------ viz tail
viz_overlay() {  # overlay element + tiler (stereo shows left|right side by side)
  TAIL=(! nvdsmast3rviz overlay=true ros-enable="${ROS}")
  if [ "${STEREO}" = 1 ]; then
    TAIL+=(! nvmultistreamtiler rows=1 columns=2 width=$((2 * W)) height="${H}")
  fi
  TAIL+=(! nvvideoconvert ! "video/x-raw(memory:NVMM),format=RGBA" ! nvdsosd)
}

case "${VIZ}" in
  none)
    if [ "${ROS}" = "true" ]; then
      TAIL=(! nvdsmast3rviz overlay=false ros-enable=true
            ! nvvideoconvert ! fakesink sync=false)
    else
      TAIL=(! nvvideoconvert ! fakesink sync=false)
    fi ;;
  window)
    case "${SINK:-auto}" in
      egl) SINK_ELEM="nveglglessink" ;;
      xv)  SINK_ELEM="xvimagesink" ;;
      *)   SINK_ELEM="autovideosink" ;;
    esac
    viz_overlay
    TAIL+=(! nvvideoconvert ! videoconvert ! "${SINK_ELEM}" sync=false) ;;
  udp)
    viz_overlay
    TAIL+=(! nvvideoconvert !
           nvv4l2h264enc bitrate=4000000 insert-sps-pps=1 idrinterval=30 !
           h264parse ! rtph264pay config-interval=1 pt=96 !
           udpsink host="${VIEW_HOST:-127.0.0.1}" port="${VIEW_PORT:-5600}"
               sync=false)
    echo "[slam] overlay stream -> udp://${VIEW_HOST:-127.0.0.1}:${VIEW_PORT:-5600}" ;;
  rviz)
    # ROS bridge on + RViz window from this very container (VISUALIZATION.md 2.4).
    TAIL=(! nvdsmast3rviz overlay=false ros-enable=true
          ! nvvideoconvert ! fakesink sync=false) ;;
  *)
    echo "unknown VIZ='${VIZ}' (use none|window|udp|rviz)" >&2; exit 1 ;;
esac

RVIZ_PID=""
if [ "${VIZ}" = "rviz" ]; then
  ROS_SETUP="/opt/ros/humble/setup.bash"
  RVIZ_CFG="${RVIZ_CFG:-${ROOT}/configs/mast3r_slam.rviz}"
  if [ ! -f "${ROS_SETUP}" ]; then
    echo "[slam] no ROS 2 in this image — rebuild with WITH_ROS2=1 (VISUALIZATION.md 2.1)" >&2
    exit 1
  fi
  # ROS setup scripts are not `set -u`-clean.
  set +u; . "${ROS_SETUP}"; set -u
  if ! command -v rviz2 >/dev/null 2>&1; then
    echo "[slam] rviz2 not found — image built with WITH_ROS2=1 before rviz2 was added; rebuild it" >&2
    exit 1
  fi
  rviz2 -d "${RVIZ_CFG}" &
  RVIZ_PID=$!
  trap '[ -n "${RVIZ_PID}" ] && kill "${RVIZ_PID}" 2>/dev/null || true' EXIT
  echo "[slam] rviz2 pid=${RVIZ_PID} config=${RVIZ_CFG} (DISPLAY=${DISPLAY:-<unset>})"
fi

gst-launch-1.0 -e "${PIPE[@]}" "${MUX[@]}" ! "${CORE[@]}" "${TAIL[@]}"
