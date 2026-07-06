#!/usr/bin/env bash
# One command: MONO SLAM from a single camera + RViz opened AUTOMATICALLY.
#
#   pipelines/run_mono_rviz.sh [/dev/videoN] [seq-name]
#
# This is a thin wrapper over run_slam.sh with VIZ=rviz (mono, ROS bridge on,
# rviz2 launched next to the pipeline and closed with it). Use it when you have
# one camera and want to watch the live 3D map + trajectory without typing env
# vars.
#
# REQUIREMENTS:
#   * image built with ROS 2 + rviz2:  WITH_ROS2=1 bash docker/build.sh
#   * X11 forwarded into the container (for the RViz window):
#       xhost +local:                                          # once, on host
#       docker run ... -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix ...
#   (full details in VISUALIZATION.md 2.4)
#
# A different RViz layout:  RVIZ_CFG=/path/my.rviz pipelines/run_mono_rviz.sh ...
# A video file instead of a camera also works:  run_mono_rviz.sh /data/clip.mp4
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CAM="${1:-/dev/video0}"
SEQ="${2:-mono}"

# This script is single-input by design. A second camera would make run_slam.sh
# switch to stereo, which is not what "mono" promises — catch that early.
case "${SEQ}" in
  /dev/*|*://*) is_dev=1 ;;
  *) [ -f "${SEQ}" ] && is_dev=1 || is_dev=0 ;;
esac
if [ "${is_dev}" = 1 ]; then
  echo "[mono-rviz] this is the single-camera script; '${SEQ}' looks like a second source." >&2
  echo "            For a stereo pair use:  VIZ=rviz pipelines/run_slam.sh <left> <right> <baseline>" >&2
  exit 2
fi

# Guard against the classic v4l2 pitfall: UVC cameras expose an extra /dev/videoN
# that only carries METADATA (caps show 'Metadata Capture', not 'Video Capture').
# Feeding that node to v4l2src fails with "not a capture device".
case "${CAM}" in
  /dev/video*)
    if command -v v4l2-ctl >/dev/null 2>&1; then
      info="$(v4l2-ctl -d "${CAM}" --info 2>/dev/null || true)"
      if [ -n "${info}" ] && ! printf '%s' "${info}" | grep -qi "Video Capture"; then
        echo "[mono-rviz] '${CAM}' is not a video-capture device (likely a metadata node)." >&2
        echo "            Cameras often add a second /dev/videoN for metadata; pick the capture one:" >&2
        echo "              v4l2-ctl --list-devices" >&2
        exit 2
      fi
    fi ;;
esac

echo "[mono-rviz] camera=${CAM} seq=${SEQ} -> mono SLAM + RViz (DISPLAY=${DISPLAY:-<unset>})"
export VIZ=rviz
exec bash "${HERE}/run_slam.sh" "${CAM}" "${SEQ}"
