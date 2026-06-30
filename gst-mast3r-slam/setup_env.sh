#!/usr/bin/env bash
# Source this to register the mast3rslam element with GStreamer:
#   source gst-mast3r-slam/setup_env.sh
#
# - GST_PLUGIN_PATH points at this dir so the python loader scans ./python/*.py
# - PYTHONPATH exposes the mast3r_slam_gst package and the repo root (mast3r_slam)
PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PLUGIN_DIR}/.." && pwd)"

export GST_PLUGIN_PATH="${PLUGIN_DIR}${GST_PLUGIN_PATH:+:${GST_PLUGIN_PATH}}"
export PYTHONPATH="${PLUGIN_DIR}/python:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "GST_PLUGIN_PATH=${GST_PLUGIN_PATH}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "Verify with: gst-inspect-1.0 mast3rslam"
