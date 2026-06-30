#!/usr/bin/env bash
# Configure + build the plugin inside an existing DeepStream 9.0 container
# (with libtorch installed). Run from anywhere.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TORCH_DIR="${Torch_DIR:-$(python3 -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"share","cmake","Torch"))')}"
BUILD_DIR="${BUILD_DIR:-${HERE}/build}"

cmake -S "${HERE}" -B "${BUILD_DIR}" \
    -DTorch_DIR="${TORCH_DIR}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo "[build] libnvdsmast3rslam.so ->"
ls -l "${BUILD_DIR}"/libnvdsmast3rslam.so
echo "Install with: cp ${BUILD_DIR}/libnvdsmast3rslam.so \\"
echo "    /opt/nvidia/deepstream/deepstream/lib/gstreamer-1.0/ && gst-inspect-1.0 nvdsmast3rslam"
