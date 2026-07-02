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
GST_PLUGIN_DIR="$(pkg-config --variable=pluginsdir gstreamer-1.0)"
echo "Install with: sudo cp ${BUILD_DIR}/libnvdsmast3rslam.so ${GST_PLUGIN_DIR}/"
echo "  then: gst-inspect-1.0 nvdsmast3rslam"
echo "  (ensure libtorch is loadable, e.g. export LD_LIBRARY_PATH=\$(python3 -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),\"lib\"))'):\$LD_LIBRARY_PATH)"
