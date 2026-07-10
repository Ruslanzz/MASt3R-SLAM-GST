#!/usr/bin/env bash
# Configure + build the plugin inside an existing DeepStream container
# (with libtorch installed). Run from anywhere.
#
# ROS 2: if the image has ROS 2 Humble (built WITH_ROS2=1), the ROS bridge is
# enabled automatically so an in-container rebuild does not silently produce a
# plugin without ROS support. Override with WITH_ROS2=0 (or =1) explicitly.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TORCH_DIR="${Torch_DIR:-$(python3 -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"share","cmake","Torch"))')}"
BUILD_DIR="${BUILD_DIR:-${HERE}/build}"
CUDA_ARCH="${CUDA_ARCH:-75}"

if [ -z "${WITH_ROS2:-}" ]; then
  if [ -f /opt/ros/humble/setup.bash ]; then WITH_ROS2=1; else WITH_ROS2=0; fi
fi
ROS_FLAGS=()
if [ "${WITH_ROS2}" = "1" ]; then
  # ament_cmake needs the ROS env (PYTHONPATH etc.); its setup scripts are not
  # `set -u`-clean.
  set +u; source /opt/ros/humble/setup.bash; set -u
  ROS_FLAGS=(-DWITH_ROS2=ON -DCMAKE_PREFIX_PATH=/opt/ros/humble)
fi
echo "[build] cuda_arch=${CUDA_ARCH} ros2=${WITH_ROS2}"

cmake -S "${HERE}" -B "${BUILD_DIR}" \
    -DTorch_DIR="${TORCH_DIR}" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    "${ROS_FLAGS[@]}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo "[build] libnvdsmast3rslam.so ->"
ls -l "${BUILD_DIR}"/libnvdsmast3rslam.so
GST_PLUGIN_DIR="$(pkg-config --variable=pluginsdir gstreamer-1.0)"
echo "Install with: sudo cp ${BUILD_DIR}/libnvdsmast3rslam.so ${GST_PLUGIN_DIR}/"
echo "  then: gst-inspect-1.0 nvdsmast3rslam"
echo "  (ensure libtorch is loadable, e.g. export LD_LIBRARY_PATH=\$(python3 -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),\"lib\"))'):\$LD_LIBRARY_PATH)"
