#!/usr/bin/env bash
# Build the DeepStream 7.1 image for the native nvdsmast3rslam plugin,
# targeting the NVIDIA GeForce GTX 1660 Ti (Turing, sm_75).
# Run from the repository root.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

IMAGE="${IMAGE:-nvdsmast3rslam:ds7.1-gtx1660ti}"
# Newest DeepStream that supports Turing on x86 dGPU.
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/deepstream:7.1-triton-multiarch}"
CUDA_ARCH="${CUDA_ARCH:-75}"
# WITH_ROS2=1 adds the ROS 2 Humble bridge (nvdsmast3rviz -> RViz), see
# VISUALIZATION.md 2.
WITH_ROS2="${WITH_ROS2:-0}"

echo "[build] image=${IMAGE} base=${BASE_IMAGE} cuda_arch=${CUDA_ARCH} ros2=${WITH_ROS2}"
docker build \
    -f deepstream-mast3r-slam/docker/Dockerfile \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg CUDA_ARCH="${CUDA_ARCH}" \
    --build-arg TORCH_CUDA_ARCH_LIST="7.5" \
    --build-arg WITH_ROS2="${WITH_ROS2}" \
    -t "${IMAGE}" \
    .
echo "[build] done -> ${IMAGE}"
