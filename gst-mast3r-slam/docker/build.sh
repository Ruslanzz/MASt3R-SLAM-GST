#!/usr/bin/env bash
# Build the DeepStream 9.0 based image for the MASt3R-SLAM GStreamer plugin.
#
# Run from the repository root (the build context must include the submodules
# and, ideally, the downloaded checkpoints under checkpoints/).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

IMAGE="${IMAGE:-mast3r-slam-gst:ds9.0}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/deepstream:9.0-triton-multiarch}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0 8.6 8.9}"

echo "[build] image=${IMAGE} base=${BASE_IMAGE} arch='${TORCH_CUDA_ARCH_LIST}'"
docker build \
    -f gst-mast3r-slam/docker/Dockerfile \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    -t "${IMAGE}" \
    .
echo "[build] done -> ${IMAGE}"
