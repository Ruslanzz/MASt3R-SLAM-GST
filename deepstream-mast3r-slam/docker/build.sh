#!/usr/bin/env bash
# Build the DeepStream 9.0 image for the native nvdsmast3rslam plugin.
# Run from the repository root.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

IMAGE="${IMAGE:-nvdsmast3rslam:ds9.0}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/deepstream:9.0-triton-multiarch}"

echo "[build] image=${IMAGE} base=${BASE_IMAGE}"
docker build \
    -f deepstream-mast3r-slam/docker/Dockerfile \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -t "${IMAGE}" \
    .
echo "[build] done -> ${IMAGE}"
