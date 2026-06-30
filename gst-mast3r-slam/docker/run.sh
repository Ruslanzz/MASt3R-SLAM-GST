#!/usr/bin/env bash
# Run the container with GPU + (optionally) camera / display passthrough.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-mast3r-slam-gst:ds9.0}"

DEVICE_ARGS=()
# USB / v4l2 cameras
for dev in /dev/video*; do
    [ -e "$dev" ] && DEVICE_ARGS+=("--device=$dev")
done

docker run --rm -it \
    --gpus all \
    --runtime nvidia \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    --network host \
    "${DEVICE_ARGS[@]}" \
    -v "${REPO_ROOT}:/opt/MASt3R-SLAM-GST" \
    -v "${REPO_ROOT}/checkpoints:/opt/MASt3R-SLAM-GST/checkpoints" \
    -w /opt/MASt3R-SLAM-GST \
    "${IMAGE}" \
    "${@:-bash}"
