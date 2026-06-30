#!/usr/bin/env bash
# Build a TensorRT engine from the exported MASt3R encoder ONNX.
#
# Usage:
#   tools/build_trt_engine.sh <encoder.onnx> [output.engine] [fp16|fp32]
#
# The engine is fixed-shape (matching the ONNX), so no optimisation profile is
# needed. FP16 is ~2x faster on Tensor Cores but will make the SLAM output drift
# from the bit-exact CUDA reference (see DESIGN.md).
set -euo pipefail

ONNX="${1:?usage: build_trt_engine.sh <encoder.onnx> [output.engine] [fp16|fp32]}"
ENGINE="${2:-${ONNX%.onnx}.engine}"
PRECISION="${3:-fp16}"

TRTEXEC="${TRTEXEC:-trtexec}"
if ! command -v "${TRTEXEC}" >/dev/null 2>&1; then
    # DeepStream / TensorRT default install location.
    TRTEXEC="/usr/src/tensorrt/bin/trtexec"
fi

PREC_FLAG=""
if [ "${PRECISION}" = "fp16" ]; then
    PREC_FLAG="--fp16"
fi

echo "[trt] building ${ENGINE} from ${ONNX} (${PRECISION})"
"${TRTEXEC}" \
    --onnx="${ONNX}" \
    --saveEngine="${ENGINE}" \
    ${PREC_FLAG} \
    --memPoolSize=workspace:4096 \
    --skipInference

echo "[trt] wrote ${ENGINE}"
echo "[trt] use it via:  mast3rslam backend=tensorrt trt-encoder-engine=${ENGINE}"
