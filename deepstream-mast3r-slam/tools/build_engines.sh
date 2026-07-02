#!/usr/bin/env bash
# One-shot: export MASt3R encoder+decoder to ONNX and build TensorRT engines.
#
#   bash deepstream-mast3r-slam/tools/build_engines.sh [fp16|fp32] [H] [W]
#
# MUST run inside the container ON the target GPU (engines are GPU- and
# TensorRT-version-specific, so they cannot be prebuilt in the docker image).
# Expects the checkpoint at checkpoints/MASt3R_..._metric.pth.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
cd "${ROOT}"

PRECISION="${1:-fp16}"
H="${2:-384}"
W="${3:-512}"
CKPT="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
ENC_ONNX="checkpoints/mast3r_encoder.onnx"
DEC_ONNX="checkpoints/mast3r_decoder.onnx"
ENC_ENGINE="checkpoints/mast3r_encoder.engine"
DEC_ENGINE="checkpoints/mast3r_decoder.engine"

if [ ! -f "${CKPT}" ]; then
    echo "[engines] checkpoint not found: ${CKPT}" >&2
    echo "  wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/" >&2
    exit 1
fi

# Locate trtexec (deb layout, tar layout, or PATH).
TRTEXEC="${TRTEXEC:-}"
if [ -z "${TRTEXEC}" ]; then
    for cand in /usr/src/tensorrt/bin/trtexec /opt/tensorrt/bin/trtexec trtexec; do
        if command -v "${cand}" >/dev/null 2>&1; then TRTEXEC="${cand}"; break; fi
    done
fi
if [ -z "${TRTEXEC}" ]; then
    echo "[engines] trtexec not found (checked /usr/src/tensorrt/bin, /opt/tensorrt/bin, PATH)." >&2
    echo "  locate it with: find / -name trtexec -type f 2>/dev/null" >&2
    echo "  then re-run:   TRTEXEC=/path/to/trtexec $0 ${PRECISION}" >&2
    exit 1
fi
echo "[engines] trtexec: ${TRTEXEC}, precision: ${PRECISION}, input: ${H}x${W}"

PREC_FLAG=""
[ "${PRECISION}" = "fp16" ] && PREC_FLAG="--fp16"

# 1) ONNX export (skipped if both files already exist; delete them to re-export).
if [ ! -f "${ENC_ONNX}" ] || [ ! -f "${DEC_ONNX}" ]; then
    echo "[engines] exporting ONNX (encoder + decoder) ..."
    python3 deepstream-mast3r-slam/tools/export_onnx.py \
        --checkpoint "${CKPT}" --height "${H}" --width "${W}" --which both \
        --out-encoder "${ENC_ONNX}" --out-decoder "${DEC_ONNX}"
else
    echo "[engines] ONNX files exist, skipping export (delete to re-export)"
fi

# 2) TensorRT engines. Each run takes several minutes on a GTX 1660 Ti;
#    workspace is capped at 2 GB to fit the 6 GB card.
echo "[engines] building encoder engine ..."
"${TRTEXEC}" --onnx="${ENC_ONNX}" --saveEngine="${ENC_ENGINE}" \
    ${PREC_FLAG} --memPoolSize=workspace:2048 --skipInference

echo "[engines] building decoder engine ..."
"${TRTEXEC}" --onnx="${DEC_ONNX}" --saveEngine="${DEC_ENGINE}" \
    ${PREC_FLAG} --memPoolSize=workspace:2048 --skipInference

echo "[engines] done:"
ls -lh "${ENC_ENGINE}" "${DEC_ENGINE}"
echo "[engines] next: bash deepstream-mast3r-slam/pipelines/run_file.sh <video.mp4>"
