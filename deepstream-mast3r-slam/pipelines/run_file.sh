#!/usr/bin/env bash
# DeepStream MASt3R-SLAM from a video file.
#   pipelines/run_file.sh /path/to/video.mp4 [seq-name]
#
# Layout: file -> decode -> nvstreammux(batch=1) -> nvinfer(MASt3R encoder)
#         -> nvdsmast3rslam (decoder + SLAM) -> fakesink
# Poses are attached as NvDsUserMeta; trajectory/.ply written to logs/<seq>.* on EOS.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"

VIDEO="${1:?usage: run_file.sh <video> [seq]}"
SEQ="${2:-$(basename "${VIDEO%.*}")}"
W="${MUX_W:-512}"
H="${MUX_H:-384}"
ENC_CFG="${ROOT}/configs/config_infer_mast3r_encoder.txt"
DEC_ENGINE="${DEC_ENGINE:-${ROOT}/../checkpoints/mast3r_decoder.engine}"

gst-launch-1.0 -e \
    filesrc location="${VIDEO}" ! decodebin3 ! nvvideoconvert ! \
    "video/x-raw(memory:NVMM),format=RGBA" ! m.sink_0 \
    nvstreammux name=m batch-size=1 width="${W}" height="${H}" \
        live-source=0 ! \
    nvinfer config-file-path="${ENC_CFG}" ! \
    nvdsmast3rslam infer-gie-id=1 decoder-engine="${DEC_ENGINE}" \
        config="${ROOT}/../config/base.yaml" save-dir="${ROOT}/../logs" \
        sequence-name="${SEQ}" ! \
    nvvideoconvert ! fakesink sync=false
