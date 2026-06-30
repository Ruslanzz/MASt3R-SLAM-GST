#!/usr/bin/env python3
"""Export the MASt3R ViT encoder (``model._encode_image``) to ONNX.

Only the encoder is exported — it is the heaviest, statically-shaped part of the
network (one forward per frame at a fixed resolution). The decoder, DPT heads and
the SLAM geometry stay on torch/CUDA (see DESIGN.md, section 2). The resulting
ONNX is turned into a TensorRT ``.engine`` with ``tools/build_trt_engine.sh`` and
consumed by ``backend=tensorrt``.

The encoder input shape must match the resized frame shape that ``resize_img``
produces for your stream (long edge 512, dims multiples of 8/16). For a 4:3
source this is typically 384x512 (HxW). Use ``--probe-image`` to derive it from a
sample frame automatically.

Caveats: CroCo/MASt3R uses RoPE positional encoding and masked attention; some
opsets struggle with it. Try ``--opset 17`` first, fall back to ``--opset 18``.
This script is a starting point, not a guaranteed one-shot export.
"""

import argparse
import sys

import numpy as np
import torch


def resized_shape_from_image(path, img_size=512):
    import cv2
    from mast3r_slam.mast3r_utils import resize_img

    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    res = resize_img(rgb, img_size)
    _, _, h, w = res["img"].shape
    return h, w


class _EncoderWrapper(torch.nn.Module):
    def __init__(self, model, true_shape):
        super().__init__()
        self.model = model
        self.register_buffer("true_shape", true_shape, persistent=False)

    def forward(self, img):
        feat, pos, _ = self.model._encode_image(img, self.true_shape)
        return feat, pos


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        default="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    )
    ap.add_argument("--output", default="checkpoints/mast3r_encoder.onnx")
    ap.add_argument("--img-height", type=int, default=384)
    ap.add_argument("--img-width", type=int, default=512)
    ap.add_argument("--probe-image", default="", help="derive HxW from this image")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from mast3r_slam.mast3r_utils import load_mast3r

    if args.probe_image:
        h, w = resized_shape_from_image(args.probe_image)
        print(f"[export] derived encoder input shape from image: {h}x{w}")
    else:
        h, w = args.img_height, args.img_width

    device = args.device
    model = load_mast3r(path=args.checkpoint, device=device).eval()

    true_shape = torch.tensor([[h, w]], dtype=torch.int32, device=device)
    wrapper = _EncoderWrapper(model, true_shape).to(device).eval()

    dummy = torch.randn(1, 3, h, w, device=device)

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (dummy,),
            args.output,
            input_names=["img"],
            output_names=["feat", "pos"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=None,  # fixed shape engine -> fastest, simplest
        )
    print(f"[export] wrote {args.output} (input img: 1x3x{h}x{w})")
    print("[export] next: tools/build_trt_engine.sh", args.output)


if __name__ == "__main__":
    sys.exit(main())
