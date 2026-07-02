#!/usr/bin/env python3
"""Export MASt3R to ONNX for the DeepStream pipeline.

Two engines are needed:
  * encoder  — run by gst-nvinfer (input: img 1x3xHxW; outputs: feat, pos)
  * decoder  — run by nvdsmast3rslam (inputs: feat1,pos1,feat2,pos2;
               outputs: pts3d_1,conf_1,desc_1,desc_conf_1,
                        pts3d_2,conf_2,desc_2,desc_conf_2)

Output layer names MUST match config_infer_mast3r_encoder.txt (feat;pos) and
lib/mast3r_slam_core.cpp (the decoder I/O names above).

Then build engines with trtexec, e.g.:
  trtexec --onnx=checkpoints/mast3r_encoder.onnx \\
          --saveEngine=checkpoints/mast3r_encoder.engine --fp16
  trtexec --onnx=checkpoints/mast3r_decoder.onnx \\
          --saveEngine=checkpoints/mast3r_decoder.engine --fp16

CAVEATS: CroCo/MASt3R uses RoPE + masked attention; ONNX export of the decoder
in particular is delicate. This is a starting point — validate the exported
graph and tweak opset/wrapper as needed.
"""
import argparse

import numpy as np
import torch


class EncoderWrapper(torch.nn.Module):
    """Inlines dust3r's ``_encode_image`` body.

    Calling ``_encode_image`` directly puts a ``(x, pos, None)`` tuple into the
    traced graph; the legacy ONNX exporter's tuple-lowering pass crashes on the
    None element with an INTERNAL ASSERT in dead_code_elimination.cpp. Inlining
    the body keeps the graph tensors-only. ``pos`` is returned as int32 so the
    TensorRT engine's output dtype matches what nvdsmast3rslam reads
    (FrameInput.pos_dev is int32).
    """

    def __init__(self, model, true_shape):
        super().__init__()
        self.model = model
        self.register_buffer("true_shape", true_shape, persistent=False)

    def forward(self, img):
        x, pos = self.model.patch_embed(img, true_shape=self.true_shape)
        for blk in self.model.enc_blocks:
            x = blk(x, pos)
        x = self.model.enc_norm(x)
        return x, pos.to(torch.int32)


class DecoderWrapper(torch.nn.Module):
    """Single decoder pass producing both views' heads (see mast3r_utils.decoder).

    ``pos1``/``pos2`` are int32 inputs (cast to long internally): the C++ core
    binds int32 tensors to the engine, and the encoder engine emits int32 pos.
    """

    def __init__(self, model, shape1, shape2):
        super().__init__()
        self.model = model
        self.register_buffer("shape1", shape1, persistent=False)
        self.register_buffer("shape2", shape2, persistent=False)

    def forward(self, feat1, pos1, feat2, pos2):
        dec1, dec2 = self.model._decoder(feat1, pos1.long(), feat2, pos2.long())
        with torch.amp.autocast(enabled=False, device_type="cuda"):
            res1 = self.model._downstream_head(1, [t.float() for t in dec1], self.shape1)
            res2 = self.model._downstream_head(2, [t.float() for t in dec2], self.shape2)
        return (
            res1["pts3d"], res1["conf"], res1["desc"], res1["desc_conf"],
            res2["pts3d"], res2["conf"], res2["desc"], res2["desc_conf"],
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",
                    default="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    ap.add_argument("--out-encoder", default="checkpoints/mast3r_encoder.onnx")
    ap.add_argument("--out-decoder", default="checkpoints/mast3r_decoder.onnx")
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--which", choices=["encoder", "decoder", "both"], default="both")
    args = ap.parse_args()

    from mast3r_slam.mast3r_utils import load_mast3r

    dev = args.device
    h, w = args.height, args.width
    model = load_mast3r(path=args.checkpoint, device=dev).eval()
    true_shape = torch.tensor([[h, w]], dtype=torch.int32, device=dev)

    # token count N = (h/16)*(w/16); feat dim 1024
    n = (h // 16) * (w // 16)

    if args.which in ("encoder", "both"):
        enc = EncoderWrapper(model, true_shape).to(dev).eval()
        dummy = torch.randn(1, 3, h, w, device=dev)
        with torch.inference_mode():
            torch.onnx.export(enc, (dummy,), args.out_encoder,
                              input_names=["img"], output_names=["feat", "pos"],
                              opset_version=args.opset, do_constant_folding=True)
        print(f"[export] encoder -> {args.out_encoder} (img 1x3x{h}x{w})")

    if args.which in ("decoder", "both"):
        dec = DecoderWrapper(model, true_shape, true_shape.clone()).to(dev).eval()
        f1 = torch.randn(1, n, 1024, device=dev)
        f2 = torch.randn(1, n, 1024, device=dev)
        # int32 to match the encoder engine's pos output and the C++ core's I/O
        p1 = torch.zeros(1, n, 2, dtype=torch.int32, device=dev)
        p2 = torch.zeros(1, n, 2, dtype=torch.int32, device=dev)
        with torch.inference_mode():
            torch.onnx.export(
                dec, (f1, p1, f2, p2), args.out_decoder,
                input_names=["feat1", "pos1", "feat2", "pos2"],
                output_names=["pts3d_1", "conf_1", "desc_1", "desc_conf_1",
                              "pts3d_2", "conf_2", "desc_2", "desc_conf_2"],
                opset_version=args.opset, do_constant_folding=True)
        print(f"[export] decoder -> {args.out_decoder} (feat 1x{n}x1024)")


if __name__ == "__main__":
    main()
