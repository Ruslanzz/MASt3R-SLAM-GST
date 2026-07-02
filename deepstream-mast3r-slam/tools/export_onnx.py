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


def make_grid_pos(h, w, patch=16):
    """Patch-grid RoPE positions, row-major (y, x) — replicates PositionGetter.

    Computed OUTSIDE any traced graph: torch.cartesian_prod / meshgrid are
    multi-output ops that trip the legacy exporter's tuple-lowering pass with
    the same INTERNAL ASSERT as tuples containing None.
    """
    hp, wp = h // patch, w // patch
    y = torch.arange(hp)
    x = torch.arange(wp)
    return torch.cartesian_prod(y, x).view(1, hp * wp, 2)


class ExportRoPE2D(torch.nn.Module):
    """Export-friendly drop-in replacement for croco's RoPE2D / cuRoPE2D.

    The stock implementations cannot go through the legacy ONNX exporter:
    cuRoPE2D is a custom CUDA autograd Function, and the PyTorch fallback uses
    ``tokens.chunk(2)`` (a multi-output node — the remaining trigger of the
    tuple-lowering INTERNAL ASSERT), a python dict cache and
    ``int(positions.max())``. This module precomputes the cos/sin tables as
    constant buffers and uses tensor slicing only.
    """

    def __init__(self, freq, max_pos, head_dims):
        super().__init__()
        for hd in sorted({int(d) for d in head_dims}):
            D = hd // 2
            inv = 1.0 / (freq ** (torch.arange(0, D, 2).float() / D))
            t = torch.arange(max_pos).float()
            fr = torch.outer(t, inv)          # (max_pos, D/2)
            fr = torch.cat((fr, fr), dim=-1)  # (max_pos, D)
            self.register_buffer(f"cos_{D}", fr.cos(), persistent=False)
            self.register_buffer(f"sin_{D}", fr.sin(), persistent=False)

    def _apply1d(self, tok, pos1d, cos, sin):
        c = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
        s = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
        h = tok.shape[-1] // 2
        rot = torch.cat((-tok[..., h:], tok[..., :h]), dim=-1)  # rotate_half
        return tok * c + rot * s

    def forward(self, tokens, positions):
        D = tokens.size(3) // 2  # python int under trace (fixed shapes)
        cos = getattr(self, f"cos_{D}")
        sin = getattr(self, f"sin_{D}")
        y = tokens[..., :D]
        x = tokens[..., D:]
        y = self._apply1d(y, positions[:, :, 0].long(), cos, sin)
        x = self._apply1d(x, positions[:, :, 1].long(), cos, sin)
        return torch.cat((x, y), dim=-1)


def swap_rope_for_export(model, h, w):
    """Replace every rope reference (self-attention, cross-attention, model) by
    one shared ExportRoPE2D instance. Head dims are derived from the attention
    modules themselves (qkv for self-attn, projq for croco cross-attn)."""
    patch = model.patch_embed.patch_size[0]
    freq = 100.0
    rp = getattr(model, "rope", None)
    if rp is not None:
        freq = float(getattr(rp, "base", getattr(rp, "freq", 100.0)))

    head_dims = set()
    holders = []
    for m in model.modules():
        if getattr(m, "rope", None) is None or not hasattr(m, "num_heads"):
            continue
        if hasattr(m, "qkv"):        # croco Attention
            head_dims.add(m.qkv.in_features // m.num_heads)
            holders.append(m)
        elif hasattr(m, "projq"):    # croco CrossAttention
            head_dims.add(m.projq.in_features // m.num_heads)
            holders.append(m)

    max_pos = max(h // patch, w // patch)
    export_rope = ExportRoPE2D(freq, max_pos, head_dims)
    for m in holders:
        m.rope = export_rope
    if rp is not None:
        model.rope = export_rope
    print(f"[export] RoPE swapped for export: freq={freq} max_pos={max_pos} "
          f"head_dims={sorted(head_dims)} ({len(holders)} attention modules)")
    return export_rope


class EncoderWrapper(torch.nn.Module):
    """Tensor-only inline of dust3r's ``_encode_image``.

    Two legacy-exporter landmines are avoided:
      * ``_encode_image`` returns ``(x, pos, None)`` — the None element crashes
        ``_jit_pass_lower_all_tuples``;
      * ``patch_embed``'s PositionGetter builds pos with torch.cartesian_prod
        (meshgrid family, same crash), so pos is precomputed in ``__init__``
        and baked into the graph as a constant buffer.
    ``pos`` is emitted as int32 so the engine's output dtype matches what
    nvdsmast3rslam reads (FrameInput.pos_dev is int32).
    """

    def __init__(self, model, h, w):
        super().__init__()
        self.model = model
        patch = model.patch_embed.patch_size[0]
        self.register_buffer("pos", make_grid_pos(h, w, patch).to(torch.int64),
                             persistent=False)

    def forward(self, img):
        pe = self.model.patch_embed
        x = pe.proj(img).flatten(2).transpose(1, 2)
        x = pe.norm(x)
        pos = self.pos.expand(x.shape[0], -1, -1)
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


def trace_export(module, ex_args, out_path, input_names, output_names, opset):
    """Warm-up + export under no_grad.

    The warm-up forward fills the python-level caches (RoPE cos/sin tables,
    PositionGetter) so they enter the trace as plain tensor constants instead
    of being re-traced; no_grad (rather than inference_mode) keeps the JIT
    passes happy with the traced tensors.
    """
    with torch.no_grad():
        module(*ex_args)
        torch.onnx.export(module, ex_args, out_path,
                          input_names=input_names, output_names=output_names,
                          opset_version=opset, do_constant_folding=True)


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
    swap_rope_for_export(model, h, w)
    model = model.to(dev)  # move the freshly attached ExportRoPE2D buffers too
    true_shape = torch.tensor([[h, w]], dtype=torch.int32, device=dev)

    # token count N = (h/16)*(w/16); feat dim 1024
    n = (h // 16) * (w // 16)

    if args.which in ("encoder", "both"):
        enc = EncoderWrapper(model, h, w).to(dev).eval()
        dummy = torch.randn(1, 3, h, w, device=dev)
        trace_export(enc, (dummy,), args.out_encoder,
                     ["img"], ["feat", "pos"], args.opset)
        print(f"[export] encoder -> {args.out_encoder} (img 1x3x{h}x{w})")

    if args.which in ("decoder", "both"):
        dec = DecoderWrapper(model, true_shape, true_shape.clone()).to(dev).eval()
        f1 = torch.randn(1, n, 1024, device=dev)
        f2 = torch.randn(1, n, 1024, device=dev)
        # dummy pos MUST be the real patch grid, not zeros: RoPE bakes its
        # cos/sin tables sized max(pos)+1 into the graph, so zero positions
        # would bake 1-row tables and the engine would gather garbage at
        # runtime. int32 matches the encoder engine output / C++ core I/O.
        grid = make_grid_pos(h, w).to(torch.int32).to(dev)
        p1 = grid.clone()
        p2 = grid.clone()
        trace_export(dec, (f1, p1, f2, p2), args.out_decoder,
                     ["feat1", "pos1", "feat2", "pos2"],
                     ["pts3d_1", "conf_1", "desc_1", "desc_conf_1",
                      "pts3d_2", "conf_2", "desc_2", "desc_conf_2"],
                     args.opset)
        print(f"[export] decoder -> {args.out_decoder} (feat 1x{n}x1024)")


if __name__ == "__main__":
    main()
