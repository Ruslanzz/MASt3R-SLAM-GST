"""Optional TensorRT backend — accelerates **only** the ViT encoder.

Rationale (see DESIGN.md, section 2): ``model._encode_image`` is the heaviest
and most static part of MASt3R (run once per frame at a fixed 512-long-edge
resolution, i.e. a fixed token count).  It is the single piece that can be moved
to TensorRT without breaking the asymmetric, dynamically-shaped decoder, the DPT
heads or the CUDA-based SLAM geometry, all of which stay on torch.

The backend loads the stock torch model (used for the decoder / heads / matching
/ optimisation) and replaces ``model._encode_image`` with a call into a prebuilt
TensorRT engine.  Build the engine with ``tools/export_encoder_onnx.py`` +
``tools/build_trt_engine.sh``.

NOTE: FP16/INT8 engines change numerics, so SLAM output will *drift* from the
reference. Use this backend when throughput matters more than bit-exactness.
"""

from __future__ import annotations

import os

from mast3r_slam_gst.backends.base import InferenceBackend


def _trt_dtype_to_torch(trt, dtype):
    import torch

    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.BOOL: torch.bool,
    }
    if hasattr(trt.DataType, "INT8"):
        mapping[trt.DataType.INT8] = torch.int8
    return mapping[dtype]


class _TRTEngine:
    """Minimal TensorRT 10 runner that uses torch CUDA tensors as I/O buffers."""

    def __init__(self, engine_path: str, device: str):
        import tensorrt as trt  # lazy
        import torch

        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        self.trt = trt
        self.torch = torch
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        self.inputs, self.outputs = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(name)
            else:
                self.outputs.append(name)

    def infer(self, feeds: dict):
        torch = self.torch
        trt = self.trt
        stream = torch.cuda.current_stream(self.device)

        for name, tensor in feeds.items():
            tensor = tensor.to(self.device).contiguous()
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())
            feeds[name] = tensor  # keep ref alive

        results = {}
        for name in self.outputs:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = _trt_dtype_to_torch(trt, self.engine.get_tensor_dtype(name))
            out = torch.empty(shape, dtype=dtype, device=self.device)
            self.context.set_tensor_address(name, out.data_ptr())
            results[name] = out

        ok = self.context.execute_async_v3(stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        stream.synchronize()
        return results


class TensorRTEncoderBackend(InferenceBackend):
    name = "tensorrt"

    def __init__(self, engine_path: str = ""):
        self.engine_path = engine_path

    def configure(self, engine_path: str):
        self.engine_path = engine_path

    def load_model(self, checkpoint: str, device: str):
        import torch
        from mast3r_slam.mast3r_utils import load_mast3r

        engine_path = self.engine_path or os.environ.get(
            "MAST3R_TRT_ENCODER_ENGINE", ""
        )
        if not engine_path:
            raise ValueError(
                "backend=tensorrt requires the 'trt-encoder-engine' property "
                "(or MAST3R_TRT_ENCODER_ENGINE env var) pointing at the .engine"
            )

        path = checkpoint if checkpoint else None
        model = load_mast3r(path=path, device=device)
        model.share_memory()

        engine = _TRTEngine(engine_path, device)
        model_dtype = next(model.parameters()).dtype

        # The engine takes the normalized image and returns the encoder tokens
        # (feat) and the patch positions (pos), mirroring the 3-tuple returned
        # by the stock ``_encode_image`` (the third element is unused / None).
        @torch.inference_mode()
        def _encode_image_trt(image, true_shape):
            feeds = {"img": image.to(device).to(model_dtype)}
            out = engine.infer(feeds)
            feat = out["feat"].to(model_dtype)
            pos = out["pos"].to(torch.long)
            return feat, pos, None

        model._encode_image = _encode_image_trt  # transparent swap
        return model
