"""Pluggable inference backend interface.

The neural network part of MASt3R-SLAM (the MASt3R two-view model) is the only
piece that *can* sensibly be accelerated by TensorRT.  Everything else — feature
matching, Sim3 pose tracking, the global Gauss-Newton optimisation and the ASMK
retrieval database — relies on torch, lietorch and the custom CUDA kernels
compiled into ``mast3r_slam_backends`` and therefore stays on CUDA regardless of
the chosen backend (see DESIGN.md, section 2).

A backend's single responsibility is to produce the ``AsymmetricMASt3R`` model
object that the rest of the unchanged ``mast3r_slam`` code drives.  The TensorRT
backend additionally swaps ``model._encode_image`` for a TRT-engine call, so all
existing call sites (``mast3r_match_asymmetric``, ``mast3r_inference_mono``,
``FactorGraph.add_factors`` ...) transparently use the engine for the heavy ViT
encoder while keeping the decoder / DPT heads / geometry on torch.
"""

from __future__ import annotations

import abc


class InferenceBackend(abc.ABC):
    """Builds and returns the torch MASt3R model used by the SLAM pipeline."""

    name: str = "base"

    @abc.abstractmethod
    def load_model(self, checkpoint: str, device: str):
        """Return a ready-to-use ``AsymmetricMASt3R`` instance on ``device``.

        Implementations may wrap ``model._encode_image`` (and only that) to run
        on an alternative runtime; the returned object must remain
        API-compatible with the stock model so the rest of ``mast3r_slam`` is
        untouched.
        """
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


def make_backend(kind: str) -> InferenceBackend:
    """Factory: ``kind`` is the value of the element's ``backend`` property."""
    kind = (kind or "torch").lower()
    if kind in ("torch", "cuda", "torch_cuda", "pytorch"):
        from mast3r_slam_gst.backends.torch_cuda import TorchCudaBackend

        return TorchCudaBackend()
    if kind in ("tensorrt", "trt", "tensorrt_encoder"):
        from mast3r_slam_gst.backends.tensorrt_encoder import TensorRTEncoderBackend

        return TensorRTEncoderBackend()
    raise ValueError(
        f"Unknown inference backend '{kind}'. Use 'torch' or 'tensorrt'."
    )
