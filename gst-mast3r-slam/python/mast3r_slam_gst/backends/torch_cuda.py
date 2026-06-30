"""Default CUDA / PyTorch backend.

This is a thin wrapper around the upstream ``load_mast3r`` so that the SLAM
output is **bit-for-bit identical** to the reference repository: nothing about
the model or its numerics is changed.
"""

from __future__ import annotations

from mast3r_slam_gst.backends.base import InferenceBackend


class TorchCudaBackend(InferenceBackend):
    name = "torch"

    def load_model(self, checkpoint: str, device: str):
        # Imported lazily so that merely scanning the plugin does not pull torch.
        from mast3r_slam.mast3r_utils import load_mast3r

        path = checkpoint if checkpoint else None
        model = load_mast3r(path=path, device=device)
        model.share_memory()
        return model
