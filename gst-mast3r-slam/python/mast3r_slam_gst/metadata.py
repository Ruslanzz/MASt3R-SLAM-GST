"""Build the ``mast3r-slam-pose`` ``Gst.Structure`` posted on the bus.

This is the *online* output channel (the reference repo uses a 3D viewer for
this). The persistent ``.txt`` / ``.ply`` / keyframe outputs are written
verbatim via ``mast3r_slam.evaluate`` on EOS, so they are identical to the repo.

Quaternion convention follows ``lietorch`` / ``save_traj``: ``(qx, qy, qz, qw)``.
"""

from __future__ import annotations

POSE_MESSAGE_NAME = "mast3r-slam-pose"


def pose_to_structure(
    frame_id: int,
    timestamp_s: float,
    t_wc,
    mode: str,
    num_keyframes: int,
    is_keyframe: bool,
):
    """``t_wc`` is a ``lietorch.Sim3``; returns a named ``Gst.Structure``."""
    from gi.repository import Gst
    from mast3r_slam.lietorch_utils import as_SE3

    se3 = as_SE3(t_wc)
    tx, ty, tz, qx, qy, qz, qw = (float(v) for v in se3.data.numpy().reshape(-1))

    # Sim3 stores [t(3), q(4), s(1)]; expose the similarity scale too.
    try:
        scale = float(t_wc.data.detach().cpu().reshape(-1)[7])
    except Exception:
        scale = 1.0

    s = Gst.Structure.new_empty(POSE_MESSAGE_NAME)
    s.set_value("frame-id", int(frame_id))
    s.set_value("timestamp", float(timestamp_s))
    s.set_value("tx", tx)
    s.set_value("ty", ty)
    s.set_value("tz", tz)
    s.set_value("qx", qx)
    s.set_value("qy", qy)
    s.set_value("qz", qz)
    s.set_value("qw", qw)
    s.set_value("scale", scale)
    s.set_value("mode", str(mode))
    s.set_value("num-keyframes", int(num_keyframes))
    s.set_value("is-keyframe", bool(is_keyframe))
    return s
