"""GStreamer integration for MASt3R-SLAM.

Modules:
* ``element``      — the ``GstBase.BaseTransform`` element ``mast3rslam``;
* ``slam_runner``  — in-process port of ``main.py`` driving ``mast3r_slam.*``;
* ``backends``     — pluggable inference backends (CUDA / TensorRT-encoder);
* ``buffer_utils`` — ``Gst.Buffer`` → RGB float conversion;
* ``metadata``     — pose ``Gst.Structure`` for the bus.
"""

__all__ = []
