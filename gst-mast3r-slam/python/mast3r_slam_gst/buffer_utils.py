"""Helpers to turn a mapped ``Gst.Buffer`` into the RGB float image the SLAM
front-end expects (``HxWx3`` float32 in ``[0, 1]``, channel order RGB), matching
``mast3r_slam.dataloader.MonocularDataset.get_image``.

The element negotiates ``video/x-raw`` with format ``RGBA`` or ``RGB`` in system
memory; upstream DeepStream elements copy NVMM → system memory with
``nvvideoconvert ! video/x-raw,format=RGBA``.
"""

from __future__ import annotations

import numpy as np


_BYTES_PER_PIXEL = {"RGBA": 4, "RGBx": 4, "RGB": 3}


def buffer_to_rgb_float(gst_buffer, video_info, fmt="RGBA") -> np.ndarray:
    """Return an ``HxWx3`` float32 RGB image in ``[0, 1]``.

    ``video_info`` is a ``GstVideo.VideoInfo`` describing the negotiated caps and
    is used to honour the row stride (NVIDIA converters often align rows). ``fmt``
    is the negotiated pixel format string (``RGBA``/``RGBx``/``RGB``), taken from
    the caps by the element so we never guess the bytes-per-pixel.
    """
    from gi.repository import Gst  # local import; gi is set up by the plugin

    bpp = _BYTES_PER_PIXEL.get((fmt or "RGBA").upper())
    if bpp is None:
        raise ValueError(f"Unsupported pixel format for MASt3R-SLAM: {fmt}")

    width = video_info.width
    height = video_info.height
    stride = video_info.get_stride(0)

    ok, mapinfo = gst_buffer.map(Gst.MapFlags.READ)
    if not ok:
        raise RuntimeError("Failed to map Gst.Buffer for reading")
    try:
        raw = np.frombuffer(mapinfo.data, dtype=np.uint8, count=stride * height)
        rows = raw.reshape(height, stride)
        # Drop any row padding, then split into pixels.
        usable = rows[:, : width * bpp].reshape(height, width, bpp)
        rgb = usable[:, :, :3]  # RGBA/RGBx → RGB (RGB byte order is already RGB)
        # Copy out of the mapped memory before unmapping.
        out = rgb.astype(np.float32) / 255.0
    finally:
        gst_buffer.unmap(mapinfo)
    return out
