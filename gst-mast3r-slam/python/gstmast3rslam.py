"""gst-python plugin entry point for the ``mast3rslam`` element.

The gst-python loader (``libgstpython``) scans the ``python`` sub-directory of
every ``GST_PLUGIN_PATH`` entry, imports each module and registers any element
declared via ``__gstelementfactory__``.

Requires ``GST_PLUGIN_PATH`` to point at ``gst-mast3r-slam`` (so this file lives
in ``<path>/python/``) and ``PYTHONPATH`` to include ``gst-mast3r-slam/python``
so that ``mast3r_slam_gst`` is importable. See ``setup_env.sh``.
"""

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GObject, Gst  # noqa: E402

from mast3r_slam_gst.element import GstMast3rSlam  # noqa: E402

GObject.type_register(GstMast3rSlam)

# (factory-name, rank, GType) — picked up automatically by the python loader.
__gstelementfactory__ = ("mast3rslam", Gst.Rank.NONE, GstMast3rSlam)
