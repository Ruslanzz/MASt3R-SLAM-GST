"""``mast3rslam`` GStreamer element (``GstBase.BaseTransform``).

The element consumes a single monocular ``video/x-raw`` stream (RGBA/RGB in
system memory, produced upstream by ``nvvideoconvert``), runs MASt3R-SLAM, and:

* passes every buffer downstream **unmodified** so official NVIDIA elements
  (``nvosd``, encoders, sinks ...) keep working;
* posts the per-frame camera pose on the bus as a ``mast3r-slam-pose`` element
  message (the online output channel);
* on EOS / shutdown writes ``<sequence-name>.txt`` (TUM trajectory),
  ``<sequence-name>.ply`` (point cloud) and keyframe PNGs via the unchanged
  ``mast3r_slam.evaluate`` — i.e. the same output as the reference repo.

Heavy imports (torch, mast3r_slam) are deferred to ``do_start`` so that
``gst-inspect`` can introspect the plugin cheaply.
"""

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import GObject, Gst, GstBase, GstVideo  # noqa: E402

FIXED_CAPS = Gst.Caps.from_string(
    "video/x-raw, "
    "format=(string){ RGBA, RGB }, "
    "width=(int)[1,2147483647], "
    "height=(int)[1,2147483647], "
    "framerate=(fraction)[0/1,2147483647/1]"
)

DEFAULTS = {
    "config": "config/base.yaml",
    "calib": "",
    "checkpoint": "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    "retrieval-checkpoint": "",
    "backend": "torch",
    "trt-encoder-engine": "",
    "device": "cuda:0",
    "save-dir": "logs",
    "sequence-name": "mast3rslam",
    "save-results": True,
    "conf-threshold": 1.5,
    "backend-thread": True,
    "max-fps": 0.0,
}


class GstMast3rSlam(GstBase.BaseTransform):
    __gstmetadata__ = (
        "MASt3R-SLAM",
        "Filter/Analyzer/Video",
        "Monocular dense SLAM (MASt3R-SLAM) running on CUDA, optional TensorRT "
        "encoder. Passthrough video; emits poses on the bus and saves "
        "trajectory/point-cloud on EOS.",
        "MASt3R-SLAM GStreamer integration",
    )

    __gsttemplates__ = (
        Gst.PadTemplate.new(
            "sink",
            Gst.PadDirection.SINK,
            Gst.PadPresence.ALWAYS,
            FIXED_CAPS,
        ),
        Gst.PadTemplate.new(
            "src",
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            FIXED_CAPS,
        ),
    )

    __gproperties__ = {
        "config": (
            str, "config", "Path to SLAM config YAML",
            DEFAULTS["config"], GObject.ParamFlags.READWRITE,
        ),
        "calib": (
            str, "calib", "Path to calibration/intrinsics YAML (enables use_calib)",
            DEFAULTS["calib"], GObject.ParamFlags.READWRITE,
        ),
        "checkpoint": (
            str, "checkpoint", "Path to MASt3R weights (.pth)",
            DEFAULTS["checkpoint"], GObject.ParamFlags.READWRITE,
        ),
        "retrieval-checkpoint": (
            str, "retrieval-checkpoint",
            "Path to retrieval weights (empty = repo default)",
            DEFAULTS["retrieval-checkpoint"], GObject.ParamFlags.READWRITE,
        ),
        "backend": (
            str, "backend", "Inference backend: 'torch' (CUDA) or 'tensorrt'",
            DEFAULTS["backend"], GObject.ParamFlags.READWRITE,
        ),
        "trt-encoder-engine": (
            str, "trt-encoder-engine",
            "Path to the TensorRT encoder .engine (backend=tensorrt)",
            DEFAULTS["trt-encoder-engine"], GObject.ParamFlags.READWRITE,
        ),
        "device": (
            str, "device", "Torch device",
            DEFAULTS["device"], GObject.ParamFlags.READWRITE,
        ),
        "save-dir": (
            str, "save-dir", "Directory for trajectory/point-cloud/keyframes",
            DEFAULTS["save-dir"], GObject.ParamFlags.READWRITE,
        ),
        "sequence-name": (
            str, "sequence-name", "Base name of output files",
            DEFAULTS["sequence-name"], GObject.ParamFlags.READWRITE,
        ),
        "save-results": (
            bool, "save-results", "Save trajectory/point-cloud/keyframes on EOS",
            DEFAULTS["save-results"], GObject.ParamFlags.READWRITE,
        ),
        "conf-threshold": (
            float, "conf-threshold",
            "Confidence threshold used when exporting the .ply",
            0.0, 1e9, DEFAULTS["conf-threshold"], GObject.ParamFlags.READWRITE,
        ),
        "backend-thread": (
            bool, "backend-thread",
            "Run the global optimisation in a background thread",
            DEFAULTS["backend-thread"], GObject.ParamFlags.READWRITE,
        ),
        "max-fps": (
            float, "max-fps",
            "Throttle SLAM processing to this rate (0 = process every frame)",
            0.0, 1e6, DEFAULTS["max-fps"], GObject.ParamFlags.READWRITE,
        ),
    }

    def __init__(self):
        super().__init__()
        self._props = dict(DEFAULTS)
        self._runner = None
        self._video_info = None
        self._format = "RGBA"
        self._last_pts_ns = None
        self._frame_index = 0
        self._error = False
        self.set_in_place(True)
        self.set_passthrough(False)

    # ---------------------------------------------------------- GObject props
    def do_get_property(self, prop):
        try:
            return self._props[prop.name]
        except KeyError:
            raise AttributeError(f"unknown property {prop.name}")

    def do_set_property(self, prop, value):
        if prop.name not in self._props:
            raise AttributeError(f"unknown property {prop.name}")
        self._props[prop.name] = value

    # --------------------------------------------------------------- lifecycle
    def do_set_caps(self, incaps, outcaps):
        self._video_info = GstVideo.VideoInfo.new_from_caps(incaps)
        structure = incaps.get_structure(0)
        fmt = structure.get_string("format") if structure else None
        self._format = fmt or "RGBA"
        return True

    def do_start(self):
        try:
            from mast3r_slam_gst.slam_runner import Mast3rSlamRunner

            self._runner = Mast3rSlamRunner(
                config_path=self._props["config"],
                calib_path=self._props["calib"],
                checkpoint=self._props["checkpoint"],
                retrieval_checkpoint=self._props["retrieval-checkpoint"],
                backend=self._props["backend"],
                trt_encoder_engine=self._props["trt-encoder-engine"],
                device=self._props["device"],
                save_dir=self._props["save-dir"],
                sequence_name=self._props["sequence-name"],
                save_results=self._props["save-results"],
                conf_threshold=self._props["conf-threshold"],
                backend_thread=self._props["backend-thread"],
            )
            self._runner.start()
        except Exception as exc:  # pragma: no cover - runtime only
            Gst.error(f"mast3rslam: failed to start SLAM runner: {exc}")
            self._error = True
            return False
        return True

    def do_stop(self):
        if self._runner is not None:
            try:
                self._runner.finish()
            except Exception as exc:  # pragma: no cover
                Gst.warning(f"mast3rslam: error finalising: {exc}")
        return True

    def do_sink_event(self, event):
        if event.type == Gst.EventType.EOS and self._runner is not None:
            try:
                self._runner.finish()
            except Exception as exc:  # pragma: no cover
                Gst.warning(f"mast3rslam: error on EOS: {exc}")
        return GstBase.BaseTransform.do_sink_event(self, event)

    # ----------------------------------------------------------- per-buffer
    def _should_process(self, pts_ns):
        max_fps = self._props["max-fps"]
        if max_fps <= 0.0 or pts_ns == Gst.CLOCK_TIME_NONE:
            return True
        if self._last_pts_ns is None:
            return True
        min_interval_ns = 1e9 / max_fps
        return (pts_ns - self._last_pts_ns) >= min_interval_ns

    def do_transform_ip(self, buf):
        # Always pass the buffer downstream unmodified, even on error.
        if self._error or self._runner is None or self._video_info is None:
            return Gst.FlowReturn.OK

        pts_ns = buf.pts
        if not self._should_process(pts_ns):
            return Gst.FlowReturn.OK
        self._last_pts_ns = pts_ns

        if pts_ns != Gst.CLOCK_TIME_NONE:
            timestamp_s = pts_ns / 1e9
        else:
            fps_n = self._video_info.fps_n or 30
            fps_d = self._video_info.fps_d or 1
            timestamp_s = self._frame_index * fps_d / fps_n

        try:
            from mast3r_slam_gst.buffer_utils import buffer_to_rgb_float

            img = buffer_to_rgb_float(buf, self._video_info, self._format)
            pose = self._runner.process(timestamp_s, img)
            self._frame_index += 1
            if pose is not None:
                self._post_pose(pose)
        except Exception as exc:  # pragma: no cover - runtime only
            Gst.warning(f"mast3rslam: frame processing error: {exc}")

        return Gst.FlowReturn.OK

    def _post_pose(self, pose):
        from mast3r_slam_gst.metadata import pose_to_structure

        structure = pose_to_structure(
            pose["frame_id"],
            pose["timestamp"],
            pose["T_WC"],
            pose["mode"],
            pose["num_keyframes"],
            pose["is_keyframe"],
        )
        self.post_message(Gst.Message.new_element(self, structure))


def register(plugin):
    GObject.type_register(GstMast3rSlam)
    return Gst.Element.register(
        plugin, "mast3rslam", Gst.Rank.NONE, GstMast3rSlam
    )
