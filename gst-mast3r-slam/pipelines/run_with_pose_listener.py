#!/usr/bin/env python3
"""Run a MASt3R-SLAM pipeline and consume the live pose messages.

This shows how a real application (not gst-launch) reads the online output: it
watches the bus for ``mast3r-slam-pose`` element messages and can stream them to
a TUM-format file as they arrive, while the persistent ``.ply``/``.txt`` are
still written by the element on EOS.

Example:
  source gst-mast3r-slam/setup_env.sh
  python gst-mast3r-slam/pipelines/run_with_pose_listener.py \\
      "filesrc location=video.mp4 ! decodebin3 ! nvvideoconvert ! \\
       video/x-raw,format=RGBA ! mast3rslam config=config/base.yaml \\
       sequence-name=video ! fakesink sync=false" \\
      --live-traj logs/video_live.txt
"""

import argparse
import sys

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

from mast3r_slam_gst.metadata import POSE_MESSAGE_NAME  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pipeline", help="gst-launch style pipeline description")
    ap.add_argument("--live-traj", default="", help="stream poses to this file")
    args = ap.parse_args()

    Gst.init(None)
    pipeline = Gst.parse_launch(args.pipeline)
    loop = GLib.MainLoop()

    traj_fh = open(args.live_traj, "w") if args.live_traj else None

    def on_message(_bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("[listener] EOS")
            pipeline.set_state(Gst.State.NULL)
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            print(f"[listener] ERROR: {err} ({dbg})", file=sys.stderr)
            pipeline.set_state(Gst.State.NULL)
            loop.quit()
        elif t == Gst.MessageType.ELEMENT:
            s = message.get_structure()
            if s and s.get_name() == POSE_MESSAGE_NAME:
                fid = s.get_value("frame-id")
                ts = s.get_value("timestamp")
                tx, ty, tz = s.get_value("tx"), s.get_value("ty"), s.get_value("tz")
                qx, qy, qz, qw = (
                    s.get_value("qx"), s.get_value("qy"),
                    s.get_value("qz"), s.get_value("qw"),
                )
                kf = "*" if s.get_value("is-keyframe") else " "
                print(
                    f"[pose]{kf} f={fid:>5} mode={s.get_value('mode'):<8} "
                    f"kf={s.get_value('num-keyframes'):>4} "
                    f"t=({tx:+.3f},{ty:+.3f},{tz:+.3f})"
                )
                if traj_fh is not None:
                    traj_fh.write(
                        f"{ts} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n"
                    )
                    traj_fh.flush()

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message)

    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except KeyboardInterrupt:
        print("[listener] interrupted -> sending EOS")
        pipeline.send_event(Gst.Event.new_eos())
        # give the element a moment to flush results before tearing down
        bus.timed_pop_filtered(5 * Gst.SECOND, Gst.MessageType.EOS)
        pipeline.set_state(Gst.State.NULL)
    finally:
        if traj_fh is not None:
            traj_fh.close()


if __name__ == "__main__":
    main()
