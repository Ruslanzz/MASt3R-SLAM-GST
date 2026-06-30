# Example pipelines

All scripts source `../setup_env.sh` first, so run them from the repository root
inside the DeepStream container (or any host where the plugin and deps are set
up). They use `gst-launch-1.0 -m -e`:

* `-m` prints bus messages, including the `mast3r-slam-pose` element messages;
* `-e` forwards EOS on Ctrl-C so the element writes `logs/<seq>.txt` (TUM
  trajectory), `logs/<seq>.ply` (point cloud) and `logs/keyframes/<seq>/*.png`.

| Script | Input transport |
|--------|-----------------|
| `v4l2_camera.sh [/dev/videoN]` | USB v4l2 camera (dGPU) or CSI/Argus (Jetson, commented) |
| `video_file.sh <video> [cfg] [calib]` | local video file, HW-decoded |
| `udp_rtsp.sh udp [port]` / `udp_rtsp.sh rtsp <url>` | RTP/H264 over UDP, or RTSP |

## Common element layout

```
<source> ! <decode> ! nvvideoconvert ! video/x-raw,format=RGBA ! mast3rslam ... ! <sink>
```

* `nvvideoconvert ! video/x-raw,format=RGBA` copies the frame from NVMM (GPU) to
  system memory in the RGBA layout the element maps to numpy.
* `mast3rslam` is **passthrough** — put any downstream NVIDIA element after it
  (`nvdsosd`, `nvv4l2h264enc`, `nv3dsink`, ...). Replace `fakesink` to e.g. also
  display the video.

## Reading poses from an application

`gst-launch` only prints the messages. A real app uses the bus, see
`run_with_pose_listener.py`, which also streams a live TUM trajectory.

## Why monocular single-source (not two cameras)

MASt3R-SLAM is monocular; the two-view "pair" the network needs is formed
internally (current frame ↔ reference keyframe). A single stream therefore
reproduces the reference trajectory output exactly. Feeding two independent
cameras would be stereo two-view reconstruction, a different task. See
`../DESIGN.md` section 3.
