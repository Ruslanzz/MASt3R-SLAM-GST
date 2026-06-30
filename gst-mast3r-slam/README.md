# MASt3R-SLAM GStreamer plugin (`mast3rslam`)

A GStreamer 1.24 element that runs [MASt3R-SLAM](../README.md) on the GPU inside a
pipeline, built to live alongside the official NVIDIA DeepStream elements. It
ingests a monocular video stream (v4l2 camera / file / UDP / RTSP), runs the full
SLAM, passes the video through unmodified, emits camera poses on the bus, and on
EOS writes **the same outputs as the reference repository**: a TUM trajectory
`.txt`, a colored point-cloud `.ply`, and keyframe PNGs.

> Design rationale — including the **CUDA vs TensorRT** trade-off and the input
> choice — is in [`DESIGN.md`](DESIGN.md). Short version: run the network on
> **CUDA/PyTorch by default** (bit-identical to the repo); TensorRT is an
> *optional* accelerator for the ViT encoder only, because the matching, pose
> tracking and global optimisation are custom CUDA kernels that cannot move to
> TensorRT.

## Layout

```
gst-mast3r-slam/
├── DESIGN.md                     # architecture + CUDA/TensorRT analysis (RU)
├── setup_env.sh                  # sets GST_PLUGIN_PATH / PYTHONPATH
├── python/
│   ├── gstmast3rslam.py          # gst-python plugin entry (__gstelementfactory__)
│   └── mast3r_slam_gst/
│       ├── element.py            # the GstBase.BaseTransform element
│       ├── slam_runner.py        # in-process port of main.py
│       ├── buffer_utils.py       # Gst.Buffer -> RGB float
│       ├── metadata.py           # pose -> Gst.Structure (bus message)
│       └── backends/             # torch_cuda (default) | tensorrt_encoder
├── docker/                       # Dockerfile (DeepStream 9.0) + build/run
├── tools/                        # ONNX export + TRT engine build (TensorRT path)
└── pipelines/                    # v4l2 / file / UDP-RTSP examples + pose listener
```

## Build (DeepStream 9.0)

```bash
# from the repository root, with submodules initialised and checkpoints/ present
bash gst-mast3r-slam/docker/build.sh
bash gst-mast3r-slam/docker/run.sh          # GPU + cameras + repo mounted
```

The image installs PyTorch, compiles the SLAM CUDA backend (`mast3r_slam_backends`,
lietorch), wires `GST_PLUGIN_PATH`/`PYTHONPATH`, and verifies the element loads
(`gst-inspect-1.0 mast3rslam`). See `docker/Dockerfile` for the `BASE_IMAGE` /
`TORCH_CUDA_ARCH_LIST` build args (set the latter to your GPU arch).

## Register the element without Docker

```bash
source gst-mast3r-slam/setup_env.sh
gst-inspect-1.0 mast3rslam
```
(Requires the repo's Python deps installed, i.e. `pip install -e thirdparty/mast3r`
and `pip install -e .`, plus `python3-gi` / `gstreamer1.0-python3-plugin-loader`.)

## Run

```bash
# checkpoints/ must contain the MASt3R + retrieval weights (see ../README.md)
bash gst-mast3r-slam/pipelines/video_file.sh /path/to/video.mp4
bash gst-mast3r-slam/pipelines/v4l2_camera.sh /dev/video0
bash gst-mast3r-slam/pipelines/udp_rtsp.sh udp 5000
# outputs: logs/<seq>.txt, logs/<seq>.ply, logs/keyframes/<seq>/*.png
```

## Element reference

`mast3rslam` — `Filter/Analyzer/Video`, passthrough, `video/x-raw {RGBA,RGB}`.

| Property | Default | Meaning |
|----------|---------|---------|
| `config` | `config/base.yaml` | SLAM config YAML |
| `calib` | `""` | intrinsics YAML → enables calibrated mode |
| `checkpoint` | MASt3R metric `.pth` | model weights |
| `retrieval-checkpoint` | repo default | retrieval weights |
| `backend` | `torch` | `torch` (CUDA) or `tensorrt` |
| `trt-encoder-engine` | `""` | encoder `.engine` for `backend=tensorrt` |
| `device` | `cuda:0` | torch device |
| `save-dir` / `sequence-name` | `logs` / `mast3rslam` | output location/name |
| `save-results` | `true` | write `.txt`/`.ply`/keyframes on EOS |
| `conf-threshold` | `1.5` | confidence filter for the `.ply` |
| `backend-thread` | `true` | run global optimisation in a thread |
| `max-fps` | `0` | throttle processing (0 = every frame) |

**Bus output** — element message `mast3r-slam-pose` with fields `frame-id`,
`timestamp`, `tx ty tz`, `qx qy qz qw` (lietorch order), `scale`, `mode`,
`num-keyframes`, `is-keyframe`. See `pipelines/run_with_pose_listener.py`.

## Optional: TensorRT encoder backend

```bash
# 1) export the encoder to ONNX (derive shape from a sample frame)
python gst-mast3r-slam/tools/export_encoder_onnx.py --probe-image sample.png \
       --output checkpoints/mast3r_encoder.onnx
# 2) build a fixed-shape engine
bash gst-mast3r-slam/tools/build_trt_engine.sh checkpoints/mast3r_encoder.onnx \
       checkpoints/mast3r_encoder.engine fp16
# 3) use it
... ! mast3rslam backend=tensorrt \
        trt-encoder-engine=checkpoints/mast3r_encoder.engine ! ...
```
FP16/INT8 changes numerics, so SLAM output drifts from the bit-exact CUDA path —
use this only when throughput matters more than reproducing the reference.

## Status / limitations

* The element copies frames NVMM→system memory for the Python/torch boundary;
  zero-copy GPU ingest is listed under "future work" in `DESIGN.md`.
* Loop closure / relocalisation need the ASMK retrieval module
  (`thirdparty/mast3r/asmk`); without it tracking still runs.
* Not runnable without a CUDA GPU, the downloaded checkpoints, and the compiled
  `mast3r_slam_backends` extension.
