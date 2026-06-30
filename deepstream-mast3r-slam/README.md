# DeepStream MASt3R-SLAM — native C/C++ plugin (`nvdsmast3rslam`)

A C/C++ GStreamer/DeepStream plugin that runs MASt3R-SLAM on the GPU, built
**around the official `gst-nvinfer`**: `gst-nvinfer` runs the MASt3R ViT encoder
as a TensorRT engine and emits its tensors as `NvDsInferTensorMeta`; the custom
`nvdsmast3rslam` element consumes those tensors and performs the decoder +
matching + Sim3 tracking + factor-graph backend, attaches the camera pose as
`NvDsUserMeta`, and on EOS writes the same outputs as the reference repo
(TUM trajectory `.txt`, colored `.ply`).

> Full rationale, data flow and the encoder/decoder split are in
> [`DESIGN.md`](DESIGN.md). This is the C/C++ / gst-nvinfer counterpart of the
> Python plugin on branch `claude/gstreamer-cuda-tensorrt-plugin-g29qxg`.

```
 source ─▶ nvstreammux(batch=1) ─▶ nvinfer (MASt3R encoder, TRT)
                                       │ NvDsInferTensorMeta(feat,pos)
                                       ▼
                              nvdsmast3rslam (C++)   ─▶ nvosd/enc/sink
                              decoder(TRT)+match+track+backend
                              ▼ NvDsUserMeta(pose) + EOS: .txt/.ply
```

## Layout

```
deepstream-mast3r-slam/
├── DESIGN.md                       # architecture + rationale (RU)
├── CMakeLists.txt                  # builds the element + SLAM core + reuses kernels
├── build_local.sh                  # cmake build inside the container
├── gst-plugin/
│   ├── gstnvdsmast3rslam.{h,cpp}    # the GStreamer element (reads tensor meta)
│   └── mast3r_slam_meta.h           # NvDsUserMeta pose type
├── lib/
│   ├── mast3r_slam_core.{h,cpp}     # libtorch SLAM core (decoder→match→track)
│   ├── trt_engine.{h,cpp}           # TensorRT 10 runner (decoder engine)
│   ├── sim3.h                       # Sim(3) Lie group (Eigen)
│   └── io.{h,cpp}                   # TUM trajectory + .ply writers
├── configs/
│   └── config_infer_mast3r_encoder.txt   # gst-nvinfer config (encoder)
├── tools/
│   └── export_onnx.py               # export encoder + decoder ONNX
├── docker/                          # Dockerfile (DeepStream 9.0) + build.sh
└── pipelines/                       # run_file / run_v4l2 / run_udp
```

## Build (DeepStream 9.0)

```bash
# from the repo root, submodules initialised, checkpoints/ present
bash deepstream-mast3r-slam/docker/build.sh
docker run --rm -it --gpus all --runtime nvidia \
    -v "$PWD:/opt/MASt3R-SLAM-GST" nvdsmast3rslam:ds9.0 bash
# inside: the .so is already in the GStreamer plugin path
gst-inspect-1.0 nvdsmast3rslam
```

The image installs libtorch, then CMake compiles `nvdsmast3rslam` and **reuses
the repository's CUDA kernels** (`mast3r_slam/backend/src/*.cu`) directly. To
rebuild after edits: `bash deepstream-mast3r-slam/build_local.sh`.

## Prepare engines

```bash
# 1) export ONNX (encoder + decoder)
python deepstream-mast3r-slam/tools/export_onnx.py --height 384 --width 512
# 2) build TensorRT engines (FP32 for best parity with the reference)
trtexec --onnx=checkpoints/mast3r_encoder.onnx --saveEngine=checkpoints/mast3r_encoder.engine
trtexec --onnx=checkpoints/mast3r_decoder.onnx --saveEngine=checkpoints/mast3r_decoder.engine
```

## Run

```bash
bash deepstream-mast3r-slam/pipelines/run_file.sh /path/to/video.mp4
bash deepstream-mast3r-slam/pipelines/run_v4l2.sh /dev/video0
bash deepstream-mast3r-slam/pipelines/run_udp.sh udp 5000
# output: logs/<seq>.txt (TUM trajectory), logs/<seq>.ply
```

## Element reference — `nvdsmast3rslam`

| Property | Default | Meaning |
|----------|---------|---------|
| `infer-gie-id` | `1` | `gie-unique-id` of the upstream `nvinfer` (encoder) |
| `config` | `config/base.yaml` | SLAM config (reserved; defaults compiled in) |
| `calib` | `""` | intrinsics YAML (calibrated mode) |
| `decoder-engine` | `""` | TensorRT engine for the decoder+heads |
| `save-dir` / `sequence-name` | `logs` / `mast3rslam` | output location/name |
| `save-results` | `true` | write `.txt`/`.ply` on EOS |
| `conf-threshold` | `1.5` | confidence filter for the `.ply` |
| `gpu-id` | `0` | CUDA device |

**Online pose output** is `NvDsUserMeta` of type `NVDS_MAST3R_SLAM_POSE_META`
(`NvDsMast3rSlamPoseMeta`, see `gst-plugin/mast3r_slam_meta.h`): `frame_id`,
`timestamp`, `t[3]`, `q[4]` (qx,qy,qz,qw), `scale`, `num_keyframes`,
`is_keyframe`, `mode`. Read it with a pad probe on the element's src pad.

## Status / limitations

* This is a **native DeepStream project**: it cannot be compiled or run without
  the DeepStream 9.0 SDK, TensorRT, CUDA and libtorch on a target GPU. The code
  follows the gst-nvinfer / `dsexample` conventions and is commented at the build
  and integration points.
* FP16/INT8 engines change numerics → output drifts from the bit-exact reference;
  build engines in FP32 for closest parity.
* The SLAM core ports the **front-end** (mono init, matching, Sim3 ray tracking,
  keyframe selection) on libtorch + the reused CUDA kernels. The **global
  factor-graph backend** and **ASMK loop-closure** are marked as integration
  points in `lib/mast3r_slam_core.cpp` (the `gauss_newton_rays_cuda` call is
  wired). Per-point `.ply` color extraction from the `NvBufSurface` is a TODO
  (currently neutral gray); the trajectory `.txt` is fully populated.
* Numerics (Sim3 retraction in `sim3.h`, decoder ONNX I/O names, descriptor
  layout) must be validated on the target before production use.
