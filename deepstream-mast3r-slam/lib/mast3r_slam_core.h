/*
 * mast3r_slam_core.h — C++ API of the MASt3R-SLAM core used by the GStreamer
 * element. Kept free of GStreamer/GObject types so it can be unit-tested and
 * compiled independently. Implemented on top of libtorch + the repository's
 * custom CUDA kernels (mast3r_slam_backends) and a TensorRT decoder engine.
 */
#ifndef MAST3R_SLAM_CORE_H
#define MAST3R_SLAM_CORE_H

#include <cstdint>
#include <memory>
#include <string>

namespace mast3r_slam {

struct CoreConfig {
  std::string config_path = "config/base.yaml";
  std::string calib_path;        // empty -> uncalibrated (rays) mode
  std::string decoder_engine;    // TensorRT engine for decoder + DPT heads
  std::string save_dir = "logs";
  std::string sequence_name = "mast3rslam";
  bool save_results = true;
  double conf_threshold = 1.5;   // .ply confidence filter
  int gpu_id = 0;
};

/* One frame's encoder output, handed over by the element. Pointers are CUDA
 * device pointers owned by the nvinfer tensor meta (valid for this call only). */
struct FrameInput {
  double timestamp = 0.0;
  const float *feat_dev = nullptr;  // (N, feat_dim) device
  const void *pos_dev = nullptr;    // (N, 2) int32 device
  int feat_n = 0;
  int feat_dim = 0;
  int pos_n = 0;
  int model_w = 0;                  // encoder input width  (resized frame)
  int model_h = 0;                  // encoder input height (resized frame)
  void *gst_buffer = nullptr;       // GstBuffer* (opaque) for color extraction
  void *frame_meta = nullptr;       // NvDsFrameMeta* (opaque)
};

struct PoseResult {
  bool valid = false;
  uint64_t frame_id = 0;
  double timestamp = 0.0;
  double t[3] = {0, 0, 0};
  double q[4] = {0, 0, 0, 1};  // qx, qy, qz, qw
  double scale = 1.0;
  int num_keyframes = 0;
  int is_keyframe = 0;
  int mode = 0;  // Mast3rSlamMode
};

class Mast3rSlamCore {
 public:
  explicit Mast3rSlamCore(const CoreConfig &cfg);
  ~Mast3rSlamCore();

  Mast3rSlamCore(const Mast3rSlamCore &) = delete;
  Mast3rSlamCore &operator=(const Mast3rSlamCore &) = delete;

  bool start();
  PoseResult process(const FrameInput &in);
  void finish();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mast3r_slam

#endif  // MAST3R_SLAM_CORE_H
