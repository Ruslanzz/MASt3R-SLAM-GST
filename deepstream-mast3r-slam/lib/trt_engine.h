/*
 * trt_engine.h — minimal TensorRT 10 runner wrapping a serialized .engine.
 *
 * Used for the MASt3R decoder + DPT heads (the encoder runs in gst-nvinfer).
 * I/O buffers are plain CUDA device pointers so the caller can bind libtorch
 * CUDA tensors (tensor.data_ptr()) directly — no host round-trip.
 */
#ifndef MAST3R_SLAM_TRT_ENGINE_H
#define MAST3R_SLAM_TRT_ENGINE_H

#include <cuda_runtime_api.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
}  // namespace nvinfer1

namespace mast3r_slam {

class TrtEngine {
 public:
  TrtEngine();
  ~TrtEngine();

  // Loads and deserializes the engine file. Returns false on failure.
  bool load(const std::string &engine_path);

  // Set the (dynamic) shape of an input tensor before infer().
  bool setInputShape(const std::string &name, const std::vector<int64_t> &dims);

  // Bind a device pointer to a named I/O tensor.
  void setTensorAddress(const std::string &name, void *device_ptr);

  // Enqueue + synchronize on the given stream (0 = default).
  bool infer(cudaStream_t stream = 0);

  std::vector<std::string> inputNames() const { return inputs_; }
  std::vector<std::string> outputNames() const { return outputs_; }
  std::vector<int64_t> tensorShape(const std::string &name) const;

 private:
  struct Deleter {
    template <typename T>
    void operator()(T *p) const;
  };
  std::unique_ptr<nvinfer1::IRuntime, Deleter> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, Deleter> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, Deleter> context_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
};

}  // namespace mast3r_slam

#endif  // MAST3R_SLAM_TRT_ENGINE_H
