#include "trt_engine.h"

#include <NvInfer.h>

#include <cstring>
#include <fstream>
#include <iostream>

namespace mast3r_slam {

namespace {
class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING) std::cerr << "[TRT] " << msg << "\n";
  }
};
Logger g_logger;
}  // namespace

template <typename T>
void TrtEngine::Deleter::operator()(T *p) const {
  if (p) delete p;  // TensorRT 10 objects are deleted with `delete`.
}

TrtEngine::TrtEngine() = default;
TrtEngine::~TrtEngine() = default;

bool TrtEngine::load(const std::string &engine_path) {
  std::ifstream f(engine_path, std::ios::binary);
  if (!f.good()) {
    std::cerr << "[TRT] cannot open engine: " << engine_path << "\n";
    return false;
  }
  f.seekg(0, std::ios::end);
  size_t size = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<char> blob(size);
  f.read(blob.data(), size);

  runtime_.reset(nvinfer1::createInferRuntime(g_logger));
  if (!runtime_) return false;
  engine_.reset(runtime_->deserializeCudaEngine(blob.data(), size));
  if (!engine_) {
    std::cerr << "[TRT] deserialize failed: " << engine_path << "\n";
    return false;
  }
  context_.reset(engine_->createExecutionContext());
  if (!context_) return false;

  inputs_.clear();
  outputs_.clear();
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char *name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
      inputs_.push_back(name);
    else
      outputs_.push_back(name);
  }
  return true;
}

bool TrtEngine::setInputShape(const std::string &name,
                              const std::vector<int64_t> &dims) {
  nvinfer1::Dims d;
  d.nbDims = (int)dims.size();
  for (int i = 0; i < d.nbDims; ++i) d.d[i] = dims[i];
  return context_->setInputShape(name.c_str(), d);
}

void TrtEngine::setTensorAddress(const std::string &name, void *device_ptr) {
  context_->setTensorAddress(name.c_str(), device_ptr);
}

std::vector<int64_t> TrtEngine::tensorShape(const std::string &name) const {
  nvinfer1::Dims d = context_->getTensorShape(name.c_str());
  std::vector<int64_t> out(d.nbDims);
  for (int i = 0; i < d.nbDims; ++i) out[i] = d.d[i];
  return out;
}

bool TrtEngine::infer(cudaStream_t stream) {
  if (!context_->enqueueV3(stream)) return false;
  return cudaStreamSynchronize(stream) == cudaSuccess;
}

}  // namespace mast3r_slam
