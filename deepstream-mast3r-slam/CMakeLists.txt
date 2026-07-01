# Builds the nvdsmast3rslam GStreamer element + the SLAM core, reusing the
# repository's CUDA kernels. Intended to be built inside the DeepStream 9.0
# container (docker/Dockerfile), where GStreamer 1.24, CUDA, TensorRT, DeepStream
# and libtorch are available.
cmake_minimum_required(VERSION 3.18)
project(nvdsmast3rslam LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- paths (override with -D... as needed) ----------------------------------
set(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH "MASt3R-SLAM repo root")
set(DEEPSTREAM_DIR "/opt/nvidia/deepstream/deepstream" CACHE PATH "DeepStream SDK")
set(TENSORRT_DIR "/usr/include/x86_64-linux-gnu" CACHE PATH "TensorRT include dir")
# Torch_DIR should point at libtorch's cmake (e.g. python -c 'import torch;print(torch.utils.cmake_prefix_path)')

# ---- dependencies -----------------------------------------------------------
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0)
find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)
find_package(Eigen3 QUIET)

# The reused .cu kernels pull in torch/extension.h -> torch/python.h -> <Python.h>.
# python3-dev installs it under /usr/include/python3.X (a sub-dir not on the
# default search path), so add that dir explicitly. No libpython linking needed:
# the kernels only use torch::Tensor, not the Python C-API.
execute_process(
  COMMAND python3 -c "import sysconfig; print(sysconfig.get_path('include'))"
  OUTPUT_VARIABLE PYTHON_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "Python include dir: ${PYTHON_INCLUDE_DIR}")

set(EIGEN_INCLUDE "${REPO_ROOT}/thirdparty/eigen")
if(Eigen3_FOUND)
  set(EIGEN_INCLUDE ${EIGEN3_INCLUDE_DIR})
endif()

# Reused CUDA kernels from the repository (compiled directly, no pybind module).
set(KERNEL_SRC
  ${REPO_ROOT}/mast3r_slam/backend/src/gn_kernels.cu
  ${REPO_ROOT}/mast3r_slam/backend/src/matching_kernels.cu
)

# GTX 1660 Ti is Turing, compute capability 7.5. Override with -DCMAKE_CUDA_ARCHITECTURES
# for other GPUs (80=Ampere A100, 86=RTX 30xx, 89=Ada).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

add_library(nvdsmast3rslam SHARED
  gst-plugin/gstnvdsmast3rslam.cpp
  lib/mast3r_slam_core.cpp
  lib/trt_engine.cpp
  lib/io.cpp
  ${KERNEL_SRC}
)

target_include_directories(nvdsmast3rslam PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/gst-plugin
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
  ${GST_INCLUDE_DIRS}
  ${EIGEN_INCLUDE}
  ${REPO_ROOT}/mast3r_slam/backend/include
  ${DEEPSTREAM_DIR}/sources/includes
  ${TENSORRT_DIR}
  ${TORCH_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIR}
)

target_link_directories(nvdsmast3rslam PRIVATE
  ${GST_LIBRARY_DIRS}
  ${DEEPSTREAM_DIR}/lib
)

target_link_libraries(nvdsmast3rslam PRIVATE
  ${GST_LIBRARIES}
  ${TORCH_LIBRARIES}
  CUDA::cudart
  nvinfer
  nvdsgst_meta
  nvds_meta
  nvbufsurface
)

# The __CUDA_NO_HALF*_ macros mirror what torch.utils.cpp_extension passes to
# nvcc: they disable CUDA's built-in __half/__nv_bfloat16 operators so the
# reused kernels use c10::Half's operators unambiguously (otherwise `score >
# max_score` on c10::Half is an ambiguous overload with CUDA 12.x).
target_compile_options(nvdsmast3rslam PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -O2>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3
    -D__CUDA_NO_HALF_OPERATORS__
    -D__CUDA_NO_HALF_CONVERSIONS__
    -D__CUDA_NO_HALF2_OPERATORS__
    -D__CUDA_NO_BFLOAT16_CONVERSIONS__>
)

# Install into the GStreamer plugin path so gst-inspect-1.0 finds it.
install(TARGETS nvdsmast3rslam
  LIBRARY DESTINATION "${GST_LIBDIR}/gstreamer-1.0")
