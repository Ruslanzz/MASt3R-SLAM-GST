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

set(EIGEN_INCLUDE "${REPO_ROOT}/thirdparty/eigen")
if(Eigen3_FOUND)
  set(EIGEN_INCLUDE ${EIGEN3_INCLUDE_DIR})
endif()

# Reused CUDA kernels from the repository (compiled directly, no pybind module).
set(KERNEL_SRC
  ${REPO_ROOT}/mast3r_slam/backend/src/gn_kernels.cu
  ${REPO_ROOT}/mast3r_slam/backend/src/matching_kernels.cu
)

set(CMAKE_CUDA_ARCHITECTURES 80 86 89)

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

target_compile_options(nvdsmast3rslam PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -O2>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Install into the GStreamer plugin path so gst-inspect-1.0 finds it.
install(TARGETS nvdsmast3rslam
  LIBRARY DESTINATION "${GST_LIBDIR}/gstreamer-1.0")
