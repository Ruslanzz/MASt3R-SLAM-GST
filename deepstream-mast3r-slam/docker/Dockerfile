# Native C++ DeepStream plugin (nvdsmast3rslam) built on DeepStream 9.0.
#
# DeepStream 9.0 (Ubuntu 24.04) provides GStreamer 1.24, the official NVIDIA
# elements (incl. gst-nvinfer), CUDA 12.x and TensorRT 10.x. We additionally
# install libtorch (to reuse the repo's CUDA kernels and the SLAM math) and
# build our element + SLAM core with CMake.
#
# NOTE: substitute a real NGC tag if 9.0 is not yet published to you. Use a
# *devel*/*triton* variant so nvcc + dev headers are present.
ARG BASE_IMAGE=nvcr.io/nvidia/deepstream:9.0-triton-multiarch
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ninja-build pkg-config \
        python3 python3-pip \
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
        libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# libtorch (C++). Using the pip wheel and exporting its cmake prefix is the
# simplest way to get a CUDA-matched libtorch + headers.
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir torch==2.5.1 --index-url ${TORCH_INDEX_URL}

WORKDIR /opt/MASt3R-SLAM-GST
COPY . /opt/MASt3R-SLAM-GST
RUN git submodule update --init --recursive || true

# Build the plugin. Torch_DIR points cmake at the pip libtorch.
RUN export Torch_DIR="$(python3 -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"share","cmake","Torch"))')" && \
    cmake -S deepstream-mast3r-slam -B /tmp/build -GNinja \
        -DTorch_DIR="${Torch_DIR}" \
        -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /tmp/build -j && \
    cp /tmp/build/libnvdsmast3rslam.so /opt/nvidia/deepstream/deepstream/lib/gstreamer-1.0/ && \
    (gst-inspect-1.0 nvdsmast3rslam || echo "[warn] gst-inspect could not load the element")

WORKDIR /opt/MASt3R-SLAM-GST
CMD ["bash"]
