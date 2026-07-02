# Native C++ DeepStream plugin (nvdsmast3rslam) for NVIDIA GeForce GTX 1660 Ti.
#
# The GTX 1660 Ti is a Turing GPU (compute capability 7.5, 6 GB, no tensor cores).
# The newest DeepStream that still supports Turing on x86 dGPU is DeepStream 7.1
# (Ubuntu 22.04, GStreamer 1.20, CUDA 12.6, TensorRT 10.3, driver >= 560). Newer
# releases may drop Turing, so 7.1 is the safe maximum for this card.
#
# If a later DeepStream release still lists Turing/sm_75 in its support matrix,
# bump BASE_IMAGE — CUDA arch stays 75.
ARG BASE_IMAGE=nvcr.io/nvidia/deepstream:7.1-triton-multiarch
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
# Torch cu124 wheels run fine on the container's CUDA 12.6 runtime and include
# Turing (sm_75) kernels.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
# Turing compute capability for our CUDA kernels + torch extension build.
ARG CUDA_ARCH=75
ARG TORCH_CUDA_ARCH_LIST=7.5
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ninja-build pkg-config \
        python3 python3-pip python3-dev \
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
        libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# libtorch (C++) via the pip wheel; export its cmake prefix for our build.
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir torch==2.5.1 --index-url ${TORCH_INDEX_URL}

WORKDIR /opt/MASt3R-SLAM-GST
COPY . /opt/MASt3R-SLAM-GST
RUN git submodule update --init --recursive || true

# Build the plugin for Turing (sm_75). Torch_DIR points cmake at the pip libtorch.
RUN export Torch_DIR="$(python3 -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"share","cmake","Torch"))')" && \
    cmake -S deepstream-mast3r-slam -B /tmp/build -GNinja \
        -DTorch_DIR="${Torch_DIR}" \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /tmp/build -j && \
    GST_PLUGIN_DIR="$(pkg-config --variable=pluginsdir gstreamer-1.0)" && \
    mkdir -p "${GST_PLUGIN_DIR}" && \
    cp /tmp/build/libnvdsmast3rslam.so "${GST_PLUGIN_DIR}/" && \
    TORCH_LIB="$(python3 -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))')" && \
    printf '%s\n' "${TORCH_LIB}" "/opt/nvidia/deepstream/deepstream/lib" \
        > /etc/ld.so.conf.d/mast3rslam.conf && \
    ldconfig && \
    (gst-inspect-1.0 nvdsmast3rslam || echo "[warn] gst-inspect could not load the element")

# Python side for the ONNX export tooling (tools/export_onnx.py):
# MASt3R (+dust3r, curope) and the repo package with its CUDA backend + lietorch.
# --no-build-isolation: curope / lietorch / the backend import torch in setup.py.
# FORCE_CUDA=1: docker build has no visible GPU, but nvcc compiles the kernels
# fine (setup.py honors this override); arch comes from TORCH_CUDA_ARCH_LIST.
RUN python3 -m pip install --no-cache-dir --no-build-isolation -e thirdparty/mast3r && \
    FORCE_CUDA=1 python3 -m pip install --no-cache-dir --no-build-isolation -e . && \
    python3 -c "import mast3r.utils.path_to_dust3r, dust3r, mast3r_slam_backends; print('mast3r + dust3r + backend OK')"

WORKDIR /opt/MASt3R-SLAM-GST
CMD ["bash"]
