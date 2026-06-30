# MASt3R-SLAM GStreamer plugin — build on top of DeepStream 9.0.
#
# DeepStream 9.0 (Ubuntu 24.04) ships GStreamer 1.24, CUDA 12.x and TensorRT 10.x
# together with the official NVIDIA GStreamer elements (nvvideoconvert,
# nvstreammux, nvv4l2*, nvinfer, nvosd ...) used in the final pipeline.
#
# NOTE: the exact NGC tag is parameterised. As of writing, public DeepStream
# images go up to 7.x; substitute a real tag from
#   https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream
# if 9.0 is not yet available to you. Prefer a *devel*/*triton* variant so the
# CUDA toolkit (nvcc) is present to compile the SLAM CUDA kernels.
ARG BASE_IMAGE=nvcr.io/nvidia/deepstream:9.0-triton-multiarch
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
# Build arch for the custom CUDA kernels (override per target GPU, e.g. 89 = Ada,
# 86 = Ampere, 87 = Orin). Space-separated list is allowed.
ARG TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# Torch wheel index matching the container CUDA version (cu124 works on 12.x).
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124

# ---------------------------------------------------------------- system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ninja-build \
        python3 python3-pip python3-dev \
        python3-gi python3-gi-cairo gir1.2-gstreamer-1.0 \
        python3-gst-1.0 gstreamer1.0-python3-plugin-loader \
        gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad gstreamer1.0-libav \
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------- python deps
# Match the torch version the repo pins (2.5.1) and the container's CUDA.
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
        torch==2.5.1 torchvision==0.20.1 --index-url ${TORCH_INDEX_URL}

WORKDIR /opt
# Expecting the repository (with submodules) to be COPYed or bind-mounted here.
# For a self-contained image, copy the whole repo:
COPY . /opt/MASt3R-SLAM-GST
WORKDIR /opt/MASt3R-SLAM-GST

# Submodules (eigen for the CUDA kernels; mast3r/dust3r for the model). If the
# build context already contains them this is a no-op.
RUN git submodule update --init --recursive || true

# MASt3R model + the SLAM CUDA backend (compiles gn_kernels.cu / matching_kernels.cu
# and lietorch). in3d (the 3D viewer) is intentionally NOT installed: the plugin
# never opens a window.
RUN python3 -m pip install --no-cache-dir -e thirdparty/mast3r && \
    (python3 -m pip install --no-cache-dir -e thirdparty/mast3r/asmk || \
     echo "[warn] asmk not installed; retrieval/loop-closure disabled") && \
    python3 -m pip install --no-cache-dir --no-build-isolation -e .

# TensorRT python bindings (already present in DeepStream, but ensure pycuda-free
# torch-based runner deps are available).
RUN python3 -c "import tensorrt; print('TensorRT', tensorrt.__version__)" || \
    echo "[warn] tensorrt python module not found; backend=tensorrt unavailable"

# --------------------------------------------------------------- plugin wiring
ENV GST_PLUGIN_PATH=/opt/MASt3R-SLAM-GST/gst-mast3r-slam${GST_PLUGIN_PATH:+:${GST_PLUGIN_PATH}}
ENV PYTHONPATH=/opt/MASt3R-SLAM-GST/gst-mast3r-slam/python:/opt/MASt3R-SLAM-GST${PYTHONPATH:+:${PYTHONPATH}}

# Sanity: the element should be inspectable (heavy imports are deferred).
RUN gst-inspect-1.0 mast3rslam || \
    echo "[warn] gst-inspect could not load mast3rslam (check GST_PLUGIN_PATH)"

WORKDIR /opt/MASt3R-SLAM-GST
CMD ["bash"]
