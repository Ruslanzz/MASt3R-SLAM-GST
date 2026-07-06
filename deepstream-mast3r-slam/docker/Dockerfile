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

# gstreamer1.0-x provides the X11 window sinks (ximagesink/xvimagesink) and
# plugins-good provides autovideosink + v4l2src — needed by the *_display.sh
# pipelines (VISUALIZATION.md 1b); no-ops if the base image already has them.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ninja-build pkg-config \
        python3 python3-pip python3-dev \
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
        gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-x \
        libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# libtorch (C++) via the pip wheel; export its cmake prefix for our build.
# torchvision is needed by dust3r's image utils (used by tools/export_onnx.py);
# 0.20.1 is the matching release for torch 2.5.1.
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
        --index-url ${TORCH_INDEX_URL}

WORKDIR /opt/MASt3R-SLAM-GST
COPY . /opt/MASt3R-SLAM-GST
RUN git submodule update --init --recursive || true

# Optional ROS 2 Humble bridge for nvdsmast3rviz (Odometry/Path/TF/PointCloud2).
# Build with:  --build-arg WITH_ROS2=1
# rviz2 is installed alongside the bridge so the 3D map/trajectory can be
# viewed from THIS container (VIZ=rviz in pipelines/run_slam.sh; needs the X11
# forwarding from VISUALIZATION.md 1b). Without the NVIDIA `graphics`
# capability rviz2 falls back to Mesa software rendering — fine at SLAM frame
# rates (1-3 FPS).
ARG WITH_ROS2=0
RUN if [ "${WITH_ROS2}" = "1" ]; then \
      : "The DeepStream base image ships an Intel RealSense apt source whose" \
        "signing key has since rotated, so apt-get update fails with" \
        "'repository is not signed'. We don't use RealSense — drop the source." && \
      (grep -rlE 'librealsense' /etc/apt/sources.list.d/ 2>/dev/null | xargs -r rm -f || true) && \
      (sed -i '/librealsense/d' /etc/apt/sources.list 2>/dev/null || true) && \
      apt-get update && apt-get install -y --no-install-recommends curl gnupg && \
      curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg && \
      echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
        > /etc/apt/sources.list.d/ros2.list && \
      apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-rclcpp ros-humble-nav-msgs ros-humble-sensor-msgs \
        ros-humble-geometry-msgs ros-humble-tf2-ros ros-humble-ros-environment \
        ros-humble-rviz2 && \
      rm -rf /var/lib/apt/lists/*; \
    fi

# Build the plugin for Turing (sm_75). Torch_DIR points cmake at the pip libtorch.
RUN export Torch_DIR="$(python3 -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"share","cmake","Torch"))')" && \
    ROS_FLAGS=""; if [ "${WITH_ROS2}" = "1" ]; then \
      ROS_FLAGS="-DWITH_ROS2=ON -DCMAKE_PREFIX_PATH=/opt/ros/humble"; fi && \
    cmake -S deepstream-mast3r-slam -B /tmp/build -GNinja \
        -DTorch_DIR="${Torch_DIR}" \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        ${ROS_FLAGS} \
        -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /tmp/build -j && \
    GST_PLUGIN_DIR="$(pkg-config --variable=pluginsdir gstreamer-1.0)" && \
    mkdir -p "${GST_PLUGIN_DIR}" && \
    cp /tmp/build/libnvdsmast3rslam.so "${GST_PLUGIN_DIR}/" && \
    TORCH_LIB="$(python3 -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))')" && \
    printf '%s\n' "${TORCH_LIB}" "/opt/nvidia/deepstream/deepstream/lib" \
        "/opt/ros/humble/lib" \
        > /etc/ld.so.conf.d/mast3rslam.conf && \
    ldconfig && \
    (gst-inspect-1.0 nvdsmast3rslam || echo "[warn] gst-inspect could not load the element") && \
    rm -rf /root/.cache/gstreamer-1.0
# ^ the registry cache MUST NOT be baked into the image: docker build has no
# GPU, so the NVIDIA GStreamer plugins (nvvideoconvert, nvinfer, ...) fail to
# load libcuda and get blacklisted; the stale blacklist would persist at
# runtime ("no element nvvideoconvert"). A fresh scan happens on first use.

# Belt-and-suspenders for the same blacklist problem: a self-healing entrypoint
# that, on every container start, re-clears the registry if the NVIDIA plugins
# are (still) not loadable — covers a registry re-poisoned at runtime (e.g. a
# scan that raced the driver) that the build-time deletion above cannot. Kept in
# /usr/local/bin so a `-v $PWD:/opt/MASt3R-SLAM-GST` mount can't shadow it.
COPY deepstream-mast3r-slam/docker/entrypoint.sh /usr/local/bin/mast3r-entrypoint.sh
RUN chmod +x /usr/local/bin/mast3r-entrypoint.sh

# Python side for the ONNX export tooling (tools/export_onnx.py):
# MASt3R (+dust3r, curope) and the repo package with its CUDA backend + lietorch.
# --no-build-isolation: curope / lietorch / the backend import torch in setup.py.
# FORCE_CUDA=1: docker build has no visible GPU, but nvcc compiles the kernels
# fine (setup.py honors this override); arch comes from TORCH_CUDA_ARCH_LIST.
RUN python3 -m pip install --no-cache-dir --no-build-isolation -e thirdparty/mast3r && \
    FORCE_CUDA=1 python3 -m pip install --no-cache-dir --no-build-isolation -e . && \
    python3 -m pip install --no-cache-dir onnx && \
    python3 -c "import mast3r.utils.path_to_dust3r, dust3r, mast3r_slam_backends, onnx; print('mast3r + dust3r + backend + onnx OK')"

WORKDIR /opt/MASt3R-SLAM-GST
ENTRYPOINT ["/usr/local/bin/mast3r-entrypoint.sh"]
CMD ["bash"]
