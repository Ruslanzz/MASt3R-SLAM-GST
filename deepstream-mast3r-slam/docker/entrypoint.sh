#!/usr/bin/env bash
# Self-healing GStreamer registry guard.
#
# WHY: the NVIDIA GStreamer plugins (nvvideoconvert, nvinfer, nvstreammux, ...)
# only load once libcuda is reachable. If ANY plugin scan runs without a usable
# GPU — a docker build (no GPU), or a `docker run` that forgot `--gpus all`, or
# a stray gst-inspect before the driver was ready — those plugins are written to
# the cached registry's blacklist and stay blacklisted for the life of that
# cache. The pipeline then dies with `no element "nvvideoconvert"` until the
# cache is cleared by hand. Deleting the cache at image-build time is not enough:
# a bad scan at runtime re-pollutes it.
#
# WHAT: on container start, probe one NVIDIA element. If it does not load, the
# registry is stale — delete it so the very next gst invocation rebuilds it with
# the GPU present. Deleting is cheap (a rescan takes a couple of seconds) and
# only happens when actually broken, so a healthy container pays nothing.
set -euo pipefail

# Registry lives under $GST_REGISTRY or $HOME/.cache/gstreamer-1.0. Cover both.
CACHE_DIRS=("${HOME:-/root}/.cache/gstreamer-1.0" "/root/.cache/gstreamer-1.0")

# Probe one NVIDIA element AND our own plugin: either can be the blacklisted
# one (nvvideoconvert after a GPU-less scan; nvdsmast3rslam after a load
# failure whose cause has since been fixed, e.g. a replaced .so).
probe_ok() {
  gst-inspect-1.0 nvvideoconvert >/dev/null 2>&1 && \
  gst-inspect-1.0 nvdsmast3rslam >/dev/null 2>&1
}

if ! probe_ok; then
  echo "[entrypoint] nvvideoconvert/nvdsmast3rslam not loadable — clearing stale GStreamer registry" >&2
  for d in "${CACHE_DIRS[@]}"; do rm -rf "$d" 2>/dev/null || true; done
  # Rebuild now (GPU is present at runtime) so the first pipeline starts clean.
  if probe_ok; then
    echo "[entrypoint] registry rebuilt — NVIDIA plugins + nvdsmast3rslam OK" >&2
  else
    echo "[entrypoint] WARNING: still not loadable after a fresh rescan." >&2
    echo "[entrypoint]   nvvideoconvert missing -> container started without --gpus all / --runtime nvidia" >&2
    echo "[entrypoint]   nvdsmast3rslam missing -> see the real cause with:" >&2
    echo "[entrypoint]   gst-inspect-1.0 \"\$(pkg-config --variable=pluginsdir gstreamer-1.0)/libnvdsmast3rslam.so\"" >&2
  fi
fi

# Fall through to the container command (default CMD is bash).
exec "$@"
