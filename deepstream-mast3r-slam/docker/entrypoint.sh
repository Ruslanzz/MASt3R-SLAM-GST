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

if ! gst-inspect-1.0 nvvideoconvert >/dev/null 2>&1; then
  echo "[entrypoint] nvvideoconvert not loadable — clearing stale GStreamer registry" >&2
  for d in "${CACHE_DIRS[@]}"; do rm -rf "$d" 2>/dev/null || true; done
  # Rebuild now (GPU is present at runtime) so the first pipeline starts clean.
  if gst-inspect-1.0 nvvideoconvert >/dev/null 2>&1; then
    echo "[entrypoint] registry rebuilt — NVIDIA plugins OK" >&2
  else
    echo "[entrypoint] WARNING: nvvideoconvert still not loadable after rescan;" \
         "check that the container was started with --gpus all / --runtime nvidia" >&2
  fi
fi

# Fall through to the container command (default CMD is bash).
exec "$@"
