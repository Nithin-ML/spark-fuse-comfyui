# ComfyUI on Spark Fuse — Flux test image.
# Models are NOT baked in; they arrive per job under /input/models/ (ShareSync).
#
# Base pinned by digest: the moving cu130-slim-v2 tag silently changed ComfyUI
# between rebuilds. This pin provides ComfyUI 0.30.1, the first dated tag with
# the built-in MiniMaxH3ImageToVideo node (added in ComfyUI 0.30.0).
# Rollback target (ComfyUI 0.28.0, production before this pin):
# FROM yanwk/comfyui-boot:cu130-slim-v2-20260717@sha256:e2fada7c3db5ed043d99b336ab2e1b68d100cac74e094ad39a28831e1bb7fffb
FROM yanwk/comfyui-boot:cu130-slim-v2-20260803@sha256:9cde099f3fa5b39a58ceadf4562d4e157b4f5e3008d1690d6c5bba9ffd0cffbb

# The base's entrypoint copies the bundle to /root on first start; we skip all
# of that — nodes are installed into the bundle itself and the runner launches
# ComfyUI from the bundle path directly.
#
# ComfyUI's Python stack (torch, pip) lives under python3.13; plain `python3`
# in this base is a bare 3.12 with no pip. Everything here must use python3.13.
ENV COMFYUI_DIR=/default-comfyui-bundle/ComfyUI \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_NO_CACHE_DIR=1

# The base ships ComfyUI-Manager as a pip package with its startup cache-update
# already patched out; this rm only guards against bundle layouts that drop a
# self-updating Manager into custom_nodes.
RUN rm -rf "$COMFYUI_DIR/custom_nodes/ComfyUI-Manager"

COPY nodes.json /build/nodes.json
COPY scripts/install_nodes.py /build/install_nodes.py
RUN python3.13 /build/install_nodes.py /build/nodes.json "$COMFYUI_DIR/custom_nodes" \
    && rm -rf /build

# Benign noise from the base image, cleaned up without moving the ComfyUI pin.
# - The deprecated pynvml distribution warns on import. Uninstalling it leaves
#   the pynvml module provided by nvidia-ml-py, at the version the base ships.
# - PyOpenGL logs a missing accelerator. Install it at PyOpenGL's own version
#   so a later base bump cannot pair mismatched builds.
# - matplotlib reports the color-emoji font only while building its font cache,
#   which a fresh container does on every job. Build the cache once here.
# - The base installs onnxruntime (CPU) and a CUDA 13 nightly over the same
#   module path. 1.30.0 is the PyPI CUDA 13 build whose GPU discovery skips
#   DRM cards that have no PCI vendor file, instead of warning about them.
# One heredoc: a second `&&` after a closing delimiter is a new Dockerfile
# instruction, not a continuation of this RUN.
RUN python3.13 - <<'PY'
import importlib.metadata
import os
import subprocess
import sys
import warnings

def installed(dist):
    try:
        importlib.metadata.version(dist)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False

subprocess.call([sys.executable, "-m", "pip", "show", "pynvml", "onnxruntime", "onnxruntime-gpu"])

if installed("pynvml"):
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "pynvml"])

pyopengl = importlib.metadata.version("PyOpenGL")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--no-cache-dir",
    f"PyOpenGL-accelerate=={pyopengl}",
])

# Both distributions own the onnxruntime package directory. Remove whichever
# are present before installing the single GPU build.
for dist in ("onnxruntime", "onnxruntime-gpu"):
    if installed(dist):
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", dist])
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-deps",
    "onnxruntime-gpu==1.30.0",
])

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import pynvml
bad = [str(item.message) for item in caught if "pynvml package is deprecated" in str(item.message)]
if bad:
    raise SystemExit(bad[0])
import OpenGL_accelerate
import onnxruntime as ort
providers = ort.get_available_providers()
print("onnxruntime", ort.__version__, providers, flush=True)
if "CUDAExecutionProvider" not in providers:
    raise SystemExit("CUDAExecutionProvider missing from onnxruntime")

os.environ["MPLBACKEND"] = "Agg"
import matplotlib.font_manager as fm
fm.fontManager
PY

COPY runner/ /runner/

ENTRYPOINT ["python3.13", "/runner/spark_fuse_run.py"]
