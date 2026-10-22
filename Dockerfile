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

COPY runner/ /runner/

ENTRYPOINT ["python3.13", "/runner/spark_fuse_run.py"]
