#!/usr/bin/env python3
"""Install pinned ComfyUI custom nodes at image build time.

Usage: install_nodes.py <nodes.json> <custom_nodes_dir>

Registry nodes are downloaded from the Comfy Registry at exact pinned
versions — the same artifacts a registry-based local install uses, so the
image matches the desktop node-for-node. Git nodes are fetched as GitHub
archive zips at exact pinned commits (no git binary required).

Each node's requirements.txt (if present) is pip-installed under a
constraints file that pins the base image's torch/torchvision/torchaudio,
so no node dependency can replace the CUDA-matched PyTorch build.
"""
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

REGISTRY_API = "https://api.comfy.org/nodes/{id}/versions/{version}"
GITHUB_ZIP = "{repo}/archive/{commit}.zip"
BASE_CONSTRAINTS = Path("/builder-scripts/constraints.txt")  # ships in the base image
CONSTRAINTS = Path("/tmp/torch-constraints.txt")
RETRIES = 3


def log(msg):
    print(f"[install-nodes] {msg}", flush=True)


def fetch(url, as_json=False):
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "spark-fuse-comfyui-build"}
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = resp.read()
            return json.loads(data) if as_json else data
        except Exception as err:
            last_err = err
            log(f"attempt {attempt}/{RETRIES} failed for {url}: {err}")
            time.sleep(5 * attempt)
    raise RuntimeError(f"download failed after {RETRIES} attempts: {url}") from last_err


def extract_zip(blob, dest: Path):
    """Extract a zip into dest; strip a single wrapping top-level directory."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            zf.extractall(tmp)
        entries = list(tmp.iterdir())
        root = entries[0] if len(entries) == 1 and entries[0].is_dir() else tmp
        dest.mkdir(parents=True)
        for item in root.iterdir():
            shutil.move(str(item), str(dest / item.name))


def write_constraints():
    if BASE_CONSTRAINTS.is_file():
        CONSTRAINTS.write_text(BASE_CONSTRAINTS.read_text())
        log(f"torch constraints (from base image): "
            f"{', '.join(CONSTRAINTS.read_text().split())}")
        return
    frozen = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True, capture_output=True, text=True,
    ).stdout
    pins = [l for l in frozen.splitlines()
            if re.match(r"^(torch|torchvision|torchaudio)==", l)]
    CONSTRAINTS.write_text("\n".join(pins) + "\n")
    log(f"torch constraints: {', '.join(pins) or 'NONE FOUND'}")


def install_requirements(node_dir: Path):
    req = node_dir / "requirements.txt"
    if not req.is_file():
        return
    log(f"pip install -r {req}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir",
         "-r", str(req), "-c", str(CONSTRAINTS)],
        check=True,
    )


# ---- Optional nodes (Halo Hues addition) -----------------------------------
# A node with "required": false is installed after every required node, and
# may only ADD Python packages: every package already installed is pinned at
# its current version. A requirement that would change one of them is skipped
# (logged), so a new node can never break the nodes the studio already relies
# on, and a node that fails to download or install never fails the build. The
# runner's early workflow check reports any node type that ended up missing.
FROZEN = Path("/tmp/frozen-constraints.txt")
OPTIONAL_PROBLEMS = []


def freeze_environment():
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                         capture_output=True, text=True, check=True).stdout
    pins = [l for l in out.splitlines() if "==" in l and " @ " not in l
            and not l.startswith(("-e", "#"))]
    FROZEN.write_text("\n".join(pins) + "\n")
    log(f"frozen {len(pins)} installed packages for optional nodes")


def _req_lines(req: Path):
    for line in req.read_text(errors="ignore").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith(("-", "git+", "http")):
            yield line


def install_optional_requirements(node_dir: Path, label: str):
    req = node_dir / "requirements.txt"
    if not req.is_file():
        return
    base = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-c", str(FROZEN)]
    if subprocess.run(base + ["-r", str(req)]).returncode == 0:
        return
    # The file as a whole conflicts: install line by line, skip what conflicts.
    skipped = [line for line in _req_lines(req)
               if subprocess.run(base + [line]).returncode != 0]
    if skipped:
        OPTIONAL_PROBLEMS.append(f"{label}: skipped requirement(s) {', '.join(skipped)}")
        log(f"WARNING {label}: skipped {', '.join(skipped)}")


def main():
    manifest = json.loads(Path(sys.argv[1]).read_text())
    target = Path(sys.argv[2])
    target.mkdir(parents=True, exist_ok=True)
    write_constraints()
    optional = [n for n in manifest["git"] if n.get("required", True) is False]
    manifest["git"] = [n for n in manifest["git"] if n.get("required", True) is not False]

    for node in manifest["registry"]:
        node_id, version = node["id"], node["version"]
        dest = target / node_id
        log(f"registry: {node_id}@{version}")
        meta = fetch(REGISTRY_API.format(id=node_id, version=version), as_json=True)
        url = meta.get("downloadUrl")
        if not url:
            raise RuntimeError(f"no downloadUrl for {node_id}@{version}: {meta}")
        extract_zip(fetch(url), dest)
        install_requirements(dest)

    for node in manifest["git"]:
        dest = target / node["folder"]
        log(f"git: {node['repo']} @ {node['commit'][:12]}")
        extract_zip(fetch(GITHUB_ZIP.format(**node)), dest)
        install_requirements(dest)

    if optional:
        freeze_environment()
    for node in optional:
        dest = target / node["folder"]
        label = f"{node['repo']} @ {node['commit'][:12]}"
        log(f"git (optional): {label}")
        try:
            extract_zip(fetch(GITHUB_ZIP.format(**node)), dest)
        except Exception as exc:  # noqa: BLE001 - optional: never fail the build
            OPTIONAL_PROBLEMS.append(f"{label}: download failed ({exc})")
            log(f"WARNING {label}: download failed ({exc})")
            continue
        install_optional_requirements(dest, node["folder"])

    installed = sorted(p.name for p in target.iterdir() if p.is_dir())
    log(f"installed nodes: {', '.join(installed)}")
    if OPTIONAL_PROBLEMS:
        log(f"{len(OPTIONAL_PROBLEMS)} optional node problem(s):")
        for line in OPTIONAL_PROBLEMS:
            log(f"  - {line}")


if __name__ == "__main__":
    main()
