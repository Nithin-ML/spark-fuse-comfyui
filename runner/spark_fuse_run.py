#!/usr/bin/env python3
"""Spark Fuse job runner — drives one or more headless ComfyUI workflows per
container, against a single ComfyUI process.

Job contract:
  /input/spark_fuse_job.json   optional manifest: {"workflows": [{"workflow":
                                {...API-format graph...}, "batch_count": N}, ...]}.
                                When present, every entry runs in manifest order
                                against ONE ComfyUI boot; outputs land under
                                /output/wf_01/, /output/wf_02/, ... (zero-padded,
                                manifest order).
  /input/workflow.json         single-workflow fallback, used only when the
                                manifest above is absent. Exact pre-manifest
                                behaviour: one workflow, BATCH_COUNT repeats,
                                flat /output. This is what an old bridge submits,
                                so it must keep working unchanged.
  /input/models/...      model files (checkpoints/, loras/, vae/, ...)
  /output/               everything written here persists to ShareSync

The ComfyUI server only ever listens on container-internal loopback; Spark
Fuse publishes no ports and none are needed.

Exit codes (single-workflow / fallback mode, unchanged):
  0  success — workflow completed, files written to /output
  1  workflow executed but ended in error
  2  bad input (workflow.json missing / unparsable / not API format,
     /output not writable, or the input directory cannot be staged)
  3  ComfyUI rejected the workflow at submission
  4  JOB_TIMEOUT_SECONDS exceeded
  5  workflow completed but produced no files in /output
  6  ComfyUI server failed to start or died mid-run

Exit codes (manifest mode):
  0  every workflow in the manifest succeeded
  1  at least one workflow failed; the others still ran (see per-workflow
     [runner] markers in the log for which)
  2  bad input (manifest missing/malformed, /output not writable, or the
     input directory cannot be staged)
  6  ComfyUI server failed to start or died mid-run (fatal — aborts the whole
     job; a dead ComfyUI process cannot run the remaining workflows)

  Codes 3 (submission rejected), 4 (JOB_TIMEOUT_SECONDS exceeded) and 5 (no
  files produced) are per-workflow conditions in manifest mode: that single
  workflow is marked failed and the job moves on to the next entry, then
  exits 1 overall rather than dying on the spot.
"""
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

COMFYUI_DIR = Path(os.environ.get("COMFYUI_DIR", "/default-comfyui-bundle/ComfyUI"))
INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))
WORKFLOW_PATH = Path(os.environ.get("WORKFLOW_PATH", str(INPUT_DIR / "workflow.json")))
MANIFEST_PATH = Path(os.environ.get("MANIFEST_PATH", str(INPUT_DIR / "spark_fuse_job.json")))
PORT = int(os.environ.get("COMFY_PORT", "8188"))
BASE_URL = f"http://127.0.0.1:{PORT}"
STARTUP_TIMEOUT = int(os.environ.get("STARTUP_TIMEOUT_SECONDS", "600"))
JOB_TIMEOUT = int(os.environ.get("JOB_TIMEOUT_SECONDS", "0"))  # 0 = no limit
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL_SECONDS", "2"))
HEARTBEAT_EVERY = 60
# Render this many images in one job (one cold start + one model load amortised
# across all of them). Set per job via the BATCH_COUNT env var; clamped to 1..100.
try:
    BATCH_COUNT = max(1, min(int(os.environ.get("BATCH_COUNT", "1")), 100))
except ValueError:
    BATCH_COUNT = 1
# Inputs whose value is randomised between batch renders so each image differs.
SEED_KEYS = ("seed", "noise_seed")
# Where ComfyUI looks for models. Defaults to the per-job /input mount today;
# set MODEL_BASE_DIR=/assets/models to use Spark Fuse's persistent assets mount
# when it ships (one submit-time --env flag, no image rebuild, no workflow edits).
MODEL_BASE_DIR = os.environ.get("MODEL_BASE_DIR", str(INPUT_DIR / "models"))
MODEL_SUBDIRS = [
    "checkpoints", "diffusion_models", "unet", "text_encoders", "clip",
    "clip_vision", "vae", "loras", "controlnet", "upscale_models",
    "embeddings", "configs",
]
# Generated at runtime from MODEL_BASE_DIR (see write_extra_model_paths()).
EXTRA_MODEL_PATHS = Path("/tmp/spark_fuse_extra_model_paths.yaml")
# Model prefetch (Halo Hues). ComfyUI's dynamic-VRAM loader reads weights from
# /assets in many small random reads while sampling ("Model Initializing"),
# which is very slow on the Fuse assets mount and its on-node cache. Copying the
# model files this job references to the node's local NVMe first, in large
# sequential reads, and pointing ComfyUI there makes every later read local.
# PREFETCH_MODELS=0 turns it off.
PREFETCH_MODELS = os.environ.get("PREFETCH_MODELS", "1").lower() not in ("0", "false", "no")
LOCAL_MODEL_DIR = Path(os.environ.get("LOCAL_MODEL_DIR", "/tmp/spark-models"))
# 2, not more: every byte read from /assets also lands in Spark's on-node asset
# cache, and several large files copied at once fill a small cache sooner.
PREFETCH_THREADS = max(1, int(os.environ.get("PREFETCH_THREADS", "2")))
# When the asset cache disk is full, reads from /assets fail with EIO until the
# cache frees space. A failed read is retried from the same offset, waiting up
# to this many seconds in total per file before giving up.
PREFETCH_EIO_WAIT = int(os.environ.get("PREFETCH_EIO_WAIT", "1200"))
# If a referenced model still could not be copied, stop with a clear error
# instead of letting ComfyUI read it from /assets, which on a full cache ends in
# "Fatal Python error: Bus error" and kills every workflow in the job.
# PREFETCH_REQUIRED=0 restores the old fall-back-to-/assets behaviour.
PREFETCH_REQUIRED = os.environ.get("PREFETCH_REQUIRED", "1").lower() not in ("0", "false", "no")
PREFETCH_FAILED = []
import threading as _threading
PREFETCH_CANCEL = _threading.Event()  # set when the job ends: stops copies at once
MODEL_EXTS = (".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx")
# ComfyUI's input directory. /input is read-only for the life of the job, and
# ComfyUI creates folders there at startup (Load3D's 3d/, uploads, and so on),
# so the runner copies the job files here and passes this path instead.
# Kept out of /tmp/comfyui because ComfyUI clears its temp directory on boot.
STAGED_INPUT_DIR = Path("/tmp/spark-fuse-input")


class WorkflowFailure(Exception):
    """One manifest entry's failure (exit codes 1, 3, 4 or 5 — never 2 or 6).

    Raised by submit()/wait_done()/_collect_new_outputs() instead of calling
    die() directly, so manifest mode can catch it, mark that entry failed, and
    continue to the next one. Fallback (single-workflow) mode catches it right
    back into die(code, message), reproducing today's immediate-exit behaviour
    exactly.
    """

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def log(msg):
    print(f"[runner] {msg}", flush=True)


def die(code, msg):
    log(f"ERROR: {msg}")
    sys.exit(code)


def http_json(path, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(BASE_URL + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body) if body else {}


def log_banner():
    log(f"spark-fuse-comfyui runner (python {sys.version.split()[0]})")
    version_file = COMFYUI_DIR / "comfyui_version.py"
    if version_file.is_file():
        match = re.search(r'__version__\s*=\s*"([^"]+)"', version_file.read_text())
        if match:
            log(f"ComfyUI version: {match.group(1)}")
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        if smi.returncode == 0:
            log(f"GPU: {smi.stdout.strip()}")
    except (OSError, subprocess.TimeoutExpired):
        log("nvidia-smi not available")


def _validate_workflow_shape(workflow, label):
    """Shared API-format check for both workflow.json and each manifest entry.

    Always fatal (exit 2) on a bad shape, even from inside a manifest — a
    malformed graph means the job can never legitimately run, same severity
    as a bad workflow.json today.
    """
    if not isinstance(workflow, dict):
        die(2, f"{label} is not a JSON object")
    if "nodes" in workflow and "links" in workflow:
        die(2, f"{label} is a UI-format export; re-export with "
               "'Save (API Format)' (enable dev mode options in ComfyUI settings)")
    bad = [k for k, v in workflow.items()
           if not (isinstance(v, dict) and "class_type" in v)]
    if bad:
        die(2, f"{label} does not look like API format "
               f"(offending keys: {bad[:5]})")


def load_workflow():
    if not WORKFLOW_PATH.is_file():
        die(2, f"workflow not found at {WORKFLOW_PATH}")
    try:
        workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        die(2, f"cannot parse {WORKFLOW_PATH}: {err}")
    _validate_workflow_shape(workflow, "workflow.json")
    log(f"workflow loaded: {len(workflow)} nodes")
    return workflow


def load_manifest():
    """Load and validate /input/spark_fuse_job.json; return [(workflow, batch_count), ...]
    in manifest order. Any structural problem is fatal (exit 2), checked entirely
    before ComfyUI starts — same severity and timing as a bad workflow.json today.
    """
    try:
        raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        die(2, f"cannot parse {MANIFEST_PATH}: {err}")
    entries = raw.get("workflows") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        die(2, f"{MANIFEST_PATH} must be a JSON object with a non-empty 'workflows' array")
    parsed = []
    for i, entry in enumerate(entries):
        label = f"{MANIFEST_PATH.name} workflows[{i}]"
        if not isinstance(entry, dict) or not isinstance(entry.get("workflow"), dict):
            die(2, f"{label} missing a 'workflow' object")
        workflow = entry["workflow"]
        _validate_workflow_shape(workflow, f"{label}.workflow")
        try:
            batch_count = max(1, min(int(entry.get("batch_count") or 1), 100))
        except (TypeError, ValueError):
            batch_count = 1
        parsed.append((workflow, batch_count))
    log(f"manifest loaded: {len(parsed)} workflow(s)")
    return parsed


def check_output_writable():
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        probe = OUTPUT_DIR / ".write-probe"
        probe.write_text("")
        probe.unlink()
    except OSError as err:
        die(2, f"{OUTPUT_DIR} is not writable: {err}")


def write_extra_model_paths():
    """Generate ComfyUI's extra_model_paths.yaml from MODEL_BASE_DIR.

    Done at runtime, not baked into the image, so the model library location is
    a submit-time choice: /input/models by default, or /assets/models once the
    Spark Fuse assets mount is available, with no rebuild and no workflow edits.
    """
    names = list(MODEL_SUBDIRS)
    extra = _extra_model_dirs()
    names += extra
    lines = []
    if PREFETCH_MODELS:
        # Listed first: ComfyUI returns the first match, so a prefetched local
        # copy wins over the /assets original. Files never prefetched are still
        # found under spark_fuse below.
        lines += ["spark_local:", f"  base_path: {LOCAL_MODEL_DIR}"]
        lines += [f"  {name}: {name}" for name in MODEL_SUBDIRS]
    lines += ["spark_fuse:", f"  base_path: {MODEL_BASE_DIR}"]
    lines += [f"  {name}: {name}" for name in names]
    EXTRA_MODEL_PATHS.write_text("\n".join(lines) + "\n")
    log(f"model search base: {MODEL_BASE_DIR}")
    if extra:
        log(f"extra model folders: {', '.join(extra)}")
    _link_extra_model_dirs(extra)


def _extra_model_dirs():
    """Top-level folders under MODEL_BASE_DIR beyond the standard MODEL_SUBDIRS.

    Custom nodes keep models in folders of their own (LLM/ for QwenVL,
    depthcrafter/, sams/, BiRefNet/, ...). Listing only the top level keeps this
    cheap on the lazily-mounted assets path.
    """
    base = Path(MODEL_BASE_DIR)
    try:
        entries = sorted(p.name for p in base.iterdir() if p.is_dir())
    except OSError:
        return []
    return [name for name in entries
            if name not in MODEL_SUBDIRS and not name.startswith(".")]


def _link_extra_model_dirs(names):
    """Symlink extra model folders into ComfyUI's own models/ directory.

    Many custom nodes ignore extra_model_paths.yaml and read
    folder_paths.models_dir/<folder> directly; without the link they download
    the model again on every job. An existing folder is replaced only when it
    holds nothing but ComfyUI's put_*_here placeholder files.
    """
    models_dir = COMFYUI_DIR / "models"
    for name in names:
        source = Path(MODEL_BASE_DIR) / name
        target = models_dir / name
        try:
            if target.is_symlink():
                continue
            if target.is_dir():
                if any(not p.name.startswith("put_") for p in target.iterdir()):
                    continue
                shutil.rmtree(target)
            elif target.exists():
                continue
            models_dir.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source, target_is_directory=True)
            log(f"linked models/{name} -> {source}")
        except OSError as err:
            log(f"could not link models/{name}: {err}")


def _referenced_model_files(workflows):
    """Model files under MODEL_BASE_DIR that these API-format workflows name.

    Any string input ending in a model extension is looked up in each standard
    model folder; every match is returned as a path relative to MODEL_BASE_DIR.
    Extra custom-node folders are left alone: they are symlinked into ComfyUI's
    own models/ directory, which ComfyUI searches before any extra path.
    """
    base = Path(MODEL_BASE_DIR)
    names = set()
    for wf in workflows:
        for node in wf.values():
            if not isinstance(node, dict):
                continue
            for value in (node.get("inputs") or {}).values():
                if isinstance(value, str) and value.lower().endswith(MODEL_EXTS):
                    names.add(value.replace("\\", "/").lstrip("/"))
    found = {}
    for name in sorted(names):
        for folder in MODEL_SUBDIRS:
            rel = Path(folder) / name
            try:
                src = base / rel
                if ".." not in rel.parts and src.is_file():
                    found[str(rel)] = src.stat().st_size
            except OSError:
                pass
    return found


def _copy_one(rel, size):
    src = Path(MODEL_BASE_DIR) / rel
    dest = LOCAL_MODEL_DIR / rel
    if dest.is_file() and dest.stat().st_size == size:
        return 0.0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    started = time.time()
    last_note = started
    copied = 0
    waited = 0.0
    chunk = 64 * 1024 * 1024
    import errno
    fd = os.open(src, os.O_RDONLY)
    try:
        with open(tmp, "wb") as fout:
            while copied < size:
                if PREFETCH_CANCEL.is_set():
                    raise RuntimeError("job ended; copy cancelled")
                try:
                    buf = os.pread(fd, min(chunk, size - copied), copied)
                except OSError as err:
                    if err.errno != errno.EIO or waited >= PREFETCH_EIO_WAIT:
                        raise
                    # Asset cache full: wait for Spark to evict, then resume
                    # from the same offset.
                    pause = min(60, 10 + waited / 4)
                    log(f"prefetch: {rel} read error at {copied / 2**30:.1f} GiB "
                        f"(asset cache full?); retrying in {pause:.0f}s")
                    time.sleep(pause)
                    waited += pause
                    continue
                if not buf:
                    break
                fout.write(buf)
                copied += len(buf)
                now = time.time()
                # Spark Fuse stops a container that prints nothing for 30
                # minutes (container_inactive), so report progress each minute.
                if now - last_note >= 60:
                    last_note = now
                    log(f"prefetch: {rel} {copied / 2**30:.1f}/{size / 2**30:.1f} GiB "
                        f"({copied / 2**20 / (now - started):.0f} MiB/s)")
    finally:
        os.close(fd)
    if copied != size:
        raise OSError(f"copied {copied} of {size} bytes")
    tmp.rename(dest)  # atomic: ComfyUI never sees a half-written model
    return time.time() - started


def prefetch_models(workflows):
    """Copy the job's model files to local NVMe. Runs in a background thread
    while ComfyUI starts; main() waits for it before submitting a workflow."""
    try:
        files = _referenced_model_files(workflows)
        if not files:
            log("prefetch: no model files to copy")
            return
        total = sum(files.values())
        free = shutil.disk_usage(LOCAL_MODEL_DIR.parent if LOCAL_MODEL_DIR.parent.exists() else "/tmp").free
        if total > free * 0.9:
            log(f"prefetch: skipped, {total / 2**30:.1f} GiB needed but only "
                f"{free / 2**30:.1f} GiB free on local disk")
            return
        log(f"prefetch: copying {len(files)} model file(s), {total / 2**30:.1f} GiB, "
            f"to local NVMe ({LOCAL_MODEL_DIR})")
        started = time.time()
        from concurrent.futures import ThreadPoolExecutor, as_completed
        # Largest first so the long copies start early.
        order = sorted(files.items(), key=lambda kv: kv[1], reverse=True)
        with ThreadPoolExecutor(max_workers=PREFETCH_THREADS) as pool:
            futures = {pool.submit(_copy_one, rel, size): (rel, size) for rel, size in order}
            for fut in as_completed(futures):
                rel, size = futures[fut]
                try:
                    secs = fut.result()
                    if secs:
                        log(f"prefetch: {rel} ({size / 2**30:.1f} GiB) in {secs:.0f}s "
                            f"({size / 2**20 / max(secs, 0.001):.0f} MiB/s)")
                except Exception as err:  # noqa: BLE001 - fall back to /assets for this file
                    LOCAL_MODEL_DIR.joinpath(rel + ".part").unlink(missing_ok=True)
                    if PREFETCH_CANCEL.is_set():
                        continue  # the job is ending; nothing to report
                    PREFETCH_FAILED.append(rel)
                    LOCAL_MODEL_DIR.joinpath(rel + ".part").unlink(missing_ok=True)
                    log(f"prefetch: {rel} failed ({err}); ComfyUI will read it from /assets")
        elapsed = time.time() - started
        if PREFETCH_CANCEL.is_set():
            log(f"prefetch: stopped after {elapsed:.0f}s (job ended)")
            return
        log(f"prefetch: done in {elapsed:.0f}s "
            f"({total / 2**20 / max(elapsed, 0.001):.0f} MiB/s overall)")
    except Exception as err:  # noqa: BLE001 - prefetch is an optimisation only
        log(f"prefetch: skipped after error ({err})")


def stage_input_directory():
    """Copy the read-only input mount to a writable directory for ComfyUI.

    A real copy, not symlinks: ComfyUI resolves input paths with realpath and
    rejects anything that lands outside its input directory. The model tree is
    left on the mount when MODEL_BASE_DIR sits under the input directory;
    extra_model_paths.yaml reads it from there, so checkpoints are not copied.
    """
    input_root = INPUT_DIR.resolve()
    skip_rel = None
    try:
        model_base = Path(MODEL_BASE_DIR).resolve()
    except OSError:
        model_base = None
    if model_base is not None and model_base != input_root and input_root in model_base.parents:
        skip_rel = model_base.relative_to(input_root)

    def ignore_models(directory, names):
        if skip_rel is None:
            return []
        try:
            rel = Path(directory).resolve().relative_to(input_root)
        except (OSError, ValueError):
            return []
        if rel == skip_rel.parent and skip_rel.name in names:
            return [skip_rel.name]
        return []

    if STAGED_INPUT_DIR.exists():
        shutil.rmtree(STAGED_INPUT_DIR)
    try:
        shutil.copytree(input_root, STAGED_INPUT_DIR, ignore=ignore_models)
        # The mount is read-only, and copytree keeps those modes. ComfyUI has
        # to be able to create folders and files in the copy.
        for dirpath, dirnames, filenames in os.walk(STAGED_INPUT_DIR):
            os.chmod(dirpath, 0o755)
            for name in filenames:
                os.chmod(Path(dirpath) / name, 0o644)
    except OSError as err:
        die(2, f"cannot stage {INPUT_DIR} to {STAGED_INPUT_DIR}: {err}")
    files = [p for p in STAGED_INPUT_DIR.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    log(f"staged input: {len(files)} file(s), {total:,} bytes -> {STAGED_INPUT_DIR}")


def start_comfyui():
    cmd = [
        sys.executable, "main.py",
        "--listen", "127.0.0.1",
        "--port", str(PORT),
        "--extra-model-paths-config", str(EXTRA_MODEL_PATHS),
        "--output-directory", str(OUTPUT_DIR),
        "--input-directory", str(STAGED_INPUT_DIR),
        "--temp-directory", "/tmp/comfyui",
    ]
    log(f"starting ComfyUI: {' '.join(cmd)}")
    # stdout/stderr inherited so ComfyUI's own logs reach the Spark Fuse stream
    return subprocess.Popen(cmd, cwd=COMFYUI_DIR)


def wait_ready(proc, busy=None):
    """Wait for ComfyUI's HTTP server.

    ComfyUI starts while the model prefetch is copying tens of GiB, and on
    some nodes that slows its start a lot (imports took 5+ minutes on one
    node). Nothing can be submitted before the prefetch finishes anyway, so
    the STARTUP_TIMEOUT only counts from the moment busy() turns False.
    """
    started = time.monotonic()
    deadline = started + STARTUP_TIMEOUT
    last_note = started
    while True:
        now = time.monotonic()
        if busy is not None and busy():
            deadline = max(deadline, now + STARTUP_TIMEOUT)
        if now >= deadline:
            break
        if proc.poll() is not None:
            die(6, f"ComfyUI exited during startup (code {proc.returncode})")
        try:
            stats = http_json("/system_stats", timeout=5)
            names = ", ".join(d.get("name", "?") for d in stats.get("devices", []))
            log(f"ComfyUI ready after {now - started:.0f}s (devices: {names or 'unknown'})")
            return
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            if now - last_note >= 60:
                last_note = now
                log(f"ComfyUI still starting ({now - started:.0f}s)")
            time.sleep(1)
    die(6, f"ComfyUI not ready after {time.monotonic() - started:.0f}s "
           f"({STARTUP_TIMEOUT}s after the model prefetch finished)")


# Save-node inputs that name an output path. A Windows-style prefix such as
# "SSI\\SHT30\\SHT30" is one file name with backslashes on Linux, not folders,
# and ShareSync cannot upload such names (the job ends in output_drain_failed).
_SAVE_PATH_INPUTS = ("filename_prefix", "output_path")


def _normalize_save_paths(workflow):
    """Turn backslashes in save-node path inputs into forward slashes so they
    become subfolders under /output, as they do on Windows."""
    for node_id, node in workflow.items():
        inputs = (node.get("inputs") or {}) if isinstance(node, dict) else {}
        for key in _SAVE_PATH_INPUTS:
            value = inputs.get(key)
            if isinstance(value, str) and "\\" in value:
                fixed = "/".join(part for part in value.replace("\\", "/").split("/") if part)
                inputs[key] = fixed
                log(f"node #{node_id}: {key} {value!r} -> {fixed!r} (Windows path separators)")


# ---- Early workflow check (before the model copy finishes) -----------------
# Node packs disagree on a few common node names. ComfyLiterals and
# ComfyUI-Logic both register "Int", "Float" and "String"; whichever loads
# last wins, so a workflow saved on a machine where the other pack won names
# the input differently ("value" vs "Number"/"String") and ComfyUI rejects it.
_LITERAL_NODES = ("Int", "Float", "String")


def _node_inputs_schema(info, class_type):
    spec = (info.get(class_type) or {}).get("input") or {}
    return list((spec.get("required") or {}).keys()), list((spec.get("optional") or {}).keys())


def _adapt_literal_nodes(workflow, info):
    """Rename the single input of Int/Float/String nodes to whatever the
    installed pack calls it. Returns how many nodes were changed."""
    changed = 0
    for node_id, node in workflow.items():
        if not isinstance(node, dict) or node.get("class_type") not in _LITERAL_NODES:
            continue
        inputs = node.get("inputs") or {}
        required, optional = _node_inputs_schema(info, node["class_type"])
        missing = [k for k in required if k not in inputs]
        extra = [k for k in inputs if k not in required and k not in optional]
        if len(missing) == 1 and len(extra) == 1:
            inputs[missing[0]] = inputs.pop(extra[0])
            changed += 1
            log(f"node #{node_id} ({node['class_type']}): input {extra[0]!r} -> {missing[0]!r} "
                "(saved with a different node pack's version of this node)")
    return changed


def _has_input(inputs, key):
    # V3 dynamic inputs (Autogrow, DynamicCombo) appear in /object_info under
    # one name ("values") but in the workflow as "values.a", "values.b", ...
    return key in inputs or any(k.startswith(key + ".") for k in inputs)


def _workflow_problems(workflow, info):
    """(blocking, warnings): node types not installed block the job; a
    missing input is only a warning, because ComfyUI's own check on submit
    is the authority there (dynamic inputs vary by node pack)."""
    problems, warnings = [], []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        cls = node.get("class_type")
        if cls not in info:
            title = (node.get("_meta") or {}).get("title") or cls
            problems.append(f"node #{node_id} ({title}): node type {cls!r} is not installed in this image")
            continue
        inputs = node.get("inputs") or {}
        required, _ = _node_inputs_schema(info, cls)
        for key in required:
            if not _has_input(inputs, key):
                warnings.append(f"node #{node_id} ({cls}): input {key!r} looks missing "
                                "(ComfyUI checks this for sure when the workflow is submitted)")
    return problems, warnings


def precheck_workflows(workflows):
    """Check every workflow against the installed nodes as soon as ComfyUI is
    up, instead of after the model copy (which can take 10+ minutes).
    Fixes the Int/Float/String clash in place; stops the job if no workflow
    can run. Returns the set of workflow indexes (1-based) with problems."""
    try:
        info = http_json("/object_info", timeout=120)
    except Exception as exc:  # noqa: BLE001 - the normal submit check still runs
        log(f"early workflow check skipped ({exc})")
        return set()
    bad = set()
    for idx, workflow in enumerate(workflows, start=1):
        _adapt_literal_nodes(workflow, info)
        problems, warnings = _workflow_problems(workflow, info)
        label = f"workflow {idx}/{len(workflows)}" if len(workflows) > 1 else "workflow"
        for w in warnings[:20]:
            log(f"{label}: warning: {w}")
        if problems:
            bad.add(idx)
            log(f"{label} will fail — {len(problems)} problem(s):")
            for p in problems[:20]:
                log(f"  {p}")
            if len(problems) > 20:
                log(f"  ... and {len(problems) - 20} more")
    if bad and len(bad) == len(workflows) and os.environ.get(
            "ALLOW_PARTIAL_OUTPUTS", "") not in ("1", "true", "yes"):
        die(3, "no workflow can run with the nodes in this image (see the problems above); "
               "stopping before the model copy finishes")
    if not bad:
        log(f"early workflow check: {len(workflows)} workflow(s) OK")
    return bad


def submit(workflow):
    _normalize_save_paths(workflow)
    payload = {"prompt": workflow, "client_id": "spark-fuse-runner"}
    try:
        resp = http_json("/prompt", payload=payload, timeout=60)
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace")
        log(f"ComfyUI rejected the workflow (HTTP {err.code}):")
        print(body, flush=True)
        raise WorkflowFailure(3, "workflow validation failed")
    prompt_id = resp.get("prompt_id")
    if not prompt_id:
        raise WorkflowFailure(3, f"no prompt_id in /prompt response: {resp}")
    node_errors = resp.get("node_errors") or {}
    if node_errors:
        # ComfyUI accepts a prompt when at least one output validates and
        # silently drops the outputs that failed; the job would then "succeed"
        # with only part of the graph run (often ending in exit 5, no output).
        _log_node_errors(node_errors)
        if os.environ.get("ALLOW_PARTIAL_OUTPUTS", "") not in ("1", "true", "yes"):
            raise WorkflowFailure(
                3, f"{len(node_errors)} node(s) failed validation; their outputs "
                   "would be skipped (set ALLOW_PARTIAL_OUTPUTS=1 to run the rest anyway)")
        log("ALLOW_PARTIAL_OUTPUTS set: running the outputs that validated")
    log(f"workflow queued, prompt_id={prompt_id}")
    return prompt_id


def _log_node_errors(node_errors):
    log("ComfyUI skipped part of the workflow; these nodes failed validation:")
    for node_id, info in node_errors.items():
        info = info if isinstance(info, dict) else {}
        log(f"  node #{node_id} ({info.get('class_type', '?')}):")
        for err in info.get("errors") or []:
            message = err.get("message", "")
            details = err.get("details", "")
            log(f"    - {message}" + (f": {details}" if details else ""))
        outputs = info.get("dependent_outputs") or []
        if outputs:
            log(f"    blocks output node(s): {', '.join(map(str, outputs))}")


def wait_done(proc, prompt_id):
    """Poll a submitted prompt to completion. start resets on every call, so
    JOB_TIMEOUT_SECONDS is (and always was) a per-render ceiling, not a whole-job
    one — calling this once per manifest entry preserves that scoping unchanged.

    Raises WorkflowFailure(1) on execution error or WorkflowFailure(4) on
    JOB_TIMEOUT_SECONDS — both recoverable, per-workflow conditions in manifest
    mode. A dead ComfyUI process (die(6)) is not recoverable in either mode: no
    remaining workflow can run without the server, so it still exits immediately.
    """
    start = time.monotonic()
    last_beat = start
    while True:
        elapsed = time.monotonic() - start
        if JOB_TIMEOUT and elapsed > JOB_TIMEOUT:
            raise WorkflowFailure(4, f"job exceeded JOB_TIMEOUT_SECONDS={JOB_TIMEOUT}")
        if proc.poll() is not None:
            die(6, f"ComfyUI died mid-run (code {proc.returncode})")
        try:
            history = http_json(f"/history/{prompt_id}", timeout=15)
        except (urllib.error.URLError, OSError):
            history = {}
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status") or {}
            if status.get("status_str") == "error":
                for item in status.get("messages", []):
                    event = item[0]
                    data = item[1] if len(item) > 1 else {}
                    if event in ("execution_error", "execution_interrupted"):
                        log(f"{event}:")
                        print(json.dumps(data, indent=2), flush=True)
                raise WorkflowFailure(1, "workflow execution failed")
            if status.get("completed"):
                log(f"workflow completed in {elapsed:.0f}s")
                return entry
        if time.monotonic() - last_beat >= HEARTBEAT_EVERY:
            log(f"still running ({elapsed:.0f}s elapsed)")
            last_beat = time.monotonic()
        time.sleep(POLL_INTERVAL)


def report_outputs():
    files = sorted(p for p in OUTPUT_DIR.rglob("*") if p.is_file())
    for f in files:
        log(f"output: {f.relative_to(OUTPUT_DIR)} ({f.stat().st_size:,} bytes)")
    if not files:
        die(5, "workflow completed but wrote nothing to /output — "
               "does the workflow contain a save node?")
    return files


def _output_snapshot():
    """Manifest mode only: every file currently under /output, for before/after diffing."""
    return {p for p in OUTPUT_DIR.rglob("*") if p.is_file()}


def _collect_new_outputs(before, subdir_name):
    """Move files that appeared under /output since *before* into /output/<subdir_name>/.

    Diffing (not a fresh-subfolder-per-entry render target) because ComfyUI's
    --output-directory is fixed for the whole process; this is the only way to
    tell one workflow's files apart from another's without touching ComfyUI's
    save nodes. run_manifest() calls this unconditionally, whether the entry
    succeeded or failed partway through, and always before the next entry's
    "before" snapshot is taken — so an already-moved prior entry's files
    (complete or partial) never look "new" here, and no manifest entry can
    steal another's outputs.

    Fallback (single-workflow) mode never calls this; report_outputs() above is
    untouched and keeps the flat /output layout old bridges expect.
    """
    after = _output_snapshot()
    new_files = sorted(after - before)
    if not new_files:
        return []
    subdir = OUTPUT_DIR / subdir_name
    subdir.mkdir(parents=True, exist_ok=True)
    moved = []
    for f in new_files:
        # Keep the save node's subfolders (FJ/sc_67/...) inside wf_NN, and turn
        # any backslash left in a file name into a folder: ShareSync cannot
        # upload names containing "\\".
        rel = f.relative_to(OUTPUT_DIR).as_posix().replace("\\", "/")
        parts = [p for p in rel.split("/") if p not in ("", ".", "..")]
        dest = subdir.joinpath(*parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        f.rename(dest)
        moved.append(dest)
        log(f"output: {dest.relative_to(OUTPUT_DIR)} ({dest.stat().st_size:,} bytes)")
    # Remove the save node's now-empty folders left at the top of /output.
    for d in sorted({p.parent for p in new_files}, key=lambda p: len(p.parts), reverse=True):
        while d != OUTPUT_DIR and OUTPUT_DIR in d.parents:
            try:
                d.rmdir()
            except OSError:
                break
            d = d.parent
    return moved


def run_manifest(proc, entries):
    """Run every manifest entry against the one already-live ComfyUI process.

    Continue-on-failure: a WorkflowFailure (raised for codes 1/3/4) from one
    entry is logged and that entry is marked failed; the loop moves on to the
    next entry rather than aborting the job. die(6) from wait_done() (a dead
    ComfyUI process) is not caught here; it exits the whole process
    immediately, same as in fallback mode, since no further entry can run.

    _collect_new_outputs() runs unconditionally, whether the entry succeeded
    or failed partway through its batch loop, so a failed entry's partial
    output (e.g. 2 of 3 batch renders done before it failed) still gets moved
    into that entry's wf_NN/ folder instead of being left unattributed at the
    flat /output root. It runs before the next entry's "before" snapshot is
    taken either way, so cross-attribution between entries stays impossible.
    An entry that produced nothing at all (success or failure) gets no wf_NN/
    folder — _collect_new_outputs only creates it when there's something to
    move.

    Returns the job's overall exit code: 0 if every entry succeeded, 1 if any
    failed (including "ran clean but produced no files").
    """
    total = len(entries)
    any_failed = False
    for idx, (workflow, batch_count) in enumerate(entries, start=1):
        log(f"workflow {idx}/{total} started")
        before = _output_snapshot()
        failure = None
        try:
            if batch_count > 1:
                log(f"workflow {idx}/{total}: batch render, {batch_count} "
                    f"image(s) in this workflow (model already warm)")
            for b in range(batch_count):
                if b > 0:
                    n_seeds = randomize_seeds(workflow)
                    log(f"workflow {idx}/{total}: render {b + 1}/{batch_count} "
                        f"(randomised {n_seeds} seed input(s))")
                prompt_id = submit(workflow)
                wait_done(proc, prompt_id)
        except WorkflowFailure as exc:
            failure = exc

        # Always collect whatever this entry wrote, success or failure, before
        # the next entry's "before" snapshot is taken.
        moved = _collect_new_outputs(before, f"wf_{idx:02d}")

        if failure is not None:
            any_failed = True
            suffix = f" ({len(moved)} partial file(s))" if moved else ""
            log(f"workflow {idx}/{total} failed: {failure.message}{suffix}")
            continue

        if not moved:
            any_failed = True
            log(f"workflow {idx}/{total} failed: workflow completed but wrote "
                "nothing to /output — does the workflow contain a save node?")
            continue

        log(f"workflow {idx}/{total} succeeded ({len(moved)} file(s))")
    return 1 if any_failed else 0


def shutdown(proc):
    if proc.poll() is not None:
        return
    log("shutting down ComfyUI")
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


def randomize_seeds(workflow):
    """Give each batch render a fresh seed so the images differ. ComfyUI caches
    node outputs, so an identical re-submit would return the same image; changing
    the seed re-runs only sampling onward, reusing the already-loaded model."""
    changed = 0
    for node in workflow.values():
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if not isinstance(inputs, dict):
            continue
        for key in inputs:
            value = inputs[key]
            if key in SEED_KEYS and isinstance(value, int) and not isinstance(value, bool):
                inputs[key] = random.randint(0, 2**32 - 1)
                changed += 1
    return changed


def main():
    log_banner()

    # Manifest presence decides the mode entirely: an old bridge never writes
    # /input/spark_fuse_job.json, so it always takes the fallback branch below,
    # unchanged from before this file gained manifest support.
    manifest = load_manifest() if MANIFEST_PATH.is_file() else None
    workflow = None if manifest is not None else load_workflow()

    check_output_writable()
    write_extra_model_paths()
    stage_input_directory()
    prefetch = None
    if PREFETCH_MODELS:
        import threading
        workflows = [wf for wf, _ in manifest] if manifest is not None else [workflow]
        prefetch = threading.Thread(target=prefetch_models, args=(workflows,), daemon=True)
        prefetch.start()
    proc = start_comfyui()
    try:
        wait_ready(proc, busy=(prefetch.is_alive if prefetch is not None else None))
        precheck_workflows([wf for wf, _ in manifest] if manifest is not None else [workflow])
        if prefetch is not None and prefetch.is_alive():
            log("waiting for model prefetch to finish before submitting")
            prefetch.join()
        if PREFETCH_FAILED and PREFETCH_REQUIRED:
            die(6, f"{len(PREFETCH_FAILED)} model file(s) could not be read from /assets "
                   "(Spark's on-node asset cache is probably full; see the [assets] lines). "
                   "Stopping now: reading them from /assets would crash ComfyUI with a bus "
                   "error. Retry the job, or set PREFETCH_REQUIRED=0 to try anyway.")
        if manifest is not None:
            exit_code = run_manifest(proc, manifest)
        else:
            if BATCH_COUNT > 1:
                log(f"batch render: {BATCH_COUNT} images in one job (model loads once)")
            try:
                for i in range(BATCH_COUNT):
                    if i > 0:
                        n = randomize_seeds(workflow)
                        log(f"render {i + 1}/{BATCH_COUNT} (randomised {n} seed input(s))")
                    elif BATCH_COUNT > 1:
                        log(f"render 1/{BATCH_COUNT}")
                    prompt_id = submit(workflow)
                    wait_done(proc, prompt_id)
            except WorkflowFailure as exc:
                die(exc.code, exc.message)
            files = report_outputs()
            log(f"job complete: {len(files)} file(s) in {OUTPUT_DIR}")
    finally:
        # The copy threads are pool workers Python waits for at exit; without
        # this a job that stops early keeps copying (and running) for minutes.
        PREFETCH_CANCEL.set()
        shutdown(proc)

    if manifest is not None:
        if exit_code == 0:
            log(f"job complete: all {len(manifest)} workflow(s) succeeded")
        else:
            log("job complete: one or more workflows failed")
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
