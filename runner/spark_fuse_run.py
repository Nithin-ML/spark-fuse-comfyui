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
STARTUP_TIMEOUT = int(os.environ.get("STARTUP_TIMEOUT_SECONDS", "300"))
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
    lines = ["spark_fuse:", f"  base_path: {MODEL_BASE_DIR}"]
    lines += [f"  {name}: {name}" for name in MODEL_SUBDIRS]
    EXTRA_MODEL_PATHS.write_text("\n".join(lines) + "\n")
    log(f"model search base: {MODEL_BASE_DIR}")


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


def wait_ready(proc):
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            die(6, f"ComfyUI exited during startup (code {proc.returncode})")
        try:
            stats = http_json("/system_stats", timeout=5)
            names = ", ".join(d.get("name", "?") for d in stats.get("devices", []))
            log(f"ComfyUI ready (devices: {names or 'unknown'})")
            return
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            time.sleep(1)
    die(6, f"ComfyUI not ready after {STARTUP_TIMEOUT}s")


def submit(workflow):
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
    log(f"workflow queued, prompt_id={prompt_id}")
    return prompt_id


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
        dest = subdir / f.name
        f.rename(dest)
        moved.append(dest)
        log(f"output: {dest.relative_to(OUTPUT_DIR)} ({dest.stat().st_size:,} bytes)")
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
    proc = start_comfyui()
    try:
        wait_ready(proc)
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
        shutdown(proc)

    if manifest is not None:
        if exit_code == 0:
            log(f"job complete: all {len(manifest)} workflow(s) succeeded")
        else:
            log("job complete: one or more workflows failed")
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
