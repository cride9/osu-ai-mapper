"""Local subprocess jobs survive browser refreshes and expose cancellation."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

from .data import atomic_json, load_json


def pid_alive(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        code = wintypes.DWORD()
        try:
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def start_job(home, action, arguments):
    root = Path(home).resolve() / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    for state_file in root.glob("*.status.json"):
        previous = load_json(state_file, {})
        if previous.get("status") in ("starting", "running"):
            pid = previous.get("pid")
            alive = False
            if pid:
                alive = pid_alive(pid)
            elif time.time() - state_file.stat().st_mtime < 30:
                alive = True
            if alive:
                raise ValueError("A job is already running. Cancel it or wait before starting another.")
    job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    path = root / f"{job_id}.json"
    atomic_json(path, {"action": action, "arguments": arguments})
    status_path = path.with_suffix(".status.json")
    atomic_json(status_path, {"status": "starting", "action": action})
    with path.with_suffix(".log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-u", "-m", "osumapper.cli", "worker", str(path)], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), cwd=Path(__file__).resolve().parent.parent)
    atomic_json(status_path, {"status": "running", "pid": process.pid, "action": action, "started": time.time()})
    return str(path)


def cancel_job(path):
    if not path:
        return "No job selected"
    Path(path).with_suffix(".cancel").touch()
    return "Cancellation requested. Training saves the last completed optimizer update."


def job_status(path):
    if not path:
        return "No job selected", ""
    path = Path(path)
    state = load_json(path.with_suffix(".status.json"), {})
    if state.get("status") == "running" and state.get("pid") and not pid_alive(state["pid"]):
        state.update(status="interrupted", message="The worker process is no longer running. Resume training from last.pt in the same run folder.")
        atomic_json(path.with_suffix(".status.json"), state)
        spec = load_json(path, {})
        if spec.get("action") == "train":
            run_status = Path(spec["arguments"]["run_dir"]) / "status.json"
            previous = load_json(run_status, {})
            if previous.get("status") == "running":
                previous["status"] = "interrupted"
                atomic_json(run_status, previous)
    log = path.with_suffix(".log")
    text = log.read_text(encoding="utf-8", errors="replace")[-14000:] if log.exists() else ""
    return json.dumps(state, indent=2), text


def worker(path):
    path = Path(path)
    job = load_json(path)
    args = job["arguments"]
    started = time.time()
    def progress(message):
        print(message, flush=True)
        atomic_json(path.with_suffix(".status.json"), {"status": "running", "pid": os.getpid(), "action": job["action"], "started": started, "message": str(message)})
    cancelled = lambda: path.with_suffix(".cancel").exists()
    try:
        if job["action"] == "prepare":
            from .data import prepare
            result = prepare(**args, progress=progress, cancelled=cancelled)
        elif job["action"] == "train":
            from .training import train, TrainConfig
            args["cfg"] = TrainConfig(**args.pop("config"))
            result = train(**args, progress=progress, cancelled=cancelled)
        elif job["action"] == "generate":
            from .generation import generate, GenerationConfig
            args["cfg"] = GenerationConfig(**args.pop("config"))
            result = generate(**args, progress=progress, cancelled=cancelled)
        elif job["action"] == "benchmark":
            from .training import benchmark
            result = benchmark(**args)
        elif job["action"] == "evaluate":
            from .training import evaluate
            result = evaluate(**args)
        else:
            raise ValueError("Unknown worker action")
        atomic_json(path.with_suffix(".status.json"), {"status": "cancelled" if cancelled() else "complete", "action": job["action"], "result": result, "elapsed_seconds": time.time() - started})
        print(result, flush=True)
    except InterruptedError:
        atomic_json(path.with_suffix(".status.json"), {"status": "cancelled", "action": job["action"]})
    except Exception as exc:
        traceback.print_exc()
        atomic_json(path.with_suffix(".status.json"), {"status": "failed", "action": job["action"], "error": str(exc)})
        raise
