"""6 GiB defaults, exclusive GPU work and atomic trusted-local checkpoints."""
from __future__ import annotations

import contextlib
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from ..data import load_json
from ..jobs import pid_alive

MAX_RESERVED_GIB=4.8
MIN_FREE_GIB=.75
_gpu_depth=0


def device_for(name="auto"):
    if name=="auto": name="cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available(): raise ValueError("CUDA is unavailable")
    return torch.device(name)


def active_v1_jobs():
    project=Path(__file__).resolve().parents[2]
    homes=[project/"local-data",project.parent.parent/"work"]
    if os.environ.get("OSUMAPPER_HOME"): homes.append(Path(os.environ["OSUMAPPER_HOME"]))
    for home in homes:
        for p in (home/"jobs").glob("*.status.json"):
            s=load_json(p,{})
            if s.get("status") in ("starting","running") and s.get("action") in ("train","generate","benchmark","evaluate") and s.get("pid")!=os.getpid() and s.get("pid") and pid_alive(s["pid"]):
                yield s


@contextlib.contextmanager
def gpu_session(device,home=None):
    global _gpu_depth
    if device.type!="cuda":
        yield; return
    if _gpu_depth:
        _gpu_depth+=1
        try: yield
        finally: _gpu_depth-=1
        return
    if list(active_v1_jobs()): raise RuntimeError("A V1 GPU job is active. V2 will not start beside it; finish or stop V1 yourself first.")
    lock=Path(tempfile.gettempdir())/"osu-mapper-v2-gpu.lock"
    if lock.exists():
        try: pid=int(lock.read_text())
        except (ValueError,OSError): pid=-1
        if pid>0 and pid_alive(pid): raise RuntimeError("Another V2 GPU job is active")
        lock.unlink(missing_ok=True)
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode()); os.close(fd)
    try:
        free,total=torch.cuda.mem_get_info(device)
        if free<MIN_FREE_GIB*1024**3: raise RuntimeError("Insufficient dedicated GPU headroom")
        index=device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(min(MAX_RESERVED_GIB*1024**3/total,(free-MIN_FREE_GIB*1024**3)/total),index)
        _gpu_depth=1
        yield
    finally:
        _gpu_depth=0; lock.unlink(missing_ok=True)


def memory_stats(device):
    if device.type!="cuda": return {"reserved_gib":0.,"peak_allocated_gib":0.,"free_gib":None}
    free,_=torch.cuda.mem_get_info(device)
    return {"reserved_gib":torch.cuda.memory_reserved(device)/1024**3,"peak_allocated_gib":torch.cuda.max_memory_allocated(device)/1024**3,"free_gib":free/1024**3}


def enforce_headroom(device):
    metrics=memory_stats(device)
    if device.type=="cuda" and (metrics["reserved_gib"]>MAX_RESERVED_GIB+.01 or metrics["free_gib"]<MIN_FREE_GIB):
        raise RuntimeError("Measured dedicated VRAM headroom failed the 6 GB policy; reduce physical batch, not context")
    return metrics


def precision_for(name,device):
    if name=="bf16" and (device.type!="cuda" or not torch.cuda.is_bf16_supported(including_emulation=False)):
        raise ValueError("Native BF16 is unavailable on this GPU; use fp16 (Quadro RTX 3000)")
    return torch.bfloat16 if name=="bf16" else torch.float16


def random_state():
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_random(s):
    random.setstate(s["python"]); np.random.set_state(s["numpy"]); torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(s["cuda"])


def save_checkpoint(path,state):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp.pt"); torch.save(state,tmp); tmp.replace(path)


def load_checkpoint(path):
    # Only user-selected locally produced checkpoints are supported.
    state=torch.load(path,map_location="cpu",weights_only=False)
    if state.get("format")!="osu-v2": raise ValueError("This is not a V2 checkpoint; V1 files must use the V1 interface")
    return state
