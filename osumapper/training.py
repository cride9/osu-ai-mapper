from __future__ import annotations

import contextlib
import json
import math
import random
import time
from itertools import accumulate
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import audio
from .data import atomic_json, load_json, snapshot
from .dataset import MapDataset, collate
from .model import Mapper, ModelConfig
from .tokenizer import Tokenizer, VERSION as TOKEN_VERSION


@dataclass
class TrainConfig:
    preset: str = "6gb"
    steps: int = 100000
    hours: float = 72
    batch_size: int = 1
    accumulation: int = 32
    lr: float = 3e-4
    warmup: int = 500
    validate_every: int = 250
    checkpoint_minutes: float = 15
    sample_every: int = 1000
    seed: int = 42
    overfit: int = 0
    device: str = "auto"
    override_batch: bool = False
    checkpointing: bool | None = None
    memory_fraction: float = 0.68
    prefetch: bool = True


class BatchPrefetch:
    """One CPU batch ahead; its RNG is independent of model/dropout state.

    Seeds are indexed by completed update and microbatch. A checkpoint can
    discard pending work and recreate it exactly without serializing a queue.
    """
    def __init__(self, dataset, cfg, cumulative_weights, pad_to=None):
        self.dataset, self.cfg = dataset, cfg
        self.weights, self.pad_to = cumulative_weights, pad_to
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="map-batch")
        self.pending = None
        self.key = None

    def prepare(self, step, micro):
        rng = random.Random(f"{self.cfg.seed}:batch:{step}:{micro}")
        indices = rng.choices(range(len(self.dataset)), cum_weights=self.weights, k=self.cfg.batch_size)
        return collate([self.dataset.sample(i, rng) for i in indices], pad_to=self.pad_to)

    def get(self, step, micro):
        if self.key != (step, micro):
            if self.pending is not None:
                self.pending.cancel()
            self.pending = self.pool.submit(self.prepare, step, micro)
        result = self.pending.result()
        self.key = (step, micro + 1) if micro + 1 < self.cfg.accumulation else (step + 1, 0)
        self.pending = self.pool.submit(self.prepare, *self.key)
        return result

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


def device_for(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; install the CUDA 12.6 PyTorch wheel")
    return torch.device(name)


def limit_gpu_memory(device, fraction):
    if not 0 < fraction <= 1:
        raise ValueError("GPU memory fraction must be greater than zero and at most one")
    if device.type == "cuda":
        # WDDM can otherwise let the caching allocator spill into shared RAM.
        # Dynamic sequence lengths need an allocator limit as well as a batch
        # benchmark. PyTorch reclaims unused cached blocks at this limit.
        torch.cuda.set_per_process_memory_fraction(fraction, device.index if device.index is not None else torch.cuda.current_device())


def amp(device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else contextlib.nullcontext()


def move(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def loss_for(model, batch):
    logits, beats = model(batch["mel"], batch["tokens"], batch["phase"])
    token_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), batch["labels"].reshape(-1), ignore_index=-100)
    beat_loss = F.binary_cross_entropy_with_logits(beats.float(), batch["beats"], reduction="none", pos_weight=torch.tensor([6.0, 12.0], device=beats.device))
    beat_loss = (beat_loss * batch["beat_mask"]).sum() / batch["beat_mask"].sum().clamp_min(1)
    loss = token_loss + 0.3 * beat_loss
    return loss, token_loss.detach(), beat_loss.detach()


def random_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_random(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(path, model, optimizer, scaler, cfg, dataset_hash, step, elapsed, best, scheduler=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": 1, "model_config": asdict(model.config), "train_config": asdict(cfg), "schedule_total": getattr(scheduler, "total_budget", cfg.steps), "tokenizer_version": TOKEN_VERSION, "audio_version": audio.VERSION, "dataset_hash": dataset_hash, "step": step, "elapsed_seconds": elapsed, "best_validation": best, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "scheduler": scheduler.state_dict() if scheduler else None, "random": random_state()}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    # Checkpoints include optimizer/RNG state. Load only local checkpoints you
    # created or trust; Python pickle is not a safe public interchange format.
    result = torch.load(path, map_location=device, weights_only=False)
    if result.get("format") != 1 or result.get("tokenizer_version") != TOKEN_VERSION or result.get("audio_version") != audio.VERSION:
        raise ValueError("Checkpoint format/tokenizer/audio version mismatch")
    return result


@torch.no_grad()
def validate(model, dataset, device, batches=12):
    state = random_state()
    was_training = model.training
    model.eval()
    values = []
    for index in np.linspace(0, len(dataset) - 1, min(len(dataset), batches)).astype(int):
        batch = move(collate([dataset[int(index)]]), device)
        with amp(device):
            loss, tokens, beats = loss_for(model, batch)
        values.append([loss.item(), tokens.item(), beats.item()])
    model.train(was_training)
    restore_random(state)
    return dict(zip(("loss", "token_loss", "beat_loss"), np.mean(values, axis=0).tolist()))


def benchmark(run_dir, preset="6gb", device="auto", steps=5, batch_size=None, checkpointing=None, memory_fraction=0.68):
    device = device_for(device)
    limit_gpu_memory(device, memory_fraction)
    cfg = ModelConfig.tiny() if preset == "tiny" else ModelConfig()
    if checkpointing is not None:
        cfg.checkpointing = checkpointing
    batch_size = batch_size if batch_size is not None else (2 if preset == "8gb" else 1)
    if batch_size < 1 or steps < 1:
        raise ValueError("Batch size and benchmark steps must be positive")
    torch.set_num_threads(4)
    model = Mapper(len(Tokenizer()), cfg).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=False)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    length = 256 if preset == "tiny" else cfg.max_tokens - 1
    frames = round(12000 / audio.FRAME_MS)
    batch = {"mel": torch.randn(batch_size, 128, frames, device=device), "phase": torch.randn(batch_size, frames, 3, device=device), "tokens": torch.randint(1, len(Tokenizer()), (batch_size, length), device=device), "labels": torch.randint(1, len(Tokenizer()), (batch_size, length), device=device), "beats": torch.zeros(batch_size, frames, 2, device=device), "beat_mask": torch.ones(batch_size, frames, 2, device=device)}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.monotonic()
    successes, attempts = 0, 0
    warmup = 2
    measured_start = None
    measured_attempts = 0
    while successes < steps + warmup:
        attempts += 1
        if attempts > steps + warmup + 24:
            raise ValueError("Benchmark could not obtain stable FP16 optimizer updates")
        optimizer.zero_grad(set_to_none=True)
        with amp(device):
            loss, _, _ = loss_for(model, batch)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite benchmark loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer); scaler.update()
        successes += int(scaler.get_scale() >= previous_scale)
        if measured_start is not None:
            measured_attempts += 1
        elif successes == warmup:
            if device.type == "cuda":
                torch.cuda.synchronize()
            measured_start = time.monotonic()
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = (time.monotonic() - measured_start) / measured_attempts
    report = {"device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "preset": preset, "parameters": sum(p.numel() for p in model.parameters()), "batch_size": batch_size, "checkpointing": cfg.checkpointing, "decoder_tokens": length, "loss": loss.item(), "successful_updates": successes, "attempts": attempts, "warmup_updates": warmup, "seconds_per_microbatch": seconds, "examples_per_second": batch_size / seconds, "seconds_per_32_examples": 32 * seconds / batch_size, "total_seconds": time.monotonic() - start, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else None, "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3 if device.type == "cuda" else None, "free_after_gib": torch.cuda.mem_get_info()[0] / 1024**3 if device.type == "cuda" else None}
    report["memory_fraction"] = memory_fraction
    atomic_json(Path(run_dir) / "benchmark.json", report)
    del model, optimizer, scaler, batch
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def train(data_dir, run_dir, cfg=None, resume=None, progress=print, cancelled=lambda: False):
    cfg = cfg or TrainConfig()
    root = Path(run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if cfg.steps < 1 or cfg.hours <= 0 or cfg.batch_size < 1 or cfg.accumulation < 1:
        raise ValueError("Training steps, hours, batch and accumulation must be positive")
    torch.set_num_threads(4)
    device = device_for(cfg.device)
    state = load_checkpoint(resume) if resume else None
    if state:
        saved_config = TrainConfig(**state["train_config"])
        # Runtime batch tuning is opt-in; keep weights, optimizer, schedule,
        # frozen data and sampling policy from the original run.
        saved_config.hours, saved_config.steps, saved_config.device = cfg.hours, cfg.steps, cfg.device
        if cfg.override_batch:
            saved_config.batch_size, saved_config.accumulation = cfg.batch_size, cfg.accumulation
        saved_config.override_batch = cfg.override_batch
        saved_config.memory_fraction = cfg.memory_fraction
        saved_config.prefetch = cfg.prefetch
        if cfg.checkpointing is not None:
            saved_config.checkpointing = cfg.checkpointing
        cfg = saved_config
        frozen = load_json(root / "dataset.json")
        if not frozen or frozen["hash"] != state["dataset_hash"]:
            raise ValueError("Resume requires the matching frozen dataset.json in the run directory")
    else:
        if (root / "last.pt").exists():
            raise ValueError("Run already has a checkpoint; resume it or choose a new run directory")
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        frozen = snapshot(data_dir, root)
    limit_gpu_memory(device, cfg.memory_fraction)
    model_cfg = ModelConfig(**state["model_config"]) if state else ModelConfig.tiny() if cfg.preset == "tiny" else ModelConfig()
    if cfg.checkpointing is not None:
        model_cfg.checkpointing = cfg.checkpointing
    progress("Indexing training windows and loading model")
    training = MapDataset(frozen, "train", model_cfg.max_tokens, augment=not bool(cfg.overfit), overfit=cfg.overfit)
    try:
        validation = MapDataset(frozen, "validation", model_cfg.max_tokens, augment=False)
    except ValueError:
        if not cfg.overfit:
            raise ValueError("No validation songs; prepare a larger dataset before full training")
        validation = training
    model = Mapper(len(Tokenizer()), model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01, foreach=False)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    schedule_total = state.get("schedule_total", state["train_config"]["steps"]) if state else cfg.steps
    def lr_factor(step):
        if step < cfg.warmup:
            return (step + 1) / max(1, cfg.warmup)
        fraction = min(1, (step - cfg.warmup) / max(1, schedule_total - cfg.warmup))
        return max(0.1, 0.5 * (1 + math.cos(math.pi * fraction)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scheduler.total_budget = schedule_total
    step, elapsed_before, best = 0, 0.0, math.inf
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        scheduler.load_state_dict(state["scheduler"])
        step, elapsed_before, best = state["step"], state["elapsed_seconds"], state["best_validation"]
        restore_random(state["random"])
    atomic_json(root / "config.json", {"training": asdict(cfg), "model": asdict(model_cfg), "dataset_hash": frozen["hash"]})
    cumulative_weights = list(accumulate(training.weights()))
    padded_length = model_cfg.max_tokens - 1 if device.type == "cuda" and cfg.batch_size > 1 else None
    prefetch = BatchPrefetch(training, cfg, cumulative_weights, padded_length) if cfg.prefetch else None
    started = last_save = time.monotonic()
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    progress(f"Training {sum(p.numel() for p in model.parameters()):,} parameters on {device}; {len(training):,} windows; batch {cfg.batch_size} x accumulation {cfg.accumulation} = {cfg.batch_size * cfg.accumulation} examples/update; gradient checkpointing {model_cfg.checkpointing}")
    status, completion_reason = "complete", None
    try:
        while step < cfg.steps and elapsed_before + time.monotonic() - started < cfg.hours * 3600:
            if cancelled():
                status = "cancelled"; break
            optimizer.zero_grad(set_to_none=True)
            update_started = time.monotonic()
            data_seconds = 0.0
            totals = np.zeros(3)
            for micro in range(cfg.accumulation):
                if cancelled():
                    status = "cancelled"; break
                data_started = time.monotonic()
                # Fixed CUDA shapes avoid caching a separate family of large
                # buffers for every random batch's longest sequence (WDDM
                # otherwise readily spills the cache into shared memory).
                if prefetch:
                    cpu_batch = prefetch.get(step, micro)
                else:
                    indices = random.choices(range(len(training)), cum_weights=cumulative_weights, k=cfg.batch_size)
                    cpu_batch = collate([training[i] for i in indices], pad_to=padded_length)
                batch = move(cpu_batch, device)
                data_seconds += time.monotonic() - data_started
                with amp(device):
                    loss, token_loss, beat_loss = loss_for(model, batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss; latest completed checkpoint retained")
                scaler.scale(loss / cfg.accumulation).backward()
                totals += [loss.item(), token_loss.item(), beat_loss.item()]
            if status == "cancelled":
                optimizer.zero_grad(set_to_none=True); break
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            if scaler.get_scale() < previous_scale:
                progress("FP16 overflow: skipped update and reduced gradient scale")
                continue
            scheduler.step(); step += 1
            elapsed = elapsed_before + time.monotonic() - started
            metrics = {"step": step, "elapsed_seconds": elapsed, "loss": float(totals[0] / cfg.accumulation), "token_loss": float(totals[1] / cfg.accumulation), "beat_loss": float(totals[2] / cfg.accumulation), "lr": scheduler.get_last_lr()[0], "gradient_norm": float(norm), "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0, "status": "running"}
            metrics.update(batch_size=cfg.batch_size, accumulation=cfg.accumulation, memory_fraction=cfg.memory_fraction, seconds_per_update=time.monotonic() - update_started, data_seconds=data_seconds, examples_per_second=cfg.batch_size * cfg.accumulation / (time.monotonic() - update_started), reserved_vram_gib=torch.cuda.memory_reserved() / 1024**3 if device.type == "cuda" else 0)
            if step % cfg.validate_every == 0 or step == 1:
                metrics["validation"] = validate(model, validation, device)
                if metrics["validation"]["loss"] < best:
                    best = metrics["validation"]["loss"]
                    save_checkpoint(root / "best.pt", model, optimizer, scaler, cfg, frozen["hash"], step, elapsed, best, scheduler)
            with (root / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")
            atomic_json(root / "status.json", metrics)
            if step == 1 or step % 10 == 0:
                progress(f"Step {step}: loss {metrics['loss']:.4f}, elapsed {elapsed/3600:.2f}h, peak VRAM {metrics['peak_vram_gib']:.2f} GiB")
            if time.monotonic() - last_save >= cfg.checkpoint_minutes * 60 or step % cfg.validate_every == 0:
                save_checkpoint(root / "last.pt", model, optimizer, scaler, cfg, frozen["hash"], step, elapsed, best, scheduler)
                last_save = time.monotonic()
            if cfg.sample_every and step % cfg.sample_every == 0:
                from .generation import sample_training_clip
                rng = random_state()
                try:
                    sample_training_clip(model, validation, root / "samples" / f"step-{step}", device, cancelled)
                    progress(f"Sample map saved at step {step}")
                except (ValueError, RuntimeError) as exc:
                    progress(f"Sample not exportable yet: {exc}")
                finally:
                    restore_random(rng); model.train()
        if elapsed_before + time.monotonic() - started >= cfg.hours * 3600:
            status = "time_limit"
            completion_reason = "time_budget"
        elif step >= cfg.steps:
            completion_reason = "step_budget"
    except KeyboardInterrupt:
        status = "cancelled"
        completion_reason = "cancelled"
    except Exception as exc:
        status = "failed"
        completion_reason = "failed"
        atomic_json(root / "error.json", {"step": step, "error": str(exc)})
        raise
    finally:
        if prefetch:
            prefetch.close()
        optimizer.zero_grad(set_to_none=True)
        elapsed = elapsed_before + time.monotonic() - started
        # Only complete optimizer updates are checkpointed; partial gradients
        # are intentionally discarded on cancellation/failure.
        save_checkpoint(root / "last.pt", model, optimizer, scaler, cfg, frozen["hash"], step, elapsed, best, scheduler)
        atomic_json(root / "status.json", {"status": status, "completion_reason": completion_reason, "step": step, "step_target": cfg.steps, "elapsed_seconds": elapsed, "hours_target": cfg.hours, "best_validation": best if math.isfinite(best) else None, "checkpoint": str(root / "last.pt")})
        progress(f"Training {status}; checkpoint saved after step {step}")
    return str(root / "last.pt")


def evaluate(checkpoint_path, run_dir, split="test", batches=100):
    device = device_for()
    saved = load_checkpoint(checkpoint_path)
    model = Mapper(len(Tokenizer()), ModelConfig(**saved["model_config"])).to(device)
    model.load_state_dict(saved["model"])
    frozen = load_json(Path(run_dir) / "dataset.json")
    if not frozen or frozen["hash"] != saved["dataset_hash"]:
        raise ValueError("Evaluation dataset does not match checkpoint")
    dataset = MapDataset(frozen, split, augment=False)
    report = {"split": split, "checkpoint_step": saved["step"], "windows": len(dataset), **validate(model, dataset, device, batches)}
    atomic_json(Path(run_dir) / f"evaluation-{split}.json", report)
    return report
