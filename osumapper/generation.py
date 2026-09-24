from __future__ import annotations

import re
import shutil
import uuid
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from . import audio, mapio
from .data import atomic_json, difficulty, digest, load_json, style_metrics
from .model import Mapper, ModelConfig
from .tokenizer import Tokenizer, Grammar
from .training import amp, device_for, load_checkpoint


@dataclass
class GenerationConfig:
    stars: float = 4.0
    preset: str = "Auto"
    aim: int | None = None
    streams: int | None = None
    rhythm: int | None = None
    ar: float | None = None
    od: float | None = None
    cs: float | None = None
    hp: float | None = None
    seed: int = 42
    variants: int = 1
    candidates: int = 3
    temperature: float = 0.8
    top_p: float = 0.95
    bpm: float | None = None
    offset: float | None = None
    timing_map: str | None = None
    device: str = "auto"

    def styles(self):
        result = {"aim": None, "streams": None, "rhythm": None}
        presets = {"Auto": {}, "Aim": {"aim": 2, "streams": 0}, "Streams": {"aim": 0, "streams": 2}, "Complex Rhythm": {"rhythm": 2}, "Custom": {"aim": self.aim, "streams": self.streams, "rhythm": self.rhythm}}
        if self.preset not in presets:
            raise ValueError("Unknown style preset")
        result.update(presets[self.preset])
        if any(v is not None and v not in (0, 1, 2) for v in result.values()):
            raise ValueError("Style levels must be Low/Medium/High or Auto")
        return result


def choose_settings(cfg, records=None):
    near = [r for r in (records or []) if abs(r["stars"] - cfg.stars) < 0.75]
    defaults = {"AR": min(9.5, max(4, 4 + cfg.stars * 0.8)), "OD": min(9, max(3, 3 + cfg.stars * 0.7)), "CS": 4, "HP": 5}
    return {key: float(getattr(cfg, key.lower())) if getattr(cfg, key.lower()) is not None else float(np.median([r["settings"][key] for r in near])) if near else defaults[key] for key in defaults}


@torch.no_grad()
def infer_timing(model, mel, duration_ms, device, cfg, cancelled=lambda: False):
    if cfg.timing_map:
        reference = mapio.read(cfg.timing_map)
        return [p for p in reference.timing if p.uninherited], {"reference": cfg.timing_map, "confidence": 1.0}
    if cfg.bpm:
        return audio.decode_timing(np.zeros((1, 2)), bpm=cfg.bpm, offset=cfg.offset)
    scores = np.zeros((mel.shape[1], 2), np.float32)
    for start in range(0, int(duration_ms) + 1, 8000):
        if cancelled():
            raise InterruptedError("Generation cancelled")
        base = max(0, start - 2000)
        features = torch.from_numpy(audio.window(mel, base))[None].to(device)
        with amp(device):
            _, beats = model.encode(features)
        values = beats.float().sigmoid()[0].cpu().numpy()
        left = round(start / audio.FRAME_MS)
        right = min(len(scores), round((start + 8000) / audio.FRAME_MS))
        origin = round(base / audio.FRAME_MS)
        if right > left:
            scores[left:right] = values[left - origin:right - origin]
    return audio.decode_timing(scores, bpm=cfg.bpm, offset=cfg.offset)


def sample_token(logits, allowed, temperature, top_p):
    if not allowed:
        raise ValueError("No legal continuation")
    indices = torch.tensor(allowed, device=logits.device)
    values = logits[indices].float()
    if not torch.isfinite(values).all():
        raise ValueError("Non-finite generation logits")
    if temperature <= 0:
        return allowed[int(values.argmax())]
    probabilities = torch.softmax(values / temperature, dim=0)
    ordered, order = torch.sort(probabilities, descending=True)
    remove = ordered.cumsum(0) - ordered >= top_p
    ordered[remove] = 0
    selected = order[torch.multinomial(ordered, 1)]
    return int(indices[selected].item())


@torch.no_grad()
def generate_section(model, mel, bm, start, end, duration_ms, settings, cfg, device, cancelled=lambda: False):
    tok = Tokenizer()
    base = max(0, start - 2000)
    history = tok.context(bm, base, start)
    prefix = tok.condition(cfg.stars, cfg.styles(), settings) + tok.time(end - 1, base) + [tok.ids["CTX"]] + history + [tok.ids["GEN"]]
    busy = max([o.time + bm.duration(o) for o in bm.objects if o.time < start] + [b for a, b in bm.breaks if a < start] + [start])
    grammar = Grammar(tok, base, start, end, duration_ms, busy)
    features = audio.window(mel, base)
    phase = audio.phase_features(bm, base, features.shape[1])
    with amp(device):
        memory, _ = model.encode(torch.from_numpy(features)[None].to(device), torch.from_numpy(phase)[None].to(device))
        logits, caches = model.decode_step(torch.tensor([prefix], device=device), memory)
    tokens, exhausted = [], False
    while len(prefix) + len(tokens) < model.config.max_tokens:
        if cancelled():
            raise InterruptedError("Generation cancelled")
        remaining = model.config.max_tokens - len(prefix) - len(tokens)
        if grammar.state == "event" and remaining < 32 and grammar.minimum < end:
            exhausted = True
        allowed = grammar.allowed(remaining)
        token = sample_token(logits[0, -1], allowed, cfg.temperature, cfg.top_p)
        tokens.append(token)
        grammar.consume(token)
        if token == tok.ids["EOS"]:
            break
        with amp(device):
            logits, caches = model.decode_step(torch.tensor([[token]], device=device), memory, caches, offset=len(prefix) + len(tokens) - 1)
    if not tokens or tokens[-1] != tok.ids["EOS"]:
        raise ValueError("Generation token limit reached inside an event")
    tok.decode(tokens, bm, base)
    # Continue dense sections from their last committed event; don't discard
    # the rest of the requested interval when the token budget runs out.
    return max(start + 1, grammar.at + 1) if exhausted else end


def candidate(model, mel, duration_ms, timing, cfg, settings, device, seed, progress=print, cancelled=lambda: False, span=None):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model.eval()
    bm = mapio.Beatmap(timing=[mapio.TimingPoint(**asdict(p)) for p in timing])
    for short, full in [("AR", "ApproachRate"), ("OD", "OverallDifficulty"), ("CS", "CircleSize"), ("HP", "HPDrainRate")]:
        bm.difficulty[full] = str(settings[short])
    start = int(span[0]) if span else 0
    stop = min(duration_ms, span[1]) if span else duration_ms
    sections = 0
    while start < stop - 1:
        end = min(start + 8000, stop)
        # Exact digital silence needs no objects. Keep timing and history.
        a, b = round(start / audio.FRAME_MS), round(end / audio.FRAME_MS)
        if b > a and np.max(mel[:, a:b]) < np.log(1e-7):
            start = end
        else:
            start = generate_section(model, mel, bm, start, end, duration_ms, settings, cfg, device, cancelled)
        sections += 1
        progress(f"Mapped {start / 1000:.1f}/{stop / 1000:.1f}s; {len(bm.objects)} objects")
        if sections > max(100, duration_ms / 20):
            raise ValueError("Generation failed to advance through song")
    bm.validate(duration_ms, generated=True)
    if not bm.objects:
        raise ValueError("Model generated no hit objects; train longer or inspect conditioning")
    return bm


def export(bm, audio_path, destination, metadata):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    safe_title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(audio_path).stem).strip(". ")[:90] or "song"
    identity = f"{safe_title}-{uuid.uuid4().hex[:8]}"
    folder = destination / identity
    folder.mkdir()
    audio_name = "audio" + Path(audio_path).suffix.lower()
    if Path(audio_path).suffix.lower() not in {".mp3", ".ogg"}:
        # Stable-compatible container; keep all timing/leading silence.
        import subprocess
        audio_name = "audio.ogg"
        completed = subprocess.run([audio.ffmpeg(), "-nostdin", "-v", "error", "-i", str(audio_path), "-map", "0:a:0", "-c:a", "libvorbis", "-q:a", "6", str(folder / audio_name)], capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if completed.returncode:
            raise ValueError(completed.stderr.decode(errors="replace"))
    else:
        shutil.copy2(audio_path, folder / audio_name)
    bm.general["AudioFilename"] = audio_name
    bm.metadata.update(Title=Path(audio_path).stem, Creator="Local AI", Version=f"{metadata.get('style', 'Auto')} {metadata['measured_stars']:.2f} stars", Tags="ai-generated local", BeatmapID="0", BeatmapSetID="-1")
    osu_path = folder / f"{identity}.osu"
    mapio.write(bm, osu_path)
    mapio.read(osu_path).validate()
    atomic_json(folder / "generation.json", metadata)
    archive = destination / f"{identity}.osz"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in folder.iterdir():
            z.write(path, path.name)
    return str(archive.resolve())


def generate(checkpoint_path, inputs, output_dir, cfg=None, progress=print, cancelled=lambda: False):
    cfg = cfg or GenerationConfig()
    if not 0 <= cfg.stars <= 20 or cfg.variants < 1 or cfg.candidates < 1:
        raise ValueError("Stars must be 0–20; variants and candidates must be positive")
    for name in ("ar", "od", "cs", "hp"):
        value = getattr(cfg, name)
        if value is not None and not 0 <= value <= 10:
            raise ValueError(f"{name.upper()} must be between 0 and 10")
    if not 0 < cfg.top_p <= 1 or cfg.temperature < 0:
        raise ValueError("Invalid sampling settings")
    cfg.styles()
    torch.set_num_threads(4)
    device = device_for(cfg.device)
    state = load_checkpoint(checkpoint_path)
    model = Mapper(len(Tokenizer()), ModelConfig(**state["model_config"])).to(device)
    model.load_state_dict(state["model"]); model.eval()
    frozen = load_json(Path(checkpoint_path).parent / "dataset.json", {})
    settings = choose_settings(cfg, frozen.get("records"))
    checkpoint_hash = digest(checkpoint_path)
    if isinstance(inputs, (str, Path)):
        source = Path(inputs)
        inputs = sorted(p for p in source.rglob("*") if p.suffix.lower() in audio.EXTENSIONS) if source.is_dir() else [source]
    if not inputs:
        raise ValueError("No supported audio files found")
    outputs, failures = [], []
    for audio_path in inputs:
        if cancelled():
            raise InterruptedError("Generation cancelled")
        progress(f"Reading {Path(audio_path).name}")
        pcm = audio.decode(audio_path)
        duration_ms = len(pcm) * 1000 / audio.SR
        mel = audio.spectrogram(pcm)
        del pcm
        timing, timing_report = infer_timing(model, mel, duration_ms, device, cfg, cancelled)
        for variant in range(cfg.variants):
            candidates, errors = [], []
            for attempt in range(cfg.candidates):
                seed = cfg.seed + variant * cfg.candidates + attempt
                progress(f"Variant {variant + 1}, candidate {attempt + 1}/{cfg.candidates}")
                try:
                    bm = candidate(model, mel, duration_ms, timing, cfg, settings, device, seed, progress, cancelled)
                    stats = difficulty(bm)
                    candidates.append((abs(stats["stars"] - cfg.stars), bm, stats, seed))
                except ValueError as exc:
                    errors.append(str(exc)); progress(f"Rejected candidate: {exc}")
            if not candidates:
                failures.append({"audio": str(audio_path), "variant": variant + 1, "errors": errors})
                continue
            miss, bm, stats, seed = min(candidates, key=lambda x: x[0])
            metadata = {"requested_stars": cfg.stars, "measured_stars": stats["stars"], "difficulty_miss": miss > 0.5, "style": cfg.preset, "settings": asdict(cfg), "seed": seed, "checkpoint": str(Path(checkpoint_path).resolve()), "checkpoint_sha256": checkpoint_hash, "checkpoint_step": state["step"], "timing": timing_report, "style_metrics": style_metrics(bm, stats), "candidate_errors": errors}
            outputs.append(export(bm, audio_path, output_dir, metadata))
            progress(f"Exported {stats['stars']:.2f} stars (requested {cfg.stars:.2f})" + (" — target missed by more than 0.5" if miss > 0.5 else ""))
    atomic_json(Path(output_dir) / "generation-report.json", {"outputs": outputs, "failures": failures})
    if not outputs:
        raise ValueError("No valid maps generated; inspect generation-report.json. An early checkpoint may need more training.")
    return outputs


def sample_training_clip(model, dataset, destination, device, cancelled=lambda: False):
    row = dataset.rows[0]
    reference = dataset.load_map(0)
    mel, _, _ = dataset.load_audio(row["audio_hash"])
    start = max(0, reference.objects[0].time - 1000)
    cfg = GenerationConfig(stars=row["stars"], candidates=1, temperature=0.6)
    timing = [p for p in reference.timing if p.uninherited]
    bm = candidate(model, mel, row["duration_ms"], timing, cfg, row["settings"], device, 1234, progress=lambda _: None, cancelled=cancelled, span=(start, start + 16000))
    stats = difficulty(bm)
    return export(bm, row["audio_path"], destination, {"style": "Training preview", "measured_stars": stats["stars"], "requested_stars": row["stars"], "reference_timing": True, "partial_song": True})
