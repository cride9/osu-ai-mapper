"""Held-out timing measurements and reproducible generation/playtest panels."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks

from . import audio
from .data import atomic_json, load_json
from .dataset import MapDataset, collate
from .generation import GenerationConfig, generate
from .model import Mapper, ModelConfig
from .tokenizer import Tokenizer
from .training import amp, device_for, load_checkpoint, move


def match_events(predicted, expected, tolerance_ms=50):
    """One-to-one matches prevent nearby predicted peaks inflating recall."""
    used, errors = set(), []
    for predicted_time in predicted:
        available = [(abs(float(predicted_time - value)), i) for i, value in enumerate(expected) if i not in used]
        if available:
            error, i = min(available)
            if error <= tolerance_ms:
                used.add(i); errors.append(error)
    return len(errors), len(predicted), len(expected), errors


@torch.no_grad()
def timing_metrics(checkpoint_path, run_dir, split="test", windows=50):
    device = device_for()
    saved = load_checkpoint(checkpoint_path)
    frozen = load_json(Path(run_dir) / "dataset.json")
    if frozen["hash"] != saved["dataset_hash"]:
        raise ValueError("Dataset mismatch")
    dataset = MapDataset(frozen, split, augment=False)
    model = Mapper(len(Tokenizer()), ModelConfig(**saved["model_config"])).to(device)
    model.load_state_dict(saved["model"]); model.eval()
    counts = np.zeros((2, 3), int)
    errors = [[], []]
    for index in np.linspace(0, len(dataset) - 1, min(windows, len(dataset))).astype(int):
        batch = move(collate([dataset[int(index)]]), device)
        with amp(device):
            _, logits = model.encode(batch["mel"])
        probabilities = logits[0].float().sigmoid().cpu().numpy()
        truth, mask = batch["beats"][0].cpu().numpy(), batch["beat_mask"][0].cpu().numpy()
        for channel in range(2):
            predicted, _ = find_peaks(probabilities[:, channel], height=0.5, distance=round(130 / audio.FRAME_MS))
            expected, _ = find_peaks(truth[:, channel], height=0.5, distance=round(130 / audio.FRAME_MS))
            predicted = predicted[mask[predicted, channel] > 0]
            expected = expected[mask[expected, channel] > 0]
            tp, npred, ntrue, matched = match_events(predicted * audio.FRAME_MS, expected * audio.FRAME_MS)
            counts[channel] += [tp, npred, ntrue]
            errors[channel].extend(matched)
    result = {}
    for channel, name in enumerate(("beat", "downbeat")):
        tp, npred, ntrue = counts[channel]
        precision, recall = tp / max(1, npred), tp / max(1, ntrue)
        result[name] = {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(1e-9, precision + recall), "matched_error_ms": float(np.mean(errors[channel])) if errors[channel] else None, "tolerance_ms": 50}
    atomic_json(Path(run_dir) / f"timing-{split}.json", result)
    return result


def generation_panel(checkpoint_path, run_dir, count=4, progress=print, cancelled=lambda: False):
    run_dir = Path(run_dir)
    frozen = load_json(run_dir / "dataset.json")
    rows = sorted((r for r in frozen["records"] if r["split"] == "test"), key=lambda r: r["stars"])
    unique, seen = [], set()
    for row in rows:
        if row["group"] not in seen:
            unique.append(row); seen.add(row["group"])
    if not unique:
        raise ValueError("No held-out songs")
    selected = [unique[i] for i in np.linspace(0, len(unique)-1, min(count, len(unique))).astype(int)]
    panel = []
    for row in selected:
        for preset in ("Auto", "Aim", "Streams", "Complex Rhythm"):
            destination = run_dir / "panel" / row["id"][:12] / preset.replace(" ", "-")
            config = GenerationConfig(stars=row["stars"], preset=preset, candidates=1, seed=2026)
            try:
                outputs = generate(checkpoint_path, row["audio_path"], destination, config, progress, cancelled)
                for archive in outputs:
                    metadata = load_json(Path(archive).with_suffix("") / "generation.json")
                    panel.append({"song": row["title"], "style": preset, "target": row["stars"], "measured": metadata["measured_stars"], "style_metrics": metadata["style_metrics"], "timing": metadata["timing"], "archive": archive, "playtest_timing_1_to_5": None, "playtest_flow_1_to_5": None, "playtest_fun_1_to_5": None})
            except ValueError as exc:
                panel.append({"song": row["title"], "style": preset, "error": str(exc)})
    valid = [p for p in panel if "measured" in p]
    report = {"maps": panel, "valid_export_fraction": len(valid) / len(panel), "star_mae": float(np.mean([abs(p["target"]-p["measured"]) for p in valid])) if valid else None, "note": "Compare style metrics within each song. Human timing/flow/fun scores remain unfilled until actually playtested."}
    atomic_json(run_dir / "panel" / "playtest-panel.json", report)
    return report
