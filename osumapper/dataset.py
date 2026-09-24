from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import audio, mapio
from .data import digest
from .tokenizer import Tokenizer


class MapDataset(Dataset):
    def __init__(self, snapshot, split="train", max_tokens=1536, augment=True, overfit=0):
        self.root = Path(snapshot["root"])
        self.rows = [r for r in snapshot["records"] if r["split"] == split]
        self.tok, self.max_tokens = Tokenizer(), max_tokens
        self.augment = augment
        self.windows = []
        if overfit:
            # Choose clips with actual hit objects, not a silent intro.
            self.windows = [(i, max(0, self.load_map(i).objects[0].time - 1000), min(self.rows[i]["duration_ms"], self.load_map(i).objects[0].time + 7000)) for i in range(min(len(self.rows), overfit))]
        else:
            for i, row in enumerate(self.rows):
                bm = self.load_map(i)
                # Pre-index every dense remainder as another example. Reserve
                # the complete context budget before deciding output spans.
                lengths = {}
                for obj in bm.objects:
                    lengths[obj.time] = lengths.get(obj.time, 0) + len(self.tok.object(bm, obj, obj.time))
                for a, b in bm.breaks:
                    lengths[a] = lengths.get(a, 0) + 7
                events = sorted(lengths.items())
                event_index, start = 0, 0
                while start < int(row["duration_ms"]):
                    end = min(start + 8000, row["duration_ms"])
                    used, cursor = 0, event_index
                    while cursor < len(events) and events[cursor][0] < end:
                        at, size = events[cursor]
                        if at >= start:
                            if used + size > self.max_tokens - 410:
                                if at <= start:
                                    raise ValueError("Simultaneous events exceed token budget")
                                end = at
                                break
                            used += size
                        cursor += 1
                    self.windows.append((i, start, end))
                    start, event_index = end, cursor
        if not self.windows:
            raise ValueError(f"No {split} examples in dataset")

    def __len__(self):
        return len(self.windows)

    @lru_cache(maxsize=32)
    def load_map(self, index):
        row = self.rows[index]
        if digest(row["map_path"]) != row["map_hash"]:
            raise ValueError("Source map changed since dataset preparation; prepare a new dataset")
        return mapio.read(row["map_path"])

    @lru_cache(maxsize=8)
    def load_audio(self, ahash):
        mel = np.load(self.root / "cache" / f"{ahash}.npy", mmap_mode="r")
        with np.load(self.root / "cache" / f"{ahash}.beats.npz") as values:
            beats, mask = values["target"].astype(np.float32), values["mask"].astype(np.float32)
        return mel, beats, mask

    def __getitem__(self, index):
        return self.sample(index)

    def sample(self, index, rng=None):
        rng = random if rng is None else rng
        row_idx, start, indexed_end = self.windows[index]
        row, tok = self.rows[row_idx], self.tok
        bm = self.load_map(row_idx)
        # Jitter the audio origin, keeping the indexed output interval intact.
        # This avoids losing dense tails or teaching arbitrary early EOS.
        base = max(0, start - (rng.randint(1500, 2500) if self.augment else 2000))
        end = indexed_end
        context = tok.context(bm, base, start)
        styles = dict(row["styles"])
        if self.augment:
            if rng.random() < 0.2:
                styles = {}
            else:
                styles = {k: None if rng.random() < 0.1 else v for k, v in styles.items()}
        condition = tok.condition(row["stars"], styles, row["settings"])
        prefix = condition + tok.time(end - 1, base) + [tok.ids["CTX"]] + context + [tok.ids["GEN"]]
        records = tok.records(bm, base, start, end)
        targets = []
        for time, record in records:
            if len(prefix) + len(targets) + len(record) + 1 > self.max_tokens:
                raise ValueError("Indexed window exceeds token budget; rebuild its index")
            targets.extend(record)
        if not targets and records and end == records[0][0]:
            raise ValueError("One event cannot fit token budget")
        sequence = prefix + targets + [tok.ids["EOS"]]
        labels = [-100] * (len(prefix) - 1) + sequence[len(prefix):]
        mel, full_beats, full_mask = self.load_audio(row["audio_hash"])
        features = audio.window(mel, base)
        count = features.shape[1]
        frame_start = round(base / audio.FRAME_MS)
        beats, mask = np.zeros((count, 2), np.float32), np.zeros((count, 2), np.float32)
        length = min(count, len(full_beats) - frame_start)
        if length > 0:
            beats[:length] = full_beats[frame_start:frame_start + length]
            mask[:length] = full_mask[frame_start:frame_start + length]
        phase = audio.phase_features(bm, base, count)
        if self.augment:
            if rng.random() < 0.2:
                phase[:] = 0
            elif rng.random() < 0.25:
                phase = np.roll(phase, rng.randint(-3, 3), axis=0)
        return {"mel": torch.from_numpy(features), "phase": torch.from_numpy(phase), "tokens": torch.tensor(sequence[:-1]), "labels": torch.tensor(labels), "beats": torch.from_numpy(beats), "beat_mask": torch.from_numpy(mask), "map_id": row["id"]}

    def weights(self):
        # Prevent songs with many difficulties/windows from dominating while
        # increasing representation of rare one-star bands.
        from collections import Counter
        songs = Counter(self.rows[i]["group"] for i, _, _ in self.windows)
        bands = Counter(int(self.rows[i]["stars"]) for i, _, _ in self.windows)
        return [1 / (songs[self.rows[i]["group"]] ** 0.5 * bands[int(self.rows[i]["stars"])] ** 0.5) for i, _, _ in self.windows]


def collate(samples, pad_to=None):
    max_len = max(len(s["tokens"]) for s in samples)
    if pad_to is not None:
        if pad_to < max_len:
            raise ValueError("Padding length cannot truncate training examples")
        max_len = pad_to
    result = {key: torch.stack([s[key] for s in samples]) for key in ("mel", "phase", "beats", "beat_mask")}
    result["tokens"] = torch.zeros(len(samples), max_len, dtype=torch.long)
    result["labels"] = torch.full((len(samples), max_len), -100, dtype=torch.long)
    for i, sample in enumerate(samples):
        result["tokens"][i, :len(sample["tokens"])] = sample["tokens"]
        result["labels"][i, :len(sample["labels"])] = sample["labels"]
    return result
