from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import audio, mapio
from .tokenizer import Tokenizer, VERSION as TOKEN_VERSION, STYLES

DATA_VERSION = 1


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for attempt in range(8):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            # Windows readers (UI/antivirus) may briefly deny replacement.
            if attempt == 7:
                raise
            time.sleep(0.02 * (attempt + 1))


def load_json(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalized(value):
    return re.sub(r"[^\w]+", "", unicodedata.normalize("NFKC", value).casefold())


def safe_extract(path, destination):
    """Only map/audio assets, bounded decompression, no traversal or symlinks."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        total = 0
        for item in z.infolist():
            relative = Path(item.filename.replace("\\", "/"))
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination) or relative.is_absolute() or ":" in item.filename:
                raise ValueError("Unsafe archive member")
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive contains symlink")
            total += item.file_size
            if total > 2 * 1024**3 or item.file_size > 1024**3:
                raise ValueError("Archive exceeds extraction size limit")
            if item.is_dir() or target.suffix.lower() not in audio.EXTENSIONS | {".osu"}:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with z.open(item) as source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)


def source_maps(source, root):
    source = Path(source).expanduser().resolve()
    if not source.exists():
        raise ValueError(f"Source does not exist: {source}")
    archives = [source] if source.suffix.lower() == ".osz" else sorted(source.rglob("*.osz")) if source.is_dir() else []
    paths = sorted(source.rglob("*.osu")) if source.is_dir() else [source] if source.suffix.lower() == ".osu" else []
    for archive in archives:
        target = root / "imports" / digest(archive)
        safe_extract(archive, target)
        paths.extend(sorted(target.rglob("*.osu")))
    return list(dict.fromkeys(paths))


def difficulty(bm):
    import rosu_pp_py as rosu
    parsed = rosu.Beatmap(content=mapio.dumps(bm))
    if hasattr(parsed, "is_suspicious") and parsed.is_suspicious():
        raise ValueError("Difficulty calculator rejected a suspicious map")
    attrs = rosu.Difficulty(lazer=False).calculate(parsed)
    return {"stars": float(attrs.stars), "aim": float(attrs.aim), "speed": float(attrs.speed)}


def style_metrics(bm, diff):
    objects = bm.objects
    if len(objects) < 3:
        return {s: 0.0 for s in STYLES}
    intervals = np.diff([o.time for o in objects]).astype(float)
    beats = np.array([bm.clock_at(o.time)[0] for o in objects[:-1]])
    fractions = intervals / beats
    # IOI variety plus syncopation; tempo-normalized and separate from density.
    quantized = np.round(fractions * 12).astype(int)
    counts = np.array(list(Counter(quantized).values()), float)
    probs = counts / counts.sum()
    entropy = float(-(probs * np.log2(probs)).sum())
    syncopation = float(np.mean(np.abs(fractions * 4 - np.round(fractions * 4)) > 0.15))
    tapping = float(np.mean((fractions <= 0.3) & (intervals >= 25)))
    return {"aim": diff["aim"] / max(diff["aim"] + diff["speed"], 0.01), "streams": tapping, "rhythm": entropy + syncopation}


def group_and_split(records):
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(records):
        for key in ("audio_hash", "fingerprint", "song_key"):
            value = row[key]
            if not value:
                continue
            signature = (key, value)
            if signature in seen:
                parent[find(i)] = find(seen[signature])
            else:
                seen[signature] = i
    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[find(i)].append(row)
    for group in groups.values():
        # Stable across ordering; song key keeps new difficulties in the group.
        identity = min(r["song_key"] or r["audio_hash"] for r in group)
        gid = hashlib.sha256(identity.encode()).hexdigest()
        bucket = int(gid[:8], 16) % 100
        split = "train" if bucket < 90 else "validation" if bucket < 95 else "test"
        for row in group:
            row.update(group=gid, split=split)


def suggest_labels(records):
    bands = defaultdict(list)
    for row in records:
        bands[int(row["stars"])].append(row)
    for band in bands.values():
        for style in STYLES:
            values = np.array([r["metrics"][style] for r in band])
            low, high = np.quantile(values, [1/3, 2/3])
            for row in band:
                value = row["metrics"][style]
                row.setdefault("suggested", {})[style] = int(value > low) + int(value > high)


def resolve_labels(root, records):
    labels = load_json(Path(root) / "labels.json", {})
    result = []
    for original in records:
        row = dict(original)
        label = labels.get(row["id"], {})
        row["excluded"] = label.get("excluded", False)
        row["styles"] = label.get("styles", row["suggested"])
        row["label_source"] = label.get("source", "suggestion")
        result.append(row)
    return result


def set_label(root, map_id, styles, excluded=False):
    if any(v is not None and v not in (0, 1, 2) for v in styles.values()):
        raise ValueError("Labels must be 0 (Low), 1 (Medium), 2 (High), or null")
    root = Path(root)
    labels = load_json(root / "labels.json", {})
    labels[map_id] = {"styles": {s: styles.get(s) for s in STYLES}, "source": "human", "excluded": excluded}
    atomic_json(root / "labels.json", labels)


def prepare(source, destination, limit=None, progress=print, cancelled=lambda: False):
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "cache").mkdir(exist_ok=True)
    paths = source_maps(source, root)
    if limit:
        paths = paths[:limit]
    token = Tokenizer()
    old = load_json(root / "manifest.json", {})
    audio_index = old.get("audio_index", {}) if old.get("audio_version") == audio.VERSION else {}
    if not old or old.get("audio_version") == audio.VERSION:
        audio_index.update(load_json(root / "audio-index.json", {}))
    records, rejected, map_ids = [], [], set()
    timing_by_audio = defaultdict(list)
    for index, path in enumerate(paths):
        if cancelled():
            raise InterruptedError("Preparation cancelled; reusable feature caches retained")
        try:
            bm = mapio.read(path)
            if not bm.objects:
                raise ValueError("No hit objects")
            audio_path = (path.parent / bm.general["AudioFilename"]).resolve()
            if not audio_path.is_relative_to(path.parent.resolve()):
                raise ValueError("Audio reference escapes map folder")
            if not audio_path.is_file():
                raise ValueError("Missing audio")
            stat = audio_path.stat()
            key = str(audio_path)
            info = audio_index.get(key)
            if info is None or info["size"] != stat.st_size or info["mtime"] != stat.st_mtime_ns:
                pcm = audio.decode(audio_path)
                ahash = audio.pcm_hash(pcm)
                feature_path = root / "cache" / f"{ahash}.npy"
                if not feature_path.exists():
                    features = audio.spectrogram(pcm)
                    temp = feature_path.with_suffix(".tmp.npy")
                    np.save(temp, features.astype(np.float16))
                    temp.replace(feature_path)
                info = {"audio_hash": ahash, "fingerprint": audio.fingerprint(pcm), "duration_ms": len(pcm) * 1000 / audio.SR, "size": stat.st_size, "mtime": stat.st_mtime_ns}
                audio_index[key] = info
                del pcm
            if not (root / "cache" / f"{info['audio_hash']}.npy").exists():
                raise ValueError("Feature cache missing; remove manifest to rebuild")
            bm.validate(info["duration_ms"])
            # Reject unsupported numeric/path extremes visibly, never truncate.
            for obj in bm.objects:
                if len(obj.points) > 64:
                    raise ValueError("Slider exceeds 64 native anchors")
                token.object(bm, obj, obj.time)
            diff = difficulty(bm)
            map_hash = digest(path)
            map_id = hashlib.sha256((map_hash + info["audio_hash"]).encode()).hexdigest()
            if map_id in map_ids:
                continue
            map_ids.add(map_id)
            artist, title = bm.metadata.get("Artist", ""), bm.metadata.get("Title", "")
            song_key = normalized(artist) + "|" + normalized(title) if artist and title else ""
            record = {"id": map_id, "map_path": str(path.resolve()), "map_hash": map_hash, "audio_path": key, **info, "song_key": song_key, "title": title, "artist": artist, "version": bm.metadata.get("Version", ""), "creator": bm.metadata.get("Creator", ""), **diff, "objects": len(bm.objects), "metrics": style_metrics(bm, diff), "settings": {short: float(bm.difficulty[long]) for short, long in [("AR", "ApproachRate"), ("OD", "OverallDifficulty"), ("CS", "CircleSize"), ("HP", "HPDrainRate")]}}
            records.append(record)
            timing_by_audio[info["audio_hash"]].append(bm)
        except (ValueError, OSError, KeyError, IndexError, OverflowError) as exc:
            rejected.append({"path": str(path), "reason": str(exc)})
        if index % 25 == 0 or index == len(paths) - 1:
            progress(f"Prepared {index + 1}/{len(paths)} files; {len(records)} accepted, {len(rejected)} rejected")
            atomic_json(root / "prepare-progress.json", {"processed": index + 1, "total": len(paths), "accepted": len(records), "rejected": len(rejected)})
            # Save audio indexing separately so cancellation/crash can reuse it.
            atomic_json(root / "audio-index.json", audio_index)
    if not records:
        atomic_json(root / "rejected.json", rejected)
        raise ValueError("No usable maps; see rejected.json")
    progress("Building timing supervision and song-group splits")
    for ahash, maps in timing_by_audio.items():
        if cancelled():
            raise InterruptedError("Preparation cancelled")
        mel = np.load(root / "cache" / f"{ahash}.npy", mmap_mode="r")
        # Consensus of unique red-line grids. Conflicting regions get no loss.
        grids = {}
        for bm in maps:
            signature = tuple((round(p.time, 1), round(p.beat_length, 3), p.meter) for p in bm.timing if p.uninherited)
            grids[signature] = bm
        arrays = np.stack([audio.beat_targets(bm, mel.shape[1]) for bm in grids.values()])
        target = np.median(arrays, axis=0)
        mask = ((arrays.max(0) - arrays.min(0)) < 0.35).astype(np.float32)
        # Also ignore regions before/after any mapped section to avoid invented
        # timing supervision in unmapped intros or incomplete maps.
        first = min(b.objects[0].time for b in maps)
        last = max(b.objects[-1].time + b.duration(b.objects[-1]) for b in maps)
        valid = (np.arange(len(target)) * audio.FRAME_MS >= max(0, first - 4000)) & (np.arange(len(target)) * audio.FRAME_MS <= last + 2000)
        mask *= valid[:, None]
        np.savez_compressed(root / "cache" / f"{ahash}.beats.npz", target=target.astype(np.float16), mask=mask.astype(np.uint8))
    group_and_split(records)
    suggest_labels(records)
    report = {"maps": len(records), "unique_audio": len(timing_by_audio), "song_groups": len({r['group'] for r in records}), "splits": dict(Counter(r["split"] for r in records)), "star_bands": dict(sorted(Counter(str(int(r["stars"])) for r in records).items(), key=lambda p: int(p[0]))), "rejected": len(rejected), "duplicates_removed": len(paths) - len(records) - len(rejected), "warning": "Bands with few songs may generalize poorly; labels are estimates until reviewed."}
    manifest = {"version": DATA_VERSION, "audio_version": audio.VERSION, "tokenizer_version": TOKEN_VERSION, "audio_index": audio_index, "records": records, "report": report}
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "rejected.json", rejected)
    atomic_json(root / "report.json", report)
    progress(json.dumps(report, indent=2))
    return report


def snapshot(root, run_dir):
    root, run_dir = Path(root).resolve(), Path(run_dir)
    manifest = load_json(root / "manifest.json")
    if not manifest:
        raise ValueError("Prepare a dataset first")
    records = [r for r in resolve_labels(root, manifest["records"]) if not r["excluded"]]
    frozen = {"version": DATA_VERSION, "audio_version": audio.VERSION, "tokenizer_version": TOKEN_VERSION, "root": str(root), "records": records}
    identity = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    frozen["hash"] = identity
    atomic_json(run_dir / "dataset.json", frozen)
    return frozen
