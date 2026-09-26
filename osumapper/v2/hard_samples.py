"""Read-only, per-window audio loss diagnostics."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from .. import audio

SPIKE_THRESHOLD = 1.0
CSV_FIELDS = (
    'timestamp', 'step', 'total_loss', 'beat_loss', 'auxiliary_loss',
    'beatmap_id', 'beatmapset_id', 'artist', 'title', 'version',
    'audio_path', 'map_path', 'audio_hash', 'source_map_count',
    'chunk_start_sec', 'chunk_end_sec', 'chunk_duration_sec',
    'bpm', 'min_bpm', 'max_bpm', 'timing_point_count',
    'uninherited_timing_point_count', 'bpm_change_count',
    'bpm_changes_in_chunk', 'beat_target_count', 'strong_beat_target_count',
    'beat_target_density', 'strong_beat_density',
)


def timing_summary(points, start_ms, end_ms):
    """Use the already-indexed representative map; never parse .osu in training."""
    points = sorted(points)
    reds = [(time, 60000 / length) for time, length, active in points if active and length > 0]
    changes = [(time, bpm) for i, (time, bpm) in enumerate(reds)
               if i and not math.isclose(bpm, reds[i - 1][1], rel_tol=1e-5)]
    spans = Counter()
    for i, (time, bpm) in enumerate(reds):
        next_time = reds[i + 1][0] if i + 1 < len(reds) else end_ms
        overlap = max(0, min(end_ms, next_time) - max(start_ms, time))
        if overlap: spans[bpm] += overlap
    values = [bpm for _, bpm in reds]
    return dict(bpm=max(spans, key=spans.get) if spans else (values[0] if values else None),
                min_bpm=min(values) if values else None, max_bpm=max(values) if values else None,
                timing_point_count=len(points), uninherited_timing_point_count=len(reds),
                bpm_change_count=len(changes),
                bpm_changes_in_chunk=any(start_ms < time < end_ms for time, _ in changes))


def target_counts(target, mask):
    """Count local maxima in the augmented, supervised beat/downbeat targets."""
    out = []
    for channel in (0, 1):
        values = np.asarray(target[:, channel], dtype=np.float32)
        valid = np.asarray(mask[:, channel]) > 0
        peaks, _ = find_peaks(values, height=.5, distance=max(1, round(100 / audio.FRAME_MS)))
        count = int(valid[peaks].sum())
        if len(values) and values[0] >= .5 and valid[0]: count += 1
        out.append(count)
    return out


def make_record(step, losses, meta, target, mask):
    source = meta['source_map']
    start = meta['start_ms']
    end = min(start + round(len(target) * audio.FRAME_MS), meta['duration_ms'])
    duration = max(0, end - start) / 1000
    beats, strong = target_counts(target, mask)
    row = dict(timestamp=datetime.now(timezone.utc).isoformat(), step=step,
               total_loss=losses[0], beat_loss=losses[1], auxiliary_loss=losses[2],
               beatmap_id=source.get('beatmap_id'), beatmapset_id=source.get('set_id'),
               artist=source.get('artist'), title=source.get('title'), version=source.get('version'),
               audio_path=meta.get('audio_path'), map_path=source.get('map_path'),
               audio_hash=meta.get('audio_hash'), source_map_count=len(meta.get('maps', ())),
               chunk_start_sec=start / 1000, chunk_end_sec=end / 1000, chunk_duration_sec=duration,
               beat_target_count=beats, strong_beat_target_count=strong,
               beat_target_density=beats / duration if duration else 0,
               strong_beat_density=strong / duration if duration else 0)
    row.update(timing_summary(meta.get('timing_points', ()), start, end))
    return row


def write_spikes(run, step, batches, progress=print, total_samples=None):
    """Append completed-update samples only; a step may contain many windows."""
    run = Path(run)
    hard = []
    for losses, metas, targets, masks in batches:
        for values, meta, target, mask in zip(losses, metas, targets, masks):
            if values[0] > SPIKE_THRESHOLD:
                hard.append((make_record(step, values, meta, target, mask), meta))
    if not hard: return
    csv_path = run / 'hard_samples.csv'
    with csv_path.open('a', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if file.tell() == 0: writer.writeheader()
        for row, _ in hard: writer.writerow(row)
        file.flush()
    with (run / 'loss-spikes.jsonl').open('a', encoding='utf-8') as file:
        for row, meta in hard:
            file.write(json.dumps(dict(row, maps=meta['maps']), ensure_ascii=False, allow_nan=False) + '\n')
        file.flush()
    for row, _ in hard:
        bpm = f"{row['bpm']:.1f}" if row['bpm'] is not None else '?'
        progress(f"AUDIO SPIKE | step={step} | loss={row['total_loss']:.4f} | beat={row['beat_loss']:.4f} | recon={row['auxiliary_loss']:.4f}\n"
                 f"  {row['artist']} - {row['title']} [{row['version']}]\n"
                 f"  map={row['beatmap_id']} set={row['beatmapset_id']}\n"
                 f"  chunk={row['chunk_start_sec']:.1f}-{row['chunk_end_sec']:.1f}s | BPM={bpm} | timing_points={row['timing_point_count']} | bpm_changes={row['bpm_change_count']}\n"
                 f"  targets={row['beat_target_count']} | strong={row['strong_beat_target_count']} | density={row['beat_target_density']:.2f}/s")
    if len(hard) > 1:
        names = Counter(f"{row['artist']} - {row['title']} [{row['version']}]" for row, _ in hard)
        progress(f"SPIKE SUMMARY step {step}: {len(hard)}/{total_samples or sum(len(x[0]) for x in batches)} samples exceeded threshold | "
                 f"mean hard loss: {np.mean([row['total_loss'] for row, _ in hard]):.4f} | "
                 f"max loss: {max(row['total_loss'] for row, _ in hard):.4f} | dominant map: {names.most_common(1)[0][0]}")
