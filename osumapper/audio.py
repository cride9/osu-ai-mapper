from __future__ import annotations

import hashlib
import os
import subprocess

import numpy as np
from scipy.signal import find_peaks

from .mapio import Beatmap, TimingPoint

SR, HOP, FFT, MELS = 22050, 220, 1024, 128
FRAME_MS = HOP * 1000 / SR
VERSION = "mel-v1-22050-220-1024-128-centered"
EXTENSIONS = {".mp3", ".ogg", ".wav", ".flac"}


def ffmpeg():
    import imageio_ffmpeg
    return os.environ.get("OSUMAPPER_FFMPEG") or imageio_ffmpeg.get_ffmpeg_exe()


def decode(path, sr=SR):
    args = [ffmpeg(), "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"]
    result = subprocess.run(args, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=600)
    if result.returncode:
        raise ValueError(f"Audio decode failed: {result.stderr.decode(errors='replace')[-500:]}")
    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Empty or non-finite audio")
    return audio


def filterbank():
    hz_to_mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    edges = 700 * (10 ** (np.linspace(hz_to_mel(20), hz_to_mel(SR / 2), MELS + 2) / 2595) - 1)
    freq = np.fft.rfftfreq(FFT, 1 / SR)
    filters = np.zeros((MELS, len(freq)), np.float32)
    for i in range(MELS):
        filters[i] = np.maximum(0, np.minimum((freq - edges[i]) / (edges[i + 1] - edges[i]), (edges[i + 2] - freq) / (edges[i + 2] - edges[i + 1])))
        filters[i] *= 2 / (edges[i + 2] - edges[i])
    return filters


def spectrogram(audio):
    # Frame i is centered at i*HOP samples. Do not trim or normalize silence.
    padded = np.pad(audio, (FFT // 2, FFT // 2))
    frames = np.lib.stride_tricks.sliding_window_view(padded, FFT)[::HOP]
    bank = filterbank()
    chunks = []
    window = np.hanning(FFT).astype(np.float32)
    for start in range(0, len(frames), 4096):
        power = np.abs(np.fft.rfft(frames[start:start + 4096] * window, axis=1)) ** 2
        chunks.append(np.log(np.maximum(power @ bank.T, 1e-8)).astype(np.float32))
    return np.concatenate(chunks).T


def pcm_hash(audio):
    return hashlib.sha256(np.asarray(np.clip(audio, -1, 1) * 32767, dtype="<i2").tobytes()).hexdigest()


def fingerprint(audio):
    """Cheap duplicate-recording fingerprint; metadata also groups song edits."""
    count = len(audio) // SR
    if count < 4:
        return pcm_hash(audio)
    blocks = audio[:count * SR].reshape(count, SR)
    energy = np.log(np.maximum((blocks ** 2).mean(1), 1e-8))
    energy = (energy - energy.mean()) / max(float(energy.std()), 0.01)
    return hashlib.sha256(np.round(energy * 2).astype(np.int8).tobytes()).hexdigest()


def window(mel, base_ms, duration_ms=12000):
    n = round(duration_ms / FRAME_MS)
    start = round(base_ms / FRAME_MS)
    result = np.full((MELS, n), np.log(1e-8), dtype=np.float32)
    left, right = max(0, start), min(mel.shape[1], start + n)
    if right > left:
        result[:, left - start:right - start] = mel[:, left:right]
    # Fixed scaling avoids a train/inference mismatch and keeps padding quiet.
    return (result + 5) / 7


def beat_targets(bm: Beatmap, frames: int):
    times = np.arange(frames) * FRAME_MS
    targets = np.zeros((frames, 2), np.float32)
    reds = [p for p in bm.timing if p.uninherited]
    for i, red in enumerate(reds):
        end = reds[i + 1].time if i + 1 < len(reds) else times[-1] + FRAME_MS
        first = int(np.floor((0 - red.time) / red.beat_length)) if red.time < 0 else 0
        for beat_idx in range(first, int(np.ceil((end - red.time) / red.beat_length))):
            at = red.time + beat_idx * red.beat_length
            if at < 0 or at >= end:
                continue
            center = round(at / FRAME_MS)
            for j in range(max(0, center - 3), min(frames, center + 4)):
                value = np.exp(-0.5 * ((times[j] - at) / 15) ** 2)
                targets[j, 0] = max(targets[j, 0], value)
                if beat_idx % red.meter == 0:
                    targets[j, 1] = max(targets[j, 1], value)
    return targets


def phase_features(bm: Beatmap, base_ms, frames):
    times = base_ms + np.arange(frames) * FRAME_MS
    result = np.zeros((frames, 3), np.float32)
    reds = [p for p in bm.timing if p.uninherited]
    for i, red in enumerate(reds):
        end = reds[i + 1].time if i + 1 < len(reds) else np.inf
        mask = (times >= (red.time if i else -np.inf)) & (times < end)
        phase = (times[mask] - red.time) / red.beat_length
        result[mask, 0] = np.sin(phase * 2 * np.pi)
        result[mask, 1] = np.cos(phase * 2 * np.pi)
        result[mask, 2] = np.cos(phase * 2 * np.pi / max(1, red.meter))
    return result


def decode_timing(activations, frame_ms=FRAME_MS, bpm=None, offset=None):
    """Dynamic-programming pulse path, followed by robust piecewise fitting.

    A transition penalty prefers consistent intervals while allowing tempo
    changes. Low confidence is returned to the UI rather than hidden.
    """
    activations = np.asarray(activations)
    score = activations[:, 0]
    if bpm is not None:
        if not 20 <= bpm <= 400:
            raise ValueError("BPM must be between 20 and 400")
        return [TimingPoint(float(offset or 0), 60000 / bpm)], {"confidence": 1.0, "manual": True}
    peaks, _ = find_peaks(score, distance=max(1, round(130 / frame_ms)), prominence=0.03)
    if len(peaks) < 4:
        # A transparent fallback makes inspection possible; it is never called
        # a successful automatic timing estimate.
        return [TimingPoint(float(offset or 0), 500)], {"confidence": 0.0, "warning": "Insufficient learned beat evidence; inspect BPM/offset (fallback 120 BPM)."}
    times = peaks * frame_ms
    strength = np.log(np.maximum(score[peaks], 1e-5)) + 3
    # Track interval state in 10 ms bins, 150..1500 ms (40..400 BPM).
    periods = np.arange(150, 1501, 10)
    costs = np.full((len(peaks), len(periods)), -1e9)
    parents = np.full((len(peaks), len(periods), 2), -1, dtype=int)
    for i in range(len(peaks)):
        costs[i] = strength[i]
        for j in range(i - 1, max(-1, i - 16), -1):
            delta = times[i] - times[j]
            if delta > 1550:
                break
            k = int(np.argmin(abs(periods - delta)))
            if abs(periods[k] - delta) > 40 or delta < 150:
                continue
            penalties = 4 * abs(np.log(periods / periods[k]))
            previous = int(np.argmax(costs[j] - penalties))
            candidate = costs[j, previous] - penalties[previous] + strength[i]
            if candidate > costs[i, k]:
                costs[i, k] = candidate
                parents[i, k] = [j, previous]
    i, k = np.unravel_index(np.argmax(costs), costs.shape)
    path = []
    while i >= 0:
        path.append(times[i])
        i, k = parents[i, k]
    beats = np.array(path[::-1])
    if len(beats) < 4:
        return [TimingPoint(float(offset or times[0]), 500)], {"confidence": 0.0, "warning": "Unreliable tempo path; inspect timing."}
    points = []
    start = 0
    while start < len(beats) - 1:
        end = min(len(beats), start + 8)
        period = float(np.median(np.diff(beats[start:end])))
        while end < len(beats) and abs((beats[end] - beats[end - 1]) - period) < max(25, period * 0.06):
            end += 1
        segment = beats[start:end]
        fitted = float(np.polyfit(np.arange(len(segment)), segment, 1)[0])
        at = float(np.median(segment - np.arange(len(segment)) * fitted))
        # Choose the first bar phase using learned downbeat strength.
        idxs = np.clip(np.round(segment[:4] / frame_ms).astype(int), 0, len(score) - 1)
        phase = int(np.argmax(activations[idxs, 1]))
        at += phase * fitted
        while at > segment[0]:
            at -= 4 * fitted
        if points and at <= points[-1].time:
            at = float(segment[0])
        points.append(TimingPoint(at, fitted))
        start = end - 1
    if offset is not None:
        delta = offset - points[0].time
        points = [TimingPoint(p.time + delta, p.beat_length, p.meter) for p in points]
    confidence = float(np.mean(score[np.clip(np.round(beats / frame_ms).astype(int), 0, len(score) - 1)]))
    report = {"confidence": confidence, "sections": len(points)}
    if confidence < 0.5:
        report["warning"] = "Low timing confidence; listen and optionally correct BPM/offset."
    return points, report
