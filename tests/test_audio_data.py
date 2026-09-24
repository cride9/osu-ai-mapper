import json
import zipfile

import numpy as np
import pytest
import soundfile as sf

from osumapper import audio
from osumapper.data import atomic_json, group_and_split, resolve_labels, safe_extract, set_label, snapshot
from osumapper.mapio import Beatmap, TimingPoint


@pytest.mark.parametrize("sr", [22050, 44100, 48000])
def test_decode_resample_preserves_impulse_time(tmp_path, sr):
    x = np.zeros(sr * 3, np.float32)
    x[sr] = 0.9
    path = tmp_path / "音楽.wav"
    sf.write(path, np.stack([x, x], axis=1), sr, subtype="FLOAT")
    pcm = audio.decode(path)
    assert abs(len(pcm) - audio.SR * 3) <= 1
    assert abs(np.argmax(abs(pcm)) / audio.SR - 1) < 1 / audio.SR
    assert np.max(abs(pcm[:audio.SR // 2])) < 1e-8
    mel = audio.spectrogram(pcm)
    maximum = np.argmax(np.exp(mel).sum(0))
    assert abs(maximum * audio.FRAME_MS - 1000) < audio.FRAME_MS
    clip = audio.window(mel, 500, 1000)
    clip_peak = np.argmax(np.exp(clip * 7 - 5).sum(0))
    assert abs(clip_peak * audio.FRAME_MS - 500) < audio.FRAME_MS * 1.5


def test_beats_and_tempo_changes():
    bm = Beatmap(timing=[TimingPoint(0, 500), TimingPoint(8000, 400)])
    targets = audio.beat_targets(bm, round(16000 / audio.FRAME_MS))
    for at in [0, 500, 2000, 8000, 8400, 8800]:
        assert targets[round(at / audio.FRAME_MS), 0] > 0.9
    points, report = audio.decode_timing(targets)
    assert report["confidence"] > 0.8
    assert any(abs(p.beat_length - 500) < 20 for p in points)
    assert any(abs(p.beat_length - 400) < 20 for p in points)


def test_group_split_prevents_song_leaks():
    rows = [dict(id=str(i), audio_hash=f"a{i}", fingerprint=f"f{i}", song_key=f"song{i}") for i in range(300)]
    rows[1]["audio_hash"] = rows[0]["audio_hash"]
    rows[2]["song_key"] = rows[1]["song_key"]
    rows[3]["fingerprint"] = rows[2]["fingerprint"]
    group_and_split(rows)
    assert len({r["group"] for r in rows[:4]}) == 1
    assert len({r["split"] for r in rows[:4]}) == 1
    expected = {r["id"]: (r["group"], r["split"]) for r in rows}
    group_and_split(rows[::-1])
    assert expected == {r["id"]: (r["group"], r["split"]) for r in rows}
    assert {r["split"] for r in rows} == {"train", "validation", "test"}


def test_labels_frozen_and_unknown_distinct_from_low(tmp_path):
    rows = [{"id": "map", "suggested": {"aim": 1, "streams": 1, "rhythm": 1}}]
    atomic_json(tmp_path / "manifest.json", {"records": rows})
    set_label(tmp_path, "map", {"aim": 0, "streams": None, "rhythm": 2})
    resolved = resolve_labels(tmp_path, rows)
    assert resolved[0]["styles"] == {"aim": 0, "streams": None, "rhythm": 2}
    frozen = snapshot(tmp_path, tmp_path / "run")
    set_label(tmp_path, "map", {"aim": 2, "streams": 2, "rhythm": 0})
    assert frozen["records"][0]["styles"]["aim"] == 0
    assert frozen["hash"] != snapshot(tmp_path, tmp_path / "other")["hash"]


def test_osz_path_traversal_rejected(tmp_path):
    archive = tmp_path / "bad.osz"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../outside.osu", "malicious")
    with pytest.raises(ValueError, match="Unsafe"):
        safe_extract(archive, tmp_path / "extracted")
    assert not (tmp_path / "outside.osu").exists()

