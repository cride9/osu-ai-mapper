import copy

import numpy as np
import pytest
import torch

from osumapper.model import Mapper, ModelConfig
from osumapper.tokenizer import Tokenizer
from osumapper import mapio
from osumapper.generation import GenerationConfig, candidate, export
from osumapper.data import difficulty


def test_cached_decoder_matches_full_causal_decoder():
    torch.set_num_threads(2)
    torch.manual_seed(1)
    model = Mapper(len(Tokenizer()), ModelConfig.tiny()).eval()
    tokens = torch.randint(1, len(Tokenizer()), (1, 25))
    memory = torch.randn(1, 20, model.config.width)
    with torch.no_grad():
        expected = model.decode(tokens, memory)
        cache = None
        values = []
        for i in range(tokens.shape[1]):
            logits, cache = model.decode_step(tokens[:, i:i+1], memory, cache, i)
            values.append(logits)
    torch.testing.assert_close(torch.cat(values, 1), expected, atol=2e-5, rtol=2e-4)


def test_causal_future_tokens_do_not_change_past():
    torch.manual_seed(2)
    model = Mapper(len(Tokenizer()), ModelConfig.tiny()).eval()
    tokens = torch.randint(1, len(Tokenizer()), (1, 20))
    other = tokens.clone(); other[:, 10:] = 100
    memory = torch.randn(1, 10, model.config.width)
    with torch.no_grad():
        a, b = model.decode(tokens, memory), model.decode(other, memory)
    torch.testing.assert_close(a[:, :10], b[:, :10])


def test_timing_head_cannot_read_ground_truth_phase():
    model = Mapper(len(Tokenizer()), ModelConfig.tiny()).eval()
    mel = torch.randn(1, 128, 100)
    with torch.no_grad():
        _, a = model.encode(mel, torch.zeros(1, 100, 3))
        _, b = model.encode(mel, torch.ones(1, 100, 3))
    torch.testing.assert_close(a, b)


def test_style_controls_are_independent():
    assert GenerationConfig(preset="Aim").styles() == {"aim": 2, "streams": 0, "rhythm": None}
    assert GenerationConfig(preset="Auto").styles() == {"aim": None, "streams": None, "rhythm": None}
    assert GenerationConfig(preset="Custom", aim=0, streams=2, rhythm=1).styles()["aim"] == 0


def test_section_boundaries_and_crossing_slider(monkeypatch):
    """Exercise the real full-song stitcher with deterministic section events."""
    import osumapper.generation as gen
    seen = []
    def section(model, mel, bm, start, end, duration, settings, cfg, device, cancelled):
        seen.append(start)
        if start == 0:
            bm.objects.append(mapio.HitObject(100, 100, 7900, "slider", points=[(300, 100)], length=560))
        else:
            active_end = bm.objects[-1].time + bm.duration(bm.objects[-1])
            at = int(max(start, active_end + 1))
            if at < end:
                bm.objects.append(mapio.HitObject(200, 200, at))
        return end
    monkeypatch.setattr(gen, "generate_section", section)
    model = Mapper(len(Tokenizer()), ModelConfig.tiny())
    mel = np.zeros((128, 2500), np.float32)
    result = candidate(model, mel, 24000, [mapio.TimingPoint(0, 500)], GenerationConfig(), {"AR": 8, "OD": 7, "CS": 4, "HP": 5}, torch.device("cpu"), 1, progress=lambda _: None)
    assert seen == [0, 8000, 16000]
    assert len(result.objects) == 3
    assert len({o.time for o in result.objects}) == 3
    assert result.objects[1].time > 9900


def test_export_is_independently_readable(tmp_path):
    import soundfile as sf
    import zipfile
    wav = tmp_path / "音楽.wav"
    sf.write(wav, np.sin(np.arange(44100 * 3) * 0.1).astype(np.float32) * 0.2, 44100)
    bm = mapio.Beatmap(timing=[mapio.TimingPoint(0, 500)], objects=[mapio.HitObject(128, 192, 1000), mapio.HitObject(256, 192, 1500)])
    stars = difficulty(bm)["stars"]
    output = export(bm, wav, tmp_path / "generated", {"measured_stars": stars, "style": "Auto"})
    with zipfile.ZipFile(output) as z:
        assert "audio.ogg" in z.namelist()
        restored = mapio.parse(z.read(next(n for n in z.namelist() if n.endswith(".osu"))).decode("utf-8"))
        assert len(restored.objects) == 2
        assert difficulty(restored)["stars"] == pytest.approx(stars)
