import copy

import pytest

from osumapper.mapio import Beatmap, HitObject, TimingPoint, dumps, read, write
from osumapper.tokenizer import Tokenizer, Grammar


def fixture_map():
    return Beatmap(timing=[TimingPoint(-50, 500), TimingPoint(1000, -50, uninherited=False), TimingPoint(4000, 400)], objects=[
        HitObject(101, 151, 123, new_combo=True, combo_skip=3, hitsound=8),
        HitObject(128, 192, 1100, "slider", curve="B", points=[(160, 100), (200, 192), (200, 192), (256, 100), (300, 192)], repeats=2, length=250.5, edge_sounds=[2, 8, 4]),
        HitObject(200, 200, 2500, "slider", curve="P", points=[(256, 150), (300, 200)], length=180),
        HitObject(128, 192, 4000, "slider", curve="L", points=[(256, 192), (256, 256)], length=180),
        HitObject(128, 100, 4800, "slider", curve="C", points=[(160, 150), (220, 180), (250, 200)], length=150),
        HitObject(256, 192, 6500, "spinner", end_time=8000)
    ], breaks=[(8100, 9500)])


def test_native_roundtrip_unicode(tmp_path):
    bm = fixture_map()
    bm.metadata["Title"] = "音楽 – Árvíztűrő"
    path = tmp_path / "譜面.osu"
    write(bm, path)
    restored = read(path)
    assert dumps(restored) == dumps(bm)
    assert restored.objects[1].points[1] == restored.objects[1].points[2]
    assert restored.objects[1].edge_sounds == [2, 8, 4]
    assert restored.clock_at(3999) == (500, 2)
    assert restored.clock_at(4000) == (400, 1)


def test_token_roundtrip_timing_and_sliders():
    bm, tok = fixture_map(), Tokenizer()
    encoded = [v for _, record in tok.records(bm, 0, 0, 10000) for v in record] + [tok.ids["EOS"]]
    restored = Beatmap(timing=copy.deepcopy(bm.timing))
    tok.decode(encoded, restored, 0)
    assert len(restored.objects) == len(bm.objects)
    assert restored.breaks == bm.breaks
    for original, result in zip(bm.objects, restored.objects):
        assert result.time == original.time
        assert result.kind == original.kind
        assert abs(result.x - original.x) <= 1
        assert abs(result.y - original.y) <= 1
        assert result.repeats == original.repeats
        assert abs(restored.duration(result) - bm.duration(original)) <= 0.50001
    assert restored.objects[1].points[1] == restored.objects[1].points[2]
    restored.validate()


def test_grammar_only_complete_legal_sequences():
    tok = Tokenizer()
    grammar = Grammar(tok, 0, 1000, 8000, 15000)
    bm = Beatmap(timing=[TimingPoint(0, 500)])
    objects = [HitObject(100, 200, 1000), HitObject(200, 200, 1200, "slider", points=[(300, 200)], length=140), HitObject(256, 192, 2500, "spinner", end_time=4000)]
    sequence = sum([tok.object(bm, obj, 0) for obj in objects], []) + [tok.ids["EOS"]]
    for token in sequence:
        assert token in grammar.allowed(), (grammar.state, tok.names[token])
        grammar.consume(token)
    assert grammar.state == "done"


def test_grammar_boundaries_and_active_duration():
    tok = Tokenizer()
    grammar = Grammar(tok, 0, 2000, 8000, 10000, busy_until=9000)
    assert grammar.allowed() == [tok.ids["EOS"]]
    grammar = Grammar(tok, 0, 1234, 1240, 10000)
    grammar.consume(tok.ids["CIRCLE"])
    grammar.consume(tok.t("T", 123))
    assert grammar.allowed() == tok.group("F", 4, 9)


def test_incomplete_event_rejected():
    tok = Tokenizer()
    bm = fixture_map()
    tokens = tok.object(bm, bm.objects[0], 0)[:-1]
    with pytest.raises(ValueError, match="Unterminated"):
        tok.decode(tokens, Beatmap(timing=copy.deepcopy(bm.timing)), 0)


def test_invalid_inputs():
    bm = fixture_map()
    bm.timing[0].beat_length = float("nan")
    with pytest.raises(ValueError):
        bm.validate()

