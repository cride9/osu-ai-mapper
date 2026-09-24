import copy
import random
from pathlib import Path

import numpy as np
import torch
import pytest

from osumapper import audio, mapio
from osumapper.data import atomic_json, digest, load_json, snapshot
from osumapper.dataset import MapDataset, collate
from osumapper.tokenizer import Tokenizer
from osumapper.training import BatchPrefetch, TrainConfig, train, load_checkpoint


def make_dataset(root, dense=False):
    root.mkdir(parents=True)
    (root / "cache").mkdir()
    rows = []
    bm = mapio.Beatmap(timing=[mapio.TimingPoint(0, 500)], objects=[mapio.HitObject(100 + (i % 4)*80, 100 + (i % 3)*70, 1000+i*(8 if dense else 200)) for i in range(700 if dense else 20)])
    path = root / "test.osu"
    mapio.write(bm, path)
    for index, split in enumerate(["train", "validation", "test"]):
        ahash = f"audio{index}"
        mel = np.random.default_rng(1).normal(size=(128, 900)).astype(np.float16)
        np.save(root / "cache" / f"{ahash}.npy", mel)
        targets = audio.beat_targets(bm, mel.shape[1])
        np.savez_compressed(root / "cache" / f"{ahash}.beats.npz", target=targets, mask=np.ones_like(targets))
        rows.append({"id": str(index), "map_path": str(path), "map_hash": digest(path), "audio_hash": ahash, "duration_ms": 8500, "split": split, "group": str(index), "stars": 4, "settings": {"AR": 8, "OD": 7, "CS": 4, "HP": 5}, "suggested": {"aim": 1, "streams": 0, "rhythm": 1}})
    atomic_json(root / "manifest.json", {"records": rows})


def test_dense_windows_cover_every_target_without_truncation(tmp_path):
    root = tmp_path / "data"
    make_dataset(root, dense=True)
    frozen = snapshot(root, tmp_path / "run")
    dataset = MapDataset(frozen, augment=False)
    assert len(dataset) > 2
    seen = []
    tok = Tokenizer()
    for index, (_, start, end) in enumerate(dataset.windows):
        example = dataset[index]
        assert len(example["tokens"]) <= 1536
        assert len(example["tokens"]) == len(example["labels"])
        target = example["labels"][example["labels"] != -100].tolist()
        result = mapio.Beatmap(timing=[mapio.TimingPoint(0, 500)])
        tok.decode(target, result, max(0, start - 2000))
        seen += [o.time for o in result.objects]
    assert seen == [o.time for o in dataset.load_map(0).objects]


def test_resume_matches_uninterrupted_cpu_training(tmp_path):
    root = tmp_path / "data"
    make_dataset(root)
    cfg = TrainConfig(preset="tiny", device="cpu", steps=6, hours=1, accumulation=1, warmup=2, overfit=1, validate_every=3, sample_every=0)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    train(root, full, cfg, progress=lambda _: None)
    def cancelled():
        status = load_json(resumed / "status.json", {})
        return status.get("step", 0) >= 3
    train(root, resumed, copy.deepcopy(cfg), progress=lambda _: None, cancelled=cancelled)
    assert load_checkpoint(resumed / "last.pt")["step"] == 3
    train(root, resumed, copy.deepcopy(cfg), resume=resumed / "last.pt", progress=lambda _: None)
    a, b = load_checkpoint(full / "last.pt"), load_checkpoint(resumed / "last.pt")
    assert a["step"] == b["step"] == 6
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    assert a["scheduler"] == b["scheduler"]


def test_resume_can_tune_batch_without_resetting_training(tmp_path):
    root, run = tmp_path / "data", tmp_path / "run"
    make_dataset(root)
    cfg = TrainConfig(preset="tiny", device="cpu", steps=3, hours=1, accumulation=2, warmup=2, overfit=1, validate_every=3, sample_every=0)
    train(root, run, cfg, progress=lambda _: None, cancelled=lambda: load_json(run / "status.json", {}).get("step", 0) >= 1)
    before = load_checkpoint(run / "last.pt")
    requested = TrainConfig(device="cpu", steps=3, batch_size=2, accumulation=1, override_batch=True, checkpointing=True)
    train(root, run, requested, resume=run / "last.pt", progress=lambda _: None)
    after = load_checkpoint(run / "last.pt")
    assert before["step"] == 1 and after["step"] == 3
    assert after["train_config"]["batch_size"] == 2
    assert after["train_config"]["accumulation"] == 1
    assert after["train_config"]["preset"] == "tiny"
    assert after["train_config"]["overfit"] == 1
    assert after["model_config"]["checkpointing"] is True
    assert after["dataset_hash"] == before["dataset_hash"]
    assert after["schedule_total"] == before["schedule_total"]
    assert after["scheduler"]["last_epoch"] == 3
    assert all(state["step"].item() == 3 for state in after["optimizer"]["state"].values())
    assert all(torch.isfinite(value).all() for value in after["model"].values())


def test_prefetch_recreates_batch_without_changing_global_rng(tmp_path):
    root = tmp_path / "data"
    make_dataset(root)
    dataset = MapDataset(snapshot(root, tmp_path / "run"), augment=True)
    cfg = TrainConfig(batch_size=2, accumulation=1)
    loader = BatchPrefetch(dataset, cfg, [float(i + 1) for i in range(len(dataset))], pad_to=1535)
    state = random.getstate()
    try:
        expected = loader.prepare(50, 0)
        actual = loader.get(50, 0)
        # Simulate discarding a prefetched batch and reconstructing on resume.
        recreated = loader.prepare(51, 0)
        following = loader.get(51, 0)
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            torch.testing.assert_close(following[key], recreated[key], rtol=0, atol=0)
        assert actual["tokens"].shape == (2, 1535)
        assert (actual["labels"][:, -1] == -100).all()
        assert random.getstate() == state
    finally:
        loader.close()
    with pytest.raises(ValueError, match="truncate"):
        collate([dataset[0]], pad_to=1)


def test_status_write_retries_transient_windows_read_lock(tmp_path, monkeypatch):
    real_replace = Path.replace
    calls = []
    def locked_twice(source, destination):
        calls.append(source)
        if len(calls) < 3:
            raise PermissionError("temporary reader lock")
        return real_replace(source, destination)
    monkeypatch.setattr(Path, "replace", locked_twice)
    path = tmp_path / "status.json"
    atomic_json(path, {"step": 42})
    assert load_json(path) == {"step": 42}
    assert len(calls) == 3
