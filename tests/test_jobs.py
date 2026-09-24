import os

from osumapper.data import atomic_json, load_json
from osumapper.jobs import job_status, pid_alive


def test_current_process_is_detected():
    assert pid_alive(os.getpid())


def test_interrupted_worker_is_not_reported_as_running(tmp_path, monkeypatch):
    import osumapper.jobs as jobs
    run = tmp_path / "run"
    job = tmp_path / "job.json"
    atomic_json(job, {"action": "train", "arguments": {"run_dir": str(run)}})
    atomic_json(job.with_suffix(".status.json"), {"status": "running", "pid": 123456})
    atomic_json(run / "status.json", {"status": "running", "step": 42})
    monkeypatch.setattr(jobs, "pid_alive", lambda _: False)
    text, log = job_status(job)
    assert '"status": "interrupted"' in text
    assert load_json(run / "status.json") == {"status": "interrupted", "step": 42}
