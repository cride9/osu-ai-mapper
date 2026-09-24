from __future__ import annotations

import os
import json
import math
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from .data import load_json, resolve_labels, set_label
from .jobs import start_job, cancel_job, job_status


def build(home):
    import gradio as gr
    from . import audio, mapio
    from .preview import plot_map
    home = Path(home).resolve()
    home.mkdir(parents=True, exist_ok=True)
    dataset_default = home / "mapper-data"
    run_default = home / "runs" / "main"
    run_settings = load_json(run_default / "config.json", {})
    performance = run_settings.get("training", {})
    latest_jobs = sorted((home / "jobs").glob("*.json")) if (home / "jobs").exists() else []
    latest_jobs = [p for p in latest_jobs if not p.name.endswith(".status.json")]
    latest = str(latest_jobs[-1]) if latest_jobs else ""
    def newest_job():
        candidates = sorted(p for p in (home / "jobs").glob("*.json") if not p.name.endswith(".status.json")) if (home / "jobs").exists() else []
        return str(candidates[-1]) if candidates else ""
    styles = ["Auto / unknown", "Low", "Medium", "High"]
    def launch(action, **args):
        try:
            return start_job(home, action, args)
        except ValueError as exc:
            raise gr.Error(str(exc))
    def report(dataset):
        return load_json(Path(dataset) / "report.json", {"message": "Choose your Songs folder, then prepare the library."})
    def training_progress(run):
        run = Path(run).expanduser()
        state = load_json(run / "status.json", {})
        config = load_json(run / "config.json", {})
        cfg = config.get("training", {})
        if not state or not cfg:
            return 0, "A készültség megjelenítéséhez válaszd ki a tréning mappáját."
        step_target = max(1, int(cfg.get("steps", 100000)))
        hours_target = max(0.01, float(cfg.get("hours", 72)))
        step, elapsed = int(state.get("step", 0)), float(state.get("elapsed_seconds", 0))
        run_status = state.get("status", "unknown")
        if run_status == "complete":
            reason = state.get("completion_reason")
            detail = f"Elérte a {step_target:,} frissítést; a tréning befejeződött." if reason == "step_budget" else "A tréning befejeződött."
            return 100, detail
        percent = min(99.9, 100 * min(step / step_target, elapsed / (hours_target * 3600)))
        if run_status in ("cancelled", "time_limit", "failed", "interrupted"):
            names = {"cancelled": "Leállítva; a mentett checkpointból folytatható.", "time_limit": "Lejárt az időkeret; a mentett checkpointból folytatható.", "failed": "Hiba történt; ellenőrizd az állapotot és a checkpointot.", "interrupted": "A háttérfolyamat megszakadt; folytasd az utolsó checkpointból."}
            return percent, f"{names[run_status]} {step:,} / {step_target:,} frissítés · {elapsed/3600:.1f} / {hours_target:g} óra."
        recent = []
        metrics = run / "metrics.jsonl"
        if metrics.exists():
            with metrics.open("rb") as handle:
                handle.seek(max(0, metrics.stat().st_size - 65_536))
                lines = handle.read().decode("utf-8", errors="replace").splitlines()
            for line in lines[-100:]:
                try:
                    duration = json.loads(line).get("seconds_per_update")
                    if duration and math.isfinite(float(duration)) and duration > 0:
                        recent.append(float(duration))
                except (ValueError, TypeError):
                    pass
        samples = recent[-50:]
        average = sum(samples) / len(samples) if samples else None
        update_hours = max(0, step_target - step) * average / 3600 if average else math.inf
        budget_hours = max(0, hours_target - elapsed / 3600)
        eta = min(update_hours, budget_hours)
        eta_text = f"kb. {eta:.1f} óra" if math.isfinite(eta) else f"legfeljebb {budget_hours:.1f} óra"
        limit = "a frissítési cél elérésekor" if update_hours <= budget_hours else f"az időkeret végén ({hours_target:g} óra)"
        speed = f" · az utóbbi {len(samples)} frissítés átlaga: {average:.2f} mp/frissítés" if average else ""
        return percent, f"Fut · {step:,} / {step_target:,} frissítés ({step/step_target:.1%}) · {elapsed/3600:.1f} / {hours_target:g} óra · becsült hátralévő idő: {eta_text}, várhatóan {limit}{speed}. A becslés változhat."
    def review(dataset, cursor, direction=0, seconds=None):
        manifest = load_json(Path(dataset) / "manifest.json", {})
        rows = resolve_labels(dataset, manifest.get("records", []))
        if not rows:
            raise gr.Error("Prepare a library first")
        cursor = max(0, min(len(rows) - 1, int(cursor) + direction))
        row = rows[cursor]
        bm = mapio.read(row["map_path"])
        at = max(0, bm.objects[0].time / 1000 - 0.5) if seconds is None else max(0, float(seconds))
        pcm = audio.decode(row["audio_path"])
        first, last = round(at * audio.SR), round((at + 30) * audio.SR)
        clip = pcm[first:last]
        heading = f"**{cursor + 1}/{len(rows)} · {row['artist']} — {row['title']}**\n\n{row['version']} · {row['stars']:.2f}★ · {row['objects']} objects · labels: {row['label_source']}" + (" · EXCLUDED" if row["excluded"] else "")
        return cursor, row["id"], heading, (audio.SR, clip) if len(clip) else None, plot_map(bm, at), at, *[styles[0 if row["styles"].get(k) is None else row["styles"][k] + 1] for k in ("aim", "streams", "rhythm")]
    def save_review(dataset, map_id, aim, streams, rhythm, excluded):
        if not map_id:
            raise gr.Error("Load a map first")
        values = {k: None if v == styles[0] else styles.index(v) - 1 for k, v in zip(("aim", "streams", "rhythm"), (aim, streams, rhythm))}
        set_label(dataset, map_id, values, excluded)
        return "Excluded from future runs." if excluded else "Labels saved. Active training keeps its frozen labels; new runs use these corrections."
    def train_job(dataset, run, preset, hours, resume, overfit, batch, accumulation, checkpointing, memory_percent, prefetch):
        from dataclasses import asdict
        from .training import TrainConfig
        if resume.strip() and overfit:
            raise gr.Error("Uncheck the small overfit test to resume a checkpoint. Run diagnostics in a separate folder.")
        cfg = TrainConfig(preset="tiny" if overfit else preset, hours=float(hours), batch_size=1 if overfit else int(batch), accumulation=1 if overfit else int(accumulation), steps=200 if overfit else 100000, warmup=10 if overfit else 500, overfit=2 if overfit else 0, validate_every=50 if overfit else 250, sample_every=100 if overfit else 1000, override_batch=True, checkpointing=False if overfit else bool(checkpointing))
        cfg.memory_fraction, cfg.prefetch = float(memory_percent) / 100, bool(prefetch)
        return launch("train", data_dir=dataset, run_dir=run, config=asdict(cfg), resume=resume.strip() or None)
    def generate_job(checkpoint, song_folder, files, output, star, preset, aim, streams, rhythm, seed, variants, candidates, bpm, offset, timing_map, ar, od, cs, hp):
        from dataclasses import asdict
        from .generation import GenerationConfig
        level = lambda x: None if x == styles[0] else styles.index(x) - 1
        cfg = GenerationConfig(stars=float(star), preset=preset, aim=level(aim), streams=level(streams), rhythm=level(rhythm), seed=int(seed), variants=int(variants), candidates=int(candidates), bpm=float(bpm) if bpm else None, offset=float(offset) if offset is not None else None, timing_map=timing_map.strip() or None, ar=ar, od=od, cs=cs, hp=hp)
        inputs = files or song_folder.strip()
        if not inputs:
            raise gr.Error("Choose audio files or a folder")
        return launch("generate", checkpoint_path=checkpoint, inputs=inputs, output_dir=output, config=asdict(cfg))
    with gr.Blocks(title="Local osu! AI mapper", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Local osu! AI mapper\nTrain on your maps. Make new maps for your music. Everything runs on this computer.")
        dataset = gr.Textbox(value=str(dataset_default), label="Prepared library folder")
        with gr.Tabs():
            with gr.Tab("Library"):
                source = gr.Textbox(value=str(Path(os.environ.get("LOCALAPPDATA", "")) / "osu!" / "Songs"), label="Songs folder or .osz archive")
                gr.Markdown("The importer reads your maps and audio, groups songs into separate training and evaluation sets, and suggests styles. Original maps stay untouched.")
                with gr.Row():
                    prepare_button = gr.Button("Prepare library", variant="primary")
                    report_button = gr.Button("Refresh library report")
                library_report = gr.JSON(value=report(str(dataset_default)), label="Coverage and rejected files")
            with gr.Tab("Labeling"):
                gr.Markdown("Review each difficulty separately. Aim, tapping/streams, and rhythm complexity are independent. Suggestions are relative to maps at similar star ratings.")
                cursor, map_id = gr.State(0), gr.State("")
                with gr.Row():
                    load_button = gr.Button("Load / refresh")
                    previous_button = gr.Button("Previous")
                    next_button = gr.Button("Skip / next")
                description = gr.Markdown("Load a map to begin.")
                player = gr.Audio(label="30-second audio preview", interactive=False)
                plot = gr.Plot(label="Pattern and density")
                with gr.Row():
                    preview_at = gr.Number(label="Preview start (seconds)", value=0, minimum=0)
                    preview_button = gr.Button("Preview here")
                with gr.Row():
                    label_aim = gr.Radio(styles, value=styles[0], label="Aim intensity")
                    label_streams = gr.Radio(styles, value=styles[0], label="Streams / tapping")
                    label_rhythm = gr.Radio(styles, value=styles[0], label="Rhythm complexity")
                with gr.Row():
                    accept_button = gr.Button("Accept / save labels", variant="primary")
                    exclude_button = gr.Button("Exclude map")
                label_status = gr.Markdown()
            with gr.Tab("Training"):
                gr.Markdown("Start with the GPU check and a small overfit test in a separate run folder. Full training saves checkpoints automatically; you can close the browser while it runs.")
                run_dir = gr.Textbox(value=str(run_default), label="Training run folder")
                with gr.Row():
                    preset = gr.Radio(["6gb", "8gb"], value="6gb", label="GPU memory preset")
                    hours = gr.Number(value=72, minimum=0.01, maximum=720, label="Total training budget (hours)")
                    overfit = gr.Checkbox(value=False, label="Small overfit test (2 maps, 200 updates)")
                resume = gr.Textbox(label="Resume checkpoint (optional)", placeholder="Path to last.pt")
                with gr.Accordion("Training speed and GPU memory", open=True):
                    gr.Markdown("Batch is the number of examples processed together on the GPU. Batch × accumulation is the number per weight update; keep this at 32 for the full model. These settings also apply when resuming. Cancel and wait for the saved checkpoint before running a memory check.")
                    with gr.Row():
                        train_batch = gr.Number(value=performance.get("batch_size", 1), minimum=1, maximum=128, precision=0, label="Batch size")
                        train_accumulation = gr.Number(value=performance.get("accumulation", 32), minimum=1, maximum=128, precision=0, label="Gradient accumulation")
                    train_checkpointing = gr.Checkbox(value=run_settings.get("model", {}).get("checkpointing", True), label="Save GPU memory with gradient checkpointing")
                    memory_percent = gr.Number(value=round(performance.get("memory_fraction", 0.68) * 100), minimum=10, maximum=90, precision=0, label="GPU memory budget for tensors and cache (%)")
                    prefetch_batches = gr.Checkbox(value=performance.get("prefetch", True), label="Prepare the next batch while the GPU trains")
                    gr.Markdown("The memory budget limits tensors and their cache. The CUDA driver, desktop, and other apps use additional memory. The tested budget on this Quadro is 68%; keep it there to leave room for those uses.")
                    gr.Markdown("Larger batches usually improve speed until the GPU is saturated. Leave memory free for dense sections and the desktop. The GPU check uses the batch and checkpointing settings shown here at maximum sequence length.")
                with gr.Row():
                    benchmark_button = gr.Button("Check GPU memory")
                    train_button = gr.Button("Start / resume training", variant="primary")
                    metrics_button = gr.Button("Refresh training metrics")
                train_metrics = gr.JSON(label="Latest training status")
                completion = gr.Slider(0, 100, value=0, step=0.1, interactive=False, label="Tréning készültsége (%)")
                completion_detail = gr.Markdown("A futás 100 000 frissítésnél vagy a beállított időkeret végén fejeződik be, amelyik előbb bekövetkezik. A kijelzett százalék ezt a két határt követi; önmagában nem jelenti azt, hogy a modell térképei már jók.")
                evaluation_checkpoint = gr.Textbox(value=str(run_default / "best.pt"), label="Checkpoint to evaluate")
                evaluate_button = gr.Button("Evaluate on held-out test songs")
                gr.Markdown("Sample .osz files appear in the run’s samples folder. They are short previews using reference timing; full-song generation evaluates automatic timing too.")
            with gr.Tab("Generate"):
                checkpoint = gr.Textbox(value=str(run_default / "best.pt"), label="Trained checkpoint")
                files = gr.File(file_count="multiple", file_types=[".mp3", ".ogg", ".wav", ".flac"], type="filepath", label="Songs")
                song_folder = gr.Textbox(label="Or a folder of songs")
                with gr.Row():
                    target_stars = gr.Slider(0.5, 15, value=4, step=0.1, label="Target stars")
                    gen_preset = gr.Radio(["Auto", "Aim", "Streams", "Complex Rhythm", "Custom"], value="Auto", label="Map style")
                with gr.Row():
                    gen_aim = gr.Radio(styles, value=styles[0], label="Custom: aim")
                    gen_streams = gr.Radio(styles, value=styles[0], label="Custom: streams / tapping")
                    gen_rhythm = gr.Radio(styles, value=styles[0], label="Custom: rhythm complexity")
                with gr.Row():
                    seed = gr.Number(value=42, precision=0, label="Seed")
                    variants = gr.Number(value=1, minimum=1, maximum=20, precision=0, label="Variants per song")
                    candidates = gr.Number(value=3, minimum=1, maximum=10, precision=0, label="Candidates per variant")
                with gr.Accordion("Timing and difficulty overrides", open=False):
                    with gr.Row():
                        bpm = gr.Number(value=None, label="BPM (optional)")
                        offset = gr.Number(value=None, label="Offset in milliseconds (optional)")
                    timing_map = gr.Textbox(label="Reference .osu timing file (optional)")
                    with gr.Row():
                        ar = gr.Number(value=None, minimum=0, maximum=10, label="AR")
                        od = gr.Number(value=None, minimum=0, maximum=10, label="OD")
                        cs = gr.Number(value=None, minimum=0, maximum=10, label="CS")
                        hp = gr.Number(value=None, minimum=0, maximum=10, label="HP")
                output_dir = gr.Textbox(value=str(home / "generated"), label="Generated maps folder")
                generate_button = gr.Button("Generate maps", variant="primary")
                refresh_files = gr.Button("Refresh generated files")
                generated_files = gr.File(file_count="multiple", label="Import these .osz files into osu!")
                gr.Markdown("Reported stars are calculated from the generated map. A target miss above 0.5 stars and uncertain automatic timing are recorded in generation-report.json / generation.json.")
        gr.Markdown("### Background job")
        job = gr.Textbox(value=latest, label="Job record", visible=False)
        with gr.Row():
            refresh_job = gr.Button("Refresh progress")
            cancel = gr.Button("Cancel current job")
        status = gr.Textbox(label="Status", lines=4)
        log = gr.Textbox(label="Progress", lines=12, max_lines=20)
        timer = gr.Timer(3)
        prepare_button.click(lambda s, d: launch("prepare", source=s, destination=d), [source, dataset], job)
        report_button.click(report, dataset, library_report)
        review_outputs = [cursor, map_id, description, player, plot, preview_at, label_aim, label_streams, label_rhythm]
        load_button.click(lambda d, c: review(d, c), [dataset, cursor], review_outputs)
        previous_button.click(lambda d, c: review(d, c, -1), [dataset, cursor], review_outputs)
        next_button.click(lambda d, c: review(d, c, 1), [dataset, cursor], review_outputs)
        preview_button.click(lambda d, c, a: review(d, c, seconds=a), [dataset, cursor, preview_at], review_outputs)
        label_inputs = [dataset, map_id, label_aim, label_streams, label_rhythm]
        accept_button.click(lambda *a: save_review(*a, False), label_inputs, label_status)
        exclude_button.click(lambda *a: save_review(*a, True), label_inputs, label_status)
        preset.change(lambda p: (2, 16) if p == "8gb" else (1, 32), preset, [train_batch, train_accumulation])
        benchmark_button.click(lambda r, p, b, c, m: launch("benchmark", run_dir=r, preset=p, batch_size=int(b), checkpointing=bool(c), memory_fraction=float(m) / 100), [run_dir, preset, train_batch, train_checkpointing, memory_percent], job)
        train_button.click(train_job, [dataset, run_dir, preset, hours, resume, overfit, train_batch, train_accumulation, train_checkpointing, memory_percent, prefetch_batches], job)
        metrics_button.click(lambda r: load_json(Path(r) / "status.json", {}), run_dir, train_metrics)
        timer.tick(training_progress, run_dir, [completion, completion_detail], show_progress="hidden")
        evaluate_button.click(lambda c, r: launch("evaluate", checkpoint_path=c, run_dir=r), [evaluation_checkpoint, run_dir], job)
        generate_button.click(generate_job, [checkpoint, song_folder, files, output_dir, target_stars, gen_preset, gen_aim, gen_streams, gen_rhythm, seed, variants, candidates, bpm, offset, timing_map, ar, od, cs, hp], job)
        refresh_files.click(lambda p: [str(f) for f in sorted(Path(p).glob("*.osz"))], output_dir, generated_files)
        refresh_job.click(job_status, job, [status, log])
        timer.tick(job_status, job, [status, log], show_progress="hidden")
        cancel.click(cancel_job, job, status)
        demo.load(newest_job, outputs=job)
    return demo


def launch(home, port=7860, open_browser=True):
    # Opening the launcher twice should reuse the existing local app.
    import json
    import urllib.request
    import webbrowser
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/config", timeout=1) as response:
            config = json.load(response)
        if config.get("title") == "Local osu! AI mapper":
            if open_browser:
                webbrowser.open(f"http://127.0.0.1:{port}")
            print(f"Mapper is already running at http://127.0.0.1:{port}")
            return
    except (OSError, ValueError):
        pass
    demo = build(home)
    demo.queue().launch(server_name="127.0.0.1", server_port=port, share=False, inbrowser=open_browser, allowed_paths=[str(Path(home).resolve())])
