from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description="Local osu!standard mapper")
    commands = p.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Import Songs/.osz and cache audio")
    prep.add_argument("source"); prep.add_argument("--data", required=True); prep.add_argument("--limit", type=int)
    label = commands.add_parser("label", help="Save per-difficulty style labels")
    label.add_argument("map_id"); label.add_argument("--data", required=True)
    for name in ("aim", "streams", "rhythm"):
        label.add_argument(f"--{name}", type=int, choices=[0, 1, 2])
    label.add_argument("--exclude", action="store_true")
    train = commands.add_parser("train", help="Train or resume a model")
    train.add_argument("--data", required=True); train.add_argument("--run", required=True)
    train.add_argument("--preset", choices=["6gb", "8gb", "tiny"], default="6gb")
    train.add_argument("--steps", type=int, default=100000); train.add_argument("--hours", type=float, default=72)
    train.add_argument("--resume"); train.add_argument("--device", default="auto")
    train.add_argument("--batch-size", type=int); train.add_argument("--accumulation", type=int)
    train.add_argument("--checkpointing", action=argparse.BooleanOptionalAction, default=None, help="Trade extra computation for lower GPU memory use")
    train.add_argument("--gpu-memory-fraction", type=float, default=0.68)
    train.add_argument("--prefetch", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--overfit", type=int, default=0); train.add_argument("--warmup", type=int, default=500)
    train.add_argument("--validate-every", type=int, default=250); train.add_argument("--sample-every", type=int, default=1000)
    bench = commands.add_parser("benchmark", help="Measure CUDA peak memory and successful updates")
    bench.add_argument("--run", required=True); bench.add_argument("--preset", default="6gb", choices=["6gb", "8gb", "tiny"]); bench.add_argument("--device", default="auto")
    bench.add_argument("--batch-size", type=int); bench.add_argument("--steps", type=int, default=5)
    bench.add_argument("--checkpointing", action=argparse.BooleanOptionalAction, default=None)
    bench.add_argument("--gpu-memory-fraction", type=float, default=0.68)
    evaluate = commands.add_parser("evaluate", help="Evaluate a held-out split")
    evaluate.add_argument("checkpoint"); evaluate.add_argument("--run", required=True); evaluate.add_argument("--split", default="test", choices=["validation", "test"]); evaluate.add_argument("--batches", type=int, default=100)
    evaluate.add_argument("--timing", action="store_true", help="Also score learned beat/downbeat alignment")
    evaluate.add_argument("--panel", type=int, default=0, help="Generate four styles on N held-out songs for playtesting")
    gen = commands.add_parser("generate", help="Generate .osz packages from songs")
    gen.add_argument("checkpoint"); gen.add_argument("input"); gen.add_argument("--output", required=True)
    gen.add_argument("--stars", type=float, default=4); gen.add_argument("--preset", choices=["Auto", "Aim", "Streams", "Complex Rhythm", "Custom"], default="Auto")
    for name in ("aim", "streams", "rhythm"):
        gen.add_argument(f"--{name}", type=int, choices=[0, 1, 2])
    for name in ("ar", "od", "cs", "hp", "bpm", "offset"):
        gen.add_argument(f"--{name}", type=float)
    gen.add_argument("--timing-map"); gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--variants", type=int, default=1); gen.add_argument("--candidates", type=int, default=3)
    gen.add_argument("--temperature", type=float, default=0.8); gen.add_argument("--top-p", type=float, default=0.95); gen.add_argument("--device", default="auto")
    ui = commands.add_parser("ui", help="Open local labeling/training/generation UI")
    ui.add_argument("--home", default=os.environ.get("OSUMAPPER_HOME", str(Path.cwd() / "local-data")))
    ui.add_argument("--port", type=int, default=7860); ui.add_argument("--no-browser", action="store_true")
    worker = commands.add_parser("worker", help=argparse.SUPPRESS); worker.add_argument("job")
    return p


def main():
    args = parser().parse_args()
    if args.command == "prepare":
        from .data import prepare
        result = prepare(args.source, args.data, args.limit)
    elif args.command == "label":
        from .data import set_label
        result = set_label(args.data, args.map_id, {k: getattr(args, k) for k in ("aim", "streams", "rhythm")}, args.exclude)
    elif args.command == "train":
        from .training import train, TrainConfig
        if args.resume and (args.batch_size is None) != (args.accumulation is None):
            raise ValueError("When tuning a resumed run, provide both --batch-size and --accumulation")
        cfg = TrainConfig(preset=args.preset, steps=args.steps, hours=args.hours, batch_size=args.batch_size if args.batch_size is not None else (2 if args.preset == "8gb" else 1), accumulation=args.accumulation if args.accumulation is not None else (16 if args.preset == "8gb" else 32), device=args.device, overfit=args.overfit, warmup=args.warmup, validate_every=args.validate_every, sample_every=args.sample_every, override_batch=args.batch_size is not None or args.accumulation is not None, checkpointing=args.checkpointing)
        cfg.memory_fraction, cfg.prefetch = args.gpu_memory_fraction, args.prefetch
        result = train(args.data, args.run, cfg, args.resume)
    elif args.command == "benchmark":
        from .training import benchmark
        result = benchmark(args.run, args.preset, args.device, steps=args.steps, batch_size=args.batch_size, checkpointing=args.checkpointing, memory_fraction=args.gpu_memory_fraction)
    elif args.command == "evaluate":
        from .training import evaluate
        result = evaluate(args.checkpoint, args.run, args.split, args.batches)
        if args.timing:
            from .evaluation import timing_metrics
            result["timing"] = timing_metrics(args.checkpoint, args.run, args.split, args.batches)
        if args.panel:
            from .evaluation import generation_panel
            result["panel"] = generation_panel(args.checkpoint, args.run, args.panel)
    elif args.command == "generate":
        from .generation import generate, GenerationConfig
        values = vars(args).copy()
        for key in ("command", "checkpoint", "input", "output"):
            values.pop(key)
        result = generate(args.checkpoint, args.input, args.output, GenerationConfig(**values))
    elif args.command == "ui":
        from .ui import launch
        launch(args.home, args.port, not args.no_browser)
        return
    else:
        from .jobs import worker
        worker(args.job)
        return
    if result is not None:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
