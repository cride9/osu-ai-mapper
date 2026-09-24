# Local osu! AI mapper

A complete local pipeline for training an osu!standard audio-to-beatmap model from scratch. Includes a custom event tokenizer, audio cache, beat/downbeat supervision, 32.4M-parameter Transformer, training/checkpoint resume, constrained generation, difficulty calculation, `.osz` export, and a browser interface.

The code is usable immediately. A randomly initialized or briefly tested checkpoint is **not a finished mapper**. Musicality, style following, and fun must be assessed after training on held-out songs and through actual playtesting.

## Open the application

On the computer where this was built, double-click **Start.cmd**. It uses the installed project environment and the prepared local data. If the interface is already running, open <http://127.0.0.1:7860>.

On another Windows machine:

1. Install **64-bit Python 3.12** and an NVIDIA driver compatible with CUDA 12.6.
2. Run `Install.ps1` in PowerShell. It creates `.venv` and installs the pinned CUDA PyTorch build and dependencies. FFmpeg is included by `imageio-ffmpeg`; a separate CUDA Toolkit or Rust compiler is unnecessary for these Windows wheels.
3. Open `Start.cmd`.

The app listens on **127.0.0.1 only**, disables Gradio analytics, and does not create a public share link. There are no pretrained weights, hosted inference services, account requirements, or automatic beatmap downloads.

## Workflow

### 1. Library

Choose an osu!stable **Songs** folder, a folder of `.osz` archives, an individual archive, or one `.osu` file. Prepare the library and refresh the report.

Original maps/audio are read only. The prepared library contains:

- `manifest.json`: source identities, stars, styles, song groups, and splits.
- `cache/`: one log-mel cache per decoded recording, plus timing targets/masks.
- `labels.json`: human corrections, kept independently of preparation.
- `report.json` and `rejected.json`: difficulty coverage and exact rejection reasons.

Maps are grouped by decoded-audio identity, a conservative audio fingerprint, and normalized artist/title. All difficulties and grouped recordings share one deterministic 90/5/5 train/validation/test assignment. Percentages are approximate because whole songs are indivisible. Small libraries may lack a validation group; full training then requires more songs.

Supported training objects: circles, Bezier/linear/perfect-circle/Catmull sliders, repeated Bezier anchors, slider repeats, SV changes, spinners, breaks, combos, and standard hitsounds. Nonstandard modes, missing audio, invalid timings, sliders exceeding 64 native anchors/repeats, out-of-vocabulary coordinates, and objects beyond the recording are reported and excluded. Videos, storyboards, custom samples, and arbitrary hitsound filenames are not learning targets. Native parser/writer round trips preserve edge sounds; generated sliders use default edge samples.

### 2. Labeling

Load a map and listen to its preview. The diagram shows four seconds of patterns; enter a different preview time to inspect another section. The audio preview covers 30 seconds starting at that position.

Use **Low / Medium / High** independently for:

- **Aim:** relative emphasis on cursor movement.
- **Streams / tapping:** frequent short beat-relative intervals.
- **Rhythm complexity:** diversity and syncopation of beat-relative intervals.

Suggestions are normalized among maps with similar stars. They are weak labels, not expert judgments. Accept them, correct them, leave a field unknown, or exclude a bad map. Unknown differs from Low. Human changes survive library preparation. A running training job retains its frozen labels; begin a new run to incorporate later edits.

### 3. Training

Use **Check GPU memory**, then a **Small overfit test** in a separate run folder. The small model is a diagnostic that should learn a few examples; use the **6gb** or **8gb** preset for the actual model.

The portable presets start conservatively: 6 GB uses batch 1 / accumulation 32, and 8 GB uses batch 2 / accumulation 16. In **Training speed and GPU memory**, increase the batch and reduce accumulation to keep their product at 32. The tuned local Quadro RTX 3000 configuration is **batch 8 / accumulation 4**, **gradient checkpointing off**, **memory budget 68%**, and **next-batch preparation on**. The existing `main` run uses these settings. The UI initially loads the saved main-run settings. Changing the GPU preset resets the batch controls to conservative values. Always inspect the displayed values before starting.

**Check GPU memory** benchmarks the displayed batch size, memory budget, and checkpointing option at maximum sequence length after warmup. Compare `examples_per_second`, peak allocation, and free memory. Memory usage alone does not measure training speed. If the tuned setting does not fit, use batch 4 / accumulation 8 and enable gradient checkpointing. The RTX 3050 has not been tested directly.

The memory budget caps PyTorch tensors and their allocator cache; CUDA context, desktop, and other applications add to the total. On Windows, an unrestricted cache can grow beyond dedicated VRAM into slow shared system memory even when active tensors are small. CUDA training now pads batches to a fixed decoder length, caps the cache, and prepares one CPU batch ahead. The batch-32 configuration was rejected after real-data monitoring exposed cache growth. Do not raise the memory budget just to fill the GPU.

Full-model defaults:

| Setting | Value |
|---|---|
| Parameters | 32,361,986 |
| Encoder / decoder | 6 / 8 layers |
| Width / heads / feed-forward | 384 / 6 / 1536 |
| Audio | 22.05 kHz mono, 128 mel bands, 220-sample hop, centered 1,024-sample FFT |
| Context | 12 seconds, normally 8 seconds of output with prior-map context |
| Decoder limit | 1,536 tokens; dense remainders become additional windows |
| Precision | FP16 autocast and scaling, FP32 parameters/AdamW state |
| Optimizer | AdamW, 3e-4 peak LR, 0.01 weight decay, warmup/cosine, gradient clipping |
| Initial budget | 72 total hours or 100,000 optimizer updates |

Timing supervision comes from uninherited timing points; conflicting grids are masked. The timing head cannot see ground-truth phase features. The map decoder receives timing features, with phase jitter/dropout during training to reduce reliance on perfect input. Auto is learned by hiding style conditions on 20% of training examples.

Checkpoints are written at validation intervals and every 15 minutes, plus on cancellation/completion. **Cancel** discards incomplete gradients and saves completed optimizer updates. Resume `last.pt` in the **same run folder**, keeping its `dataset.json`. Resume restores optimizer/scaler/scheduler and random states, and does not silently use new labels. Increase the total hours/steps to extend a completed budget.

The Training tab's live progress indicator tracks the earlier of the update limit and total-hour limit. At 100%, that run is done; its final status says whether it reached the update goal or time limit, was cancelled, or failed. The time estimate uses recent measured update speed. A finished run still needs held-out evaluation and human playtesting before its maps can be called good.

The UI applies the displayed batch, accumulation, gradient-checkpointing, memory-budget, and prefetch settings when resuming, while retaining learned weights, optimizer/scheduler state, and frozen data. With the CLI, specify both `--batch-size` and `--accumulation` to override a resumed run; omit both to keep its settings. `--checkpointing` / `--no-checkpointing` controls the memory/computation tradeoff; `--gpu-memory-fraction` defaults to 0.68, and `--no-prefetch` disables background batch preparation. A changed batch grouping or a switch from the legacy sampler need not reproduce the exact numerical trajectory of the old run. The new prefetch sampler uses independent per-update seeds, so its pending work can be recreated on resume without disturbing model randomness.

`best.pt` tracks validation loss; `last.pt` tracks the latest completed update. `metrics.jsonl` retains losses, validation, learning rate, gradient norm, elapsed time, memory, seconds per update, data preparation time, and examples per second. Samples are written every 1,000 updates; these are explicitly marked short previews using reference timing. Early checkpoints may not produce exportable samples.

Checkpoint files use PyTorch serialization with optimizer/RNG state. Load your own checkpoints or ones you trust. A snapshot also verifies source map hashes when maps are loaded; prepare a new dataset if source content changes.

### 4. Generate

Select a trained checkpoint, audio files or a folder, target stars, style, seed, and variant count. **Custom** allows independent style levels; Auto leaves the decision to the learned model. AR/OD/CS/HP default to statistics from nearby training difficulties unless overridden.

Automatic timing predicts beat/downbeat activations and fits a piecewise tempo path. Provide BPM/offset or an existing timing map when it needs correction. Low confidence and fallback timing are recorded clearly. Songs with drifting tempo, sparse percussion, or timing unlike the corpus can require correction.

Generation uses cached attention and typed event masks. It carries prior patterns and active objects across windows, handles digital silence, and continues dense sections instead of dropping their remaining time. It preserves millisecond timestamps, quantizes positions to two pixels, and derives slider length consistently from duration, tempo, repeats, and SV. The constrained decoder avoids overlapping new notes during generated sliders/spinners.

Three candidates per variant are evaluated by default. The closest valid candidate to the target is exported; the application reports **measured stars**, and flags deviations above 0.5. Style scores and stars are checks, not guarantees of enjoyable play. Failed candidates and runs are reported rather than replaced with heuristic fake maps.

Each result includes a `.osz`, an unpacked map/audio folder, and `generation.json` with seed, checkpoint identity, actual stars, settings, and timing confidence. MP3/OGG is copied unchanged; WAV/FLAC is converted to OGG for osu!stable. Double-click an `.osz` to import it into osu!, then inspect and playtest it.

## Commands

After activating `.venv`, the `osumapper` command and `python -m osumapper.cli` expose the same backend as the UI:

```powershell
osumapper prepare 'C:\Users\you\AppData\Local\osu!\Songs' --data local-data/library
osumapper benchmark --run local-data/checks --preset 6gb
osumapper train --data local-data/library --run local-data/overfit --preset tiny --overfit 2 --steps 2000 --batch-size 1 --accumulation 1 --warmup 30
osumapper train --data local-data/library --run local-data/main --preset 6gb --hours 72
osumapper train --data local-data/library --run local-data/main --resume local-data/main/last.pt --hours 144
osumapper benchmark --run local-data/batch-check --batch-size 8 --no-checkpointing --gpu-memory-fraction 0.68
osumapper train --data local-data/library --run local-data/main --resume local-data/main/last.pt --batch-size 8 --accumulation 4 --no-checkpointing --gpu-memory-fraction 0.68
osumapper generate local-data/main/best.pt 'C:\Music\song.mp3' --output local-data/generated --stars 5.5 --preset Aim
osumapper generate local-data/main/best.pt 'C:\Music' --output local-data/generated --preset Custom --aim 0 --streams 2 --rhythm 1 --variants 2
osumapper evaluate local-data/main/best.pt --run local-data/main --timing
osumapper evaluate local-data/main/best.pt --run local-data/main --panel 4
osumapper ui --home local-data
```

`evaluate --panel 4` generates Auto/Aim/Streams/Complex Rhythm versions of four held-out songs spanning available stars. It records star error, timing confidence, style metrics, and blank human timing/flow/fun ratings. Fill those only after real playtesting. A normal loss evaluation uses teacher forcing and does not substitute for this panel.

## Validation and known limits

Run `python -m pytest -q`. Tests cover native curve/timing round trips, audio impulse alignment at three sample rates, song leakage, frozen labels, dense-window coverage, grammar boundaries, cached-decoder equivalence, causal masking, independent timing prediction, exports, and exact interrupted/resumed CPU training.

This is a compact first implementation, not a rankable-map quality guarantee. The current corpus determines styles and difficulty coverage. Full-song structure is learned indirectly through local audio context and preceding patterns; there is no separate song-level planner. Rhythm/style statistics are approximations. The labeling plot approximates Catmull paths; exported paths retain the actual native type. Optional artwork and custom audio samples are not generated.

The application has no cloud publishing or automatic submission to osu!. Generation quality on new songs must be measured after training. See `VERIFICATION.md` for the checks actually completed on this machine.

## References

- [osu! file format](https://osu.ppy.sh/wiki/en/Client/File_formats/osu_(file_format))
- [PyTorch AMP recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
- [rosu-pp-py](https://github.com/MaxOhn/rosu-pp-py), pinned to 3.1.0 with stable difficulty settings
- [Mapperatorinator](https://github.com/OliBomby/Mapperatorinator), related research inspiration; no weights or implementation are copied
