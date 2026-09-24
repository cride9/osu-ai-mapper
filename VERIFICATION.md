# Verification on this computer

Checked with Python 3.12.14, PyTorch 2.8.0+cu126, and the Quadro RTX 3000 6 GB. The RTX 3050 was not connected and has not been tested directly.

## Completed checks

- **26 automated tests passed**, including parser/writer/tokenizer round trips, repeated slider anchors, tempo/SV changes, audio impulse alignment at 22.05/44.1/48 kHz, song split isolation, frozen labels, dense-section coverage, constrained decoding, cached attention, independent beat supervision, export, exact checkpoint resume on CPU with prefetching, batch changes on resume without resetting optimizer progress, isolated/reproducible prefetch randomness, transient Windows status-file locks, and interrupted-job reporting.
- Static checks found no undefined/unused references in the application modules. The installed environment passes `pip check`.
- The **32,361,986-parameter** model completed successful CUDA forward/backward/AdamW updates at a **1,535-token decoder length**. The synthetic benchmark measured about **0.536 GiB peak allocated** and **0.668 GiB peak reserved** GPU memory.
- The real-data full-model run reached 122 updates before its worker stopped. Its observed peak allocation was about **0.660 GiB**. Loss decreased from **8.699** at the initial step to **5.290** at step 122. These are early learning checks, not evidence of mapping quality.
- A separate **1.052M-parameter diagnostic** memorized two real-map clips over **2,000 updates**, reaching approximately **0.087 diagnostic loss**. This loss is on the small overfit examples, not held-out songs.
- Diagnostic sample `.osz` packages were written at steps 1,000 and 2,000.
- A held-out **79.7-second song** completed automatic timing estimation, full-song generation with **three candidates**, difficulty ranking, and `.osz` export. The selected diagnostic output measured **3.968 stars** against a 4.0 target. This single result does not establish difficulty calibration or musical quality.
- The Gradio interface was opened in the browser. Real-song audio previews, pattern plotting, automatic label values, and the tab layout were checked visually.

## Prepared dataset

| Item | Count |
|---|---:|
| Input map files, all modes | 7,009 |
| Accepted standard maps | 6,204 |
| Rejected unsupported/invalid files | 784 |
| Exact duplicate map/audio pairs removed | 21 |
| Distinct cached recordings | 1,307 |
| Song groups | 947 |
| Training difficulties | 5,732 |
| Validation difficulties | 204 |
| Test difficulties | 268 |
| Indexed training windows | 102,204 |

Most exclusions were other game modes. Another 110 maps exceeded the configured 64-anchor slider limit. Remaining exclusions and exact source paths are recorded in the prepared library's `rejected.json`.

## Initial training handoff

The original full-model run's most recent saved checkpoint was **step 115**; updates after that were not saved before the old worker stopped. It was successfully loaded for resume. At the user's request, the resumed worker was **cancelled before further optimizer updates**, and `main/last.pt` was saved again at step 115.

No training worker was left running at that initial handoff. The user subsequently resumed training. The existing `main` run uses the original suggested labels. Start a new run after manual label corrections if those corrections should be included.

## GPU tuning on September 24

The user requested higher GPU utilization. Training was safely cancelled and saved at step 168, with a separate `before-batch-tuning-step168.pt` backup. Resume now supports changing batch size, accumulation, and gradient checkpointing while retaining the checkpoint's learned state. Weighted sampling now precomputes cumulative weights instead of rebuilding them for each microbatch.

Maximum-length synthetic CUDA benchmark (1,535 decoder tokens, 12-second audio, full model, two successful warmup updates; eight measured updates except batch 8 with checkpointing, which used six):

| Batch | Gradient checkpointing | Seconds / 32 examples | Peak allocated GiB | Peak reserved GiB |
|---:|:---:|---:|---:|---:|
| 1 | on | 8.50 | 0.536 | 0.668 |
| 8 | on | 1.85 | 1.192 | 1.447 |
| 16 | on | 1.56 | 2.002 | 2.469 |
| 32 | on | 1.47 | 3.611 | 4.498 |
| 8 | off | 1.53 | 3.430 | 3.660 |

These initial synthetic checks did not reproduce allocator growth under varying real-data shapes. The first batch-32 trial averaged about 2.04 seconds/update, but its allocator reserve grew to 10.74 GiB and Windows spilled into shared RAM. It was stopped, and the learned state was saved at step 242. A transient Windows status-file read lock also occurred during cancellation; bounded retry now handles that case. The initial batch-32 choice was rejected.

The corrected implementation fixes CUDA batch shapes at 1,535 decoder tokens, caps tensor/cache allocations at 68% of VRAM (about 4.08 GiB), and prepares one CPU batch ahead with independent reproducible randomness. Batch 32 fails a maximum-length benchmark at this cap. The selected configuration is **batch 8 / accumulation 4, gradient checkpointing off, prefetch on**. It passed 22 successful synthetic updates (2 warmup + 20 measured), averaging **1.578 seconds per 32 examples**, with **3.430 GiB peak allocated / 3.660 GiB reserved**, and about **0.976 GiB free** at the end of the check.

Real training resumes from step 242. Desktop/driver memory is additional to PyTorch allocation. Raw measurements, rejected settings, and the monitored real-training results are in `PERFORMANCE.json`. The RTX 3050 has not been checked directly.

The corrected real-data run remained at **3.544 GiB peak allocated / 3.820 GiB reserved** over the observed updates. A 60-sample, roughly one-minute NVIDIA monitor measured total dedicated GPU usage of **4,592–4,752 MiB**, with **1,204–1,364 MiB free**. GPU utilization averaged **59.8%** (35–77%); it is not claimed to be a sustained 100%. Updates averaged approximately **1.8 seconds**, compared with **7.70 seconds** over 28 baseline intervals: about **4.3× faster**. Waiting for CPU data plus transfer averaged about **0.008 seconds/update**. The step-250 checkpoint was saved with the new settings; training continues in the background.

## Not claimed or completed

- A fully trained, generalizing mapper or consistently enjoyable output.
- Verified style adherence and star accuracy across the full held-out panel.
- Human playtesting or a confirmed visual import/play session in osu!stable. Exports were independently parsed and difficulty-calculated; actual game/editor inspection is the user's next quality gate.
- Direct testing of the RTX 3050 8 GB preset on that GPU.

The tutorial describes how to finish training and assess these remaining quality questions.
