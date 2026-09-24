# Your first osu! AI training run

The application is installed on this computer. Your library has already been prepared, so you can begin by reviewing labels. The model still needs substantial training before its maps should be judged for fun or consistency.

## 1. Open the app

Double-click **Start.cmd** in this project folder. If it is already running, open:

**http://127.0.0.1:7860**

The four tabs are **Library → Labeling → Training → Generate**. The background-job panel at the bottom shows progress and has a Cancel button.

The program and prepared data are separate. Keep both this project folder and its associated `work` folder. On another PC, use `Install.ps1` first; Python 3.12 is recommended. There is no need to install a separate CUDA Toolkit.

## 2. Check your prepared library

On **Library**, click **Refresh library report**. The existing dataset contains:

| What | Count |
|---|---:|
| Usable standard difficulties | 6,204 |
| Distinct audio recordings | 1,307 |
| Song groups, including alternate versions | 947 |
| Training difficulties | 5,732 |
| Validation difficulties | 204 |
| Test difficulties | 268 |

You do **not** need to prepare these again. Most of your data is around 1–9 stars; there are far fewer examples above 10 stars. Expect the model to learn the well-represented ranges more reliably.

The original library contained 7,009 files across all game modes. Preparation rejected 784 unsupported/invalid files and removed 21 exact duplicate map/audio pairs. Details are in `rejected.json` inside the prepared-library folder. Most exclusions are other game modes; 110 maps exceed the first version's slider-anchor limit.

If you add more maps later, run **Prepare library** again using the same prepared-library folder. Existing audio caches and human labels are reused. Do not prepare or modify source maps during an active training run.

## 3. Review style labels

Open **Labeling** and click **Load / refresh**.

- The audio player previews 30 seconds of the song.
- The diagram shows the next four seconds of object patterns.
- Change **Preview start (seconds)** and click **Preview here** to inspect a different section.
- Labels belong to one difficulty, not the entire song.

Use the three independent controls:

| Control | Low | High |
|---|---|---|
| Aim intensity | Less emphasis on cursor movement | Large/fast jumps or demanding cursor movement |
| Streams / tapping | Mostly isolated notes or slower tapping | Frequent bursts, streams, or tapping pressure |
| Rhythm complexity | Regular, predictable rhythms | Varied intervals, syncopation, more complex rhythms |

**Medium** is the middle ground. **Auto / unknown** means you are not assigning that characteristic; it does not mean Low.

The preselected labels are statistical suggestions. Listen and inspect before accepting them. **Accept / save labels** saves your corrections. **Skip / next** moves on without accepting them. **Exclude map** removes that difficulty from future training runs.

For your first pass, review a representative selection—aim maps, stream maps, unusual rhythms, and several difficulties. You do not need to label all 6,204 maps before experimenting; unreviewed maps use their suggested labels. A few hundred thoughtful labels are a more practical starting task than rushing the whole library. This is a workflow suggestion, not a guaranteed quality threshold.

Save corrections **before starting a new run**. Each run freezes its dataset and labels. Later edits do not change that run when you resume it.

## 4. Choose your first training run

The installation has already passed a CUDA memory test and a two-map memorization test. You do not need to repeat those unless you change hardware or dependencies.

On **Training**:

1. Set **Training run folder** to a new folder. For example, replace the final `main` in the displayed path with `my-first-run`.
2. Choose **6gb** for the Quadro RTX 3000, or **8gb** for the RTX 3050.
3. Leave **Small overfit test** unchecked for the real model.
4. Set your total training budget. **72 hours** is the planned first experiment; you can cancel earlier and resume later.
5. Leave **Resume checkpoint** empty for this new run.
6. In **Training speed and GPU memory**, use **Batch size 8**, **Gradient accumulation 4**, **gradient checkpointing off**, **memory budget 68%**, and **Prepare the next batch on** for this Quadro. On different hardware, run **Check GPU memory** with the chosen settings first. If necessary, use batch 4 / accumulation 8 and turn gradient checkpointing on.
7. Click **Start / resume training**.

The first stage says **Indexing training windows and loading model**. On this library it takes a few minutes before the GPU starts training. That is normal. The dataset produces about **102,204 training windows**.

The full model has **32.4 million parameters**. It starts from random weights. No pretrained mapper or audio model is being downloaded.

The small overfit checkbox instead trains a roughly one-million-parameter diagnostic on two maps. It is useful for testing the pipeline, not for creating your final model. Keep overfit tests in their own run folder.

### Existing checkpoint from setup

An initial full-model experiment was started during implementation and then stopped at your request. Its run is the existing `main` folder. You may resume its `last.pt` if you want, but its labels were frozen **before your manual review**.

For training with your corrected labels, use the new run folder described above. Do not overwrite or reuse `main` as a fresh run.

## 5. Understand the progress

The background log prints a summary every ten optimizer updates. Click **Refresh training metrics** to see the latest detailed status.

The **Training progress** indicator updates while the app is open. It shows progress toward the earlier limit: **100,000 optimizer updates** or **72 accumulated training hours**. The text below it shows the update count, elapsed time, recent speed, and estimated remaining time. The estimate changes with training speed and checkpoints. At 100%, the run has ended; the job and run status say whether it completed, reached the time limit, was cancelled, or failed. Reaching 100% means the run reached its limit, not that its maps are guaranteed to be good. Check held-out results and playtest the generated maps.

- **Step:** completed optimizer updates. With batch 8 / accumulation 4, each update combines four batches of eight examples. Batch 1 / accumulation 32 processes the same number one at a time.
- **Loss / token loss:** how well the model predicts beatmap events. A downward trend is encouraging; it is not a direct fun score.
- **Beat loss:** performance on beat/downbeat supervision.
- **Validation loss:** performance on songs excluded from training. Compare it with training loss over time.
- **Peak VRAM:** actual model allocation measured by PyTorch. Desktop applications and driver memory add to the total reported by Windows.

The original batch-1 run took roughly **8 seconds per update**. The selected batch-8 configuration takes about **1.58 seconds per 32 examples** in the maximum-length GPU benchmark, using **3.43 GiB peak PyTorch allocation / 3.66 GiB reserved**. Actual training also loads and prepares examples, so use **seconds_per_update** and **examples_per_second** in the live metrics for real speed. These are observations on this Quadro, not guarantees for another computer. The RTX 3050 preset has not been checked on that physical GPU.

Keep the **68% memory budget**: it limits tensor/cache memory to about 4.08 GiB on the 6 GiB card, leaving room for CUDA and the desktop. Training uses fixed buffer sizes and prepares the next batch in the background. An earlier batch-32 trial grew into shared system memory; it was stopped and replaced. In Windows Task Manager, distinguish **Dedicated GPU memory** from **Shared GPU memory**. Shared-memory growth into multiple gigabytes is not useful VRAM utilization.

With the corrected settings, the observed real run takes approximately **1.8 seconds per update**, about **4.3 times faster** than the original configuration. A one-minute monitor showed **4.48–4.64 GiB total dedicated GPU usage**, with at least **1.17 GiB free**. GPU utilization averaged about 60%; use measured training speed and stable dedicated-memory usage, rather than a constant 100% indicator, to judge this setup.

The program attempts a short preview export every 1,000 updates; samples appear under the run's `samples` folder. Early checkpoints may not produce a valid sample yet.

Preview maps deliberately use reference timing and cover only a short excerpt. They test whether the mapper is learning patterns. Use **Generate** later to assess whole-song mapping and automatic timing.

There is no universal loss value at which maps become good. Check validation trends and periodically listen/playtest generated output. If training improves while validation repeatedly worsens, the model may be memorizing the corpus rather than improving on new songs.

## 6. Stop and resume safely

Click **Cancel current job**, then wait until the job shows **cancelled** and confirms that the checkpoint was saved. During startup indexing, cancellation may wait for indexing to finish.

The run contains:

| File | Purpose |
|---|---|
| `last.pt` | Latest saved training state; use this to resume |
| `best.pt` | Checkpoint with the best measured validation loss |
| `dataset.json` | Frozen map list, labels, splits, and dataset identity |
| `config.json` | Model and training settings |
| `metrics.jsonl` | Progress history |
| `samples/` | Periodic diagnostic preview maps |

To resume:

1. Keep **Training run folder** on the same run.
2. Set **Resume checkpoint** to that run's `last.pt`.
3. Leave the overfit checkbox off for the full model.
4. Check the speed settings: **batch 8 / accumulation 4**, **gradient checkpointing off**, **68% memory budget**, **next-batch preparation on**. They also apply when resuming. Weights, optimizer state, and dataset are retained.
5. Click **Start / resume training**.

The hours setting is a **total accumulated budget**. If you used 10 of 72 hours, resuming with 72 allows about 62 more. Set 144 to extend the total budget to 144 hours.

You may close the browser tab without stopping the worker. The computer must remain awake. Before shutdown or reboot, cancel and wait for the checkpoint. An unexpected stop can lose work since the last save; the interface marks a dead worker as interrupted so you can resume it.

Do not launch two separate training commands against the same run folder. The UI permits only one background job at a time. Pause training before generating or running GPU checks through the interface.

## 7. Generate your first new maps

After the model has had meaningful training, cancel safely and open **Generate**.

1. Set **Trained checkpoint** to the run's `best.pt`.
2. Add one song first. MP3, OGG, WAV, and FLAC are supported.
3. Pick a star target well represented in the training data—around 3–6 stars is a sensible first check for this library.
4. Choose a style:
   - **Auto:** the learned model chooses the pattern emphasis.
   - **Aim:** high aim with low stream emphasis.
   - **Streams:** high tapping with low aim emphasis.
   - **Complex Rhythm:** high rhythmic complexity; other controls remain automatic.
   - **Custom:** independently set aim, streams, and rhythmic complexity.
5. Leave **Candidates per variant** at 3. The app chooses the valid result nearest your target stars.
6. Keep **Variants per song** at 1 for the first check.
7. Click **Generate maps**, then **Refresh generated files** when it finishes.

Double-click the resulting `.osz` to import it into osu!stable. The result is local; it is not submitted to osu! or ranked.

The displayed difficulty is the **measured** star rating. Requested stars are a conditioning target, not an exact guarantee. A miss greater than 0.5 stars is reported. The pinned difficulty calculator may differ from the version in your client.

The same seed/settings/checkpoint make comparisons more repeatable. Change the seed for another variation. Once one-song generation works to your satisfaction, use a song folder and increase the variant count.

## 8. Fix timing when needed

Listen in the editor before judging the patterns. Automatic timing is learned, and it can be wrong even when its confidence is high.

If notes drift against the music or the beat grid is wrong, open **Timing and difficulty overrides**:

- Supply **BPM** and **Offset in milliseconds** for a song with one steady tempo.
- Supply a reference **.osu timing file** for changing tempo or more precise timing. It must correspond to the same audio timeline.
- AR/OD/CS/HP overrides change those settings; they do not directly set star rating.

Inspect the result's `generation.json` for timing confidence, fallback warnings, actual stars, seed, and checkpoint identity. Regenerate after correcting timing. A high difficulty match by itself does not establish correct musical timing.

## 9. Evaluate usefulness, not just successful export

Start with several unfamiliar songs and compare styles using the same star target and seed. Ask:

- Are notes aligned with the musical events you hear?
- Are jumps readable and comfortable at the requested difficulty?
- Do streams start and end naturally?
- Do sliders flow into the next object?
- Are quieter sections and breaks sensible?
- Do Aim, Streams, and Complex Rhythm produce noticeably different maps?
- Would you voluntarily play the map again?

The Training tab can run held-out loss evaluation. The command-line panel in README generates four styles on held-out songs and leaves space for your timing/flow/fun ratings. A program cannot fill those ratings honestly without playtesting.

## Troubleshooting

| Symptom | What to do |
|---|---|
| It stays on indexing at startup | Allow a few minutes for the full library; this stage is CPU/file work. |
| The job says interrupted | The old worker is gone. Resume `last.pt` in the same run folder. |
| CUDA is unavailable | Use the project's launcher/environment; reinstall with `Install.ps1` if necessary and check the NVIDIA driver. |
| Out of GPU memory | Close GPU-heavy apps. Resume `last.pt` with batch 4 / accumulation 8 and gradient checkpointing enabled. Keep the 68% memory budget. Run the memory check before resuming. |
| Total GPU memory exceeds dedicated VRAM | Check Shared GPU memory. Cancel safely, keep the memory budget at 68%, and use the latest launcher/code with fixed batch buffers. Avoid increasing the cap to fill the card. |
| No objects or invalid early candidates | The checkpoint may be too early. Continue training and check validation/sample progress. |
| Wrong rhythm or drift | Inspect BPM/offset and try a timing reference matching the audio. |
| My new labels did not affect resumed training | Expected: the run freezes labels. Start a new run after corrections. |
| My run already has a checkpoint | Resume it, or choose a different run folder for fresh training. |
| The interface says a job is already running | Wait or cancel the current job. Refresh the page if you reopened the app. |

For guidance later, share the run's step count, recent training/validation losses, a description of a generated map's problems, and the settings you used. That is enough to decide whether to keep training, improve labels/data, or correct timing first.
