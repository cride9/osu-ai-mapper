# Introducing Lyra

**Lyra-1.0-Flash** and **Lyra-1.0** are two planned local models for generating osu!standard beatmaps from music. This repository contains the V2 preparation, training, evaluation, and generation code. The model names describe architecture candidates; they do not yet identify released, fully trained checkpoints.

Creating a good rhythm-game map takes more than finding a BPM. A mapper chooses which sounds to emphasize, when patterns should become dense or sparse, and how individual objects form a playable sequence. Lyra is an experiment in learning those decisions from human-created beatmaps.

The system trains a dedicated audio encoder on rhythm targets and masked audio reconstruction. A separate autoregressive mapper then uses the learned audio features, recent objects, and longer-range song context to generate beatmap events. At inference time, **both trained components are needed**.

## Two model sizes

| Candidate | Project configuration | Audio encoder | Mapper | Combined parameters |
| --- | --- | ---: | ---: | ---: |
| **Lyra-1.0-Flash** | V2-S | 39,445,572 | 40,923,456 | **80,369,028** |
| **Lyra-1.0** | V2-L | 39,445,572 | 77,903,040 | **117,348,612** |

The candidates share the same encoder architecture and prepared dataset. The larger mapper has more capacity, but improved quality or speed is **not yet established**; the two need comparable training and evaluation. Both begin with randomly initialized weights. The audio encoder is trained first and frozen for mapper training.

## Data and learning

The **prepared V2 corpus** contains **148,556 usable osu!standard difficulties**, grouped into **29,385 song groups** and **34,132 unique decoded recordings** with approximately **1,799.94 hours of unique audio**. These numbers describe validated data available to the pipeline. They do **not** mean that either candidate has completed training on every map.

| Split by song group | Beatmaps |
| --- | ---: |
| Train | 133,587 |
| Validation | 7,743 |
| Test | 7,226 |

The collection spans many difficulty levels and mapping styles. Human timing points supervise beat and downbeat prediction; masked mel reconstruction provides an auxiliary audio-learning task. The mapper learns a versioned event vocabulary for circles, native sliders and repeats, spinners, timing and slider-velocity changes, combos, breaks, coordinates, and hitsounds. It works section by section using local audio, a full-song summary, and object history.

This learned approach can represent mapping decisions beyond a fixed beat grid, including different beat subdivisions and emphasis. Extreme tempo, unusual timing, long-range coherence, and playability remain evaluation challenges. Difficult examples stay in the usable dataset; they are not automatically discarded because their loss is high.

## What the project provides

- Local preparation from the selected SQLite dataset rows and downloaded `.osu`/audio or `.osz` files, with validation and a report of rejected entries.
- Separate audio-encoder and mapper training, resumable checkpoints, and a local Gradio interface.
- Beatmap generation with timing correction, star and style controls, validation, and `.osu`/`.osz` export.
- Evaluation commands for held-out timing and mapping, plus model-card drafts for [Flash](model-cards/Lyra-1.0-Flash/README.md) and [the larger candidate](model-cards/Lyra-1.0/README.md).

Generated maps are intended for experimentation and mapper-assisted creation. Human review and osu! playtesting remain necessary. A valid export or low validation loss alone does not establish that a map is fun.

## Run locally

On Windows, run `Install.ps1` to create the project environment, then `StartV2.cmd` to open the local interface at <http://127.0.0.1:7861>. The UI provides Library, Training, and Generate workflows. You supply your own selected dataset and audio files; they are not included in this repository. Training does not start automatically.

For command-line use, the equivalent entry point is `python -m osumapper.v2` (or the installed `osumapper-v2` command). For example:

```powershell
python -m osumapper.v2 prepare --source-db C:\path\to\osu_dataset.sqlite --files-root C:\path\to\osu_files --data local-data-v2\dataset-full --version v2
python -m osumapper.v2 train --stage audio --model v2-s --data local-data-v2\dataset-full --run local-data-v2\runs\audio
python -m osumapper.v2 features local-data-v2\runs\audio\best.pt --data local-data-v2\dataset-full
python -m osumapper.v2 train --stage mapper --model v2-s --data local-data-v2\dataset-full --run local-data-v2\runs\mapper
python -m osumapper.v2 generate local-data-v2\runs\mapper\best.pt C:\path\to\song.mp3 --output local-data-v2\generated --stars 5
```

The mapper must be trained against the matching audio encoder and dataset version. Use a run's `last.pt` to resume training and its evaluated `best.pt` for generation. The local generation path also expects matching manifests and style data, so uploading only weight files would not yet provide standalone inference on Hugging Face.

## Release status

The repository has code and an examined prepared corpus. **No completed full-dataset Lyra-1.0-Flash or Lyra-1.0 checkpoint, held-out quality result, or osu! playtest result is asserted here.** The [model-card drafts](model-cards/README.md) list the evidence and artifacts to add before publishing either model. Audio and beatmap rights must be checked separately before sharing datasets or examples.

The earlier V1 mapper remains available in this repository. Lyra refers to the V2 candidates; V1 results should not be presented as V2 evaluation.

Run `python -m pytest -q` for the automated tests.
