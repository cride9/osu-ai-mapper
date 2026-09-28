---
model_name: Lyra-1.0-Flash
tags:
  - osu
  - beatmap-generation
  - music-conditioned-generation
  - pytorch
---

# Lyra-1.0-Flash

**Status: model card draft.** The architecture and prepared dataset described below are verified in the local project. A completed full-dataset Lyra-1.0-Flash training run, its weights, and held-out/playtest results have not yet been provided for release. Replace this notice and add measured results only after checking the actual checkpoint.

Lyra-1.0-Flash is the smaller V2 candidate for generating local osu!standard beatmaps from audio. It consists of a separately trained audio encoder and an autoregressive mapper. Both components start from random weights. The trained encoder is frozen while the mapper is trained; inference uses both together.

| Item | Specification |
| --- | --- |
| Audio encoder | 39,445,572 parameters; shared with the larger candidate |
| Mapper | 40,923,456 parameters |
| Combined system | **80,369,028 parameters** |
| Audio input | Mono 22.05 kHz; 128-bin log-mel features at approximately 10 ms spacing |
| Mapper | Width 512, 8 decoder layers, 8 query heads / 2 key-value heads, SwiGLU feed-forward width 1408 |
| Output | osu!standard `.osu` and local `.osz` packages through the project backend |

The encoder has a residual convolutional frontend and six Conformer layers. The mapper uses local audio context, a full-song summary, recent object history, and a section planner. It generates one section at a time while carrying prior objects forward. The event tokenizer represents circles, native sliders, repeats, spinners, timing and slider-velocity changes, breaks, combos, positions, and hitsounds. Generation also validates object order and geometry. Star and style controls are available in the local backend.

## Prepared dataset

The **prepared V2 corpus**, before training-set selection, contains 148,556 validated osu!standard difficulties from 34,132 unique decoded recordings in 29,385 song groups. Unique recordings total approximately 1,799.94 audio hours. These are preparation results, **not evidence that a released checkpoint has trained on every map**. The 150,823 selected SQLite rows were examined; 2,267 were rejected. Downloaded `.osz` members were read without requiring extraction.

| Split | Beatmaps |
| --- | ---: |
| Train | 133,587 |
| Validation | 7,743 |
| Test | 7,226 |

Splits are grouped using decoded audio identity, an audio fingerprint, beatmapset identity, and normalized song metadata. Validation and test recordings are excluded from audio pretraining. Source maps and audio are not included in this model card. The prepared collection has broad difficulty coverage, but high-star ranges are sparse; the presence of unusual rhythmic examples does not establish that the model handles them well.

## Training objectives

The audio stage learns beat/downbeat predictions from beatmap timing labels and reconstructs masked mel features:

`audio_loss = beat_loss + 0.1 × reconstruction_loss`

Conflicting timing labels from difficulties sharing a recording are masked. The mapper is trained by teacher forcing on next event tokens, with section-planning supervision and an auxiliary cost for out-of-bounds circle and slider-head coordinates:

`mapper_loss = token_loss + 0.1 × plan_loss + 0.05 × boundary_loss`

The displayed loss is a training diagnostic. It does not measure musical quality or playability. Prepared-data and feature caches can be bounded and rotated without removing eligible maps, but a particular time-limited run may stop before completing a full dataset cycle. Record completed cycles and checkpoint dataset identity when reporting final training.

## Intended use and limitations

Intended uses are local beatmap prototyping, mapper-assisted editing, and research on music-conditioned level generation. Generated maps should be reviewed and playtested before sharing. Rhythm alignment, difficulty consistency, pattern variety, full-song flow, and automatic timing on unseen songs need empirical evaluation. Unusual timing, tempo changes, syncopation, sparse sections, and dense patterns may remain difficult. Grammar and export validation reduce malformed output; they do not guarantee an enjoyable map.

The project does not yet provide a portable, self-contained Hugging Face inference package. Its current generation command expects a mapper checkpoint together with its dataset/feature manifests, the matching audio-encoder checkpoint, and access to the prepared training style data. Those paths must be packaged or adapted before publishing downloadable weights. No full-dataset validation scores, timing metrics, or human playtest results are claimed here.
