---
model_name: Lyra-1.0
tags:
  - osu
  - beatmap-generation
  - music-conditioned-generation
  - pytorch
---

# Lyra-1.0

**Status: model card draft.** Lyra-1.0 is the planned larger V2 mapper candidate. Its architecture and prepared dataset are verified; no completed full-dataset Lyra-1.0 checkpoint or comparative evaluation is available here. Higher capacity is a design choice, **not a demonstrated quality improvement**. Replace this notice only after training and testing the released weights.

Lyra-1.0 combines the same separately trained audio encoder used by Lyra-1.0-Flash with a wider and deeper autoregressive beatmap mapper. Every learned component starts from random initialization. After audio training, the encoder is frozen for mapper training. Both components are needed to generate a beatmap.

| Item | Specification |
| --- | --- |
| Shared audio encoder | 39,445,572 parameters |
| Larger mapper | 77,903,040 parameters |
| Combined system | **117,348,612 parameters** |
| Audio input | Mono 22.05 kHz; 128-bin log-mel features at approximately 10 ms spacing |
| Mapper | Width 640, 12 decoder layers, 10 query heads / 2 key-value heads, SwiGLU feed-forward width 1728 |
| Output | osu!standard `.osu` and local `.osz` packages through the project backend |

The shared audio network uses a residual convolutional frontend and six Conformer layers. The mapper receives detailed local audio, a compressed full-song view, recent object history, and section-level planning features. It generates successive sections while retaining history. Its versioned event vocabulary covers native osu!standard object and timing structures. The backend offers star and style settings and checks complete generated maps before export.

## Prepared dataset

The **available prepared V2 corpus** contains 148,556 validated osu!standard difficulties, 34,132 unique decoded recordings, 29,385 song groups, and approximately 1,799.94 hours of unique audio. These are corpus statistics, not a claim that a Lyra-1.0 checkpoint has consumed all of them.

| Split | Beatmaps |
| --- | ---: |
| Train | 133,587 |
| Validation | 7,743 |
| Test | 7,226 |

The import examined all 150,823 SQLite-selected rows and rejected 2,267. Song grouping uses decoded audio identity, an audio fingerprint, beatmapset identity, and normalized song metadata; held-out recordings do not participate in audio pretraining. The prepared data spans many difficulty bands, although very high-star maps are scarce. Source songs and community beatmaps are not redistributed with this card.

## Training objectives

The encoder predicts beats and downbeats and reconstructs masked mel features:

`audio_loss = beat_loss + 0.1 × reconstruction_loss`

The mapper learns the next event token with teacher forcing, predicts section characteristics, and receives an auxiliary cost for out-of-bounds circle and slider-head coordinates:

`mapper_loss = token_loss + 0.1 × plan_loss + 0.05 × boundary_loss`

The larger model is a candidate to compare against Lyra-1.0-Flash on identical training time and held-out songs. No superiority claim follows from parameter count alone. A time-limited training run may not visit every prepared map even though the cache does not exclude it; final reporting should include dataset cycles and the exact checkpoint identity.

## Intended use and limitations

The intended use is local osu!standard map prototyping and research on audio-conditioned level generation. Generated maps need manual inspection and playtesting. The model may choose unusual rhythms, repetitive patterns, inconsistent difficulty, or weak long-range structure. Timing estimation for unseen tracks, tempo changes, syncopation, sparse music, and extreme density all require measurement. A syntactically valid `.osz` package is not proof of musical quality.

The current local generation code requires both mapper and audio-encoder checkpoints, matching dataset/feature manifests, and prepared training style data. It is not yet a self-contained Hugging Face inference package. Publish measured validation, timing, export-validity, and human playtest results with the actual released weights; none are asserted in this draft.
