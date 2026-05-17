# Theory Judge + TTS Vocals Design

## Summary

Replace Qwen2.5-Omni-7B RLAIF judge with a deterministic, modular music theory judge.
Add TTS vocal rendering via Bark. Keep soft scale rewards (no hard quantization).

## Architecture

Three subsystems:

1. **Theory Judge** (`src/reward/theory_judge.py`) — 8 pure-function reward modules operating on raw frequency/duration/velocity arrays. No audio rendering needed for scoring.
2. **Vocal Renderer** (`src/render/vocals.py`) — Bark TTS integration for melody vocals, mixed with instrumental.
3. **Training loop** — New `--judge theory` flag in `rlaif_train.py`. Reward = weighted sum of theory modules.

## Theory Judge Modules

| Module | Input | Signal |
|--------|-------|--------|
| `key_adherence` | freqs, root_hz, scale | Deviation from nearest scale degree (cents) |
| `interval_quality` | freqs | Consonance class of consecutive intervals |
| `melodic_contour` | freqs | Stepwise motion, gap-fill, climax placement |
| `cadence_detection` | freqs, durations | V-I resolution at phrase boundaries |
| `voice_leading` | voices (multi-voice) | Parallel 5ths/8ves penalty, contrary motion |
| `rhythm_analysis` | durations | Tempo grid coherence, pattern repetition |
| `tension_resolution` | freqs, durations | Tension arc shape (build + resolve) |
| `dynamic_shaping` | velocities | Phrasing arcs, metric accents, dynamic range |

Composite: `reward = Σ(weight_i * module_i(...))`

Default weights: key_adherence=3.0, interval_quality=2.0, melodic_contour=1.5, cadence=1.0, voice_leading=1.0, rhythm=1.0, tension=0.8, dynamics=0.5

## Key Decisions

- Theory judge scores **structure** (freqs/durs/vels), not rendered audio
- Soft scale rewards via key_adherence penalty, not hard quantization
- TTS vocals are render-only — theory judge doesn't score the vocal audio
- Bark chosen for TTS: local inference, singing support, Apache 2.0, MPS compatible
- All modules are pure functions with no state — deterministic, testable, fast

## Integration

- New CLI: `--judge theory` replaces `--judge qwen`
- Physics rewards from psychoacoustic.py remain available as separate weight
- Adapters gain `vocals=True` flag for TTS rendering on eval checkpoints
