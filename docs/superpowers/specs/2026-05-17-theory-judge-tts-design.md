# Theory Judge + TTS Vocals Design

## Summary

Replace Qwen2.5-Omni-7B RLAIF judge with a deterministic, modular music theory judge.
Add TTS vocal rendering via Bark. Use scale snap quantization and per-note credit
assignment for effective REINFORCE training.

## Architecture

Three subsystems:

1. **Theory Judge** (`src/reward/theory_judge.py`, ~750 lines) — 8 pure-function reward modules operating on raw frequency/duration/velocity arrays. No audio rendering needed for scoring. Includes per-note credit assignment for REINFORCE.
2. **Vocal Renderer** (`src/render/vocals.py`) — Bark TTS integration for melody vocals, mixed with instrumental. PyTorch 2.6 compatibility fix via monkey-patching `torch.load`.
3. **Training loop** — `--judge theory` flag in `rlaif_train.py`. Blended loss: 30% scalar REINFORCE + 70% per-note REINFORCE. Scale snap and freq std clamp flags.

## Theory Judge Modules

| Module | Input | Signal |
|--------|-------|--------|
| `key_adherence` | freqs, root_hz, scale | Exp-decay distance to nearest scale degree |
| `interval_quality` | freqs | Consonance class of consecutive intervals |
| `melodic_contour` | freqs | Stepwise motion, gap-fill, climax, motif repetition |
| `cadence_detection` | freqs, durations | V-I/IV-I/vii-I patterns at phrase boundaries |
| `voice_leading` | voices (multi-voice) | Parallel 5ths/8ves penalty, contrary motion |
| `rhythm_analysis` | durations | Grid alignment, syncopation, pattern repetition |
| `tension_resolution` | freqs, durations | Tension arc (build in first 70%, resolve in last 30%) |
| `dynamic_shaping` | velocities | Phrasing arcs, metric accents, dynamic range |

Composite: `reward = sum(weight_i * module_i(...))`

Default weights: key_adherence=3.0, interval_quality=2.0, melodic_contour=1.5, cadence=1.0, voice_leading=1.0, rhythm=1.0, tension=0.8, dynamics=0.5

## Per-Note Credit Assignment

`theory_reward_per_note()` returns shape (N,) combining seven per-note signals with weights:

| Signal | Weight | Description |
|--------|--------|-------------|
| Key adherence | 5.0 | Exp-decay distance to nearest scale degree |
| Interval consonance | 2.0 | Consonance of adjacent intervals |
| Stepwise motion | 1.5 | Reward steps, penalize large leaps |
| Rhythm grid | 1.0 | Alignment to common duration ratios |
| Cadence position | 1.0 | Tonic at phrase ends, dominant before tonic |
| Tension phase | 0.8 | Dissonance in build, consonance in resolve |
| Dynamics arch | 0.5 | Proximity to ideal sin-arch velocity |

Training blends `0.3 * scalar_loss + 0.7 * per_note_loss` where per_note_loss uses
the generator's per-note log-probs multiplied by per-note advantages.

## Scale Snap Quantization

Hard snap via `_snap_to_scale()` using argmin (not softmax). Configurable:

- `--scale-snap 0.8` — interpolate 80% toward nearest scale tone after sampling
- `--scale-key Bb_blues` — any of 17 roots (C through B, all sharps/flats) x 9 scales
- Applied in generator's `sample()` after Gaussian sampling, before return

ROOTS dict in `theory_judge.py` covers all 12 chromatic pitches (with enharmonic equivalents).
SCALES dict covers: major, natural_minor, harmonic_minor, melodic_minor, pentatonic_major, pentatonic_minor, dorian, mixolydian, blues.

## Freq Std Clamp

`--freq-std-clamp -2.0` limits the Gaussian policy's log-std to max -2.0, reducing sampling
noise from 18 semitones to 2.4 semitones. This lets REINFORCE learn meaningful pitch means
instead of relying on random exploration.

## Key Decisions

- Theory judge scores **structure** (freqs/durs/vels), not rendered audio — 1000x faster than Qwen
- Scale snap is hard quantization (argmin), not soft (softmax temperature was producing out-of-key midpoints)
- Per-note credit assignment gives REINFORCE gradient signal for positional patterns (cadences, tension arcs)
- Bark TTS uses monkey-patched `torch.load` with `weights_only=False` for PyTorch 2.6 compatibility
- All modules are pure functions with no state — deterministic, testable, fast

## Training Results

Best run (500 steps, batch=16, lr=1e-4, scale_snap=0.8, freq_std_clamp=-2.0):

- Mean theory score: 5.9 (up from 4.4 with Qwen judge)
- Key adherence: 0.78 (up from random ~0.50)
- Best total reward: 6.57
- Theoretical max for hand-crafted melody: 7.15

Bottlenecks (architecture-limited, not judge-limited):
- Cadence detection: 0.27 (feedforward can't learn position-dependent patterns)
- Dynamic shaping: 0.31 (velocity head lacks temporal structure)

## Integration

- CLI: `--judge theory --freq-std-clamp -2.0 --scale-snap 0.8 --scale-key A_natural_minor`
- Render: `scripts/render_theory_samples.py` with matching flags
- Vocal adapter: `melody_v8_vocal` renders through Bark + instrumental mix
- Physics rewards from psychoacoustic.py remain as separate `--physics-weight`

## Test Coverage

135+ tests across `test_theory_judge.py` (61) and `test_vocals.py` (19), plus existing test files.
