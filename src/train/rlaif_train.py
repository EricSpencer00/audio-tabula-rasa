"""
RLAIF training loop. The reward signal is a weighted sum of:

  1. The Qwen2.5-Omni-7B music critic's 1-10 score (the AI feedback)
  2. The existing physics-based reward already used in the phase-2/3
     trainers (kept at a small weight so the generator doesn't drift
     into something the LLM "likes" but is dissonant nonsense).

Reward shape per sample:
    r = qwen_weight * (qwen_score - 5) / 5  +  phys_weight * physics_reward

REINFORCE update against this scalar reward, on top of the existing
Phase-2/3 weights (so we *fine-tune* with the LLM, we don't restart
from scratch).

CLI:
    python -m src.train.rlaif_train \
        --generator melody --judge qwen \
        --steps 200 --batch-size 4 \
        --out-dir results/rlaif/melody_qwen

    python -m src.train.rlaif_train \
        --generator chord_progression --judge qwen \
        --steps 100 --batch-size 4 \
        --out-dir results/rlaif/chord_qwen

Each step is `--batch-size` renders + `--batch-size` Qwen calls. The
expensive thing is the Qwen forward pass, so a step on M-series with
batch=4 takes roughly 60-90 s.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from src.generator.autoregressive_melody import AutoregressiveMelodyGenerator
from src.generator.chord_generator import ChordProgressionGenerator
from src.generator.counterpoint_generator import CounterpointGenerator
from src.generator.melody_generator import ExpressiveMelodyGenerator, MelodyGenerator
from src.render.instruments import PRESETS
from src.render.synth import SAMPLE_RATE as _SR, render_chord, render_melody, simple_reverb, snap_to_scale
from src.reward.counterpoint import counterpoint_reward
from src.reward.psychoacoustic import (
    expressive_melody_reward,
    melody_reward,
    progression_reward,
)
from src.reward.theory_judge import (
    theory_reward, theory_reward_breakdown, theory_reward_per_note,
)


SAMPLE_RATE = 44_100   # synth.SAMPLE_RATE


# ---------- generator adapters ----------------------------------------


@dataclass
class _Adapter:
    name: str
    build: Callable[[], torch.nn.Module]
    init_weights: str
    sample_to_audio: Callable[[np.ndarray], np.ndarray]
    physics_reward: Callable[[np.ndarray], float]
    output_format: str = "freqs"  # "freqs", "expressive", "voices"


def _melody_to_audio(freqs: np.ndarray) -> np.ndarray:
    return render_melody(freqs, note_duration=0.35, gap=0.03)


def _expressive_melody_to_audio(combined: np.ndarray) -> np.ndarray:
    n = len(combined) // 3
    freqs, durs, vels = combined[:n], combined[n:2*n], combined[2*n:]
    return render_melody(freqs, gap=0.03, durations=durs, velocities=vels,
                         use_fm=True)


def _quantized_melody_to_audio(combined: np.ndarray) -> np.ndarray:
    n = len(combined) // 3
    freqs, durs, vels = combined[:n], combined[n:2*n], combined[2*n:]
    freqs_q = np.array([snap_to_scale(f) for f in freqs])
    return render_melody(freqs_q, gap=0.03, durations=durs, velocities=vels,
                         use_fm=True)


def _instrument_melody_to_audio(combined: np.ndarray) -> np.ndarray:
    n = len(combined) // 3
    freqs, durs, vels = combined[:n], combined[n:2*n], combined[2*n:]
    freqs_q = np.array([snap_to_scale(f) for f in freqs])
    inst = PRESETS["lead"]
    chunks = []
    for f, d, v in zip(freqs_q, durs, vels):
        chunks.append(inst.render(float(f), float(d), velocity=float(v)))
        chunks.append(np.zeros(int(0.03 * _SR)))
    return np.concatenate(chunks)


def _reverb_instrument_melody_to_audio(combined: np.ndarray) -> np.ndarray:
    n = len(combined) // 3
    freqs, durs, vels = combined[:n], combined[n:2*n], combined[2*n:]
    freqs_q = np.array([snap_to_scale(f) for f in freqs])
    inst = PRESETS["lead"]
    chunks = []
    for f, d, v in zip(freqs_q, durs, vels):
        chunks.append(inst.render(float(f), float(d), velocity=float(v)))
        chunks.append(np.zeros(int(0.01 * _SR)))
    dry = np.concatenate(chunks)
    return simple_reverb(dry, decay=0.35, delay_ms=45.0, n_taps=6)


def _layered_melody_to_audio(combined: np.ndarray) -> np.ndarray:
    """Melody + pad drone + phrase grouping + dynamic arc + reverb."""
    n = len(combined) // 3
    freqs, durs, vels = combined[:n], combined[n:2*n], combined[2*n:]
    freqs_q = np.array([snap_to_scale(f) for f in freqs])

    # Dynamic arc: bell-shaped velocity envelope for musical phrasing
    arc = np.sin(np.linspace(0, np.pi, n)) * 0.4 + 0.6
    vels_shaped = vels * arc

    lead = PRESETS["lead"]
    chunks = []
    for i, (f, d, v) in enumerate(zip(freqs_q, durs, vels_shaped)):
        chunks.append(lead.render(float(f), float(d), velocity=float(v)))
        gap = 0.08 if (i + 1) % 4 == 0 else 0.01
        chunks.append(np.zeros(int(gap * _SR)))
    melody = np.concatenate(chunks)

    # Pad drone on the median frequency (root) + fifth above
    root_freq = float(snap_to_scale(np.median(freqs_q)))
    fifth_freq = root_freq * 1.5
    pad = PRESETS["pad"]
    total_dur = len(melody) / _SR
    drone = (0.15 * pad.render(root_freq, total_dur, velocity=0.3)
             + 0.10 * pad.render(fifth_freq, total_dur, velocity=0.2))
    min_len = min(len(melody), len(drone))
    mixed = melody[:min_len] + drone[:min_len]

    return simple_reverb(mixed, decay=0.35, delay_ms=45.0, n_taps=6)


def _expressive_melody_reward(combined: np.ndarray) -> float:
    n = len(combined) // 3
    return expressive_melody_reward(combined, n_notes=n)


def _autoregressive_melody_to_audio(freqs: np.ndarray) -> np.ndarray:
    """Render autoregressive melody with octave doubling + drone + reverb.

    Richer rendering reduces Qwen refusal rate (deterministic per audio
    content — thinner clips are refused more often).
    """
    freqs_q = np.array([snap_to_scale(f) for f in freqs])
    lead = PRESETS["lead"]
    pad = PRESETS["pad"]
    chunks = []
    for i, f in enumerate(freqs_q):
        dur = 0.35 + 0.10 * np.sin(i * 0.8)
        vel = 0.6 + 0.2 * np.sin(i * 0.5 + 1.0)
        note = lead.render(float(f), dur, velocity=vel)
        octave = 0.3 * lead.render(float(f) * 0.5, dur, velocity=vel * 0.5)
        min_n = min(len(note), len(octave))
        note[:min_n] += octave[:min_n]
        chunks.append(note)
        gap = 0.06 if (i + 1) % 4 == 0 else 0.02
        chunks.append(np.zeros(int(gap * _SR)))
    melody = np.concatenate(chunks)
    root_freq = float(snap_to_scale(np.median(freqs_q)))
    fifth_freq = root_freq * 1.5
    total_dur = len(melody) / _SR
    drone = 0.18 * pad.render(root_freq, total_dur, velocity=0.3)
    drone5 = 0.10 * pad.render(fifth_freq, total_dur, velocity=0.2)
    min_len = min(len(melody), len(drone), len(drone5))
    mixed = melody[:min_len] + drone[:min_len] + drone5[:min_len]
    return simple_reverb(mixed, decay=0.35, delay_ms=45.0, n_taps=6)


def _counterpoint_to_audio(voices: np.ndarray) -> np.ndarray:
    """Render multi-voice counterpoint with repetition + drone + reverb.

    voices shape: (n_voices, n_notes) — frequencies in Hz.
    Repeats the pattern twice with slight variation to produce ~8s clips
    (short clips trigger high Qwen refusal rates).
    """
    lead = PRESETS["lead"]
    pad = PRESETS["pad"]
    n_voices = voices.shape[0]
    voice_audios = []
    for v_idx, voice in enumerate(voices):
        freqs_q = np.array([snap_to_scale(f) for f in voice])
        chunks = []
        for rep in range(2):
            for i, f in enumerate(freqs_q):
                dur = 0.40 + 0.08 * np.sin(i * 0.7 + v_idx + rep * 0.3)
                vel = 0.55 + 0.15 * np.sin(i * 0.5 + v_idx * 1.5 + rep)
                note = lead.render(float(f), dur, velocity=vel)
                octave = 0.25 * lead.render(float(f) * 0.5, dur,
                                            velocity=vel * 0.4)
                mn = min(len(note), len(octave))
                note[:mn] += octave[:mn]
                chunks.append(note)
                gap = 0.05 if (i + 1) % 4 == 0 else 0.01
                chunks.append(np.zeros(int(gap * _SR)))
            chunks.append(np.zeros(int(0.08 * _SR)))
        voice_audios.append(np.concatenate(chunks))
    max_len = max(len(a) for a in voice_audios)
    mixed = np.zeros(max_len)
    for a in voice_audios:
        mixed[:len(a)] += a / max(n_voices, 2)
    # Normalize to consistent RMS regardless of voice count
    rms = np.sqrt(np.mean(mixed ** 2))
    if rms > 1e-6:
        mixed *= 0.06 / rms
    all_freqs = voices.flatten()
    root_freq = float(snap_to_scale(np.median(all_freqs)))
    total_dur = max_len / _SR
    drone = 0.15 * pad.render(root_freq, total_dur, velocity=0.25)
    drone5 = 0.08 * pad.render(root_freq * 1.5, total_dur, velocity=0.15)
    min_len = min(max_len, len(drone), len(drone5))
    mixed[:min_len] += drone[:min_len] + drone5[:min_len]
    return simple_reverb(mixed, decay=0.35, delay_ms=45.0, n_taps=6)


def _counterpoint_reward_wrapper(voices: np.ndarray) -> float:
    return counterpoint_reward(voices)


def _vocal_melody_to_audio(combined: np.ndarray) -> np.ndarray:
    """Render expressive melody with TTS vocals mixed in."""
    from src.render.vocals import render_melody_with_vocals
    n = len(combined) // 3
    freqs, durs, vels = combined[:n], combined[n:2*n], combined[2*n:]
    return render_melody_with_vocals(
        freqs, durs, vels,
        instrumental_renderer=_layered_melody_to_audio,
    )


def _progression_to_audio(seqs: np.ndarray) -> np.ndarray:
    chunks = []
    for c in seqs:
        chunks.append(render_chord(c, duration=0.7))
        chunks.append(np.zeros(int(0.06 * SAMPLE_RATE), dtype=np.float64))
    return np.concatenate(chunks)


def _progression_to_arpeggio(seqs: np.ndarray) -> np.ndarray:
    """Render chord progression as arpeggiated melody with pad drone + reverb.

    Block chords cause high Qwen refusal rates on MPS. This renders each
    chord as an ascending arpeggio pattern (3 reps per chord) over a
    sustained pad drone, producing ~8s clips similar to melody training.
    """
    inst = PRESETS["lead"]
    pad = PRESETS["pad"]

    chunks = []
    for chord in seqs:
        sorted_freqs = np.sort(chord)
        for _rep in range(3):
            for freq in sorted_freqs:
                chunks.append(inst.render(float(freq), 0.20, velocity=0.7))
                chunks.append(np.zeros(int(0.02 * _SR)))
        chunks.append(np.zeros(int(0.06 * _SR)))

    melody = np.concatenate(chunks)
    total_dur = len(melody) / _SR

    root_freq = float(np.median(seqs))
    drone = 0.15 * pad.render(root_freq, total_dur, velocity=0.25)
    min_len = min(len(melody), len(drone))
    mixed = melody[:min_len] + drone[:min_len]

    return simple_reverb(mixed, decay=0.3, delay_ms=40.0, n_taps=4)


_ADAPTERS = {
    "melody": _Adapter(
        name="melody",
        build=lambda: MelodyGenerator(latent_dim=16, hidden=128, n_notes=8),
        init_weights="results/phase3_melodies/melody_generator.pt",
        sample_to_audio=_melody_to_audio,
        physics_reward=melody_reward,
    ),
    "melody_v2": _Adapter(
        name="melody_v2",
        build=lambda: MelodyGenerator(latent_dim=32, hidden=256, n_notes=16),
        init_weights="results/rlaif/melody_v2/melody_generator.pt",
        sample_to_audio=_melody_to_audio,
        physics_reward=melody_reward,
    ),
    "melody_v3": _Adapter(
        name="melody_v3",
        build=lambda: ExpressiveMelodyGenerator(
            latent_dim=32, hidden=256, n_notes=16),
        init_weights="results/rlaif/melody_v3/melody_generator.pt",
        sample_to_audio=_expressive_melody_to_audio,
        physics_reward=_expressive_melody_reward,
        output_format="expressive",
    ),
    "melody_v4": _Adapter(
        name="melody_v4",
        build=lambda: ExpressiveMelodyGenerator(
            latent_dim=32, hidden=256, n_notes=16),
        init_weights="results/rlaif/melody_v3_stepwise3/rlaif_generator_best.pt",
        sample_to_audio=_quantized_melody_to_audio,
        physics_reward=_expressive_melody_reward,
        output_format="expressive",
    ),
    "melody_v5": _Adapter(
        name="melody_v5",
        build=lambda: ExpressiveMelodyGenerator(
            latent_dim=32, hidden=256, n_notes=16),
        init_weights="results/rlaif/melody_v3_stepwise3/rlaif_generator_best.pt",
        sample_to_audio=_instrument_melody_to_audio,
        physics_reward=_expressive_melody_reward,
        output_format="expressive",
    ),
    "melody_v6": _Adapter(
        name="melody_v6",
        build=lambda: ExpressiveMelodyGenerator(
            latent_dim=32, hidden=256, n_notes=16),
        init_weights="results/rlaif/melody_v5_qwen/rlaif_generator_best.pt",
        sample_to_audio=_reverb_instrument_melody_to_audio,
        physics_reward=_expressive_melody_reward,
        output_format="expressive",
    ),
    "melody_v7": _Adapter(
        name="melody_v7",
        build=lambda: ExpressiveMelodyGenerator(
            latent_dim=32, hidden=256, n_notes=16),
        init_weights="results/rlaif/melody_v5_qwen/rlaif_generator_best.pt",
        sample_to_audio=_layered_melody_to_audio,
        physics_reward=_expressive_melody_reward,
        output_format="expressive",
    ),
    "autoregressive": _Adapter(
        name="autoregressive",
        build=lambda: AutoregressiveMelodyGenerator(
            hidden=128, n_notes=16, init_noise_dim=16),
        init_weights="",
        sample_to_audio=_autoregressive_melody_to_audio,
        physics_reward=melody_reward,
    ),
    "chord_progression": _Adapter(
        name="chord_progression",
        build=lambda: ChordProgressionGenerator(
            latent_dim=16, hidden=128, n_chords=4, n_voices=3),
        init_weights="results/phase2_progressions/progression_generator.pt",
        sample_to_audio=_progression_to_audio,
        physics_reward=progression_reward,
    ),
    "chord_arpeggio": _Adapter(
        name="chord_arpeggio",
        build=lambda: ChordProgressionGenerator(
            latent_dim=16, hidden=128, n_chords=4, n_voices=3),
        init_weights="results/phase2_progressions/progression_generator.pt",
        sample_to_audio=_progression_to_arpeggio,
        physics_reward=progression_reward,
    ),
    "counterpoint": _Adapter(
        name="counterpoint",
        build=lambda: CounterpointGenerator(
            latent_dim=24, hidden=192, n_voices=2, n_notes=8),
        init_weights="results/phase7_counterpoint/counterpoint_generator.pt",
        sample_to_audio=_counterpoint_to_audio,
        physics_reward=_counterpoint_reward_wrapper,
        output_format="voices",
    ),
    "counterpoint_3v": _Adapter(
        name="counterpoint_3v",
        build=lambda: CounterpointGenerator(
            latent_dim=24, hidden=192, n_voices=3, n_notes=8),
        init_weights="results/phase13_3voice_counterpoint/counterpoint_generator.pt",
        sample_to_audio=_counterpoint_to_audio,
        physics_reward=_counterpoint_reward_wrapper,
        output_format="voices",
    ),
    "counterpoint_4v": _Adapter(
        name="counterpoint_4v",
        build=lambda: CounterpointGenerator(
            latent_dim=24, hidden=192, n_voices=4, n_notes=8),
        init_weights="results/phase13_4voice_counterpoint/counterpoint_generator.pt",
        sample_to_audio=_counterpoint_to_audio,
        physics_reward=_counterpoint_reward_wrapper,
        output_format="voices",
    ),
    "melody_v8_vocal": _Adapter(
        name="melody_v8_vocal",
        build=lambda: ExpressiveMelodyGenerator(
            latent_dim=32, hidden=256, n_notes=16),
        init_weights="results/rlaif/melody_v5_qwen/rlaif_generator_best.pt",
        sample_to_audio=_vocal_melody_to_audio,
        physics_reward=_expressive_melody_reward,
        output_format="expressive",
    ),
}


# ---------- judges ----------------------------------------------------


class _StubJudge:
    """Stand-in judge that returns ``5`` for everything — used by tests
    so we can exercise the loop without needing a 14 GB model."""

    def score_audio(self, audio, sample_rate=SAMPLE_RATE):
        return type("R", (), {"score": 5,
                              "critique": "(stub judge)"})()


class _TheoryJudge:
    """Rule-based music theory judge — no model, no GPU, no refusals."""

    is_theory = True

    def score(self, data: np.ndarray, output_format: str = "freqs") -> "type[object]":
        """Score raw generator output based on the adapter's output_format."""
        per_note = None
        if output_format == "voices":
            freqs_flat = data.flatten()
            score = theory_reward(freqs_flat, voices=data)
            breakdown = theory_reward_breakdown(freqs_flat, voices=data)
            per_note = theory_reward_per_note(freqs_flat)
        elif output_format == "expressive":
            n_notes = len(data) // 3
            f = data[:n_notes]
            d = data[n_notes:2 * n_notes]
            v = data[2 * n_notes:]
            score = theory_reward(f, durations=d, velocities=v)
            breakdown = theory_reward_breakdown(f, durations=d, velocities=v)
            per_note = theory_reward_per_note(f, durations=d, velocities=v)
        else:
            score = theory_reward(data)
            breakdown = theory_reward_breakdown(data)
            per_note = theory_reward_per_note(data)

        critique = " | ".join(f"{k}={val:.2f}" for k, val in breakdown.items())
        return type("R", (), {"score": score, "critique": critique,
                              "per_note": per_note})()


def _build_judge(name: str, qwen_model: str, qwen_device: str,
                 prompt_style: str = "original"):
    if name == "stub":
        return _StubJudge()
    if name == "theory":
        return _TheoryJudge()
    if name == "qwen":
        from src.analysis.qwen_judge import QwenJudge
        return QwenJudge(model=qwen_model, device=qwen_device,
                         prompt_style=prompt_style)
    raise ValueError(f"unknown judge {name!r}")


# ---------- training --------------------------------------------------


def _score_batch(audios: List[np.ndarray], adapter: _Adapter,
                 judge) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (qwen_score_per_sample, physics_reward_per_sample, critiques)."""
    n = len(audios)
    qwen = np.full(n, np.nan, dtype=np.float32)
    phys = np.zeros(n, dtype=np.float32)
    crits: List[str] = []
    for i, a in enumerate(audios):
        try:
            r = judge.score_audio(a, sample_rate=SAMPLE_RATE)
            qwen[i] = float(r.score) if r.score is not None else np.nan
            crits.append(r.critique[:600])
        except Exception as e:    # noqa: BLE001
            crits.append(f"(judge failed: {e!r})")
        # Physics reward operates on the (already-detached) freqs, but
        # we don't have them here — caller will fill phys via param.
    return qwen, phys, crits


def _render_and_score(freqs_np: np.ndarray, adapter: _Adapter,
                      judge, max_retries: int = 2,
                      ) -> Tuple[np.ndarray, np.ndarray, List[str],
                                 Optional[np.ndarray]]:
    n = len(freqs_np)
    phys_rewards = np.zeros(n, dtype=np.float32)
    critiques: List[str] = []

    if getattr(judge, "is_theory", False):
        theory_scores = np.zeros(n, dtype=np.float32)
        per_note_list = []
        for i, f in enumerate(freqs_np):
            r = judge.score(f, output_format=adapter.output_format)
            theory_scores[i] = float(r.score)
            critiques.append(r.critique[:800])
            phys_rewards[i] = float(adapter.physics_reward(f))
            if r.per_note is not None:
                per_note_list.append(r.per_note)
        per_note_all = np.stack(per_note_list) if per_note_list else None
        return theory_scores, phys_rewards, critiques, per_note_all

    # Qwen/stub judge: render audio first, then score.
    audios = [adapter.sample_to_audio(f).astype(np.float32) for f in freqs_np]
    qwen_scores = np.full(n, np.nan, dtype=np.float32)
    for i, (a, f) in enumerate(zip(audios, freqs_np)):
        try:
            r = judge.score_audio(a, sample_rate=SAMPLE_RATE,
                                  max_retries=max_retries)
            qwen_scores[i] = float(r.score) if r.score is not None \
                else np.nan
            critiques.append(r.critique[:800])
        except Exception as e:    # noqa: BLE001
            critiques.append(f"(judge failed: {e!r})")
        phys_rewards[i] = float(adapter.physics_reward(f))
    return qwen_scores, phys_rewards, critiques, None


def train(generator: str,
          judge: str = "qwen",
          n_steps: int = 200,
          batch_size: int = 4,
          lr: float = 3e-5,
          seed: int = 0,
          qwen_weight: float = 1.0,
          physics_weight: float = 0.3,
          eval_every: int = 5,
          qwen_model: str = "Qwen/Qwen2.5-Omni-7B",
          qwen_device: str = "mps",
          init_from: Optional[str] = None,
          out_dir: str = "results/rlaif/run",
          prompt_style: str = "original",
          max_retries: int = 2,
          freq_std_clamp: Optional[float] = None,
          scale_snap: float = 0.0,
          scale_key: Optional[str] = None):
    if generator not in _ADAPTERS:
        raise ValueError(f"unknown generator {generator!r}")
    adapter = _ADAPTERS[generator]

    torch.manual_seed(seed)
    np.random.seed(seed)

    gen = adapter.build()

    if freq_std_clamp is not None and hasattr(gen, "_freq_std_clamp"):
        gen._freq_std_clamp = freq_std_clamp
        print(f"freq_std_clamp set to {freq_std_clamp}", flush=True)
    if scale_snap > 0 and hasattr(gen, "scale_snap"):
        gen.scale_snap = scale_snap
        print(f"scale_snap set to {scale_snap}", flush=True)
    if scale_key and hasattr(gen, "_scale_table"):
        from src.generator.melody_generator import _build_scale_table
        _ROOTS = {"C": -9, "D": -7, "E": -5, "F": -4, "G": -2, "A": 0, "B": 2}
        from src.reward.theory_judge import SCALES as _SCALES
        parts = scale_key.split("_", 1)
        root_name = parts[0].upper()
        scale_name = parts[1] if len(parts) > 1 else "major"
        if root_name in _ROOTS and scale_name in _SCALES:
            gen._scale_table = _build_scale_table(
                _SCALES[scale_name], _ROOTS[root_name])
            print(f"scale set to {root_name} {scale_name}", flush=True)

    src_weights = init_from or adapter.init_weights
    if src_weights and Path(src_weights).is_file():
        state = torch.load(src_weights, map_location="cpu")
        gen.load_state_dict(state)
        print(f"loaded init weights from {src_weights}", flush=True)
    else:
        print(f"WARNING: init weights {src_weights} not found — "
              f"starting from scratch", flush=True)

    # Low LR — we're fine-tuning, not restarting.
    opt = torch.optim.Adam(gen.parameters(), lr=lr)

    print(f"building judge: {judge} on {qwen_device} "
          f"(prompt={prompt_style})", flush=True)
    j = _build_judge(judge, qwen_model, qwen_device,
                     prompt_style=prompt_style)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    history: List[dict] = []
    best_reward = -float("inf")
    best_state = None
    t_start = time.time()

    # Running stats for reward normalization — makes REINFORCE work
    # even when Qwen scores cluster in a narrow range.
    reward_running_mean = 0.0
    reward_running_var = 1.0
    reward_count = 0

    is_theory = getattr(j, "is_theory", False)

    for step in range(n_steps):
        freqs, log_prob = gen.sample(batch_size)
        freqs_np = freqs.detach().cpu().numpy()

        judge_scores, phys_rewards, critiques, per_note_rewards = \
            _render_and_score(freqs_np, adapter, j, max_retries=max_retries)

        if is_theory:
            reward_np = judge_scores + physics_weight * phys_rewards
        else:
            valid_mask = ~np.isnan(judge_scores)
            if valid_mask.any():
                fill = float(np.nanmean(judge_scores))
            else:
                fill = 5.0
            qwen_filled = np.where(valid_mask, judge_scores, fill)
            norm_q = (qwen_filled - 5.0) / 5.0
            reward_np = qwen_weight * norm_q + physics_weight * phys_rewards

        rewards = torch.tensor(reward_np, dtype=torch.float32)

        for r in reward_np:
            reward_count += 1
            delta = r - reward_running_mean
            reward_running_mean += delta / reward_count
            reward_running_var += delta * (r - reward_running_mean)

        if reward_count > 2:
            std = max(np.sqrt(reward_running_var / reward_count), 0.01)
            adv = (rewards - reward_running_mean) / std
        else:
            adv = rewards - rewards.mean()

        # Per-note credit assignment: use per-note log-probs * per-note
        # advantages when available (theory judge + expressive generator).
        per_note_lp = getattr(gen, "_last_per_note_lp", None)
        if (per_note_rewards is not None and per_note_lp is not None
                and per_note_rewards.shape[-1] == per_note_lp.shape[-1]):
            pn_adv = torch.tensor(per_note_rewards, dtype=torch.float32)
            pn_mean = pn_adv.mean()
            pn_std = max(float(pn_adv.std()), 0.01)
            pn_adv = (pn_adv - pn_mean) / pn_std
            per_note_loss = -(per_note_lp * pn_adv).sum(-1).mean()
            scalar_loss = -(log_prob * adv).mean()
            loss = 0.3 * scalar_loss + 0.7 * per_note_loss
        else:
            loss = -(log_prob * adv).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
        opt.step()

        mean_judge = float(judge_scores.mean())
        mean_phys = float(phys_rewards.mean())
        mean_total = float(rewards.mean())
        elapsed = time.time() - t_start
        entry = {
            "step": step,
            "mean_judge": mean_judge,
            "mean_physics": mean_phys,
            "mean_reward": mean_total,
            "loss": float(loss.item()),
            "elapsed_sec": round(elapsed, 1),
            "critiques": critiques,
            "judge_scores": [float(s) for s in judge_scores.tolist()],
        }
        history.append(entry)
        label = "theory" if is_theory else "qwen"
        print(f"[{step:3d}/{n_steps}] {label}={mean_judge:.2f} "
              f"phys={mean_phys:+.3f} reward={mean_total:+.3f} "
              f"loss={loss.item():+.3f}  ({elapsed:.0f}s)", flush=True)

        if mean_total > best_reward:
            best_reward = mean_total
            best_state = {k: v.detach().clone()
                          for k, v in gen.state_dict().items()}

        # Stream snapshots so a Ctrl-C still leaves a usable run.
        if (step % eval_every == 0) or (step == n_steps - 1):
            torch.save(gen.state_dict(), out_path / "rlaif_generator.pt")
            if best_state is not None:
                torch.save(best_state, out_path / "rlaif_generator_best.pt")
            (out_path / "history.json").write_text(
                json.dumps({
                    "generator": generator,
                    "judge": judge,
                    "qwen_model": qwen_model,
                    "qwen_device": qwen_device,
                    "prompt_style": prompt_style,
                    "batch_size": batch_size,
                    "lr": lr,
                    "qwen_weight": qwen_weight,
                    "physics_weight": physics_weight,
                    "best_reward": best_reward,
                    "history": history,
                }, indent=2),
            )

    print(f"done — best mean_reward={best_reward:.3f}, "
          f"saved to {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generator", required=True,
                   choices=list(_ADAPTERS.keys()))
    p.add_argument("--judge", default="qwen", choices=["qwen", "theory", "stub"])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--qwen-weight", type=float, default=1.0)
    p.add_argument("--physics-weight", type=float, default=0.3)
    p.add_argument("--qwen-model", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--qwen-device",
                   default=os.environ.get("QWEN_DEVICE", "mps"))
    p.add_argument("--init-from", default=None,
                   help="path to .pt to initialise the policy from "
                        "(default: the phase-2/3 best checkpoint)")
    p.add_argument("--out-dir", default="results/rlaif/run")
    p.add_argument("--prompt-style", default="original",
                   choices=["original", "neutral"])
    p.add_argument("--max-retries", type=int, default=2,
                   help="Qwen retries per sample on refusal (default 2)")
    p.add_argument("--freq-std-clamp", type=float, default=None,
                   help="max log-std for freq sampling (default: 1.0; "
                        "try -1.5 for theory training)")
    p.add_argument("--scale-snap", type=float, default=0.0,
                   help="soft scale quantization strength 0-1 "
                        "(0=off, 1=hard snap)")
    p.add_argument("--scale-key", type=str, default=None,
                   help="scale for snap, e.g. 'C_major', 'A_natural_minor', "
                        "'D_pentatonic_major' (default: C major)")
    args = p.parse_args()

    train(
        generator=args.generator,
        judge=args.judge,
        n_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        qwen_weight=args.qwen_weight,
        physics_weight=args.physics_weight,
        qwen_model=args.qwen_model,
        qwen_device=args.qwen_device,
        init_from=args.init_from,
        out_dir=args.out_dir,
        prompt_style=args.prompt_style,
        max_retries=args.max_retries,
        freq_std_clamp=args.freq_std_clamp,
        scale_snap=args.scale_snap,
        scale_key=args.scale_key,
    )


if __name__ == "__main__":
    main()
