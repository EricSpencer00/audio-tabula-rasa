"""
RLAIF training: optimize a generator against the Qwen2.5-Omni-7B
audio critic's score.

For each REINFORCE step:
  1. Sample a batch of generator outputs.
  2. Render each one to audio with the same synth used in the preview.
  3. Score each audio with the Qwen audio judge.
  4. Combine the Qwen score with the original physics reward (weighted
     sum) and run a policy-gradient update.

Designed to run on a 64 GB MacBook Pro: the 7B audio model takes
~14 GB in bf16 and inference per clip is ~5–15 s on Apple Silicon.
A useful schedule on that hardware:

  --batch-size 4 --steps 200

Per step that's 4 audio renders + 4 Qwen scores ≈ 1 minute.

Default target generator is the Phase-3 melody generator. Pass
`--generator chord_progression` to RLAIF the chord generator instead.

Important: this is opt-in / local-only. CI cannot run it (no Qwen,
no MPS). It writes results to results/rlaif/<generator>/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from src.analysis.qwen_judge import QwenAudioJudge


# -- generator factories --------------------------------------------

def _make_melody_generator():
    from src.generator.melody_generator import MelodyGenerator
    gen = MelodyGenerator(latent_dim=16, hidden=128, n_notes=8)
    weight = "results/phase3_melodies/melody_generator.pt"
    if Path(weight).exists():
        gen.load_state_dict(torch.load(weight, map_location="cpu"))
    return gen


def _make_chord_progression_generator():
    from src.generator.chord_generator import ChordProgressionGenerator
    gen = ChordProgressionGenerator(latent_dim=16, hidden=128,
                                     n_chords=4, n_voices=3)
    weight = "results/phase2_progressions/progression_generator.pt"
    if Path(weight).exists():
        gen.load_state_dict(torch.load(weight, map_location="cpu"))
    return gen


GENERATOR_FACTORIES: dict[str, Callable] = {
    "melody": _make_melody_generator,
    "chord_progression": _make_chord_progression_generator,
}


# -- physics reward fallbacks (so the model doesn't drift even with
#    very few Qwen evaluations) --------------------------------------

def _physics_reward_melody(freqs: np.ndarray) -> float:
    from src.reward.psychoacoustic import melody_reward
    return float(melody_reward(freqs))


def _physics_reward_progression(seq: np.ndarray) -> float:
    from src.reward.psychoacoustic import progression_reward
    return float(progression_reward(seq, voice_leading_weight=0.5))


PHYSICS_REWARDS = {
    "melody": _physics_reward_melody,
    "chord_progression": _physics_reward_progression,
}


# -- per-sample audio renderers (kept lightweight) -------------------

def _render_melody_audio(freqs: np.ndarray) -> tuple[np.ndarray, int]:
    from src.render.song import (Note, Song, Track, write_wav,
                                  arrange_melody_track)
    from src.render.instruments import SAMPLE_RATE, get
    # 8 notes at 100 BPM, ~5 s
    inst = get("lead")
    audio_chunks = []
    for f in freqs:
        chunk = inst.render(float(f), duration=0.45)
        audio_chunks.append(chunk)
    audio = np.concatenate(audio_chunks)
    return audio.astype(np.float32), SAMPLE_RATE


def _render_progression_audio(seq: np.ndarray) -> tuple[np.ndarray, int]:
    from src.render.instruments import SAMPLE_RATE, get
    inst = get("pad")
    chunks = []
    for chord in seq:
        # mix the voices
        per_voice = [inst.render(float(f), duration=1.2) for f in chord]
        max_len = max(len(a) for a in per_voice)
        m = np.zeros(max_len)
        for a in per_voice:
            m[: len(a)] += a / len(per_voice)
        chunks.append(m)
        chunks.append(np.zeros(int(0.1 * SAMPLE_RATE)))
    return np.concatenate(chunks).astype(np.float32), SAMPLE_RATE


RENDERERS = {
    "melody": _render_melody_audio,
    "chord_progression": _render_progression_audio,
}


# -- training ---------------------------------------------------------

def train_rlaif(
    generator_name: str = "melody",
    n_steps: int = 200,
    batch_size: int = 4,
    lr: float = 1e-4,
    physics_weight: float = 0.3,
    qwen_weight: float = 1.0,
    seed: int = 0,
    log_every: int = 5,
    qwen_model: str = QwenAudioJudge.DEFAULT_MODEL,
    qwen_device: str = "auto",
    out_dir: str = None,
):
    if generator_name not in GENERATOR_FACTORIES:
        raise ValueError(
            f"unknown generator {generator_name}; "
            f"options: {list(GENERATOR_FACTORIES)}"
        )
    out_path = Path(out_dir or f"results/rlaif/{generator_name}")
    out_path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    gen = GENERATOR_FACTORIES[generator_name]()
    opt = torch.optim.Adam(gen.parameters(), lr=lr)
    judge = QwenAudioJudge(model_name=qwen_model, device=qwen_device)
    physics_fn = PHYSICS_REWARDS[generator_name]
    render_fn = RENDERERS[generator_name]

    history = []
    best_eval = -float("inf")
    best_state = None

    for step in range(n_steps):
        out = gen.sample(batch_size)
        # generator may return (freqs, log_prob) or (freqs, onsets, log_prob)
        if len(out) == 2:
            freqs, log_prob = out
            np_outputs = freqs.detach().cpu().numpy()
        else:
            freqs, onsets, log_prob = out
            np_outputs = freqs.detach().cpu().numpy()

        physics_rewards = np.array([physics_fn(o) for o in np_outputs])

        # Score each sample with Qwen — this is the slow part
        qwen_scores = []
        for sample in np_outputs:
            try:
                audio, sr = render_fn(sample)
                r = judge.score(audio, sr)
                qwen_scores.append(float(r.score) if r.score else 5.0)
            except Exception as e:    # noqa: BLE001
                print(f"[rlaif] qwen scoring failed: {e}")
                qwen_scores.append(5.0)
        qwen_scores = np.array(qwen_scores)

        rewards = (qwen_weight * (qwen_scores - 5.0) / 5.0
                   + physics_weight * physics_rewards)
        adv = rewards - rewards.mean()
        adv_t = torch.tensor(adv, dtype=torch.float32)
        loss = -(log_prob * adv_t).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
        opt.step()

        entry = {
            "step": step,
            "mean_qwen_score": float(qwen_scores.mean()),
            "mean_physics_reward": float(physics_rewards.mean()),
            "mean_combined_reward": float(rewards.mean()),
        }
        history.append(entry)
        if step % log_every == 0:
            print(f"[{step:4d}/{n_steps}] qwen={entry['mean_qwen_score']:.2f}/10  "
                  f"physics={entry['mean_physics_reward']:+.3f}  "
                  f"combined={entry['mean_combined_reward']:+.3f}")

        if entry["mean_combined_reward"] > best_eval:
            best_eval = entry["mean_combined_reward"]
            best_state = {k: v.clone() for k, v in gen.state_dict().items()}

    if best_state is not None:
        gen.load_state_dict(best_state)
    torch.save(gen.state_dict(), out_path / "rlaif_generator.pt")
    (out_path / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[rlaif] best combined reward {best_eval:+.3f}, "
          f"saved to {out_path}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", default="melody",
                        choices=list(GENERATOR_FACTORIES))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--physics-weight", type=float, default=0.3)
    parser.add_argument("--qwen-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--qwen-model",
                        default=QwenAudioJudge.DEFAULT_MODEL)
    parser.add_argument("--qwen-device", default="auto")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    train_rlaif(
        generator_name=args.generator,
        n_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        physics_weight=args.physics_weight,
        qwen_weight=args.qwen_weight,
        seed=args.seed,
        qwen_model=args.qwen_model,
        qwen_device=args.qwen_device,
        out_dir=args.out_dir,
    )
