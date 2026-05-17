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

from src.generator.chord_generator import ChordProgressionGenerator
from src.generator.melody_generator import MelodyGenerator
from src.render.synth import render_chord, render_melody
from src.reward.psychoacoustic import melody_reward, progression_reward


SAMPLE_RATE = 44_100   # synth.SAMPLE_RATE


# ---------- generator adapters ----------------------------------------


@dataclass
class _Adapter:
    name: str
    build: Callable[[], torch.nn.Module]
    init_weights: str
    sample_to_audio: Callable[[np.ndarray], np.ndarray]
    physics_reward: Callable[[np.ndarray], float]


def _melody_to_audio(freqs: np.ndarray) -> np.ndarray:
    # MelodyGenerator → 8 frequencies. ~3.0 s of audio.
    return render_melody(freqs, note_duration=0.35, gap=0.03)


def _progression_to_audio(seqs: np.ndarray) -> np.ndarray:
    # ChordProgressionGenerator → (4, 3) frequencies. ~3.2 s of audio.
    chunks = []
    for c in seqs:
        chunks.append(render_chord(c, duration=0.7))
        chunks.append(np.zeros(int(0.06 * SAMPLE_RATE), dtype=np.float64))
    return np.concatenate(chunks)


_ADAPTERS = {
    "melody": _Adapter(
        name="melody",
        build=lambda: MelodyGenerator(latent_dim=16, hidden=128, n_notes=8),
        init_weights="results/phase3_melodies/melody_generator.pt",
        sample_to_audio=_melody_to_audio,
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
}


# ---------- judges ----------------------------------------------------


class _StubJudge:
    """Stand-in judge that returns ``5`` for everything — used by tests
    so we can exercise the loop without needing a 14 GB model."""

    def score_audio(self, audio, sample_rate=SAMPLE_RATE):
        return type("R", (), {"score": 5,
                              "critique": "(stub judge)"})()


def _build_judge(name: str, qwen_model: str, qwen_device: str):
    if name == "stub":
        return _StubJudge()
    if name == "qwen":
        from src.analysis.qwen_judge import QwenJudge
        return QwenJudge(model=qwen_model, device=qwen_device)
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
                      judge) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    audios = [adapter.sample_to_audio(f).astype(np.float32) for f in freqs_np]
    qwen_scores = np.full(len(audios), np.nan, dtype=np.float32)
    phys_rewards = np.zeros(len(audios), dtype=np.float32)
    critiques: List[str] = []
    for i, (a, f) in enumerate(zip(audios, freqs_np)):
        try:
            r = judge.score_audio(a, sample_rate=SAMPLE_RATE)
            qwen_scores[i] = float(r.score) if r.score is not None \
                else np.nan
            critiques.append(r.critique[:800])
        except Exception as e:    # noqa: BLE001
            critiques.append(f"(judge failed: {e!r})")
        phys_rewards[i] = float(adapter.physics_reward(f))
    return qwen_scores, phys_rewards, critiques


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
          out_dir: str = "results/rlaif/run"):
    if generator not in _ADAPTERS:
        raise ValueError(f"unknown generator {generator!r}")
    adapter = _ADAPTERS[generator]

    torch.manual_seed(seed)
    np.random.seed(seed)

    gen = adapter.build()
    src_weights = init_from or adapter.init_weights
    if Path(src_weights).exists():
        state = torch.load(src_weights, map_location="cpu")
        gen.load_state_dict(state)
        print(f"loaded init weights from {src_weights}", flush=True)
    else:
        print(f"WARNING: init weights {src_weights} not found — "
              f"starting from scratch", flush=True)

    # Low LR — we're fine-tuning, not restarting.
    opt = torch.optim.Adam(gen.parameters(), lr=lr)

    print(f"building judge: {judge} on {qwen_device}", flush=True)
    j = _build_judge(judge, qwen_model, qwen_device)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    history: List[dict] = []
    best_reward = -float("inf")
    best_state = None
    t_start = time.time()

    for step in range(n_steps):
        # Resample with grad enabled so we get log_prob through the
        # policy at the sampled freqs.
        freqs, log_prob = gen.sample(batch_size)
        freqs_np = freqs.detach().cpu().numpy()

        qwen_scores, phys_rewards, critiques = _render_and_score(
            freqs_np, adapter, j,
        )

        # If the LLM failed to emit a score for a sample, fall back to
        # the batch mean rather than letting NaN poison the gradient.
        valid_mask = ~np.isnan(qwen_scores)
        if valid_mask.any():
            fill = float(np.nanmean(qwen_scores))
        else:
            fill = 5.0
        qwen_filled = np.where(valid_mask, qwen_scores, fill)
        norm_q = (qwen_filled - 5.0) / 5.0    # ∈ [-0.8, 1.0]
        reward_np = qwen_weight * norm_q + physics_weight * phys_rewards
        rewards = torch.tensor(reward_np, dtype=torch.float32)

        adv = rewards - rewards.mean()

        loss = -(log_prob * adv).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
        opt.step()

        mean_qwen = float(np.nanmean(qwen_scores)) \
            if valid_mask.any() else float("nan")
        mean_phys = float(phys_rewards.mean())
        mean_total = float(rewards.mean())
        elapsed = time.time() - t_start
        entry = {
            "step": step,
            "mean_qwen": mean_qwen,
            "mean_physics": mean_phys,
            "mean_reward": mean_total,
            "loss": float(loss.item()),
            "elapsed_sec": round(elapsed, 1),
            "critiques": critiques,
            "qwen_scores": [None if np.isnan(s) else float(s)
                            for s in qwen_scores.tolist()],
        }
        history.append(entry)
        print(f"[{step:3d}/{n_steps}] qwen={mean_qwen:.2f} "
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
    p.add_argument("--judge", default="qwen", choices=["qwen", "stub"])
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
    )


if __name__ == "__main__":
    main()
