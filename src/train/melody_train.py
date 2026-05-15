"""
REINFORCE training loop for Phase 3 — short melodies.

A monophonic generator outputs N notes per latent. The reward combines
sequential consonance, implied-fundamental salience (Terhardt virtual
pitch), contour smoothness, and a soft pitch-class diversity term.
The agent is expected to discover scale-like structure with no music
data — pitches that fit a common implied root, organized into modest
melodic steps.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.generator.melody_generator import MelodyGenerator
from src.reward.psychoacoustic import (
    implied_fundamental_salience,
    melody_reward,
    melody_step_smoothness,
    pitch_class_diversity,
    sequential_consonance,
)


def _score_melodies(freqs: torch.Tensor, **kw) -> np.ndarray:
    np_freqs = freqs.detach().cpu().numpy()
    out = np.empty(np_freqs.shape[0], dtype=np.float32)
    for i, m in enumerate(np_freqs):
        out[i] = melody_reward(m, **kw)
    return out


def _entropy_coef(step: int, n_steps: int,
                  start: float = 0.01, end: float = -0.02) -> float:
    frac = min(1.0, max(0.0, step / max(1, n_steps - 1)))
    return start + (end - start) * frac


def _pitch_class_histogram(freqs, n_bins: int = 24) -> np.ndarray:
    """24-bin (50-cent) pitch-class histogram aggregated over melodies."""
    pc = (np.log2(freqs) * 12.0) % 12.0
    edges = np.linspace(0, 12, n_bins + 1)
    h, _ = np.histogram(pc.flatten(), bins=edges)
    return h


def train_melodies(n_steps=3000, batch_size=64, lr=3e-4, log_every=50,
                   seed=0, n_notes=8,
                   tonal_weight=5.0, contour_weight=0.05,
                   diversity_weight=1.5, min_unique=4,
                   ent_start=0.01, ent_end=-0.02,
                   out_dir="results/phase3_melodies"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    gen = MelodyGenerator(latent_dim=16, hidden=128, n_notes=n_notes)
    opt = torch.optim.Adam(gen.parameters(), lr=lr)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    history = []
    best_eval_reward = -float("inf")
    best_state = None
    kw = dict(tonal_weight=tonal_weight,
              contour_weight=contour_weight,
              diversity_weight=diversity_weight,
              min_unique=min_unique)

    for step in range(n_steps):
        z = torch.randn(batch_size, gen.latent_dim)
        log_mean, std = gen(z)
        dist = torch.distributions.Normal(log_mean, std)
        log_freqs_raw = dist.rsample()
        log_freqs = log_freqs_raw.clamp(min=gen._log_lo, max=gen._log_hi)
        freqs = torch.exp(log_freqs)
        log_prob = dist.log_prob(log_freqs).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        rewards = torch.tensor(_score_melodies(freqs, **kw))

        # No std-normalization: melody rewards have a long left tail
        # (bad melodies can score very negatively) and normalizing by
        # std washes out the signal from rare good samples.
        adv = rewards - rewards.mean()

        ent_coef = _entropy_coef(step, n_steps, ent_start, ent_end)
        loss = -(log_prob * adv).mean() - ent_coef * entropy

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
        opt.step()

        if step % log_every == 0 or step == n_steps - 1:
            with torch.no_grad():
                ef, _ = gen.sample(128)
                ef_np = ef.cpu().numpy()
                er = _score_melodies(ef, **kw)
                tonal = np.mean([implied_fundamental_salience(m) for m in ef_np])
                diss = np.mean([sequential_consonance(m) for m in ef_np])
                contour = np.mean([melody_step_smoothness(m) for m in ef_np])
                npc = np.mean([pitch_class_diversity(m) for m in ef_np])
            entry = {
                "step": step,
                "mean_reward": float(er.mean()),
                "mean_tonal_salience": float(tonal),
                "mean_dissonance": float(diss),
                "mean_contour": float(contour),
                "mean_pitch_classes": float(npc),
            }
            history.append(entry)
            print(f"[{step:5d}] r={entry['mean_reward']:+.3f}  "
                  f"tonal={entry['mean_tonal_salience']:.3f}  "
                  f"diss={entry['mean_dissonance']:.3f}  "
                  f"contour={entry['mean_contour']:.2f}  "
                  f"#pc={entry['mean_pitch_classes']:.1f}")
            if entry["mean_reward"] > best_eval_reward:
                best_eval_reward = entry["mean_reward"]
                best_state = {k: v.clone() for k, v in gen.state_dict().items()}

    if best_state is not None:
        print(f"Best eval reward: {best_eval_reward:+.3f}; "
              "saving best checkpoint")
        gen.load_state_dict(best_state)
    torch.save(gen.state_dict(), out_path / "melody_generator.pt")
    with open(out_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    with torch.no_grad():
        sample_freqs, _ = gen.sample(512)
    np.save(out_path / "final_melodies.npy", sample_freqs.cpu().numpy())
    print(f"\nSaved melody model + history + sample to {out_path}/")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-notes", type=int, default=8)
    parser.add_argument("--tonal-weight", type=float, default=5.0)
    parser.add_argument("--contour-weight", type=float, default=0.05)
    parser.add_argument("--diversity-weight", type=float, default=1.5)
    parser.add_argument("--min-unique", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default="results/phase3_melodies")
    args = parser.parse_args()
    train_melodies(
        n_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        n_notes=args.n_notes,
        tonal_weight=args.tonal_weight,
        contour_weight=args.contour_weight,
        diversity_weight=args.diversity_weight,
        min_unique=args.min_unique,
        out_dir=args.out_dir,
    )
