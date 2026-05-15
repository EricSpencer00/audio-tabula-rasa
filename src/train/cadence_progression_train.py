"""
Phase-12 cadence-aware chord-progression training.

Extends the Phase-2 progression training with a cadence-arc reward:
the model is encouraged to produce K-chord sequences whose middle
chords have higher dissonance (more "tension") than the endpoints
(more "resolved").

Keeps every other Phase-2 term (per-chord consonance + voice-leading
+ voice-spread). The cadence term is additive and re-uses the
already-saved progression generator architecture.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.generator.chord_generator import ChordProgressionGenerator
from src.reward.cadence import cadence_arc, expectation_arc
from src.reward.psychoacoustic import (
    chord_dissonance,
    progression_reward,
    triad_label,
    voice_leading_cost,
)


def _score(freqs: torch.Tensor, n_chords: int,
           spread_weight: float, vl_weight: float,
           cadence_weight: float, min_semitones: float) -> np.ndarray:
    np_seqs = freqs.detach().cpu().numpy()
    out = np.empty(np_seqs.shape[0], dtype=np.float32)
    for i, seq in enumerate(np_seqs):
        base = progression_reward(seq,
                                   spread_weight=spread_weight,
                                   voice_leading_weight=vl_weight,
                                   min_semitones=min_semitones)
        arc = cadence_arc(seq)
        out[i] = base + cadence_weight * arc
    return out


def _entropy_coef(step: int, n_steps: int,
                  start: float = 0.01, end: float = -0.02) -> float:
    frac = min(1.0, max(0.0, step / max(1, n_steps - 1)))
    return start + (end - start) * frac


def train_cadence_progressions(n_steps=2500, batch_size=64, lr=1e-3,
                               log_every=50, seed=0, n_chords=4,
                               spread_weight=2.0, vl_weight=0.5,
                               cadence_weight=2.0, min_semitones=1.5,
                               ent_start=0.01, ent_end=-0.02,
                               out_dir="results/phase12_cadence_progressions"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    gen = ChordProgressionGenerator(latent_dim=16, hidden=128,
                                    n_chords=n_chords, n_voices=3)
    opt = torch.optim.Adam(gen.parameters(), lr=lr)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    history = []
    best_eval_reward = -float("inf")
    best_state = None

    for step in range(n_steps):
        z = torch.randn(batch_size, gen.latent_dim)
        mean, std = gen(z)
        dist = torch.distributions.Normal(mean, std)
        freqs_raw = dist.rsample()
        freqs_flat = freqs_raw.clamp(min=gen.F_MIN, max=gen.F_MAX)
        log_prob = dist.log_prob(freqs_flat).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        freqs = freqs_flat.view(batch_size, gen.n_chords, gen.n_voices)

        rewards = torch.tensor(_score(freqs, n_chords, spread_weight,
                                       vl_weight, cadence_weight,
                                       min_semitones))
        adv = rewards - rewards.mean()
        adv = adv / (rewards.std() + 1e-8)

        ent_coef = _entropy_coef(step, n_steps, ent_start, ent_end)
        loss = -(log_prob * adv).mean() - ent_coef * entropy

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
        opt.step()

        if step % log_every == 0 or step == n_steps - 1:
            with torch.no_grad():
                ef, _ = gen.sample(128)
                np_seqs = ef.cpu().numpy()
                er = _score(ef, n_chords, spread_weight, vl_weight,
                             cadence_weight, min_semitones)
                arcs = np.array([cadence_arc(s) for s in np_seqs])
                exp_arcs = np.array([expectation_arc(s) for s in np_seqs])
                diss = np.mean([chord_dissonance(c)
                                 for s in np_seqs for c in s])
            entry = {
                "step": step,
                "mean_reward": float(er.mean()),
                "mean_cadence_arc": float(arcs.mean()),
                "mean_expectation_arc": float(exp_arcs.mean()),
                "mean_dissonance": float(diss),
            }
            history.append(entry)
            print(f"[{step:5d}] r={entry['mean_reward']:+.3f}  "
                  f"cad_arc={entry['mean_cadence_arc']:+.3f}  "
                  f"exp_arc={entry['mean_expectation_arc']:+.3f}  "
                  f"diss={entry['mean_dissonance']:.3f}")
            if entry["mean_reward"] > best_eval_reward:
                best_eval_reward = entry["mean_reward"]
                best_state = {k: v.clone() for k, v in gen.state_dict().items()}

    if best_state is not None:
        print(f"Best eval reward: {best_eval_reward:+.3f}; saving best ckpt")
        gen.load_state_dict(best_state)
    torch.save(gen.state_dict(), out_path / "progression_generator.pt")
    with open(out_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with torch.no_grad():
        sample, _ = gen.sample(256)
    np.save(out_path / "final_progressions.npy", sample.cpu().numpy())
    print(f"\nSaved Phase-12 model + history + samples to {out_path}/")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-chords", type=int, default=4)
    parser.add_argument("--cadence-weight", type=float, default=2.0)
    parser.add_argument("--out-dir", type=str,
                        default="results/phase12_cadence_progressions")
    args = parser.parse_args()
    train_cadence_progressions(
        n_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        n_chords=args.n_chords,
        cadence_weight=args.cadence_weight,
        out_dir=args.out_dir,
    )
