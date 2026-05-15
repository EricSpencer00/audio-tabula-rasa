"""
REINFORCE for the autoregressive melody generator (Phase 11).

Same reward as Phase 3 — sequential consonance + Terhardt virtual
pitch + PC diversity. The only difference is the architecture, which
unrolls a GRU and so can in principle learn motif structure.

Adds a motif-autocorrelation reward term: the log-frequency sequence's
autocorrelation at lags 2..N/2 is computed, and the *maximum* of those
values (excluding lag 0) is added to the reward. Repeating patterns
(motifs at any lag) score positive.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.generator.autoregressive_melody import AutoregressiveMelodyGenerator
from src.reward.psychoacoustic import (
    implied_fundamental_salience,
    melody_reward,
    pitch_class_diversity,
)


def motif_autocorrelation(log_freqs, min_lag: int = 2):
    """Max normalized autocorrelation of the log-frequency sequence
    at lags in [min_lag, len/2]. Captures repeated motifs."""
    lf = np.asarray(log_freqs, dtype=np.float64)
    lf = lf - lf.mean()
    if lf.std() < 1e-6:
        return 0.0
    n = len(lf)
    max_lag = max(min_lag + 1, n // 2 + 1)
    best = 0.0
    for lag in range(min_lag, max_lag):
        a = lf[:-lag]
        b = lf[lag:]
        ac = float(np.mean(a * b) / (lf.var() + 1e-8))
        if ac > best:
            best = ac
    return best


def _score(freqs, motif_weight: float = 1.0):
    out = np.empty(freqs.shape[0], dtype=np.float32)
    np_f = freqs.detach().cpu().numpy()
    for i, m in enumerate(np_f):
        mr = melody_reward(m)
        motif = motif_autocorrelation(np.log2(m))
        out[i] = mr + motif_weight * motif
    return out


def _entropy_coef(step: int, n_steps: int,
                  start: float = 0.01, end: float = -0.02) -> float:
    frac = min(1.0, max(0.0, step / max(1, n_steps - 1)))
    return start + (end - start) * frac


def train_autoregressive_melody(n_steps=1500, batch_size=64, lr=3e-4,
                                log_every=50, seed=0, n_notes=16,
                                motif_weight=1.0,
                                ent_start=0.01, ent_end=-0.02,
                                out_dir="results/phase11_autoregressive_melody"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    gen = AutoregressiveMelodyGenerator(hidden=96, n_notes=n_notes)
    opt = torch.optim.Adam(gen.parameters(), lr=lr)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    history = []
    best_eval_reward = -float("inf")
    best_state = None

    for step in range(n_steps):
        freqs, log_prob = gen.sample(batch_size)
        entropy = gen.entropy_at(batch_size)

        rewards = torch.tensor(_score(freqs, motif_weight=motif_weight))
        adv = rewards - rewards.mean()

        ent_coef = _entropy_coef(step, n_steps, ent_start, ent_end)
        loss = -(log_prob * adv).mean() - ent_coef * entropy

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
        opt.step()

        if step % log_every == 0 or step == n_steps - 1:
            with torch.no_grad():
                f, _ = gen.sample(128)
                fn = f.cpu().numpy()
                er = _score(f, motif_weight=motif_weight)
                tonal = np.mean([implied_fundamental_salience(m) for m in fn])
                pcs = np.mean([pitch_class_diversity(m) for m in fn])
                motifs = np.mean([motif_autocorrelation(np.log2(m))
                                   for m in fn])
            entry = {
                "step": step,
                "mean_reward": float(er.mean()),
                "mean_tonal": float(tonal),
                "mean_pcs": float(pcs),
                "mean_motif_ac": float(motifs),
            }
            history.append(entry)
            print(f"[{step:5d}] r={entry['mean_reward']:+.3f}  "
                  f"tonal={entry['mean_tonal']:.3f}  "
                  f"#pc={entry['mean_pcs']:.1f}  "
                  f"motif_ac={entry['mean_motif_ac']:.3f}")
            if entry["mean_reward"] > best_eval_reward:
                best_eval_reward = entry["mean_reward"]
                best_state = {k: v.clone() for k, v in gen.state_dict().items()}

    if best_state is not None:
        print(f"Best eval reward: {best_eval_reward:+.3f}; "
              "saving best checkpoint")
        gen.load_state_dict(best_state)
    torch.save(gen.state_dict(), out_path / "autoregressive_melody_generator.pt")
    with open(out_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with torch.no_grad():
        sample, _ = gen.sample(256)
    np.save(out_path / "final_melodies.npy", sample.cpu().numpy())
    print(f"\nSaved Phase-11 model + history + samples to {out_path}/")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-notes", type=int, default=16)
    parser.add_argument("--motif-weight", type=float, default=1.0)
    parser.add_argument("--out-dir", type=str,
                        default="results/phase11_autoregressive_melody")
    args = parser.parse_args()
    train_autoregressive_melody(
        n_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        n_notes=args.n_notes,
        motif_weight=args.motif_weight,
        out_dir=args.out_dir,
    )
