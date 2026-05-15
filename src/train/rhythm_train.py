"""
REINFORCE training loop for Phase 4 — rhythm.

A rhythm generator outputs N onset times via cumulative-sum of a soft-
positive IOI policy. The reward is autocorrelation-peak entrainment in
the musical tempo lag window (Large–Kolen-inspired) plus a sparsity /
min-IOI guard. The model is expected to converge on a roughly periodic
pulse train.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.generator.rhythm_generator import RhythmGenerator
from src.reward.rhythm import (
    autocorr_peak,
    best_period,
    best_period_phase,
    inter_onset_intervals,
    phase_coherence,
    rhythm_reward,
)


def _score_rhythms(onsets: torch.Tensor, duration: float, **kw) -> np.ndarray:
    np_o = onsets.detach().cpu().numpy()
    out = np.empty(np_o.shape[0], dtype=np.float32)
    for i, seq in enumerate(np_o):
        out[i] = rhythm_reward(seq, duration=duration, **kw)
    return out


def _entropy_coef(step: int, n_steps: int,
                  start: float = 0.01, end: float = -0.02) -> float:
    frac = min(1.0, max(0.0, step / max(1, n_steps - 1)))
    return start + (end - start) * frac


def train_rhythms(n_steps=2000, batch_size=64, lr=3e-4, log_every=50,
                  seed=0, n_onsets=8, duration=4.0,
                  entrainment_weight=4.0, diversity_weight=0.5,
                  sparsity_weight=0.5, min_onsets=4, min_ioi=0.07,
                  ent_start=0.01, ent_end=-0.02,
                  out_dir="results/phase4_rhythms"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    gen = RhythmGenerator(latent_dim=16, hidden=128,
                          n_onsets=n_onsets, duration=duration)
    opt = torch.optim.Adam(gen.parameters(), lr=lr)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    history = []
    best_eval_reward = -float("inf")
    best_state = None
    kw = dict(entrainment_weight=entrainment_weight,
              diversity_weight=diversity_weight,
              sparsity_weight=sparsity_weight,
              min_onsets=min_onsets,
              min_ioi=min_ioi)

    for step in range(n_steps):
        onsets, log_prob = gen.sample(batch_size)
        # Recover entropy from the underlying distribution.
        z = torch.randn(batch_size, gen.latent_dim)
        mean_raw, std = gen(z)
        ent_dist = torch.distributions.Normal(mean_raw, std)
        entropy = ent_dist.entropy().sum(dim=-1).mean()

        rewards = torch.tensor(_score_rhythms(onsets, duration, **kw))
        adv = rewards - rewards.mean()

        ent_coef = _entropy_coef(step, n_steps, ent_start, ent_end)
        loss = -(log_prob * adv).mean() - ent_coef * entropy

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
        opt.step()

        if step % log_every == 0 or step == n_steps - 1:
            with torch.no_grad():
                eo, _ = gen.sample(128)
                eo_np = eo.cpu().numpy()
                er = _score_rhythms(eo, duration, **kw)
                pc = np.mean([phase_coherence(o) for o in eo_np])
                ac = np.mean([autocorr_peak(o, duration=duration) for o in eo_np])
                periods = [best_period_phase(o) for o in eo_np]
                iois_all = np.concatenate([inter_onset_intervals(o) for o in eo_np])
            entry = {
                "step": step,
                "mean_reward": float(er.mean()),
                "mean_phase_coherence": float(pc),
                "mean_autocorr": float(ac),
                "mean_best_period": float(np.mean(periods)),
                "median_ioi": float(np.median(iois_all)),
                "std_ioi": float(np.std(iois_all)),
            }
            history.append(entry)
            print(f"[{step:5d}] r={entry['mean_reward']:+.3f}  "
                  f"pc={entry['mean_phase_coherence']:.3f}  "
                  f"ac={entry['mean_autocorr']:.3f}  "
                  f"period={entry['mean_best_period']:.3f}s  "
                  f"ioi_med={entry['median_ioi']:.3f}±{entry['std_ioi']:.3f}")
            if entry["mean_reward"] > best_eval_reward:
                best_eval_reward = entry["mean_reward"]
                best_state = {k: v.clone() for k, v in gen.state_dict().items()}

    if best_state is not None:
        print(f"Best eval reward: {best_eval_reward:+.3f}; "
              "saving best checkpoint")
        gen.load_state_dict(best_state)
    torch.save(gen.state_dict(), out_path / "rhythm_generator.pt")
    with open(out_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    with torch.no_grad():
        sample_o, _ = gen.sample(256)
    np.save(out_path / "final_rhythms.npy", sample_o.cpu().numpy())
    print(f"\nSaved rhythm model + history + sample to {out_path}/")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-onsets", type=int, default=8)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--out-dir", type=str, default="results/phase4_rhythms")
    args = parser.parse_args()
    train_rhythms(
        n_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        n_onsets=args.n_onsets,
        duration=args.duration,
        out_dir=args.out_dir,
    )
