"""
REINFORCE training loop for the toy interval generator.

This is the simplest possible policy gradient — no critic, no PPO clipping,
just baseline-subtracted reward. The point is to demonstrate that consonant
intervals emerge from psychoacoustic reward alone, with no music data.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.generator.toy_generator import ToyIntervalGenerator
from src.reward.psychoacoustic import consonance_reward, ratio_label


def compute_rewards(freqs: torch.Tensor) -> torch.Tensor:
    """Score each (f1, f2) pair using psychoacoustic reward model."""
    rewards = []
    for f in freqs.detach().cpu().numpy():
        rewards.append(consonance_reward(float(f[0]), float(f[1])))
    return torch.tensor(rewards, dtype=torch.float32)


def train(
    n_steps: int = 2000,
    batch_size: int = 64,
    lr: float = 1e-3,
    log_every: int = 50,
    seed: int = 0,
    out_dir: str = "results",
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    gen = ToyIntervalGenerator()
    opt = torch.optim.Adam(gen.parameters(), lr=lr)

    history = []
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for step in range(n_steps):
        freqs, log_prob = gen.sample(batch_size)
        rewards = compute_rewards(freqs)

        # baseline = batch mean (variance reduction)
        advantage = rewards - rewards.mean()
        advantage = advantage / (rewards.std() + 1e-8)

        loss = -(log_prob * advantage).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
        opt.step()

        if step % log_every == 0 or step == n_steps - 1:
            with torch.no_grad():
                eval_freqs, _ = gen.sample(256)
                eval_rewards = compute_rewards(eval_freqs).numpy()
                ratios = (eval_freqs.max(dim=1).values
                          / eval_freqs.min(dim=1).values).numpy()
                labels = [ratio_label(float(f[0]), float(f[1]))
                          for f in eval_freqs.numpy()]
                from collections import Counter
                top = Counter(labels).most_common(3)
            entry = {
                "step": step,
                "mean_reward": float(eval_rewards.mean()),
                "median_ratio": float(np.median(ratios)),
                "top_intervals": top,
            }
            history.append(entry)
            print(f"[{step:5d}] reward={entry['mean_reward']:+.3f}  "
                  f"median_ratio={entry['median_ratio']:.3f}  "
                  f"top={top}")

    # Save artifacts
    torch.save(gen.state_dict(), out_path / "toy_generator.pt")
    with open(out_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved model and history to {out_path}/")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()
    train(
        n_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        out_dir=args.out_dir,
    )
