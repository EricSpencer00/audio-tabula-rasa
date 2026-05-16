"""
Phase 9 — continuous timbre interpolation sweep.

Train Phase 1's toy interval generator at a series of alpha mix values
between the natural harmonic series (alpha=1.0) and the odd-partials
Bohlen-Pierce timbre (alpha=0.0). At each alpha we save the median
discovered ratio. As alpha drops from 1 to 0 we expect the discovered
ratio to swing continuously from the M6/octave region to the
tritone/P5 region.

Demonstrates that the scale-emergence is not a binary harmonic/non-
harmonic phenomenon — it's a smooth function of how much energy lives
in the even harmonics.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.generator.toy_generator import ToyIntervalGenerator
from src.reward.psychoacoustic import consonance_reward
from src.train.reinforce import compute_rewards, train


def sweep(alphas, n_steps: int = 1500, seeds=(0, 1, 2),
          out_dir: str = "results/phase9_timbre_sweep"):
    """For each α, train one model per seed and report the pooled
    discovered-ratio distribution. With REINFORCE the dissonance
    landscape has multiple shallow minima, so any one seed may pick a
    different one — pooling across seeds gives the right picture of
    where the timbre's actual consonance minima are."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    sweep_results = []
    for alpha in alphas:
        print(f"\n{'='*40}  alpha = {alpha:.2f}  {'='*40}")
        all_ratios = []
        for seed in seeds:
            alpha_dir = out_path / f"alpha_{alpha:.2f}_seed{seed}"
            train(
                n_steps=n_steps,
                seed=seed,
                partials=alpha,
                out_dir=str(alpha_dir),
            )
            gen = ToyIntervalGenerator()
            gen.load_state_dict(torch.load(alpha_dir / "toy_generator.pt"))
            gen.eval()
            torch.manual_seed(seed)
            with torch.no_grad():
                f, _ = gen.sample(512)
            rs = (f.max(dim=1).values / f.min(dim=1).values).cpu().numpy()
            all_ratios.append(rs)
        pooled = np.concatenate(all_ratios)
        entry = {
            "alpha": float(alpha),
            "n_seeds": len(seeds),
            "median_ratio": float(np.median(pooled)),
            "p25_ratio": float(np.percentile(pooled, 25)),
            "p75_ratio": float(np.percentile(pooled, 75)),
            "p10_ratio": float(np.percentile(pooled, 10)),
            "p90_ratio": float(np.percentile(pooled, 90)),
            "min_ratio": float(pooled.min()),
            "max_ratio": float(pooled.max()),
        }
        sweep_results.append(entry)
        print(f"alpha={alpha:.2f}: median {entry['median_ratio']:.3f} "
              f"IQR [{entry['p25_ratio']:.3f}, {entry['p75_ratio']:.3f}] "
              f"P10–P90 [{entry['p10_ratio']:.3f}, {entry['p90_ratio']:.3f}]")

    with open(out_path / "sweep.json", "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSaved sweep results to {out_path}/sweep.json")
    return sweep_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=1500)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-alphas", type=int, default=7)
    parser.add_argument("--out-dir", type=str,
                        default="results/phase9_timbre_sweep")
    args = parser.parse_args()
    alphas = np.linspace(0.0, 1.0, args.n_alphas)
    sweep(alphas, n_steps=args.n_steps,
          seeds=tuple(range(args.n_seeds)),
          out_dir=args.out_dir)
