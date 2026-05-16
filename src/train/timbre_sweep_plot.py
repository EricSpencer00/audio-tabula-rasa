"""Visualize the Phase-9 continuous timbre sweep."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.generator.toy_generator import ToyIntervalGenerator
from src.reward.psychoacoustic import total_dissonance


def plot_sweep(results_dir="results/phase9_timbre_sweep"):
    p = Path(results_dir)
    with open(p / "sweep.json") as f:
        sweep = json.load(f)
    alphas = np.array([s["alpha"] for s in sweep])
    median = np.array([s["median_ratio"] for s in sweep])
    p25 = np.array([s["p25_ratio"] for s in sweep])
    p75 = np.array([s["p75_ratio"] for s in sweep])
    p10 = np.array([s.get("p10_ratio", s["p25_ratio"]) for s in sweep])
    p90 = np.array([s.get("p90_ratio", s["p75_ratio"]) for s in sweep])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Median ratio vs alpha
    axes[0].fill_between(alphas, p10, p90, alpha=0.15, color="C0",
                          label="P10–P90 (pooled over seeds)")
    axes[0].fill_between(alphas, p25, p75, alpha=0.3, color="C0",
                          label="IQR")
    axes[0].plot(alphas, median, "o-", color="C0", lw=2, label="median")
    axes[0].set_xlabel("Even-harmonic weight α  "
                       "(0 = odd partials only, 1 = full harmonic series)")
    axes[0].set_ylabel("Discovered frequency ratio")
    axes[0].set_title("Phase 9: discovered ratio vs timbre")
    axes[0].grid(alpha=0.3)
    for ratio, name in [(1.5, "P5"), (1.667, "M6"), (1.414, "tritone"),
                        (2.0, "octave"), (1.260, "M3")]:
        axes[0].axhline(ratio, color="gray", ls=":", alpha=0.5)
        axes[0].text(alphas[-1] * 1.01, ratio, name, fontsize=8,
                     va="center", color="gray")
    axes[0].legend(fontsize=8)

    # 2. Dissonance landscape at a few alphas (with picks marked)
    F0 = 220.0
    grid = np.linspace(1.0, 3.05, 400)
    for k, alpha in enumerate(alphas):
        diss = np.array([total_dissonance(F0, F0 * r, partials=float(alpha))
                         for r in grid])
        axes[1].plot(grid, diss + k * 0.05, lw=2,
                     color=plt.cm.viridis(alpha),
                     label=f"α={alpha:.2f}")
    axes[1].set_xlabel("Frequency ratio")
    axes[1].set_ylabel("Sethares dissonance (offset per α)")
    axes[1].set_title("Reward landscape across timbres")
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].grid(alpha=0.3)

    # 3. Discovered ratio histograms stacked — pool seeds
    for k, s in enumerate(sweep):
        alpha = s["alpha"]
        rs_pool = []
        for seed in range(s.get("n_seeds", 1)):
            alpha_dir = p / f"alpha_{alpha:.2f}_seed{seed}"
            if not alpha_dir.exists():
                alpha_dir = p / f"alpha_{alpha:.2f}"   # fallback to old layout
            try:
                gen = ToyIntervalGenerator()
                gen.load_state_dict(torch.load(alpha_dir / "toy_generator.pt"))
                gen.eval()
                torch.manual_seed(seed)
                with torch.no_grad():
                    f, _ = gen.sample(512)
                rs = (f.max(dim=1).values / f.min(dim=1).values).cpu().numpy()
                rs_pool.append(rs)
            except FileNotFoundError:
                continue
        if not rs_pool:
            continue
        rs_all = np.concatenate(rs_pool)
        axes[2].hist(rs_all, bins=60, range=(1.0, 3.0),
                     histtype="step", color=plt.cm.viridis(alpha),
                     lw=2, label=f"α={alpha:.2f}")
    axes[2].set_xlim(1.0, 3.0)
    axes[2].set_xlabel("Discovered ratio")
    axes[2].set_ylabel("Sample count (of 512)")
    axes[2].set_title("Discovered-ratio distributions")
    axes[2].legend(fontsize=8, ncol=2)
    axes[2].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out_file = p / "timbre_sweep_summary.png"
    plt.savefig(out_file, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_file}")
    return out_file


if __name__ == "__main__":
    plot_sweep()
