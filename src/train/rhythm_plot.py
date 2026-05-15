"""Visualize Phase-4 rhythm training and the entrainment landscape."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.reward.rhythm import (
    autocorr_peak,
    best_period_phase,
    inter_onset_intervals,
    phase_coherence,
)


def plot_rhythms(results_dir="results/phase4_rhythms"):
    p = Path(results_dir)
    with open(p / "history.json") as f:
        history = json.load(f)
    final = np.load(p / "final_rhythms.npy")  # (N, n_onsets)

    steps = [h["step"] for h in history]
    rewards = [h["mean_reward"] for h in history]
    pcs = [h["mean_phase_coherence"] for h in history]
    periods = [h["mean_best_period"] for h in history]
    ioi_med = [h["median_ioi"] for h in history]
    ioi_std = [h["std_ioi"] for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    axes[0, 0].plot(steps, rewards, color="C0", lw=2)
    axes[0, 0].set_xlabel("Training step")
    axes[0, 0].set_ylabel("Mean rhythm reward")
    axes[0, 0].set_title("Reward over training")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(steps, pcs, color="C2", lw=2)
    axes[0, 1].axhline(1.0, color="gray", ls="--", alpha=0.5)
    axes[0, 1].axhline(0.5, color="gray", ls=":", alpha=0.5)
    axes[0, 1].text(steps[-1] * 0.6, 0.52, "uniform-random baseline",
                    fontsize=8, color="gray")
    axes[0, 1].set_xlabel("Training step")
    axes[0, 1].set_ylabel("Phase coherence")
    axes[0, 1].set_title("Periodicity over training")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].grid(alpha=0.3)

    axes[0, 2].plot(steps, periods, color="C3", lw=2, label="argmax period")
    ax2 = axes[0, 2].twinx()
    ax2.plot(steps, ioi_med, color="C4", lw=2, ls="--", label="median IOI")
    ax2.fill_between(steps,
                     np.array(ioi_med) - np.array(ioi_std),
                     np.array(ioi_med) + np.array(ioi_std),
                     color="C4", alpha=0.15)
    axes[0, 2].set_xlabel("Training step")
    axes[0, 2].set_ylabel("Best period (s)", color="C3")
    ax2.set_ylabel("Median IOI ± std (s)", color="C4")
    axes[0, 2].set_title("Tempo discovered")
    axes[0, 2].grid(alpha=0.3)

    # Onset raster for a sample of final rhythms
    rng = np.random.default_rng(0)
    idx = rng.choice(len(final), size=16, replace=False)
    for k, i in enumerate(idx):
        axes[1, 0].scatter(final[i], [k] * final.shape[1],
                           s=20, color="C0", alpha=0.8)
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Sample rhythm #")
    axes[1, 0].set_title("Onset raster (16 sampled rhythms)")
    axes[1, 0].set_xlim(0, 4.0)
    for t in np.arange(0, 4.0, 0.5):
        axes[1, 0].axvline(t, color="gray", ls=":", alpha=0.3)
    axes[1, 0].grid(alpha=0.3, axis="x")

    # IOI histogram
    all_iois = np.concatenate([inter_onset_intervals(r) for r in final])
    axes[1, 1].hist(all_iois, bins=60, color="C2", alpha=0.8)
    axes[1, 1].set_xlabel("Inter-onset interval (s)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("IOI distribution at convergence")
    axes[1, 1].set_xlim(0, 1.5)
    axes[1, 1].grid(alpha=0.3, axis="y")

    # Period distribution
    best_periods = [best_period_phase(r) for r in final]
    axes[1, 2].hist(best_periods, bins=40, color="C3", alpha=0.8)
    axes[1, 2].set_xlabel("Best period from phase-coherence search (s)")
    axes[1, 2].set_ylabel(f"Count (of {len(final)} rhythms)")
    axes[1, 2].set_title("Discovered tempo")
    for bpm in [60, 90, 120, 180]:
        T = 60.0 / bpm
        axes[1, 2].axvline(T, color="gray", ls=":", alpha=0.4)
        axes[1, 2].text(T, axes[1, 2].get_ylim()[1] * 0.95,
                        f"{bpm}", fontsize=8, ha="center", color="gray")
    axes[1, 2].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out_file = p / "rhythm_summary.png"
    plt.savefig(out_file, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_file}")
    return out_file


if __name__ == "__main__":
    plot_rhythms()
