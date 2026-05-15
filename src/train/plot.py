"""Visualize training history and the dissonance landscape."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.reward.psychoacoustic import consonance_reward


def plot_results(results_dir: str = "results"):
    p = Path(results_dir)
    with open(p / "history.json") as f:
        history = json.load(f)

    steps = [h["step"] for h in history]
    rewards = [h["mean_reward"] for h in history]
    ratios = [h["median_ratio"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1. Reward curve
    axes[0].plot(steps, rewards, lw=2, color="C0")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Mean consonance reward")
    axes[0].set_title("Reward over training")
    axes[0].grid(alpha=0.3)

    # 2. Median ratio over time, with consonant interval reference lines
    axes[1].plot(steps, ratios, lw=2, color="C1")
    for ratio, name in [(1.0, "unison"), (1.25, "M3"), (1.333, "P4"),
                        (1.5, "P5"), (1.667, "M6"), (2.0, "octave")]:
        axes[1].axhline(ratio, color="gray", ls="--", alpha=0.4)
        axes[1].text(steps[-1] * 1.01, ratio, name, fontsize=8, va="center")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Median frequency ratio")
    axes[1].set_title("Interval discovery trajectory")
    axes[1].grid(alpha=0.3)

    # 3. The Sethares dissonance landscape itself
    base = 220.0
    ratios_grid = np.linspace(1.0, 2.05, 400)
    diss = np.array([-consonance_reward(base, base * r) for r in ratios_grid])
    axes[2].plot(ratios_grid, diss, color="C3", lw=2)
    for ratio, name in [(1.0, "1:1"), (1.2, "6:5"), (1.25, "5:4"),
                        (1.333, "4:3"), (1.5, "3:2"),
                        (1.667, "5:3"), (2.0, "2:1")]:
        axes[2].axvline(ratio, color="gray", ls=":", alpha=0.5)
        axes[2].text(ratio, diss.max() * 1.02, name,
                     fontsize=8, ha="center", rotation=0)
    axes[2].set_xlabel("Frequency ratio")
    axes[2].set_ylabel("Sethares dissonance")
    axes[2].set_title("Reward landscape (lower = more consonant)")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out_file = p / "training_summary.png"
    plt.savefig(out_file, dpi=120, bbox_inches="tight")
    print(f"Saved {out_file}")
    return out_file


if __name__ == "__main__":
    plot_results()
