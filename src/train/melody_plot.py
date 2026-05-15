"""Visualize Phase-3 melody-training history and pitch-class structure."""
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.reward.psychoacoustic import (
    implied_fundamental_salience,
    pitch_class_diversity,
    sequential_consonance,
)


def plot_melodies(results_dir="results/phase3_melodies"):
    p = Path(results_dir)
    with open(p / "history.json") as f:
        history = json.load(f)
    final = np.load(p / "final_melodies.npy")     # (N, n_notes)

    steps = [h["step"] for h in history]
    rewards = [h["mean_reward"] for h in history]
    tonal = [h["mean_tonal_salience"] for h in history]
    diss = [h["mean_dissonance"] for h in history]
    npc = [h["mean_pitch_classes"] for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    axes[0, 0].plot(steps, rewards, color="C0", lw=2)
    axes[0, 0].set_xlabel("Training step")
    axes[0, 0].set_ylabel("Mean melody reward")
    axes[0, 0].set_title("Reward over training")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(steps, tonal, color="C2", lw=2)
    axes[0, 1].set_xlabel("Training step")
    axes[0, 1].set_ylabel("Implied-fundamental salience")
    axes[0, 1].set_title("Tonal coherence over training")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].axhline(0.5, color="gray", ls="--", alpha=0.5)
    axes[0, 1].text(steps[-1] * 0.6, 0.51, "uniform-pitch baseline",
                    fontsize=8, color="gray")

    axes[0, 2].plot(steps, diss, color="C3", lw=2, label="seq. dissonance")
    ax2 = axes[0, 2].twinx()
    ax2.plot(steps, npc, color="C4", lw=2, ls="--", label="# pitch classes")
    axes[0, 2].set_xlabel("Training step")
    axes[0, 2].set_ylabel("Sum of sequential Sethares", color="C3")
    ax2.set_ylabel("Mean # pitch classes", color="C4")
    axes[0, 2].set_title("Dissonance and PC diversity")
    axes[0, 2].grid(alpha=0.3)

    # Pitch-class histogram across all final melodies
    pc_all = (np.log2(final.flatten()) * 12.0) % 12.0
    axes[1, 0].hist(pc_all, bins=48, color="C0", alpha=0.75)
    axes[1, 0].set_xlabel("Pitch class (semitones above A)")
    axes[1, 0].set_ylabel("Note count")
    axes[1, 0].set_title("Pitch-class distribution at convergence")
    axes[1, 0].set_xlim(0, 12)
    for k in range(12):
        axes[1, 0].axvline(k, color="gray", ls=":", alpha=0.3)
    axes[1, 0].grid(alpha=0.3, axis="y")

    # Pitch-class diversity histogram per melody
    diversities = [pitch_class_diversity(m) for m in final]
    counts = Counter(diversities)
    keys = sorted(counts)
    axes[1, 1].bar(keys, [counts[k] for k in keys], color="C2", alpha=0.85)
    axes[1, 1].set_xlabel("Distinct pitch classes per melody")
    axes[1, 1].set_ylabel(f"Count (of {len(final)} melodies)")
    axes[1, 1].set_title("Scale size discovered")
    axes[1, 1].set_xticks(range(0, 9))
    axes[1, 1].grid(alpha=0.3, axis="y")

    # Sample 8 melodies overlaid as line plots (log-frequency)
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(final), size=8, replace=False)
    for idx in sample_idx:
        axes[1, 2].plot(np.log2(final[idx]) * 12, marker="o",
                        markersize=4, alpha=0.7)
    axes[1, 2].set_xlabel("Note position")
    axes[1, 2].set_ylabel("log₂(f) · 12  (semitones above A0)")
    axes[1, 2].set_title("Sample melodies (8 random)")
    axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()
    out_file = p / "melody_summary.png"
    plt.savefig(out_file, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_file}")
    return out_file


if __name__ == "__main__":
    plot_melodies()
