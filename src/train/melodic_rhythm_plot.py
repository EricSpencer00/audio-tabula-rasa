"""Visualize Phase-4.5 joint melodic-rhythm training."""
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.reward.psychoacoustic import (
    implied_fundamental_salience,
    pitch_class_diversity,
)
from src.reward.rhythm import (
    best_period_phase,
    inter_onset_intervals,
    phase_coherence,
)


def plot_melodic_rhythm(results_dir="results/phase4_5_melodic_rhythm"):
    p = Path(results_dir)
    with open(p / "history.json") as f:
        history = json.load(f)
    freqs = np.load(p / "final_freqs.npy")
    onsets = np.load(p / "final_onsets.npy")

    steps = [h["step"] for h in history]
    rewards = [h["mean_reward"] for h in history]
    tonal = [h["mean_tonal"] for h in history]
    pcs = [h["mean_phase_coherence"] for h in history]
    npc = [h["mean_pitch_classes"] for h in history]
    period = [h["mean_period"] for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    axes[0, 0].plot(steps, rewards, color="C0", lw=2)
    axes[0, 0].set_xlabel("Training step")
    axes[0, 0].set_ylabel("Mean joint reward")
    axes[0, 0].set_title("Reward over training")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(steps, tonal, color="C2", lw=2, label="tonal salience")
    ax2 = axes[0, 1].twinx()
    ax2.plot(steps, npc, color="C4", lw=2, ls="--", label="# pitch classes")
    axes[0, 1].set_xlabel("Training step")
    axes[0, 1].set_ylabel("Tonal salience", color="C2")
    ax2.set_ylabel("Mean # pitch classes", color="C4")
    axes[0, 1].set_title("Pitch structure over training")
    axes[0, 1].grid(alpha=0.3)

    axes[0, 2].plot(steps, pcs, color="C3", lw=2, label="phase coherence")
    ax3 = axes[0, 2].twinx()
    ax3.plot(steps, period, color="C5", lw=2, ls="--")
    axes[0, 2].set_xlabel("Training step")
    axes[0, 2].set_ylabel("Phase coherence", color="C3")
    ax3.set_ylabel("Best period (s)", color="C5")
    axes[0, 2].set_title("Rhythm structure over training")
    axes[0, 2].grid(alpha=0.3)

    # Piano roll: pitch vs time for several sampled (melody, rhythm) pairs
    rng = np.random.default_rng(0)
    idx = rng.choice(len(freqs), size=6, replace=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(idx)))
    for k, i in enumerate(idx):
        f = freqs[i]
        o = onsets[i]
        # log2(f) in semitones for visualization
        st = np.log2(f) * 12.0
        axes[1, 0].scatter(o, st + k * 0.5, s=40, color=colors[k], alpha=0.9)
        for ti, si in zip(o, st):
            axes[1, 0].plot([ti, ti], [si + k * 0.5 - 0.1, si + k * 0.5 + 0.1],
                            color=colors[k], lw=2)
    axes[1, 0].set_xlabel("Onset time (s)")
    axes[1, 0].set_ylabel("Pitch (semitones above A0, ↑ offset per sample)")
    axes[1, 0].set_title("Piano-roll view (6 sampled patterns)")
    axes[1, 0].grid(alpha=0.3)

    # IOI histogram
    iois = np.concatenate([inter_onset_intervals(o) for o in onsets])
    axes[1, 1].hist(iois, bins=60, color="C2", alpha=0.8)
    axes[1, 1].set_xlim(0, 1.5)
    axes[1, 1].set_xlabel("Inter-onset interval (s)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("IOI distribution at convergence")
    axes[1, 1].grid(alpha=0.3, axis="y")

    # Pitch-class histogram
    pc_all = (np.log2(freqs.flatten()) * 12.0) % 12.0
    axes[1, 2].hist(pc_all, bins=48, color="C0", alpha=0.8)
    axes[1, 2].set_xlim(0, 12)
    axes[1, 2].set_xlabel("Pitch class (semitones above A)")
    axes[1, 2].set_ylabel("Note count")
    axes[1, 2].set_title("Pitch-class distribution")
    for k in range(12):
        axes[1, 2].axvline(k, color="gray", ls=":", alpha=0.3)
    axes[1, 2].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out_file = p / "melodic_rhythm_summary.png"
    plt.savefig(out_file, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_file}")
    return out_file


if __name__ == "__main__":
    plot_melodic_rhythm()
