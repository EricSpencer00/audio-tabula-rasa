"""Visualize Phase-7 counterpoint training."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.reward.counterpoint import (
    shared_tonal_salience,
    vertical_dissonance,
    voice_crossings,
)


def plot_counterpoint(results_dir="results/phase7_counterpoint"):
    p = Path(results_dir)
    with open(p / "history.json") as f:
        history = json.load(f)
    voices = np.load(p / "final_voices.npy")  # (N, V, T)

    steps = [h["step"] for h in history]
    rewards = [h["mean_reward"] for h in history]
    vert = [h["mean_vert_diss"] for h in history]
    crosses = [h["mean_voice_crossings"] for h in history]
    shared = [h["mean_shared_tonal"] for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    axes[0, 0].plot(steps, rewards, color="C0", lw=2)
    axes[0, 0].set_xlabel("Training step")
    axes[0, 0].set_ylabel("Mean counterpoint reward")
    axes[0, 0].set_title("Reward over training")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(steps, vert, color="C3", lw=2)
    axes[0, 1].set_xlabel("Training step")
    axes[0, 1].set_ylabel("Mean vertical dissonance")
    axes[0, 1].set_title("Harmonic clash over training")
    axes[0, 1].grid(alpha=0.3)

    axes[0, 2].plot(steps, crosses, color="C4", lw=2, label="voice crossings")
    ax2 = axes[0, 2].twinx()
    ax2.plot(steps, shared, color="C2", lw=2, ls="--", label="shared tonal")
    axes[0, 2].set_xlabel("Training step")
    axes[0, 2].set_ylabel("Mean voice crossings", color="C4")
    ax2.set_ylabel("Shared tonal salience", color="C2")
    axes[0, 2].set_title("Voice independence & tonal coherence")
    axes[0, 2].grid(alpha=0.3)

    # Per-sample voice trajectories
    rng = np.random.default_rng(0)
    idx = rng.choice(len(voices), size=4, replace=False)
    n_voices = voices.shape[1]
    colors_per_voice = [plt.cm.tab10(i / max(1, n_voices - 1))
                        for i in range(n_voices)]
    flat_axes = [axes[1, 0], axes[1, 1]]
    for k, i in enumerate(idx[:2]):
        ax = flat_axes[k]
        for vi, voice in enumerate(voices[i]):
            ax.plot(np.log2(voice) * 12, marker="o", lw=2,
                    color=colors_per_voice[vi], label=f"voice {vi}")
        ax.set_xlabel("Note position")
        ax.set_ylabel("Pitch (semitones above A0)")
        ax.set_title(f"Sample counterpoint {i}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    # Vertical-interval distribution (between voices, at each position)
    intervals = []
    for vs in voices:
        for t in range(vs.shape[1]):
            sorted_pitches = np.sort(np.log2(vs[:, t]) * 12)
            for k in range(len(sorted_pitches) - 1):
                intervals.append(sorted_pitches[k + 1] - sorted_pitches[k])
    intervals = np.array(intervals)
    axes[1, 2].hist(intervals, bins=60, color="C2", alpha=0.8)
    axes[1, 2].set_xlim(0, 24)
    axes[1, 2].set_xlabel("Vertical interval (semitones)")
    axes[1, 2].set_ylabel("Count")
    axes[1, 2].set_title("Vertical-interval distribution")
    for st, name in [(0, "unison"), (3, "m3"), (4, "M3"), (5, "P4"),
                     (7, "P5"), (12, "octave")]:
        axes[1, 2].axvline(st, color="gray", ls=":", alpha=0.4)
        axes[1, 2].text(st, axes[1, 2].get_ylim()[1] * 0.95, name,
                        fontsize=7, ha="center", color="gray", rotation=90)
    axes[1, 2].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out_file = p / "counterpoint_summary.png"
    plt.savefig(out_file, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_file}")
    return out_file


if __name__ == "__main__":
    plot_counterpoint()
