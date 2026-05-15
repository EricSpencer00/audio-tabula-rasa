"""Visualize Phase-2 training history and the chord-tone reward landscape."""
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.reward.psychoacoustic import (
    chord_dissonance,
    triad_label,
    voice_leading_cost,
)


_REFERENCE_RATIOS = [
    (1.000, 1.000, "1:1:1"),     # unison
    (1.250, 1.500, "4:5:6 maj"),
    (1.200, 1.500, "10:12:15 min"),
    (1.260, 1.587, "aug"),
    (1.189, 1.414, "dim"),
    (1.333, 1.500, "6:8:9 sus4"),
    (1.125, 1.500, "8:9:12 sus2"),
]


def _scatter_chords(ax, freqs_2d, title):
    """freqs_2d: (N, 3) frequencies; plot the (r1, r2) ratio scatter."""
    f = np.sort(freqs_2d, axis=1)
    r1 = f[:, 1] / f[:, 0]
    r2 = f[:, 2] / f[:, 0]
    ax.scatter(r1, r2, s=8, alpha=0.4, color="C0")
    for x, y, name in _REFERENCE_RATIOS:
        ax.plot(x, y, "rx", ms=10, mew=2)
        ax.annotate(name, (x, y), xytext=(4, 4),
                    textcoords="offset points", fontsize=8, color="darkred")
    ax.set_xlabel("ratio voice2 / voice1")
    ax.set_ylabel("ratio voice3 / voice1")
    ax.set_title(title)
    ax.set_xlim(1.0, 2.1)
    ax.set_ylim(1.0, 4.2)
    ax.grid(alpha=0.3)


def plot_triads(results_dir="results/phase2_triads"):
    p = Path(results_dir)
    with open(p / "history.json") as f:
        history = json.load(f)
    final = np.load(p / "final_chords.npy")

    steps = [h["step"] for h in history]
    rewards = [h["mean_reward"] for h in history]
    diss = [h["mean_dissonance"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].plot(steps, rewards, color="C0", lw=2, label="mean reward")
    ax2 = axes[0].twinx()
    ax2.plot(steps, diss, color="C3", lw=2, ls="--", label="mean dissonance")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Mean chord reward", color="C0")
    ax2.set_ylabel("Mean dissonance", color="C3")
    axes[0].set_title("Triad training progress")
    axes[0].grid(alpha=0.3)

    _scatter_chords(axes[1], final, "Final chord cloud (sorted ratios)")

    # Label distribution
    labels = [triad_label(f) for f in final]
    common = Counter(labels).most_common(8)
    names = [c[0][:18] for c in common]
    counts = [c[1] for c in common]
    axes[2].barh(range(len(common)), counts, color="C2")
    axes[2].set_yticks(range(len(common)))
    axes[2].set_yticklabels(names, fontsize=9)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Count (of 512 samples)")
    axes[2].set_title("Triad inventory at convergence")
    axes[2].grid(alpha=0.3, axis="x")

    plt.tight_layout()
    out_file = p / "triad_summary.png"
    plt.savefig(out_file, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_file}")
    return out_file


def plot_progressions(results_dir="results/phase2_progressions"):
    p = Path(results_dir)
    with open(p / "history.json") as f:
        history = json.load(f)
    final = np.load(p / "final_progressions.npy")  # (N, K, 3)

    steps = [h["step"] for h in history]
    rewards = [h["mean_reward"] for h in history]
    diss = [h["mean_dissonance"] for h in history]
    vl = [h["mean_voice_leading"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(steps, rewards, color="C0", lw=2)
    axes[0, 0].set_xlabel("Training step")
    axes[0, 0].set_ylabel("Mean progression reward")
    axes[0, 0].set_title("Progression reward over training")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(steps, diss, color="C3", lw=2, label="dissonance/chord")
    ax2 = axes[0, 1].twinx()
    ax2.plot(steps, vl, color="C2", lw=2, ls="--", label="voice-leading cost")
    axes[0, 1].set_xlabel("Training step")
    axes[0, 1].set_ylabel("Mean dissonance per chord", color="C3")
    ax2.set_ylabel("Mean voice-leading cost", color="C2")
    axes[0, 1].set_title("Decomposed reward components")
    axes[0, 1].grid(alpha=0.3)

    flat = final.reshape(-1, final.shape[-1])
    _scatter_chords(axes[1, 0], flat, "Chord cloud over all chord positions")

    labels = [triad_label(c) for s in final for c in s]
    common = Counter(labels).most_common(8)
    names = [c[0][:20] for c in common]
    counts = [c[1] for c in common]
    axes[1, 1].barh(range(len(common)), counts, color="C2")
    axes[1, 1].set_yticks(range(len(common)))
    axes[1, 1].set_yticklabels(names, fontsize=9)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel(f"Count (of {final.shape[0] * final.shape[1]} chords)")
    axes[1, 1].set_title("Triad inventory at convergence")
    axes[1, 1].grid(alpha=0.3, axis="x")

    plt.tight_layout()
    out_file = p / "progression_summary.png"
    plt.savefig(out_file, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_file}")
    return out_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["triad", "progression", "both"],
                        default="both")
    args = parser.parse_args()
    if args.mode in ("triad", "both"):
        plot_triads()
    if args.mode in ("progression", "both"):
        plot_progressions()
