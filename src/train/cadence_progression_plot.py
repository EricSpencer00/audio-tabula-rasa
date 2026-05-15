"""Plot for Phase-12 cadence-aware chord progressions."""
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.reward.cadence import cadence_arc, expectation_arc
from src.reward.psychoacoustic import chord_dissonance, triad_label


def plot_cadence_progressions(
    results_dir="results/phase12_cadence_progressions",
):
    p = Path(results_dir)
    with open(p / "history.json") as f:
        history = json.load(f)
    seqs = np.load(p / "final_progressions.npy")  # (N, K, V)

    steps = [h["step"] for h in history]
    rewards = [h["mean_reward"] for h in history]
    arc = [h["mean_cadence_arc"] for h in history]
    exp_arc = [h["mean_expectation_arc"] for h in history]
    diss = [h["mean_dissonance"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    axes[0, 0].plot(steps, rewards, color="C0", lw=2)
    axes[0, 0].set_xlabel("Training step")
    axes[0, 0].set_ylabel("Mean reward")
    axes[0, 0].set_title("Reward over training")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(steps, arc, color="C3", lw=2, label="cadence_arc")
    axes[0, 1].plot(steps, exp_arc, color="C2", lw=2, ls="--",
                     label="expectation_arc")
    axes[0, 1].axhline(0, color="gray", ls=":", alpha=0.5)
    axes[0, 1].set_xlabel("Training step")
    axes[0, 1].set_ylabel("Arc")
    axes[0, 1].set_title("Tension-arc and tonal-expectation-arc over training")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(alpha=0.3)

    # Per-chord-position dissonance averaged across samples
    diss_per_pos = np.array([
        [chord_dissonance(c) for c in seq]
        for seq in seqs
    ])
    mean_per_pos = diss_per_pos.mean(axis=0)
    std_per_pos = diss_per_pos.std(axis=0)
    pos = np.arange(1, len(mean_per_pos) + 1)
    axes[1, 0].errorbar(pos, mean_per_pos, yerr=std_per_pos, color="C3",
                          marker="o", lw=2, capsize=4)
    axes[1, 0].set_xlabel("Chord position in progression")
    axes[1, 0].set_ylabel("Mean Sethares dissonance ± std")
    axes[1, 0].set_title("Tension arc: mean dissonance per position")
    axes[1, 0].grid(alpha=0.3)

    # Inventory of chord labels at endpoint vs middle
    n_chords = seqs.shape[1]
    endpoint_labels = []
    middle_labels = []
    for seq in seqs:
        for k, c in enumerate(seq):
            lbl = triad_label(c)
            if k == 0 or k == n_chords - 1:
                endpoint_labels.append(lbl)
            else:
                middle_labels.append(lbl)
    end_counts = Counter(endpoint_labels).most_common(6)
    mid_counts = Counter(middle_labels).most_common(6)
    x = np.arange(max(len(end_counts), len(mid_counts)))
    width = 0.4
    end_names = [n[:12] for n, _ in end_counts]
    mid_names = [n[:12] for n, _ in mid_counts]
    # Show top-6 of each
    axes[1, 1].barh(np.arange(len(end_counts)) - 0.2,
                     [c for _, c in end_counts], height=0.4,
                     color="C2", label="endpoint chords")
    axes[1, 1].barh(np.arange(len(mid_counts)) + 0.2,
                     [c for _, c in mid_counts], height=0.4,
                     color="C3", label="middle chords")
    axes[1, 1].set_yticks(range(max(len(end_counts), len(mid_counts))))
    axes[1, 1].set_yticklabels(
        [f"{e[:14]} | {m[:14]}" for e, m in
         zip([n for n, _ in end_counts] + [""] * 10,
             [n for n, _ in mid_counts] + [""] * 10)][:max(len(end_counts),
                                                            len(mid_counts))],
        fontsize=7,
    )
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("Count")
    axes[1, 1].set_title("Top chord types at endpoints (left bar) vs middle (right bar)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3, axis="x")

    plt.tight_layout()
    out_file = p / "cadence_progression_summary.png"
    plt.savefig(out_file, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_file}")
    return out_file


if __name__ == "__main__":
    plot_cadence_progressions()
