"""Side-by-side comparison of 2-, 3-, and 4-voice counterpoint results."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.reward.counterpoint import (
    shared_tonal_salience,
    vertical_dissonance,
    voice_crossings,
)


def plot_voice_count_comparison(
    sources=(("2 voices", "results/phase7_counterpoint"),
             ("3 voices", "results/phase13_3voice_counterpoint"),
             ("4 voices", "results/phase13_4voice_counterpoint")),
    out_file="results/phase13_voice_count_comparison.png",
):
    fig, axes = plt.subplots(2, len(sources), figsize=(6 * len(sources), 8))
    if len(sources) == 1:
        axes = axes.reshape(2, 1)

    for col, (label, src) in enumerate(sources):
        voices_path = Path(src) / "final_voices.npy"
        if not voices_path.exists():
            for r in (0, 1):
                axes[r, col].text(0.5, 0.5,
                                   f"missing {voices_path}",
                                   ha="center", va="center",
                                   transform=axes[r, col].transAxes)
            continue
        voices = np.load(voices_path)

        # Top row: piano-roll of 4 sampled excerpts (offset)
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(len(voices), size=4, replace=False)
        colors = ["C0", "C3", "C2", "C4"]
        ax = axes[0, col]
        for k, idx in enumerate(sample_idx):
            cp = voices[idx]
            for v, voice in enumerate(cp):
                st = np.log2(voice) * 12.0 + k * 50
                ax.plot(st, marker="o", markersize=4,
                         color=colors[v % len(colors)],
                         lw=1.5, alpha=0.85)
        ax.set_xlabel("Note position")
        ax.set_ylabel("Pitch (semitones, offset per excerpt)")
        ax.set_title(f"{label} — sample excerpts")
        ax.grid(alpha=0.3)

        # Bottom row: vertical-interval histogram
        all_intervals = []
        for vs in voices:
            for t in range(vs.shape[1]):
                sorted_t = np.sort(np.log2(vs[:, t]) * 12.0)
                for kk in range(len(sorted_t) - 1):
                    all_intervals.append(sorted_t[kk + 1] - sorted_t[kk])
        all_intervals = np.array(all_intervals)
        axes[1, col].hist(all_intervals, bins=60, color="C2", alpha=0.8)
        axes[1, col].set_xlim(0, 35)
        axes[1, col].set_xlabel("Vertical interval (semitones)")
        axes[1, col].set_ylabel("Count")
        axes[1, col].set_title(f"{label} — vertical-interval distribution")
        for st, name in [(0, "uni"), (7, "P5"), (12, "8va"), (19, "12th"),
                          (24, "2 8va")]:
            axes[1, col].axvline(st, color="gray", ls=":", alpha=0.4)
            axes[1, col].text(st, axes[1, col].get_ylim()[1] * 0.95,
                              name, fontsize=7, ha="center", color="gray",
                              rotation=90)
        axes[1, col].grid(alpha=0.3, axis="y")

        vd = np.mean([vertical_dissonance(v) for v in voices])
        vc = np.mean([voice_crossings(v) for v in voices])
        st_mean = np.mean([shared_tonal_salience(v) for v in voices])
        axes[0, col].set_xlabel(axes[0, col].get_xlabel()
                                  + f"\n(mean V-diss {vd:.3f}, crossings {vc:.2f}, "
                                  f"shared-tonal {st_mean:.3f})")

    plt.tight_layout()
    out_path = Path(out_file)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


if __name__ == "__main__":
    plot_voice_count_comparison()
