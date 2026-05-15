"""
Side-by-side comparison plot for harmonic-vs-odd triad discovery
(Phase 8b extension of Phase 8).

Loads the two triad checkpoints (Phase 2 harmonic baseline and Phase
8b odd-partial run) and overlays their final chord-ratio clouds.
"""
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.reward.psychoacoustic import chord_dissonance, triad_label


_HARMONIC_TEMPLATES = [
    (1.25, 1.5, "maj 4:5:6"),
    (1.2, 1.5, "min 10:12:15"),
    (1.189, 1.414, "dim"),
    (1.26, 1.587, "aug"),
    (1.333, 1.5, "sus4 6:8:9"),
    (1.125, 1.5, "sus2 8:9:12"),
]
# Bohlen-Pierce / odd-integer chord templates
# Examples from BP theory: 3:5:7, 5:7:9, 5:9:15, 7:9:11, 3:7:9
_BP_TEMPLATES = [
    (5/3, 7/3, "3:5:7"),
    (7/5, 9/5, "5:7:9"),
    (9/5, 3.0, "5:9:15"),
    (9/7, 11/7, "7:9:11"),
    (7/3, 3.0, "3:7:9"),
    (5/3, 3.0, "3:5:9"),
]


def _final_chords(path: Path):
    return np.load(path / "final_chords.npy")


def _scatter(ax, freqs_2d, title, templates):
    f = np.sort(freqs_2d, axis=1)
    r1 = f[:, 1] / f[:, 0]
    r2 = f[:, 2] / f[:, 0]
    ax.scatter(r1, r2, s=8, alpha=0.35, color="C0")
    for x, y, name in templates:
        ax.plot(x, y, "rx", ms=11, mew=2)
        ax.annotate(name, (x, y), xytext=(5, 4),
                    textcoords="offset points", fontsize=9, color="darkred")
    ax.set_xlabel("ratio voice2 / voice1")
    ax.set_ylabel("ratio voice3 / voice1")
    ax.set_title(title)
    ax.set_xlim(1.0, 2.5)
    ax.set_ylim(1.0, 4.2)
    ax.grid(alpha=0.3)


def plot_comparison(harmonic_dir="results/phase2_triads",
                    odd_dir="results/phase8b_bp_triads",
                    out_dir="results/phase8b_bp_triads"):
    harm_path = Path(harmonic_dir)
    odd_path = Path(odd_dir)
    out_path = Path(out_dir)

    harm_chords = _final_chords(harm_path)
    odd_chords = _final_chords(odd_path)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    _scatter(axes[0, 0], harm_chords, "Harmonic timbre — chord cloud",
             _HARMONIC_TEMPLATES)
    _scatter(axes[0, 1], odd_chords, "Odd-partial timbre — chord cloud",
             _BP_TEMPLATES)

    # Bottom row: pairwise dissonance under each timbre, evaluated
    # *on the harmonic chord cloud* and on the BP chord cloud, both with
    # both timbres. This shows the chords each model finds are low-
    # dissonance under their OWN timbre but not the other.
    for col, (label, chords) in enumerate([
        ("Harmonic-trained chords", harm_chords),
        ("Odd-partial-trained chords", odd_chords),
    ]):
        diss_h = np.array([chord_dissonance(c, partials="harmonic")
                           for c in chords])
        diss_o = np.array([chord_dissonance(c, partials="odd")
                           for c in chords])
        axes[1, col].scatter(diss_h, diss_o, s=8, alpha=0.4)
        # Diagonal
        m = max(diss_h.max(), diss_o.max())
        axes[1, col].plot([0, m], [0, m], color="gray", ls="--", alpha=0.5)
        axes[1, col].set_xlabel("dissonance under harmonic timbre")
        axes[1, col].set_ylabel("dissonance under odd-partial timbre")
        axes[1, col].set_title(label)
        axes[1, col].grid(alpha=0.3)

    plt.tight_layout()
    out_file = out_path / "bp_triads_summary.png"
    plt.savefig(out_file, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_file}")
    return out_file


if __name__ == "__main__":
    plot_comparison()
