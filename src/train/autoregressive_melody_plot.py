"""Visualize Phase-11 autoregressive melody training and motifs."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _motif_ac_curve(log_freqs, min_lag=1, max_lag=None):
    lf = np.asarray(log_freqs, dtype=np.float64)
    lf = lf - lf.mean()
    if lf.std() < 1e-6:
        return None
    n = len(lf)
    if max_lag is None:
        max_lag = n // 2
    out = []
    for lag in range(min_lag, max_lag + 1):
        a = lf[:-lag]
        b = lf[lag:]
        out.append(np.mean(a * b) / (lf.var() + 1e-8))
    return np.array(out)


def plot_autoregressive_melody(results_dir="results/phase11_autoregressive_melody"):
    p = Path(results_dir)
    with open(p / "history.json") as f:
        history = json.load(f)
    melodies = np.load(p / "final_melodies.npy")

    steps = [h["step"] for h in history]
    rewards = [h["mean_reward"] for h in history]
    tonal = [h["mean_tonal"] for h in history]
    pcs = [h["mean_pcs"] for h in history]
    motifs = [h["mean_motif_ac"] for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    axes[0, 0].plot(steps, rewards, color="C0", lw=2)
    axes[0, 0].set_xlabel("Training step")
    axes[0, 0].set_ylabel("Mean reward")
    axes[0, 0].set_title("Reward over training")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(steps, motifs, color="C3", lw=2)
    axes[0, 1].set_xlabel("Training step")
    axes[0, 1].set_ylabel("Motif autocorrelation (max non-zero lag)")
    axes[0, 1].set_title("Motif structure over training")
    axes[0, 1].grid(alpha=0.3)

    axes[0, 2].plot(steps, tonal, color="C2", lw=2, label="tonal")
    ax2 = axes[0, 2].twinx()
    ax2.plot(steps, pcs, color="C4", lw=2, ls="--", label="#pc")
    axes[0, 2].set_xlabel("Training step")
    axes[0, 2].set_ylabel("Tonal salience", color="C2")
    ax2.set_ylabel("Mean # pitch classes", color="C4")
    axes[0, 2].set_title("Pitch structure over training")
    axes[0, 2].grid(alpha=0.3)

    # Sample melodies
    rng = np.random.default_rng(0)
    idx = rng.choice(len(melodies), size=6, replace=False)
    for k, i in enumerate(idx):
        m = melodies[i]
        axes[1, 0].plot(np.log2(m) * 12, marker="o", markersize=4,
                         lw=1.5, alpha=0.85)
    axes[1, 0].set_xlabel("Note position")
    axes[1, 0].set_ylabel("Pitch (semitones above A0)")
    axes[1, 0].set_title("6 sample melodies (full autoregressive run)")
    axes[1, 0].grid(alpha=0.3)

    # Autocorrelation curves for several samples
    for k, i in enumerate(idx):
        m = melodies[i]
        curve = _motif_ac_curve(np.log2(m))
        if curve is None:
            continue
        axes[1, 1].plot(np.arange(1, len(curve) + 1), curve, lw=1.5, alpha=0.7)
    axes[1, 1].axhline(0, color="gray", ls="--", alpha=0.5)
    axes[1, 1].set_xlabel("Lag (notes)")
    axes[1, 1].set_ylabel("Normalized autocorrelation")
    axes[1, 1].set_title("Per-sample autocorrelation of log-pitch")
    axes[1, 1].grid(alpha=0.3)

    # Pitch class histogram
    pc_all = (np.log2(melodies.flatten()) * 12.0) % 12.0
    axes[1, 2].hist(pc_all, bins=48, color="C0", alpha=0.85)
    axes[1, 2].set_xlim(0, 12)
    axes[1, 2].set_xlabel("Pitch class (semitones above A)")
    axes[1, 2].set_ylabel("Note count")
    axes[1, 2].set_title("Pitch-class distribution")
    for k in range(12):
        axes[1, 2].axvline(k, color="gray", ls=":", alpha=0.3)
    axes[1, 2].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out_file = p / "autoregressive_melody_summary.png"
    plt.savefig(out_file, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_file}")
    return out_file


if __name__ == "__main__":
    plot_autoregressive_melody()
