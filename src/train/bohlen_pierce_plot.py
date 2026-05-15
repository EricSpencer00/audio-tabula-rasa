"""Plot the Phase-8 Bohlen-Pierce experiment.

Side-by-side comparison of:
- the dissonance curve under harmonic vs. odd-only partial timbres
- the interval-ratio distribution discovered by REINFORCE under each
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.reward.psychoacoustic import ratio_label, total_dissonance


def _load_history(results_dir: Path):
    with open(results_dir / "history.json") as f:
        return json.load(f)


def _final_ratios(results_dir: Path):
    """Recover the final ratios from the saved generator + samples."""
    try:
        import torch

        from src.generator.toy_generator import ToyIntervalGenerator
        gen = ToyIntervalGenerator()
        gen.load_state_dict(torch.load(results_dir / "toy_generator.pt"))
        gen.eval()
        torch.manual_seed(0)
        with torch.no_grad():
            f, _ = gen.sample(512)
        f = f.cpu().numpy()
        return (f.max(axis=1) / f.min(axis=1))
    except FileNotFoundError:
        return None


def plot_comparison(harmonic_dir="results",
                    odd_dir="results/phase8_bohlen_pierce",
                    out_dir="results/phase8_bohlen_pierce"):
    harmonic_dir = Path(harmonic_dir)
    odd_dir = Path(odd_dir)
    out_path = Path(out_dir)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    # Top: dissonance curves
    F0 = 220.0
    ratios_grid = np.linspace(1.0, 3.1, 600)
    for col, partials in enumerate(["harmonic", "odd"]):
        diss = np.array([total_dissonance(F0, F0 * r, partials=partials)
                         for r in ratios_grid])
        axes[0, col].plot(ratios_grid, diss, color="C3", lw=2)
        axes[0, col].set_xlabel("Frequency ratio")
        axes[0, col].set_ylabel("Sethares dissonance")
        axes[0, col].set_title(f"Reward landscape — {partials} partials")
        axes[0, col].grid(alpha=0.3)
        # Mark canonical intervals
        if partials == "harmonic":
            marks = [(1.0, "1:1"), (1.25, "5:4"), (1.333, "4:3"),
                     (1.5, "3:2"), (1.667, "5:3"), (2.0, "2:1"),
                     (2.5, "5:2"), (3.0, "3:1")]
        else:
            # Odd-partial / Bohlen-Pierce relevant ratios
            marks = [(1.0, "1:1"), (1.286, "9:7"), (1.4, "7:5"),
                     (1.5, "3:2"), (1.667, "5:3"), (1.857, "13:7"),
                     (2.143, "15:7"), (2.333, "7:3"), (3.0, "3:1")]
        for r, name in marks:
            axes[0, col].axvline(r, color="gray", ls=":", alpha=0.5)
            axes[0, col].text(r, diss.max() * 1.02, name,
                              fontsize=7, ha="center", rotation=0,
                              color="gray")

    # Bottom: discovered ratio distributions
    for col, (label, src) in enumerate([
        ("harmonic timbre", harmonic_dir),
        ("odd-partial timbre", odd_dir),
    ]):
        rs = _final_ratios(src)
        if rs is None:
            axes[1, col].text(0.5, 0.5,
                              f"missing {src/'toy_generator.pt'}",
                              ha="center", va="center",
                              transform=axes[1, col].transAxes)
            continue
        axes[1, col].hist(rs, bins=80, range=(1.0, 3.1),
                          color="C0", alpha=0.85)
        axes[1, col].set_xlim(1.0, 3.1)
        axes[1, col].set_xlabel("Discovered frequency ratio")
        axes[1, col].set_ylabel("Sample count (of 512)")
        axes[1, col].set_title(f"Discovered intervals — {label}")
        axes[1, col].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out_file = out_path / "bohlen_pierce_summary.png"
    plt.savefig(out_file, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_file}")
    return out_file


if __name__ == "__main__":
    plot_comparison()
