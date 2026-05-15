"""
Quantitative analysis of the trained generators — what musical
structure actually emerged?

Loads each phase's saved generator + sample artifacts and reports
the most informative statistics: interval / chord / scale-size /
tempo / vertical-interval distributions.

This is a *post hoc* analysis — nothing here feeds back into training.
It's how we describe the result for the README and PR body.
"""
import json
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np


# ---------- helpers ---------------------------------------------------

def _maybe_load_npy(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return np.load(path)


def _hist_bucket(values, edges, labels):
    counts = np.histogram(values, bins=edges)[0]
    return list(zip(labels, counts.tolist()))


def _table(rows, headers):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows))
              for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    out = [line, "  ".join("-" * w for w in widths)]
    for r in rows:
        out.append("  ".join(str(v).ljust(w) for v, w in zip(r, widths)))
    return "\n".join(out)


# ---------- per-phase reports ----------------------------------------

def phase1_intervals(results_root="results"):
    from src.reward.psychoacoustic import ratio_label
    print("=" * 60)
    print("Phase 1 — intervals (harmonic timbre)")
    print("=" * 60)
    import torch
    from src.generator.toy_generator import ToyIntervalGenerator
    gen = ToyIntervalGenerator()
    gen.load_state_dict(torch.load(Path(results_root) / "toy_generator.pt"))
    gen.eval()
    torch.manual_seed(0)
    with torch.no_grad():
        f, _ = gen.sample(512)
    rs = (f.max(dim=1).values / f.min(dim=1).values).cpu().numpy()
    labels = [ratio_label(float(p[0]), float(p[1])) for p in f.cpu().numpy()]
    top = Counter(labels).most_common(5)
    print(f"  median ratio: {np.median(rs):.3f}")
    print(f"  top labels:")
    for k, v in top:
        print(f"    {v:4d}  {k}")
    print()


def phase8_intervals_odd(results_root="results/phase8_bohlen_pierce"):
    from src.reward.psychoacoustic import ratio_label
    print("=" * 60)
    print("Phase 8 — intervals (ODD-partial timbre)")
    print("=" * 60)
    import torch
    from src.generator.toy_generator import ToyIntervalGenerator
    gen = ToyIntervalGenerator()
    gen.load_state_dict(torch.load(Path(results_root) / "toy_generator.pt"))
    gen.eval()
    torch.manual_seed(0)
    with torch.no_grad():
        f, _ = gen.sample(512)
    rs = (f.max(dim=1).values / f.min(dim=1).values).cpu().numpy()
    print(f"  median ratio: {np.median(rs):.3f}")
    # Bohlen-Pierce relevant labels
    bp_targets = [
        (1.0, "1:1"),
        (1.190, "25:21"),
        (1.286, "9:7"),
        (1.4, "7:5"),
        (1.5, "3:2"),
        (1.667, "5:3"),
        (1.8, "9:5"),
        (1.857, "13:7"),
        (2.0, "2:1"),
        (2.143, "15:7"),
        (2.333, "7:3"),
        (2.5, "5:2"),
        (3.0, "3:1 tritave"),
    ]

    def bp_label(r):
        best = min(bp_targets, key=lambda t: abs(t[0] - r))
        if abs(best[0] - r) / best[0] < 0.04:
            return best[1]
        return f"other (r={r:.3f})"

    labels = [bp_label(r) for r in rs]
    print(f"  top BP-style labels:")
    for k, v in Counter(labels).most_common(5):
        print(f"    {v:4d}  {k}")
    print()


def phase2_triads(results_root="results/phase2_triads"):
    from src.reward.psychoacoustic import (chord_dissonance, triad_label,
                                            voice_spread_penalty)
    print("=" * 60)
    print("Phase 2 — triads (harmonic timbre)")
    print("=" * 60)
    chords = _maybe_load_npy(Path(results_root) / "final_chords.npy")
    if chords is None:
        print("  (no final_chords.npy)")
        return
    labels = [triad_label(c) for c in chords]
    top = Counter(labels).most_common(8)
    diss = np.array([chord_dissonance(c) for c in chords])
    sp = np.array([voice_spread_penalty(c) for c in chords])
    print(f"  N samples: {len(chords)}")
    print(f"  mean dissonance: {diss.mean():.3f}")
    print(f"  pct samples with spread penalty > 0.01: "
          f"{(sp > 0.01).mean() * 100:.1f}%")
    print("  top triad labels:")
    for k, v in top:
        print(f"    {v:4d}  {k}")
    print()


def phase8b_triads_odd(results_root="results/phase8b_bp_triads"):
    from src.reward.psychoacoustic import chord_dissonance
    print("=" * 60)
    print("Phase 8b — triads (ODD-partial timbre)")
    print("=" * 60)
    chords = _maybe_load_npy(Path(results_root) / "final_chords.npy")
    if chords is None:
        print("  (no final_chords.npy)")
        return
    sorted_c = np.sort(chords, axis=1)
    r1 = sorted_c[:, 1] / sorted_c[:, 0]
    r2 = sorted_c[:, 2] / sorted_c[:, 0]
    diss_h = np.array([chord_dissonance(c, partials="harmonic") for c in chords])
    diss_o = np.array([chord_dissonance(c, partials="odd") for c in chords])
    print(f"  N samples: {len(chords)}")
    print(f"  mean dissonance (under odd partials): {diss_o.mean():.3f}")
    print(f"  mean dissonance (under harmonic timbre, for comparison): "
          f"{diss_h.mean():.3f}")
    print(f"  median r1: {np.median(r1):.3f},  median r2: {np.median(r2):.3f}")
    print()


def phase3_melodies(results_root="results/phase3_melodies"):
    from src.analysis.scale_identify import melody_scale_distribution
    from src.reward.psychoacoustic import (implied_fundamental_salience,
                                            pitch_class_diversity)
    print("=" * 60)
    print("Phase 3 — melodies")
    print("=" * 60)
    melodies = _maybe_load_npy(Path(results_root) / "final_melodies.npy")
    if melodies is None:
        return
    tonal = np.array([implied_fundamental_salience(m) for m in melodies])
    pcs = np.array([pitch_class_diversity(m) for m in melodies])
    print(f"  N samples: {len(melodies)}")
    print(f"  mean tonal salience: {tonal.mean():.3f}")
    print(f"  PC count distribution: "
          f"{dict(Counter(pcs.tolist()).most_common())}")
    # Scale identification (Western tuning baseline)
    counter = melody_scale_distribution(melodies)
    top = counter.most_common(5)
    print("  closest Western scale matches:")
    for (name, root), n in top:
        print(f"    {n:4d}  {name:18s} root={root}")
    print()


def phase4_rhythms(results_root="results/phase4_rhythms"):
    from src.reward.rhythm import best_period_phase, phase_coherence
    print("=" * 60)
    print("Phase 4 — rhythms")
    print("=" * 60)
    rhythms = _maybe_load_npy(Path(results_root) / "final_rhythms.npy")
    if rhythms is None:
        return
    pcs = np.array([phase_coherence(r) for r in rhythms])
    periods = np.array([best_period_phase(r) for r in rhythms])
    bpm = 60.0 / periods
    print(f"  N samples: {len(rhythms)}")
    print(f"  mean phase coherence: {pcs.mean():.3f}")
    print(f"  median best period: {np.median(periods):.3f}s "
          f"({60/np.median(periods):.1f} BPM)")
    print(f"  IQR period: "
          f"{np.percentile(periods, 25):.3f}–{np.percentile(periods, 75):.3f}s")
    print()


def phase7_counterpoint(results_root="results/phase7_counterpoint"):
    from src.reward.counterpoint import (shared_tonal_salience,
                                          vertical_dissonance,
                                          voice_crossings)
    print("=" * 60)
    print("Phase 7 — counterpoint")
    print("=" * 60)
    voices = _maybe_load_npy(Path(results_root) / "final_voices.npy")
    if voices is None:
        return
    vd = np.array([vertical_dissonance(v) for v in voices])
    vc = np.array([voice_crossings(v) for v in voices])
    st = np.array([shared_tonal_salience(v) for v in voices])
    # Vertical intervals
    all_intervals = []
    for vs in voices:
        for t in range(vs.shape[1]):
            sorted_t = np.sort(np.log2(vs[:, t]) * 12.0)
            for k in range(len(sorted_t) - 1):
                all_intervals.append(sorted_t[k + 1] - sorted_t[k])
    all_intervals = np.array(all_intervals)
    print(f"  N samples: {len(voices)}")
    print(f"  mean vertical dissonance: {vd.mean():.3f}")
    print(f"  mean voice crossings (out of {voices.shape[2]}): {vc.mean():.2f}")
    print(f"  mean shared tonal salience: {st.mean():.3f}")
    print(f"  vertical-interval percentiles: "
          f"25%={np.percentile(all_intervals, 25):.1f}st, "
          f"50%={np.percentile(all_intervals, 50):.1f}st, "
          f"75%={np.percentile(all_intervals, 75):.1f}st")
    print()


def phase8c_intervals_inharmonic(results_root="results/phase8c_inharmonic"):
    from src.reward.psychoacoustic import ratio_label
    print("=" * 60)
    print("Phase 8c — intervals (INHARMONIC partials, negative control)")
    print("=" * 60)
    import torch
    from src.generator.toy_generator import ToyIntervalGenerator
    p = Path(results_root) / "toy_generator.pt"
    if not p.exists():
        print(f"  missing {p}")
        return
    gen = ToyIntervalGenerator()
    gen.load_state_dict(torch.load(p))
    gen.eval()
    torch.manual_seed(0)
    with torch.no_grad():
        f, _ = gen.sample(512)
    rs = (f.max(dim=1).values / f.min(dim=1).values).cpu().numpy()
    labels = [ratio_label(float(p[0]), float(p[1])) for p in f.cpu().numpy()]
    top = Counter(labels).most_common(5)
    print(f"  median ratio: {np.median(rs):.3f}")
    print(f"  top labels (under Western tuning):")
    for k, v in top:
        print(f"    {v:4d}  {k}")
    print()


def phase13_n_voice(results_root: str, n_voices_label: str):
    from src.reward.counterpoint import (shared_tonal_salience,
                                          vertical_dissonance,
                                          voice_crossings)
    print("=" * 60)
    print(f"Phase 13 — {n_voices_label}-voice counterpoint")
    print("=" * 60)
    voices = _maybe_load_npy(Path(results_root) / "final_voices.npy")
    if voices is None:
        print("  (no final_voices.npy)")
        return
    vd = np.array([vertical_dissonance(v) for v in voices])
    vc = np.array([voice_crossings(v) for v in voices])
    st = np.array([shared_tonal_salience(v) for v in voices])
    print(f"  N samples: {len(voices)}")
    print(f"  mean vertical dissonance: {vd.mean():.3f}")
    print(f"  mean voice crossings: {vc.mean():.2f}")
    print(f"  mean shared tonal salience: {st.mean():.3f}")
    print()


def run_all():
    print()
    phase1_intervals()
    phase2_triads()
    phase3_melodies()
    phase4_rhythms()
    phase7_counterpoint()
    phase8_intervals_odd()
    phase8b_triads_odd()
    phase8c_intervals_inharmonic()
    phase13_n_voice("results/phase13_3voice_counterpoint", "3")
    phase13_n_voice("results/phase13_4voice_counterpoint", "4")


if __name__ == "__main__":
    run_all()
