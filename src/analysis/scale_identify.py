"""
Scale identifier: take a set of pitch classes (semitones mod 12,
clustered) and find the closest standard scale.

Used for *post-hoc* description of Phase-3 melody discoveries. It is
intentionally Western-tuning-centric — the model isn't trained on
scales, but if it happens to land near one of them it's useful to
report which one. For Bohlen-Pierce-trained models (Phase 8), use
`closest_bp_pattern` instead, which compares within the tritave.
"""
from typing import Iterable, Tuple

import numpy as np


# 12-TET scale templates as pitch-class sets (semitones 0..11)
_SCALES_12TET = {
    "major":          {0, 2, 4, 5, 7, 9, 11},
    "natural minor":  {0, 2, 3, 5, 7, 8, 10},
    "dorian":         {0, 2, 3, 5, 7, 9, 10},
    "phrygian":       {0, 1, 3, 5, 7, 8, 10},
    "lydian":         {0, 2, 4, 6, 7, 9, 11},
    "mixolydian":     {0, 2, 4, 5, 7, 9, 10},
    "locrian":        {0, 1, 3, 5, 6, 8, 10},
    "harmonic minor": {0, 2, 3, 5, 7, 8, 11},
    "pent major":     {0, 2, 4, 7, 9},
    "pent minor":     {0, 3, 5, 7, 10},
    "blues":          {0, 3, 5, 6, 7, 10},
    "whole tone":     {0, 2, 4, 6, 8, 10},
    "chromatic":      set(range(12)),
}


def _cluster_pitch_classes(pcs: Iterable[float],
                            tol_cents: float = 50.0) -> np.ndarray:
    """Round/cluster pitch classes to nearest semitone within tol."""
    pcs = np.asarray(pcs, dtype=np.float64) % 12.0
    out = []
    for pc in pcs:
        nearest = int(round(pc)) % 12
        if abs(pc - nearest) * 100 <= tol_cents:
            out.append(nearest)
    return np.unique(out)


def closest_12tet_scale(pcs: Iterable[float],
                        tol_cents: float = 50.0) -> Tuple[str, int, float]:
    """
    Find the closest 12-TET scale + root (a transposition) for a given
    pitch-class set.

    Returns (scale_name, root_semitone, jaccard_score). Higher is better.
    """
    discovered = set(int(p) for p in _cluster_pitch_classes(pcs, tol_cents))
    if not discovered:
        return ("none", 0, 0.0)
    best = ("none", 0, -1.0)
    for name, scale in _SCALES_12TET.items():
        for root in range(12):
            shifted = {(p + root) % 12 for p in scale}
            inter = discovered & shifted
            union = discovered | shifted
            jaccard = len(inter) / len(union)
            if jaccard > best[2]:
                best = (name, root, jaccard)
    return best


def _cluster_log_ratios(ratios: Iterable[float],
                         tol_cents: float = 50.0) -> np.ndarray:
    """Reduce ratios into the (1, tritave) range by multiplication and
    cluster log positions."""
    rs = np.asarray(ratios, dtype=np.float64)
    # Fold into [1, 3) range
    log_t = np.log(3.0)
    folded = np.exp(np.log(rs) % log_t)
    return folded


def closest_bp_pattern(ratios: Iterable[float],
                       tol_cents: float = 50.0) -> Tuple[float]:
    """
    For Bohlen-Pierce-style scales, return the distribution of folded
    log positions modulo the tritave. Useful diagnostic; not a
    classifier.
    """
    folded = _cluster_log_ratios(ratios, tol_cents)
    # Standard BP scale steps (13 ascending steps in a tritave)
    # Just-intoned BP ratios from Heinz Bohlen's original 1972 paper:
    bp_targets = [1.0, 1.080, 1.190, 1.286, 1.4, 1.5, 1.667, 1.8,
                  1.929, 2.077, 2.231, 2.428, 2.625, 3.0]
    # For each folded ratio, find the closest BP target
    counts = {f"{t:.3f}": 0 for t in bp_targets}
    for r in folded:
        best = min(bp_targets, key=lambda t: abs(np.log(t) - np.log(r)))
        if abs(np.log(best) - np.log(r)) * 1200 / np.log(2) < tol_cents:
            counts[f"{best:.3f}"] += 1
    return counts


def melody_scale_distribution(melodies, tol_cents: float = 50.0):
    """For a batch of melodies, identify each one's closest scale and
    return a Counter of (scale_name, root) tuples."""
    from collections import Counter
    out = Counter()
    for m in melodies:
        pcs = (np.log2(m) * 12.0) % 12.0
        name, root, _ = closest_12tet_scale(pcs, tol_cents)
        out[(name, root)] += 1
    return out
