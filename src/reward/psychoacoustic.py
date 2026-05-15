"""
Psychoacoustic reward model based on Sethares (1993) dissonance curve,
which itself extends Plomp & Levelt (1965) critical band roughness.

Grounded in physics: roughness arises when two partials fall within a
critical bandwidth of the basilar membrane. Consonant intervals (octave,
fifth, fourth, major third) are NOT hardcoded — they emerge from the
mathematics of harmonic series overlap.

Reference:
  Sethares, W. A. (1993). "Local consonance and the relationship between
  timbre and scale." JASA 94(3), 1218-1228.
"""
import numpy as np


def _sethares_pair(f1: float, f2: float, a1: float = 1.0, a2: float = 1.0) -> float:
    """Roughness between two pure sinusoidal partials."""
    fmin = min(f1, f2)
    fdif = abs(f1 - f2)
    if fdif < 1e-9:
        return 0.0
    s = 0.24 / (0.0207 * fmin + 18.96)
    return a1 * a2 * (np.exp(-3.5 * s * fdif) - np.exp(-5.75 * s * fdif))


def total_dissonance(f1: float, f2: float, n_harmonics: int = 6) -> float:
    """
    Compute total roughness between two complex tones each having
    `n_harmonics` harmonic partials with 1/n amplitude rolloff.

    This is where consonance structure emerges: simple integer ratios
    cause harmonics to coincide rather than collide, yielding low roughness.
    """
    diss = 0.0
    for i in range(1, n_harmonics + 1):
        for j in range(1, n_harmonics + 1):
            amp_i = 1.0 / i
            amp_j = 1.0 / j
            diss += _sethares_pair(i * f1, j * f2, amp_i, amp_j)
    return diss


def consonance_reward(f1: float, f2: float, n_harmonics: int = 6) -> float:
    """
    Reward = negative dissonance. Higher = more consonant.
    Frequencies in Hz. Typical musical range: 80–2000 Hz.
    """
    return -total_dissonance(f1, f2, n_harmonics)


def ratio_label(f1: float, f2: float, tol: float = 0.02) -> str:
    """Diagnostic helper: identify what interval (if any) was learned."""
    if f1 < 1 or f2 < 1:
        return "invalid"
    r = max(f1, f2) / min(f1, f2)
    named = {
        1.000: "unison",
        1.067: "minor_second",
        1.125: "major_second",
        1.200: "minor_third",
        1.250: "major_third",
        1.333: "perfect_fourth",
        1.414: "tritone",
        1.500: "perfect_fifth",
        1.600: "minor_sixth",
        1.667: "major_sixth",
        1.800: "minor_seventh",
        1.875: "major_seventh",
        2.000: "octave",
    }
    best = min(named, key=lambda k: abs(k - r))
    if abs(best - r) / best < tol:
        return named[best]
    return f"non-musical (ratio={r:.3f})"
