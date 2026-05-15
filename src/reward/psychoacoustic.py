"""
Psychoacoustic reward model based on Sethares (1993) dissonance curve,
which itself extends Plomp & Levelt (1965) critical band roughness.

Grounded in physics: roughness arises when two partials fall within a
critical bandwidth of the basilar membrane. Consonant intervals (octave,
fifth, fourth, major third) are NOT hardcoded — they emerge from the
mathematics of harmonic series overlap.

References:
  Plomp, R. & Levelt, W. J. M. (1965). "Tonal consonance and critical
  bandwidth." JASA 38, 548-560.
  Sethares, W. A. (1993). "Local consonance and the relationship between
  timbre and scale." JASA 94(3), 1218-1228.
"""
from itertools import permutations

import numpy as np


def _sethares_pair(f1: float, f2: float, a1: float = 1.0, a2: float = 1.0) -> float:
    """Roughness between two pure sinusoidal partials."""
    fmin = min(f1, f2)
    fdif = abs(f1 - f2)
    if fdif < 1e-9:
        return 0.0
    s = 0.24 / (0.0207 * fmin + 18.96)
    return a1 * a2 * (np.exp(-3.5 * s * fdif) - np.exp(-5.75 * s * fdif))


def total_dissonance(f1: float, f2: float, n_harmonics: int = 6,
                     partials: str = "harmonic") -> float:
    """
    Compute total roughness between two complex tones whose partials
    are drawn from a configurable timbre.

    `partials` selects the partial layout:
      "harmonic":  k = 1, 2, 3, ..., n_harmonics
                   — the natural vibrating-string / open-pipe spectrum.
                   This is where octave-based consonance comes from:
                   simple integer ratios cause harmonics to coincide.
      "odd":       k = 1, 3, 5, ..., 2*n_harmonics-1
                   — odd-only partials like a square wave or a clarinet's
                   low register. Bohlen-Pierce showed that this timbre
                   has its consonance minima at *tritave* (3:1) intervals
                   and odd-ratio subdivisions, not octave-based ratios.
      "inharmonic": k = 1, sqrt(2), 2, sqrt(8), ..., a stretched series
                   that does not have any integer relations — used as
                   a control to verify that consonance structure depends
                   on the partial layout, not just on having multiple
                   partials.

    Amplitude rolloff is 1/k for "harmonic" and "odd" (preserving the
    visual amplitude of the corresponding harmonic series), and 1/k for
    "inharmonic" too.
    """
    if partials == "harmonic":
        ks = np.arange(1, n_harmonics + 1, dtype=np.float64)
    elif partials == "odd":
        ks = np.arange(1, 2 * n_harmonics, 2, dtype=np.float64)
    elif partials == "inharmonic":
        # Powers of sqrt(2): 1, 1.414, 2, 2.828, 4, ...  — no integer rels
        ks = 2.0 ** (np.arange(n_harmonics, dtype=np.float64) / 2.0)
    else:
        raise ValueError(f"unknown partials={partials!r}")

    amps = 1.0 / ks
    diss = 0.0
    for i, ki in enumerate(ks):
        for j, kj in enumerate(ks):
            diss += _sethares_pair(ki * f1, kj * f2, amps[i], amps[j])
    return diss


def consonance_reward(f1: float, f2: float, n_harmonics: int = 6,
                      partials: str = "harmonic") -> float:
    """
    Reward = negative dissonance. Higher = more consonant.
    Frequencies in Hz. Typical musical range: 80–2000 Hz.
    `partials` selects the timbre: "harmonic", "odd", or "inharmonic".
    """
    return -total_dissonance(f1, f2, n_harmonics, partials=partials)


def chord_dissonance(freqs, n_harmonics: int = 6) -> float:
    """
    Total Sethares dissonance over all pairs of voices in a chord.
    `freqs` is a 1D iterable of fundamental frequencies in Hz.

    A 3-note chord has 3 pairs (C(3,2)). For 4 voices, 6 pairs. Each
    pair contributes its own n_harmonics x n_harmonics partial roughness.

    By summing pairwise roughness we keep the model purely additive in
    the physical sense: each pair of voices excites the basilar membrane
    independently. There is no chord-level prior.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    n = len(freqs)
    diss = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            diss += total_dissonance(float(freqs[i]), float(freqs[j]), n_harmonics)
    return diss


def voice_spread_penalty(freqs, min_semitones: float = 1.5) -> float:
    """
    Penalty for voices collapsing onto each other.

    Sethares dissonance is zero at exact unison (the partials of two
    identical tones never collide — they superpose perfectly). That
    means a "3-note chord" of three identical pitches would score as
    the most consonant chord possible, which trivializes the task of
    *producing a chord*. We require the K voices to occupy distinct
    log-frequency positions: anything closer than `min_semitones`
    incurs a hinge penalty proportional to the gap deficit.

    This is a constraint on the *task* (output 3 distinct pitches), not
    on perception. We keep it explicit and minimal.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    f_sorted = np.sort(freqs)
    log_ratios = np.log2(f_sorted[1:] / f_sorted[:-1]) * 12.0  # semitones
    gap_deficit = np.clip(min_semitones - log_ratios, 0.0, None)
    return float(np.sum(gap_deficit ** 2))


def voice_leading_cost(chord_a, chord_b) -> float:
    """
    Minimum total log-frequency movement when assigning voices in
    chord A to voices in chord B. Measured in octaves^2 so that
    a one-octave leap costs the same regardless of register.

    For each permutation σ of B, compute Σ_i (log2(A_i) - log2(B_σ(i)))^2
    and take the minimum. This is the assignment-problem definition of
    voice leading used in transformational music theory (Tymoczko 2006).

    Both chords are assumed to have the same number of voices.
    """
    a = np.log2(np.asarray(chord_a, dtype=np.float64))
    b = np.log2(np.asarray(chord_b, dtype=np.float64))
    if len(a) != len(b):
        raise ValueError("chord_a and chord_b must have the same number of voices")
    best = float("inf")
    for perm in permutations(range(len(b))):
        cost = float(np.sum((a - b[list(perm)]) ** 2))
        if cost < best:
            best = cost
    return best


def chord_reward(freqs, n_harmonics: int = 6, spread_weight: float = 1.0,
                 min_semitones: float = 1.5) -> float:
    """
    Reward for a single chord: negative dissonance minus a soft spread
    penalty. The spread penalty is a *task constraint* (we want a chord,
    not a unison) and is the only non-Sethares ingredient.
    """
    diss = chord_dissonance(freqs, n_harmonics=n_harmonics)
    spread = voice_spread_penalty(freqs, min_semitones=min_semitones)
    return -diss - spread_weight * spread


def progression_reward(chord_seq, n_harmonics: int = 6,
                       spread_weight: float = 1.0,
                       voice_leading_weight: float = 0.5,
                       min_semitones: float = 1.5) -> float:
    """
    Reward for a sequence of chords:
      Σ chord_reward(c_k)  −  λ_vl · Σ voice_leading_cost(c_k, c_{k+1})

    With λ_vl = 0 this is just per-chord consonance. With λ_vl > 0 the
    generator is also pushed toward smooth voice-leading transitions.
    """
    chord_seq = np.asarray(chord_seq, dtype=np.float64)
    r = 0.0
    for c in chord_seq:
        r += chord_reward(c, n_harmonics=n_harmonics,
                          spread_weight=spread_weight,
                          min_semitones=min_semitones)
    if voice_leading_weight > 0 and len(chord_seq) > 1:
        vl = 0.0
        for k in range(len(chord_seq) - 1):
            vl += voice_leading_cost(chord_seq[k], chord_seq[k + 1])
        r -= voice_leading_weight * vl
    return float(r)


def sequential_consonance(freqs, n_harmonics: int = 6) -> float:
    """
    Sum of Sethares dissonance over consecutive note pairs in a melody.

    A *melodic* (sequential) interval excites the basilar membrane in
    roughly the same way as a harmonic (simultaneous) interval — the
    decaying neural representation of note i overlaps with the onset of
    note i+1 for a few hundred milliseconds. So summed pairwise Sethares
    is a reasonable physics-grounded proxy for melodic smoothness.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    diss = 0.0
    for i in range(len(freqs) - 1):
        diss += total_dissonance(float(freqs[i]), float(freqs[i + 1]), n_harmonics)
    return diss


def implied_fundamental_salience(freqs, n_roots: int = 200,
                                 root_min: float = 27.5,
                                 root_max: float = 110.0,
                                 max_harmonic: int = 16,
                                 sigma_cents: float = 35.0) -> float:
    """
    Terhardt-Parncutt-style virtual-pitch / implied-fundamental salience.

    For each candidate root f_r in [root_min, root_max] Hz, score the set
    of notes by how cleanly each one fits *some* low harmonic of f_r:

        score(f_r) = Σ_i  max_{n=1..N_harm}  exp(-(cents_error)^2 / σ^2)

    where cents_error = 1200 · log2(f_i / (n · f_r)). The salience of the
    note set is max_{f_r} score(f_r) / len(freqs), a number in (0, 1].

    Physical interpretation: a set of notes implies a virtual fundamental
    when they collectively trace out a low-integer harmonic series. This
    is exactly the mechanism underlying *tonality* — a key or scale is
    just a collection of pitches that share a low common implied root.

    Reference:
      Terhardt, E. (1974). "Pitch, consonance, and harmony." JASA 55,
      1061-1069. Parncutt, R. (1989). "Harmony: A psychoacoustical
      approach." Springer.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    if (freqs <= 0).any():
        return 0.0
    sigma = sigma_cents / 1200.0  # convert cents to log2 units
    roots = np.linspace(root_min, root_max, n_roots)
    # Vectorize: for each f_r, find best harmonic fit per note.
    log_f = np.log2(freqs)[None, :]                  # (1, N)
    log_r = np.log2(roots)[:, None]                  # (R, 1)
    # implied harmonic numbers (continuous), clamped to [1, max_harmonic]
    log_n = log_f - log_r                            # (R, N)
    n_cont = 2 ** log_n
    n_int = np.clip(np.round(n_cont), 1, max_harmonic)
    err = np.log2(freqs[None, :] / (n_int * roots[:, None]))  # (R, N)
    fit = np.exp(-(err ** 2) / (sigma ** 2))         # (R, N)
    score_per_root = fit.sum(axis=1)                  # (R,)
    return float(score_per_root.max() / len(freqs))


def melody_step_smoothness(freqs, soft_cap_semitones: float = 12.0) -> float:
    """
    Soft contour cost: zero for any step at or below `soft_cap_semitones`
    (a perfect fifth), quadratic above. Linear absolute value beyond a
    semitone is musically natural — small leaps are fine, just not huge
    ones — and avoiding a target keeps the model from being forced to
    always step in a specific interval.

    We rely on `sequential_consonance` (Sethares between consecutive
    notes) to do most of the "smoothness" work; this term is just a
    soft brake on overly large leaps so the policy doesn't wander too
    far between adjacent notes.
    """
    f = np.asarray(freqs, dtype=np.float64)
    semitones = np.abs(np.log2(f[1:] / f[:-1])) * 12.0
    excess = np.clip(semitones - soft_cap_semitones, 0.0, None) ** 2
    return float(excess.mean())


def pitch_class_diversity(freqs, min_unique: int = 3,
                          cluster_cents: float = 50.0) -> int:
    """
    Number of distinct pitch classes (mod octave) up to `cluster_cents`
    tolerance. Used for diagnostics and as a soft variety constraint.
    """
    pc = (np.log2(np.asarray(freqs, dtype=np.float64)) * 12.0) % 12.0
    pc_sorted = np.sort(pc)
    # treat wrap-around: also consider pc + 12
    gaps = np.diff(np.concatenate([pc_sorted, [pc_sorted[0] + 12.0]]))
    tol = cluster_cents / 100.0
    n_clusters = int((gaps > tol).sum())
    return n_clusters


def melody_reward(freqs, n_harmonics: int = 6,
                  tonal_weight: float = 5.0,
                  contour_weight: float = 0.05,
                  diversity_weight: float = 1.5,
                  min_unique: int = 4) -> float:
    """
    Composite reward for a melody:

      −Σ Sethares(f_i, f_{i+1})           (sequential consonance)
      +λ_tonal · implied_root_salience    (Terhardt virtual pitch)
      −λ_contour · contour_excess         (soft cap on huge leaps)
      −λ_diversity · max(0, min_unique − #unique_pitch_classes)^2

    Sethares between consecutive notes already prefers consonant
    intervals (octave, fifth, fourth, third) over the dissonance peak
    near a semitone, so it serves as the primary smoothness signal.
    The contour term only kicks in for octave-plus leaps; the diversity
    term blocks collapse to all-unison.
    """
    diss = sequential_consonance(freqs, n_harmonics=n_harmonics)
    tonal = implied_fundamental_salience(freqs)
    contour = melody_step_smoothness(freqs)
    n_unique = pitch_class_diversity(freqs)
    diversity_deficit = max(0, min_unique - n_unique) ** 2
    return float(
        -diss
        + tonal_weight * tonal
        - contour_weight * contour
        - diversity_weight * diversity_deficit
    )


_TRIAD_TEMPLATES = {
    "major_4_5_6":      (4.0, 5.0, 6.0),
    "minor_10_12_15":   (10.0, 12.0, 15.0),
    "diminished":       (1.0, 1.18921, 1.41421),    # ~minor third + tritone
    "augmented":        (1.0, 1.25992, 1.58740),    # stacked major thirds (2^(4/12))
    "sus2_8_9_12":      (8.0, 9.0, 12.0),
    "sus4_6_8_9":       (6.0, 8.0, 9.0),
}


def triad_label(freqs, tol: float = 0.03) -> str:
    """Diagnostic: identify a 3-note chord by its sorted ratio pattern."""
    freqs = np.asarray(freqs, dtype=np.float64)
    if (freqs < 1).any():
        return "invalid"
    f_sorted = np.sort(freqs)
    r1 = f_sorted[1] / f_sorted[0]
    r2 = f_sorted[2] / f_sorted[0]
    best_name, best_err = "non-musical", float("inf")
    for name, tpl in _TRIAD_TEMPLATES.items():
        t1 = tpl[1] / tpl[0]
        t2 = tpl[2] / tpl[0]
        err = max(abs(t1 - r1) / t1, abs(t2 - r2) / t2)
        if err < best_err:
            best_err = err
            best_name = name
    if best_err < tol:
        return best_name
    return f"non-musical (r=[{r1:.3f},{r2:.3f}])"


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
