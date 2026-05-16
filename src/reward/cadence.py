"""
Phase-12 cadence reward.

A musical *cadence* is a tension-then-release arc over a chord
progression. The simplest physics-grounded formalization: a cadence
exists when a sequence of chords has higher dissonance in its middle
than at its endpoints. We measure the "arc" as

    arc(c1..cK) = mean(diss(c2), ..., diss(c_{K-1})) − mean(diss(c1), diss(cK))

and reward progressions where this quantity is *positive*. That is,
the cadence rewards starting and ending on lower-dissonance ("more
resolved") chords, and passing through higher-dissonance ones in the
middle.

Combined with the existing Phase-2 progression reward (per-chord
consonance + voice-leading), this should push the model toward an
unambiguous tonic — middle chords serving as "tension" away from a
home harmony at the endpoints.

Notes:
- This is still purely psychoacoustic. We don't impose V-I or any
  Western functional-harmony rule, only the *shape* of tension.
- A pure-consonance progression (all chords low-dissonance) gets a
  near-zero arc bonus and is therefore neither rewarded nor penalized
  by this term; the standard chord_reward handles it.
"""
import numpy as np

from src.reward.psychoacoustic import chord_dissonance


def cadence_arc(chord_seq, n_harmonics: int = 6,
                partials: str = "harmonic") -> float:
    """Return mean(middle dissonance) − mean(endpoint dissonance)."""
    seq = np.asarray(chord_seq, dtype=np.float64)
    if seq.shape[0] < 3:
        return 0.0
    diss = np.array([
        chord_dissonance(c, n_harmonics=n_harmonics, partials=partials)
        for c in seq
    ])
    endpoint = 0.5 * (diss[0] + diss[-1])
    middle = float(np.mean(diss[1:-1]))
    return float(middle - endpoint)


def cadence_reward(chord_seq, cadence_weight: float = 1.0,
                   n_harmonics: int = 6, partials: str = "harmonic") -> float:
    """Bonus proportional to the tension-arc."""
    return cadence_weight * cadence_arc(chord_seq, n_harmonics=n_harmonics,
                                         partials=partials)


def expectation_arc(chord_seq) -> float:
    """
    Tonal-expectation arc: the shared implied-fundamental salience of
    *all* chords (concatenated) minus the shared salience of just the
    middle chords.

    Positive when the first and last chords help anchor a common
    fundamental but the middle chords introduce some tonal ambiguity
    — i.e. the endpoints are "more resolved" in tonal-expectation
    terms. This is a purely psychoacoustic proxy for cadence: it does
    not impose V-I or any functional rule, just the *shape* of the
    arc in implied-fundamental confidence.
    """
    from src.reward.psychoacoustic import implied_fundamental_salience
    seq = np.asarray(chord_seq, dtype=np.float64)
    if seq.shape[0] < 3:
        return 0.0
    flat_all = seq.flatten()
    middle = seq[1:-1].flatten()
    overall = implied_fundamental_salience(flat_all)
    middle_only = implied_fundamental_salience(middle) if len(middle) else 0.0
    return float(overall - middle_only)
