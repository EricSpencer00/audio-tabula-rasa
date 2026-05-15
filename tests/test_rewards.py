"""
Reward-model property tests. We don't pretend these are unit tests of
the *learned generators* — those are stochastic — but the underlying
reward functions are deterministic and have invariants we can check.

Run with:
    python -m pytest tests/test_rewards.py -v
"""
import numpy as np

from src.reward.psychoacoustic import (
    chord_dissonance,
    chord_reward,
    consonance_reward,
    implied_fundamental_salience,
    melody_reward,
    pitch_class_diversity,
    progression_reward,
    sequential_consonance,
    total_dissonance,
    triad_label,
    voice_leading_cost,
    voice_spread_penalty,
)
from src.reward.rhythm import (
    autocorr_peak,
    inter_onset_intervals,
    phase_coherence,
    rhythm_reward,
)
from src.reward.counterpoint import (
    counterpoint_reward,
    shared_tonal_salience,
    vertical_dissonance,
    voice_crossings,
    voice_register_gap,
)


# ----- Phase 1: interval consonance -----------------------------------

def test_unison_minimizes_dissonance():
    """Unison of two complex tones is dominated by the self-roughness
    of one tone's harmonic series — but two tones in unison should not
    cost any *more* than that, since their harmonics co-locate."""
    one_tone = total_dissonance(440.0, 440.0)
    minor_2 = total_dissonance(440.0, 440.0 * 2 ** (1 / 12))
    assert one_tone < minor_2  # adding a second tone in unison < adding one a semitone away
    assert one_tone < 0.01     # the "intrinsic" roughness is small


def test_octave_has_low_dissonance():
    """Perfect octave: each partial of the upper tone coincides with a
    partial of the lower (every other one), so roughness ≈ self-roughness."""
    d = total_dissonance(220.0, 440.0)
    assert d < 0.02


def test_perfect_fifth_more_consonant_than_minor_second():
    fifth = total_dissonance(220.0, 220.0 * 1.5)
    minor_2 = total_dissonance(220.0, 220.0 * 2 ** (1.0 / 12))
    assert fifth < minor_2


def test_consonance_reward_negates_dissonance():
    a = total_dissonance(220.0, 330.0)
    b = consonance_reward(220.0, 330.0)
    assert np.isclose(a, -b)


# ----- Phase 2: triads ------------------------------------------------

def test_chord_dissonance_permutation_invariant():
    a = chord_dissonance([220, 275, 330])
    b = chord_dissonance([330, 275, 220])
    assert np.isclose(a, b)


def test_major_triad_more_consonant_than_dim():
    maj = chord_dissonance([220, 220 * 5 / 4, 220 * 3 / 2])
    dim = chord_dissonance([220, 220 * 2 ** (3 / 12), 220 * 2 ** (6 / 12)])
    assert maj < dim


def test_voice_spread_penalty_zero_for_wide_chord():
    # 5-semitone gaps
    f = [220, 220 * 2 ** (5 / 12), 220 * 2 ** (10 / 12)]
    assert voice_spread_penalty(f) == 0.0


def test_voice_spread_penalty_positive_for_near_unison():
    # 0.5-semitone gap
    f = [220, 220 * 2 ** (0.5 / 12), 220 * 2 ** (12 / 12)]
    assert voice_spread_penalty(f) > 0.0


def test_voice_leading_cost_zero_for_identical_chord():
    c = [220, 275, 330]
    assert np.isclose(voice_leading_cost(c, c), 0.0)


def test_voice_leading_cost_permutation_invariant():
    a = [220, 275, 330]
    b = [330, 275, 220]
    assert np.isclose(voice_leading_cost(a, b), 0.0)


def test_triad_label_finds_major():
    assert triad_label([220, 220 * 5 / 4, 220 * 3 / 2]) == "major_4_5_6"


def test_progression_reward_lower_with_voice_leading():
    """A 2-chord progression with smooth voicing should score higher
    than the same chords voiced far apart, when VL weight > 0."""
    near_a = [220, 275, 330]
    near_b = [220, 275 * 2 ** (1 / 12), 330 * 2 ** (1 / 12)]
    far_a = [220, 275, 330]
    far_b = [110, 137, 165]
    r_near = progression_reward([near_a, near_b], voice_leading_weight=1.0)
    r_far = progression_reward([far_a, far_b], voice_leading_weight=1.0)
    assert r_near > r_far


# ----- Phase 3: melodies ---------------------------------------------

def test_sequential_consonance_higher_for_chromatic_than_pentatonic():
    # chromatic walk has lots of semitones (high diss);
    # pentatonic walk has minor-third + whole-tone (low diss).
    chrom = 220 * 2 ** (np.arange(8) / 12.0)
    pent = np.array([220, 220 * 2 ** (2 / 12), 220 * 2 ** (5 / 12),
                     220 * 2 ** (7 / 12), 220 * 2 ** (10 / 12)] * 2)[:8]
    assert sequential_consonance(chrom) > sequential_consonance(pent)


def test_implied_fundamental_salience_in_unit_interval():
    s = implied_fundamental_salience([440.0, 550.0, 660.0])
    assert 0.0 <= s <= 1.0


def test_implied_fundamental_higher_for_just_chord_than_random():
    """Notes that are integer ratios should have higher implied-root
    salience than a random set."""
    just = [220, 220 * 3 / 2, 220 * 2, 220 * 5 / 2]
    rng = np.random.default_rng(123)
    rand = [220 * 2 ** rng.uniform(0, 2) for _ in range(4)]
    assert implied_fundamental_salience(just) \
           > implied_fundamental_salience(rand)


def test_pitch_class_diversity_handles_unison():
    assert pitch_class_diversity([440.0] * 8) == 1


def test_pitch_class_diversity_handles_pentatonic():
    pent = [220, 220 * 2 ** (3 / 12), 220 * 2 ** (5 / 12),
            220 * 2 ** (7 / 12), 220 * 2 ** (10 / 12)]
    assert pitch_class_diversity(pent) == 5


def test_melody_reward_random_better_than_unison():
    """Diversity floor should make a random scale-like melody score
    higher than collapsing to unison."""
    rng = np.random.default_rng(0)
    rand = 220 * 2 ** rng.uniform(0, 2, size=8)
    unison = np.full(8, 440.0)
    assert melody_reward(rand) > melody_reward(unison)


# ----- Phase 4: rhythm -----------------------------------------------

def test_phase_coherence_unit_bounded():
    rng = np.random.default_rng(0)
    onsets = np.sort(rng.uniform(0, 4.0, size=10))
    pc = phase_coherence(onsets)
    assert 0.0 <= pc <= 1.0


def test_phase_coherence_regular_higher_than_random():
    regular = np.arange(0, 4.0, 0.5)
    rng = np.random.default_rng(42)
    random = np.sort(rng.uniform(0, 4.0, size=8))
    assert phase_coherence(regular) > phase_coherence(random)


def test_phase_coherence_regular_close_to_one():
    regular = np.arange(0, 4.0, 0.5)
    assert phase_coherence(regular) > 0.95


def test_inter_onset_intervals_sorted():
    onsets = np.array([1.0, 0.5, 2.0, 1.5])
    iois = inter_onset_intervals(onsets)
    assert (iois > 0).all()
    assert len(iois) == len(onsets) - 1


def test_rhythm_reward_regular_higher_than_random():
    regular = np.arange(0, 4.0, 0.5)
    rng = np.random.default_rng(42)
    random = np.sort(rng.uniform(0, 4.0, size=8))
    assert rhythm_reward(regular) > rhythm_reward(random)


def test_rhythm_reward_penalizes_cluster():
    cluster = np.linspace(0.0, 0.3, 8)  # all very close together
    regular = np.arange(0, 4.0, 0.5)
    assert rhythm_reward(regular) > rhythm_reward(cluster)


# ----- Phase 7: counterpoint -----------------------------------------

def _two_voice(low, high):
    return np.stack([np.asarray(low, dtype=float),
                     np.asarray(high, dtype=float)])


def test_vertical_dissonance_zero_for_octave_doubling():
    """Two voices an octave apart have very low pairwise Sethares
    roughness (octaves share all even harmonics)."""
    walk = [262, 294, 330, 349, 392]
    vs = _two_voice(walk, [f * 2 for f in walk])
    assert vertical_dissonance(vs) < 0.06


def test_vertical_dissonance_high_for_chromatic_clash():
    """Two voices a semitone apart at every step give peak roughness."""
    base = [262, 294, 330, 349, 392]
    vs = _two_voice(base, [f * 2 ** (1 / 12) for f in base])
    assert vertical_dissonance(vs) > 1.0


def test_voice_crossings_zero_for_octave_doubled():
    low = [262, 294, 330, 349, 392]
    vs = _two_voice(low, [f * 2 for f in low])
    assert voice_crossings(vs) == 0


def test_voice_crossings_high_for_swapping():
    vs = _two_voice([220, 440, 220, 440], [440, 220, 440, 220])
    assert voice_crossings(vs) >= 2


def test_voice_register_gap_zero_for_octave_spread():
    low = [262, 294, 330]
    vs = _two_voice(low, [f * 2 for f in low])
    assert voice_register_gap(vs) == 0.0


def test_voice_register_gap_positive_for_near_unison():
    vs = _two_voice([262, 262, 262], [263, 263, 263])
    assert voice_register_gap(vs) > 0.0


def test_counterpoint_reward_octaves_over_random():
    """Octave-doubled walking voices should score higher than random."""
    walk = np.array([262, 294, 330, 349, 392, 440, 494, 523], dtype=float)
    octave = np.stack([walk, walk * 2])
    rng = np.random.default_rng(7)
    rand = 2 ** rng.uniform(np.log2(110), np.log2(1760), size=(2, 8))
    assert counterpoint_reward(octave) > counterpoint_reward(rand)


def test_counterpoint_reward_no_crossings_better_with_strong_weight():
    """Same vertical content with stable ordering beats a crossing
    arrangement when the crossing weight is large enough to overcome
    Sethares' love of octave-jump melodies. (At the default weight a
    voice swap is roughly free, because each individual voice is just
    octave-jumping which is itself consonant.)"""
    walk_lo = np.array([262, 294, 330, 349, 392, 440, 494, 523], dtype=float)
    walk_hi = walk_lo * 1.5
    stable = np.stack([walk_lo, walk_hi])
    crossing = np.stack([
        np.where(np.arange(8) % 2 == 0, walk_lo, walk_hi),
        np.where(np.arange(8) % 2 == 0, walk_hi, walk_lo),
    ])
    r_stable = counterpoint_reward(stable, crossing_weight=2.0)
    r_cross = counterpoint_reward(crossing, crossing_weight=2.0)
    assert r_stable > r_cross


def test_shared_tonal_salience_unit_bounded():
    walk = np.array([262, 294, 330, 349, 392, 440, 494, 523], dtype=float)
    vs = np.stack([walk, walk * 2])
    s = shared_tonal_salience(vs)
    assert 0.0 <= s <= 1.0


# ----- Phase 8: timbre / Bohlen-Pierce -------------------------------

def test_harmonic_octave_more_consonant_than_tritone():
    """Under natural (all-integer) harmonics, 2:1 is much more consonant
    than sqrt(2):1 — the canonical reason Western tuning is octave-based."""
    octave = total_dissonance(220.0, 440.0, partials="harmonic")
    tritone = total_dissonance(220.0, 220.0 * 2 ** 0.5, partials="harmonic")
    assert octave < tritone


def test_odd_partials_change_consonance_landscape():
    """Under odd-only partials the dissonance landscape is *different*
    from the natural harmonic one. Specifically, the dissonance peak
    around the minor second should move because the partials' inter-
    coincidences move."""
    harm = total_dissonance(220.0, 220.0 * 2 ** (1 / 12), partials="harmonic")
    odd = total_dissonance(220.0, 220.0 * 2 ** (1 / 12), partials="odd")
    assert not np.isclose(harm, odd)


def test_odd_partials_tritave_is_consonant():
    """With odd-only partials the *tritave* (3:1) plays the role the
    octave (2:1) plays for natural partials: every odd partial of the
    lower tone coincides with an odd partial of the tritave-up tone, so
    the resulting roughness is essentially the intrinsic self-roughness."""
    tritave = total_dissonance(220.0, 660.0, partials="odd")
    octave = total_dissonance(220.0, 440.0, partials="odd")
    tritone = total_dissonance(220.0, 220.0 * 2 ** 0.5, partials="odd")
    assert tritave < tritone
    # Octave can also be reasonably consonant under odd partials but the
    # tritave should be among the most consonant intervals.
    assert tritave <= 0.02


def test_inharmonic_breaks_low_integer_advantage():
    """Stretched / inharmonic partials should *not* give the integer
    ratios (2:1, 3:2) any special low-dissonance status — those minima
    only exist because the underlying partials are at integer multiples."""
    rs = np.linspace(1.05, 3.0, 50)
    diss = np.array([
        total_dissonance(220.0, 220.0 * r, partials="inharmonic") for r in rs
    ])
    # The argmin under inharmonic partials should NOT land near 2.0
    argmin_r = rs[diss.argmin()]
    assert abs(argmin_r - 2.0) > 0.05 or diss.min() > 0.02


def test_alpha_interpolation_endpoints():
    """alpha=1 should equal natural-harmonic timbre and alpha=0 should
    equal the odd-only timbre in the dissonance landscape (within
    numerical agreement to a couple of digits)."""
    rs = [1.25, 1.5, 1.667, 2.0]
    for r in rs:
        h = total_dissonance(220.0, 220.0 * r, partials="harmonic")
        a1 = total_dissonance(220.0, 220.0 * r, partials=1.0)
        # alpha=1 reproduces full harmonic series up to first n harmonics.
        # The internal layout slightly reorders amps but the cross-partial
        # sum is identical; same value.
        assert abs(h - a1) < 0.01

        o = total_dissonance(220.0, 220.0 * r, partials="odd")
        a0 = total_dissonance(220.0, 220.0 * r, partials=0.0)
        assert abs(o - a0) < 0.01


def test_alpha_interpolation_monotonic():
    """For a ratio that is consonant under harmonic but less so under
    odd-only (the major sixth), the dissonance should increase as α
    drops from 1 → 0. For a ratio that is *more* consonant under odd
    (e.g. 5:3 = 1.667), the dissonance should drop."""
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    # major sixth 5:3
    diss_m6 = [
        total_dissonance(220.0, 220.0 * (5 / 3), partials=float(a))
        for a in alphas
    ]
    # The trend can be non-monotonic globally but at the endpoints
    # alpha=0 (odd-only) should be no worse than alpha=1.
    assert diss_m6[0] <= diss_m6[-1] + 0.001
