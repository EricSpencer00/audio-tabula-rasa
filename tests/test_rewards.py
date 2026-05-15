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
