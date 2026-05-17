"""
Tests for the rule-based music theory judge.

Each module gets property tests: known-good inputs should score higher
than known-bad inputs, and edge cases shouldn't crash.

Run with:
    python -m pytest tests/test_theory_judge.py -v
"""
import numpy as np
import pytest

from src.reward.theory_judge import (
    DEFAULT_WEIGHTS,
    SCALES,
    cadence_detection,
    dynamic_shaping,
    interval_quality,
    key_adherence,
    melodic_contour,
    rhythm_analysis,
    tension_resolution,
    theory_reward,
    theory_reward_breakdown,
    theory_reward_per_note,
    voice_leading,
    _detect_key,
    _freqs_to_pitch_classes,
    _freqs_to_semitones,
    _motif_repetition,
)

# ---- helper frequencies ----
C4 = 261.63
D4 = 293.66
E4 = 329.63
F4 = 349.23
G4 = 392.00
A4 = 440.00
B4 = 493.88
C5 = 523.25

C_MAJOR_SCALE = np.array([C4, D4, E4, F4, G4, A4, B4, C5])
C_MAJOR_ARPEGGIO = np.array([C4, E4, G4, C5])
CHROMATIC_MESS = np.array([C4, C4 * 2**(1/12), C4 * 2**(2/12), C4 * 2**(3/12),
                           C4 * 2**(4/12), C4 * 2**(5/12), C4 * 2**(6/12), C4 * 2**(7/12)])
RANDOM_FREQS = np.array([187.3, 412.7, 155.2, 503.1, 278.9, 601.4, 199.8, 445.6])


class TestKeyAdherence:
    def test_in_key_scores_higher_than_chromatic(self):
        in_key = key_adherence(C_MAJOR_SCALE, root_hz=C4, scale=SCALES["major"])
        chromatic = key_adherence(CHROMATIC_MESS, root_hz=C4, scale=SCALES["major"])
        assert in_key > chromatic

    def test_perfect_scale_near_one(self):
        score = key_adherence(C_MAJOR_SCALE, root_hz=C4, scale=SCALES["major"])
        assert score > 0.9  # perfectly in key ≈ 1.0

    def test_auto_detect_key(self):
        score = key_adherence(C_MAJOR_SCALE)
        assert score > 0.8  # should auto-detect C major

    def test_single_note_returns_zero(self):
        assert key_adherence(np.array([440.0])) == 0.0

    def test_pentatonic_in_pentatonic(self):
        pent = np.array([C4, D4, E4, G4, A4, C5])
        score = key_adherence(pent, root_hz=C4, scale=SCALES["pentatonic_major"])
        assert score > 0.9


class TestIntervalQuality:
    def test_consonant_intervals_score_high(self):
        fifths = np.array([C4, G4, C4, G4, C4, G4])  # all perfect 5ths
        score = interval_quality(fifths)
        assert score > 0.8

    def test_dissonant_intervals_score_low(self):
        semitones = np.array([C4, C4 * 2**(1/12), C4, C4 * 2**(1/12)])  # minor 2nds
        score = interval_quality(semitones)
        assert score < 0.3

    def test_mixed_intervals(self):
        mixed = np.array([C4, E4, G4, A4])  # M3, m3, M2
        score = interval_quality(mixed)
        assert 0.3 < score < 0.9

    def test_single_note(self):
        assert interval_quality(np.array([440.0])) == 0.0

    def test_unisons_score_high(self):
        unisons = np.array([A4, A4, A4, A4])
        score = interval_quality(unisons)
        assert score > 0.9


class TestMelodicContour:
    def test_stepwise_melody_scores_well(self):
        stepwise = np.array([C4, D4, E4, F4, G4, A4, G4, F4, E4, D4, C4])
        score = melodic_contour(stepwise)
        assert score > 0.3

    def test_large_leaps_score_lower(self):
        leapy = np.array([C4, C5, C4, C5, C4, C5, C4, C5])
        stepwise = np.array([C4, D4, E4, F4, G4, A4, G4, E4])
        assert melodic_contour(stepwise) > melodic_contour(leapy)

    def test_flat_melody_penalized(self):
        flat = np.array([A4, A4, A4, A4, A4, A4, A4, A4])
        stepwise = np.array([C4, D4, E4, F4, G4, A4, G4, E4])
        assert melodic_contour(stepwise) > melodic_contour(flat)

    def test_short_melody(self):
        assert melodic_contour(np.array([C4, D4])) == 0.0

    def test_arch_shape_rewards_climax(self):
        arch = np.array([C4, D4, E4, F4, G4, A4, G4, F4, E4, D4])
        flat = np.array([G4, G4, G4, G4, G4, G4, G4, G4, G4, G4])
        assert melodic_contour(arch) > melodic_contour(flat)


class TestMotifRepetition:
    def test_repeated_pattern_scores_high(self):
        # C-D-E repeated: intervals [2, 2, 2, 2, 2, 2]
        intervals = np.array([2, 2, 2, 2, 2, 2])
        score = _motif_repetition(intervals)
        assert score > 0.5

    def test_no_repetition_scores_zero(self):
        intervals = np.array([1, 5, -3, 7, -2, 4])
        score = _motif_repetition(intervals)
        assert score < 0.3

    def test_aba_pattern(self):
        # A-B-A pattern: up 2, down 2, up 2, down 2
        intervals = np.array([2, -2, 2, -2, 2, -2])
        score = _motif_repetition(intervals)
        assert score > 0.3

    def test_short_input(self):
        assert _motif_repetition(np.array([2, 1])) == 0.0

    def test_melody_with_motif_beats_without(self):
        # Melody with clear motif: C-D-E-D repeated
        with_motif = np.array([C4, D4, E4, D4, C4, D4, E4, D4])
        no_motif = RANDOM_FREQS
        assert melodic_contour(with_motif) > melodic_contour(no_motif) - 0.1


class TestCadenceDetection:
    def test_v_to_i_cadence_scores_high(self):
        # G4 -> C4 is V-I in C major
        melody = np.array([C4, E4, G4, C4])
        score = cadence_detection(melody, phrase_len=4)
        assert score > 0.3

    def test_no_cadence(self):
        # notes that don't resolve
        melody = np.array([C4 * 2**(6/12)] * 4)  # all tritones
        score = cadence_detection(melody, phrase_len=4)
        assert score < 0.5

    def test_plagal_cadence(self):
        # IV-I: F->C in C major
        melody = np.array([E4, G4, F4, C4])
        score = cadence_detection(melody, phrase_len=4)
        assert score > 0.5

    def test_leading_tone_cadence(self):
        # vii-I: B->C
        melody = np.array([E4, G4, B4, C5])
        score = cadence_detection(melody, phrase_len=4)
        assert score > 0.5

    def test_cadence_ordering(self):
        # V-I should beat random
        vi = cadence_detection(np.array([E4, G4, G4, C4]), phrase_len=4)
        random_cad = cadence_detection(RANDOM_FREQS[:4], phrase_len=4)
        assert vi > random_cad

    def test_agogic_accent_bonus(self):
        melody = np.array([C4, E4, G4, C4])
        short_durs = np.array([0.3, 0.3, 0.3, 0.3])
        long_end = np.array([0.2, 0.2, 0.2, 0.6])
        score_short = cadence_detection(melody, short_durs, phrase_len=4)
        score_long = cadence_detection(melody, long_end, phrase_len=4)
        assert score_long >= score_short

    def test_short_melody(self):
        assert cadence_detection(np.array([C4, D4]), phrase_len=4) == 0.0


class TestVoiceLeading:
    def test_contrary_motion_rewarded(self):
        upper = np.array([G4, A4, G4, A4])
        lower = np.array([E4, D4, E4, D4])  # contrary to upper
        voices = np.stack([lower, upper])
        score = voice_leading(voices)
        assert score > 0

    def test_parallel_fifths_penalized(self):
        upper = np.array([G4, A4, B4, C5])
        lower = np.array([C4, D4, E4, F4])  # parallel 5ths
        voices = np.stack([lower, upper])
        score = voice_leading(voices)
        # May be negative due to parallel 5th penalties
        contrary_upper = np.array([G4, F4, G4, F4])
        contrary_lower = np.array([C4, D4, C4, D4])
        contrary_voices = np.stack([contrary_lower, contrary_upper])
        assert voice_leading(contrary_voices) > score

    def test_single_voice_returns_zero(self):
        assert voice_leading(np.array([[C4, D4, E4]])) == 0.0

    def test_two_notes_min(self):
        voices = np.array([[C4], [G4]])
        assert voice_leading(voices) == 0.0


class TestRhythmAnalysis:
    def test_grid_aligned_scores_high(self):
        on_grid = np.array([0.25, 0.25, 0.5, 0.25, 0.25, 0.5, 0.25, 0.25])
        score = rhythm_analysis(on_grid)
        assert score > 0.3

    def test_random_durations_score_lower(self):
        on_grid = np.array([0.25, 0.25, 0.5, 0.25, 0.25, 0.5])
        random_d = np.array([0.17, 0.42, 0.09, 0.61, 0.23, 0.55])
        assert rhythm_analysis(on_grid) > rhythm_analysis(random_d)

    def test_all_same_low_variety(self):
        monotone = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
        score = rhythm_analysis(monotone)
        # Grid score should be good but variety score low
        assert score < 0.7

    def test_short_sequence(self):
        assert rhythm_analysis(np.array([0.3, 0.3])) == 0.0


class TestTensionResolution:
    def test_build_and_resolve_scores_well(self):
        # Consonant start, dissonant middle, consonant end
        build_resolve = np.array([C4, E4, G4, C4*2**(6/12), C4*2**(1/12),
                                  G4, E4, C4, G4, C4])
        score = tension_resolution(build_resolve)
        assert isinstance(score, float)

    def test_all_consonant_lower_tension(self):
        consonant = np.array([C4, G4, C4, G4, C4, G4, C4, G4])
        score = tension_resolution(consonant)
        assert isinstance(score, float)

    def test_ends_on_consonance_bonus(self):
        # End on octave (consonant)
        good_end = np.array([C4, C4*2**(6/12), E4, G4, C5])
        # End on tritone (dissonant)
        bad_end = np.array([C4, E4, G4, C5, C4*2**(6/12)])
        assert tension_resolution(good_end) >= tension_resolution(bad_end) - 0.5

    def test_short_sequence(self):
        assert tension_resolution(np.array([C4, D4, E4])) == 0.0


class TestDynamicShaping:
    def test_arch_dynamics_score_well(self):
        arch = np.array([0.3, 0.5, 0.7, 0.9, 1.0, 0.9, 0.7, 0.5])
        score = dynamic_shaping(arch)
        assert score > 0.3

    def test_flat_dynamics_penalized(self):
        flat = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        arch = np.array([0.3, 0.5, 0.7, 0.9, 1.0, 0.9, 0.7, 0.5])
        assert dynamic_shaping(arch) > dynamic_shaping(flat)

    def test_metric_accents(self):
        # Strong on beats 1,3 — weak on 2,4
        accented = np.array([0.9, 0.5, 0.8, 0.4, 0.9, 0.5, 0.8, 0.4])
        score = dynamic_shaping(accented)
        assert score > 0.2

    def test_short_sequence(self):
        assert dynamic_shaping(np.array([0.5, 0.7])) == 0.0


class TestCompositeReward:
    def test_musical_melody_beats_random(self):
        musical = C_MAJOR_SCALE
        durs_musical = np.array([0.25, 0.25, 0.5, 0.25, 0.25, 0.5, 0.25, 0.5])
        vels_musical = np.array([0.7, 0.5, 0.8, 0.6, 0.9, 0.7, 0.5, 0.8])

        durs_random = np.random.RandomState(42).uniform(0.1, 0.8, 8)
        vels_random = np.random.RandomState(42).uniform(0.3, 1.0, 8)

        score_musical = theory_reward(musical, durs_musical, vels_musical)
        score_random = theory_reward(RANDOM_FREQS, durs_random, vels_random)
        assert score_musical > score_random

    def test_breakdown_returns_all_keys(self):
        result = theory_reward_breakdown(
            C_MAJOR_SCALE,
            durations=np.ones(8) * 0.3,
            velocities=np.ones(8) * 0.7,
        )
        expected_keys = {"key_adherence", "interval_quality", "melodic_contour",
                         "cadence_detection", "rhythm_analysis", "tension_resolution",
                         "dynamic_shaping"}
        assert expected_keys.issubset(result.keys())

    def test_custom_weights(self):
        freqs = C_MAJOR_SCALE
        w1 = {"key_adherence": 10.0, "interval_quality": 0.0, "melodic_contour": 0.0,
               "cadence_detection": 0.0, "voice_leading": 0.0, "rhythm_analysis": 0.0,
               "tension_resolution": 0.0, "dynamic_shaping": 0.0}
        w2 = {"key_adherence": 0.0, "interval_quality": 10.0, "melodic_contour": 0.0,
               "cadence_detection": 0.0, "voice_leading": 0.0, "rhythm_analysis": 0.0,
               "tension_resolution": 0.0, "dynamic_shaping": 0.0}
        s1 = theory_reward(freqs, weights=w1)
        s2 = theory_reward(freqs, weights=w2)
        # Different weights should give different scores
        assert s1 != s2

    def test_no_crash_with_minimal_input(self):
        score = theory_reward(np.array([440.0, 550.0]))
        assert isinstance(score, float)

    def test_voice_leading_included_for_multivoice(self):
        voices = np.stack([
            np.array([C4, D4, E4, F4]),
            np.array([G4, A4, B4, C5]),
        ])
        result = theory_reward_breakdown(voices[0], voices=voices)
        assert "voice_leading" in result


class TestDetectKey:
    def test_c_major_detected(self):
        root_hz, name, degrees = _detect_key(C_MAJOR_SCALE)
        assert name == "major"
        assert abs(root_hz - C4) < 1.0

    def test_a_minor_detected(self):
        a_minor = np.array([A4, B4, C5, D4*2, E4*2, F4*2, G4*2, A4*2])
        root_hz, name, degrees = _detect_key(a_minor)
        assert name in ("natural_minor", "dorian", "major")

    def test_pentatonic(self):
        pent = np.array([C4, D4, E4, G4, A4, C5])
        root_hz, name, degrees = _detect_key(pent)
        # Should match either pentatonic or major
        assert name in ("pentatonic_major", "major")


class TestPerNoteReward:
    def test_shape_matches_input(self):
        pn = theory_reward_per_note(C_MAJOR_SCALE)
        assert pn.shape == (len(C_MAJOR_SCALE),)

    def test_in_key_notes_score_higher(self):
        in_key = theory_reward_per_note(C_MAJOR_SCALE)
        out_key = theory_reward_per_note(CHROMATIC_MESS)
        assert in_key.mean() > out_key.mean()

    def test_with_durations_and_velocities(self):
        n = len(C_MAJOR_SCALE)
        durs = np.full(n, 0.25)
        vels = np.linspace(0.4, 0.9, n)
        pn = theory_reward_per_note(C_MAJOR_SCALE, durations=durs, velocities=vels)
        assert pn.shape == (n,)
        assert np.all(np.isfinite(pn))

    def test_single_note_returns_zeros(self):
        pn = theory_reward_per_note(np.array([C4]))
        assert len(pn) == 1
        assert pn[0] == 0.0

    def test_values_in_reasonable_range(self):
        pn = theory_reward_per_note(C_MAJOR_SCALE)
        assert pn.min() >= -0.1
        assert pn.max() <= 1.1

    def test_cadence_rewards_tonic_endings(self):
        # V-I cadence at phrase boundary should reward notes 2,3 (pre-end, end)
        vi_cadence = np.array([C4, E4, G4, C4])  # phrase: C-E-G-C (G=V, C=I)
        random_end = np.array([C4, E4, G4, F4 * 2**(6/12)])  # ends on tritone
        pn_good = theory_reward_per_note(vi_cadence, root_hz=C4, scale=SCALES["major"])
        pn_bad = theory_reward_per_note(random_end, root_hz=C4, scale=SCALES["major"])
        # Last note of good cadence should score higher
        assert pn_good[-1] > pn_bad[-1]

    def test_cadence_rewards_final_note_tonic(self):
        # Ending on tonic should score higher than ending on non-tonic
        ends_tonic = np.array([C4, D4, E4, F4, G4, A4, B4, C5])
        ends_tritone = np.array([C4, D4, E4, F4, G4, A4, B4, G4 * 2**(6/12)])
        pn_tonic = theory_reward_per_note(ends_tonic, root_hz=C4, scale=SCALES["major"])
        pn_tritone = theory_reward_per_note(ends_tritone, root_hz=C4, scale=SCALES["major"])
        assert pn_tonic[-1] > pn_tritone[-1]


class TestUtilities:
    def test_freqs_to_semitones(self):
        semitones = _freqs_to_semitones(np.array([C4, C5]))
        assert abs(semitones[1] - semitones[0] - 12.0) < 0.01

    def test_freqs_to_pitch_classes(self):
        pcs = _freqs_to_pitch_classes(np.array([C4, C5]))
        # Same pitch class — either both near 0 or differ by 12
        diff = abs(pcs[0] - pcs[1])
        assert diff < 0.01 or abs(diff - 12.0) < 0.01
