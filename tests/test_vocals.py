"""Tests for src/render/vocals.py — non-Bark utility functions."""
import numpy as np
import pytest

from src.render.vocals import (
    SAMPLE_RATE,
    BARK_RATE,
    _SYLLABLES,
    generate_syllables,
    _freq_to_pitch_hint,
    mix_vocal_instrumental,
)


class TestGenerateSyllables:
    def test_correct_count(self):
        result = generate_syllables(8)
        assert len(result.split()) == 8

    def test_single_note(self):
        result = generate_syllables(1)
        assert result in _SYLLABLES

    def test_deterministic_with_seed(self):
        a = generate_syllables(10, seed=42)
        b = generate_syllables(10, seed=42)
        assert a == b

    def test_different_seeds_differ(self):
        a = generate_syllables(10, seed=0)
        b = generate_syllables(10, seed=99)
        assert a != b

    def test_all_syllables_valid(self):
        result = generate_syllables(100, seed=7)
        for s in result.split():
            assert s in _SYLLABLES


class TestFreqToPitchHint:
    def test_low(self):
        assert _freq_to_pitch_hint(100.0) == "low"
        assert _freq_to_pitch_hint(199.0) == "low"

    def test_medium(self):
        assert _freq_to_pitch_hint(200.0) == "medium"
        assert _freq_to_pitch_hint(349.0) == "medium"

    def test_high(self):
        assert _freq_to_pitch_hint(350.0) == "high"
        assert _freq_to_pitch_hint(880.0) == "high"


class TestMixVocalInstrumental:
    def test_equal_length(self):
        v = np.ones(1000, dtype=np.float32) * 0.5
        inst = np.ones(1000, dtype=np.float32) * 0.5
        mixed = mix_vocal_instrumental(v, inst)
        assert mixed.shape == (1000,)
        assert mixed.dtype == np.float32

    def test_vocal_shorter_padded(self):
        v = np.ones(500, dtype=np.float32) * 0.5
        inst = np.ones(1000, dtype=np.float32) * 0.5
        mixed = mix_vocal_instrumental(v, inst)
        assert mixed.shape == (1000,)

    def test_instrumental_shorter_padded(self):
        v = np.ones(1000, dtype=np.float32) * 0.5
        inst = np.ones(500, dtype=np.float32) * 0.5
        mixed = mix_vocal_instrumental(v, inst)
        assert mixed.shape == (1000,)

    def test_normalized_peak(self):
        v = np.ones(1000, dtype=np.float32)
        inst = np.ones(1000, dtype=np.float32)
        mixed = mix_vocal_instrumental(v, inst)
        assert abs(np.abs(mixed).max() - 0.9) < 1e-5

    def test_gains_applied(self):
        v = np.ones(100, dtype=np.float32)
        inst = np.zeros(100, dtype=np.float32)
        mixed = mix_vocal_instrumental(v, inst, vocal_gain=0.5, instrumental_gain=0.0)
        peak = np.abs(mixed).max()
        assert peak == pytest.approx(0.9, abs=1e-5)

    def test_silence_handling(self):
        v = np.zeros(100, dtype=np.float32)
        inst = np.zeros(100, dtype=np.float32)
        mixed = mix_vocal_instrumental(v, inst)
        assert np.allclose(mixed, 0.0)

    def test_custom_gains(self):
        rng = np.random.RandomState(0)
        v = rng.randn(200).astype(np.float32)
        inst = rng.randn(200).astype(np.float32)
        mixed_default = mix_vocal_instrumental(v, inst)
        mixed_custom = mix_vocal_instrumental(v, inst, vocal_gain=0.9, instrumental_gain=0.1)
        assert not np.allclose(mixed_default, mixed_custom)


class TestConstants:
    def test_sample_rate(self):
        assert SAMPLE_RATE == 44_100

    def test_bark_rate(self):
        assert BARK_RATE == 24_000

    def test_syllable_palette_not_empty(self):
        assert len(_SYLLABLES) > 0

    def test_syllables_are_short(self):
        for s in _SYLLABLES:
            assert len(s) <= 4
