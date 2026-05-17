"""
TTS vocal renderer using Bark.

Takes melody frequencies + auto-generated phoneme/syllable sequences,
renders a vocal track via Bark, and mixes with the instrumental audio.

Bark is used because it handles singing/humming, runs locally on MPS,
and is Apache 2.0 licensed.
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Sequence

SAMPLE_RATE = 44_100
BARK_RATE = 24_000

# Simple syllable palette for melody vocalization
_SYLLABLES = [
    "la", "da", "na", "ma", "ta", "ra",
    "lo", "do", "no", "mo", "to", "ro",
    "lee", "dee", "nee", "mee", "tee", "ree",
    "lu", "du", "nu", "mu", "tu", "ru",
]

_bark_model = None


def _load_bark():
    global _bark_model
    if _bark_model is not None:
        return _bark_model
    from bark import SAMPLE_RATE as BARK_SR
    from bark.generation import (
        load_codec_model,
        preload_models,
    )
    preload_models()
    _bark_model = True
    return _bark_model


def generate_syllables(n_notes: int, seed: int = 0) -> str:
    """Generate a syllable sequence for n_notes melody notes."""
    rng = np.random.RandomState(seed)
    syllables = [_SYLLABLES[rng.randint(len(_SYLLABLES))] for _ in range(n_notes)]
    return " ".join(syllables)


def _freq_to_pitch_hint(freq: float) -> str:
    """Convert Hz to rough pitch description for Bark prompt."""
    if freq < 200:
        return "low"
    elif freq < 350:
        return "medium"
    else:
        return "high"


def render_vocal_track(freqs: np.ndarray,
                       durations: Optional[np.ndarray] = None,
                       text: Optional[str] = None,
                       voice_preset: str = "v2/en_speaker_6",
                       target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Render a vocal track for the given melody.

    Args:
        freqs: melody frequencies in Hz, shape (N,)
        durations: note durations in seconds, shape (N,) (optional)
        text: custom lyrics/syllables (auto-generated if None)
        voice_preset: Bark voice preset
        target_sr: output sample rate

    Returns:
        Vocal audio at target_sr, mono float32.
    """
    from bark import generate_audio, SAMPLE_RATE as BARK_SR
    _load_bark()

    n_notes = len(freqs)
    if text is None:
        text = generate_syllables(n_notes)

    # Bark generates speech from text — we use a singing-style prompt
    prompt = f"♪ {text} ♪"

    audio = generate_audio(prompt, history_prompt=voice_preset)

    # Resample from Bark's 24kHz to target
    if BARK_SR != target_sr:
        import scipy.signal
        n_out = int(len(audio) * target_sr / BARK_SR)
        audio = scipy.signal.resample(audio, n_out).astype(np.float32)

    # Normalize
    peak = np.abs(audio).max()
    if peak > 1e-6:
        audio = audio / peak * 0.7

    return audio


def mix_vocal_instrumental(vocal: np.ndarray,
                           instrumental: np.ndarray,
                           vocal_gain: float = 0.6,
                           instrumental_gain: float = 0.8) -> np.ndarray:
    """Mix vocal and instrumental tracks.

    Pads the shorter track with silence, applies gains, and normalizes.
    """
    max_len = max(len(vocal), len(instrumental))
    v = np.zeros(max_len, dtype=np.float32)
    inst = np.zeros(max_len, dtype=np.float32)
    v[:len(vocal)] = vocal * vocal_gain
    inst[:len(instrumental)] = instrumental * instrumental_gain
    mixed = v + inst

    peak = np.abs(mixed).max()
    if peak > 1e-6:
        mixed = mixed / peak * 0.9

    return mixed


def render_melody_with_vocals(freqs: np.ndarray,
                              durations: Optional[np.ndarray] = None,
                              velocities: Optional[np.ndarray] = None,
                              instrumental_renderer=None,
                              text: Optional[str] = None,
                              voice_preset: str = "v2/en_speaker_6",
                              vocal_gain: float = 0.6,
                              instrumental_gain: float = 0.8) -> np.ndarray:
    """Full pipeline: render instrumental + vocals, mix together.

    Args:
        freqs: melody frequencies in Hz
        durations: note durations (optional)
        velocities: note velocities (optional)
        instrumental_renderer: callable(freqs) -> audio, or None for vocals-only
        text: custom lyrics (auto-generated if None)
        voice_preset: Bark voice preset
        vocal_gain: vocal track gain in mix
        instrumental_gain: instrumental track gain in mix

    Returns:
        Mixed audio at 44.1kHz, mono float32.
    """
    vocal = render_vocal_track(freqs, durations, text, voice_preset)

    if instrumental_renderer is not None:
        # Build the combined array if durations/velocities provided
        if durations is not None and velocities is not None:
            combined = np.concatenate([freqs, durations, velocities])
            instrumental = instrumental_renderer(combined)
        else:
            instrumental = instrumental_renderer(freqs)
        return mix_vocal_instrumental(vocal, instrumental, vocal_gain, instrumental_gain)

    return vocal
