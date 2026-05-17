"""
Additive-synthesis renderer with ADSR envelopes and velocity dynamics.

Sine-bank synthesis — the pitches come entirely from the learned
generators; this module just renders them to audio.
"""
import struct
import wave
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SAMPLE_RATE = 44100

def _envelope(n_samples: int, attack_ms: float = 12.0,
              release_ms: float = 80.0) -> np.ndarray:
    env = np.ones(n_samples)
    a = max(1, int(attack_ms * SAMPLE_RATE / 1000))
    r = max(1, int(release_ms * SAMPLE_RATE / 1000))
    env[:a] = np.linspace(0.0, 1.0, a)
    if r < n_samples:
        env[-r:] = np.linspace(1.0, 0.0, r)
    return env


def render_harmonic_tone(freq: float, duration: float,
                         n_harmonics: int = 6,
                         sample_rate: int = SAMPLE_RATE,
                         velocity: float = 1.0) -> np.ndarray:
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate
    out = np.zeros(n)
    for h in range(1, n_harmonics + 1):
        out += (velocity / h) * np.sin(2 * np.pi * h * freq * t)
    return out * _envelope(n)


def render_chord(freqs: Sequence[float], duration: float = 1.5,
                 n_harmonics: int = 6) -> np.ndarray:
    n = int(duration * SAMPLE_RATE)
    out = np.zeros(n)
    for f in freqs:
        out += render_harmonic_tone(float(f), duration, n_harmonics)
    return out


def render_melody(freqs: Sequence[float], note_duration: float = 0.4,
                  gap: float = 0.05, n_harmonics: int = 6) -> np.ndarray:
    chunks = []
    for f in freqs:
        chunks.append(render_harmonic_tone(float(f), note_duration,
                                            n_harmonics))
        chunks.append(np.zeros(int(gap * SAMPLE_RATE)))
    return np.concatenate(chunks)


def render_drum(decay_ms: float = 80.0,
                fundamental: float = 120.0) -> np.ndarray:
    """A short percussive blip: damped sine + noise burst."""
    n = int(decay_ms * SAMPLE_RATE / 1000)
    t = np.arange(n) / SAMPLE_RATE
    decay = np.exp(-30 * t)
    body = np.sin(2 * np.pi * fundamental * t) * decay
    click = (np.random.default_rng(0).standard_normal(n) * decay * 0.4)
    return body + click


def render_rhythm(onsets: Sequence[float], duration: float = 4.0,
                  drum: np.ndarray = None) -> np.ndarray:
    if drum is None:
        drum = render_drum()
    n_total = int(duration * SAMPLE_RATE)
    out = np.zeros(n_total + len(drum))
    for t in onsets:
        idx = int(t * SAMPLE_RATE)
        if 0 <= idx < n_total:
            out[idx : idx + len(drum)] += drum
    return out[:n_total]


def render_melodic_rhythm(freqs: Sequence[float],
                          onsets: Sequence[float],
                          duration: float = 4.0,
                          note_duration: float = 0.35,
                          n_harmonics: int = 6) -> np.ndarray:
    n_total = int(duration * SAMPLE_RATE)
    out = np.zeros(n_total + int(note_duration * SAMPLE_RATE))
    for f, t in zip(freqs, onsets):
        if t < 0 or t >= duration:
            continue
        tone = render_harmonic_tone(float(f), note_duration, n_harmonics)
        idx = int(t * SAMPLE_RATE)
        end = idx + len(tone)
        out[idx:end] += tone
    return out[:n_total]


def normalize(audio: np.ndarray, headroom_db: float = -3.0) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak < 1e-9:
        return audio
    target = 10 ** (headroom_db / 20.0)
    return audio * (target / peak)


def write_wav(path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    audio = normalize(audio)
    pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())
