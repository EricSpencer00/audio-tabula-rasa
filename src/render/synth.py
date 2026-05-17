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
    # Vibrato: ~5 Hz, ±0.3% pitch (subtle, humanizing)
    vibrato = 1.0 + 0.003 * np.sin(2 * np.pi * 5.0 * t)
    for h in range(1, n_harmonics + 1):
        # Slight per-harmonic detuning (chorus effect, ~0.1% spread)
        detune = 1.0 + 0.001 * (h - 1) * (1 if h % 2 == 0 else -1)
        # Higher harmonics decay faster over time (spectral roll-off)
        decay = np.exp(-0.5 * (h - 1) * t / max(duration, 0.01))
        out += (velocity / h) * decay * np.sin(
            2 * np.pi * h * freq * detune * vibrato * t)
    return out * _envelope(n)


def render_fm_tone(freq: float, duration: float,
                   sample_rate: int = SAMPLE_RATE,
                   velocity: float = 1.0) -> np.ndarray:
    """Gentle FM + additive blend: warm electric-piano-like timbre."""
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate
    vibrato = 1.0 + 0.003 * np.sin(2 * np.pi * 5.0 * t)
    # Gentle FM: mod_index 0.4, ratio 2 (octave), decaying modulation
    mod_idx = 0.4 * (0.2 + 0.8 * np.exp(-4.0 * t / max(duration, 0.01)))
    mod = mod_idx * np.sin(2 * np.pi * 2.0 * freq * vibrato * t)
    fm_part = np.sin(2 * np.pi * freq * vibrato * t + mod)
    # Blend with 2 additive harmonics for body
    h2 = 0.3 * np.sin(2 * np.pi * 2 * freq * vibrato * t)
    h3 = 0.1 * np.sin(2 * np.pi * 3 * freq * vibrato * t)
    out = velocity * (0.7 * fm_part + 0.2 * h2 + 0.1 * h3)
    return out * _envelope(n)


def render_chord(freqs: Sequence[float], duration: float = 1.5,
                 n_harmonics: int = 6) -> np.ndarray:
    n = int(duration * SAMPLE_RATE)
    out = np.zeros(n)
    for f in freqs:
        out += render_harmonic_tone(float(f), duration, n_harmonics)
    return out


def snap_to_scale(freq: float, scale_semitones: Sequence[int] = (0, 2, 4, 7, 9),
                  root_hz: float = 261.63) -> float:
    """Snap a frequency to the nearest note in the given scale.

    ``scale_semitones`` defines pitch classes relative to the root
    (default: C major pentatonic). The root repeats at every octave.
    """
    if freq <= 0:
        return freq
    semitones_from_root = 12.0 * np.log2(freq / root_hz)
    pc = semitones_from_root % 12.0
    octave = semitones_from_root - pc
    best = min(scale_semitones, key=lambda s: min(abs(pc - s), abs(pc - s - 12), abs(pc - s + 12)))
    return root_hz * 2.0 ** ((octave + best) / 12.0)


def render_melody(freqs: Sequence[float], note_duration: float = 0.4,
                  gap: float = 0.05, n_harmonics: int = 6,
                  durations: Sequence[float] | None = None,
                  velocities: Sequence[float] | None = None,
                  use_fm: bool = False) -> np.ndarray:
    chunks = []
    for i, f in enumerate(freqs):
        dur = float(durations[i]) if durations is not None else note_duration
        vel = float(velocities[i]) if velocities is not None else 1.0
        if use_fm:
            chunks.append(render_fm_tone(float(f), dur, velocity=vel))
        else:
            chunks.append(render_harmonic_tone(float(f), dur, n_harmonics,
                                               velocity=vel))
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


def simple_reverb(audio: np.ndarray, decay: float = 0.3,
                   delay_ms: float = 40.0, n_taps: int = 5) -> np.ndarray:
    """Multi-tap delay reverb — fills gaps between notes without smearing."""
    out = audio.copy()
    for i in range(1, n_taps + 1):
        delay_samples = int(i * delay_ms * SAMPLE_RATE / 1000)
        gain = decay ** i
        if delay_samples < len(out):
            out[delay_samples:] += gain * audio[:len(out) - delay_samples]
    peak = np.max(np.abs(out))
    if peak > 1e-9:
        out *= np.max(np.abs(audio)) / peak
    return out


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
