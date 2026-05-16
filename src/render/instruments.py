"""
A small library of synthesized "instruments" for the song renderer.

All instruments expose `render(freq, duration, velocity=1.0)` and return
a mono float32 numpy waveform at SAMPLE_RATE. None of these are trained
on real audio — they're closed-form formulas, the same shape as the
sine-bank used in `synth.py` but stretched out into recognizable timbres
so the listener can tell instruments apart.

The instrument list is deliberately compact:

  HarmonicAdditive — sum of integer-multiple sines with configurable
                     amplitude rolloff and an ADSR envelope. Most of
                     the "Phase 1-7" sounds are this with different
                     spectra (see `pad`, `organ`, `reed`).
  Karplus-Strong   — short delay line + lowpass feedback, the canonical
                     plucked-string algorithm. Used for bass and pluck
                     leads.
  FMSynth          — one-operator FM (carrier + modulator) with a
                     decaying mod-index envelope. Bell/electric-piano
                     family.
  PercussionHit    — noise burst + low tonal body, no pitch parameter.

We also expose a small palette of preset instruments tuned by hand
(reed, pad, organ, bass_pluck, lead, bell, kick, snare, hihat) so the
song renderer can refer to them by name.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

SAMPLE_RATE = 44100


# ----- envelopes -----------------------------------------------------

def adsr(n_samples: int, attack_ms: float = 8.0, decay_ms: float = 80.0,
         sustain: float = 0.7, release_ms: float = 120.0) -> np.ndarray:
    """Linear ADSR envelope shaped to the requested duration."""
    a = max(1, int(attack_ms * SAMPLE_RATE / 1000))
    d = max(1, int(decay_ms * SAMPLE_RATE / 1000))
    r = max(1, int(release_ms * SAMPLE_RATE / 1000))
    s_len = max(1, n_samples - a - d - r)
    a = min(a, n_samples)
    d = min(d, n_samples - a)
    r = min(r, n_samples - a - d)
    s_len = max(0, n_samples - a - d - r)

    env = np.zeros(n_samples)
    if a:
        env[:a] = np.linspace(0.0, 1.0, a)
    if d:
        env[a:a + d] = np.linspace(1.0, sustain, d)
    if s_len:
        env[a + d:a + d + s_len] = sustain
    if r:
        env[-r:] = np.linspace(sustain, 0.0, r)
    return env


# ----- harmonic additive instrument ----------------------------------

@dataclass
class HarmonicAdditive:
    """Sum of sines at integer multiples of `freq`."""
    name: str
    n_partials: int = 8
    # `amps_fn(k)` returns the relative amplitude of the k-th partial
    amps_fn: Callable[[int], float] = lambda k: 1.0 / k
    # Optional per-partial detune in cents (for chorus / pad effect)
    detune_cents: float = 0.0
    vibrato_hz: float = 0.0
    vibrato_cents: float = 0.0
    attack_ms: float = 8.0
    decay_ms: float = 80.0
    sustain: float = 0.7
    release_ms: float = 120.0
    odd_only: bool = False

    def render(self, freq: float, duration: float,
               velocity: float = 1.0) -> np.ndarray:
        n = int(duration * SAMPLE_RATE)
        t = np.arange(n) / SAMPLE_RATE
        out = np.zeros(n)

        # Optional sinusoidal vibrato of the carrier frequency
        if self.vibrato_hz > 0 and self.vibrato_cents > 0:
            vib = (np.sin(2 * np.pi * self.vibrato_hz * t)
                   * self.vibrato_cents / 1200.0)
            phase = 2 * np.pi * np.cumsum(freq * (2.0 ** vib)) / SAMPLE_RATE
        else:
            phase = 2 * np.pi * freq * t

        ks = (range(1, 2 * self.n_partials, 2) if self.odd_only
              else range(1, self.n_partials + 1))
        for k in ks:
            amp = self.amps_fn(k)
            if amp == 0:
                continue
            # Tiny detune on each partial keeps it from sounding electric
            if self.detune_cents:
                rng = np.random.default_rng(k)
                d = (rng.uniform(-1, 1) * self.detune_cents) / 1200.0
                kf = k * (2.0 ** d)
            else:
                kf = float(k)
            out += amp * np.sin(kf * phase)

        env = adsr(n, self.attack_ms, self.decay_ms,
                   self.sustain, self.release_ms)
        return velocity * env * out / max(1, len(list(ks)) ** 0.5)


# ----- Karplus-Strong plucked-string ---------------------------------

@dataclass
class KarplusStrong:
    name: str = "pluck"
    decay: float = 0.9965
    excite_ms: float = 20.0   # noise-burst length
    attack_ms: float = 1.0
    release_ms: float = 60.0

    def render(self, freq: float, duration: float,
               velocity: float = 1.0) -> np.ndarray:
        n = int(duration * SAMPLE_RATE)
        period = max(2, int(SAMPLE_RATE / max(20.0, freq)))
        rng = np.random.default_rng(int(freq * 10))
        # Excitation: short white noise
        excite_n = max(period, int(self.excite_ms * SAMPLE_RATE / 1000))
        buf = rng.standard_normal(period)
        out = np.zeros(n)
        for i in range(n):
            v = (buf[i % period] + buf[(i - 1) % period]) * 0.5 * self.decay
            out[i] = v
            buf[i % period] = v
        # Soft attack + release
        env = adsr(n, attack_ms=self.attack_ms, decay_ms=10.0,
                   sustain=1.0, release_ms=self.release_ms)
        return velocity * env * out


# ----- single-operator FM synth (bell / EP) --------------------------

@dataclass
class FMSynth:
    name: str = "bell"
    mod_ratio: float = 1.4              # modulator freq / carrier freq
    mod_index_start: float = 5.0
    mod_index_end: float = 0.4
    attack_ms: float = 4.0
    decay_ms: float = 200.0
    sustain: float = 0.0                # bell-like: no sustain, long release
    release_ms: float = 800.0

    def render(self, freq: float, duration: float,
               velocity: float = 1.0) -> np.ndarray:
        n = int(duration * SAMPLE_RATE)
        t = np.arange(n) / SAMPLE_RATE
        # Decaying mod index
        idx = (self.mod_index_start
               * (self.mod_index_end / max(1e-3, self.mod_index_start))
               ** (t / max(1e-3, duration)))
        mod = idx * np.sin(2 * np.pi * self.mod_ratio * freq * t)
        out = np.sin(2 * np.pi * freq * t + mod)
        env = adsr(n, self.attack_ms, self.decay_ms,
                   self.sustain, self.release_ms)
        return velocity * env * out


# ----- percussion ----------------------------------------------------

@dataclass
class PercussionHit:
    name: str
    body_freq: float = 60.0       # for tonal kick / snare body
    body_decay_ms: float = 80.0
    noise_decay_ms: float = 40.0
    noise_weight: float = 0.5
    body_weight: float = 0.5
    duration_ms: float = 250.0
    highpass_cutoff_hz: float = 0.0   # for hi-hat, set high

    def render(self, freq: float = 0.0, duration: float = None,
               velocity: float = 1.0) -> np.ndarray:
        dur_ms = duration * 1000 if duration else self.duration_ms
        n = int(dur_ms * SAMPLE_RATE / 1000)
        t = np.arange(n) / SAMPLE_RATE
        rng = np.random.default_rng(int(self.body_freq * 100))

        # Body: damped sine
        body = (np.sin(2 * np.pi * self.body_freq * t)
                * np.exp(-1000.0 / max(1, self.body_decay_ms) * t))
        # Noise burst
        noise = (rng.standard_normal(n)
                 * np.exp(-1000.0 / max(1, self.noise_decay_ms) * t))

        if self.highpass_cutoff_hz > 0:
            # Single-pole highpass via first-difference of a smoothed signal
            tau = 1.0 / (2 * np.pi * self.highpass_cutoff_hz)
            alpha = SAMPLE_RATE * tau / (SAMPLE_RATE * tau + 1.0)
            hp = np.zeros(n)
            prev = 0.0
            for i in range(1, n):
                hp[i] = alpha * (hp[i - 1] + noise[i] - noise[i - 1])
            noise = hp

        out = self.body_weight * body + self.noise_weight * noise
        return velocity * out


# ----- preset palette ------------------------------------------------

PRESETS = {
    "pad": HarmonicAdditive(
        name="pad",
        n_partials=10,
        amps_fn=lambda k: (1.0 / k) * (0.5 if k % 2 == 0 else 1.0),
        detune_cents=4.0,
        attack_ms=180.0, decay_ms=200.0, sustain=0.55, release_ms=350.0,
        vibrato_hz=4.5, vibrato_cents=12.0,
    ),
    "organ": HarmonicAdditive(
        name="organ",
        n_partials=8,
        amps_fn=lambda k: 1.0 / (1 + (k - 1) ** 1.4),
        attack_ms=12.0, decay_ms=40.0, sustain=0.92, release_ms=80.0,
    ),
    "reed": HarmonicAdditive(
        name="reed",
        n_partials=8,
        amps_fn=lambda k: 1.0 / k,
        odd_only=True,
        attack_ms=20.0, decay_ms=60.0, sustain=0.85, release_ms=140.0,
        vibrato_hz=5.5, vibrato_cents=18.0,
    ),
    "lead": HarmonicAdditive(
        name="lead",
        n_partials=12,
        amps_fn=lambda k: 1.0 / k,
        attack_ms=4.0, decay_ms=180.0, sustain=0.4, release_ms=120.0,
        vibrato_hz=6.0, vibrato_cents=15.0,
    ),
    "bell": FMSynth(name="bell", mod_ratio=1.4,
                    mod_index_start=4.0, mod_index_end=0.3,
                    attack_ms=2.0, decay_ms=300.0, sustain=0.0,
                    release_ms=900.0),
    "epiano": FMSynth(name="epiano", mod_ratio=1.0,
                       mod_index_start=2.5, mod_index_end=0.2,
                       attack_ms=2.0, decay_ms=250.0, sustain=0.05,
                       release_ms=400.0),
    "bass_pluck": KarplusStrong(name="bass_pluck", decay=0.997,
                                 excite_ms=18.0, release_ms=80.0),
    "kick": PercussionHit(
        name="kick", body_freq=55.0, body_decay_ms=180.0,
        noise_decay_ms=10.0, noise_weight=0.15, body_weight=0.85,
        duration_ms=300.0,
    ),
    "snare": PercussionHit(
        name="snare", body_freq=200.0, body_decay_ms=70.0,
        noise_decay_ms=120.0, noise_weight=0.65, body_weight=0.35,
        duration_ms=250.0,
    ),
    "hihat": PercussionHit(
        name="hihat", body_freq=8000.0, body_decay_ms=15.0,
        noise_decay_ms=40.0, noise_weight=1.0, body_weight=0.0,
        duration_ms=120.0, highpass_cutoff_hz=4000.0,
    ),
}


def get(name: str):
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}. options: {list(PRESETS)}")
    return PRESETS[name]
