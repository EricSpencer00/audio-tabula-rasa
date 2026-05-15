"""
Phase-4 rhythm reward: periodicity / entrainment energy.

The model of Large & Kolen (1994) describes meter perception as the
emergent behavior of a *bank of nonlinear oscillators* driven by an
onset train: oscillators with natural frequencies at and around the
period of the input phase-lock to it, and a hierarchy of harmonic /
subharmonic oscillators also entrain (giving rise to perceived meter).

For a tabula-rasa reward we use a tractable *linear* approximation:
a bank of damped harmonic oscillators

    ẍ + 2 ζ ω₀ ẋ + ω₀² x = s(t)

with natural frequencies ω₀ spanning the musical tempo range
(30–480 BPM). The steady-state amplitude of each oscillator driven by
an onset train s(t) is computed in closed form via the FFT, since the
transfer function is

    H(ω) = 1 / (ω₀² − ω² + 2j ζ ω₀ ω).

Summing the energy across all oscillators rewards onset patterns
whose spectrum has power concentrated at *some* low-frequency
oscillator resonance — i.e. patterns that are *periodic in the tempo
range*. Random onset trains have spectrally flat power and entrain
weakly; regular pulse trains and hierarchical meters entrain strongly.

This loses Large-Kolen's nonlinear cross-resonance at integer-ratio
harmonics, but it preserves the central qualitative property —
periodicity in the tempo range is what the reward measures — and is
fast enough to use inside a REINFORCE loop.

References:
  Large, E. W. & Kolen, J. F. (1994). "Resonance and the perception of
  musical meter." Connection Science 6(2-3), 177-208.
"""
from typing import Sequence

import numpy as np


_DEFAULT_TEMPO_BAND_BPM = (40.0, 320.0)


def _impulse_train(onsets: Sequence[float], duration: float,
                   sample_rate: int) -> np.ndarray:
    """Render onset times into a binary impulse-train signal."""
    n = int(duration * sample_rate)
    s = np.zeros(n, dtype=np.float64)
    for t in onsets:
        if 0.0 <= t < duration:
            idx = int(t * sample_rate)
            if 0 <= idx < n:
                s[idx] += 1.0
    return s


def _oscillator_energies(onsets: Sequence[float],
                         duration: float,
                         sample_rate: int,
                         n_oscillators: int,
                         tempo_band_bpm: tuple,
                         damping: float):
    """Return the steady-state energy delivered to each oscillator."""
    s = _impulse_train(onsets, duration, sample_rate)
    if s.sum() == 0:
        return np.zeros(n_oscillators), np.zeros(n_oscillators), 0.0
    X = np.fft.rfft(s)
    freqs_hz = np.fft.rfftfreq(len(s), d=1.0 / sample_rate)
    omega = 2.0 * np.pi * freqs_hz

    omega_min = 2.0 * np.pi * tempo_band_bpm[0] / 60.0
    omega_max = 2.0 * np.pi * tempo_band_bpm[1] / 60.0
    omega_0 = np.exp(np.linspace(np.log(omega_min), np.log(omega_max),
                                  n_oscillators))

    denom = (omega_0[:, None] ** 2 - omega[None, :] ** 2) \
            + 2j * damping * omega_0[:, None] * omega[None, :]
    H = 1.0 / denom
    energies = np.sum(np.abs(X[None, :] * H) ** 2, axis=1)
    sig_energy = float(np.sum(np.abs(X) ** 2))
    return energies, omega_0, sig_energy


def entrainment_peak_ratio(onsets: Sequence[float],
                           duration: float = 4.0,
                           sample_rate: int = 200,
                           n_oscillators: int = 60,
                           tempo_band_bpm: tuple = _DEFAULT_TEMPO_BAND_BPM,
                           damping: float = 0.07) -> float:
    """
    Peak-to-mean oscillator-bank response.

    The total energy summed across the bank is roughly conserved
    (Parseval-like), so it does *not* distinguish a periodic signal
    from a random one. What does distinguish them is *concentration*:
    a periodic onset train delivers most of its energy to a single
    oscillator at the matching natural frequency, while a random
    train spreads energy across the bank. We score this as
    max(E_i) / mean(E_i) − 1, which is 0 for a flat response and
    grows as the response concentrates on a single resonance.
    """
    energies, _, _ = _oscillator_energies(
        onsets, duration, sample_rate, n_oscillators,
        tempo_band_bpm, damping)
    if energies.sum() == 0:
        return 0.0
    mean_e = energies.mean()
    if mean_e <= 0:
        return 0.0
    return float(energies.max() / mean_e - 1.0)


def entrainment_energy(onsets: Sequence[float],
                       duration: float = 4.0,
                       sample_rate: int = 200,
                       n_oscillators: int = 60,
                       tempo_band_bpm: tuple = _DEFAULT_TEMPO_BAND_BPM,
                       damping: float = 0.07) -> float:
    """Total energy delivered to the oscillator bank (Parseval-bounded)."""
    energies, _, _ = _oscillator_energies(
        onsets, duration, sample_rate, n_oscillators,
        tempo_band_bpm, damping)
    return float(energies.sum())


def phase_coherence(onsets: Sequence[float],
                    period_candidates: np.ndarray = None,
                    period_min_sec: float = 0.2,
                    period_max_sec: float = 1.5,
                    n_periods: int = 80) -> float:
    """
    Smooth periodicity measure: max over candidate periods T of the
    phase-coherence of the onset times mod T.

    For a perfectly periodic pulse train at period T, every onset has
    the same phase mod T and the resultant magnitude

        |⟨exp(2πi · t/T)⟩|

    equals 1. For a uniformly random train this magnitude is
    O(1/√N). Taking the max over candidate periods picks out the
    natural period of the input — same qualitative information as
    autocorrelation, but smooth in the onset times so REINFORCE has a
    usable gradient.
    """
    o = np.asarray(onsets, dtype=np.float64)
    if len(o) < 2:
        return 0.0
    if period_candidates is None:
        period_candidates = np.linspace(period_min_sec, period_max_sec,
                                        n_periods)
    # Vectorize over (period, onset)
    phases = (o[None, :] / period_candidates[:, None]) % 1.0
    z = np.exp(2j * np.pi * phases).mean(axis=1)
    return float(np.abs(z).max())


def best_period_phase(onsets: Sequence[float],
                      period_min_sec: float = 0.2,
                      period_max_sec: float = 1.5,
                      n_periods: int = 200) -> float:
    """Argmax-period using the phase-coherence measure."""
    o = np.asarray(onsets, dtype=np.float64)
    if len(o) < 2:
        return 0.0
    periods = np.linspace(period_min_sec, period_max_sec, n_periods)
    phases = (o[None, :] / periods[:, None]) % 1.0
    z = np.exp(2j * np.pi * phases).mean(axis=1)
    return float(periods[int(np.argmax(np.abs(z)))])


def autocorr_peak(onsets: Sequence[float],
                  duration: float = 4.0,
                  sample_rate: int = 200,
                  min_lag_sec: float = 0.2,
                  max_lag_sec: float = 2.0) -> float:
    """
    Maximum autocorrelation of the impulse train at a non-trivial lag,
    normalized by the zero-lag value.

    For a regular onset train at period T, the autocorrelation at
    lag = T equals (number_of_aligned_pairs) ≈ N − 1, so the
    normalized peak approaches 1 in the limit of many onsets. For a
    uniformly random train of N pulses over duration D, the expected
    autocorrelation at any non-zero lag is ~N · ΔT / D, which is much
    smaller. The `min_lag_sec` cutoff excludes degenerate "very fast"
    rhythms (everything clustered), keeping the reward focused on the
    actual musical tempo range.
    """
    s = _impulse_train(onsets, duration, sample_rate)
    if s.sum() == 0:
        return 0.0
    ac = np.correlate(s, s, mode="full")
    n = len(s)
    ac = ac[n - 1 :]
    min_lag = int(min_lag_sec * sample_rate)
    max_lag = min(int(max_lag_sec * sample_rate), len(ac))
    if min_lag >= max_lag:
        return 0.0
    ac0 = ac[0]
    if ac0 <= 0:
        return 0.0
    return float(ac[min_lag:max_lag].max() / ac0)


def best_period(onsets: Sequence[float],
                duration: float = 4.0,
                sample_rate: int = 200,
                min_lag_sec: float = 0.2,
                max_lag_sec: float = 2.0) -> float:
    """Argmax-lag in seconds; diagnostic only."""
    s = _impulse_train(onsets, duration, sample_rate)
    if s.sum() == 0:
        return 0.0
    ac = np.correlate(s, s, mode="full")
    n = len(s)
    ac = ac[n - 1 :]
    min_lag = int(min_lag_sec * sample_rate)
    max_lag = min(int(max_lag_sec * sample_rate), len(ac))
    if min_lag >= max_lag:
        return 0.0
    rel = ac[min_lag:max_lag]
    return float((min_lag + int(np.argmax(rel))) / sample_rate)


def beat_count_diversity(onsets: Sequence[float],
                         min_count: int = 4) -> float:
    """
    Hinge penalty for very sparse onset patterns. We do not want the
    reward to be trivially won by emitting one onset (or none), so we
    require at least `min_count` distinct onsets falling inside the
    window.
    """
    if len(onsets) < min_count:
        return float(min_count - len(onsets))
    return 0.0


def inter_onset_intervals(onsets: Sequence[float]) -> np.ndarray:
    o = np.sort(np.asarray(onsets, dtype=np.float64))
    return np.diff(o)


def rhythm_reward(onsets: Sequence[float],
                  duration: float = 4.0,
                  entrainment_weight: float = 4.0,
                  diversity_weight: float = 0.5,
                  min_onsets: int = 4,
                  min_ioi: float = 0.07,
                  sparsity_weight: float = 0.5,
                  min_lag_sec: float = 0.2,
                  max_lag_sec: float = 2.0) -> float:
    """
    Composite Phase-4 reward:

      λ_entrain · autocorr_peak(onsets)
      − λ_diversity · max(0, min_onsets − N)
      − λ_sparsity · #(IOIs below min_ioi)

    `autocorr_peak` is the normalized autocorrelation maximum of the
    onset impulse train within the musical-tempo lag window
    [min_lag_sec, max_lag_sec]. It approaches 1 for a regular pulse
    train and is much smaller for random or clustered patterns. The
    sparsity / min-IOI term forbids the model from collapsing onsets
    onto the same time, and the diversity term enforces a minimum
    number of beats inside the window. Otherwise the model has no
    music prior — only periodicity in the tempo range, which is the
    qualitative property captured by the Large–Kolen oscillator model.
    """
    e = phase_coherence(onsets,
                        period_min_sec=min_lag_sec,
                        period_max_sec=max_lag_sec)
    d = beat_count_diversity(onsets, min_count=min_onsets)
    iois = inter_onset_intervals(onsets)
    too_close = float(np.sum(iois < min_ioi))
    return float(
        entrainment_weight * e
        - diversity_weight * d
        - sparsity_weight * too_close
    )
