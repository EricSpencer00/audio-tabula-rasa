"""
Phase-7 counterpoint reward.

For V voices each of length N notes:

  Σ_voice  melody_reward(voice)                          # horizontal
  − Σ_t  Σ_{i<j} Sethares(voice_i[t], voice_j[t])        # vertical harmony
  − λ_cross  · #(voice-crossings)                        # voice independence
  + λ_shared_root · shared_implied_fundamental_salience  # joint tonality

The vertical-harmony term reuses the Phase-1/2 Sethares model: every
pair of simultaneous notes contributes its full harmonic-series
roughness. This is what punishes near-unison clashes between voices
*without* hard-coding "voices live in different octaves".

The voice-crossing term counts time steps where the relative ordering
of the voices flips (so an upper voice doesn't dive below a lower
voice). This is a *task* constraint — the model has to maintain
voice identity — not a stylistic rule like "no parallel fifths".

The shared-root term measures Terhardt virtual-pitch salience using
the *concatenation* of all voices, so the model is rewarded for
finding pitches that collectively fit one implied tonal center.
"""
import numpy as np

from src.reward.psychoacoustic import (
    implied_fundamental_salience,
    melody_reward,
    total_dissonance,
)


def vertical_dissonance(voices, n_harmonics: int = 6) -> float:
    """
    Sum over time steps t, then over voice pairs (i,j), of pairwise
    Sethares roughness between voice_i[t] and voice_j[t].

    `voices` is shape (V, N).
    """
    v = np.asarray(voices, dtype=np.float64)
    V, N = v.shape
    total = 0.0
    for t in range(N):
        for i in range(V):
            for j in range(i + 1, V):
                total += total_dissonance(float(v[i, t]), float(v[j, t]),
                                          n_harmonics=n_harmonics)
    return total


def voice_crossings(voices) -> int:
    """
    Count time steps where the voice ordering differs from the order
    of the voices' median pitches.

    Using *median* instead of mean is more robust to outlier notes —
    we want to capture the dominant register of each voice. A
    counterpoint with stable voice roles (each voice stays in its own
    octave band) has zero crossings; jagged outputs where the upper
    voice dips below the lower one rack up crossings.
    """
    v = np.asarray(voices, dtype=np.float64)
    V, N = v.shape
    if V < 2:
        return 0
    target_rank = np.argsort(np.argsort(np.median(v, axis=1)))

    crossings = 0
    for t in range(N):
        rank_t = np.argsort(np.argsort(v[:, t]))
        if not np.array_equal(rank_t, target_rank):
            crossings += 1
    return int(crossings)


def voice_register_gap(voices, min_semitones: float = 3.0) -> float:
    """
    Hinge penalty: at every time step, the gap between adjacent voices
    (in their stable rank order) should exceed min_semitones. Penalty
    is the squared deficit.

    Strictly a *task* constraint to force voice independence; the
    vertical-dissonance term already discourages clashing pitches but
    has zero penalty at unison itself.
    """
    v = np.asarray(voices, dtype=np.float64)
    V, N = v.shape
    if V < 2:
        return 0.0
    total = 0.0
    for t in range(N):
        sorted_v = np.sort(v[:, t])
        log_gaps = np.log2(sorted_v[1:] / sorted_v[:-1]) * 12.0
        deficit = np.clip(min_semitones - log_gaps, 0.0, None)
        total += float(np.sum(deficit ** 2))
    return total / N


def shared_tonal_salience(voices) -> float:
    """Implied fundamental salience of the *concatenated* voice set."""
    v = np.asarray(voices, dtype=np.float64).flatten()
    return implied_fundamental_salience(v)


def counterpoint_reward(voices,
                        per_voice_weight: float = 1.0,
                        vertical_weight: float = 1.0,
                        crossing_weight: float = 0.5,
                        register_weight: float = 1.0,
                        shared_root_weight: float = 2.0,
                        n_harmonics: int = 6) -> float:
    """Composite reward for a V-voice counterpoint of shape (V, N)."""
    v = np.asarray(voices, dtype=np.float64)
    V = v.shape[0]
    horiz = 0.0
    for i in range(V):
        horiz += melody_reward(v[i])
    vert = vertical_dissonance(v, n_harmonics=n_harmonics)
    crosses = voice_crossings(v)
    reg = voice_register_gap(v)
    shared = shared_tonal_salience(v)
    return float(
        per_voice_weight * horiz
        - vertical_weight * vert
        - crossing_weight * crosses
        - register_weight * reg
        + shared_root_weight * shared
    )
