"""
Rule-based music theory judge — replaces Qwen RLAIF scoring.

Eight modular reward components operating on raw frequency/duration/velocity
arrays. Each is a pure function returning a float reward. The composite
theory_reward() combines them with configurable weights.

Scoring operates on musical *structure*, not rendered audio.
"""
import numpy as np
from typing import Optional, Sequence, Tuple

# --------------- scale / pitch utilities ---------------

SCALES = {
    "major":            (0, 2, 4, 5, 7, 9, 11),
    "natural_minor":    (0, 2, 3, 5, 7, 8, 10),
    "harmonic_minor":   (0, 2, 3, 5, 7, 8, 11),
    "melodic_minor":    (0, 2, 3, 5, 7, 9, 11),
    "pentatonic_major": (0, 2, 4, 7, 9),
    "pentatonic_minor": (0, 3, 5, 7, 10),
    "dorian":           (0, 2, 3, 5, 7, 9, 10),
    "mixolydian":       (0, 2, 4, 5, 7, 9, 10),
    "blues":            (0, 3, 5, 6, 7, 10),
}

INTERVAL_CONSONANCE = {
    0: 1.0,    # unison
    1: 0.1,    # minor 2nd
    2: 0.3,    # major 2nd
    3: 0.7,    # minor 3rd
    4: 0.7,    # major 3rd
    5: 0.8,    # perfect 4th
    6: 0.2,    # tritone
    7: 0.9,    # perfect 5th
    8: 0.6,    # minor 6th
    9: 0.6,    # major 6th
    10: 0.3,   # minor 7th
    11: 0.4,   # major 7th
}


def _freqs_to_semitones(freqs: np.ndarray, ref_hz: float = 261.63) -> np.ndarray:
    f = np.asarray(freqs, dtype=np.float64)
    f = np.clip(f, 20.0, 20000.0)
    return 12.0 * np.log2(f / ref_hz)


def _freqs_to_pitch_classes(freqs: np.ndarray) -> np.ndarray:
    return _freqs_to_semitones(freqs) % 12.0


def _detect_key(freqs: np.ndarray) -> Tuple[float, str, Tuple[int, ...]]:
    """Auto-detect the most likely key (root_hz, scale_name, scale_degrees).

    Tests all 12 roots x all scales, picks the one with lowest total deviation.
    """
    pcs = _freqs_to_pitch_classes(freqs)
    best_score = float("inf")
    best = (261.63, "major", SCALES["major"])

    for root_offset in range(12):
        shifted = (pcs - root_offset) % 12.0
        for name, degrees in SCALES.items():
            deg_arr = np.array(degrees, dtype=np.float64)
            # min distance from each note to nearest scale degree
            dists = np.array([
                min(abs(s - d) for d in deg_arr)
                for s in shifted
            ])
            score = float(dists.sum())
            if score < best_score:
                best_score = score
                root_hz = 261.63 * 2.0 ** (root_offset / 12.0)
                best = (root_hz, name, degrees)

    return best


# --------------- Module 1: Key Adherence ---------------

def key_adherence(freqs: np.ndarray,
                  root_hz: Optional[float] = None,
                  scale: Optional[Tuple[int, ...]] = None) -> float:
    """Reward for staying in key. Returns negative mean deviation in cents.

    Higher (closer to 0) = better adherence. Typical range: [-50, 0].
    """
    f = np.asarray(freqs, dtype=np.float64)
    if len(f) < 2:
        return 0.0

    if root_hz is None or scale is None:
        root_hz, _, scale = _detect_key(f)

    semitones = _freqs_to_semitones(f, ref_hz=root_hz)
    pcs = semitones % 12.0
    deg_arr = np.array(scale, dtype=np.float64)

    deviations = np.array([
        min(abs(pc - d) if abs(pc - d) <= 6 else 12 - abs(pc - d)
            for d in deg_arr)
        for pc in pcs
    ])

    # Transform to [0, 1]: perfectly in key = 1.0, 1+ semitone average deviation ≈ 0
    mean_dev = float(deviations.mean())
    return float(np.exp(-mean_dev * 3.0))


# --------------- Module 2: Interval Quality ---------------

def interval_quality(freqs: np.ndarray) -> float:
    """Reward based on consonance class of consecutive intervals.

    Returns mean consonance score in [0, 1].
    """
    f = np.asarray(freqs, dtype=np.float64)
    if len(f) < 2:
        return 0.0

    semitones = np.abs(_freqs_to_semitones(f[1:]) - _freqs_to_semitones(f[:-1]))
    interval_classes = np.round(semitones) % 12
    scores = np.array([
        INTERVAL_CONSONANCE.get(int(ic), 0.3)
        for ic in interval_classes
    ])
    return float(scores.mean())


# --------------- Module 3: Melodic Contour ---------------

def _motif_repetition(intervals: np.ndarray, max_motif: int = 4) -> float:
    """Detect short repeated interval patterns (motifs) in a melody.

    Checks if any sub-sequence of 2-4 intervals appears more than once.
    Returns score in [0, 1] based on the fraction of the melody covered
    by repeated motifs.
    """
    n = len(intervals)
    if n < 4:
        return 0.0

    # Quantize intervals to nearest semitone for pattern matching
    quantized = np.round(intervals).astype(int)
    best_coverage = 0.0

    for motif_len in range(2, min(max_motif + 1, n // 2 + 1)):
        for start in range(n - motif_len + 1):
            motif = tuple(quantized[start:start + motif_len])
            count = 0
            i = 0
            while i <= n - motif_len:
                if tuple(quantized[i:i + motif_len]) == motif:
                    count += 1
                    i += motif_len  # non-overlapping
                else:
                    i += 1
            if count >= 2:
                coverage = (count * motif_len) / n
                best_coverage = max(best_coverage, coverage)

    return min(best_coverage, 1.0)


def melodic_contour(freqs: np.ndarray) -> float:
    """Reward for singable melody shape.

    Components:
    - Stepwise motion preference (peak at 1-3 semitones)
    - Gap-fill: large leaps followed by step in opposite direction
    - Single climax point near 60-75% through phrase
    - Penalize flat/stuck melodies
    """
    f = np.asarray(freqs, dtype=np.float64)
    n = len(f)
    if n < 3:
        return 0.0

    semitones = _freqs_to_semitones(f)
    intervals = np.diff(semitones)
    abs_intervals = np.abs(intervals)

    # Stepwise preference: Gaussian centered at 2 semitones, sigma=2
    step_reward = float(np.exp(-((abs_intervals - 2.0) ** 2) / 8.0).mean())

    # Gap-fill: after a leap (>4 semitones), next interval should be in opposite direction
    gap_fill_score = 0.0
    gap_fill_count = 0
    for i in range(len(intervals) - 1):
        if abs_intervals[i] > 4.0:
            gap_fill_count += 1
            if intervals[i] * intervals[i + 1] < 0:  # opposite direction
                gap_fill_score += 1.0
            if abs_intervals[i + 1] <= 3.0:  # step after leap
                gap_fill_score += 0.5
    gap_fill = (gap_fill_score / max(gap_fill_count, 1))

    # Climax: highest note should appear roughly 60-75% through
    peak_idx = int(np.argmax(semitones))
    ideal_peak = n * 0.67
    peak_closeness = 1.0 - min(abs(peak_idx - ideal_peak) / n, 1.0)

    # Anti-flatness: penalize low pitch range
    pitch_range = float(semitones.max() - semitones.min())
    range_score = min(pitch_range / 12.0, 1.0)  # reward up to 1 octave range

    # Motif repetition: reward short interval patterns that recur
    motif_score = _motif_repetition(intervals)

    return (0.25 * step_reward + 0.20 * gap_fill + 0.15 * peak_closeness
            + 0.15 * range_score + 0.25 * motif_score)


# --------------- Module 4: Cadence Detection ---------------

def _pc_distance(a: float, b: float) -> float:
    """Shortest distance between two pitch classes on the circle of 12."""
    d = abs(a - b) % 12.0
    return min(d, 12.0 - d)


# Cadence patterns: (pre-end pitch class, end pitch class, reward)
# Pitch classes relative to key root (0 = tonic)
_CADENCE_PATTERNS = [
    (7.0, 0.0, 1.0),    # V-I (authentic) — strongest
    (5.0, 0.0, 0.7),    # IV-I (plagal)
    (2.0, 7.0, 0.5),    # ii-V (half of ii-V-I, pre-dominant)
    (7.0, 9.0, 0.3),    # V-vi (deceptive)
    (11.0, 0.0, 0.6),   # vii°-I (leading tone resolution)
]


def cadence_detection(freqs: np.ndarray,
                      durations: Optional[np.ndarray] = None,
                      phrase_len: int = 4,
                      _root_hz: Optional[float] = None,
                      _scale: Optional[Tuple[int, ...]] = None) -> float:
    """Reward for cadential patterns at phrase boundaries.

    Recognizes: V-I (authentic), IV-I (plagal), ii-V (pre-dominant),
    V-vi (deceptive), vii-I (leading tone). Also rewards agogic accent
    and final note landing on tonic.
    """
    f = np.asarray(freqs, dtype=np.float64)
    n = len(f)
    if n < phrase_len:
        return 0.0

    # Always try the last note and first note as candidate roots in
    # addition to the detected/provided key. Short sequences often have
    # ambiguous key detection, but cadences resolve to specific notes.
    if _root_hz is not None:
        candidate_roots = [_root_hz]
    else:
        det_root, _, _ = _detect_key(f)
        candidate_roots = [det_root]
    for freq in [f[-1], f[0]]:
        if not any(abs(freq - r) < 1.0 for r in candidate_roots):
            candidate_roots.append(float(freq))

    best_total = 0.0
    for root_hz in candidate_roots:
        semitones = _freqs_to_semitones(f, ref_hz=root_hz)
        pcs = semitones % 12.0

        cadence_score = 0.0
        n_phrases = 0

        for phrase_end in range(phrase_len - 1, n, phrase_len):
            n_phrases += 1
            end_pc = pcs[phrase_end]
            pre_pc = pcs[phrase_end - 1] if phrase_end > 0 else end_pc

            # Find best-matching cadence pattern
            best_pattern_score = 0.0
            for target_pre, target_end, reward in _CADENCE_PATTERNS:
                pre_dist = _pc_distance(pre_pc, target_pre)
                end_dist = _pc_distance(end_pc, target_end)
                if pre_dist < 1.5 and end_dist < 1.5:
                    match_quality = (1.0 - pre_dist / 1.5) * (1.0 - end_dist / 1.5)
                    best_pattern_score = max(best_pattern_score, reward * match_quality)

            cadence_score += best_pattern_score

            # Bonus: final note on tonic
            if _pc_distance(end_pc, 0.0) < 0.5:
                cadence_score += 0.2

            # Agogic accent: longer note at phrase end
            if durations is not None:
                d = np.asarray(durations, dtype=np.float64)
                if phrase_end < len(d):
                    phrase_start = max(0, phrase_end - phrase_len + 1)
                    phrase_durs = d[phrase_start:phrase_end + 1]
                    if len(phrase_durs) > 1 and d[phrase_end] > phrase_durs[:-1].mean():
                        cadence_score += 0.2

        best_total = max(best_total, cadence_score / max(n_phrases, 1))

    return float(best_total)


# --------------- Module 5: Voice Leading ---------------

def voice_leading(voices: np.ndarray) -> float:
    """Reward for proper voice leading in multi-voice textures.

    voices: shape (V, N) — V voices, N notes each, in Hz.

    Penalizes parallel 5ths and octaves, rewards contrary motion,
    penalizes voice crossings.
    """
    v = np.asarray(voices, dtype=np.float64)
    if v.ndim != 2 or v.shape[0] < 2:
        return 0.0

    n_voices, n_notes = v.shape
    if n_notes < 2:
        return 0.0

    semitones = np.array([_freqs_to_semitones(v[i]) for i in range(n_voices)])
    total_score = 0.0
    n_pairs = 0

    for i in range(n_voices):
        for j in range(i + 1, n_voices):
            n_pairs += 1
            intervals = (semitones[j] - semitones[i]) % 12.0
            motions_i = np.diff(semitones[i])
            motions_j = np.diff(semitones[j])

            pair_score = 0.0
            for t in range(n_notes - 1):
                curr_interval = intervals[t]
                next_interval = intervals[t + 1]

                # Parallel 5ths/octaves: both voices move in same direction
                # to another 5th/octave
                is_parallel = (motions_i[t] * motions_j[t] > 0 and
                               abs(motions_i[t] - motions_j[t]) < 0.5)
                is_fifth_or_octave = (abs(next_interval - 7.0) < 0.5 or
                                      abs(next_interval) < 0.5 or
                                      abs(next_interval - 12.0) < 0.5)

                if is_parallel and is_fifth_or_octave:
                    pair_score -= 1.0  # parallel 5th/octave penalty
                elif motions_i[t] * motions_j[t] < 0:
                    pair_score += 0.3  # contrary motion bonus
                elif abs(motions_i[t] * motions_j[t]) < 0.01:
                    pair_score += 0.1  # oblique motion (one voice holds)

            # Voice crossing penalty
            crossings = np.sum(semitones[i][:-1] > semitones[j][:-1])
            if i < j:  # lower voice index should be lower pitch
                pair_score -= 0.2 * crossings

            total_score += pair_score / max(n_notes - 1, 1)

    return float(total_score / max(n_pairs, 1))


# --------------- Module 6: Rhythm Analysis ---------------

def rhythm_analysis(durations: np.ndarray) -> float:
    """Reward for rhythmic coherence.

    Components:
    - Duration clustering: durations should fall on a tempo grid
    - Pattern repetition via autocorrelation
    - Anti-monotony: not all same, not all different
    """
    d = np.asarray(durations, dtype=np.float64)
    if len(d) < 3:
        return 0.0

    # Grid coherence: find best-fit unit duration, measure fit
    # Try common subdivisions of the median duration
    median_d = float(np.median(d))
    if median_d < 1e-6:
        return 0.0

    ratios = d / median_d
    # How close are ratios to simple fractions (1, 0.5, 2, 1.5, 0.75)?
    grid_targets = np.array([0.5, 0.75, 1.0, 1.5, 2.0])
    grid_dists = np.array([
        min(abs(r - g) for g in grid_targets)
        for r in ratios
    ])
    grid_score = float(np.exp(-grid_dists.mean() * 4.0))

    # Pattern repetition: autocorrelation of duration sequence
    d_centered = d - d.mean()
    norm = float(np.sum(d_centered ** 2))
    if norm < 1e-9:
        autocorr_score = 0.0
    else:
        # Check lags 2, 3, 4 for repetition
        best_corr = 0.0
        for lag in [2, 3, 4]:
            if lag < len(d_centered):
                corr = float(np.sum(d_centered[:-lag] * d_centered[lag:])) / norm
                best_corr = max(best_corr, corr)
        autocorr_score = max(0.0, best_corr)

    # Anti-monotony: coefficient of variation in sweet spot [0.15, 0.5]
    cv = float(d.std() / d.mean()) if d.mean() > 1e-6 else 0.0
    if cv < 0.05:
        variety_score = 0.0  # all same duration
    elif cv > 0.8:
        variety_score = 0.2  # too chaotic
    else:
        variety_score = min(cv / 0.4, 1.0)

    return 0.4 * grid_score + 0.3 * autocorr_score + 0.3 * variety_score


# --------------- Module 7: Tension-Resolution ---------------

def tension_resolution(freqs: np.ndarray,
                       durations: Optional[np.ndarray] = None) -> float:
    """Reward for tension arcs: build in first 60-75%, resolve in final 25-40%.

    Tension is measured by cumulative dissonance of intervals.
    Resolution = drop in tension toward the end.
    """
    f = np.asarray(freqs, dtype=np.float64)
    n = len(f)
    if n < 4:
        return 0.0

    semitones = _freqs_to_semitones(f)
    intervals = np.abs(np.diff(semitones))
    interval_classes = np.round(intervals) % 12

    # Per-step tension: inverse of consonance
    tension_per_step = np.array([
        1.0 - INTERVAL_CONSONANCE.get(int(ic), 0.3)
        for ic in interval_classes
    ])

    # Running tension with decay
    running = np.zeros(len(tension_per_step))
    decay = 0.7
    running[0] = tension_per_step[0]
    for i in range(1, len(running)):
        running[i] = decay * running[i - 1] + tension_per_step[i]

    if len(running) < 3:
        return 0.0

    # Split into build phase (first 70%) and resolve phase (last 30%)
    split = max(2, int(len(running) * 0.7))
    split = min(split, len(running) - 2)  # ensure resolve has at least 2 points
    build_phase = running[:split]
    resolve_phase = running[split:]

    # Build reward: tension should generally increase
    # Use sigmoid instead of hard clip for smoother gradient
    if len(build_phase) >= 2:
        build_trend = float(np.polyfit(np.arange(len(build_phase)), build_phase, 1)[0])
        build_score = float(1.0 / (1.0 + np.exp(-build_trend * 8.0)))
    else:
        build_score = 0.5

    # Resolve reward: tension should decrease
    if len(resolve_phase) >= 2:
        resolve_trend = float(np.polyfit(np.arange(len(resolve_phase)), resolve_phase, 1)[0])
        resolve_score = float(1.0 / (1.0 + np.exp(resolve_trend * 8.0)))
    else:
        resolve_score = 0.5

    # Final note should be consonant (low tension)
    final_tension = float(tension_per_step[-1])
    resolution_bonus = float(np.exp(-final_tension * 2.0))

    return 0.35 * build_score + 0.35 * resolve_score + 0.3 * resolution_bonus


# --------------- Module 8: Dynamic Shaping ---------------

def dynamic_shaping(velocities: np.ndarray) -> float:
    """Reward for musical velocity contours.

    Components:
    - Phrasing arcs (crescendo/decrescendo shapes)
    - Metric accents (stronger on beats 1 and 3 in groups of 4)
    - Dynamic range (not all loud, not all quiet)
    """
    v = np.asarray(velocities, dtype=np.float64)
    n = len(v)
    if n < 3:
        return 0.0

    # Phrasing arc: reward bell-shaped or arch contours
    # Compare to ideal arch: sin(pi * t / n)
    ideal_arch = np.sin(np.linspace(0, np.pi, n))
    # Normalize both to [0,1]
    v_range = v.max() - v.min()
    if v_range < 1e-6:
        arch_score = 0.0  # flat dynamics = no arch
    else:
        v_norm = (v - v.min()) / v_range
        arch_corr = float(np.corrcoef(v_norm, ideal_arch)[0, 1])
        arch_score = max(0.0, arch_corr) if not np.isnan(arch_corr) else 0.0

    # Metric accents: in groups of 4, beats 0 and 2 should be louder
    accent_score = 0.0
    n_groups = 0
    for start in range(0, n - 3, 4):
        group = v[start:start + 4]
        if len(group) == 4:
            n_groups += 1
            strong = (group[0] + group[2]) / 2.0
            weak = (group[1] + group[3]) / 2.0
            if strong > weak:
                accent_score += 1.0
    accent_score = accent_score / max(n_groups, 1)

    # Dynamic range: should be moderate (not flat, not extreme)
    dyn_range = float(v.max() - v.min())
    if dyn_range < 0.05:
        range_score = 0.0  # flat dynamics
    elif dyn_range > 0.7:
        range_score = 0.7  # extreme range, slight penalty
    else:
        range_score = min(dyn_range / 0.4, 1.0)

    return 0.4 * arch_score + 0.3 * accent_score + 0.3 * range_score


# --------------- Composite Theory Reward ---------------

DEFAULT_WEIGHTS = {
    "key_adherence": 3.0,
    "interval_quality": 2.0,
    "melodic_contour": 1.5,
    "cadence_detection": 1.0,
    "voice_leading": 1.0,
    "rhythm_analysis": 1.0,
    "tension_resolution": 0.8,
    "dynamic_shaping": 0.5,
}


def theory_reward(freqs: np.ndarray,
                  durations: Optional[np.ndarray] = None,
                  velocities: Optional[np.ndarray] = None,
                  voices: Optional[np.ndarray] = None,
                  weights: Optional[dict] = None,
                  root_hz: Optional[float] = None,
                  scale: Optional[Tuple[int, ...]] = None) -> float:
    """Composite theory reward — weighted sum of all modules.

    Args:
        freqs: shape (N,) melody frequencies in Hz
        durations: shape (N,) note durations in seconds (optional)
        velocities: shape (N,) note velocities in [0, 1] (optional)
        voices: shape (V, N) multi-voice frequencies (optional, for voice_leading)
        weights: override DEFAULT_WEIGHTS
        root_hz: root frequency for key analysis (auto-detected if None)
        scale: scale degrees (auto-detected if None)

    Returns:
        Composite reward (float).
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    # Detect key once, share across modules that need it
    if root_hz is None or scale is None:
        root_hz, _, scale = _detect_key(np.asarray(freqs))

    score = 0.0
    score += w["key_adherence"] * key_adherence(freqs, root_hz, scale)
    score += w["interval_quality"] * interval_quality(freqs)
    score += w["melodic_contour"] * melodic_contour(freqs)
    score += w["cadence_detection"] * cadence_detection(freqs, durations,
                                                         _root_hz=root_hz,
                                                         _scale=scale)

    if voices is not None and voices.ndim == 2 and voices.shape[0] >= 2:
        score += w["voice_leading"] * voice_leading(voices)

    if durations is not None:
        score += w["rhythm_analysis"] * rhythm_analysis(durations)

    if durations is not None:
        score += w["tension_resolution"] * tension_resolution(freqs, durations)

    if velocities is not None:
        score += w["dynamic_shaping"] * dynamic_shaping(velocities)

    return float(score)


def theory_reward_per_note(freqs: np.ndarray,
                           durations: Optional[np.ndarray] = None,
                           velocities: Optional[np.ndarray] = None,
                           root_hz: Optional[float] = None,
                           scale: Optional[Tuple[int, ...]] = None,
                           ) -> np.ndarray:
    """Per-note reward for fine-grained REINFORCE credit assignment.

    Returns shape (N,) combining local pitch, rhythm, and dynamics signals.
    Key adherence is heavily weighted since it's the primary quality driver.
    """
    f = np.asarray(freqs, dtype=np.float64)
    n = len(f)
    if n < 2:
        return np.zeros(n, dtype=np.float64)

    if root_hz is None or scale is None:
        root_hz, _, scale = _detect_key(f)

    # --- Per-note key adherence (dominant signal) ---
    semitones = _freqs_to_semitones(f, ref_hz=root_hz)
    pcs = semitones % 12.0
    deg_arr = np.array(scale, dtype=np.float64)
    key_per_note = np.array([
        float(np.exp(-min(abs(pc - d) if abs(pc - d) <= 6 else 12 - abs(pc - d)
                         for d in deg_arr) * 3.0))
        for pc in pcs
    ])

    # --- Per-note interval consonance ---
    all_semitones = _freqs_to_semitones(f)
    raw_intervals = np.abs(np.diff(all_semitones))
    interval_classes = np.round(raw_intervals) % 12
    consonance = np.array([
        INTERVAL_CONSONANCE.get(int(ic), 0.3)
        for ic in interval_classes
    ])
    interval_per_note = np.zeros(n)
    interval_per_note[:-1] += consonance * 0.5
    interval_per_note[1:] += consonance * 0.5

    # --- Per-note stepwise motion ---
    step_per_note = np.zeros(n)
    for i in range(len(raw_intervals)):
        if raw_intervals[i] <= 2.5:
            step_per_note[i] += 0.5
            step_per_note[i + 1] += 0.5
        elif raw_intervals[i] > 7.0:
            step_per_note[i] -= 0.15
            step_per_note[i + 1] -= 0.15
    step_per_note = np.clip(step_per_note, 0.0, 1.0)

    # --- Per-note rhythm (grid alignment) ---
    rhythm_per_note = np.zeros(n)
    if durations is not None:
        d = np.asarray(durations, dtype=np.float64)
        if len(d) == n:
            median_d = float(np.median(d))
            if median_d > 1e-6:
                ratios = d / median_d
                grid_targets = np.array([0.5, 0.75, 1.0, 1.5, 2.0])
                for i_n in range(n):
                    dist = min(abs(ratios[i_n] - g) for g in grid_targets)
                    rhythm_per_note[i_n] = float(np.exp(-dist * 4.0))

    # --- Per-note dynamics (arch shape) ---
    dynamics_per_note = np.zeros(n)
    if velocities is not None:
        v = np.asarray(velocities, dtype=np.float64)
        if len(v) == n:
            ideal_arch = np.sin(np.linspace(0, np.pi, n))
            v_range = v.max() - v.min()
            if v_range > 1e-6:
                v_norm = (v - v.min()) / v_range
                dynamics_per_note = 1.0 - np.abs(v_norm - ideal_arch)
                dynamics_per_note = np.clip(dynamics_per_note, 0.0, 1.0)

    # --- Per-note cadence (phrase boundary resolution) ---
    cadence_per_note = np.zeros(n)
    phrase_len = 4
    root_st = _freqs_to_semitones(np.array([root_hz]), ref_hz=root_hz)[0]
    for phrase_end in range(phrase_len - 1, n, phrase_len):
        end_pc = pcs[phrase_end]
        # Reward notes at phrase endings that land on tonic
        tonic_dist = min(abs(end_pc - 0) if abs(end_pc - 0) <= 6
                         else 12 - abs(end_pc - 0), 6.0)
        cadence_per_note[phrase_end] = float(np.exp(-tonic_dist * 2.0))
        # Reward penultimate note being dominant (V) for V-I cadence
        if phrase_end > 0:
            pre_pc = pcs[phrase_end - 1]
            dom_dist = min(abs(pre_pc - 7) if abs(pre_pc - 7) <= 6
                           else 12 - abs(pre_pc - 7), 6.0)
            cadence_per_note[phrase_end - 1] += 0.5 * float(np.exp(-dom_dist * 2.0))
    # Final note tonic bonus
    final_pc = pcs[-1]
    tonic_dist = min(abs(final_pc - 0) if abs(final_pc - 0) <= 6
                     else 12 - abs(final_pc - 0), 6.0)
    cadence_per_note[-1] = max(cadence_per_note[-1],
                               float(np.exp(-tonic_dist * 1.5)))

    # Combine: key adherence dominates, all others contribute
    w_total = 5.0 + 2.0 + 1.5 + 1.0 + 0.5 + 1.0
    per_note = (5.0 * key_per_note
                + 2.0 * interval_per_note
                + 1.5 * step_per_note
                + 1.0 * rhythm_per_note
                + 0.5 * dynamics_per_note
                + 1.0 * cadence_per_note)
    per_note /= w_total

    return per_note.astype(np.float32)


def theory_reward_breakdown(freqs: np.ndarray,
                            durations: Optional[np.ndarray] = None,
                            velocities: Optional[np.ndarray] = None,
                            voices: Optional[np.ndarray] = None,
                            root_hz: Optional[float] = None,
                            scale: Optional[Tuple[int, ...]] = None) -> dict:
    """Return per-module scores for diagnostics."""
    if root_hz is None or scale is None:
        root_hz, _, scale = _detect_key(np.asarray(freqs))

    result = {
        "key_adherence": key_adherence(freqs, root_hz, scale),
        "interval_quality": interval_quality(freqs),
        "melodic_contour": melodic_contour(freqs),
        "cadence_detection": cadence_detection(freqs, durations,
                                                _root_hz=root_hz,
                                                _scale=scale),
        "tension_resolution": tension_resolution(freqs, durations) if durations is not None else 0.0,
        "rhythm_analysis": rhythm_analysis(durations) if durations is not None else 0.0,
        "dynamic_shaping": dynamic_shaping(velocities) if velocities is not None else 0.0,
    }
    if voices is not None and voices.ndim == 2 and voices.shape[0] >= 2:
        result["voice_leading"] = voice_leading(voices)
    return result
