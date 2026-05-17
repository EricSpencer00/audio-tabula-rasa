"""
A drop-in stand-in for QwenAudioJudge for environments that can't run
the real model (no internet, < 14 GB RAM, etc).

Computes a 1-10 score from the audio features alone. The score is a
weighted combination of properties that empirically correlate with
"sounds less robotic" in the project so far:

  + dynamic range  (compressed mixes feel sterile)
  + key confidence (well-defined tonality reads as musical)
  + chroma concentration on a few PCs
  + a moderate, not-extreme IOI std (totally regular = mechanical;
                                       totally irregular = noise)
  + spectral centroid in a "natural" 1-4 kHz range
  − too-low onset density (silent / sparse stretches feel dead)

The exact weights are heuristic — this is *not* meant to replace the
audio-aware Qwen judge in any rigorous sense. It exists so the RLAIF
training loop can run end-to-end on a CPU container that lacks GPU,
internet, or 64 GB unified memory; when Qwen is available, swap the
reward source and the rest of the pipeline is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.analysis.audio_features import AudioFeatures, _read_wav


@dataclass
class MockJudgeResult:
    file: str
    score: int                 # 1..10
    breakdown: dict
    response: str

    def to_dict(self):
        return {"file": self.file, "score": self.score,
                "breakdown": self.breakdown, "response": self.response}


def _bell(x: float, target: float, sigma: float) -> float:
    """Gaussian bell centered at target, value in [0, 1]."""
    return float(np.exp(-((x - target) / sigma) ** 2))


def _ramp(x: float, low: float, high: float) -> float:
    """0 below low, linearly up to 1 at high, then 1."""
    if x <= low:
        return 0.0
    if x >= high:
        return 1.0
    return (x - low) / (high - low)


def _score_features(f: AudioFeatures) -> tuple[int, dict]:
    """Return (score 1-10, breakdown dict)."""
    s = f.summary()
    # Sub-scores in [0, 1]
    dyn_score = _ramp(s["dynamic_range_db"], 3.0, 18.0)
    key_score = max(0.0, min(1.0, s["key_confidence"]))
    chroma_arr = np.asarray(f.chroma)
    if chroma_arr.sum() > 0:
        # entropy-like spread: 1.0 if perfectly concentrated, 0 if uniform
        norm = chroma_arr / chroma_arr.sum()
        h = float(-np.sum(norm * np.log(norm + 1e-12))) / np.log(12)
        chroma_score = 1.0 - h
    else:
        chroma_score = 0.0
    # rhythmic regularity: peaks around CV~0.1-0.3
    if s["ioi_mean_s"] > 0:
        cv = s["ioi_std_s"] / max(s["ioi_mean_s"], 1e-3)
        rhythm_score = _bell(cv, target=0.2, sigma=0.25)
    else:
        rhythm_score = 0.0
    centroid_score = _bell(np.log10(max(s["spectral_centroid_hz"], 1e-3)),
                            target=np.log10(2000.0), sigma=0.5)
    onset_score = _ramp(s["onset_density_per_s"], 0.5, 5.0)

    breakdown = {
        "dynamic_range": round(dyn_score, 3),
        "key_confidence": round(key_score, 3),
        "chroma_concentration": round(chroma_score, 3),
        "rhythmic_naturalness": round(rhythm_score, 3),
        "spectral_centroid": round(centroid_score, 3),
        "onset_density": round(onset_score, 3),
    }
    weights = {
        "dynamic_range": 1.5,
        "key_confidence": 1.5,
        "chroma_concentration": 1.0,
        "rhythmic_naturalness": 1.0,
        "spectral_centroid": 0.7,
        "onset_density": 0.8,
    }
    raw = sum(breakdown[k] * w for k, w in weights.items())
    max_raw = sum(weights.values())
    norm = raw / max_raw  # in [0, 1]
    score = int(round(1 + 9 * norm))
    return score, breakdown


class MockAudioJudge:
    """API-compatible with QwenAudioJudge but uses extracted features."""

    DEFAULT_MODEL = "mock-features"
    SAMPLE_RATE = 22050   # we don't actually care, but kept for symmetry

    def __init__(self, *args, **kwargs):
        # Accept the same kwargs as QwenAudioJudge for drop-in use
        pass

    def score(self, audio: np.ndarray, sample_rate: int,
              **_kw) -> MockJudgeResult:
        # Build a minimal AudioFeatures-equivalent without re-reading a WAV
        # Easiest: wrap as a temp WAV in-memory? Too messy. Instead,
        # re-implement the bare minimum here.
        from src.analysis.audio_features import (
            _onset_envelope, _estimate_tempo, _chroma, _estimate_key,
            _onset_times, _spectral_stats,
        )
        env, env_sr = _onset_envelope(audio, sample_rate)
        tempo = _estimate_tempo(env, env_sr)
        chroma = _chroma(audio, sample_rate)
        root, mode, conf = _estimate_key(chroma)
        onsets = _onset_times(env, env_sr)
        iois = np.diff(onsets) if len(onsets) > 1 else np.array([])
        cent, bw = _spectral_stats(audio, sample_rate)
        peak = float(np.max(np.abs(audio))) + 1e-12
        rms_local = np.sqrt(
            np.convolve(audio ** 2, np.ones(2048) / 2048, mode="same")
        )
        rms_local = rms_local[rms_local > 1e-10]
        if len(rms_local) > 100:
            low = float(np.percentile(rms_local, 10))
            high = float(np.percentile(rms_local, 99))
            dyn_db = 20.0 * np.log10(max(high, 1e-9) / max(low, 1e-9))
        else:
            dyn_db = 0.0
        duration = len(audio) / max(1, sample_rate)
        feats = AudioFeatures(
            path="<inline>",
            sample_rate=sample_rate,
            duration_s=duration,
            peak_db=20 * float(np.log10(peak)),
            dynamic_range_db=float(dyn_db),
            estimated_tempo_bpm=float(tempo),
            estimated_key=root,
            estimated_mode=mode,
            key_confidence=float(conf),
            chroma=[float(c) for c in chroma],
            onset_count=int(len(onsets)),
            onset_density_per_s=(len(onsets) / duration) if duration else 0.0,
            ioi_mean_s=float(iois.mean()) if len(iois) else 0.0,
            ioi_std_s=float(iois.std()) if len(iois) else 0.0,
            spectral_centroid_hz=cent,
            spectral_bandwidth_hz=bw,
        )
        score, br = _score_features(feats)
        return MockJudgeResult(
            file="<inline>",
            score=score,
            breakdown=br,
            response=(f"mock judge: score {score}/10  breakdown={br}"),
        )

    def score_wav(self, path) -> MockJudgeResult:
        audio, sr = _read_wav(path)
        r = self.score(audio.astype(np.float32), sr)
        r.file = str(path)
        return r
