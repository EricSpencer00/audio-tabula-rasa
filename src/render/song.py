"""
Song-level composition: lay tracks on a shared tempo grid and render
the whole thing to one mixed waveform.

A `Song` has a tempo (BPM), a duration in bars, and N tracks. Each
track binds an instrument to a list of `Note` events. Notes carry an
absolute start time (in beats), a duration (in beats), a pitch (Hz),
and a velocity. The renderer realizes each note via its instrument
and mixes everything to a single mono waveform, normalized to a small
headroom.

Composition functions in this module take a few of the trained
generators (Phase 2 chord progressions, Phase 4 rhythm, Phase 3
melodies) and arrange their outputs into a 16-bar piece. No new
training. The arrangement layer is the new thing — the source notes
already exist in `results/`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import numpy as np

from src.render.instruments import (
    SAMPLE_RATE,
    PRESETS,
    get,
)


# ----- core data ------------------------------------------------------

@dataclass
class Note:
    pitch: float          # Hz; 0 → percussion
    start: float          # beats since song start
    duration: float       # beats
    velocity: float = 1.0


@dataclass
class Track:
    instrument_name: str
    notes: List[Note] = field(default_factory=list)
    gain: float = 1.0


@dataclass
class Song:
    tempo_bpm: float = 110.0
    n_bars: int = 16
    beats_per_bar: int = 4
    tracks: List[Track] = field(default_factory=list)

    @property
    def beat_seconds(self) -> float:
        return 60.0 / self.tempo_bpm

    @property
    def total_beats(self) -> float:
        return self.n_bars * self.beats_per_bar

    @property
    def total_seconds(self) -> float:
        return self.total_beats * self.beat_seconds

    def add(self, track: Track) -> "Song":
        self.tracks.append(track)
        return self

    def render(self) -> np.ndarray:
        bs = self.beat_seconds
        total_n = int(self.total_seconds * SAMPLE_RATE) + SAMPLE_RATE  # tail
        out = np.zeros(total_n, dtype=np.float64)

        for track in self.tracks:
            inst = get(track.instrument_name)
            for note in track.notes:
                start_s = note.start * bs
                dur_s = note.duration * bs
                rendered = inst.render(
                    freq=note.pitch, duration=dur_s,
                    velocity=note.velocity * track.gain,
                )
                start_idx = int(start_s * SAMPLE_RATE)
                end_idx = start_idx + len(rendered)
                if end_idx > total_n:
                    rendered = rendered[: total_n - start_idx]
                    end_idx = total_n
                out[start_idx:end_idx] += rendered

        return out


def normalize(audio: np.ndarray, headroom_db: float = -3.0) -> np.ndarray:
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-9:
        return audio
    target = 10 ** (headroom_db / 20.0)
    return audio * (target / peak)


def write_wav(path, audio: np.ndarray,
              sample_rate: int = SAMPLE_RATE) -> None:
    import wave
    audio = normalize(audio)
    pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


# ----- composition helpers -------------------------------------------

def quantize_to_scale(freqs: Sequence[float],
                      scale_pcs: Sequence[int],
                      reference_freq: float = 220.0) -> np.ndarray:
    """
    Snap each frequency to the nearest pitch in the given scale (set of
    pitch-class semitones, 0..11). Returns the snapped frequencies.

    `reference_freq` is the tuning anchor used to convert log-frequency
    to semitones.
    """
    fs = np.asarray(freqs, dtype=np.float64)
    semitones = 12.0 * np.log2(fs / reference_freq)
    scale_set = set(int(p) % 12 for p in scale_pcs)
    snapped = []
    for s in semitones:
        oct_n, pc = divmod(s, 12.0)
        candidates = [(oct_n - 1) * 12 + p
                      for p in scale_set] + [oct_n * 12 + p
                                              for p in scale_set] \
                      + [(oct_n + 1) * 12 + p for p in scale_set]
        nearest = min(candidates, key=lambda c: abs(c - s))
        snapped.append(reference_freq * 2.0 ** (nearest / 12.0))
    return np.asarray(snapped)


def basic_drum_pattern(song: Song, kick_pattern=(0, 2),
                       snare_pattern=(1, 3), hh_subdiv: int = 2) -> Song:
    """Add a 4/4 backbeat (kick on 1 & 3, snare on 2 & 4) and hi-hat
    on every `1/hh_subdiv` beat."""
    kick = Track(instrument_name="kick", gain=0.95)
    snare = Track(instrument_name="snare", gain=0.7)
    hihat = Track(instrument_name="hihat", gain=0.45)

    for bar in range(song.n_bars):
        bar_start = bar * song.beats_per_bar
        for beat in kick_pattern:
            kick.notes.append(Note(0.0, bar_start + beat, 0.5))
        for beat in snare_pattern:
            snare.notes.append(Note(0.0, bar_start + beat, 0.5))
        for tick in range(song.beats_per_bar * hh_subdiv):
            hihat.notes.append(
                Note(0.0, bar_start + tick / hh_subdiv, 0.25,
                     velocity=0.8 if tick % 2 == 0 else 0.5)
            )
    return song.add(kick).add(snare).add(hihat)


def arrange_chord_track(song: Song, chord_seqs: np.ndarray,
                        instrument: str = "pad",
                        bars_per_chord: int = 1) -> Song:
    """
    Lay one chord per `bars_per_chord` bars, sustained.

    `chord_seqs` is shape (M, C, V): M sampled progressions of C chords,
    each with V voices. We unroll them around the song length.
    """
    track = Track(instrument_name=instrument, gain=0.55)
    chords = chord_seqs.reshape(-1, chord_seqs.shape[-1])
    n_chords_needed = song.n_bars // bars_per_chord
    for ci in range(n_chords_needed):
        chord = chords[ci % len(chords)]
        start = ci * bars_per_chord * song.beats_per_bar
        dur = bars_per_chord * song.beats_per_bar - 0.05
        for f in chord:
            track.notes.append(Note(float(f), start, dur, velocity=0.85))
    return song.add(track)


def arrange_bass_track(song: Song, chord_seqs: np.ndarray,
                       instrument: str = "bass_pluck",
                       beats_per_bass: int = 2) -> Song:
    """
    Walking-ish bass: play the lowest note of the current chord on
    every `beats_per_bass`-th beat.
    """
    track = Track(instrument_name=instrument, gain=0.9)
    chords = chord_seqs.reshape(-1, chord_seqs.shape[-1])
    bars_per_chord = max(1, song.n_bars // max(1, len(chords)))
    n_hits = int(song.total_beats / beats_per_bass)
    for k in range(n_hits):
        beat = k * beats_per_bass
        bar = int(beat // song.beats_per_bar)
        chord_idx = (bar // bars_per_chord) % len(chords)
        bass_freq = float(np.min(chords[chord_idx])) / 2.0   # one octave down
        track.notes.append(Note(bass_freq, beat,
                                 beats_per_bass - 0.1,
                                 velocity=0.95))
    return song.add(track)


def arrange_melody_track(song: Song, melodies: np.ndarray,
                         scale_pcs: Sequence[int],
                         instrument: str = "lead",
                         notes_per_bar: int = 4,
                         reference_freq: float = 220.0) -> Song:
    """
    Place the trained-generator melody notes on a uniform sub-beat grid
    inside each bar. Notes are *quantized to the scale* — without this
    constraint a tabula-rasa melody plays out-of-tune relative to the
    fixed-scale chord track.
    """
    track = Track(instrument_name=instrument, gain=0.6)
    flat = melodies.flatten()
    flat = quantize_to_scale(flat, scale_pcs, reference_freq=reference_freq)
    n_total = song.n_bars * notes_per_bar
    sub_beat = song.beats_per_bar / notes_per_bar
    for k in range(n_total):
        f = float(flat[k % len(flat)])
        beat = k * sub_beat
        track.notes.append(Note(f, beat, sub_beat * 0.9, velocity=0.85))
    return song.add(track)
