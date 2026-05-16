"""
Compose a first "song" out of trained-generator outputs.

We layer:
  - drums  (4/4 backbeat)
  - chords (Phase-2 progressions, sustained one chord per bar)
  - bass   (lowest chord tone, plucked, two beats per bar)
  - melody (Phase-3 melodies, quantized to a major-pentatonic so it
            doesn't fight the chord track tuning)

This is *arrangement* on top of existing trained models — no new
training. Output: results/audio/song_first.wav.
"""
from pathlib import Path

import numpy as np
import torch

from src.generator.chord_generator import ChordProgressionGenerator
from src.generator.melody_generator import MelodyGenerator
from src.render.song import (
    Song,
    arrange_bass_track,
    arrange_chord_track,
    arrange_melody_track,
    basic_drum_pattern,
    write_wav,
)


SCALE_PCS_MAJOR_PENTATONIC = [0, 2, 4, 7, 9]   # C major pentatonic


def compose_song(out_file: str = "results/audio/song_first.wav",
                 tempo_bpm: float = 110.0,
                 n_bars: int = 16,
                 seed: int = 0):
    torch.manual_seed(seed)

    # 1. Sample a few chord progressions (Phase 2)
    chord_gen = ChordProgressionGenerator(latent_dim=16, hidden=128,
                                           n_chords=4, n_voices=3)
    chord_gen.load_state_dict(
        torch.load("results/phase2_progressions/progression_generator.pt",
                   map_location="cpu")
    )
    chord_gen.eval()
    with torch.no_grad():
        prog, _ = chord_gen.sample(2)
    chord_seqs = prog.cpu().numpy()    # (2, 4, 3)

    # 2. Sample melodies (Phase 3)
    melody_gen = MelodyGenerator(latent_dim=16, hidden=128, n_notes=8)
    melody_gen.load_state_dict(
        torch.load("results/phase3_melodies/melody_generator.pt",
                   map_location="cpu")
    )
    melody_gen.eval()
    with torch.no_grad():
        mel, _ = melody_gen.sample(8)
    melodies = mel.cpu().numpy()       # (8, 8)

    # 3. Compose
    song = Song(tempo_bpm=tempo_bpm, n_bars=n_bars, beats_per_bar=4)
    basic_drum_pattern(song, kick_pattern=(0, 2),
                       snare_pattern=(1, 3), hh_subdiv=2)
    arrange_chord_track(song, chord_seqs, instrument="pad",
                         bars_per_chord=1)
    arrange_bass_track(song, chord_seqs, instrument="bass_pluck",
                        beats_per_bass=2)
    arrange_melody_track(song, melodies,
                          scale_pcs=SCALE_PCS_MAJOR_PENTATONIC,
                          instrument="lead",
                          notes_per_bar=4,
                          reference_freq=220.0)

    audio = song.render()
    write_wav(out_file, audio)
    print(f"saved {out_file}  ({song.total_seconds:.1f} s, "
          f"{tempo_bpm:.0f} BPM, {n_bars} bars)")
    return audio


if __name__ == "__main__":
    compose_song()
