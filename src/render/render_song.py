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
import argparse
from pathlib import Path

import numpy as np
import torch

from src.generator.chord_generator import ChordProgressionGenerator
from src.generator.melody_generator import MelodyGenerator, ExpressiveMelodyGenerator
from src.render.song import (
    Song,
    arrange_bass_track,
    arrange_chord_track,
    arrange_melody_track,
    basic_drum_pattern,
    write_wav,
)


SCALE_PCS_MAJOR_PENTATONIC = [0, 2, 4, 7, 9]   # C major pentatonic


def _load_melody_freqs(weights_path: str, n_samples: int = 8,
                       seed: int = 0):
    """Load melody generator and return frequency arrays.

    Auto-detects whether the checkpoint is MelodyGenerator or
    ExpressiveMelodyGenerator based on key names.
    """
    torch.manual_seed(seed)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    has_dur_head = any("dur_head" in k for k in state)

    if has_dur_head:
        gen = ExpressiveMelodyGenerator(latent_dim=32, hidden=256, n_notes=16)
        gen.load_state_dict(state)
        gen.eval()
        with torch.no_grad():
            combined, _ = gen.sample(n_samples)
        n = gen.n_notes
        freqs = combined[:, :n].cpu().numpy()
    else:
        gen = MelodyGenerator(latent_dim=16, hidden=128, n_notes=8)
        gen.load_state_dict(state)
        gen.eval()
        with torch.no_grad():
            freqs, _ = gen.sample(n_samples)
        freqs = freqs.cpu().numpy()
    return freqs


def compose_song(out_file: str = "results/audio/song_first.wav",
                 tempo_bpm: float = 110.0,
                 n_bars: int = 16,
                 seed: int = 0,
                 melody_weights: str = "results/phase3_melodies/melody_generator.pt",
                 chord_weights: str = "results/phase2_progressions/progression_generator.pt"):
    torch.manual_seed(seed)

    # 1. Sample a few chord progressions
    chord_gen = ChordProgressionGenerator(latent_dim=16, hidden=128,
                                           n_chords=4, n_voices=3)
    chord_gen.load_state_dict(
        torch.load(chord_weights, map_location="cpu", weights_only=True)
    )
    chord_gen.eval()
    with torch.no_grad():
        prog, _ = chord_gen.sample(2)
    chord_seqs = prog.cpu().numpy()    # (2, 4, 3)

    # 2. Sample melodies (auto-detects generator architecture)
    melodies = _load_melody_freqs(melody_weights, n_samples=8, seed=seed)

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
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/audio/song_first.wav")
    p.add_argument("--melody-weights",
                   default="results/phase3_melodies/melody_generator.pt")
    p.add_argument("--chord-weights",
                   default="results/phase2_progressions/progression_generator.pt")
    p.add_argument("--tempo", type=float, default=110.0)
    p.add_argument("--bars", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    compose_song(out_file=args.out, tempo_bpm=args.tempo,
                 n_bars=args.bars, seed=args.seed,
                 melody_weights=args.melody_weights,
                 chord_weights=args.chord_weights)
