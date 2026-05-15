"""
Render WAVs from every phase's trained generator so a human can listen
to the tabula-rasa output. This module is one-way: it reads the saved
artifacts under `results/phase*/` and writes audio under
`results/audio/`. Nothing here flows back into training.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from src.generator.toy_generator import ToyIntervalGenerator
from src.generator.chord_generator import (
    ChordProgressionGenerator,
    TriadGenerator,
)
from src.generator.melody_generator import MelodyGenerator
from src.generator.rhythm_generator import RhythmGenerator
from src.render.synth import (
    SAMPLE_RATE,
    render_chord,
    render_melodic_rhythm,
    render_melody,
    render_rhythm,
    write_wav,
)


def _safe_load(model, weights_path):
    try:
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        return True
    except FileNotFoundError:
        print(f"  (skipped — {weights_path} not found)")
        return False


def render_phase1(out_dir: Path, seed: int = 0):
    print("Phase 1: intervals")
    gen = ToyIntervalGenerator()
    if not _safe_load(gen, "results/toy_generator.pt"):
        return
    gen.eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        f, _ = gen.sample(8)
    chunks = []
    for pair in f.cpu().numpy():
        chunks.append(render_chord(pair, duration=1.0))
        chunks.append(np.zeros(int(0.2 * SAMPLE_RATE)))
    audio = np.concatenate(chunks)
    write_wav(out_dir / "phase1_intervals.wav", audio)
    print(f"  saved {out_dir/'phase1_intervals.wav'}")


def render_phase2_triads(out_dir: Path, seed: int = 0):
    print("Phase 2: triads")
    gen = TriadGenerator()
    if not _safe_load(gen, "results/phase2_triads/triad_generator.pt"):
        return
    gen.eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        f, _ = gen.sample(8)
    chunks = []
    for tri in f.cpu().numpy():
        chunks.append(render_chord(tri, duration=1.4))
        chunks.append(np.zeros(int(0.3 * SAMPLE_RATE)))
    audio = np.concatenate(chunks)
    write_wav(out_dir / "phase2_triads.wav", audio)
    print(f"  saved {out_dir/'phase2_triads.wav'}")


def render_phase2_progressions(out_dir: Path, seed: int = 0):
    print("Phase 2: progressions")
    gen = ChordProgressionGenerator(latent_dim=16, hidden=128,
                                    n_chords=4, n_voices=3)
    if not _safe_load(gen, "results/phase2_progressions/progression_generator.pt"):
        return
    gen.eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        seqs, _ = gen.sample(4)
    chunks = []
    for prog in seqs.cpu().numpy():
        for c in prog:
            chunks.append(render_chord(c, duration=0.8))
            chunks.append(np.zeros(int(0.05 * SAMPLE_RATE)))
        chunks.append(np.zeros(int(0.6 * SAMPLE_RATE)))
    audio = np.concatenate(chunks)
    write_wav(out_dir / "phase2_progressions.wav", audio)
    print(f"  saved {out_dir/'phase2_progressions.wav'}")


def render_phase3_melodies(out_dir: Path, seed: int = 0):
    print("Phase 3: melodies")
    gen = MelodyGenerator()
    if not _safe_load(gen, "results/phase3_melodies/melody_generator.pt"):
        return
    gen.eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        f, _ = gen.sample(6)
    chunks = []
    for mel in f.cpu().numpy():
        chunks.append(render_melody(mel, note_duration=0.35, gap=0.03))
        chunks.append(np.zeros(int(0.4 * SAMPLE_RATE)))
    audio = np.concatenate(chunks)
    write_wav(out_dir / "phase3_melodies.wav", audio)
    print(f"  saved {out_dir/'phase3_melodies.wav'}")


def render_phase4_rhythms(out_dir: Path, seed: int = 0):
    print("Phase 4: rhythms")
    gen = RhythmGenerator()
    if not _safe_load(gen, "results/phase4_rhythms/rhythm_generator.pt"):
        return
    gen.eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        o, _ = gen.sample(4)
    chunks = []
    for ons in o.cpu().numpy():
        chunks.append(render_rhythm(ons, duration=4.0))
        chunks.append(np.zeros(int(0.4 * SAMPLE_RATE)))
    audio = np.concatenate(chunks)
    write_wav(out_dir / "phase4_rhythms.wav", audio)
    print(f"  saved {out_dir/'phase4_rhythms.wav'}")


def render_phase34_combined(out_dir: Path, seed: int = 0):
    """Use the Phase-3 melody as a sequence of pitches and the Phase-4
    onset pattern as the timing — pure synthesis, no joint training."""
    print("Phase 3+4: melodic rhythm")
    mgen = MelodyGenerator()
    rgen = RhythmGenerator()
    if not _safe_load(mgen, "results/phase3_melodies/melody_generator.pt"):
        return
    if not _safe_load(rgen, "results/phase4_rhythms/rhythm_generator.pt"):
        return
    mgen.eval()
    rgen.eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        m, _ = mgen.sample(4)
        o, _ = rgen.sample(4)
    chunks = []
    for mel, ons in zip(m.cpu().numpy(), o.cpu().numpy()):
        chunks.append(render_melodic_rhythm(mel, ons, duration=4.0))
        chunks.append(np.zeros(int(0.5 * SAMPLE_RATE)))
    audio = np.concatenate(chunks)
    write_wav(out_dir / "phase34_melodic_rhythm.wav", audio)
    print(f"  saved {out_dir/'phase34_melodic_rhythm.wav'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="results/audio")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    render_phase1(out_dir, args.seed)
    render_phase2_triads(out_dir, args.seed)
    render_phase2_progressions(out_dir, args.seed)
    render_phase3_melodies(out_dir, args.seed)
    render_phase4_rhythms(out_dir, args.seed)
    render_phase34_combined(out_dir, args.seed)
