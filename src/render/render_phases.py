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
from src.generator.counterpoint_generator import CounterpointGenerator
from src.generator.melodic_rhythm_generator import MelodicRhythmGenerator
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


def render_phase45_melodic_rhythm(out_dir: Path, seed: int = 0):
    """Joint-trained generator: pitch and onset come from one network."""
    print("Phase 4.5: joint melodic rhythm")
    gen = MelodicRhythmGenerator()
    if not _safe_load(
        gen, "results/phase4_5_melodic_rhythm/melodic_rhythm_generator.pt"
    ):
        return
    gen.eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        f, o, _ = gen.sample(4)
    chunks = []
    for mel, ons in zip(f.cpu().numpy(), o.cpu().numpy()):
        chunks.append(render_melodic_rhythm(mel, ons, duration=4.0))
        chunks.append(np.zeros(int(0.5 * SAMPLE_RATE)))
    audio = np.concatenate(chunks)
    write_wav(out_dir / "phase45_melodic_rhythm.wav", audio)
    print(f"  saved {out_dir/'phase45_melodic_rhythm.wav'}")


def _render_n_voice_counterpoint(weights_path: str, out_file: Path,
                                  n_voices: int, seed: int):
    gen = CounterpointGenerator(n_voices=n_voices)
    if not _safe_load(gen, weights_path):
        return
    gen.eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        v, _ = gen.sample(4)
    chunks = []
    for cp in v.cpu().numpy():
        voice_audios = []
        for voice in cp:
            voice_audios.append(render_melody(voice, note_duration=0.45,
                                              gap=0.0))
        max_len = max(len(a) for a in voice_audios)
        mixed = np.zeros(max_len)
        for a in voice_audios:
            mixed[: len(a)] += a / len(voice_audios)
        chunks.append(mixed)
        chunks.append(np.zeros(int(0.4 * SAMPLE_RATE)))
    audio = np.concatenate(chunks)
    write_wav(out_file, audio)
    print(f"  saved {out_file}")


def render_phase13_3voice(out_dir: Path, seed: int = 0):
    print("Phase 13: 3-voice counterpoint")
    _render_n_voice_counterpoint(
        "results/phase13_3voice_counterpoint/counterpoint_generator.pt",
        out_dir / "phase13_3voice_counterpoint.wav",
        n_voices=3, seed=seed,
    )


def render_phase13_4voice(out_dir: Path, seed: int = 0):
    print("Phase 13: 4-voice counterpoint")
    _render_n_voice_counterpoint(
        "results/phase13_4voice_counterpoint/counterpoint_generator.pt",
        out_dir / "phase13_4voice_counterpoint.wav",
        n_voices=4, seed=seed,
    )


def render_phase8b_bp_triads(out_dir: Path, seed: int = 0):
    """BP triads rendered with odd-only partials so the listener actually
    hears the Bohlen-Pierce-like consonance the model was trained on."""
    print("Phase 8b: BP triads (odd partials)")
    gen = TriadGenerator()
    if not _safe_load(gen,
                       "results/phase8b_bp_triads/triad_generator.pt"):
        return
    gen.eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        f, _ = gen.sample(8)

    def render_odd_chord(freqs, duration=1.4, n_partials=6):
        """Mix sines at odd-multiple partials of each chord pitch."""
        n = int(duration * SAMPLE_RATE)
        t = np.arange(n) / SAMPLE_RATE
        out = np.zeros(n)
        for f0 in freqs:
            for k in range(1, 2 * n_partials, 2):
                out += (1.0 / k) * np.sin(2 * np.pi * k * float(f0) * t)
        # Envelope
        from src.render.synth import _envelope
        return out * _envelope(n)

    chunks = []
    for tri in f.cpu().numpy():
        chunks.append(render_odd_chord(tri))
        chunks.append(np.zeros(int(0.3 * SAMPLE_RATE)))
    audio = np.concatenate(chunks)
    write_wav(out_dir / "phase8b_bp_triads_odd_timbre.wav", audio)
    print(f"  saved {out_dir/'phase8b_bp_triads_odd_timbre.wav'}")


def render_phase7_counterpoint(out_dir: Path, seed: int = 0):
    """V-voice counterpoint: render all voices simultaneously."""
    print("Phase 7: counterpoint")
    gen = CounterpointGenerator()
    if not _safe_load(
        gen, "results/phase7_counterpoint/counterpoint_generator.pt"
    ):
        return
    gen.eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        v, _ = gen.sample(4)
    chunks = []
    for cp in v.cpu().numpy():     # shape (V, N)
        # Render each voice as a sequence and mix
        voice_audios = []
        for voice in cp:
            voice_audios.append(render_melody(voice, note_duration=0.45,
                                              gap=0.0))
        max_len = max(len(a) for a in voice_audios)
        mixed = np.zeros(max_len)
        for a in voice_audios:
            mixed[: len(a)] += a / len(voice_audios)
        chunks.append(mixed)
        chunks.append(np.zeros(int(0.4 * SAMPLE_RATE)))
    audio = np.concatenate(chunks)
    write_wav(out_dir / "phase7_counterpoint.wav", audio)
    print(f"  saved {out_dir/'phase7_counterpoint.wav'}")


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
    render_phase45_melodic_rhythm(out_dir, args.seed)
    render_phase7_counterpoint(out_dir, args.seed)
    render_phase8b_bp_triads(out_dir, args.seed)
    render_phase13_3voice(out_dir, args.seed)
    render_phase13_4voice(out_dir, args.seed)
