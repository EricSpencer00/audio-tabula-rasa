"""
Render audio samples from a theory-trained checkpoint and save as WAV.

Usage:
    python scripts/render_theory_samples.py \
        --checkpoint results/rlaif/theory_v1_100/rlaif_generator_best.pt \
        --generator melody_v7 \
        --n-samples 5 \
        --out-dir results/audio/theory_samples

Also prints the theory judge breakdown for each sample.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.render.synth import SAMPLE_RATE
from src.reward.theory_judge import theory_reward_breakdown


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--generator", required=True,
                   choices=["melody_v7", "melody_v3", "melody_v4",
                            "melody_v5", "melody_v6", "autoregressive",
                            "counterpoint", "counterpoint_3v",
                            "counterpoint_4v", "melody_v8_vocal"])
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--out-dir", default="results/audio/theory_samples")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # Lazy import to avoid loading rlaif_train's full dependency tree
    from src.train.rlaif_train import _ADAPTERS

    adapter = _ADAPTERS[args.generator]
    gen = adapter.build()

    state = torch.load(args.checkpoint, map_location="cpu")
    gen.load_state_dict(state)
    gen.eval()
    print(f"Loaded {args.checkpoint}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = []
    for i in range(args.n_samples):
        with torch.no_grad():
            freqs, _ = gen.sample(1)
        freqs_np = freqs[0].cpu().numpy()

        # Render audio
        audio = adapter.sample_to_audio(freqs_np).astype(np.float32)

        # Score with theory judge
        if adapter.output_format == "expressive":
            n_notes = len(freqs_np) // 3
            f, d, v = freqs_np[:n_notes], freqs_np[n_notes:2*n_notes], freqs_np[2*n_notes:]
            breakdown = theory_reward_breakdown(f, durations=d, velocities=v)
        elif adapter.output_format == "voices":
            breakdown = theory_reward_breakdown(freqs_np.flatten(), voices=freqs_np)
        else:
            breakdown = theory_reward_breakdown(freqs_np)

        # Save WAV
        import scipy.io.wavfile
        wav_path = out_path / f"sample_{i}.wav"
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        scipy.io.wavfile.write(str(wav_path), SAMPLE_RATE, audio_int16)

        total = sum(breakdown.values())
        results.append({"file": str(wav_path), "total": total, **breakdown})
        scores_str = " | ".join(f"{k}={v:.2f}" for k, v in breakdown.items())
        print(f"  [{i}] total={total:.2f} | {scores_str} → {wav_path}")

    # Save summary
    summary_path = out_path / "theory_scores.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {args.n_samples} samples to {out_path}")
    print(f"Theory scores summary: {summary_path}")


if __name__ == "__main__":
    main()
