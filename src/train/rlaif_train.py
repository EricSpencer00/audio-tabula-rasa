"""
RLAIF training loop: REINFORCE with a Qwen2.5-Omni audio judge.

The generator (currently only the Phase-3 monophonic melody generator)
samples a batch of N notes. We render each sample as audio at 44.1 kHz,
hand it to the Qwen-Omni judge for a 0-10 musicality score, and use
that score as the REINFORCE reward.

The generator is *initialized* from the existing Phase-3 baseline
(`results/phase3_melodies/melody_generator.pt`) so the 80-step budget
is spent fine-tuning, not bootstrapping from random. Score 0-10 is
treated as the raw reward; we subtract a running batch mean as the
baseline (no std normalization — same convention as the Phase-3
psychoacoustic training).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from src.generator.melody_generator import MelodyGenerator
from src.render.synth import SAMPLE_RATE, render_melody


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    import librosa
    return librosa.resample(audio.astype(np.float32),
                            orig_sr=src_sr, target_sr=dst_sr)


def _render_melody_for_judge(freqs: np.ndarray, judge_sr: int,
                             note_duration: float = 0.35,
                             gap: float = 0.03) -> np.ndarray:
    audio = render_melody(freqs, note_duration=note_duration, gap=gap)
    return _resample(audio, SAMPLE_RATE, judge_sr)


def train_melody_rlaif(
    out_dir: str,
    qwen_model: str = "Qwen/Qwen2.5-Omni-3B",
    qwen_device: str = "cpu",
    qwen_dtype: str = "bfloat16",
    init_weights: str = "results/phase3_melodies/melody_generator.pt",
    n_steps: int = 80,
    batch_size: int = 2,
    lr: float = 1e-4,
    seed: int = 0,
    n_notes: int = 8,
    note_duration: float = 0.35,
    gap: float = 0.03,
    log_every: int = 1,
    checkpoint_every: int = 20,
    score_fallback: float = 0.0,
):
    from src.analysis.qwen_judge import QwenJudge

    torch.manual_seed(seed)
    np.random.seed(seed)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    gen = MelodyGenerator(latent_dim=16, hidden=128, n_notes=n_notes)
    if init_weights and Path(init_weights).exists():
        gen.load_state_dict(torch.load(init_weights, map_location="cpu"))
        print(f"Loaded init weights from {init_weights}", flush=True)
    else:
        print(f"WARNING: init weights not found at {init_weights}; "
              "training from random init.", flush=True)
    opt = torch.optim.Adam(gen.parameters(), lr=lr)

    print(f"Loading Qwen judge from {qwen_model} on {qwen_device}...",
          flush=True)
    t_load = time.time()
    judge = QwenJudge(model_id=qwen_model, device=qwen_device,
                      dtype=qwen_dtype, max_new_tokens=48)
    print(f"Judge loaded in {time.time() - t_load:.1f}s "
          f"(audio_sr={judge.audio_sr})", flush=True)

    history: List[dict] = []
    best_eval_reward = -float("inf")
    best_state = None
    running_baseline = None
    baseline_decay = 0.9

    for step in range(n_steps):
        t_step = time.time()
        z = torch.randn(batch_size, gen.latent_dim)
        log_mean, std = gen(z)
        dist = torch.distributions.Normal(log_mean, std)
        log_freqs_raw = dist.rsample()
        log_freqs = log_freqs_raw.clamp(min=gen._log_lo, max=gen._log_hi)
        freqs = torch.exp(log_freqs)
        log_prob = dist.log_prob(log_freqs).sum(dim=-1)

        freqs_np = freqs.detach().cpu().numpy()
        rewards: List[float] = []
        raws: List[str] = []
        for i in range(batch_size):
            wav = _render_melody_for_judge(
                freqs_np[i], judge_sr=judge.audio_sr,
                note_duration=note_duration, gap=gap,
            )
            res = judge.score_audio("<rendered>", waveform=wav)
            score = res.score if res.score is not None else score_fallback
            rewards.append(score)
            raws.append(res.raw_text)

        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        mean_r = float(rewards_t.mean())
        if running_baseline is None:
            running_baseline = mean_r
        else:
            running_baseline = (baseline_decay * running_baseline
                                + (1 - baseline_decay) * mean_r)
        adv = rewards_t - running_baseline

        loss = -(log_prob * adv).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
        opt.step()

        dt = time.time() - t_step
        elapsed_steps = step + 1
        entry = {
            "step": step,
            "mean_reward": mean_r,
            "baseline": running_baseline,
            "loss": float(loss.detach()),
            "rewards": rewards,
            "step_seconds": dt,
        }
        history.append(entry)
        if step % log_every == 0 or step == n_steps - 1:
            print(
                f"[elapsed_steps={elapsed_steps}/{n_steps}] "
                f"r={mean_r:+.2f}  baseline={running_baseline:+.2f}  "
                f"loss={float(loss.detach()):+.3f}  "
                f"dt={dt:.1f}s  rewards={rewards}",
                flush=True,
            )

        if mean_r > best_eval_reward:
            best_eval_reward = mean_r
            best_state = {k: v.clone() for k, v in gen.state_dict().items()}

        if checkpoint_every and (step + 1) % checkpoint_every == 0:
            torch.save(gen.state_dict(),
                       out_path / f"melody_generator_step{step+1}.pt")
            with open(out_path / "history.json", "w") as f:
                json.dump(history, f, indent=2)

    if best_state is not None:
        print(f"Best mean_reward across training: {best_eval_reward:+.2f}; "
              "saving best checkpoint as melody_generator.pt", flush=True)
        gen.load_state_dict(best_state)
    torch.save(gen.state_dict(), out_path / "melody_generator.pt")
    torch.save(gen.state_dict(), out_path / "melody_generator_final.pt")
    with open(out_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    with torch.no_grad():
        sample_freqs, _ = gen.sample(64)
    np.save(out_path / "final_melodies.npy",
            sample_freqs.cpu().numpy())
    print(f"Saved to {out_path}/ — DONE", flush=True)
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", type=str, default="melody",
                        choices=["melody"])
    parser.add_argument("--judge", type=str, default="qwen",
                        choices=["qwen"])
    parser.add_argument("--qwen-model", type=str,
                        default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument("--qwen-device", type=str, default="cpu")
    parser.add_argument("--qwen-dtype", type=str, default="bfloat16")
    parser.add_argument("--init-weights", type=str,
                        default="results/phase3_melodies/melody_generator.pt")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-notes", type=int, default=8)
    parser.add_argument("--note-duration", type=float, default=0.35)
    parser.add_argument("--gap", type=float, default=0.03)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--out-dir", type=str,
                        default="results/rlaif/melody_qwen3b")
    args = parser.parse_args()

    if args.generator != "melody":
        raise NotImplementedError(
            f"generator={args.generator!r} not implemented yet")
    if args.judge != "qwen":
        raise NotImplementedError(
            f"judge={args.judge!r} not implemented yet")

    train_melody_rlaif(
        out_dir=args.out_dir,
        qwen_model=args.qwen_model,
        qwen_device=args.qwen_device,
        qwen_dtype=args.qwen_dtype,
        init_weights=args.init_weights,
        n_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        n_notes=args.n_notes,
        note_duration=args.note_duration,
        gap=args.gap,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
