"""
REINFORCE training loop for Phase 4.5 — joint melodic rhythm.

A single generator emits per-note (pitch, IOI). The reward sums the
Phase-3 melody reward (on the pitches) and the Phase-4 rhythm reward
(on the cumulative-sum onsets). No explicit cross-modal coupling.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.generator.melodic_rhythm_generator import MelodicRhythmGenerator
from src.reward.psychoacoustic import (
    implied_fundamental_salience,
    melody_reward,
    pitch_class_diversity,
    sequential_consonance,
)
from src.reward.rhythm import (
    autocorr_peak,
    best_period_phase,
    inter_onset_intervals,
    phase_coherence,
    rhythm_reward,
)


def _score(freqs: torch.Tensor, onsets: torch.Tensor,
           duration: float, melody_w: float, rhythm_w: float) -> np.ndarray:
    f = freqs.detach().cpu().numpy()
    o = onsets.detach().cpu().numpy()
    out = np.empty(f.shape[0], dtype=np.float32)
    for i in range(f.shape[0]):
        mr = melody_reward(f[i])
        rr = rhythm_reward(o[i], duration=duration)
        out[i] = melody_w * mr + rhythm_w * rr
    return out


def _entropy_coef(step: int, n_steps: int,
                  start: float = 0.01, end: float = -0.02) -> float:
    frac = min(1.0, max(0.0, step / max(1, n_steps - 1)))
    return start + (end - start) * frac


def train_melodic_rhythm(n_steps=2500, batch_size=64, lr=3e-4, log_every=50,
                         seed=0, n_notes=8, duration=4.0,
                         melody_weight=1.0, rhythm_weight=1.0,
                         ent_start=0.01, ent_end=-0.02,
                         out_dir="results/phase4_5_melodic_rhythm"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    gen = MelodicRhythmGenerator(latent_dim=24, hidden=192,
                                 n_notes=n_notes, duration=duration)
    opt = torch.optim.Adam(gen.parameters(), lr=lr)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    history = []
    best_eval_reward = -float("inf")
    best_state = None

    for step in range(n_steps):
        freqs, onsets, log_prob = gen.sample(batch_size)
        z = torch.randn(batch_size, gen.latent_dim)
        entropy = gen.entropy_at(z)

        rewards = torch.tensor(_score(freqs, onsets, duration,
                                       melody_weight, rhythm_weight))
        adv = rewards - rewards.mean()

        ent_coef = _entropy_coef(step, n_steps, ent_start, ent_end)
        loss = -(log_prob * adv).mean() - ent_coef * entropy

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
        opt.step()

        if step % log_every == 0 or step == n_steps - 1:
            with torch.no_grad():
                f, o, _ = gen.sample(128)
                fn = f.cpu().numpy()
                on = o.cpu().numpy()
                er = _score(f, o, duration, melody_weight, rhythm_weight)
                tonal = np.mean([implied_fundamental_salience(m) for m in fn])
                diss = np.mean([sequential_consonance(m) for m in fn])
                npc = np.mean([pitch_class_diversity(m) for m in fn])
                pc = np.mean([phase_coherence(x) for x in on])
                period = np.mean([best_period_phase(x) for x in on])
            entry = {
                "step": step,
                "mean_reward": float(er.mean()),
                "mean_tonal": float(tonal),
                "mean_seq_diss": float(diss),
                "mean_pitch_classes": float(npc),
                "mean_phase_coherence": float(pc),
                "mean_period": float(period),
            }
            history.append(entry)
            print(f"[{step:5d}] r={entry['mean_reward']:+.3f}  "
                  f"ton={entry['mean_tonal']:.3f}  "
                  f"#pc={entry['mean_pitch_classes']:.1f}  "
                  f"phase_coh={entry['mean_phase_coherence']:.3f}  "
                  f"period={entry['mean_period']:.3f}s")
            if entry["mean_reward"] > best_eval_reward:
                best_eval_reward = entry["mean_reward"]
                best_state = {k: v.clone() for k, v in gen.state_dict().items()}

    if best_state is not None:
        print(f"Best eval reward: {best_eval_reward:+.3f}; "
              "saving best checkpoint")
        gen.load_state_dict(best_state)
    torch.save(gen.state_dict(), out_path / "melodic_rhythm_generator.pt")
    with open(out_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    with torch.no_grad():
        sf, so, _ = gen.sample(256)
    np.save(out_path / "final_freqs.npy", sf.cpu().numpy())
    np.save(out_path / "final_onsets.npy", so.cpu().numpy())
    print(f"\nSaved Phase-4.5 model + history + samples to {out_path}/")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-notes", type=int, default=8)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--melody-weight", type=float, default=1.0)
    parser.add_argument("--rhythm-weight", type=float, default=1.0)
    parser.add_argument("--out-dir", type=str,
                        default="results/phase4_5_melodic_rhythm")
    args = parser.parse_args()
    train_melodic_rhythm(
        n_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        n_notes=args.n_notes,
        duration=args.duration,
        melody_weight=args.melody_weight,
        rhythm_weight=args.rhythm_weight,
        out_dir=args.out_dir,
    )
