"""
REINFORCE training loop for Phase 7 — counterpoint.

A V-voice generator emits V × N log-frequencies. The reward sums each
voice's Phase-3 melody reward (horizontal) and a vertical pairwise
Sethares chord reward at every time step, plus voice-crossing and
voice-register-gap constraints.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.generator.counterpoint_generator import CounterpointGenerator
from src.reward.counterpoint import (
    counterpoint_reward,
    shared_tonal_salience,
    vertical_dissonance,
    voice_crossings,
)


def _score(voices: torch.Tensor, **kw) -> np.ndarray:
    np_v = voices.detach().cpu().numpy()
    out = np.empty(np_v.shape[0], dtype=np.float32)
    for i in range(np_v.shape[0]):
        out[i] = counterpoint_reward(np_v[i], **kw)
    return out


def _entropy_coef(step: int, n_steps: int,
                  start: float = 0.01, end: float = -0.02) -> float:
    frac = min(1.0, max(0.0, step / max(1, n_steps - 1)))
    return start + (end - start) * frac


def train_counterpoint(n_steps=2000, batch_size=64, lr=3e-4, log_every=50,
                       seed=0, n_voices=2, n_notes=8,
                       per_voice_weight=1.0, vertical_weight=2.0,
                       crossing_weight=2.0, register_weight=3.0,
                       shared_root_weight=2.0,
                       ent_start=0.01, ent_end=-0.02,
                       out_dir="results/phase7_counterpoint"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    gen = CounterpointGenerator(latent_dim=24, hidden=192,
                                n_voices=n_voices, n_notes=n_notes)
    opt = torch.optim.Adam(gen.parameters(), lr=lr)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    history = []
    best_eval_reward = -float("inf")
    best_state = None
    kw = dict(per_voice_weight=per_voice_weight,
              vertical_weight=vertical_weight,
              crossing_weight=crossing_weight,
              register_weight=register_weight,
              shared_root_weight=shared_root_weight)

    for step in range(n_steps):
        voices, log_prob = gen.sample(batch_size)
        z = torch.randn(batch_size, gen.latent_dim)
        entropy = gen.entropy_at(z)

        rewards = torch.tensor(_score(voices, **kw))
        adv = rewards - rewards.mean()

        ent_coef = _entropy_coef(step, n_steps, ent_start, ent_end)
        loss = -(log_prob * adv).mean() - ent_coef * entropy

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
        opt.step()

        if step % log_every == 0 or step == n_steps - 1:
            with torch.no_grad():
                v, _ = gen.sample(128)
                vn = v.cpu().numpy()
                er = _score(v, **kw)
                vd = np.mean([vertical_dissonance(x) for x in vn])
                vc = np.mean([voice_crossings(x) for x in vn])
                st = np.mean([shared_tonal_salience(x) for x in vn])
            entry = {
                "step": step,
                "mean_reward": float(er.mean()),
                "mean_vert_diss": float(vd),
                "mean_voice_crossings": float(vc),
                "mean_shared_tonal": float(st),
            }
            history.append(entry)
            print(f"[{step:5d}] r={entry['mean_reward']:+.3f}  "
                  f"vert_diss={entry['mean_vert_diss']:.3f}  "
                  f"crosses={entry['mean_voice_crossings']:.2f}  "
                  f"shared_ton={entry['mean_shared_tonal']:.3f}")
            if entry["mean_reward"] > best_eval_reward:
                best_eval_reward = entry["mean_reward"]
                best_state = {k: v.clone() for k, v in gen.state_dict().items()}

    if best_state is not None:
        print(f"Best eval reward: {best_eval_reward:+.3f}; saving best checkpoint")
        gen.load_state_dict(best_state)
    torch.save(gen.state_dict(), out_path / "counterpoint_generator.pt")
    with open(out_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    with torch.no_grad():
        sv, _ = gen.sample(256)
    np.save(out_path / "final_voices.npy", sv.cpu().numpy())
    print(f"\nSaved counterpoint model + history + samples to {out_path}/")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-voices", type=int, default=2)
    parser.add_argument("--n-notes", type=int, default=8)
    parser.add_argument("--out-dir", type=str,
                        default="results/phase7_counterpoint")
    args = parser.parse_args()
    train_counterpoint(
        n_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        n_voices=args.n_voices,
        n_notes=args.n_notes,
        out_dir=args.out_dir,
    )
