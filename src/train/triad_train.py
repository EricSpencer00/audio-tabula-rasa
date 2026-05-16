"""
REINFORCE training loop for Phase 2 — triads and chord progressions.

Same structure as the toy case: Gaussian policy, baseline-subtracted
advantage. The reward swaps in the chord/progression reward from
psychoacoustic.py, with optional voice-leading penalty.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.generator.chord_generator import (
    ChordProgressionGenerator,
    TriadGenerator,
)
from src.reward.psychoacoustic import (
    chord_dissonance,
    chord_reward,
    progression_reward,
    triad_label,
    voice_leading_cost,
)


def _score_triads(freqs: torch.Tensor, spread_weight: float,
                  min_semitones: float,
                  partials: str = "harmonic") -> np.ndarray:
    out = np.empty(freqs.shape[0], dtype=np.float32)
    np_freqs = freqs.detach().cpu().numpy()
    for i, f in enumerate(np_freqs):
        out[i] = chord_reward(f, spread_weight=spread_weight,
                              min_semitones=min_semitones,
                              partials=partials)
    return out


def _score_progressions(freqs: torch.Tensor, spread_weight: float,
                        vl_weight: float, min_semitones: float) -> np.ndarray:
    out = np.empty(freqs.shape[0], dtype=np.float32)
    np_freqs = freqs.detach().cpu().numpy()
    for i, seq in enumerate(np_freqs):
        out[i] = progression_reward(seq, spread_weight=spread_weight,
                                    voice_leading_weight=vl_weight,
                                    min_semitones=min_semitones)
    return out


def _entropy_coef(step: int, n_steps: int,
                  start: float = 0.01, end: float = -0.02) -> float:
    """Linear schedule for the entropy coefficient.

    Starts positive (encourage exploration: higher policy entropy is
    rewarded) and ends negative (force the policy to commit: lower
    entropy is rewarded). This counteracts REINFORCE's well-known
    inability to collapse policy variance on its own.
    """
    frac = min(1.0, max(0.0, step / max(1, n_steps - 1)))
    return start + (end - start) * frac


def train_triads(n_steps=2000, batch_size=64, lr=1e-3, log_every=50,
                 seed=0, spread_weight=2.0, min_semitones=1.5,
                 ent_start=0.01, ent_end=-0.02,
                 partials="harmonic",
                 out_dir="results/phase2_triads"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    gen = TriadGenerator(latent_dim=8, hidden=64, n_voices=3)
    opt = torch.optim.Adam(gen.parameters(), lr=lr)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    history = []
    best_eval_reward = -float("inf")
    best_state = None

    for step in range(n_steps):
        # We need the distribution for entropy as well as log_prob,
        # so rebuild it here rather than calling sample() directly.
        z = torch.randn(batch_size, gen.latent_dim)
        mean, std = gen(z)
        dist = torch.distributions.Normal(mean, std)
        freqs_raw = dist.rsample()
        freqs = freqs_raw.clamp(min=gen.F_MIN, max=gen.F_MAX)
        log_prob = dist.log_prob(freqs).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        rewards = torch.tensor(_score_triads(freqs, spread_weight, min_semitones,
                                              partials=partials))

        adv = rewards - rewards.mean()
        adv = adv / (rewards.std() + 1e-8)

        ent_coef = _entropy_coef(step, n_steps, ent_start, ent_end)
        loss = -(log_prob * adv).mean() - ent_coef * entropy

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
        opt.step()

        if step % log_every == 0 or step == n_steps - 1:
            with torch.no_grad():
                ef, _ = gen.sample(256)
                er = _score_triads(ef, spread_weight, min_semitones,
                                    partials=partials)
                labels = [triad_label(f) for f in ef.cpu().numpy()]
                top = Counter(labels).most_common(3)
                diss = np.mean([
                    chord_dissonance(f, partials=partials)
                    for f in ef.cpu().numpy()
                ])
            entry = {
                "step": step,
                "mean_reward": float(er.mean()),
                "mean_dissonance": float(diss),
                "top_triads": top,
            }
            history.append(entry)
            print(f"[{step:5d}] reward={entry['mean_reward']:+.3f}  "
                  f"diss={entry['mean_dissonance']:.3f}  "
                  f"top={top}")
            if entry["mean_reward"] > best_eval_reward:
                best_eval_reward = entry["mean_reward"]
                best_state = {k: v.clone() for k, v in gen.state_dict().items()}

    if best_state is not None:
        print(f"Best eval reward: {best_eval_reward:+.3f}; "
              "saving best checkpoint")
        gen.load_state_dict(best_state)
    torch.save(gen.state_dict(), out_path / "triad_generator.pt")
    with open(out_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Dump a sample for downstream plotting / analysis.
    with torch.no_grad():
        sample_freqs, _ = gen.sample(512)
    np.save(out_path / "final_chords.npy", sample_freqs.cpu().numpy())
    print(f"\nSaved triad model + history + sample chords to {out_path}/")
    return history


def train_progressions(n_steps=2500, batch_size=64, lr=1e-3, log_every=50,
                       seed=0, n_chords=4, spread_weight=2.0,
                       vl_weight=0.5, min_semitones=1.5,
                       ent_start=0.01, ent_end=-0.02,
                       out_dir="results/phase2_progressions"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    gen = ChordProgressionGenerator(latent_dim=16, hidden=128,
                                    n_chords=n_chords, n_voices=3)
    opt = torch.optim.Adam(gen.parameters(), lr=lr)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    history = []
    best_eval_reward = -float("inf")
    best_state = None

    for step in range(n_steps):
        z = torch.randn(batch_size, gen.latent_dim)
        mean, std = gen(z)
        dist = torch.distributions.Normal(mean, std)
        freqs_raw = dist.rsample()
        freqs_flat = freqs_raw.clamp(min=gen.F_MIN, max=gen.F_MAX)
        log_prob = dist.log_prob(freqs_flat).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        freqs = freqs_flat.view(batch_size, gen.n_chords, gen.n_voices)

        rewards = torch.tensor(
            _score_progressions(freqs, spread_weight, vl_weight, min_semitones))

        adv = rewards - rewards.mean()
        adv = adv / (rewards.std() + 1e-8)

        ent_coef = _entropy_coef(step, n_steps, ent_start, ent_end)
        loss = -(log_prob * adv).mean() - ent_coef * entropy

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
        opt.step()

        if step % log_every == 0 or step == n_steps - 1:
            with torch.no_grad():
                ef, _ = gen.sample(128)
                er = _score_progressions(ef, spread_weight, vl_weight,
                                         min_semitones)
                np_seqs = ef.cpu().numpy()
                # Mean per-chord dissonance and mean voice-leading
                diss = np.mean([chord_dissonance(c) for s in np_seqs for c in s])
                vl = np.mean([
                    voice_leading_cost(s[i], s[i + 1])
                    for s in np_seqs for i in range(n_chords - 1)
                ])
                labels = [triad_label(c) for s in np_seqs for c in s]
                top = Counter(labels).most_common(3)
            entry = {
                "step": step,
                "mean_reward": float(er.mean()),
                "mean_dissonance": float(diss),
                "mean_voice_leading": float(vl),
                "top_triads": top,
            }
            history.append(entry)
            print(f"[{step:5d}] reward={entry['mean_reward']:+.3f}  "
                  f"diss={entry['mean_dissonance']:.3f}  "
                  f"vl={entry['mean_voice_leading']:.4f}  "
                  f"top={top}")
            if entry["mean_reward"] > best_eval_reward:
                best_eval_reward = entry["mean_reward"]
                best_state = {k: v.clone() for k, v in gen.state_dict().items()}

    if best_state is not None:
        print(f"Best eval reward: {best_eval_reward:+.3f}; "
              "saving best checkpoint")
        gen.load_state_dict(best_state)
    torch.save(gen.state_dict(), out_path / "progression_generator.pt")
    with open(out_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    with torch.no_grad():
        sample_seqs, _ = gen.sample(256)
    np.save(out_path / "final_progressions.npy", sample_seqs.cpu().numpy())
    print(f"\nSaved progression model + history + sample sequences to {out_path}/")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["triad", "progression"],
                        default="triad")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-chords", type=int, default=4)
    parser.add_argument("--spread-weight", type=float, default=2.0)
    parser.add_argument("--vl-weight", type=float, default=0.5)
    parser.add_argument("--min-semitones", type=float, default=1.5)
    parser.add_argument("--partials", type=str, default="harmonic",
                        choices=["harmonic", "odd", "inharmonic"])
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "triad":
        train_triads(
            n_steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            spread_weight=args.spread_weight,
            min_semitones=args.min_semitones,
            partials=args.partials,
            out_dir=args.out_dir or "results/phase2_triads",
        )
    else:
        train_progressions(
            n_steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            n_chords=args.n_chords,
            spread_weight=args.spread_weight,
            vl_weight=args.vl_weight,
            min_semitones=args.min_semitones,
            out_dir=args.out_dir or "results/phase2_progressions",
        )
