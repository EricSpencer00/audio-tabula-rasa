"""
Overnight "ralph loop": iterate RLAIF passes over every available
generator, replace the in-tree weights when the new model scores
higher than the old one on the audio judge, re-render the preview,
and push.

Usage:
    nohup python scripts/ralph_loop.py \
        --judge mock \
        --rounds 24 \
        --steps-per-round 60 \
        > /tmp/ralph_loop.log 2>&1 &

The default --judge is `mock` (the feature-based stand-in that runs
without HF / Qwen). When you whitelist huggingface.co in the env
policy, swap it out:

    python scripts/ralph_loop.py --judge qwen --rounds 8 --steps-per-round 50

Per round (per generator):
  1. Sample 32 audio clips from the current weights, score, average.
  2. Run RLAIF on a fresh copy of those weights (`steps_per_round`).
  3. Sample 32 clips from the new weights, score, average.
  4. If new average > old average, replace results/<phase>/<...>.pt.
  5. Re-render the preview audio and commit + push.

The loop is robust to per-generator failures: if one generator's
RLAIF crashes, the next one in the rotation still runs.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import torch

from src.train.rlaif_train import (
    GENERATOR_FACTORIES, JUDGE_FACTORIES,
    PHYSICS_REWARDS, RENDERERS, train_rlaif,
)


# (generator_name, in_tree_weight_path) — the weight file we replace
# if RLAIF improves the average score.
GENERATORS = [
    ("melody", "results/phase3_melodies/melody_generator.pt"),
    ("chord_progression",
     "results/phase2_progressions/progression_generator.pt"),
]


def _eval_avg_score(generator_name: str, judge_name: str,
                    weight_path: str, n: int = 32) -> float:
    factory = GENERATOR_FACTORIES[generator_name]
    gen = factory()
    if Path(weight_path).exists():
        gen.load_state_dict(torch.load(weight_path, map_location="cpu"))
    gen.eval()
    judge = JUDGE_FACTORIES[judge_name](None, "auto")
    render = RENDERERS[generator_name]
    scores = []
    torch.manual_seed(0)
    with torch.no_grad():
        out = gen.sample(n)
    samples = out[0].detach().cpu().numpy()
    for sample in samples:
        try:
            audio, sr = render(sample)
            r = judge.score(audio, sr)
            if r.score is not None:
                scores.append(float(r.score))
        except Exception:    # noqa: BLE001
            continue
    return float(np.mean(scores)) if scores else 0.0


def _git_commit_and_push(message: str) -> None:
    try:
        subprocess.run(["git", "add", "-A"], check=True)
        # nothing-to-commit is fine
        result = subprocess.run(
            ["git", "commit", "-m", message],
            capture_output=True, text=True,
        )
        if "nothing to commit" in (result.stdout + result.stderr):
            return
        subprocess.run(["git", "push"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ralph] git operation failed: {e}")


def _rerender_preview() -> None:
    try:
        subprocess.run(
            ["python3", "-m", "src.render.render_phases"],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["python3", "-m", "src.render.render_song"],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[ralph] re-render failed: {e}")


def run_ralph(rounds: int = 24, steps_per_round: int = 60,
              batch_size: int = 4, judge: str = "mock",
              n_eval: int = 32, log_path: str = "results/ralph_loop.log",
              push_every: int = 1):
    Path("results").mkdir(exist_ok=True)
    log = []
    log_path = Path(log_path)

    for round_idx in range(rounds):
        round_start = time.time()
        print(f"\n========== ralph round {round_idx + 1}/{rounds} ==========")
        for gen_name, weight_path in GENERATORS:
            t0 = time.time()
            print(f"\n[round {round_idx + 1}] generator: {gen_name}")
            try:
                old_avg = _eval_avg_score(gen_name, judge, weight_path,
                                          n=n_eval)
                print(f"  baseline avg score: {old_avg:.3f}")

                # Train a candidate
                tmp_dir = (Path("results/ralph_runs") /
                           f"round{round_idx:03d}_{gen_name}")
                tmp_dir.mkdir(parents=True, exist_ok=True)
                train_rlaif(
                    generator_name=gen_name,
                    n_steps=steps_per_round,
                    batch_size=batch_size,
                    judge=judge,
                    out_dir=str(tmp_dir),
                    seed=round_idx,
                )
                cand_path = tmp_dir / "rlaif_generator.pt"
                if not cand_path.exists():
                    print("  no candidate produced, skipping")
                    continue

                new_avg = _eval_avg_score(gen_name, judge,
                                          str(cand_path), n=n_eval)
                print(f"  candidate avg score: {new_avg:.3f}")

                replaced = False
                if new_avg > old_avg + 0.01:   # tiny tolerance
                    Path(weight_path).parent.mkdir(parents=True,
                                                    exist_ok=True)
                    shutil.copy2(cand_path, weight_path)
                    replaced = True
                    print(f"  REPLACED {weight_path} "
                          f"({old_avg:.3f} → {new_avg:.3f})")
                else:
                    print(f"  kept old (Δ={new_avg - old_avg:+.3f})")

                log.append({
                    "round": round_idx,
                    "generator": gen_name,
                    "old_score": old_avg,
                    "new_score": new_avg,
                    "replaced": replaced,
                    "duration_s": round(time.time() - t0, 1),
                })
                log_path.write_text(json.dumps(log, indent=2))
            except Exception as e:    # noqa: BLE001
                print(f"  ERROR in {gen_name}: {e}")
                traceback.print_exc()
                log.append({
                    "round": round_idx,
                    "generator": gen_name,
                    "error": str(e),
                })
                log_path.write_text(json.dumps(log, indent=2))

        # End of round: re-render and push
        try:
            _rerender_preview()
        except Exception as e:    # noqa: BLE001
            print(f"[ralph] re-render error: {e}")
        if (round_idx + 1) % push_every == 0:
            _git_commit_and_push(
                f"ralph loop: round {round_idx + 1}/{rounds}\n\n"
                f"Iterative RLAIF over {len(GENERATORS)} generators "
                f"with judge={judge}; see results/ralph_loop.log."
            )
        elapsed = time.time() - round_start
        print(f"\n[ralph] round {round_idx + 1} done in {elapsed:.1f}s")

    print("\n[ralph] loop complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--steps-per-round", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--judge", default="mock",
                        choices=list(JUDGE_FACTORIES))
    parser.add_argument("--n-eval", type=int, default=32)
    parser.add_argument("--push-every", type=int, default=1)
    args = parser.parse_args()
    run_ralph(
        rounds=args.rounds,
        steps_per_round=args.steps_per_round,
        batch_size=args.batch_size,
        judge=args.judge,
        n_eval=args.n_eval,
        push_every=args.push_every,
    )
