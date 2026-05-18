"""Integration tests for theory judge training pipeline.

Runs minimal training steps to verify the full loop:
generator -> theory judge -> per-note REINFORCE -> weight update.

These tests are fast (~1s each) since theory scoring doesn't render audio.
"""
import numpy as np
import pytest
import torch
from pathlib import Path

from src.train.rlaif_train import _ADAPTERS, _render_and_score


class _TheoryJudgeForTest:
    """Minimal theory judge matching the real _TheoryJudge interface."""
    is_theory = True

    def score(self, data, output_format="freqs"):
        from src.reward.theory_judge import (
            theory_reward, theory_reward_breakdown, theory_reward_per_note,
        )
        per_note = None
        if output_format == "voices":
            freqs_flat = data.flatten()
            score = theory_reward(freqs_flat, voices=data)
            breakdown = theory_reward_breakdown(freqs_flat, voices=data)
            per_note = theory_reward_per_note(freqs_flat)
        elif output_format == "expressive":
            n = len(data) // 3
            f, d, v = data[:n], data[n:2*n], data[2*n:]
            score = theory_reward(f, durations=d, velocities=v)
            breakdown = theory_reward_breakdown(f, durations=d, velocities=v)
            per_note = theory_reward_per_note(f, durations=d, velocities=v)
        else:
            score = theory_reward(data)
            breakdown = theory_reward_breakdown(data)
            per_note = theory_reward_per_note(data)
        critique = " | ".join(f"{k}={val:.2f}" for k, val in breakdown.items())
        return type("R", (), {"score": score, "critique": critique,
                              "per_note": per_note})()


class TestRenderAndScore:
    """Test _render_and_score with theory judge for all output formats."""

    def test_expressive_format(self):
        adapter = _ADAPTERS["melody_v7"]
        gen = adapter.build()
        torch.manual_seed(0)
        with torch.no_grad():
            freqs, _ = gen.sample(2)
        scores, phys, critiques, pn = _render_and_score(
            freqs.numpy(), adapter, _TheoryJudgeForTest())
        assert scores.shape == (2,)
        assert phys.shape == (2,)
        assert len(critiques) == 2
        assert pn is not None
        assert pn.shape == (2, 16)

    def test_voices_format(self):
        adapter = _ADAPTERS["counterpoint"]
        gen = adapter.build()
        torch.manual_seed(0)
        with torch.no_grad():
            freqs, _ = gen.sample(2)
        scores, phys, critiques, pn = _render_and_score(
            freqs.numpy(), adapter, _TheoryJudgeForTest())
        assert scores.shape == (2,)
        assert pn is not None

    def test_freqs_format(self):
        adapter = _ADAPTERS["melody"]
        gen = adapter.build()
        torch.manual_seed(0)
        with torch.no_grad():
            freqs, _ = gen.sample(2)
        scores, phys, critiques, pn = _render_and_score(
            freqs.numpy(), adapter, _TheoryJudgeForTest())
        assert scores.shape == (2,)
        assert pn is not None


class TestTrainingStep:
    """Test a single REINFORCE training step with theory judge."""

    def test_expressive_training_step(self):
        """Full forward-score-backward cycle for expressive generator."""
        adapter = _ADAPTERS["melody_v7"]
        gen = adapter.build()
        gen._freq_std_clamp = -2.0
        gen.scale_snap = 0.8
        opt = torch.optim.Adam(gen.parameters(), lr=1e-3)

        torch.manual_seed(42)
        params_before = {k: v.clone() for k, v in gen.named_parameters()}

        freqs, log_prob = gen.sample(4)
        freqs_np = freqs.detach().cpu().numpy()

        scores, phys, critiques, per_note_rewards = _render_and_score(
            freqs_np, adapter, _TheoryJudgeForTest())

        rewards = scores + 0.05 * phys
        adv = rewards - rewards.mean()

        per_note_lp = getattr(gen, "_last_per_note_lp", None)
        assert per_note_lp is not None, "Expressive generator must have per-note LP"
        assert per_note_rewards is not None

        pn_adv = torch.tensor(per_note_rewards, dtype=torch.float32)
        pn_adv = (pn_adv - pn_adv.mean()) / max(float(pn_adv.std()), 0.01)
        per_note_loss = -(per_note_lp * pn_adv).sum(-1).mean()
        scalar_loss = -(log_prob * torch.tensor(adv)).mean()
        loss = 0.3 * scalar_loss + 0.7 * per_note_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        # Verify weights actually changed
        changed = False
        for k, v in gen.named_parameters():
            if not torch.allclose(params_before[k], v):
                changed = True
                break
        assert changed, "Weights should update after training step"

    def test_scale_snap_produces_in_key(self):
        """With scale_snap=1.0, all notes should be perfectly in key."""
        from src.reward.theory_judge import key_adherence, SCALES
        adapter = _ADAPTERS["melody_v7"]
        gen = adapter.build()
        gen.scale_snap = 1.0

        torch.manual_seed(0)
        with torch.no_grad():
            combined, _ = gen.sample(1)
        arr = combined[0].numpy()
        f = arr[:16]
        ka = key_adherence(f, root_hz=261.63, scale=SCALES["major"])
        assert ka > 0.99, f"With scale_snap=1.0, key_adherence should be ~1.0, got {ka}"

    def test_freq_std_clamp_reduces_spread(self):
        """Clamping freq std should reduce pitch spread across samples."""
        adapter = _ADAPTERS["melody_v7"]

        gen_wide = adapter.build()
        gen_tight = adapter.build()
        gen_tight._freq_std_clamp = -2.0

        torch.manual_seed(0)
        with torch.no_grad():
            wide, _ = gen_wide.sample(10)
        torch.manual_seed(0)
        with torch.no_grad():
            tight, _ = gen_tight.sample(10)

        # Compare frequency spread (std across samples for each note position)
        wide_f = wide[:, :16].numpy()
        tight_f = tight[:, :16].numpy()
        assert tight_f.std() < wide_f.std(), \
            "Tighter std clamp should reduce frequency spread"
