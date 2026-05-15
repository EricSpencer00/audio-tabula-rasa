"""
Phase-11 autoregressive melody generator.

A GRU emits N notes one at a time, conditioning each next-note
distribution on the embedding of the previous note(s). Same Phase-3
log-frequency Gaussian policy at each step, just temporally
unrolled — this is the architecture you reach for when you want
*motif* structure (repeated subsequences) to be learnable.

Output shape after sampling: (B, N) log-frequencies in [F_MIN, F_MAX].
"""
import math

import torch
import torch.nn as nn


class AutoregressiveMelodyGenerator(nn.Module):
    F_MIN = 110.0
    F_MAX = 880.0

    def __init__(self, hidden: int = 96, n_notes: int = 16,
                 init_noise_dim: int = 8):
        super().__init__()
        self.n_notes = n_notes
        self.hidden = hidden
        self.init_noise_dim = init_noise_dim
        self._log_lo = math.log(self.F_MIN)
        self._log_hi = math.log(self.F_MAX)

        # Initial hidden state is conditioned on a latent z so the
        # generator can produce different "melodies" given different
        # z's. Without z, all melodies collapse to the same trajectory.
        self.init_proj = nn.Sequential(
            nn.Linear(init_noise_dim, hidden),
            nn.Tanh(),
        )
        # Input at each step is the previous note (1 scalar in
        # normalized log-freq) + position embedding
        self.position_emb = nn.Embedding(n_notes, 8)
        self.gru = nn.GRUCell(input_size=1 + 8, hidden_size=hidden)
        self.out_proj = nn.Linear(hidden, 2)   # mean, log_std

    def _expand_init(self, batch_size: int, device: str):
        z = torch.randn(batch_size, self.init_noise_dim, device=device)
        h = self.init_proj(z)
        return h

    def sample(self, batch_size: int, device: str = "cpu"):
        h = self._expand_init(batch_size, device)
        # Bootstrap with a neutral starting "note" embedding
        prev_log = torch.zeros(batch_size, 1, device=device)
        notes = []
        log_probs = []
        for t in range(self.n_notes):
            pos = torch.full((batch_size,), t, dtype=torch.long, device=device)
            pe = self.position_emb(pos)
            inp = torch.cat([prev_log, pe], dim=1)
            h = self.gru(inp, h)
            params = self.out_proj(h)
            mean_raw = params[:, 0]
            log_std = params[:, 1].clamp(min=-4.0, max=1.0)
            mean = (torch.sigmoid(mean_raw)
                    * (self._log_hi - self._log_lo)
                    + self._log_lo)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            sample_log = dist.rsample().clamp(min=self._log_lo, max=self._log_hi)
            log_probs.append(dist.log_prob(sample_log))
            # Feed the normalized log-freq as the next input
            prev_log = ((sample_log - self._log_lo)
                        / (self._log_hi - self._log_lo)).unsqueeze(-1)
            notes.append(torch.exp(sample_log))

        freqs = torch.stack(notes, dim=1)
        log_prob = torch.stack(log_probs, dim=1).sum(dim=-1)
        return freqs, log_prob

    def entropy_at(self, batch_size: int, device: str = "cpu"):
        """Quick estimate of the policy's entropy over a fresh batch."""
        h = self._expand_init(batch_size, device)
        prev_log = torch.zeros(batch_size, 1, device=device)
        total_entropy = 0.0
        for t in range(self.n_notes):
            pos = torch.full((batch_size,), t, dtype=torch.long, device=device)
            pe = self.position_emb(pos)
            inp = torch.cat([prev_log, pe], dim=1)
            h = self.gru(inp, h)
            params = self.out_proj(h)
            mean_raw = params[:, 0]
            log_std = params[:, 1].clamp(min=-4.0, max=1.0)
            mean = (torch.sigmoid(mean_raw)
                    * (self._log_hi - self._log_lo)
                    + self._log_lo)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            total_entropy = total_entropy + dist.entropy().mean()
            # Use the dist mean as the next input — deterministic
            sample_log = mean
            prev_log = ((sample_log - self._log_lo)
                        / (self._log_hi - self._log_lo)).unsqueeze(-1)
        return total_entropy / self.n_notes
