"""
Phase 4.5 — joint melodic-rhythm generator.

Outputs N (log-frequency, IOI) pairs from a single shared MLP. The
sampled pitches feed the Phase-3 melody reward; the sampled onsets
(cumulative-sum of the IOIs) feed the Phase-4 rhythm reward. The
generator shares enough capacity that pitch and timing can co-vary if
the reward signals favor it.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MelodicRhythmGenerator(nn.Module):
    F_MIN = 110.0
    F_MAX = 880.0

    def __init__(self, latent_dim: int = 24, hidden: int = 192,
                 n_notes: int = 8, duration: float = 4.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_notes = n_notes
        self.duration = duration
        self._log_lo = math.log(self.F_MIN)
        self._log_hi = math.log(self.F_MAX)
        self.out_dim = 4 * n_notes  # pitch_mean, pitch_log_std, ioi_mean, ioi_log_std

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.out_dim),
        )

    def forward(self, z: torch.Tensor):
        params = self.net(z)
        n = self.n_notes
        pitch_mean_raw = params[:, : n]
        pitch_log_std = params[:, n : 2 * n].clamp(min=-4.0, max=1.0)
        ioi_mean_raw = params[:, 2 * n : 3 * n]
        ioi_log_std = params[:, 3 * n :].clamp(min=-3.0, max=1.0)
        log_pitch_mean = torch.sigmoid(pitch_mean_raw) \
                         * (self._log_hi - self._log_lo) + self._log_lo
        return log_pitch_mean, torch.exp(pitch_log_std), \
               ioi_mean_raw, torch.exp(ioi_log_std)

    def sample(self, batch_size: int, device: str = "cpu"):
        z = torch.randn(batch_size, self.latent_dim, device=device)
        log_pitch_mean, pitch_std, ioi_mean_raw, ioi_std = self.forward(z)

        pitch_dist = torch.distributions.Normal(log_pitch_mean, pitch_std)
        ioi_dist = torch.distributions.Normal(ioi_mean_raw, ioi_std)

        log_pitch = pitch_dist.rsample().clamp(min=self._log_lo, max=self._log_hi)
        ioi_raw = ioi_dist.rsample()
        freqs = torch.exp(log_pitch)

        # Phase-4 onset construction
        iois = F.softplus(ioi_raw) + 0.02
        max_total = self.duration * 0.95
        scale = (max_total / (iois.sum(dim=-1, keepdim=True) + 1e-8)).clamp(max=1.0)
        iois = iois * scale
        onsets = torch.cumsum(iois, dim=-1)

        log_prob = (
            pitch_dist.log_prob(log_pitch).sum(dim=-1)
            + ioi_dist.log_prob(ioi_raw).sum(dim=-1)
        )
        return freqs, onsets, log_prob

    def entropy_at(self, z: torch.Tensor):
        log_pitch_mean, pitch_std, ioi_mean_raw, ioi_std = self.forward(z)
        pd = torch.distributions.Normal(log_pitch_mean, pitch_std)
        od = torch.distributions.Normal(ioi_mean_raw, ioi_std)
        return (pd.entropy().sum(dim=-1) + od.entropy().sum(dim=-1)).mean()
