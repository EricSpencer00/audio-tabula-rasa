"""
Phase-7 counterpoint generator.

Outputs V monophonic voices, each a sequence of N log-frequencies. The
voices are not constrained to occupy fixed registers — register
separation, if it emerges, must come from the reward (the cross-voice
roughness term punishes near-unison clashes).

Each voice is parameterized by independent Gaussian heads in
log-frequency, exactly like the Phase-3 melody generator. The whole
network is one shared MLP from a common latent — so cross-voice
coordination has somewhere to live (in the weights from z to per-head
means).
"""
import math

import torch
import torch.nn as nn


class CounterpointGenerator(nn.Module):
    """
    V-voice counterpoint generator.

    Each voice has its own log-frequency band so the network can break
    the symmetry between voices. The bands overlap (~half-octave per
    voice's worth on each side) so the reward can still discover its
    own ordering — this is a task constraint, not a music prior.
    """
    F_MIN = 110.0    # A2
    F_MAX = 1760.0   # A6

    def __init__(self, latent_dim: int = 24, hidden: int = 192,
                 n_voices: int = 2, n_notes: int = 8,
                 band_overlap_octaves: float = 0.5):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_voices = n_voices
        self.n_notes = n_notes
        self.out_dim = n_voices * n_notes
        log_lo = math.log(self.F_MIN)
        log_hi = math.log(self.F_MAX)
        total_octaves = (log_hi - log_lo) / math.log(2)
        # Per-voice band width: total / V, centered evenly with overlap.
        band_octaves = total_octaves / n_voices + 2 * band_overlap_octaves
        band_log_width = band_octaves * math.log(2)

        # Voice i band center
        centers = []
        for i in range(n_voices):
            frac = (i + 0.5) / n_voices
            centers.append(log_lo + frac * (log_hi - log_lo))
        per_voice_log_lo = torch.tensor(centers) - band_log_width / 2.0
        per_voice_log_hi = torch.tensor(centers) + band_log_width / 2.0
        # Clip to global range
        per_voice_log_lo = torch.clamp(per_voice_log_lo, min=log_lo)
        per_voice_log_hi = torch.clamp(per_voice_log_hi, max=log_hi)

        # Repeat the per-voice bounds to (V*N,) for the output heads
        self.register_buffer("log_lo_per_head",
                             per_voice_log_lo.repeat_interleave(n_notes))
        self.register_buffer("log_hi_per_head",
                             per_voice_log_hi.repeat_interleave(n_notes))

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * self.out_dim),
        )

    def forward(self, z: torch.Tensor):
        params = self.net(z)
        mean_raw = params[:, : self.out_dim]
        log_std = params[:, self.out_dim :].clamp(min=-4.0, max=1.0)
        log_mean = (
            torch.sigmoid(mean_raw)
            * (self.log_hi_per_head - self.log_lo_per_head)
            + self.log_lo_per_head
        )
        return log_mean, torch.exp(log_std)

    def sample(self, batch_size: int, device: str = "cpu"):
        z = torch.randn(batch_size, self.latent_dim, device=device)
        log_mean, std = self.forward(z)
        dist = torch.distributions.Normal(log_mean, std)
        log_freqs = dist.rsample()
        log_freqs = torch.clamp(log_freqs,
                                min=self.log_lo_per_head,
                                max=self.log_hi_per_head)
        log_prob = dist.log_prob(log_freqs).sum(dim=-1)
        freqs = torch.exp(log_freqs)
        freqs = freqs.view(batch_size, self.n_voices, self.n_notes)
        return freqs, log_prob

    def entropy_at(self, z: torch.Tensor):
        log_mean, std = self.forward(z)
        return torch.distributions.Normal(log_mean, std).entropy().sum(dim=-1).mean()
