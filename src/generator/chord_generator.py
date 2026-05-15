"""
Tabula-rasa generators for Phase 2:

- TriadGenerator           : outputs a single 3-note chord.
- ChordProgressionGenerator: outputs K consecutive 3-note chords.

Both have no built-in concept of pitch class, key, or interval. They
just emit frequencies in [F_MIN, F_MAX] via a Gaussian policy. All
musical structure must come from the reward.
"""
import torch
import torch.nn as nn


class TriadGenerator(nn.Module):
    """
    Input : latent z ∈ R^latent_dim
    Output: (f1, f2, f3) Gaussian-policy sample of 3 frequencies.
    """
    F_MIN = 110.0   # A2
    F_MAX = 880.0   # A5

    def __init__(self, latent_dim: int = 8, hidden: int = 64, n_voices: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_voices = n_voices
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * n_voices),  # means + log_stds
        )

    def forward(self, z: torch.Tensor):
        params = self.net(z)
        mean_raw = params[:, : self.n_voices]
        log_std = params[:, self.n_voices :].clamp(min=-3.0, max=2.0)
        mean = torch.sigmoid(mean_raw) * (self.F_MAX - self.F_MIN) + self.F_MIN
        std = torch.exp(log_std)
        return mean, std

    def sample(self, batch_size: int, device: str = "cpu"):
        z = torch.randn(batch_size, self.latent_dim, device=device)
        mean, std = self.forward(z)
        dist = torch.distributions.Normal(mean, std)
        freqs = dist.rsample()
        freqs = freqs.clamp(min=self.F_MIN, max=self.F_MAX)
        log_prob = dist.log_prob(freqs).sum(dim=-1)
        return freqs, log_prob


class ChordProgressionGenerator(nn.Module):
    """
    Outputs K consecutive 3-note chords, all in one shot from a latent.

    Output shape after sampling: (B, K, n_voices) frequencies in Hz.
    log_prob is summed over all (K * n_voices) Gaussian dimensions.

    The network has no recurrence — every chord is generated jointly.
    That keeps the architecture as minimal as the toy case while still
    giving the voice-leading term something to grab onto.
    """
    F_MIN = 110.0
    F_MAX = 880.0

    def __init__(self, latent_dim: int = 16, hidden: int = 128,
                 n_chords: int = 4, n_voices: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_chords = n_chords
        self.n_voices = n_voices
        self.out_dim = n_chords * n_voices

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
        log_std = params[:, self.out_dim :].clamp(min=-3.0, max=2.0)
        mean = torch.sigmoid(mean_raw) * (self.F_MAX - self.F_MIN) + self.F_MIN
        std = torch.exp(log_std)
        return mean, std

    def sample(self, batch_size: int, device: str = "cpu"):
        z = torch.randn(batch_size, self.latent_dim, device=device)
        mean, std = self.forward(z)
        dist = torch.distributions.Normal(mean, std)
        freqs = dist.rsample()
        freqs = freqs.clamp(min=self.F_MIN, max=self.F_MAX)
        log_prob = dist.log_prob(freqs).sum(dim=-1)
        freqs = freqs.view(batch_size, self.n_chords, self.n_voices)
        return freqs, log_prob
