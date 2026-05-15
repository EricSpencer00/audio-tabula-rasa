"""
Toy generator: tabula-rasa policy that outputs two frequencies (an interval).
Initialized randomly. Has no concept of music. Learns only via reward signal
from the psychoacoustic reward model.
"""
import torch
import torch.nn as nn


class ToyIntervalGenerator(nn.Module):
    """
    Input: latent noise z ∈ R^8
    Output: distribution over (f1, f2) in Hz, sampled from a Gaussian policy.
    Frequency range squashed to [F_MIN, F_MAX] via sigmoid.
    """
    F_MIN = 110.0   # A2
    F_MAX = 880.0   # A5

    def __init__(self, latent_dim: int = 8, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4),   # 2 means + 2 log_stds
        )
        self.latent_dim = latent_dim

    def forward(self, z: torch.Tensor):
        params = self.net(z)
        mean_raw = params[:, :2]
        log_std = params[:, 2:].clamp(min=-2.0, max=3.0)
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
