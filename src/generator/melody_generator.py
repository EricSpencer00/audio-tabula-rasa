"""
Phase-3 monophonic-melody generator.

Output: a sequence of N notes (frequencies, in Hz) sampled from a
Gaussian policy. No recurrence, no music prior — same tabula-rasa
philosophy as the earlier phases. The temporal structure of melody
must come from the reward model alone (consonance, tonal coherence,
contour smoothness).
"""
import torch
import torch.nn as nn


class MelodyGenerator(nn.Module):
    """
    Output: n_notes frequencies in [F_MIN, F_MAX] via a Gaussian policy
    in log-frequency. Perceptual distance in pitch is logarithmic in
    Hz, so it is more natural for both exploration noise and the
    sigmoid squash to live in log space.
    """
    F_MIN = 110.0   # A2
    F_MAX = 880.0   # A5

    def __init__(self, latent_dim: int = 16, hidden: int = 128,
                 n_notes: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_notes = n_notes
        # log-Hz range
        import math
        self._log_lo = math.log(self.F_MIN)
        self._log_hi = math.log(self.F_MAX)

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * n_notes),
        )

    def forward(self, z: torch.Tensor):
        params = self.net(z)
        mean_raw = params[:, : self.n_notes]
        log_std = params[:, self.n_notes :].clamp(min=-4.0, max=1.0)
        log_mean = torch.sigmoid(mean_raw) * (self._log_hi - self._log_lo) + self._log_lo
        std = torch.exp(log_std)
        return log_mean, std

    def sample(self, batch_size: int, device: str = "cpu"):
        z = torch.randn(batch_size, self.latent_dim, device=device)
        log_mean, std = self.forward(z)
        dist = torch.distributions.Normal(log_mean, std)
        log_freqs = dist.rsample()
        log_freqs = log_freqs.clamp(min=self._log_lo, max=self._log_hi)
        log_prob = dist.log_prob(log_freqs).sum(dim=-1)
        freqs = torch.exp(log_freqs)
        return freqs, log_prob
