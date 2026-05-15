"""
Phase-4 rhythm generator.

Outputs N onset times in [0, T_window] via a Gaussian policy. We
parameterize the policy in *inter-onset intervals* (IOIs) so that the
generator's output is already a sorted onset sequence: each head
produces a positive duration sampled from a softplus-Gaussian, and the
onsets are cumulative sums clipped to the window.

There is no built-in beat, tempo, or grid — the policy must rediscover
periodicity from the rhythmic-entrainment reward alone.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RhythmGenerator(nn.Module):
    def __init__(self, latent_dim: int = 16, hidden: int = 128,
                 n_onsets: int = 8, duration: float = 4.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_onsets = n_onsets
        self.duration = duration

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * n_onsets),
        )

    def forward(self, z: torch.Tensor):
        params = self.net(z)
        mean_raw = params[:, : self.n_onsets]
        log_std = params[:, self.n_onsets :].clamp(min=-3.0, max=1.0)
        return mean_raw, torch.exp(log_std)

    def sample(self, batch_size: int, device: str = "cpu"):
        """
        Returns (onsets, log_prob).

        Onsets are produced as cumulative sums of soft-positive IOIs
        derived from a Gaussian policy in pre-softplus space. The
        cumulative-sum step is the path through which the policy
        gradient sees inter-onset timing.
        """
        z = torch.randn(batch_size, self.latent_dim, device=device)
        mean_raw, std = self.forward(z)
        dist = torch.distributions.Normal(mean_raw, std)
        raw = dist.rsample()
        log_prob = dist.log_prob(raw).sum(dim=-1)

        # Soft-positive IOI scale: softplus keeps gradients smooth and
        # avoids the heavy-tailed behavior of exp. We also add a small
        # floor so the policy can never park every onset at the same
        # time, which the min_ioi penalty would catch anyway.
        iois = F.softplus(raw) + 0.02       # at least 20 ms apart
        # Normalize so that cumulative IOI sum sits within the window.
        # We allow a slight overflow because clipping is cheap and
        # rescaling would introduce a non-physical bias.
        max_total = self.duration * 0.95
        scale = (max_total / (iois.sum(dim=-1, keepdim=True) + 1e-8)).clamp(max=1.0)
        iois = iois * scale
        onsets = torch.cumsum(iois, dim=-1)
        return onsets, log_prob
