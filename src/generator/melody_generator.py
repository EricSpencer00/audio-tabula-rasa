"""
Phase-3 monophonic-melody generator.

Output: a sequence of N notes (frequencies, in Hz) sampled from a
Gaussian policy. No recurrence, no music prior — same tabula-rasa
philosophy as the earlier phases. The temporal structure of melody
must come from the reward model alone (consonance, tonal coherence,
contour smoothness).
"""
import math

import numpy as np
import torch
import torch.nn as nn


# C major scale tones as semitones relative to A4 = 440 Hz
# Covers 110 Hz (A2) to 880 Hz (A5): 3 octaves
_SCALE_SEMITONES_C_MAJOR = []
for octave_offset in range(-24, 25):  # wide range
    for degree in (0, 2, 4, 5, 7, 9, 11):  # C major intervals from C
        st = degree + octave_offset * 12 - 9  # offset from A4
        hz = 440.0 * 2.0 ** (st / 12.0)
        if 100.0 <= hz <= 900.0:
            _SCALE_SEMITONES_C_MAJOR.append(math.log(hz))
_SCALE_LOG_HZ = torch.tensor(sorted(set(_SCALE_SEMITONES_C_MAJOR)),
                              dtype=torch.float32)


def _snap_to_scale(log_freqs: torch.Tensor) -> torch.Tensor:
    """Snap log-frequencies to nearest C major scale tone (hard quantize)."""
    scale = _SCALE_LOG_HZ.to(log_freqs.device)  # (S,)
    # distances: (B, N, S)
    dists = (log_freqs.unsqueeze(-1) - scale.unsqueeze(0).unsqueeze(0)).abs()
    nearest_idx = dists.argmin(dim=-1)  # (B, N)
    return scale[nearest_idx]  # (B, N)


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


class ExpressiveMelodyGenerator(nn.Module):
    """Outputs frequencies + durations + velocities per note.

    The ``sample()`` return is ``(combined, log_prob)`` where *combined*
    has shape ``(B, 3*n_notes)`` laid out as ``[freqs | durations | velocities]``.
    Frequencies and durations are sampled from independent Gaussian
    policies (both contribute to log_prob); velocities are deterministic.
    """

    F_MIN, F_MAX = 110.0, 880.0
    DUR_MIN, DUR_MAX = 0.10, 0.80
    VEL_MIN, VEL_MAX = 0.3, 1.0

    def __init__(self, latent_dim: int = 32, hidden: int = 256,
                 n_notes: int = 16, freq_std_clamp: float = 1.0,
                 scale_snap: float = 0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_notes = n_notes
        self._log_lo = math.log(self.F_MIN)
        self._log_hi = math.log(self.F_MAX)
        self._log_dur_lo = math.log(self.DUR_MIN)
        self._log_dur_hi = math.log(self.DUR_MAX)
        self._freq_std_clamp = freq_std_clamp
        self.scale_snap = scale_snap

        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.freq_head = nn.Linear(hidden, 2 * n_notes)
        self.dur_head = nn.Linear(hidden, 2 * n_notes)
        self.vel_head = nn.Linear(hidden, n_notes)

    def forward(self, z: torch.Tensor):
        h = self.backbone(z)

        fp = self.freq_head(h)
        freq_mean = (torch.sigmoid(fp[:, :self.n_notes])
                     * (self._log_hi - self._log_lo) + self._log_lo)
        freq_std = torch.exp(fp[:, self.n_notes:].clamp(-4.0, self._freq_std_clamp))

        dp = self.dur_head(h)
        dur_mean = (torch.sigmoid(dp[:, :self.n_notes])
                    * (self._log_dur_hi - self._log_dur_lo) + self._log_dur_lo)
        dur_std = torch.exp(dp[:, self.n_notes:].clamp(-4.0, 0.0))

        vel = (torch.sigmoid(self.vel_head(h))
               * (self.VEL_MAX - self.VEL_MIN) + self.VEL_MIN)

        return freq_mean, freq_std, dur_mean, dur_std, vel

    def sample(self, batch_size: int, device: str = "cpu"):
        z = torch.randn(batch_size, self.latent_dim, device=device)
        freq_mean, freq_std, dur_mean, dur_std, vel = self.forward(z)

        fd = torch.distributions.Normal(freq_mean, freq_std)
        log_freqs = fd.rsample().clamp(self._log_lo, self._log_hi)

        if self.scale_snap > 0:
            snapped = _snap_to_scale(log_freqs)
            log_freqs = (1 - self.scale_snap) * log_freqs + self.scale_snap * snapped

        dd = torch.distributions.Normal(dur_mean, dur_std)
        log_durs = dd.rsample().clamp(self._log_dur_lo, self._log_dur_hi)

        freq_lp = fd.log_prob(log_freqs)   # (B, N)
        dur_lp = dd.log_prob(log_durs)     # (B, N)
        log_prob = freq_lp.sum(-1) + dur_lp.sum(-1)  # (B,)

        combined = torch.cat([torch.exp(log_freqs),
                              torch.exp(log_durs),
                              vel], dim=-1)
        self._last_per_note_lp = freq_lp + dur_lp  # (B, N)
        return combined, log_prob
