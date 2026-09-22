"""Streaming, globally aggregated reconstruction and feature-usage metrics.

All errors are token weighted. FVU/NMSE divides total squared error by variation
around the *global* feature-wise input mean. Explained variance instead centers
residuals, so it does not penalize a constant reconstruction bias. Zero-variance
inputs are flagged; ratio denominators are floored at ``eps`` in that case.
Feature death is observed over the complete evaluation stream, never per batch.
"""
from __future__ import annotations

import math
import torch


class ReconstructionMetrics:
    def __init__(self, *, dense_threshold: float = .1, activation_threshold: float = 0., eps: float = 1e-12):
        if not 0 < dense_threshold <= 1 or activation_threshold < 0 or eps <= 0:
            raise ValueError('Invalid metric thresholds')
        self.dense_threshold = dense_threshold
        self.activation_threshold = activation_threshold
        self.eps = eps
        self.n_tokens = 0
        self.d_in = self.d_sae = None
        self._mean_x = self._mean_error = None
        self._m2_x = self._m2_error = 0.
        self._sse = self._cosine = self._relative_error = self._norm_ratio = 0.
        self._feature_counts = self._l0_hist = None

    @torch.no_grad()
    def update(self, x: torch.Tensor, recon: torch.Tensor, z: torch.Tensor) -> None:
        if x.ndim != 2 or recon.shape != x.shape or z.ndim != 2 or len(x) != len(z):
            raise ValueError('Expected x/recon [tokens,d_in] and z [tokens,d_sae]')
        if not len(x):
            return
        if x.shape[1] == 0 or z.shape[1] == 0:
            raise ValueError('Input and dictionary dimensions must be positive')
        if self.d_in is not None and (x.shape[1] != self.d_in or z.shape[1] != self.d_sae):
            raise ValueError('Metric dimensions changed during evaluation')
        if not all(torch.isfinite(v).all().item() for v in (x, recon, z)):
            raise ValueError('Non-finite input, reconstruction or feature activation')
        # Float64 batch sufficient statistics; only small statistics persist on CPU.
        x = x.detach().to(dtype=torch.float64)
        recon = recon.detach().to(device=x.device, dtype=torch.float64)
        error = recon - x
        n = len(x)
        mean_x, mean_error = x.mean(0).cpu(), error.mean(0).cpu()
        m2_x = ((x - mean_x.to(x.device)) ** 2).sum().item()
        m2_error = ((error - mean_error.to(x.device)) ** 2).sum().item()
        if self.n_tokens:
            weight = self.n_tokens * n / (self.n_tokens + n)
            self._m2_x += m2_x + (mean_x - self._mean_x).square().sum().item() * weight
            self._m2_error += m2_error + (mean_error - self._mean_error).square().sum().item() * weight
            self._mean_x += (mean_x - self._mean_x) * n / (self.n_tokens + n)
            self._mean_error += (mean_error - self._mean_error) * n / (self.n_tokens + n)
        else:
            self.d_in, self.d_sae = x.shape[1], z.shape[1]
            self._mean_x, self._mean_error = mean_x, mean_error
            self._m2_x, self._m2_error = m2_x, m2_error
            self._feature_counts = torch.zeros(self.d_sae, dtype=torch.int64)
            self._l0_hist = torch.zeros(self.d_sae + 1, dtype=torch.int64)
        x_norm, recon_norm = x.norm(dim=-1), recon.norm(dim=-1)
        self._sse += error.square().sum().item()
        self._cosine += ((x * recon).sum(-1) / (x_norm * recon_norm).clamp_min(self.eps)).sum().item()
        self._relative_error += (error.norm(dim=-1) / x_norm.clamp_min(self.eps)).sum().item()
        self._norm_ratio += (recon_norm / x_norm.clamp_min(self.eps)).sum().item()
        active = z.detach().abs() > self.activation_threshold
        self._feature_counts += active.sum(0).cpu()
        self._l0_hist += torch.bincount(active.sum(-1), minlength=self.d_sae + 1).cpu()
        self.n_tokens += n

    @property
    def feature_frequencies(self) -> torch.Tensor:
        if not self.n_tokens:
            raise ValueError('No evaluated tokens')
        return self._feature_counts.to(torch.float64) / self.n_tokens

    @property
    def feature_counts(self) -> torch.Tensor:
        if not self.n_tokens:
            raise ValueError('No evaluated tokens')
        return self._feature_counts.clone()

    def _l0_quantile(self, q: float) -> float:
        rank = q * (self.n_tokens - 1)
        cumulative = self._l0_hist.cumsum(0)
        def at(index):
            return torch.searchsorted(cumulative, torch.tensor(index + 1)).item()
        lo, hi = math.floor(rank), math.ceil(rank)
        return at(lo) + (at(hi) - at(lo)) * (rank - lo)

    def compute(self) -> dict[str, float]:
        if not self.n_tokens:
            raise ValueError('No evaluated tokens')
        freq = self.feature_frequencies
        l0_values = torch.arange(self.d_sae + 1, dtype=torch.float64)
        l0_mean = (l0_values * self._l0_hist).sum().item() / self.n_tokens
        l0_var = ((l0_values - l0_mean).square() * self._l0_hist).sum().item() / self.n_tokens
        fvu = self._sse / max(self._m2_x, self.eps)
        result = {
            'n_tokens': float(self.n_tokens),
            'mse': self._sse / (self.n_tokens * self.d_in),
            'l2_squared_error_mean': self._sse / self.n_tokens,
            'nmse': fvu, 'fvu': fvu, 'fraction_variance_explained': 1 - fvu,
            'explained_variance': 1 - self._m2_error / max(self._m2_x, self.eps),
            'variance_is_zero': float(self._m2_x <= self.eps),
            'cosine_similarity': self._cosine / self.n_tokens,
            'relative_norm_error': self._relative_error / self.n_tokens,
            'reconstruction_norm_ratio': self._norm_ratio / self.n_tokens,
            'l0_mean': l0_mean, 'l0_std': math.sqrt(max(l0_var, 0.)),
            'l0_p05': self._l0_quantile(.05), 'l0_p50': self._l0_quantile(.5),
            'l0_p95': self._l0_quantile(.95), 'l0_p99': self._l0_quantile(.99),
            'dead_feature_count': float((self._feature_counts == 0).sum().item()),
            'dead_feature_fraction': (freq == 0).double().mean().item(),
            'dense_feature_fraction': (freq >= self.dense_threshold).double().mean().item(),
            'feature_frequency_mean': freq.mean().item(),
            'feature_frequency_p50': freq.quantile(.5).item(),
            'feature_frequency_p95': freq.quantile(.95).item(),
            'feature_frequency_max': freq.max().item(),
        }
        return result
