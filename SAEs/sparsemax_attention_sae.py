"""Sparsemax attention SAE adapted from ChemDFM_SAE_Training c9b6d55.

The reference's learned query/keys, decoder scales and participation-ratio L0
objective are retained. Numerical fixes use float32 sparsemax, handle an all-IDF-
masked dictionary, and count firing rates across any leading input dimensions.
The encoder bias participates in scores (the reference registered but never used it).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SparsemaxAttentionSAEConfig:
    d_in: int
    d_sae: int
    dtype: str = "float32"
    device: str = "cpu"
    decoder_init_norm: float = 0.1
    key_dim: int | None = None
    preselect_k: int | None = 2048
    activation_mode: str = "probs"
    use_input_norm: bool = True
    use_idf_mask: bool = False
    idf_threshold: float = 0.1
    mse_loss_scale: float = 1.0
    score_scale: float = 2.0
    l0_target: float | None = 32.0
    l0_coefficient: float = 10.0
    cosine_loss_coefficient: float = 0.0
    norm_loss_coefficient: float = 0.0
    value_scale_init: float = 1.0
    global_output_scale_init: float = 1.0
    key_init_std: float | None = None
    eps: float = 1e-8
    architecture: str = "sparsemax_attention"

    def __post_init__(self):
        cfg = self
        if cfg.activation_mode not in {"probs", "masked_scores"}:
            raise ValueError("activation_mode must be 'probs' or 'masked_scores'")
        if cfg.key_dim is None:
            cfg.key_dim = min(1024, cfg.d_in)
        if type(cfg.key_dim) is not int or cfg.key_dim <= 0:
            raise ValueError("key_dim must be a positive integer")
        if cfg.preselect_k is not None and (type(cfg.preselect_k) is not int or cfg.preselect_k <= 0):
            raise ValueError("preselect_k must be a positive integer or null")
        if cfg.eps <= 0 or cfg.decoder_init_norm <= 0:
            raise ValueError("eps and decoder_init_norm must be positive")
        if cfg.l0_target is not None and cfg.l0_target <= 0:
            raise ValueError("l0_target must be positive or null")
        for name in ("mse_loss_scale", "l0_coefficient", "cosine_loss_coefficient", "norm_loss_coefficient"):
            value = getattr(cfg, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


class SparsemaxAttentionSAE(nn.Module):
    """Sparsemax selection with an independently learned decoder dictionary."""

    def __init__(self, cfg: SparsemaxAttentionSAEConfig):
        super().__init__()
        self.cfg = cfg
        factory = {"device": cfg.device, "dtype": getattr(torch, cfg.dtype)}
        self.W_dec = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_in, **factory))
        self.W_q = nn.Parameter(torch.empty(cfg.d_in, cfg.key_dim, **factory))
        self.W_key = nn.Parameter(torch.empty(cfg.d_sae, cfg.key_dim, **factory))
        self.value_scale = nn.Parameter(torch.full((cfg.d_sae,), cfg.value_scale_init, **factory))
        self.global_output_scale = nn.Parameter(torch.tensor(cfg.global_output_scale_init, **factory))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae, **factory))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_in, **factory))
        self.register_buffer("idf_score", torch.zeros(cfg.d_sae, device=cfg.device, dtype=torch.float32))
        self.register_buffer("num_encode_calls", torch.zeros((), device=cfg.device, dtype=torch.long))
        with torch.no_grad():
            nn.init.normal_(self.W_dec)
            self.W_dec.copy_(F.normalize(self.W_dec, dim=-1) * cfg.decoder_init_norm)
            nn.init.normal_(self.W_q, std=cfg.d_in ** -0.5)
            nn.init.normal_(self.W_key, std=cfg.key_init_std or cfg.key_dim ** -0.5)

    def get_coefficients(self):
        return {}

    @staticmethod
    def sparsemax(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Euclidean simplex projection; exact zeros with autograd on support."""
        work = scores.float() if scores.dtype in (torch.float16, torch.bfloat16) else scores
        z = work - work.amax(dim=dim, keepdim=True)
        ordered = z.sort(dim=dim, descending=True).values
        ranks = torch.arange(1, z.shape[dim] + 1, device=z.device, dtype=z.dtype)
        shape = [1] * z.ndim
        shape[dim] = -1
        support = 1 + ordered * ranks.view(shape) > ordered.cumsum(dim)
        count = support.sum(dim=dim, keepdim=True).clamp_min(1)
        tau = (ordered.masked_fill(~support, 0).sum(dim=dim, keepdim=True) - 1) / count
        return (z - tau).clamp_min(0).to(scores.dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.W_dec.device, dtype=self.W_dec.dtype)
        input_norm = x.norm(dim=-1, keepdim=True).clamp_min(self.cfg.eps) if self.cfg.use_input_norm else None
        normalized = x / input_norm if input_norm is not None else x
        queries = F.normalize(normalized @ self.W_q, dim=-1, eps=self.cfg.eps)
        keys = F.normalize(self.W_key, dim=-1, eps=self.cfg.eps)
        scores = (queries @ keys.T) * self.cfg.score_scale + self.b_enc
        if self.cfg.use_idf_mask:
            masked = self.idf_score > self.cfg.idf_threshold
            if masked.all():
                masked = masked.clone()
                masked[self.idf_score.argmin()] = False
            scores = scores.masked_fill(masked, torch.finfo(scores.dtype).min)
        k = self.cfg.preselect_k
        if k is not None and k < self.cfg.d_sae:
            selected, indices = scores.topk(k, dim=-1)
            probs = self.sparsemax(selected)
            selected_acts = selected * (probs > 0) if self.cfg.activation_mode == "masked_scores" else probs
            acts = torch.zeros_like(scores).scatter(-1, indices, selected_acts)
        else:
            probs = self.sparsemax(scores)
            acts = scores * (probs > 0) if self.cfg.activation_mode == "masked_scores" else probs
        if input_norm is not None:
            acts = acts * input_norm
        if self.training:
            with torch.no_grad():
                fired = (acts.detach() != 0).float().reshape(-1, self.cfg.d_sae).mean(0)
                self.num_encode_calls.add_(1)
                self.idf_score.add_((fired - self.idf_score) / self.num_encode_calls)
        return acts

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        feature_acts = feature_acts.to(device=self.W_dec.device, dtype=self.W_dec.dtype)
        return ((feature_acts * self.value_scale) @ self.W_dec) * self.global_output_scale + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        acts = self.encode(x)
        return self.decode(acts), acts

    def training_forward_pass(self, step_input):
        x = step_input.sae_in.to(device=self.W_dec.device, dtype=self.W_dec.dtype)
        recon, acts = self(x)
        weights = acts.float().abs()
        l0_proxy = (weights.sum(-1).square() / weights.square().sum(-1).clamp_min(self.cfg.eps)).mean()
        mse_loss = (recon.float() - x.float()).square().mean() * self.cfg.mse_loss_scale
        l0_loss = x.new_zeros((), dtype=torch.float32)
        if self.cfg.l0_target is not None and self.cfg.l0_coefficient > 0:
            target = self.cfg.l0_target
            l0_loss = self.cfg.l0_coefficient * ((l0_proxy - target) / max(1.0, target)).square()
        cosine_raw = 1 - F.cosine_similarity(recon.float(), x.float(), dim=-1, eps=self.cfg.eps).mean()
        norm_in = x.float().norm(dim=-1).clamp_min(self.cfg.eps)
        norm_rel_error = ((recon.float().norm(dim=-1) - norm_in) / norm_in).square().mean()
        cosine_loss = cosine_raw * self.cfg.cosine_loss_coefficient
        norm_loss = norm_rel_error * self.cfg.norm_loss_coefficient
        loss = mse_loss + l0_loss + cosine_loss + norm_loss
        return SimpleNamespace(
            sae_in=x, sae_out=recon, feature_acts=acts, hidden_pre=None, loss=loss,
            losses={"mse_loss": mse_loss, "l0_target_loss": l0_loss,
                    "cosine_loss": cosine_loss, "norm_loss": norm_loss},
            metrics={"l0_proxy": l0_proxy.detach(), "cosine_raw": cosine_raw.detach(),
                     "norm_rel_error": norm_rel_error.detach()},
        )
