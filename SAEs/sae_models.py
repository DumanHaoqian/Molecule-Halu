"""SAE-Lens 6.51 model adapters for cached ChemDFM activation rows.

Training keeps the library's native objectives and decoder parameterization.
Use ``reconstruct`` for evaluation: native BatchTopK ``encode`` is batch-coupled
even in eval mode, whereas this adapter uses the same per-row JumpReLU gate as
SAE-Lens inference export. A zero (not yet calibrated) threshold falls back to
per-row TopK, with ceil(k) active features at most.
"""
from __future__ import annotations

import copy
import math
import os
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from torch import nn

ARCHITECTURES = ("batchtopk", "topk", "jumprelu", "matryoshka", "sparsemax_attention")


def _model_components(model_cfg: dict[str, Any], device: str):
    """Resolve native configuration and model class without model allocation.

    In addition to architecture/d_in/d_sae/k, architecture-specific native
    SAE-Lens config fields are accepted. Unknown fields fail rather than silently
    falling back to a library default. ``k`` is a comparison target only for
    JumpReLU and sparsemax; their actual sparsity is controlled by their losses.
    """
    cfg = copy.deepcopy(model_cfg)
    arch = cfg.pop("architecture", "batchtopk").lower()
    arch = {"matryoshka_batchtopk": "matryoshka", "batch_topk": "batchtopk"}.get(arch, arch)
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown SAE architecture {arch!r}; choose from {ARCHITECTURES}")
    for key in ("d_in", "d_sae"):
        value = cfg.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    k = cfg.pop("k", min(32, cfg["d_sae"]))
    if isinstance(k, bool) or not isinstance(k, (int, float)) or not math.isfinite(k) or not 0 < k <= cfg["d_sae"]:
        raise ValueError("k must be finite and satisfy 0 < k <= d_sae")
    if arch == "topk" and int(k) != k:
        raise ValueError("TopK requires integer k")
    if cfg.get("normalize_activations", "none") != "none":
        raise ValueError("normalize_activations must be 'none'; the pipeline normalizer owns scaling")
    if cfg.get("dtype", "float32") != "float32":
        raise ValueError("Use float32 SAE parameters; mixed precision is controlled by the trainer")
    if cfg.get("use_sparse_activations", False):
        raise ValueError("This pipeline requires dense feature tensors; use_sparse_activations must be false")
    for key in ("decoder_init_norm", "jumprelu_bandwidth", "jumprelu_tanh_scale"):
        value = cfg.get(key)
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{key} must be finite and positive")
    for key in ("aux_loss_coefficient", "l0_coefficient", "pre_act_loss_coefficient", "jumprelu_init_threshold"):
        value = cfg.get(key)
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
            raise ValueError(f"{key} must be finite and nonnegative")
    if "topk_threshold_lr" in cfg and not 0 < cfg["topk_threshold_lr"] <= 1:
        raise ValueError("topk_threshold_lr must lie in (0, 1]")
    if "l0_warm_up_steps" in cfg and (type(cfg["l0_warm_up_steps"]) is not int or cfg["l0_warm_up_steps"] < 0):
        raise ValueError("l0_warm_up_steps must be a nonnegative integer")
    if arch == "jumprelu":
        mode = cfg.get("jumprelu_sparsity_loss_mode", "step")
        if mode not in ("step", "tanh", "quadratic"):
            raise ValueError("jumprelu_sparsity_loss_mode must be step, tanh, or quadratic")
        if mode == "quadratic" and not 0 < cfg.get("target_l0", k) <= cfg["d_sae"]:
            raise ValueError("quadratic target_l0 must lie in (0, d_sae]")
    cfg.update(device=str(device), dtype="float32")

    if arch == "sparsemax_attention":
        from sparsemax_attention_sae import SparsemaxAttentionSAE, SparsemaxAttentionSAEConfig
        cfg.pop("normalize_activations", None)
        cfg.setdefault("l0_target", float(k))
        cfg_class, model_class = SparsemaxAttentionSAEConfig, SparsemaxAttentionSAE
    else:
        from sae_lens import (
            BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig,
            JumpReLUTrainingSAE, JumpReLUTrainingSAEConfig,
            MatryoshkaBatchTopKTrainingSAE, MatryoshkaBatchTopKTrainingSAEConfig,
            TopKTrainingSAE, TopKTrainingSAEConfig,
        )
        cfg_class, model_class = {
            "batchtopk": (BatchTopKTrainingSAEConfig, BatchTopKTrainingSAE),
            "topk": (TopKTrainingSAEConfig, TopKTrainingSAE),
            "jumprelu": (JumpReLUTrainingSAEConfig, JumpReLUTrainingSAE),
            "matryoshka": (MatryoshkaBatchTopKTrainingSAEConfig, MatryoshkaBatchTopKTrainingSAE),
        }[arch]
        cfg.setdefault("normalize_activations", "none")
        cfg.setdefault("decoder_init_norm", 0.1)
        if arch != "jumprelu":
            cfg["k"] = int(k) if arch == "topk" else float(k)
            cfg.setdefault("rescale_acts_by_decoder_norm", True)
        elif cfg.get("jumprelu_sparsity_loss_mode") == "quadratic":
            cfg.setdefault("target_l0", float(k))
        if arch == "matryoshka":
            d = cfg["d_sae"]
            cfg.setdefault("matryoshka_widths", sorted({max(1, d // 16), max(1, d // 8), max(1, d // 2), d}))
            widths = cfg["matryoshka_widths"]
            if not isinstance(widths, list) or not widths or any(type(w) is not int or w <= 0 or w > d for w in widths) or any(a >= b for a, b in zip(widths, widths[1:])):
                raise ValueError("matryoshka_widths must be strictly increasing integers in [1, d_sae]")
            if widths[-1] != d:
                cfg["matryoshka_widths"] = widths + [d]
            cfg.setdefault("use_matryoshka_aux_loss", True)
    unknown = set(cfg) - {f.name for f in fields(cfg_class)}
    if unknown:
        raise ValueError(f"Unsupported {arch} options: {', '.join(sorted(unknown))}")
    native_cfg = cfg_class(**cfg)
    resolved = {**copy.deepcopy(cfg), "architecture": arch, "k": k}
    resolved.pop("device", None)
    return native_cfg, model_class, resolved


def resolve_model_config(model_cfg: dict[str, Any], device: str = "cpu") -> dict[str, Any]:
    """Validate and fill adapter defaults without allocating model parameters.

    The returned portable mapping can be passed directly to ``build_sae``. Native
    SAE-Lens defaults not explicitly overridden remain governed by the pinned
    library version. Normalization must be owned by the outer data pipeline.
    """
    return _model_components(model_cfg, device)[2]


def build_sae(model_cfg: dict[str, Any], device: str = "cpu") -> nn.Module:
    """Build a native training SAE from shared and architecture-specific fields.

    Unknown options fail explicitly. ``k`` controls the TopK-family selection;
    for JumpReLU and sparsemax it is a comparison target, with actual sparsity
    controlled by their architecture-specific objectives.
    """
    native_cfg, model_class, resolved = _model_components(model_cfg, device)
    sae = model_class(native_cfg)
    sae.model_config = resolved
    return sae


def training_step(sae: nn.Module, x: torch.Tensor, step: int = 0, dead_mask: torch.Tensor | None = None):
    """Return the native TrainStepOutput (or compatible sparsemax output).

    Warmup is 1-based: step 0 uses 1/warm_up_steps of each target coefficient.
    SAE-Lens MSE sums over input dimensions and averages over activation rows;
    sparsemax retains the reference's per-element MSE loss convention.
    """
    from sae_lens.saes.sae import TrainCoefficientConfig, TrainStepInput
    if step < 0:
        raise ValueError("step must be nonnegative")
    if x.ndim != 2 or x.shape[-1] != sae.cfg.d_in or x.shape[0] == 0:
        raise ValueError(f"Expected nonempty activation rows [N, {sae.cfg.d_in}]")
    x = x.to(device=sae.W_dec.device, dtype=sae.W_dec.dtype)
    if dead_mask is not None:
        if dead_mask.shape != (sae.cfg.d_sae,):
            raise ValueError("dead_mask must have shape [d_sae]")
        dead_mask = dead_mask.to(device=x.device, dtype=torch.bool)
    coefficients = {}
    for name, coefficient in sae.get_coefficients().items():
        if isinstance(coefficient, TrainCoefficientConfig):
            warmup = coefficient.warm_up_steps
            coefficients[name] = coefficient.value * (min(1.0, (step + 1) / warmup) if warmup > 0 else 1.0)
        else:
            coefficients[name] = float(coefficient)
    return sae.training_forward_pass(TrainStepInput(
        sae_in=x, coefficients=coefficients, dead_neuron_mask=dead_mask,
        n_training_steps=step, is_logging_step=False,
    ))


def reconstruct(sae: nn.Module, x: torch.Tensor, width: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return batch-independent inference reconstruction and dense feature acts.

    An optional width zeros all features outside the prefix before native decode,
    preserving the decoder bias and norm scaling for Matryoshka evaluation. This
    function does not change module train/eval mode or disable gradients; callers
    should use ``sae.eval()`` and ``torch.no_grad()`` for evaluation.
    """
    if x.ndim < 2 or x.shape[-1] != sae.cfg.d_in:
        raise ValueError(f"Expected activation rows with final dimension {sae.cfg.d_in}")
    x = x.to(device=sae.W_dec.device, dtype=sae.W_dec.dtype)
    if hasattr(sae, "topk_threshold"):
        hidden = sae.hook_sae_acts_pre(sae.process_sae_in(x) @ sae.W_enc + sae.b_enc)
        if sae.cfg.rescale_acts_by_decoder_norm:
            hidden = hidden * sae.W_dec.norm(dim=-1)
        threshold = sae.topk_threshold.to(hidden.dtype)
        if threshold.item() > 0:
            acts = hidden.relu() * (hidden > threshold)
        else:
            values, indices = hidden.relu().topk(min(sae.cfg.d_sae, math.ceil(sae.cfg.k)), dim=-1)
            acts = torch.zeros_like(hidden).scatter(-1, indices, values)
        acts = sae.hook_sae_acts_post(acts)
    else:
        acts = sae.encode(x)
    if width is not None:
        if type(width) is not int or not 0 < width <= sae.cfg.d_sae:
            raise ValueError("width must be an integer in [1, d_sae]")
        acts = acts * (torch.arange(sae.cfg.d_sae, device=acts.device) < width)
    return sae.decode(acts), acts


@torch.no_grad()
def post_optimizer_step(sae: nn.Module) -> None:
    """Preserve JumpReLU's native nonnegative threshold after the optimizer.

    Decoder parameters are intentionally left in SAE-Lens' native parameterization;
    TopK-family models already include row-norm scaling in encode and decode.
    """
    if hasattr(sae, "threshold") and isinstance(sae.threshold, nn.Parameter):
        sae.threshold.clamp_(min=0.0)


def save_sae(sae: nn.Module, path: str | Path, model_cfg: dict[str, Any] | None = None) -> None:
    """Atomically save native training state and portable model configuration."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(sae.model_config if model_cfg is None else model_cfg)
    payload = {"format_version": 1, "model_config": cfg,
               "sae_state_dict": {k: v.detach().cpu().clone() for k, v in sae.state_dict().items()}}
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_sae(path: str | Path, device: str = "cpu") -> nn.Module:
    """Load ``save_sae`` output using tensor-only deserialization, in eval mode."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported SAE checkpoint format")
    sae = build_sae(payload["model_config"], device=device)
    sae.load_state_dict(payload["sae_state_dict"], strict=True)
    return sae.eval()
