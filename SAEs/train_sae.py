"""Train an SAE from a verified layer-26 cache, with exact resumable row order.

No language model is loaded here. Run ``--dry-run`` to verify a cache/configuration
without constructing an SAE, initializing CUDA, starting W&B, or writing a run.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch
import yaml

from activation_store import (ActivationStore, ShuffledActivationStream, denormalize,
                              fit_normalizer, normalize, sha256_file)
from metrics import ReconstructionMetrics


DEFAULTS = {
    "cache_dir": "activations/layer26/pilot_10m", "output_dir": "runs/batchtopk",
    "seed": 20260921, "device": "cuda:4", "verify_cache_hashes": True,
    "sae": {"architecture": "batchtopk", "d_in": 5120, "d_sae": 16384, "k": 64},
    "training": {"batch_size": 1024, "max_tokens": 10000474, "lr": 3e-4,
                 "warmup_steps": 100, "lr_schedule": "cosine", "min_lr_ratio": 0.1,
                 "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0,
                 "grad_clip": 1.0, "chunk_rows": 8192, "dead_feature_tokens": 262144,
                 "log_every": 10, "checkpoint_every": 500, "validate_every": 500},
    "normalization": {"sample_tokens": 262144, "center": True},
    "evaluation": {"batch_size": 1024, "max_tokens": 100000},
    "wandb": {"mode": "online", "project": "chemdfm-layer26-sae", "entity": None,
              "name": None, "tags": ["layer26", "train-only-normalization"]},
}


def _merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def resolve_config(config: dict, base_dir: str | Path | None = None) -> dict:
    cfg = _merge(DEFAULTS, config)
    base = Path(base_dir or Path.cwd())
    for key in ("cache_dir", "output_dir"):
        value = Path(cfg[key]).expanduser()
        cfg[key] = str((base / value).resolve() if not value.is_absolute() else value.resolve())
    if cfg["wandb"]["mode"] not in {"online", "offline", "disabled"}:
        raise ValueError("wandb.mode must be online, offline, or disabled")
    if int(cfg["seed"]) < 0:
        raise ValueError("seed must be nonnegative")
    for section, fields in (("training", ["batch_size", "max_tokens", "chunk_rows", "log_every", "checkpoint_every", "validate_every", "dead_feature_tokens"]),
                             ("normalization", ["sample_tokens"]), ("evaluation", ["batch_size", "max_tokens"]),
                             ("sae", ["d_in", "d_sae"])):
        for key in fields:
            if int(cfg[section][key]) <= 0:
                raise ValueError(f"{section}.{key} must be positive")
    train = cfg["training"]
    if train["lr"] <= 0 or train["eps"] <= 0 or train["grad_clip"] <= 0 or train["warmup_steps"] < 0:
        raise ValueError("Invalid optimizer or warmup configuration")
    if train["lr_schedule"] not in {"constant", "cosine"} or not 0 <= train["min_lr_ratio"] <= 1:
        raise ValueError("Invalid learning-rate schedule")
    return cfg


def load_config(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Training YAML must contain a mapping")
    return resolve_config(config, path.parent)


def inspect_config(config: dict) -> dict:
    """Check model/configuration and estimate memory without constructing an SAE."""
    from sae_models import resolve_model_config
    cfg = resolve_config(config)
    model_cfg = resolve_model_config(cfg["sae"], device="cpu")
    d_in, d_sae = int(model_cfg["d_in"]), int(model_cfg["d_sae"])
    if model_cfg["architecture"] == "sparsemax_attention":
        key_dim = model_cfg.get("key_dim") or min(1024, d_in)
        parameters = d_in * d_sae + d_in * key_dim + d_sae * key_dim + 2 * d_sae + d_in + 1
    else:
        parameters = 2 * d_in * d_sae + d_in + d_sae
        if model_cfg["architecture"] == "jumprelu":
            parameters += d_sae
    cache_exists = (Path(cfg["cache_dir"]) / "manifest.json").exists()
    report = {"config": cfg, "resolved_model_config": model_cfg,
              "estimated_parameters": parameters,
              "estimated_parameter_gib": parameters * 4 / 2 ** 30,
              "estimated_training_state_gib": parameters * 16 / 2 ** 30,
              "memory_estimate_note": "FP32 parameters, gradients and two Adam moments only; activations/workspaces require additional memory",
              "cache_exists": cache_exists, "cache_verified": False}
    if cache_exists:
        store = ActivationStore(cfg["cache_dir"], verify_hashes=bool(cfg["verify_cache_hashes"]))
        if store.d_in != d_in:
            raise ValueError("SAE d_in differs from cache activation width")
        report.update(cache_verified=bool(cfg["verify_cache_hashes"]), cache_fingerprint=store.fingerprint,
                      train_tokens=store.n_tokens("train"), validation_tokens=store.n_tokens("validation"))
    return report


def config_fingerprint(cfg: dict) -> str:
    # Device, output location, and telemetry may change when resuming elsewhere.
    fixed = {key: value for key, value in cfg.items() if key not in {"device", "output_dir", "wandb"}}
    return hashlib.sha256(json.dumps(fixed, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic_json(path: Path, payload: dict):
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_checkpoint(path: str | Path, payload: dict):
    """Atomic checkpoint plus a receipt verified before every resume."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        digest = sha256_file(Path(tmp))
        os.replace(tmp, path)
        _atomic_json(Path(str(path) + ".receipt.json"), {
            "format_version": 1, "path": path.name, "sha256": digest,
            "step": payload["step"], "tokens_seen": payload["tokens_seen"],
            "cache_fingerprint": payload["cache_fingerprint"],
            "config_fingerprint": payload["config_fingerprint"],
        })
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_checkpoint(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    receipt_path = Path(str(path) + ".receipt.json")
    if not receipt_path.exists():
        raise ValueError(f"Checkpoint receipt is missing: {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    if sha256_file(path) != receipt["sha256"]:
        raise ValueError("Checkpoint checksum/hash does not match its receipt")
    # Only load trusted local checkpoints: optimizer/RNG state includes Python objects.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    for key in ("step", "tokens_seen", "cache_fingerprint", "config_fingerprint"):
        if payload.get(key) != receipt.get(key):
            raise ValueError(f"Checkpoint {key} does not match its receipt")
    return payload


def _rng_state(device: torch.device) -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def _restore_rng(state: dict, device: torch.device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)


def learning_rate(step: int, total_steps: int, train: dict) -> float:
    warmup = int(train["warmup_steps"])
    if step < warmup:
        multiplier = (step + 1) / max(warmup, 1)
    elif train["lr_schedule"] == "cosine":
        progress = (step - warmup) / max(total_steps - warmup - 1, 1)
        multiplier = train["min_lr_ratio"] + (1 - train["min_lr_ratio"]) * (1 + math.cos(math.pi * min(progress, 1))) / 2
    else:
        multiplier = 1.0
    return float(train["lr"]) * multiplier


def evaluate(sae, store: ActivationStore, normalizer: dict, cfg: dict, device: torch.device) -> dict:
    from sae_models import reconstruct
    metrics = ReconstructionMetrics()
    was_training = sae.training
    rng = _rng_state(device)
    try:
        sae.eval()
        with torch.inference_mode():
            for array in store.iter_batches("validation", int(cfg["batch_size"]), int(cfg["max_tokens"])):
                x = torch.from_numpy(array).to(device=device, dtype=torch.float32)
                recon, z = reconstruct(sae, normalize(x, normalizer))
                if not torch.isfinite(recon).all() or not torch.isfinite(z).all():
                    raise FloatingPointError("Nonfinite validation reconstruction or features")
                metrics.update(x, denormalize(recon, normalizer), z)
        return metrics.compute()
    finally:
        sae.train(was_training)
        _restore_rng(rng, device)


def _versions() -> dict:
    result = {"torch": torch.__version__, "numpy": np.__version__}
    for package in ("sae-lens", "wandb", "transformers"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _code_hashes() -> dict:
    root = Path(__file__).resolve().parent
    return {name: sha256_file(root / name) for name in
            ("train_sae.py", "activation_store.py", "sae_models.py", "sparsemax_attention_sae.py", "metrics.py")}


def run_training(config: dict, *, resume: str | Path | None = None,
                 max_steps: int | None = None, dry_run: bool = False) -> dict:
    cfg = resolve_config(config)
    from sae_models import resolve_model_config
    resolve_model_config(cfg["sae"], device="cpu")
    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be nonnegative")
    store = ActivationStore(cfg["cache_dir"], verify_hashes=bool(cfg["verify_cache_hashes"]))
    if store.d_in != int(cfg["sae"]["d_in"]):
        raise ValueError("SAE d_in differs from cache activation width")
    train = cfg["training"]
    target_tokens = min(int(train["max_tokens"]), store.n_tokens("train"))
    total_steps = math.ceil(target_tokens / int(train["batch_size"]))
    summary = {"cache_fingerprint": store.fingerprint, "train_tokens": store.n_tokens("train"),
               "validation_tokens": store.n_tokens("validation"), "target_tokens": target_tokens,
               "total_steps": total_steps, "d_in": store.d_in, "config": cfg,
               "requested_max_tokens": int(train["max_tokens"]),
               "budget_limited_by_single_pass": int(train["max_tokens"]) > store.n_tokens("train")}
    if dry_run:
        return summary

    output = Path(cfg["output_dir"])
    previous = load_checkpoint(resume) if resume else None
    if previous:
        if previous["cache_fingerprint"] != store.fingerprint:
            raise ValueError("Resume cache fingerprint does not match checkpoint")
        if previous["config_fingerprint"] != config_fingerprint(cfg):
            raise ValueError("Resume config fingerprint differs; architecture, data, normalization and optimizer settings must match")
        original_output = Path(previous["config"]["output_dir"]).resolve()
        if output.exists() and any(output.iterdir()) and output.resolve() != original_output:
            raise FileExistsError("Refusing to resume into another existing run directory")
        latest_path = output / "last.pt"
        if latest_path.exists() and load_checkpoint(latest_path)["step"] > previous["step"]:
            raise ValueError("Resume checkpoint is older than this run's last checkpoint; choose an empty output directory")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {output}. Use --resume or a new output_dir")
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg["device"])
    if device.type == "cuda":
        if device.index is None:
            raise ValueError("Select an explicit CUDA device, e.g. cuda:4")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; use --device cpu for synthetic tests")
        torch.cuda.set_device(device)
    random.seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]) % 2 ** 32)
    torch.manual_seed(int(cfg["seed"]))
    from sae_models import build_sae, post_optimizer_step, training_step
    sae = build_sae(cfg["sae"], device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=float(train["lr"]),
                                 betas=tuple(train["betas"]), eps=float(train["eps"]),
                                 weight_decay=float(train["weight_decay"]))
    stream = ShuffledActivationStream(store, seed=int(cfg["seed"]), chunk_rows=int(train["chunk_rows"]))
    normalizer = previous["normalizer"] if previous else fit_normalizer(
        store, sample_tokens=int(cfg["normalization"]["sample_tokens"]),
        center=bool(cfg["normalization"]["center"]), seed=int(cfg["seed"]),
        batch_size=int(train["batch_size"]))
    step = tokens_seen = 0
    best_nmse = float("inf")
    last_fired = torch.zeros(int(cfg["sae"]["d_sae"]), dtype=torch.int64, device=device)
    firing_counts = torch.zeros_like(last_fired)
    if previous:
        sae.load_state_dict(previous["sae_state_dict"], strict=True)
        optimizer.load_state_dict(previous["optimizer_state_dict"])
        stream.load_state_dict(previous["stream_state"])
        step, tokens_seen = int(previous["step"]), int(previous["tokens_seen"])
        if tokens_seen != stream.tokens_emitted:
            raise ValueError("Checkpoint progress differs from activation stream cursor")
        best_nmse = float(previous["best_nmse"])
        last_fired.copy_(previous["last_fired_tokens"].to(device))
        firing_counts.copy_(previous["firing_counts"].to(device))
    normalizer = dict(normalizer, mean=normalizer["mean"].to(device))
    versions = _versions()
    code_hashes = _code_hashes()
    run = None
    if cfg["wandb"]["mode"] != "disabled":
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("Install the pinned W&B SDK or set --wandb-mode disabled") from error
        run = wandb.init(project=cfg["wandb"]["project"], entity=cfg["wandb"]["entity"],
                         name=cfg["wandb"]["name"] or output.name, mode=cfg["wandb"]["mode"],
                         dir=str(output), tags=cfg["wandb"]["tags"],
                         config={**cfg, "cache_fingerprint": store.fingerprint, "versions": versions, "code_hashes": code_hashes},
                         id=previous.get("wandb_run_id") if previous else None,
                         resume="allow" if previous and cfg["wandb"]["mode"] == "online" else None)
        run.define_metric("step")
        run.define_metric("train/*", step_metric="step")
        run.define_metric("val/*", step_metric="step")
    # Constructing the model and telemetry may consume randomness. Restore last.
    if previous:
        _restore_rng(previous["rng_state"], device)
    _atomic_json(output / "config.json", {**cfg, "cache_fingerprint": store.fingerprint, "versions": versions, "code_hashes": code_hashes})
    metrics_handle = (output / "metrics.jsonl").open("a")
    started = time.monotonic()
    start_tokens = tokens_seen
    latest_validation = previous.get("validation", {}) if previous else {}
    validation_step = int(previous.get("validation_step", 0)) if previous else 0
    exit_code = 1

    def log(values: dict):
        record = {"step": step, "tokens_seen": tokens_seen, **values}
        metrics_handle.write(json.dumps(record, allow_nan=False) + "\n")
        metrics_handle.flush()
        if run:
            # The custom optimizer-step axis permits separate train/val records.
            # Let W&B advance its internal history step to avoid duplicate writes.
            run.log(record)

    def payload() -> dict:
        return {"format_version": 1, "sae_state_dict": sae.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "step": step,
                "tokens_seen": tokens_seen, "best_nmse": best_nmse,
                "normalizer": dict(normalizer, mean=normalizer["mean"].detach().cpu()),
                "stream_state": stream.state_dict(), "rng_state": _rng_state(device),
                "last_fired_tokens": last_fired.detach().cpu(),
                "firing_counts": firing_counts.detach().cpu(),
                "config": cfg, "config_fingerprint": config_fingerprint(cfg),
                "cache_fingerprint": store.fingerprint, "versions": versions,
                "cache_manifest": store.manifest, "resolved_model_config": sae.model_config,
                "code_hashes": code_hashes,
                "validation": latest_validation, "validation_step": validation_step,
                "wandb_run_id": run.id if run else None,
                "normalization_contract": "x_norm=(x-mean)/scale; x_recon=recon_norm*scale+mean"}

    try:
        sae.train()
        while tokens_seen < target_tokens and (max_steps is None or step < max_steps):
            array = stream.next_batch(min(int(train["batch_size"]), target_tokens - tokens_seen))
            x = normalize(torch.from_numpy(array).to(device=device, dtype=torch.float32), normalizer)
            if not torch.isfinite(x).all():
                raise FloatingPointError("Nonfinite training activations")
            lr = learning_rate(step, total_steps, train)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            dead_mask = tokens_seen - last_fired >= int(train["dead_feature_tokens"])
            result = training_step(sae, x, step=step, dead_mask=dead_mask)
            if not torch.isfinite(result.loss):
                raise FloatingPointError("Nonfinite training loss")
            result.loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(sae.parameters(), float(train["grad_clip"]), error_if_nonfinite=True)
            optimizer.step()
            post_optimizer_step(sae)
            tokens_seen += len(x)
            step += 1
            with torch.no_grad():
                fired = (result.feature_acts.detach() != 0).sum(0)
                firing_counts += fired
                last_fired[fired > 0] = tokens_seen
            if step % int(train["log_every"]) == 0 or step == 1:
                with torch.no_grad():
                    mse = float((result.sae_out.detach() - x).square().mean())
                    values = {"train/loss": float(result.loss.detach()), "train/mse": mse,
                              "train/mse_raw": mse * normalizer["scale"] ** 2,
                              "train/l0": float((result.feature_acts.detach() != 0).sum(-1).float().mean()),
                              "train/lr": lr, "train/grad_norm": float(grad_norm),
                              "train/parameter_norm": math.sqrt(sum(float(p.detach().float().square().sum()) for p in sae.parameters())),
                              "train/dead_feature_fraction": float(dead_mask.float().mean()),
                              "train/never_fired_fraction": float((firing_counts == 0).float().mean()),
                              "train/tokens_per_second": (tokens_seen - start_tokens) / max(time.monotonic() - started, 1e-9)}
                    for key, value in result.losses.items():
                        values[f"train/components/{key}"] = float(value.detach()) if torch.is_tensor(value) else float(value)
                log(values)
            if step % int(train["validate_every"]) == 0:
                latest_validation = evaluate(sae, store, normalizer, cfg["evaluation"], device)
                validation_step = step
                log({f"val/{key}": value for key, value in latest_validation.items()})
                if latest_validation["nmse"] < best_nmse:
                    best_nmse = latest_validation["nmse"]
                    save_checkpoint(output / "best.pt", payload())
            if step % int(train["checkpoint_every"]) == 0:
                save_checkpoint(output / "last.pt", payload())
        # Assess the precise saved weights, including an intentional short run.
        if validation_step != step or not latest_validation:
            latest_validation = evaluate(sae, store, normalizer, cfg["evaluation"], device)
            validation_step = step
            log({f"val/{key}": value for key, value in latest_validation.items()})
        if latest_validation["nmse"] < best_nmse:
            best_nmse = latest_validation["nmse"]
            save_checkpoint(output / "best.pt", payload())
        save_checkpoint(output / "last.pt", payload())
        result_summary = {"step": step, "tokens_seen": tokens_seen, "target_tokens": target_tokens,
                          "complete": tokens_seen >= target_tokens, "validation": latest_validation,
                          "best_nmse": best_nmse, "checkpoint": str(output / "last.pt"),
                          "cache_fingerprint": store.fingerprint, "code_hashes": code_hashes,
                          "requested_max_tokens": int(train["max_tokens"]),
                          "budget_limited_by_single_pass": int(train["max_tokens"]) > store.n_tokens("train")}
        _atomic_json(output / "summary.json", result_summary)
        if run:
            run.summary.update(result_summary)
        exit_code = 0
        return result_summary
    finally:
        metrics_handle.close()
        if run:
            run.finish(exit_code=exit_code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", help="Explicit torch device; no GPU allocation in --dry-run")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    parser.add_argument("--resume", type=Path, help="Trusted checkpoint with a matching .receipt.json")
    parser.add_argument("--max-steps", type=int, help="Stop at this absolute optimizer step; resume keeps the original LR schedule")
    parser.add_argument("--dry-run", action="store_true", help="Verify manifests, hashes and configuration only")
    parser.add_argument("--validate-config", action="store_true", help="Validate model options and estimate memory without allocating SAE parameters; verify cache if present")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    if args.wandb_mode:
        cfg["wandb"]["mode"] = args.wandb_mode
    result = inspect_config(cfg) if args.validate_config else run_training(cfg, resume=args.resume, max_steps=args.max_steps, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
