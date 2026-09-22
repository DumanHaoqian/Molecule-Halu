"""Small CPU-only integration tests; no production corpus or model is loaded."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_cache(root: Path, d_in=8):
    rng = np.random.default_rng(41)
    manifest = {"format_version": 1, "status": "complete", "layer": 26,
                "block_index": 25, "activation_site": "post_block_residual",
                "d_in": d_in, "dtype": "float16", "splits": {}}
    for split, counts in (("train", (29, 31, 37)), ("validation", (23,))):
        shards = []
        for i, n in enumerate(counts):
            path = root / split / f"{i:06d}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            # Held-out distribution intentionally shifted to detect normalization leakage.
            values = rng.normal(size=(n, d_in)) + (0 if split == "train" else 100)
            np.save(path, values.astype(np.float16))
            shards.append({"path": str(path.relative_to(root)), "n_rows": n,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        manifest["splits"][split] = {"n_tokens": sum(counts), "shards": shards}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def tiny_config(cache, output):
    return {"cache_dir": str(cache), "output_dir": str(output), "seed": 42,
            "device": "cpu", "sae": {"architecture": "batchtopk", "d_in": 8,
            "d_sae": 24, "k": 3}, "training": {"batch_size": 11,
            "max_tokens": 88, "lr": 0.002, "warmup_steps": 2,
            "chunk_rows": 17, "checkpoint_every": 2, "log_every": 1,
            "validate_every": 2}, "normalization": {"sample_tokens": 97,
            "center": True}, "evaluation": {"batch_size": 7, "max_tokens": 23},
            "wandb": {"mode": "disabled", "project": "sae-cpu-test"}}


def test_cache_integrity_and_training_only_normalization(tmp_path):
    from activation_store import ActivationStore, fit_normalizer
    root = make_cache(tmp_path / "cache")
    store = ActivationStore(root)
    norm = fit_normalizer(store, sample_tokens=97, center=True, seed=7, batch_size=13)
    all_train = np.concatenate([np.load(root / entry["path"]).astype(np.float32)
        for entry in store.manifest["splits"]["train"]["shards"]])
    np.testing.assert_allclose(norm["mean"].numpy(), all_train.mean(0), atol=1e-6)
    assert abs(norm["scale"] - np.sqrt(np.square(all_train-all_train.mean(0)).mean())) < 1e-6
    shard = root / "train/000000.npy"
    with shard.open("r+b") as fh:
        fh.seek(-2, 2)
        fh.write(b"\x00\x00")
    with pytest.raises(ValueError, match="hash|checksum"):
        ActivationStore(root)


def test_stream_resume_preserves_order_and_tail(tmp_path):
    from activation_store import ActivationStore, ShuffledActivationStream
    store = ActivationStore(make_cache(tmp_path / "cache"))
    stream = ShuffledActivationStream(store, seed=15, chunk_rows=17)
    first = stream.next_batch(19)
    saved = stream.state_dict()
    expected = [stream.next_batch(19) for _ in range(5)]
    resumed = ShuffledActivationStream(store, seed=15, chunk_rows=17)
    resumed.load_state_dict(saved)
    actual = [resumed.next_batch(19) for _ in range(5)]
    assert first.shape == (19, 8)
    assert sum(map(len, expected)) + len(first) == 97
    for a, b in zip(expected, actual):
        np.testing.assert_array_equal(a, b)


def test_cpu_training_resume_is_equivalent_and_refuses_overwrite(tmp_path):
    from train_sae import run_training
    torch.set_num_threads(2)
    cache = make_cache(tmp_path / "cache")
    whole = run_training(tiny_config(cache, tmp_path / "whole"))
    config = tiny_config(cache, tmp_path / "resumed")
    partial = run_training(config, max_steps=3)
    assert partial["step"] == 3
    result = run_training(config, resume=tmp_path / "resumed/last.pt")
    assert result["step"] == whole["step"] == 8
    a = torch.load(tmp_path / "whole/last.pt", weights_only=False, map_location="cpu")
    b = torch.load(tmp_path / "resumed/last.pt", weights_only=False, map_location="cpu")
    assert a["tokens_seen"] == b["tokens_seen"] == 88
    for key in a["sae_state_dict"]:
        torch.testing.assert_close(a["sae_state_dict"][key], b["sae_state_dict"][key], rtol=0, atol=0)
    assert (tmp_path / "resumed/last.pt.receipt.json").exists()
    assert (tmp_path / "resumed/best.pt").exists()
    assert b["normalizer"]["n_tokens"] == 97
    assert all(abs(v) < 1 for v in b["normalizer"]["mean"])
    with pytest.raises(FileExistsError):
        run_training(config)


def test_resume_rejects_configuration_change(tmp_path):
    from train_sae import run_training
    cfg = tiny_config(make_cache(tmp_path / "cache"), tmp_path / "run")
    run_training(cfg, max_steps=2)
    cfg["training"]["lr"] = 0.1
    with pytest.raises(ValueError, match="config"):
        run_training(cfg, resume=tmp_path / "run/last.pt")


@pytest.mark.skipif(importlib.util.find_spec("wandb") is None, reason="optional W&B SDK unavailable")
def test_wandb_offline_cpu_smoke(tmp_path):
    from train_sae import run_training
    cfg = tiny_config(make_cache(tmp_path / "cache"), tmp_path / "run")
    cfg["wandb"]["mode"] = "offline"
    result = run_training(cfg, max_steps=1)
    assert result["step"] == 1
    files = list((tmp_path / "run/wandb").glob("offline-run-*/run-*.wandb"))
    assert files
    from wandb.sdk.internal.datastore import DataStore
    from wandb.proto import wandb_internal_pb2
    reader = DataStore()
    reader.open_for_scan(str(files[0]))
    history_keys = set()
    while (data := reader.scan_data()) is not None:
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.HasField("history"):
            for item in record.history.item:
                history_keys.add(item.key or "/".join(item.nested_key))
    assert "train/loss" in history_keys
    assert "val/nmse" in history_keys


def test_validate_config_without_cache_or_model_allocation(tmp_path):
    from train_sae import inspect_config
    cfg = tiny_config(tmp_path / "not-extracted", tmp_path / "run")
    report = inspect_config(cfg)
    assert report["estimated_parameters"] == 2 * 8 * 24 + 8 + 24
    assert report["cache_exists"] is False
    assert not (tmp_path / "run").exists()


def test_metadata_hash_and_incomplete_cache_are_rejected(tmp_path):
    from activation_store import ActivationStore
    root = make_cache(tmp_path / "cache")
    manifest = json.loads((root / "manifest.json").read_text())
    metadata = root / "train/000000.metadata.jsonl"
    metadata.write_text('{"row_start":0,"row_end":29}\n')
    manifest["splits"]["train"]["shards"][0].update(
        metadata_path="train/000000.metadata.jsonl", metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest())
    (root / "manifest.json").write_text(json.dumps(manifest))
    ActivationStore(root)
    metadata.write_text("changed")
    with pytest.raises(ValueError, match="metadata checksum"):
        ActivationStore(root)
    manifest["status"] = "extracting"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="status"):
        ActivationStore(root)


def test_dry_run_rejects_invalid_architecture_without_writing(tmp_path):
    from train_sae import run_training
    cfg = tiny_config(make_cache(tmp_path / "cache"), tmp_path / "run")
    cfg["sae"]["architecture"] = "typo"
    with pytest.raises(ValueError, match="architecture"):
        run_training(cfg, dry_run=True)
    assert not (tmp_path / "run").exists()


def test_checkpoint_corruption_is_rejected_before_loading(tmp_path):
    from train_sae import run_training, load_checkpoint
    cfg = tiny_config(make_cache(tmp_path / "cache"), tmp_path / "run")
    run_training(cfg, max_steps=1)
    path = tmp_path / "run/last.pt"
    with path.open("r+b") as handle:
        handle.seek(-2, 2)
        handle.write(b"XX")
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(path)


@pytest.mark.parametrize("architecture", ["topk", "jumprelu", "matryoshka", "sparsemax_attention"])
def test_all_architectures_train_on_tiny_cpu_cache(tmp_path, architecture):
    from train_sae import run_training
    cfg = tiny_config(make_cache(tmp_path / "cache"), tmp_path / "run")
    cfg["sae"]["architecture"] = architecture
    if architecture == "sparsemax_attention":
        cfg["sae"].update(key_dim=4, preselect_k=12)
    report = run_training(cfg, max_steps=2)
    assert report["step"] == 2
    assert np.isfinite(report["validation"]["nmse"])
