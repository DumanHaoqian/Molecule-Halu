"""Validated, bounded-memory access to separately extracted activation splits."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


def sha256_file(path: Path, block_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


class ActivationStore:
    """Only accepts complete FP16 caches with explicit train/validation manifests.

    Shards stay memory mapped. Hash verification reads files incrementally and is
    enabled by default, including on resume; it never promotes a shard to FP32.
    """

    def __init__(self, cache_dir: str | Path, verify_hashes: bool = True):
        self.root = Path(cache_dir).expanduser().resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        manifest = self.manifest
        expected = {"format_version": 1, "status": "complete", "layer": 26,
                    "block_index": 25, "activation_site": "post_block_residual",
                    "dtype": "float16"}
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"Cache {key} must be {value!r}; got {manifest.get(key)!r}")
        self.d_in = int(manifest.get("d_in", 0))
        if self.d_in <= 0:
            raise ValueError("Cache d_in must be positive")
        self._paths: dict[str, list[Path]] = {}
        seen: set[Path] = set()
        for split in ("train", "validation"):
            spec = manifest.get("splits", {}).get(split)
            if not spec or not spec.get("shards"):
                raise ValueError(f"Cache needs a separate nonempty {split} split")
            rows = 0
            self._paths[split] = []
            for entry in spec["shards"]:
                path = (self.root / entry["path"]).resolve()
                if not path.is_relative_to(self.root) or path in seen:
                    raise ValueError(f"Invalid or overlapping cache shard: {path}")
                seen.add(path)
                digest = entry.get("sha256", "")
                if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError(f"Missing or invalid sha256 checksum: {path}")
                if verify_hashes and sha256_file(path) != digest:
                    raise ValueError(f"Cache shard checksum/hash mismatch: {path}")
                if "metadata_path" in entry or "metadata_sha256" in entry or "meta_sha256" in entry:
                    metadata = (self.root / entry.get("metadata_path", "")).resolve()
                    meta_digest = entry.get("metadata_sha256", entry.get("meta_sha256", ""))
                    if not metadata.is_relative_to(self.root) or not metadata.is_file() or len(meta_digest) != 64:
                        raise ValueError(f"Invalid cache metadata path/checksum: {metadata}")
                    if verify_hashes and sha256_file(metadata) != meta_digest:
                        raise ValueError(f"Cache metadata checksum/hash mismatch: {metadata}")
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                n_rows = int(entry.get("n_rows", 0))
                if n_rows <= 0 or array.shape != (n_rows, self.d_in) or array.dtype != np.float16:
                    raise ValueError(f"Cache shard shape/dtype mismatch: {path}")
                rows += n_rows
                self._paths[split].append(path)
            if rows != spec.get("n_tokens"):
                raise ValueError(f"Cache {split} token count does not match its shards")
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        self.fingerprint = hashlib.sha256(canonical).hexdigest()

    def n_tokens(self, split: str) -> int:
        return int(self.manifest["splits"][split]["n_tokens"])

    def get_shard(self, split: str, index: int) -> np.ndarray:
        return np.load(self._paths[split][index], mmap_mode="r", allow_pickle=False)

    def iter_batches(self, split: str, batch_size: int,
                     max_tokens: int | None = None) -> Iterator[np.ndarray]:
        if batch_size <= 0 or (max_tokens is not None and max_tokens < 0):
            raise ValueError("Invalid batch size or token limit")
        remaining = self.n_tokens(split) if max_tokens is None else min(max_tokens, self.n_tokens(split))
        for index in range(len(self._paths[split])):
            if remaining <= 0:
                break
            shard = self.get_shard(split, index)
            for start in range(0, len(shard), batch_size):
                count = min(batch_size, len(shard) - start, remaining)
                if not count:
                    return
                # Copy only a minibatch: torch never sees a read-only mmap view.
                yield np.array(shard[start:start + count], copy=True)
                remaining -= count


class ShuffledActivationStream:
    """One training epoch, shuffled by shard/chunk and then within each chunk.

    Only one chunk's row permutation and one output batch occupy RAM. Chunk
    descriptors are small; payload remains memory mapped. State stores the exact
    cursor and NumPy RNG state so interrupted runs consume precisely the same rows.
    """

    def __init__(self, store: ActivationStore, seed: int = 0, chunk_rows: int = 8192):
        if chunk_rows <= 0 or seed < 0:
            raise ValueError("chunk_rows must be positive and seed nonnegative")
        self.store, self.seed, self.chunk_rows = store, int(seed), int(chunk_rows)
        self.rng = np.random.default_rng(seed)
        shards = self.rng.permutation(len(store._paths["train"]))
        chunks = [(int(i), start, min(start + chunk_rows, entry["n_rows"]))
                  for i in shards
                  for entry in [store.manifest["splits"]["train"]["shards"][int(i)]]
                  for start in range(0, entry["n_rows"], chunk_rows)]
        self.chunks = [chunks[int(i)] for i in self.rng.permutation(len(chunks))]
        self.chunk_index = self.row_offset = self.tokens_emitted = 0
        self._array = self._permutation = None

    def _ensure_chunk(self):
        if self._array is None and self.chunk_index < len(self.chunks):
            shard, start, end = self.chunks[self.chunk_index]
            self._array = self.store.get_shard("train", shard)[start:end]
            # Independent per-chunk RNG allows reconstruction at any cursor.
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, shard, start]))
            self._permutation = rng.permutation(end - start)

    def next_batch(self, batch_size: int) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        count = min(batch_size, self.store.n_tokens("train") - self.tokens_emitted)
        batch = np.empty((count, self.store.d_in), dtype=np.float16)
        written = 0
        while written < count:
            self._ensure_chunk()
            available = len(self._array) - self.row_offset
            take = min(count - written, available)
            indices = self._permutation[self.row_offset:self.row_offset + take]
            batch[written:written + take] = self._array[indices]
            self.row_offset += take
            written += take
            self.tokens_emitted += take
            if self.row_offset == len(self._array):
                self.chunk_index += 1
                self.row_offset = 0
                self._array = self._permutation = None
        return batch

    def state_dict(self) -> dict:
        return {"format_version": 1, "cache_fingerprint": self.store.fingerprint,
                "seed": self.seed, "chunk_rows": self.chunk_rows,
                "chunk_index": self.chunk_index, "row_offset": self.row_offset,
                "tokens_emitted": self.tokens_emitted,
                "rng_state": copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state: dict):
        for key, expected in (("format_version", 1), ("cache_fingerprint", self.store.fingerprint),
                              ("seed", self.seed), ("chunk_rows", self.chunk_rows)):
            if state.get(key) != expected:
                raise ValueError(f"Activation stream {key} differs on resume")
        index, offset = int(state["chunk_index"]), int(state["row_offset"])
        if not 0 <= index <= len(self.chunks):
            raise ValueError("Invalid activation stream chunk cursor")
        chunk_length = self.chunks[index][2] - self.chunks[index][1] if index < len(self.chunks) else 0
        if offset < 0 or (offset >= chunk_length and not (offset == chunk_length == 0)):
            raise ValueError("Invalid activation stream row cursor")
        emitted = sum(end - start for _, start, end in self.chunks[:index]) + offset
        if emitted != state["tokens_emitted"]:
            raise ValueError("Activation stream token count differs from cursor")
        self.chunk_index, self.row_offset, self.tokens_emitted = index, offset, emitted
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        self._array = self._permutation = None


def fit_normalizer(store: ActivationStore, sample_tokens: int = 262144,
                   center: bool = True, seed: int = 0, batch_size: int = 1024) -> dict:
    """Fit one train-only mean and scalar RMS; never normalize each token separately."""
    count = min(int(sample_tokens), store.n_tokens("train"))
    if count <= 0 or batch_size <= 0:
        raise ValueError("normalization.sample_tokens and batch_size must be positive")
    stream = ShuffledActivationStream(store, seed=seed, chunk_rows=max(batch_size, 8192))
    total = torch.zeros(store.d_in, dtype=torch.float64)
    squares = torch.zeros(store.d_in, dtype=torch.float64)
    consumed = 0
    while consumed < count:
        x = torch.from_numpy(stream.next_batch(min(batch_size, count - consumed))).to(torch.float64)
        if not torch.isfinite(x).all():
            raise ValueError("Nonfinite training activation during normalization")
        total += x.sum(0)
        squares += x.square().sum(0)
        consumed += len(x)
    mean = total / count if center else torch.zeros_like(total)
    variance = (squares / count - mean.square()).mean().clamp_min(0)
    scale = float(variance.sqrt())
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("Degenerate training activation normalization scale")
    return {"mean": mean.float(), "scale": scale, "center": bool(center), "n_tokens": count,
            "data_mean": (total / count).float(),
            "method": "train_global_mean_scalar_rms", "seed": int(seed)}


def normalize(x: torch.Tensor, normalizer: dict) -> torch.Tensor:
    return (x.float() - normalizer["mean"].to(x.device)) / float(normalizer["scale"])


def denormalize(x: torch.Tensor, normalizer: dict) -> torch.Tensor:
    return x * float(normalizer["scale"]) + normalizer["mean"].to(x.device)
