# ChemDFM-R activation storage

T051 extracts full-token `bfloat16` `resid_post` tensors from
`model.model.layers[26]`. The token and label axes are identical; the alignment
is `post_token_h_t` with `label_shift = 0`.

The tensor payload is intentionally server-only because the complete release is
approximately 18.7 GB before small container overhead. On the approved 5090 it
is stored under:

```text
/home/haoqian/Data/Molecule-Halu/Pilot/HallucinationDataset/
  activations/chemdfm_r/layer_26/
```

Git contains the JSON manifest and one JSON sidecar per tensor shard. Each
sidecar records the ordered record IDs, per-record token counts, row offsets,
tensor shape, layer, alignment, dtype, byte size, and relative tensor path. The
binary `.pt` files and incomplete `.pt.partial` files are excluded by exact
repository ignore rules.

No digest is computed or verified in this T051 workflow. Checkpoint provenance
is recorded as the approved local path plus its validated Qwen2 configuration;
that path and configuration are an operator-approved provenance claim, not
proof of byte-exact checkpoint or tokenizer identity. Reproduction therefore
requires provisioning the same approved local snapshot through the controlled
server environment.

To resume an interrupted extraction, rerun the same T051 command with the same
release root and shard size. A shard is reused only when both its tensor and
sidecar exist, the sidecar matches the deterministic record plan, and strict
resume validation loads the tensor on CPU and checks its ordered IDs, token
counts, row offsets, `bfloat16` dtype, `[tokens, 5120]` shape, layer, and exact
post-token alignment. Partial or unpublished work is regenerated without
changing completed shards.

All `tensor_path` and `metadata_path` entries in the activation manifest are
relative to the manifest's `layer_26` directory and are resolved with traversal
and root-escape checks. The manifest records the approved server directory only
as environment provenance; Git does not contain the tensor payloads.

Each tensor file stores:

```text
format_version
activation_alignment
label_shift
layer_index
hook_path
record_ids
token_counts
row_offsets
activations  # [sum(token_counts), 5120], bfloat16
```
