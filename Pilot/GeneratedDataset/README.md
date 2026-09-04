# Generated outputs

This directory intentionally contains no prebuilt records. Outputs from the previous
template-rewrite protocol were removed. Each step now has an explicit `copy`,
`occurrence_patch`, or `derivation_rewrite` contract. Simple claims use occurrence-specific
temporary HALLU markers; incomplete or derived prose may be rewritten completely, but every
affected claim occurrence is still marked separately as `node_id.NN`.
Old-claim, arithmetic, missing/duplicate-marker, and exact-FORMAL validation all run before
local span construction. Only marked claim values produce spans; a derivation rewrite never
labels the complete body. Poe returns natural-language bodies only; the Step header and
modified FORMAL are always assembled locally.

After exporting `POE_API_KEY`, generate records with:

```bash
molhallulens-generate --output GeneratedDataset/example.jsonl
```

Validated, secret-free Poe responses are stored in `.poe_text_cache/` so interrupted
builds can resume without repeating completed requests.
