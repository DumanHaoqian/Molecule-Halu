# Generated outputs

This directory intentionally contains no prebuilt records. Outputs from the previous
template-rewrite protocol were removed. Generation emits adjacent matched H/N records;
each step has an explicit `copy`,
`occurrence_patch`, or `derivation_rewrite` contract. Simple claims use occurrence-specific
temporary HALLU markers; incomplete or derived prose may be rewritten completely, but every
affected claim occurrence is still marked separately as `node_id.NN`.
Old-claim, arithmetic, missing/duplicate-marker, and exact-FORMAL validation all run before
local span construction. Only marked claim values produce spans; a derivation rewrite never
labels the complete body. Poe returns natural-language bodies only; the Step header and
modified FORMAL are always assembled locally.

H and N share `pair_id`. H owns non-empty `hallucination_spans`; N has no hallucination
spans and instead owns one truth-valued `control_span` per H span. For steps marked
`byte_identical`, replacing H span contents with the corresponding controls reconstructs
the N text exactly. A truth swap that fails FORMAL, residual-value, or arithmetic checks is
regenerated with Poe and explicitly released as `pair_alignment: regenerated`. This weaker
alignment class must be reported separately in downstream experiments. Every paired
occurrence also records whether its two values have the same character length.

After exporting `POE_API_KEY`, generate records with:

```bash
molhallulens-generate --output GeneratedDataset/example.jsonl

# Optional bounded smoke run.
molhallulens-generate --max-origins 3 --output GeneratedDataset/example.jsonl
```

Validated, secret-free Poe responses are stored in `.poe_text_cache/` so interrupted
builds can resume without repeating completed requests.
