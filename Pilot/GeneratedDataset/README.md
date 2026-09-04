# Generated outputs

This directory intentionally contains no prebuilt records. Outputs from the previous
template-rewrite protocol were removed: Poe must now minimally edit each original
`step_text` from its modified `formal_ab`, and occurrence-specific temporary HALLU
markers must pass missing/duplicate/value validation before local span construction.

After exporting `POE_API_KEY`, generate records with:

```bash
molhallulens-generate --output GeneratedDataset/example.jsonl
```

Validated, secret-free Poe responses are stored in `.poe_text_cache/` so interrupted
builds can resume without repeating completed requests.
