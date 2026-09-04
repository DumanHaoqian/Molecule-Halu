# Generated outputs

This directory intentionally contains no prebuilt records. The previous rule-rendered
JSONL files were removed because final `step_text` must now pass through the Poe agent.

After exporting `POE_API_KEY`, generate records with:

```bash
molhallulens-generate --output GeneratedDataset/hallucinations.jsonl
```

Validated, secret-free Poe responses are stored in `.poe_text_cache/` so interrupted
builds can resume without repeating completed requests.
