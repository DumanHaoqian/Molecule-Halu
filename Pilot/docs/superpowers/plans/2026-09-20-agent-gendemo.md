# Agent dataset replay implementation plan

**Goal:** Let the existing gendemo.py entry point display the saved agent H against original N without rewriting experimental data or calling models.

**Architecture:** Dispatch agent pairs to a separate read-only viewer using adjacent origins/<origin>/accepted.json and input.json. Keep the legacy viewer as-is. Default dataset selection remains explicit through --records; the old default is preserved.

**Tech stack:** Existing Python/Gradio HTML comparison components and saved Unicode annotations/token records; no new dependencies, model downloads, or API calls.

- [x] Write tests for actual v17 original N versus paired N, exact sidecar binding, corrupted annotations, missing sidecars, safe HTML and navigation callbacks.
- [x] Implement agent record validation and read-only original-N/H, paired-N/H, exact error highlights, dependency tables and saved per-model token coverage.
- [x] Dispatch gendemo.build_demo for minimal agent pair records, retaining old event schema behavior.
- [x] Test all four cases and UI callbacks, run focused legacy checks and a local HTTP/API smoke test; document launch command.

Text diffs are not labels. Original N comes from frozen input.row.N_visible, with any historical complete-answer clause already removed. Paired N comes from the actual experiment pair. Root and propagated errors come only from accepted annotations; source-side edited substrings are patch context, not a one-to-one truth token alignment. Show diagnostic-only and four-origin scope explicitly. Keep GT and full wrong products out of the visible comparison. Never compute a generic tokenizer count and label it as the model's saved count. Do not mutate, resave or regenerate datasets.

## Verification

21 focused legacy/Agent viewer tests passed. The actual localhost Gradio service returned HTTP 200; all four select endpoints, previous/next navigation, saved token tables and root/propagation HTML were verified through the real client. The client omits button-update outputs, unlike direct callback tests. No browser screenshot verification was available.

A local proxy problem was reproduced: Gradio prints the serving URL before making its startup-events self-request; inherited proxy settings blocked that request, leaving the queue unstarted. The CLI now preserves existing NO_PROXY/no_proxy entries and adds loopback plus its bind host before launch. The actual queued requests passed after this correction.

Viewer is served on 127.0.0.1:7838, PID recorded in /tmp/gendemo-v17.pid; log is /tmp/gendemo-v17.log. Original dataset artifacts remain unchanged.
