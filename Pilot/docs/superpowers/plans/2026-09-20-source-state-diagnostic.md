# Source-state CoT diagnostic (v13)

## Evidence and hypothesis

The v12 larger structural intervention passed construction checks but failed the
predeclared two-model development gate. ChemDFM A/B were 11/14 and 12/14; Chem-R
A/B were 9/14 and 7/14. The two harmful Chem-R flips affected stereochemical
specification, without matching the full H plan or an executable nonempty root
subset. These observations motivate a different test; they do not establish a
new mechanism or predict success.

The new hypothesis is that an incorrect molecular source representation inside
CoT can affect the answer when the subsequent edit is internally coherent.
Only the supplied CoT changes. Original question, instruction, source strings,
model system/prefix placement, and evaluation settings remain fixed.

## Construction and interpretation

1. Freeze the benchmark reference edit I on the actual source S and verify its
   exact product against GT. Render N independently of H candidates.
2. Construct a mistaken perceived source S′ by omitting one or more remote
   branches, including a ring. Preserve spectators, at least half of the active
   component, all reference-edit atoms and their immediate neighbors. Require
   connected, closed-shell, stereo-safe execution through the existing tools.
3. Apply exactly I to S′, obtaining P′. Also execute the composed omission and I
   directly on S and require the same complete product, including stereo. This
   second path supports an auditable composition check and construction metrics.
4. Both N and H start with a source-representation SMILES and its formula, then
   use the same typed edit renderer without the optional local-product window.
   H source counts and subsequent product
   counts describe S′ and P′ consistently. The original instruction remains
   outside CoT and unchanged. Neither complete GT nor P′ is rendered.
5. `source_graph` is one root. Changed formula/count claims
   inherit computational provenance from it; canceled or unchanged values are
   not labeled erroneous. A whole false source-SMILES value is a semantic claim,
   not a declaration that every constituent atom or character is false.
6. A bounded gpt-5.4-mini selector judges source-reconstruction plausibility;
   clean, blind, and conditional-conformance reviews remain separate. Preserve
   disagreements and all failures. Do not select examples using target outputs.

Before any v13 API generation, local-product windows were removed from both
conditions: reducing the permitted radius can change a displayed SMILES while
leaving it a correct partial description of the intended product. That is not
an unambiguous hallucination label. The six graph/formula/count errors suffice
without labeling this representation change or adding artificial numeric errors.

This is a controlled source-state intervention, not a fitted distribution of
natural hallucinations or a pure rewording of v12. N now also contains the actual
input source representation, so N differs from v12. Complete-source copying is
a possible mechanism/confound and must be tested by matching generated answers
to S′, P′, S, and GT. Full-source and local-fragment effects must not be conflated.

## Implementation boundaries

- New `source_state_candidates.py`: bounded source omissions and two-path
  execution; no model outputs.
- New `source_state_renderer.py`: independent N, conditional H, exact offsets
  and known claim provenance; no API calls.
- New `source_state_orchestrator.py` and `generate_source_agent_pairs.py`:
  isolated roles, resumable artifacts, fixed 18/132 split, source snapshots and
  both model tokenizers. Separate worker processes avoid RDKit/Python thread
  contention. No existing v12 source module is rewritten during its evaluation.
- Reuse the existing evaluator and immutable question checks. Freeze a fresh
  evaluation directory. Keep all 18 planned development origins, including
  construction or API failures; do not replace failures with easier examples.

## Validation and expansion

Verify replay/composition, protected atoms, spectator/stereo preservation,
reference-edit applicability, no complete-product strings, conditional
arithmetic, offset/token labels and resume behavior. Require at least one root,
six distinct wrong rendered nodes and forty error tokens under each tokenizer;
never add fictitious numeric roots to satisfy density.

Only after independent artifact review run ABCDE, GT probability probes and
answer entropy on the accepted development batch. Report every denominator,
paired flips and exact McNemar, and separate stereochemical from connectivity
errors. Do not infer reasoning-plan adoption from an accuracy drop alone.

Expansion requires B accuracy strictly below A on both models, evaluated on all
accepted development origins. This is an exploratory screening criterion, not a
significance claim. If passed, freeze construction and evaluate all 132 heldout
origins without target-output filtering; otherwise preserve the failed result
and revise the hypothesis. The construction and screening rules above were
written before evaluation; the completed observation is recorded separately below.


## Completed v13 observation

The 18-origin batch retained 17 accepted origins and one construction rejection. Both model runs finished at 2026-09-20 04:29:47 +08:00 with launcher status0: 85 ABCDE answers, 68 GT probes and 68 n=8 entropy prompts per model. Greedy A/B/C/D/E correct counts were ChemDFM 14/15/14/15/15 and Chem-R 9/12/11/8/8 out of17; both B>A, so the screening gate failed and no heldout expansion occurred. B−A harmful/rescue counts were DFM0/1 and R1/4, exact McNemar p=1/.375. Neither model's B matched S′ or P′ on any origin, with or without stereo. Chem-R's sole harmful flip (add0172) changes connectivity; no complete-graph match identifies internal execution. Mean GT-token logprob H−N/H−A was DFM−.001893/−.011506 and R−.004225/−.007227 nats. Mean canonical-SMILES entropy bits A/B/C/E were DFM .126802/.118571/.154268/.156096 and R .707731/.464940/.455785/.522414. Secondary origin-averaged sampled accuracies were DFM83.09/88.24/84.56/86.03% and R51.47/68.38/63.97/49.26%; each origin's correct/8 is one repeated-sampling diagnostic, not eight independent questions. Probability and entropy do not replace the failed greedy gate. Exact [archives](../../../Experiments/agent_corruption_v13/development18/evaluation/archive/terminal_validation.json), [gate](../../../Experiments/agent_corruption_v13/development18/v13briefgate.json), [source-state signatures](../../../Experiments/agent_corruption_v13/development18/review/source_state_mechanisms.md), and [sampled accuracy](../../../Experiments/agent_corruption_v13/development18/review/sampled_accuracy.md) retain full coverage and invalid/missing denominators.
