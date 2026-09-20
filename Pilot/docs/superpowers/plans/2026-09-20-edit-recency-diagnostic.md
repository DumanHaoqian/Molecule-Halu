# Edit-order diagnostic following v13

## Observed motivation

The complete v13 greedy development sets failed the two-model screening gate:
ChemDFM A/B were 14/17 and 15/17, Chem-R A/B were 9/17 and 12/17.
Neither model's B outputs matched the perceived source or its intended edited
product, with or without stereochemistry. Larger source reconstruction errors
therefore did not establish the desired answer degradation.

The next bounded hypothesis is that an explicit erroneous edit immediately
before the answer has more influence than the same edit followed by atom/ring
accounting. This is a presentation/recency diagnostic, not a claim about the
cause of the previous null results.

## Fixed chemical content and paired conditions

Reuse **all 14 accepted v12 parent origins** and their exact selected structural
plans, roots and executed products. Retain the four original construction
rejections in the 18-origin planning denominator. Do not resample candidates or
select origins using either target model's outputs.

Render the same no-local-window content in two orders:

- `account_last`: identification, apply edit, account for atoms/rings.
- `edit_last`: identification, account for atoms/rings, apply edit.

The optional local-product window is absent in **both** conditions, eliminating
the previous ambiguity in treating a changed permitted radius as a false local
claim. The original-order no-local condition is required: a comparison only to
v12 would confound order with removal of the local window.

Only the step order and corresponding step numbers may differ. Reconstruct all
bindings, exact character offsets and both tokenizer labels. N/H share the same
order within each condition. Chemical values, root identities, claims, question,
instruction, scoring GT, model prompts and decoding settings stay fixed. Check
each pair of ordered texts becomes byte-identical after undoing the permutation
and normalizing the three step numbers.

The actual CPU materialization retains 7–12 wrong semantic nodes per parent.
Both orders have 65–260 error tokens under ChemDFM and 51–204 under Chem-R,
with zero density exceptions. These are construction measurements, not behavior
results. Any later density exception must be reported rather than filtered out.

## Implementation and checks

Implemented the separate CPU-only [derive_edit_order_diagnostic.py](../../../scripts/derive_edit_order_diagnostic.py)
and [focused tests](../../../tests/test_agent_edit_order_diagnostic.py).
The 10 focused tests passed after the initial missing-script failures; the combined
88-test derivation, label, renderer, severity and evaluator suite, including those
10 tests, passed. Tests cover
inverse text normalization, unchanged chemistry, shifted spans/bindings, density
retention, parent drift rejection, inherited rejections and stable E donors.
No v12/v13 frozen module or data was changed.

The tool replays both parent executions, verifies every nonlocal node and the
parent text outside the removed window, then permutes complete blocks and
renumbers only their step headings. It rebuilds offsets and token labels from
the parent's frozen tokenizer backends and recomputes formal checks, leakage
checks and severity. Inverse permutation must restore byte-identical content.
Semantic labels retain the disclosed v12 reference-relative scope; no new
chemical claim or local-fragment truth rule is introduced.

Both generation directories were atomically published under
`Pilot/Experiments/agent_corruption_v14/`: [account_last](../../../Experiments/agent_corruption_v14/account_last/summary.json)
and [edit_last](../../../Experiments/agent_corruption_v14/edit_last/summary.json).
Each has `planned=18`, `processed=18`, `accepted=14`, `rejected=4`,
`actual_api_calls=0` and `production_accepted=0`. The four rejections are explicitly
marked `inherited_parent_rejection`; no original parent was dropped or rescued.
Density reporting is non-gating, with no violations in the published artifacts.

Each manifest uses `protocol=agent_edit_order_diagnostic_v1` plus its `order`.
The two isolated directories share pair IDs and sorted origin order so that
evaluation seeds and E donor origins remain aligned. Each contains 28 pair rows,
all 18 origin terminal records, N/H semantic bindings, character annotations,
token labels, current implementation snapshots and untouched parent snapshots.
Old Agent opinions are retained in `historical_parent_reviews`; current review
status is `unknown_not_repeated`, not a fresh approval. The release contract is
`diagnostic_only` with `production_eligible=false`.

The following is the exact CPU derivation command already used, provided for
reproduction rather than for rerunning the existing output. The tool requires
the entire `--output` root to be absent and refuses overwrite/resume; a future
derivation must choose a fresh root. Neither this command nor its tests invokes
an API, model weights or GPU.

```bash
cd /mnt_nas1/haoqian/Data/Molecule
PYTHONPATH=Pilot /home/haoqian/.venvs/chemllm-outcome/bin/python \
  Pilot/scripts/derive_edit_order_diagnostic.py \
  --parent Pilot/Experiments/agent_corruption_v12/development18 \
  --output Pilot/Experiments/agent_corruption_v14
```

Independent selected-artifact chemistry review and evaluator CPU preflight are
separate checks assigned after publication. At this checkpoint, no v14 GPU
behavioral result is claimed.

## GPU entry and result isolation

The dedicated [run_edit_order_exp.sh](../../../scripts/run_edit_order_exp.sh)
accepts no arguments and reads the two published v14 directories. Its launch
entry, after independent artifact review and CPU preflight, is:

```bash
cd /mnt_nas1/haoqian/Data/Molecule/Pilot
bash scripts/run_edit_order_exp.sh
```

The independent log is `Pilot/scripts/run_edit_order_exp.log`; its PID file is
`Pilot/scripts/run_edit_order_exp.pid`. These do not overwrite v13's
`run_agent_exp.log` or PID file. Each condition writes its own evaluation
directories and `evaluation/agent_corruption_result.md`. The worker respects the
shared chemistry GPU locks and waits for GPU capacity rather than interrupting
other jobs. A running worker must not be started twice.

Run ABCDE under each condition on GPU 4–7, with ChemDFM account_last then edit_last,
followed by Chem-R account_last then edit_last. There are 70 greedy requests per
model/condition, with the existing evaluator/system settings, TP=4, batch size=8
and GPU memory utilization 0.72. This initial launcher does not rerun probability
or entropy sampling. Each condition uses its own fresh
directory and the same deterministic E donor selection. Report all denominators
and compare B(edit_last) with B(account_last) on the same origins, alongside
B−A and B−C within each condition. Verify repeated A requests as a runtime
control. Do not pool repeated origins as independent observations.

This diagnostic alone cannot authorize heldout expansion. If the new order
shows the desired B<A direction on both models, freeze the revised generation
protocol and validate it on the complete planned development batch before
accessing the untouched 132-origin heldout. Otherwise preserve the failed result
and reconsider the construction mechanism. The design above was frozen before
evaluation. The completed greedy-only observation is recorded separately below;
no v14 probability or entropy run was performed.


## Completed v14 observation

All four greedy sets completed with supervisor status 0 at 2026-09-20 04:46:53 +08:00, each covering the same 14 accepted parent origins plus four inherited construction rejections in the planning denominator. All 280 outputs were valid, with no missing or truncated answers. DFM account_last and edit_last both had A/B/C/D/E correct counts 11/12/12/11/12. Chem-R account_last had 8/7/12/9/10; edit_last had 8/8/11/9/9. Primary and full isomeric correctness counts coincide. B(edit_last)−B(account_last) was zero for DFM and +1/14 for Chem-R, with delete0255 rescued, no harmful flips and exact McNemar p=1. Neither condition met the two-model B<A direction. This does not support the predicted worsening from putting the edit last, and it does not authorize heldout expansion or post-hoc winner selection.

Within-condition B−A/B−C always uses the current condition's observed A/C: DFM +1/14 and 0 under both orders; Chem-R account_last −1/14 and −5/14 (B−C p=.0625), edit_last 0 and −3/14 (B−C p=.25). Tests are exploratory and unadjusted for multiple comparisons. Chem-R's historical v12 A=9/14 is not substituted for this round's A=8/14, and no cause is inferred for the cross-round change.

Both models' repeated A outputs were identical on all 14 origins at the complete generated-text, parsed-SMILES and correctness levels. D complete text matched on 12/14 DFM and 10/14 Chem-R origins, while parsed SMILES, full stereochemical graphs and correctness matched on all 14 in both models. Text variability therefore did not imply answer variability. Full wrong-plan product matches under B were 0/14 for every model/order, also 0/14 without stereo. Chem-R B GT full-stereo/achiral counts were 7/8 for account_last and 8/10 for edit_last; full-product equality does not identify internal execution or exclude partial changes.

The [paired comparison](../../../Experiments/agent_corruption_v14/review/order_comparison.md) and its JSON retain all per-origin metrics, harmful/rescue identities, Wilson intervals, exact McNemar tests and repeated controls. [Exact terminal log](../../../Experiments/agent_corruption_v14/review/run_edit_order_exp.log), [terminal validation](../../../Experiments/agent_corruption_v14/review/terminal_validation.json), [independent construction audit](../../../Experiments/agent_corruption_v14/review/independent_order_audit.md) and [evaluator preflight](../../../Experiments/agent_corruption_v14/review/evaluator_preflight.md) preserve the run and construction evidence. No new Agent review or production approval is claimed.
