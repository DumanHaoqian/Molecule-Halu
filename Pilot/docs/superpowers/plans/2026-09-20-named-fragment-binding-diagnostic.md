# Named-fragment binding diagnostic after v14

## Evidence and hypothesis

The complete v14 paired-order experiment failed its intended screening direction
on ChemDFM: both orders had A=11/14 and B=12/14. Chem-R had A=8/14 in both orders,
with B=7/14 for account-last and 8/14 for edit-last. Both models repeated every A
answer exactly. No complete H product was generated under either order, with or
without stereochemistry. Reordering the existing ledger did not establish
greater susceptibility.

Original N contains explicit chemical relationships omitted by that ledger:
which source functional group is attached to the carbonyl carbon of a named
incoming fragment. The next narrow diagnostic tests whether retaining that
original expression changes uptake of a fixed erroneous name-to-graph binding.
This is a development hypothesis, not a conclusion or a fitted natural-error
model.

## Fixed cohort and chemistry

Use the same previously declared 18 development origins. This diagnostic's
eligibility is addition by N/O acylation, including acetylation: add0098,
add0172, add0193 and add0194. All four remain included regardless of either
target model's outputs. Record the other 14 as outside this diagnostic's stated
chemical scope, not as newly observed behavior failures.

Keep the correct original source anchor, single-bond attachment, original-source
hydrogen adjustment, all source atoms and all other reference operations. Replace
only the incoming fragment with `C(=O)c1ccc(C(F)(F)F)cc1`, attaching atom index 0.
The attachment-qualified moiety is 4-(trifluoromethyl)benzoyl. Its standalone
hydrogen-capped representation has formula C8H5F3O; its attached acyl moiety has
formula C8H4F3O. These formula scopes must never be conflated.

The independent CPU feasibility check covered all four origins, with 129 checks
and zero failures. Correct anchors are N18, O28, N24 and N20 respectively. GT to
wrong-product heavy-atom differences are +7, -7, +7 and +9; ring-count differences
are +1, +1, 0 and +1. In particular, add0193 retains its correct ring count even
though ring identity changes.

## Two expressions of the same error

- `ledger`: existing typed no-local-window renderer of the clean and fixed wrong
  plans, with exact program-controlled annotations.
- `native`: N is the original N with complete product/answer occurrences removed
  by the existing symmetric projection. H changes only an explicitly recorded,
  independently checked inventory of dependent source-text spans.

The old ledger does not explicitly display the original group names. Keep that
control unchanged and record `ledger_names_displayed=False`. The comparison is
therefore the complete original-name/source-role/connection-language expression
versus the ledger, with the same underlying graph intervention. It does **not**
isolate a name-only effect or a pure wording effect.

The intended original group names remain unchanged in native N and H, including
accurate restatements of the instruction. The erroneous binding is **original
name -> wrong attachment-qualified graph**. Do not substitute the true name of
the new group into the trace and silently turn the test into a misread-instruction
experiment. Preserve the original source-role and carbonyl-attachment language.

Replace every displayed incoming-fragment SMILES with the fixed wrong fragment.
Update the dependent composition, topology, product counts, deltas and all their
repetitions. Long-aliphatic-chain, acyclic and cyclopropane descriptions require
explicit treatment. Equality words such as "also" must be checked when counts
diverge. Unchanged values, especially add0193's ring count, receive no new error
label. Neither complete GT nor the wrong product may be displayed.

There is one binding root. Repeated SMILES and derived assertions share that
root; repetition does not create independent root events. Unchanged original
names are correct references to the task symbol and are not themselves labeled
as changed values. The mapping claim is intentionally chemically false.
Only the execution and numerical consequences *conditional on that false
binding* are internally coherent; the entire prose must not be called chemically
contradiction-free. Labels identify specified false semantic values/relations,
not independent truth values of every constituent token.

## Bounded implementation and review

Implement a separate CPU diagnostic compiler and a declarative source-span
inventory for these four development examples. Every source quote must resolve
exactly; reject overlapping or missing spans. After compilation, verify all
unmodified text remains byte-identical and every changed chemical assertion has
typed graph evidence and exact labels. Validate the complete inventory independently
against original N and both executed graphs before target evaluation. No
unconstrained free-text rewrite or extra numerical error may be used to satisfy
density thresholds.

This four-example inventory is a mechanism prototype, not a general Agent
generation system or production dataset. Preserve historical Agent provenance,
and record any fresh gpt-5.4-mini review separately if performed. Do not invent
new Agent approval. Both styles retain the same four examples; report each
style's real node/token density, including exceptions, without filtering.

Freeze a fresh v15 directory with both styles, original questions/GT, selected
plans, implementation and tokenizer snapshots, all annotations and eligibility
denominators. Use the unchanged ABCDE prefix evaluator and exact v12 runtime
settings, GPU 4-7, ChemDFM before Chem-R. Compare H-native versus H-ledger and
within-style B-A/B-C on the same four origins; check repeated A and D. Report
complete and stereo-ignored wrong-product agreement and actual graph changes.

The materializer validates the fixed development partition before opening any
origin inputs, rejects mutation of the frozen input or chemistry bundle by a
renderer, copies each published record, and publishes both styles with Linux
atomic no-replace rename. The actual NAS rejects the required rename flags;
on that filesystem, an atomic symlink publishes the complete private sibling
directory, which remains permanent backing storage. A concurrently created empty
destination and a dangling symlink are protected as well as a populated
experiment directory. These are provenance
guards; they do not strengthen the empirical hypothesis.

This is exploratory and very small. It cannot authorize a claim of general
success or immediate heldout expansion. If useful, a general bounded Agent
span-inventory construction protocol must be frozen and validated on the full
planned development cohort before the untouched 132-origin heldout is evaluated.
If unsuccessful, retain the failure and reconsider the mechanism. No target
results are asserted by this plan.

## Observed v15 results — 2026-09-20

The design above was frozen before target evaluation. Both expressions retained
all four eligible origins; the other fourteen development origins remained
outside diagnostic scope. No fresh API/model review or production acceptance
was claimed. The independent chemistry audit passed 899 checks; the label audit
passed 8,924 checks, including 67 native patches. CPU evaluator preflight passed
104,513 checks over all 80 requests and 20,553 exact rendered-prompt trace-token
records. Original source, instruction and GT were unchanged.

All four greedy ABCDE stages completed on GPUs 4–7, ChemDFM ledger/native before
Chem-R ledger/native. The supervisor exited status 0 at
2026-09-20T05:28:28+08:00. All 80 outputs were valid, with no missing or truncated
answers. No v15 teacher-forcing probe or entropy stages were run.

| Model | Expression | A | B | C | D | E |
|---|---|---:|---:|---:|---:|---:|
| ChemDFM-R-14B | ledger | 2/4 | 2/4 | 2/4 | 2/4 | 2/4 |
| ChemDFM-R-14B | native | 2/4 | 3/4 | 3/4 | 2/4 | 3/4 |
| Chem-R-8B | ledger | 2/4 | 2/4 | 3/4 | 2/4 | 2/4 |
| Chem-R-8B | native | 2/4 | 3/4 | 2/4 | 2/4 | 2/4 |

Primary-component and full stereo correctness agreed on all 80 outputs. For
both models, B(native)−B(ledger) was +25 percentage points, with one rescue and
zero harmful flips among two ledger-correct origins; exact McNemar p=1 and
Wilson95 harmful-flip interval [0%,65.76%]. ChemDFM rescued add0194; Chem-R
rescued add0172. Within-expression B−A was zero for ledger and +25 points for
native in both models. B−C was zero for ChemDFM in both expressions; Chem-R
ledger had −25 points (one harmful flip, add0098, out of three C-correct
origins; Wilson95 [6.15%,79.23%], p=1), while native had +25 points (one rescue,
add0172; p=1). This is a four-origin exploratory comparison, without
multiple-comparison correction or evidence of a stable general effect.

Every B answer failed to match the complete wrong product P′: 0/4 for all four
model/expression combinations, with or without stereochemistry. No B error was
only a GT stereochemistry discrepancy. ChemDFM ledger errors were add0172 and
add0194, versus add0172 under native; Chem-R ledger errors were add0098 and
add0172, versus add0098 under native. Each was another valid full molecular
graph, not P′. The companion report includes all sixteen B classifications;
complete P′ disagreement does not prove internal trace neglect or rule out
partial structural changes.

Repeated A requests yielded identical complete generated text, parsed SMILES,
full graphs and correctness in 4/4 cases for both models. Repeated D requests
yielded identical complete text in only 1/4 and full stereo graphs in 3/4, while
correctness agreed in 4/4. ChemDFM add0194 differed in connectivity and was wrong
both times; Chem-R add0172 differed only in stereo specification and matched
GT connectivity but failed full stereo equality both times. These observed run
differences are retained, without assigning an unsupported cause.

The expressions did not have equal false-node/token density: ledger nodes were
7/7/5/7 and native nodes 7/9/6/7 for origins 0098/0172/0193/0194. Ledger add0193
retained its five-node exception. Full-prompt H error tokens were 59–65 versus
80–85 for ChemDFM and 47–53 versus 70–74 for Chem-R. This remains a whole
expression-package comparison, not a pure wording or name-only effect.

Neither expression met the dual-model B<A direction. No heldout evaluation was
authorized or performed, and neither expression nor any origin was selected as
a target-derived winner. The bounded hypothesis did not produce the predicted
accuracy degradation or complete wrong-plan outputs on these four cases.

A Bash dynamic-scope issue made each model's ledger `MODEL DONE` log label say
native. The start labels, immutable dataset/output paths and all prepared and
actual request records identify the correct stages; independent analysis uses
those files. The exact executed launcher and log remain archived, and the
launcher was corrected only after termination.

Evidence: [paired analysis](../../../Experiments/agent_corruption_v15/review/named_binding_comparison.md),
[per-origin records](../../../Experiments/agent_corruption_v15/review/named_binding_comparison.json),
[CPU evaluator preflight](../../../Experiments/agent_corruption_v15/review/evaluator_preflight.md),
and [terminal archive validation](../../../Experiments/agent_corruption_v15/review/terminal_validation.json).
