# Fixed-graph task interpretation diagnostic

The v16 connection augmentation did not make B accuracy lower than A for both
models. None of its decidable B outputs adopted the fixed wrong connection.
The next exploratory hypothesis is that an internally inconsistent group name
and graph lets the model recover the requested edit. Close that inconsistency
inside H while retaining the original external question unchanged.

Protocol `agent_task_intent_diagnostic_v1`, output v17. Retain the same fixed
four development acylations (0098, 0172, 0193, 0194), the full 18-case development
denominator and 14 explicit outside-scope records. The 132 heldout cases remain
unevaluated. No per-origin target-based selection, replacement, padding, new
candidate search or API review is performed by this bounded compiler.

Both conditions derive from frozen v16 `native_with_connections`:

- `binding`: exact parent N/H text and annotations. The one root is the original
  false name-to-fragment relation (`r1`, `fragment`, `structural`).
- `intent`: N text stays byte-identical. Replace H's original group aliases by
  `4-(trifluoromethyl)benzoyl`, the actual chemical name of the already-fixed
  wrong graph. Also replace 0194's task paraphrase `acetylating` naturally with
  `acylating ... with a 4-(trifluoromethyl)benzoyl group`. Correct grammar is an
  unlabelled edit. The one root now means a wrong perceived task group
  (`r1`, `task_group`, `task_interpretation`); the fragment is its propagated
  consequence. Repeated aliases are repeated mentions, not independent roots.

There are eleven expected task-group root mentions: two for 0098, four for
0172, two for 0193 and three for 0194. Preserve the same original source anchor,
wrong incoming fragment `C(=O)c1ccc(C(F)(F)F)cc1`, carbonyl attachment index 0,
single bond, source hydrogen adjustment, formulas, counts and appended open
connection value. Chemistry graph amplitude does not increase in this contrast.
The intervention changes the CoT's interpretation of the task; it does not edit
the true instruction or resolve its external conflict with H.

Rebuild complete N-to-H patches and token offsets for both model tokenizers.
Rebase fragment annotations from root to propagation; remove obsolete current
false-name-to-graph evidence. Preserve old root/binding semantics only as clearly
marked historical provenance. N's metadata may be renamed while its text stays
exact. Check chemistry independently, including unchanged source roles,
open-boundary port reconstruction, full-product/component exclusion and exact
physical plans across conditions. A full answer string is forbidden, but the
partial graph and specified edit still contain reconstructible answer information.

Use immutable inputs, atomic two-condition publication and no output overwrite.
Freeze implementation, tokenizer and alias inventory before target execution.
Actual full prompts for A/C/D/E must match across conditions; only B changes.
Do not modify evaluator prompts or runtime to force obedience.

Run four sequential stages on GPU 4-7: DFM binding, DFM intent, Chem-R binding,
Chem-R intent. Retain the same v12 runtime (TP4, bfloat16, max context 16384,
batch 8, seed 42, temperature 0, repetition penalty 1.05, max output 2048),
20 ABCDE requests per stage, without entropy or teacher-forcing probes.

Report all four origins, paired B-A/B-C, between-condition B change and repeated
A/C/D/E stability; no historical baseline substitution. Report accuracy,
similarity, complete wrong-product agreement, and the frozen v16 local connection
classifier's N/H/other/unknown categories. Retain the 4096-embedding cap,
anchor/projection invariance criterion, dummy/draft-copy diagnostics and all
unknowns; a local match is not evidence of an internal reasoning path. Accuracy
decline alone is not proof of adopting the intended corruption.

These four repeatedly used development cases are exploratory. Even a favorable
result cannot authorize immediate heldout expansion: general bounded Agent
generation and a frozen full-development validation must come first. An
unfavorable result is retained in full rather than filtered away.


## Measured v17 result (complete)

All four stages completed on 2026-09-20 at 06:21:51 +08:00 with supervisor
status 0: 80/80 greedy answers, all parseable and dummy-free, no missing or
length-truncated answers. No probes, entropy, new API review or heldout run.
The four fixed origins remain paired, and fourteen remain outside scope.

| Model | Style | A | B | C | D | E |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ChemDFM-R-14B | binding | 2/4 | 3/4 | 3/4 | 2/4 | 3/4 |
| ChemDFM-R-14B | intent | 2/4 | 2/4 | 3/4 | 2/4 | 3/4 |
| Chem-R-8B | binding | 3/4 | 2/4 | 4/4 | 2/4 | 2/4 |
| Chem-R-8B | intent | 3/4 | 3/4 | 4/4 | 2/4 | 2/4 |

DFM B changed by −25 percentage points, with add0194 harmful (1/3 baseline
correct; Wilson95 6.15–79.23%, exact McNemar p=1). Chem-R B changed by +25
points, with add0172 rescued, zero harmful among two baseline-correct (p=1).
Both styles fail the both-model H<A direction. All within-style comparisons
use current A/C: DFM binding/intent B−A +25/0 and B−C 0/−25 points; Chem-R
−25/0 and −50/−25 points. B full-product stereo correctness equals primary
correctness; B Morgan similarity changes 0.935714→0.873214 for DFM and
0.903991→0.950658 for Chem-R.

A/C/D/E complete requests are identical across styles. Each model repeats all
four A/C/E complete texts and answers exactly. D full generated text repeats
only 1/4, while parsed answers and full molecular graphs repeat 4/4. No cause
is assigned to D's text variation.

No B output matches full P′, either with or without stereo (0/4 in every set),
and no B output is merely a GT stereo discrepancy. No dummy or complete N/H
open-port draft copies occur. DFM binding/intent local classification is
4/4 decidable in each, with N/H counts 3/0 and 2/0. Chem-R is 3/4 and 4/4
decidable, N/H 2/0 and 3/0; binding add0172 is unknown_source_not_intact,
not a negative H-adoption observation. The frozen classifier's local-source
intactness does not guarantee remote hydrogen, substituent or stereo accuracy,
and output agreement cannot establish an internal reasoning path.

Independent chemistry: 199 checks; label audit: 13,800 checks; evaluator
preflight: 165,797 checks including 32,722 full-prompt token records. All pass.
Each style retains one current root per origin, but the root ontologies differ
as specified. Wrong nodes are binding 8/10/7/8 versus intent 9/11/8/9. No density
exceptions, filtering or chemical amplitude changes occurred.

The failed directional gate and complete results are retained. This four-case
exploratory diagnostic does not authorize heldout expansion. See the
[paired report](../../../Experiments/agent_corruption_v17/review/task_intent_comparison.md),
[all scored records](../../../Experiments/agent_corruption_v17/review/task_intent_comparison.json),
[exact log](../../../Experiments/agent_corruption_v17/review/run_task_intent_exp.log) and
[terminal validation](../../../Experiments/agent_corruption_v17/review/terminal_validation.json).
The pre-execution plan remains separately archived unchanged.
