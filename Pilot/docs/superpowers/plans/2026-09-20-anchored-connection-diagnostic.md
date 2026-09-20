# Fixed-plan anchored product-connection diagnostic

## Evidence and bounded hypothesis

The completed v15 experiment did not meet H<A in either expression package.
On the same four predeclared development acylations, both models had A=2/4;
B was 2/4 for the ledger and 3/4 for the original-prose package. Neither model
emitted the full executed wrong product. This is a small exploratory failure,
not evidence that the models ignore all reasoning. Repeated D answers also
varied in molecular structure despite unchanged correctness, so current paired
controls must be rerun rather than substituted with historical results.

The next hypothesis concerns representation of the realized connection:
does explicitly displaying the fragment attached to the original source anchor
make an otherwise identical erroneous plan affect the generated graph?
The previous v12 local windows used adaptive radii and hydrogen caps. The new
representation uses a fixed operation-defined boundary, keeps real product
hydrogens and valence, and explicitly represents open connections to the omitted
source. It is not merely rewriting v15's bonding instruction in the past tense.

## Fixed cohort and intervention

Protocol: `agent_anchored_connections_diagnostic_v1`, fresh directory v16.
Use exactly v15's four origins: add0098, add0172, add0193 and add0194. Preserve
all 18 development inputs and the 14 explicit outside-scope records. Never
replace a failed construction with another origin. No heldout rows are opened.
All wrong plans, executed products, original instructions, indexed source
molecules, group-name bindings and the single fragment root remain frozen.

Two paired styles:

- `native`: exact v15 native N and H, rerun as the concurrent control.
- `native_with_connections`: the same N/H with a symmetric final block showing
  their respective realized, open-boundary product-connection representation.

Retain the entire incoming operation and the same source anchor. Replace each
cut anchor-to-source-neighbor bond by a dummy port carrying that omitted source
atom's map. The same source anchor, port identities and presentation rule apply
to N and H; incoming runtime map numbers are not cross-graph atom identities.
All four boundaries are single bonds without boundary stereo annotations. The
stearoyl reference includes 20 real heavy atoms; do not impose an unrelated
16-atom cap or choose another radius to obtain acceptance.

## Chemical and label contract

`[*:k]` means an open connection to omitted original source atom k, not an actual
element replacement or a hydrogen cap. Preserve each retained real atom's
product hydrogens, charge, valence and bonds. Replacing ports by their saved
external source context must recover the entire corresponding executed product,
including specified stereochemistry. Reject ambiguous or unsupported boundaries.

The N18 ports remain connected through an omitted piperidine path; no claim of
acyclic chemistry follows from the partial serialization. External stereo is
preserved in the full graph, not asserted to be represented by this local value.
Do not infer full-product rings or formula from the cropped graph.

The H connection value is a single propagated node with parent `fragment` and
root `r1`, conditional on the existing false named-fragment binding. It is marked
only after verifying that the anchored connection differs from the correct
product at that same source site. Correct anchor/port tokens are not counted as
independent false claims. Whole-value labeling locates a false graph relation,
not per-atom falsehood. The original native annotations remain unchanged.

Check both explicit SMILES and the real-atom projection after dummy removal
against complete GT/P' molecules and every complete component. No complete
answer may appear. This does not claim the trace contains no answer information:
the unchanged source plus a fully specified editing operation already permits
product reconstruction. The new factor is explicit partial product structure,
and its information/format advantage must be acknowledged.

## Implementation, verification and decision

Use separate chemistry/renderer modules with exact replay, positive and rejection
tests. The compiler copies frozen v15 native records, verifies original inputs
and partition membership, rebuilds token labels for both frozen tokenizers, and
publishes both styles together without replacing existing artifacts. Store all
boundary witnesses, text edits, implementation/tokenizer snapshots and failures.
This bounded compiler makes no new API calls and claims no fresh model approval.

Run the unchanged evaluator on GPU4-7 with the exact v12 runtime: DFM native,
DFM augmented, Chem-R native, Chem-R augmented; ABCDE, 20 requests each. E uses
the corresponding donor N from the same fixed four-origin pool. Report lengths
and densities without padding or outcome-based filtering.

Check B-A and B-C within each style, B/C changes across styles, A/D repeated
controls, complete wrong-product agreement with/without stereo, and adoption of
the predeclared wrong connection where output-to-source mapping is unambiguous.
If mapping is ambiguous or the source graph is not retained, report the local
mechanism as undecidable rather than calling it absent. Correct N augmentation
is necessary to distinguish useful structure context from harmful corruption.

For this auxiliary connectivity diagnostic, enumerate all valid retained-source
embeddings up to a fixed limit. Remote symmetry may produce several full-source
embeddings; it is admissible only when every valid embedding identifies the same
answer atom as the original anchor and yields the same canonical connection
projection with original-source port identities. Never choose a favorable
embedding. Reaching the enumeration limit, disagreement about the anchor or
projection, and a changed source-induced graph are separate undecidable cases.
Outputs containing dummy atoms and exact N/H open-port draft copies are separate
format/copy outcomes, not evidence of producing a concrete wrong molecule.

No improvement may be attributed to greater chemical edit amplitude: plans are
identical. Four reused development origins cannot establish general success or
authorize direct heldout expansion. If the new block yields no error uptake,
retain that falsification rather than adding more copies or choosing individual
successful outputs. A general Agent generator and a frozen full-development
validation remain prerequisites for a later 132-origin heldout experiment.

## Frozen auxiliary mechanism classification

Before v16 target evaluation, the local-connection diagnostic is defined as
follows. It does not alter any request, origin, primary accuracy denominator or
heldout decision. RDKit-parseable answers containing dummy atoms are counted
separately as `predicted_has_dummy`, not as concrete complete products. Exact
map-normalized copies of either N or H open-port draft are also reported
separately, with stereo and achiral comparisons; copying such a draft is not
evidence of producing the complete wrong product.

For concrete answers, enumerate source heavy-atom embeddings with a fixed cap
of 4,096. Reaching that cap means `unknown_truncated_mappings`; do not assume the
enumeration is complete. Check element, isotope, charge, aromatic state and the
induced source-internal bond graph for each embedding. Source hydrogen counts
and stereochemistry are not required to match the unedited source in this
connectivity diagnostic; full product stereochemistry is reported separately.
No complete intact source embedding means `unknown_source_not_intact`.

Remote source symmetries may produce multiple embeddings. A local N/H/other
decision requires every intact embedding to map the declared source anchor to
the same predicted atom and to produce the same canonical open-port connection
projection with original source-port identities retained. Any anchor or
projection disagreement is `unknown_ambiguous_mapping`; never select the most
favorable embedding. Unsupported bridge boundaries are also unknown. The
connection projection retains the actual anchor and its attached new-atom
component(s), preserves their observed hydrogen/charge state, and replaces cuts
back into the omitted source by the declared ports. Compare that projection to
the frozen N/H connection values without stereochemistry, preserving isotope,
charge and port maps. Every unknown category is reported separately from
decidable absence or presence of the wrong connection.

## Observed v16 results — 2026-09-20

The frozen four-origin cohort, both styles and all 80 greedy ABCDE requests
completed. The supervisor exited status 0 at 2026-09-20T05:54:40+08:00. All
answers were RDKit-parseable and dummy-free, with no missing or truncated
answers; no v16 probe or entropy stages were run. The published artifact audits
passed 334 independent chemistry checks, 11,906 label checks and 150,861 actual
request/token checks over 29,781 trace-token records. No fresh API review or
production acceptance was claimed.

| Model | Condition | A | B | C | D | E |
|---|---|---:|---:|---:|---:|---:|
| ChemDFM-R-14B | native | 2/4 | 3/4 | 3/4 | 2/4 | 3/4 |
| ChemDFM-R-14B | native_with_connections | 2/4 | 3/4 | 3/4 | 2/4 | 3/4 |
| Chem-R-8B | native | 3/4 | 3/4 | 2/4 | 2/4 | 2/4 |
| Chem-R-8B | native_with_connections | 2/4 | 2/4 | 4/4 | 2/4 | 2/4 |

Primary and full stereo correctness agreed for all 80 outputs. ChemDFM's B and C
did not change across conditions. Chem-R's B decreased 25 percentage points,
with add0172 flipping wrong (1/3 baseline-correct; Wilson95 [6.15%,79.23%], exact
McNemar p=1); its C increased 50 points with two rescues (p=0.5). Within-style
B−A was +25 points for ChemDFM in both styles and zero for Chem-R in both.
Chem-R's augmented B−C was −50 points, with add0098/add0172 harmful flips out of
four C-correct cases (Wilson95 [15.00%,85.00%], p=0.5). These are exploratory
four-origin statistics without multiple-comparison correction.

The repeated A control was identical in 4/4 cases for ChemDFM and 3/4 for
Chem-R. Chem-R add0172 A was fully correct in the native stage but lost stereo
specification in the augmented stage, while retaining GT connectivity. Its B
flip at that origin also changed connectivity. These errors are distinct; the
unchanged A prompt contained no augmented connection block, so no such causal
attribution is made. Each within-condition contrast uses its own current A/C,
not a historical or cross-condition replacement. D's full generated text was
identical in only 1/4 cases per model, but all parsed answers, full graphs and
correctness repeated in 4/4.

All sixteen B outputs had zero complete P′ matches, with or without stereo, and
none was a GT-connectivity-only stereo error. No N/H open-port draft was copied
as a complete answer in any of the 80 outputs; no dummy atoms were present.
The eight frozen program-reference GT/P′ positive controls all recovered the
proper N/H connection under 16/8/2/2 source embeddings for the four origins.
For B, local connectivity was decidable in 4/4 cases for both ChemDFM styles
(N=3,H=0 each) and Chem-R native (N=3,H=0). Chem-R augmented was decidable in
3/4 (N=2,H=0); add0172 had no intact source embedding and remains
`unknown_source_not_intact`, not evidence of absent H adoption. The other
decidable wrong answers matched neither frozen N nor H connection. These
output classifications do not establish an internal model reasoning path.

The independent chemistry reviewer inspected the classifier and reran all 13
mechanism tests without reading target outputs; no blocking defect was found.
Its `source_intact` check covers heavy-atom identity and induced source
connectivity, not every remote hydrogen, substituent or stereochemical state.
See the [independent classifier review](../../../Experiments/agent_corruption_v16/review/classifier_independent_review.md).

Both styles retained all four cases and had no density exceptions. Each
augmentation added exactly one propagated node, leaving the single fragment
root and chemical plans unchanged. Nodes increased from 7/9/6/7 to 8/10/7/8;
H full-prompt error tokens increased from 80–85 to 104–115 for ChemDFM and
70–74 to 92–101 for Chem-R. The result does not demonstrate greater chemical
edit amplitude, general error susceptibility, or successful uptake of the
specified wrong graph. Neither condition met dual-model B<A. No heldout
evaluation, target-derived origin filtering or condition winner selection
followed this diagnostic.

Evidence: [complete paired and mechanism report](../../../Experiments/agent_corruption_v16/review/anchored_connection_comparison.md),
[per-origin records](../../../Experiments/agent_corruption_v16/review/anchored_connection_comparison.json),
[reference controls](../../../Experiments/agent_corruption_v16/review/local_connection_reference_controls.json),
and [terminal archive](../../../Experiments/agent_corruption_v16/review/terminal_validation.json).
