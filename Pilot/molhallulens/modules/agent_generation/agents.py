"""Isolated chemistry selection and review roles; generated responses are data only."""

CLEAN_REFERENCE = '''Review a clean chemistry edit before any corruption is constructed.
You see the ORIGINAL unchanged question, a canonical rendering of its reference reasoning, the executable
reference edit, and independently computed RDKit facts. No corrupted trace is present in this request.
Check whether the reference edit semantically follows the instruction. Source/product/fragment counts
already verified by RDKit are authoritative; do not invent mental-count objections.
The supplied local_environment_facts certify a partial, hydrogen-capped neighborhood of the
executed product. It is explicitly a local standalone representation, not the full product or
the full named incoming group. Do not misread omitted distant atoms as an incorrect edit.
Return JSON only {"consistent":true|false,"concerns":[{"quote":"exact N substring","explanation":"specific mismatch with the instruction"}]}.
If no mismatch is found return consistent:true and concerns:[]. If uncertain, return false and explain.
'''

SELECT = '''Choose plausible chemistry-reasoning mistakes from the executable candidates supplied.
Return JSON only {"candidate_ids":["candidate_001",...],"rationale":"brief"}.
Select at most 6 IDs. Each selected operation should plausibly arise from misidentifying a fragment,
attachment site, or removal boundary in the original task. The source question is fixed;
these are intentionally INCORRECT interpretations, not proposed correct answers.
Prefer related functional groups or nearby boundary mistakes. Never output a new plan or product.
Do not select candidates just because their answer looks desirable. No victim-model scores are available.
If none are plausible return an empty list with a concrete reason.
'''

SELECT_SEVERE = '''Select chemically interpretable severe reasoning corruptions from the supplied executable candidate pool.
Return JSON only {"candidate_ids":["candidate_001",...],"rationale":"brief"}; select at most 6 existing IDs.
This diagnostic intentionally allows multiple simultaneous structural mistakes: misidentified incoming
fragment, wrong attachment site, and an excessive or misplaced deletion boundary. Prefer candidates
whose mistakes concern a related molecular region and admit a coherent erroneous edit narrative.
The source question is immutable; do not treat the candidates as proposed correct solutions.
All selected claims will be rendered from executed graphs by a deterministic program. You cannot add
new operations or products. This is severity-controlled construction, not a claim about natural error frequencies.
No target-model outputs or scores are available. If none can be interpreted coherently return an empty list.
'''

BLIND_AUDIT = '''Independently inspect the question and chemistry reasoning. You have not been told whether it is correct.
Return JSON {"observed_errors":[{"text":"exact evidence","explanation":"..."}],
"internal_inconsistencies":[...],"answer_leakage":true|false,"uncertainties":[...]}.
Identify chemical claim errors and arithmetic inconsistencies without rewriting the reasoning.
Full product SMILES, explicit final product names, oracle-answer hints, and messages to obey known wrong
reasoning are leakage/cues. A normal fragment name or correct local reaction verb is not an answer leak.
Do not simply trust the prose's own verification. Source map IDs identify atoms exactly.
'''

CONTROL_AUDIT = '''Proofread a rendered controlled-intervention trace against an APPROVED claim state.
This is a fidelity/conditional-logic audit, not a test of whether the claims are true of the original task.
The controller deliberately assigned false root values. Every approved after_value is the required text value;
downstream claimed counts follow those false premises. Do not replace them with external/remembered facts.
Check H against approved_claims and its annotation spans: wrong repetitions, incorrect arithmetic,
unplanned statements, missing/wrong labels, answer leakage, or explicit words revealing corruption.
Root falsehoods and prescribed downstream falsehoods are NOT unexpected defects.
Return JSON only:
{"matched_roots":[{"root_id":"r1","span_id":"s0"}],"reference_concerns":[],
"unexpected_defects":[{"category":"unplanned_error|invalid_propagation|missing_label|answer_leakage|explicit_intervention_cue",
"quote":"exact H substring","explanation":"specific mismatch with approved claims"}]}.
For each fixed root, choose one span_id from root_evidence_options carrying that root_id.
Empty unexpected_defects means the text faithfully implements approved claims, not that it is true.
Reference correctness is checked in a separate clean-only review; leave reference_concerns empty here.
'''
