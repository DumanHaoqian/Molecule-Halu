# SAE text preparation v1

User authorized implementation after the corpus audit. Scope: prepare text, source manifests, conservative leakage exclusions, grouped splits and reproducible activation sampling selections; no model inference, GPU activation extraction, downloads or SAE training.

## Design

- New isolated output directory `sae/data/v1`; preserve all original data and Pilot annotations. Refuse overwriting completed output. Build staging files with a completion marker only after validation.
- Main corpus: ChemCoT, selecting one CoT field per record, and all four Mol-LLaMA instruction categories. Optional supplement: original PubChem descriptions and ChEBI-20 train, never the redundant enriched-description view. Candidate generation pool: OpenMolIns xlarge and MoleculeQA train; no assistant targets from these enter the main corpus.
- Plain UTF-8 JSONL, source-specific shards, stable IDs, original source row and checksum, task, molecule/CID keys, standard chat messages and exact tokenizer counts. Resolve `<mol>` using actual molecular input; references and metadata never enter model messages. Preserve existing assistant text without fabricating thinking tags. Keep full dialogs.
- Exclude all 600 Pilot questions by normalized text, caption CID and molecular connectivity keys from source inputs/references. Normalize SMILES through RDKit; include whole structures and the largest component, ignoring stereochemistry conservatively. Do not equate this with scaffold/semantic deduplication.
- Quarantine malformed input, missing molecule context, explicit answer cues and unsupported context length (>16,384 tokens). Log every excluded source row and reason. Invalid generated output remains eligible when it is part of authentic reasoning; source input metadata validity is checked separately.
- Group main/supplement records by molecular connectivity/CID, identical contextual prompts and identical nontrivial assistant text using connected components; stable seeded 98/2 train/validation assignment. Exact full-dialog deduplication across sources. Prompt pool is `candidate`, not an SAE split; future rollouts must join the existing grouping policy before activation use.
- Count actual eligible assistant subwords in rendered ChemDFM chat context, excluding whitespace-only and control/template tokens. Counts and ranges, not huge dense activations, are stored. Preserve original text and identify teacher-forced origin.
- Create a reproducible main-corpus selection near 10M train assistant tokens (30% ChemCoT, 70% Mol-LLaMA) and 0.5M validation, without replacement and without truncating selected documents. Report actual budgets and shortfalls. PubChem/ChEBI optional, not included in initial selection.

## Implementation and validation

1. Add focused tests for chemical identity, metadata/reference separation, fallback CoT selection, answer-cue quarantine, assistant token masking, cross-source connected-component grouping and Pilot exclusion.
2. Implement normalization helpers and a streaming builder with bounded CPU workers. Source fingerprints and tool versions go in the manifest. Store candidate records once in intermediate JSONL; no modifications to source files.
3. Run helper tests and a small end-to-end fixture before the full build. Full build normalizes sources, excludes/deduplicates, groups, writes final shards and deterministic selections.
4. Independently validate output IDs, source accounting, content hashes, molecular/question separation, token budgets, selection membership and source/role constraints. Record limitations, exact counts and reproduction commands in README and verification JSON.

Implementation uses existing local Python/RDKit/transformers/tokenizers. Work remains in the new `sae/` directory in the current checkout. No broad refactor or changes to previous experiment results.
