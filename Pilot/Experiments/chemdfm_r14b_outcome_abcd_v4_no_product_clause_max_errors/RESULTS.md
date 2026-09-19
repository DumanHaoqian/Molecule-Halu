# ChemDFM-R-14B outcome experiment v4

## Design

- 150 paired MolEdit questions: 50 add, 50 delete, 50 substitute.
- A: direct answer; B: maximum-edit H reasoning; C: normal N reasoning; D: self-generated CoT.
- For every supplied H/N CoT, the entire `--> PRODUCT_SMILES("...")` clause was deleted. No product placeholder or literal ground-truth product remains.
- H retained all 500 available root mutations: 100 samples have 3 errors and 50 have 4 errors.
- Greedy decoding with ChemDFM-R-14B on physical GPUs 0/1 only.

## Results

| Group | Correct | Accuracy | Mean FTS |
|---|---:|---:|---:|
| A | 121/150 | 80.67% | 0.9574 |
| B | 132/150 | 88.00% | 0.9664 |
| C | 133/150 | 88.67% | 0.9694 |
| D | 119/150 | 79.33% | 0.9464 |

B vs C: B-only correct 2, C-only correct 3, exact McNemar p=1.0. Their normalized outputs are identical on 144/150 questions.

B vs A: +7.33 percentage points, B-only 16/A-only 5, exact McNemar p=0.02660.
C vs A: +8.00 percentage points, C-only 17/A-only 5, exact McNemar p=0.01690.

## Diagnosis

Removing the product clause does not make the existing H construction causally harmful. Most of its 500 mutations alter verification counts or locally inconsistent labels. Only 10/150 H records mutate `add_fragment`, and only 6/150 mutate `remove_group`. The model can ignore the noisy fields and still follow the unchanged natural-language edit and often unchanged product-determining fragment.

The 3-error versus 4-error split is confounded with subtask: all 50 four-error records are add examples. It therefore cannot be interpreted as a clean dose-response result.

The next causal dataset should corrupt every product-determining field, especially ADD_FRAGMENT/REMOVE_GROUP and anchor identity, and propagate each corruption consistently through the prose and FORMAL statements. Numeric verification errors should be secondary diagnostics rather than the main hallucination intervention.

## Integrity

- 600/600 predictions completed; 600 valid SMILES; 600 stopped on `</answer>`; 0 missing; 0 truncated; 0 atom-mapped outputs.
- Script SHA256: `420b75d308f6a3dbd8d503771777a32db63ce704198262ad66b72d5adc6c77e0`.
- Requests SHA256: `2859b304b138c5c2da42c5b827845a58eebd549b705aadad34a75a02d92a0652`.
- Predictions SHA256: `e4149cf7012d0f087be14e4eb27d4579c921a2221dd9b833e3c33cb8a4b48cd8`.
