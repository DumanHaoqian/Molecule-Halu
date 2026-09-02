# MolHalluLens Molecule Editing 幻觉数据集详细实施计划

> 版本：v0.2  
> 日期：2026-08-28  
> 当前范围：ChemCoTBench-V2 Molecule Editing Pilot  
> 目标规模：150 origins，1,200 records，train/validation/test = 800/200/200  
> LLM Provider：Poe Creator Platform；首选模型 `gpt-5.4-mini`  
> 文档状态：实施设计；本文件不表示生成代码和最终数据已经完成

---

## 0. 本版本冻结的决定

本计划以以下决定为不可变实现约束。后续代码、schema、测试和数据构建都必须显式验证这些约束。

1. Detector 的可见序列固定为：

   ```text
   source/indexed SMILES
   → instruction
   → candidate reasoning chain
   → candidate final answer
   ```

2. `gt_smiles` 不进入 detector 输入。它只作为构建期、验证期和标注期的隐藏 oracle 保存。
3. 只保留 post-token detection：token `t` 的标签与已经读入 token `t` 后的 `h_t` 对齐，不再构建或宣称 pre-token prediction。
4. 正确 process trace 被解析为 typed State DAG；扰动先发生在结构化 state/edge 上，再渲染成文本，禁止直接对整段字符串做无约束修改。
5. 四种 propagation 都是正式数据组成：`STOP/LOCAL`、`PARTIAL`、`FULL_CF`、`TERMINAL`。`TERMINAL` 与另外三类拥有相同配额和独立评测结果。
6. 每个 origin 生成四组 H/N matched pair，即 8 条 records：

   ```text
   H_LOCAL       + N_LOCAL
   H_PARTIAL     + N_PARTIAL
   H_FULL_CF     + N_FULL_CF
   H_TERMINAL    + N_TERMINAL
   ```

7. 150 origins × 8 records = 1,200 records；按 origin/leakage group 切分为 100/25/25 origins，对应 800/200/200 records。
8. 字符 span 是 canonical annotation；ChemDFM-R token mask 是 tokenizer-specific derived artifact。
9. Token label 使用多标签、正交轴设计，至少同时表达：是否幻觉、错误类型、editing subtype、root/propagated/terminal causal role。
10. Perturbator 使用面向对象的 family/subtask 继承层次；候选来源、传播、渲染、验证使用 composition/Strategy 注入，避免把所有逻辑塞入一个巨型基类。
11. 化学结构候选由“确定性规则/RDKit + 不确定性 LLM Agent”共同提出；LLM 可以发明候选结构，但确定性 validator 是唯一接受裁决者。
12. Split manifest 必须在 cross-record donor pool 和 LLM candidate generation 之前冻结；任何 donor 只能来自同一 split。
13. 所有 GPT 调用通过 Poe API；首选 Poe model ID 为 `gpt-5.4-mini`。`POE_API_KEY` 只能来自环境变量，不能使用 OpenAI 直连 key/base URL。

### 0.1 v0.2 Poe API 变更

相对 v0.1，本版本将原来的 OpenAI 直连假设改为 Poe Creator Platform：

- API base URL：`https://api.poe.com/v1`；
- primary model ID：`gpt-5.4-mini`；
- display/bot name：`GPT-5.4-Mini`；
- 认证变量：`POE_API_KEY`；
- 构建前通过 `GET /v1/models` 校验模型存在、endpoint 和 tool capabilities；
- proposal/tool agent 优先使用 Poe Responses API；保留 Poe Chat Completions 和 `fastapi_poe` adapter 作为兼容 fallback；
- 不依赖 `seed`、function `strict` 或 Chat Completions `response_format`；所有输出均本地 schema validate；
- 记录 Poe request ID、model catalog hash、token usage、point balance/usage history和重试信息。

### 0.2 对 MolHalluLens v1.5 的同步修改

当前决定取代原规划中“`h_(t-1)` 是在线检测主实验”的要求。相应地，论文的 RQ1/H1 应重写为：

> ChemDFM-R 在读入一个候选 token 后，其 residual stream 中是否出现可定位、可分类、可跨任务泛化的 token-level hallucination signal？

本 Pilot 不再声称预测“模型下一 token 是否即将产生幻觉”。如果未来重新研究 pre-generation risk，应使用模型 native decoding 和 prefix risk 定义另立实验，不与本 counterfactual teacher-forcing 数据混用。

---

## 1. 项目目标、边界与最终产物

### 1.1 目标

基于当前 150 条正确 Molecule Editing process traces，构建一个：

- tool-grounded；
- token-level；
- 有 matched faithful controls；
- 支持 root error propagation；
- 支持细粒度错误类型；
- 可直接用于 ChemDFM-R `resid_post` 特征提取；
- 难以通过 renderer 风格、固定数字、固定片段或字符串相等 shortcut 解决；
- 可扩展到 `mol_opt`、`mol_und`、`rxn_pred`；

的 counterfactual hallucination Pilot。

### 1.2 本阶段不做的事情

- 不构建 pre-token target。
- 不把 `gt_smiles` 序列化到 detector prompt。
- 不实现另外三个 task family 的具体 operators；只预留接口和空的 abstract family 类。
- 不把 LLM 自报的“valid/correct/plausible”当标签或验证证据。
- 不把 clean/corrupted 全文 string diff 直接当 hallucination mask。
- 不把 1,200 records 当成 1,200 个独立统计样本；统计独立单位仍是 origin/leakage group。
- 不用裸 SMILES 字符串相等判定分子是否一致。

### 1.3 预期产物

实施完成后至少产生：

```text
Pilot/
├── molhallulens/                         # 构建代码
├── HallucinationDataset/
│   ├── dataset_manifest.json
│   ├── split_manifest.csv
│   ├── records/
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   └── test.jsonl
│   ├── oracle/                          # 不进入 detector serializer
│   ├── state_graphs/
│   ├── tokenized/chemdfm_r/
│   ├── provenance/
│   └── reports/
└── MOLHALLULENS_MOLEDIT_HALLUCINATION_IMPLEMENTATION_PLAN.md
```

---

## 2. 当前数据源和已知数据特征

### 2.1 三类输入文件的职责

当前数据根目录：

```text
Pilot/Dataset/
```

每个 subtask 的数据职责如下：

| 来源 | 构建时用途 |
|---|---|
| `raw_benchmark_data/mol_edit/*.json` | `indexed_smiles`、instruction、`gt_smiles`、原始 metadata |
| `process_evaluation_data/mol_edit/*.json` | verified clean CoT、`parsed_reference_state`、step/natural/formal 文本 |
| `formal_templates/mol_edit/*.json` | step fields、reference fields、legacy verifier field inventory |

构建程序应按 `anonymous_sample_id` join raw/process records，不依赖数组位置。

### 2.2 subtask 名称映射

Pilot 文件名与 ChemCoTBench parser/verifier package 名不同，必须显式保存：

```text
add_pilot_origin        → source_subtask=add_v2        → normalized_subtask=add
delete_pilot_origin     → source_subtask=delete_v2     → normalized_subtask=delete
substitute_pilot_origin → source_subtask=substitute_v2 → normalized_subtask=substitute
```

不允许直接使用 `add_pilot_origin` 动态导入原始 verifier，因为仓库中不存在对应 Python package。

### 2.3 已知分布和异常

- Addition：50 条全部为纯 addition，`LEAVING="none"`；anchor 为 N 35 条、O 15 条。
- Delete：49 条是 Deprotection；`mol_edit.delete_v2.0081` 是 delete-with-replacement 异常类型，不能无条件套用 `heavy_delta=-remove_heavy`。
- Substitute：44/50 的 anchor 为 C；49/50 的 leaving group 为单一重原子；非常适合构造同元素错误 site 和 matched leaving-group confusion。
- 当前 150 条均有正确 process trace，但部分 `answer_smiles` 与 `gt_smiles` 裸字符串不同而图等价，所有等价判断必须使用 isomeric graph equivalence。
- 初步 scaffold 审计得到 146 个 unique Murcko scaffolds；存在 3 个重复 source/scaffold groups，共 7 个 origins。这些 group 必须整体进入同一个 split。

### 2.4 现有 verifier 的使用边界

现有 verifier 可作为 `LegacyVerifierAdapter`，但不能作为最终构建 gate。它主要能检查：

- index 范围和 index/element 一致性；
- fragment/group/product 是否可解析；
- heavy/ring count 和局部算术；
- process product 与 GT 的分子等价性。

它不能充分检查：

- instruction 指定的真实 edit site；
- anchor 与 remove/leaving group 的邻接关系；
- fragment attachment atom；
- 声称的 edit 是否真正生成 candidate product；
- cross-step state consistency；
- natural language 与 FORMAL/state 的一致性；
- final Answer 的完整语义。

因此必须新增严格的 graph-edit、propagation、renderer 和 token-label validators。

---

## 3. Detector 输入、序列化和 activation 对齐

### 3.1 唯一允许的 detector 输入格式

建议固定可版本化的序列化模板：

```text
<MOLECULE>
{indexed_smiles}

<INSTRUCTION>
{instruction}

<REASONING>
{candidate_reasoning_chain}

<FINAL_ANSWER>
{candidate_final_answer}
```

精确 delimiter 可以在实现前冻结，但字段顺序不能改变。

### 3.2 GT 的隔离方式

`gt_smiles`、reference DAG、clean reference state、operator、candidate provenance 都可以存在于构建 artifact，但不能进入 `DetectorPromptSerializer`。

建议从类型接口上防止泄漏：

```python
class DetectorPromptSerializer:
    def serialize(
        self,
        indexed_smiles: str,
        instruction: str,
        reasoning_chain: str,
        final_answer: str,
    ) -> SerializedDetectorInput:
        ...
```

该接口根本不接收 `gt_smiles`。

State DAG 节点使用 visibility：

```python
class Visibility(Enum):
    BUILD_ONLY = "build_only"          # GT、reference-only evidence
    PROMPT_PREFIX = "prompt_prefix"    # source、instruction
    CANDIDATE_OUTPUT = "candidate_output"
```

### 3.3 Post-token 对齐

唯一 activation-label 对齐定义：

```text
token x_t 已经进入 ChemDFM-R
→ 读取 layer l 的 resid_post h_t^(l)
→ 预测 token t 的标签 y_t
```

实现中不得 shift labels：

```python
assert activation_alignment == "post_token_h_t"
assert token_labels.shape[0] == hidden_states.shape[0]
```

### 3.4 哪些 token 参与评测

| Segment | attention | evaluation label |
|---|---:|---:|
| source/indexed SMILES | 1 | `-100` / ignore |
| instruction | 1 | `-100` / ignore |
| reasoning chain | 1 | 0 或细粒度 positive label |
| final answer | 1 | 0 或细粒度 positive label |
| special/padding token | 按模型配置 | `-100` / ignore |

这样 source/instruction 不会成为大量 trivial negative tokens。

---

## 4. Typed State DAG

### 4.1 为什么需要 DAG

同一个语义事实可能在 natural language、FORMAL 行和最终 Answer 中重复出现。文本位置不是逻辑状态，step 也不等于单一状态。

State DAG 的目标是：

1. 把每个可验证 claim 表示为 typed node；
2. 把推导、等价、计数、图编辑等关系表示为 typed edge；
3. 从一个 root mutation 精确找到合法 descendants；
4. 让 propagation 修改 state，而不是字符串；
5. 让同一 state 的多个文本 mention 共享一个语义标签来源。

### 4.2 静态 schema 与实例值分离

```python
@dataclass(frozen=True)
class StateNodeSpec:
    node_id: str
    value_type: ValueType
    step_index: int | None
    role: NodeRole
    visibility: Visibility
    mutable: bool
    comparator: ComparatorKind
    renderer_slot: str | None

@dataclass(frozen=True)
class StateEdge:
    source: str
    target: str
    relation: DependencyType

@dataclass(frozen=True)
class StateDAG:
    schema: StateSchema
    values: Mapping[str, ClaimValue]
```

每个实例 value 不应只是一个裸字符串或整数。建议统一包装：

```python
@dataclass(frozen=True)
class ClaimValue:
    raw_value: Any
    normalized_value: Any
    value_type: ValueType
    provenance: ValueProvenance       # REFERENCE / RULE / RDKIT / LLM / PROPAGATED
    locally_valid: bool | None
    oracle_match: bool | None
    confidence: float | None
    mention_ids: tuple[str, ...]
```

其中 `oracle_match` 和 build provenance 只存在于构建端，不能进入 detector serialization。一个 state 可以在 natural language 和 FORMAL 中有多个 `mention_id`；这些 mentions 共用同一个语义状态，但各自拥有独立 char span。

建议 node roles：

```text
EVIDENCE        source、instruction
INTERNAL_TRUTH  gt_smiles、oracle graph diff
PRIMARY_CLAIM   anchor、remove group、add fragment
DERIVED_CLAIM   product、counts、delta
FINAL_ANSWER    与 process product 分开的独立节点
```

建议 edge relations：

```text
DERIVED_FROM
COUNT_OF
DELTA_OF
MUST_EQUAL
MOLECULARLY_EQUIVALENT_TO
EDIT_PRODUCES
CONSTRAINED_BY_INSTRUCTION
ATTACHED_TO
REMOVED_FROM
```

节点值和关系都可能产生 hallucination。比如 anchor node 和 product node 各自可以有值，而“该 anchor 加上该 fragment 产生该 product”是 `EDIT_PRODUCES` edge claim。

### 4.3 Addition DAG

```text
source + instruction ──> anchor_idx / anchor_element / leaving
instruction          ──> add_fragment ──> fragment_heavy
source + anchor + leaving + fragment ──> product
source                ──> source_heavy / source_rings
product               ──> product_heavy / product_rings
source/product counts ──> heavy_delta / ring_delta
product               ──> final_answer
oracle_gt             ──> build-time correctness checks only
```

对应当前字段：

```text
step1_anchor_idx
step1_anchor_element
step1_leaving_smiles
step2_frag_smiles
step2_heavy_atoms
step3_product_smiles
step4_n_heavy_src / step4_n_heavy_prod / step4_heavy_delta
step5_n_rings_src / step5_n_rings_prod / step5_ring_delta
final_answer
```

### 4.4 Deletion DAG

```text
source + instruction ──> anchor / remove_group
remove_group          ──> remove_heavy
source + anchor + remove_group ──> product
source/product        ──> heavy/ring counts and deltas
product               ──> final_answer
```

Step 1 和 Step 2 中重复出现的 remove group 是同一个 semantic state 的两个 mentions；如果故意制造 cross-step contradiction，则 candidate graph 中需要显式创建两个 claim instances 并用 `MUST_EQUAL` edge 连接。

### 4.5 Substitution DAG

```text
source + instruction ──> anchor / remove_group / add_fragment
remove_group          ──> remove_heavy
add_fragment          ──> add_heavy
source + anchor + remove_group + add_fragment ──> product
remove/add counts + source/product ──> heavy delta
source/product        ──> ring counts and delta
product               ──> final_answer
```

### 4.6 Reference graph、candidate graph 和 GraphDelta

每条 variant 都应保存：

```python
reference_graph: StateDAG
candidate_graph: StateDAG
graph_delta: tuple[MutationEvent, ...]
```

```python
@dataclass(frozen=True)
class MutationEvent:
    node_or_edge_id: str
    before: ClaimValue
    after: ClaimValue
    causal_role: CausalRole
    hallucination_types: frozenset[HallucinationType]
    edit_subtypes: frozenset[EditErrorSubtype]
    operator_id: str
    root_event_id: str
```

### 4.7 EditTruth 必须由图差异复核

`parsed_reference_state` 只作为 seed。最终 `EditTruth` 应由 RDKit 对 source 和 GT 做 graph comparison/MCS/mapping，并结合 instruction 与原 trace 消除歧义：

```python
@dataclass(frozen=True)
class EditTruth:
    source_smiles: str
    gt_smiles: str
    valid_anchor_indices: tuple[int, ...]
    symmetry_equivalent_anchors: tuple[tuple[int, ...], ...]
    removed_atom_maps: frozenset[int]
    added_atoms: tuple[AtomDescriptor, ...]
    broken_bonds: tuple[BondEdit, ...]
    formed_bonds: tuple[BondEdit, ...]
    remove_fragment: FragmentSpec | None
    add_fragment: FragmentSpec | None
    source_descriptors: MoleculeDescriptors
    product_descriptors: MoleculeDescriptors
    mapping_confidence: float
```

如果映射存在多个对称等价解，保存 equivalence class，而不是任选一个并把其他等价位置标错。

---

## 5. 面向对象架构

### 5.1 Family 继承层次

仓库中四个正式 family 是：

```text
Perturbator (ABC)
├── MoleculeEditingPerturbator (ABC)       family="mol_edit"
│   ├── AdditionPerturbator                subtask="add"
│   ├── DeletionPerturbator                subtask="delete"
│   └── SubstitutionPerturbator            subtask="substitute"
├── MolecularOptimizationPerturbator       family="mol_opt"   [future]
├── MoleculeUnderstandingPerturbator       family="mol_und"   [future]
└── ReactionPredictionPerturbator           family="rxn_pred"  [future]
```

Python class 使用 `SubstitutionPerturbator`，注册键仍兼容数据中的 `substitute`。

### 5.2 继承和组合的边界

- Inheritance 只表达稳定的 family/subtask 类型和 template hooks。
- Candidate source、propagation、renderer、validator、label projector 使用 composition。
- Operator 是 concrete subtask 的成员方法，但通过 decorator registry 管理 metadata。
- Operator 只产生 root patch，禁止自行修改 downstream state 或最终文本。

### 5.3 抽象基类

```python
class Perturbator(ABC, Generic[TruthT]):
    family: ClassVar[str]

    def __init__(
        self,
        candidate_engine: CandidateEngine,
        propagator: PropagationEngine,
        renderer: TraceRenderer,
        validators: ValidatorChain,
        label_projector: LabelProjector,
    ) -> None:
        ...

    @abstractmethod
    def parse_record(self, raw: dict, process: dict) -> TaskRecord:
        ...

    @abstractmethod
    def build_reference_dag(self, record: TaskRecord) -> StateDAG:
        ...

    @abstractmethod
    def derive_truth(self, record: TaskRecord, dag: StateDAG) -> TruthT:
        ...

    @abstractmethod
    def state_schema(self) -> StateSchema:
        ...

    def perturb_one(
        self,
        record: TaskRecord,
        recipe: PerturbationRecipe,
    ) -> PerturbationResult:
        """不可被 subclass 重写的 Template Method。"""
```

固定 Template Method：

```text
ingest record
→ build/validate reference DAG
→ enumerate candidate root patches
→ select root patch
→ apply propagation
→ build Trace AST
→ render text and exact char spans
→ project token labels
→ validate artifact and bundle
→ write immutable PerturbationResult
```

### 5.4 Editing family 提供的公共能力

```python
class MoleculeEditingPerturbator(Perturbator[EditTruth], ABC):
    def derive_graph_diff(...) -> MolecularGraphDiff: ...
    def apply_edit_action(...) -> Chem.Mol: ...
    def enumerate_attachment_sites(...) -> list[AtomSite]: ...
    def enumerate_removable_groups(...) -> list[FragmentSpec]: ...
    def compare_molecules(...) -> MoleculeComparison: ...
    def validate_edit_family(...) -> ValidationReport: ...

    @abstractmethod
    def expected_edit_kind(self) -> EditKind: ...
```

### 5.5 Operator 成员方法和 registry

```python
@operator(
    operator_id="mol_edit.add.alternate_anchor",
    root_fields={"step1.anchor_idx", "step1.anchor_element"},
    policies={STOP, PARTIAL, FULL},
    candidate_sources={RULE, RDKIT, LLM, HYBRID},
    default_types={CONTRADICTION, REASONING_ERROR},
)
def perturb_alternate_anchor(
    self,
    context: PerturbationContext,
) -> CandidatePool:
    ...
```

Decorator 只注册 `OperatorSpec`；方法返回 `CandidatePatch` 候选，不能直接改字符串。

```python
@dataclass(frozen=True)
class OperatorSpec:
    operator_id: str
    root_fields: frozenset[str]
    supported_policies: frozenset[PropagationPolicy]
    supported_sources: frozenset[CandidateSourceType]
    hallucination_types: frozenset[HallucinationType]
    diagnostic_only: bool = False

@dataclass(frozen=True)
class CandidatePatch:
    root_node_id: str
    old_value: ClaimValue
    new_value: ClaimValue
    edit_action: EditAction | None
    source: CandidateSourceType
    metadata: Mapping[str, Any]
```

### 5.6 Factory 和 subtask normalization

```python
@PerturbatorRegistry.register("mol_edit", "add")
class AdditionPerturbator(MoleculeEditingPerturbator):
    ...

perturbator = PerturbatorFactory.from_record(record)
```

`SubtaskNormalizer` 将 `add_v2`、`add_pilot_origin` 等归一为 `add`，避免主程序出现任务相关的 `if/elif`。

### 5.7 关键 domain objects

```python
TaskRecord
EditTruth
StateDAG
OperatorSpec
CandidatePatch
PerturbationRecipe
MutationEvent
TraceDocument
RenderedExample
CharAnnotation
TokenLabelSet
ValidationIssue
ValidationReport
PerturbationResult
OriginBundle
BuildProvenance
```

所有核心对象尽量使用 frozen dataclass；需要改变 state 时返回新对象，便于审计 root 和 propagation。

---

## 6. 四类 propagation 和 matched bundle

### 6.1 四种 policy 的精确定义

```python
class PropagationPolicy(Enum):
    STOP = "local"
    PARTIAL = "partial"
    FULL = "full_cf"
    TERMINAL = "terminal"
```

#### STOP / LOCAL

- 只改变一个 root node/edge。
- 所有 downstream state value 保留 reference。
- 如果 root 与 downstream 产生依赖冲突，记录 violated edge；只有在文本中显式表达该关系时才给关系 span 标签。
- 正确 downstream product/Answer 不因“位于错误 root 之后”而自动被标错。

典型 phenotype：reasoning wrong，Answer correct。

#### PARTIAL

- 改变 root；
- 在 root descendants 中沿合法连通子图传播；
- 到 recipe 指定的 cut set 停止；
- 不能随机挑一组互不相连的字段。

```python
partial_cut_nodes: frozenset[NodeId]
```

需要区分：

- `PROPAGATED_FALSE`：相对 candidate parent state 也计算错误；
- `PROPAGATED_CONDITIONAL`：在错误 root 世界里局部计算正确，但属于 off-task branch。

#### FULL_CF

- 改变一个结构性 root；
- 按 DAG 拓扑顺序重算所有可达 descendants；
- product 必须通过真实 RDKit graph edit 得到；
- heavy/ring/delta 全部由 candidate product 重算；
- final Answer 与 candidate product 分子等价；
- 整条链内部自洽，但不满足原 instruction/oracle truth。

#### TERMINAL

- reasoning graph 与 reference 保持一致；
- 只允许 `final_answer` 节点作为 root 变化；
- final Answer 必须满足该 terminal operator 的目标，例如 valid-but-wrong、stereo/connectivity near-miss 或明确 diagnostic invalidity；
- `TERMINAL` 是完整、等额的数据类别，不作为附属样本处理。

### 6.2 PropagationEngine

```python
class DerivationRule(Protocol):
    output_node: str
    input_nodes: tuple[str, ...]

    def derive(self, state: StateDAG, context: PerturbationContext) -> ClaimValue:
        ...

class PropagationEngine:
    def propagate(
        self,
        reference: StateDAG,
        root_patch: CandidatePatch,
        recipe: PerturbationRecipe,
    ) -> PropagatedState:
        ...
```

Changed-set invariants：

```text
STOP_CHANGED_SET     == root patch fields
PARTIAL_CHANGED_SET  == root + 合法连通下游子图，且小于完整 closure
FULL_CHANGED_SET     == root + 所有可推导 descendants
TERMINAL_CHANGED_SET == {final_answer}
```

### 6.3 每个 origin 的 8-record bundle

```python
class MatchedBundleBuilder:
    def build(
        self,
        origin: TaskRecord,
        policies=(STOP, PARTIAL, FULL, TERMINAL),
    ) -> OriginBundle:
        ...
```

| Pair | H | matched N |
|---|---|---|
| LOCAL | 一个 root claim 错，downstream reference 保持 | 同 step、同 field、同 style、相同重写量，事实正确 |
| PARTIAL | root 加部分传播 | 相同 dependency path 被 faithful 地重渲染 |
| FULL_CF | 自洽错误分支 | 全 downstream faithful 重算/重渲染 |
| TERMINAL | reasoning 正确，Answer 错 | reasoning 正确，Answer 使用另一份等价 SMILES serialization |

每组 H/N 必须共享：

```text
origin_id
split
input_view_id
target step/field
renderer backend/style bucket
rewrite budget
candidate difficulty bucket
```

N control 不是简单复制同一 clean trace四次，而是与对应 H 在 surface process 上 matched 的 faithful variant。

### 6.4 数量不变量

```text
每个 origin：4 H + 4 N = 8 records
每个 subtask：50 origins × 8 = 400 records
全体：150 origins × 8 = 1,200 records
H/N：600/600
每种 propagation：150 H + 150 N = 300 records
```

---

## 7. Molecule Editing perturbation operators

### 7.1 通用候选难度要求

结构性 hard candidate 应尽量满足：

- sanitized、价态合法；
- 与 reference 不等价；
- atom/fragment/bond edit 可实际执行；
- anchor 尽量同 element、aromaticity、degree、hybridization；
- 排除 automorphism/symmetry-equivalent site；
- fragment 尽量匹配 attachment element、heavy count、ring count、formal charge、heteroatom vector；
- product 尽量高结构相似，并匹配 heavy/ring count；
- 数字值来自真实 donor/经验分布，不使用固定 `±1` 作为主模式。

Invalid SMILES、明显越界 index、单纯格式错误可以保留为独立 error type，但不得支配 core。建议最终 H 中至少 80% 为 chemically valid structural/relational errors。

### 7.2 AdditionPerturbator 成员 operators

```python
perturb_alternate_anchor_same_element()
perturb_neighborhood_matched_anchor()
perturb_fragment_bucket_swap()
perturb_fragment_attachment_atom()
perturb_attachment_bond_order()
perturb_valid_wrong_site_product()
perturb_valid_regioisomer_product()
perturb_heavy_count_claim()
perturb_ring_count_claim()
perturb_internal_relation_claim()
perturb_terminal_answer()
```

实现重点：

- 当前 Add 的 `LEAVING` 全是 `none`。`none→Br/Cl` 不能成为核心高频 operator，否则标签与词值强相关。
- Wrong anchor：RDKit 先枚举同元素非对称等价 N/O；LLM 再按局部环境和 instruction plausibility 排序。
- Wrong fragment：从同 split donor bucket 或 LLM 提议同类近邻片段，经 RDKit 验证。
- FULL_CF：在错误 site 接入正确 fragment，或在正确 site 接入 matched wrong fragment，真正生成 sanitized product。
- Attachment orientation 必须保存 `fragment_attachment_atom`；单独 canonical fragment SMILES 不足以表达该语义。

### 7.3 DeletionPerturbator 成员 operators

```python
perturb_wrong_group_occurrence()
perturb_wrong_adjacent_group()
perturb_group_boundary_contract()
perturb_group_boundary_expand()
perturb_partial_deletion()
perturb_over_deletion()
perturb_matched_remove_group()
perturb_alternative_deprotection_product()
perturb_cross_step_group_identity()
perturb_heavy_count_claim()
perturb_ring_count_claim()
perturb_terminal_answer()
```

实现重点：

- 优先利用同一分子内多个保护基/多个相似 occurrence，构造 chemically valid wrong deprotection。
- Group boundary operator 必须由 source graph 的连接子图定义，不允许仅修改一段 fragment 字符串。
- FULL_CF 需要真实切 bond、补充合理 implicit H/valence，并验证结果不等价于 GT。
- `delete_v2.0081` 必须标为 `operation_subtype=delete_with_replacement`，仅启用适用 operators；不得使用普通 deletion delta rule。

### 7.4 SubstitutionPerturbator 成员 operators

```python
perturb_alternate_substitution_site()
perturb_wrong_leaving_occurrence()
perturb_incoming_fragment_bucket_swap()
perturb_fragment_attachment_atom()
perturb_attachment_bond_order()
perturb_leaving_group_swap()
perturb_partial_substitution()
perturb_valid_wrong_regioisomer()
perturb_add_remove_role_claim()
perturb_heavy_count_claim()
perturb_ring_count_claim()
perturb_terminal_answer()
```

实现重点：

- 44/50 anchor 为 C，同元素错误 site 是主要 hard operator。
- Halogen swap 必须循环/交叉平衡，使 F/Cl/Br/I 均在其他上下文中作为正确值出现。
- Incoming fragment 采用同 split donor bucket 或 LLM structure proposal。
- Fragment attachment atom 和 source anchor 是两个独立维度，必须分别保存和验证。
- FULL_CF 使用错误 regio/site、错误 incoming fragment 或错误 attachment orientation 构造有效产品，随后程序重算所有 downstream state。

### 7.5 Operator-policy compatibility

配置文件必须声明哪些 operator 支持哪些 policy。例如：

| Operator family | STOP | PARTIAL | FULL_CF | TERMINAL |
|---|---:|---:|---:|---:|
| wrong anchor/site | ✓ | ✓ | ✓ | – |
| wrong fragment/group | ✓ | ✓ | ✓ | – |
| attachment/bond edit | ✓ | ✓ | ✓ | – |
| numeric/count claim | ✓ | ✓ | – | – |
| NL–FORMAL/internal relation | ✓ | ✓ | – | – |
| final answer identity | – | – | – | ✓ |

Scheduler 只能从兼容集合抽取；不能在候选失败后静默改变 propagation phenotype。

### 7.6 初始 operator/root-subtype 配额

四种 propagation 的数量是硬配额；具体 operator 是第二级配额。每个 subtask、每种 H policy 有 50 条，建议初始分配为：

| Policy | Root/operator family | 每 50 条目标数 |
|---|---|---:|
| LOCAL | anchor/site grounding | 15 |
| LOCAL | group/fragment identity | 15 |
| LOCAL | attachment/internal relation | 10 |
| LOCAL | heavy/ring/count claim | 10 |
| PARTIAL | entity error + 部分依赖传播 | 18 |
| PARTIAL | product/dependency cross-step propagation | 17 |
| PARTIAL | count/ring propagation | 10 |
| PARTIAL | NL–FORMAL/internal relation | 5 |
| FULL_CF | valid wrong site/occurrence/regioisomer | 18 |
| FULL_CF | valid wrong group/fragment | 15 |
| FULL_CF | wrong attachment atom/bond | 10 |
| FULL_CF | alternate valid edit/no-op/边界操作 | 7 |
| TERMINAL | valid high-similarity wrong molecule | 35 |
| TERMINAL | stereo/connectivity/regio near-miss | 10 |
| TERMINAL | invalid/format/truncation diagnostic | 5 |

这是跨 subtask 的 root-family 目标。每个 concrete perturbator 再把它映射到适用 member methods，例如 Delete 的 “site” 对应 wrong occurrence/alternative deprotection，Substitute 的 “group/fragment” 对应 leaving/incoming fragment。

Quota scheduler 要求：

- 先按 policy 锁定 50 条，再按 operator family 分配；
- operator 不适用于某 origin 时，在同一 family 内换 operator；
- family 完全不可行时才由预先声明的 fallback matrix 调整，并记录 quota deviation；
- 不根据 LLM 是否容易成功而改变标签比例；
- headline edit subtype 在 validation/test 中尽量各覆盖至少 5 个独立 origins，否则只保留 annotation，不作为独立 headline metric。

---

## 8. 确定性规则/RDKit + 不确定性 LLM Agent

### 8.1 三层职责分离

```text
Deterministic Orchestrator
  规定 split、phenotype、operator、约束和 quota
          ↓
Hybrid Candidate Engine
  Rule/RDKit enumeration + LLM Agent structure invention
          ↓
Deterministic Validator / Propagation / State Builder
  唯一 accept/reject；重算 downstream state
          ↓
Label-blind Renderer
  只表达已经锁定的 state
          ↓
Final detector text + char spans + token labels + private provenance
```

### 8.2 CandidateEngine

```text
CandidateEngine
├── RuleCandidateSource
├── RDKitCandidateSource
├── LLMAgentCandidateSource
└── HybridCandidateSelector
```

统一接口：

```python
class CandidateSource(Protocol):
    def propose(self, request: CandidateRequest) -> Sequence[CandidatePatch]:
        ...

class HybridCandidateEngine:
    def build_pool(self, request: CandidateRequest) -> CandidatePool:
        """合并、canonicalize、去重、验证、匹配难度并排序。"""
```

建议总体 candidate generation mix：

```text
40% deterministic rule/RDKit candidates
40% LLM-proposed structure candidates + deterministic validation
20% hybrid graph-edit candidates：规则限定空间，LLM 发明/选择，RDKit 执行
```

这是初始目标，不是硬编码常数；最终 manifest 报告每类真实占比。至少一半 structural H 应让 LLM 实质参与候选结构或位点选择，而不只是润色文本。

### 8.3 LLM Proposal Agent 可以做什么

- 提出 plausible alternate anchor/group/fragment；
- 发明 matched near-neighbor fragment；
- 提议错误 regioisomer、attachment orientation、near-miss product；
- 在 RDKit 枚举候选中选择最符合 instruction 语境、最容易被误信的候选；
- 根据 validator reject codes 修正候选。

LLM 不得决定：

- hallucination label；
- candidate 是否通过；
- downstream 数字；
- token span；
- root/propagated role；
- split 或 operator quota。

### 8.4 Agent tools

只暴露无副作用、结构化、可缓存的 chemistry tools：

```text
inspect_atoms
enumerate_alternate_anchors
analyze_smiles
find_group_at_anchor
enumerate_removable_groups
simulate_edit
compute_descriptors
compare_molecules
check_candidate_signature
```

示例：

```json
{
  "tool": "simulate_edit",
  "arguments": {
    "family": "substitute",
    "source_smiles": "...",
    "anchor_idx": 23,
    "remove_group_smiles": "Br",
    "add_fragment_smiles": "N1CCCC1",
    "fragment_attachment_atom": 0,
    "bond_type": "SINGLE"
  }
}
```

工具返回所有 sanitized products、graph diff、descriptors 和 reject reasons，不相信 LLM 手工拼接的长 SMILES。

### 8.5 Proposal request contract

Orchestrator 先产生确定性请求：

```json
{
  "schema_version": "1.0",
  "request_id": "...",
  "origin_id": "mol_edit.substitute_v2.0216",
  "operator_id": "mol_edit.substitute.fragment_bucket_swap",
  "propagation": "FULL_CF",
  "root_fields": ["step1.add_fragment"],
  "candidate_source_mode": "HYBRID",
  "constraints": {
    "sanitized": true,
    "not_equivalent_to_reference": true,
    "same_attachment_element": true,
    "match_heavy_count": true,
    "match_ring_count": true
  },
  "derived_seed": 1839483
}
```

`derived_seed`：

```text
SHA256(global_seed + dataset_version + origin_id + operator_id + policy + variant_index)
```

不要依赖 Python 内置 hash。

### 8.6 Proposal response contract

LLM 优先通过 Poe Responses `text.format` 请求结构化输出并返回 3–5 个候选；由于 Poe compatibility 不保证所有 schema/strict 参数都被下游模型严格执行，以下对象仍必须通过本地 Pydantic/JSON Schema 校验：

```json
{
  "proposal_version": "1.0",
  "request_id": "...",
  "candidates": [
    {
      "candidate_id": "c1",
      "root_field": "step1.add_fragment",
      "replacement": {
        "smiles": "N1CCCC1",
        "attachment_atom": 0
      },
      "bond_edits": [],
      "minimal_surface_realization": "a pyrrolidin-1-yl group",
      "plausibility_reason": "Same fragment class and similar local size."
    }
  ],
  "abstain_reason": null
}
```

模型不输出 `accepted=true` 或最终标签。

### 8.7 Poe provider 和 `gpt-5.4-mini`

所有 LLM 请求通过 Poe Creator Platform。2026-08-28 对 Poe 公共模型目录的校验结果为：

```text
API model id:       gpt-5.4-mini
Display/bot name:   GPT-5.4-Mini
Owner:              OpenAI
Supported features: tools, web_search
Supported endpoints:
  /v1/responses
  /v1/chat/completions
  /v1/messages
Context length:     400,000
Max output tokens:  128,000
```

模型目录是运行时真源：[Poe `GET /v1/models`](https://creator.poe.com/api-reference/listModels)。构建启动时必须重新解析目录，而不是只相信文档中的展示名称：

```python
catalog = poe_model_registry.refresh()
model = catalog.require("gpt-5.4-mini")

assert "/v1/responses" in model.supported_endpoints
assert "tools" in model.supported_features
```

如果模型不存在、endpoint/tool capability 发生变化，build 应 fail closed。不得静默换用其他 bot，否则不同模型的语言风格和结构候选分布会污染数据版本。

当前 Poe catalog 没有为 `gpt-5.4-mini` 暴露可选择的上游 snapshot ID。因此 provenance 保存：

```text
requested_model_id
response.model
model_catalog_entry
model_catalog_entry_sha256
catalog_fetched_at
```

不能在论文或数据卡中声称已经 pin 住 OpenAI upstream snapshot；正式复现依赖冻结的 Poe raw response cache。

### 8.8 Poe client/transport 分层

推荐 provider-neutral interface：

```python
class LLMClient(Protocol):
    def propose(self, request: ProposalRequest) -> RawLLMResult: ...
    def render(self, request: RenderRequest) -> RawLLMResult: ...

class PoeResponsesClient(LLMClient): ...       # primary
class PoeChatCompletionsClient(LLMClient): ... # tool-loop compatibility fallback
class FastApiPoeTextClient(LLMClient): ...     # smoke/simple text fallback
```

#### Primary：Poe Responses API

Poe 的 `/v1/responses` 支持 tools、reasoning、streaming 和 `text.format` JSON schema，适合作为 proposal agent 主路径：[Poe Create response](https://creator.poe.com/api-reference/createResponse)。可以继续使用 OpenAI Python SDK，但 base URL 和 key 都属于 Poe，请求不经过 OpenAI 直连账户：

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["POE_API_KEY"],
    base_url="https://api.poe.com/v1",
)

response = client.responses.create(
    model="gpt-5.4-mini",
    instructions=PROPOSAL_SYSTEM_PROMPT,
    input=proposal_input,
    tools=CHEMISTRY_TOOLS,
    tool_choice="auto",
    parallel_tool_calls=False,
    reasoning={"effort": "medium"},
    text={"format": PROPOSAL_JSON_SCHEMA},
)
```

`web_search_preview` 不用于本数据构建。Agent 只能访问本项目提供的只读 chemistry tools，防止外部搜索内容造成不可复现 evidence。

#### Poe Chat Completions fallback

如果某次 Poe Responses tool loop 与目标 bot 存在兼容问题，允许退回：

```python
client.chat.completions.create(
    model="gpt-5.4-mini",
    messages=messages,
    tools=CHEMISTRY_TOOLS,
    tool_choice="auto",
    parallel_tool_calls=False,
)
```

客户端显式执行 tool call，再把 tool result 加入 messages；最多 3 agent turns、每个 proposal 最多 6 次 chemistry tool calls。Transport fallback 不得改变 model、operator、constraints 或 phenotype，并必须写入 provenance。

#### `fastapi_poe` adapter

用户给出的 `fastapi_poe` 调用可用于 API key/model smoke test，以及不需要 tool loop 的 simple renderer fallback：

```python
import fastapi_poe as fp
import os

message = fp.ProtocolMessage(role="user", content=render_prompt)

for partial in fp.get_bot_response_sync(
    messages=[message],
    bot_name="GPT-5.4-Mini",
    api_key=os.environ["POE_API_KEY"],
):
    handle_partial(partial)
```

OpenAI-compatible endpoint 使用 catalog `id="gpt-5.4-mini"`；`fastapi_poe` 使用 Poe bot/display name。两者由 `PoeModelRegistry` 显式映射，禁止在业务代码中到处混用大小写字符串。

### 8.9 Poe compatibility 防线

用户提供的 Poe 文档明确指出 compatibility 层存在 best-effort/ignored parameters。因此实现必须遵守：

1. Poe Responses 的 Structured Outputs 使用 `text.format`；不能把 Chat Completions 的 `response_format` 当约束，因为该字段可能被忽略。
2. Function calling 的 `strict` 可能被忽略，tool arguments 不能直接执行，必须先经过 Pydantic/JSON Schema 校验和 allow-list dispatch。
3. `seed` 会被忽略，不能用于 API 级逐字复现；本地 derived seed 只控制 scheduler、candidate ordering 和 deterministic fallback。
4. `metadata`、`service_tier` 等字段不能作为构建正确性依赖；provenance 全部由本地 ledger 写入。
5. 某些 unsupported fields 可能被静默忽略；构建前需要 capability probe，不能仅根据 HTTP 200 判断功能可用。
6. 即使 `text.format` 返回了合法 JSON，也必须再次做本地 schema、semantic slot、SMILES 和 graph validation。
7. Tool calls 顺序影响 agent state，因此设置 `parallel_tool_calls=False`，由本地 orchestrator 串行执行并计数。

建议启动时对 `gpt-5.4-mini` 运行三个 capability smoke tests：

```text
plain response
JSON-schema response
single local function tool call + tool-result continuation
```

任何一项失败，都不能直接开始 1,200-record build。

### 8.10 Validate–retry–fallback

```text
Attempt 1：LLM 返回 3–5 candidates → 全部确定性验证
Attempt 2：只反馈 reject codes 和必要 RDKit facts → 禁止重复候选
Attempt 3：最后一次修正

仍失败：
  → 使用同一 operator/phenotype 允许的 deterministic fallback
  → fallback 仍失败则 recipe 失败并由 quota scheduler backfill
```

绝不允许为了凑数而：

- 放宽 sanitize/valence；
- 接受 reference-equivalent H；
- 把 FULL 静默改成 STOP；
- 改变 Answer correctness；
- 让 renderer 修补化学 state。

### 8.11 LLM renderer 必须 label-blind

Proposal Agent 可以看到 clean reference、GT、operator 和 verifier reject codes；Renderer 不可以。

Renderer 只接收：

```json
{
  "step_name": "ADD_FRAGMENT_SIZE",
  "locked_slots": {
    "fragment_smiles": "N1CCCC1",
    "heavy_atoms": 5
  },
  "style_id": "style_07",
  "original_style_excerpt": "..."
}
```

推荐让 renderer 返回带 placeholder 的模板，而不是完整事实文本：

```text
The incoming {{fragment_smiles}} fragment contains
{{heavy_atoms}} heavy atoms.
```

程序填充 placeholder 时直接生成 char offsets。FORMAL 行、数字和 Answer 均由程序确定性渲染。

### 8.12 Poe rate limit、错误和 point budget

Poe 当前文档给出的 request-based rate limit 为 500 RPM。实现建议：

```text
configured ceiling: 450 RPM
max concurrency:     8（初始值，按延迟压测调整）
retry base delay:    250 ms
retry strategy:      exponential backoff + jitter
429/503:             优先遵守 Retry-After
```

错误分类：

| HTTP | 处理 |
|---:|---|
| 400 | request/schema bug；不重试，记录 fatal configuration issue |
| 401 | `POE_API_KEY` 无效；立即停止 |
| 402 | points 不足；停止新请求，保留 cache/checkpoint |
| 403 | permission/moderation；人工检查，不盲重试 |
| 404 | model ID/endpoint 错；刷新 model catalog 后 fail closed |
| 408 | 可重试 |
| 429 | 按 `Retry-After` 重试 |
| 500/502/503/529 | transient/provider/overload；有限重试 |

每次 full build 前后检查：

- [`GET /usage/current_balance`](https://creator.poe.com/api-reference/getCurrentBalance)；
- [`GET /usage/points_history`](https://creator.poe.com/api-reference/getPointsHistory)。

本地 `PoeUsageLedger` 保存：

```text
pre_build_point_balance
post_build_point_balance
query_id / response_id / x-request-id
model id
prompt/output token usage
cost_points（从 points history 对账）
attempt/retry count
```

Usage API 的 point 历史只保留有限时间，因此正式 release 前必须导出到 private provenance；不保存 `POE_API_KEY`。

### 8.13 Cache、审计和可复现性

API sampling 不能仅靠 temperature/seed 保证逐字重现。正式可复现性依靠：

- 固定 Poe model ID，并保存当时的 model catalog entry/hash；
- canonical request hash；
- 完整 request/response/tool transcript；
- immutable response cache；
- deterministic validator；
- accepted artifact content hash。

```text
cache/
├── proposals/<request_hash>.json
├── tool_runs/<tool_run_hash>.json
├── renders/<render_hash>.json
└── accepted/<artifact_hash>.json
```

Cache key 至少包含：

```text
source_record_sha256
canonical PerturbationRequest
Poe model ID + model catalog entry hash
prompt/schema hash
tool schema hash
operator version
attempt index
```

Cache/provenance 还要保存：

```text
provider = poe
base_url = https://api.poe.com/v1
transport = responses | chat_completions | fastapi_poe
requested_model_id = gpt-5.4-mini
response.model
response.id / query_id / x-request-id
raw request/response
tool transcript
token usage / cost_points
```

`POE_API_KEY`、Authorization header 和任何原始 secret 不得写入数据、日志或 Git。

---

## 9. Token-level hallucination taxonomy

### 9.1 不使用单一互斥 class

同一个 token 可能同时是：

- 与 instruction/reference state 矛盾；
- reasoning error；
- wrong anchor；
- root error。

因此 canonical label 必须是多个正交 axes，而不是单个 `class_id`。

### 9.2 通用 semantic type axis

沿用 MolHalluLens taxonomy：

| ID | Type | 说明 |
|---:|---|---|
| 0 | `CONTRADICTION` | 与 source、instruction、内部显式 claim 或 verified oracle state 矛盾 |
| 1 | `UNSUPPORTED` | 输入和可执行证据无法支持的新增事实 |
| 2 | `REASONING_ERROR` | 中间结构、算术、因果或图编辑推理错误 |
| 3 | `INVALID_CHEMISTRY` | 非法 SMILES、价态或不可执行结构 |
| 4 | `CONSTRAINT_VIOLATION` | 化学结构有效，但不满足 instruction |
| 5 | `FORMAT_ERROR` | FORMAL/schema/单位/Answer 格式错误 |
| 6 | `OMISSION` | 缺失必要 claim；通常无直接 token span |
| 7 | `UNVERIFIABLE` | 当前 evidence/tool 不足，不能强制判真伪 |

Canonical tensor：

```text
semantic_type_mask: [T, 8] multi-hot
```

同时派生两个二分类口径：

```text
hallucination_core_mask = CONTRADICTION OR UNSUPPORTED
error_any_mask = OR of all adjudicated positive types except UNVERIFIABLE
```

这样既保持严格 hallucination 定义，又允许训练更宽的 molecular error detector。

### 9.3 Editing-specific subtype axis

| Code | Label |
|---|---|
| E01 | `ANCHOR_GROUNDING` |
| E02 | `REMOVE_OR_LEAVING_GROUP_IDENTIFICATION` |
| E03 | `ADD_FRAGMENT_IDENTIFICATION` |
| E04 | `ATTACHMENT_OR_BOND_EDIT` |
| E05 | `PRODUCT_CONSTRUCTION` |
| E06 | `HEAVY_ATOM_COUNT` |
| E07 | `HEAVY_ATOM_ARITHMETIC` |
| E08 | `RING_COUNT` |
| E09 | `RING_ARITHMETIC` |
| E10 | `CHEMICAL_VALIDITY` |
| E11 | `INSTRUCTION_CONSTRAINT` |
| E12 | `FINAL_ANSWER_IDENTITY` |
| E13 | `INTERNAL_INCONSISTENCY` |
| E14 | `FORMAT_SCHEMA` |
| E15 | `UNSUPPORTED_NATURAL_CLAIM` |

更具体的 slot 单独保存：

```text
anchor.idx
anchor.element
remove_group.smiles
add_fragment.smiles
attachment.source_atom
attachment.fragment_atom
bond_edit
product.smiles
remove_heavy
add_heavy
source_heavy
product_heavy
heavy_delta
source_rings
product_rings
ring_delta
final_answer.smiles
```

### 9.4 Causal-role axis

```text
ROOT
PROPAGATED_FALSE
PROPAGATED_CONDITIONAL
TERMINAL
```

- `ROOT`：本 variant 唯一独立采样的初始错误。
- `PROPAGATED_FALSE`：由 root 导致，且相对 candidate parent state 也不成立。
- `PROPAGATED_CONDITIONAL`：在错误 candidate state 下局部计算正确，但属于 off-task branch。
- `TERMINAL`：reasoning 正确，final answer 是错误 root。

派生 masks：

```text
root_error_mask
propagated_error_mask
terminal_error_mask
local_falsehood_mask
off_task_branch_mask
```

例如 FULL_CF 中由错误 product 正确计算出的 heavy count：

```text
local_falsehood_mask = 0
off_task_branch_mask = 1
causal_role = PROPAGATED_CONDITIONAL
```

### 9.5 Evidence-relation axis

每个 positive span 可多选：

```text
CONTRADICTS_SOURCE
CONTRADICTS_INSTRUCTION
CONTRADICTS_REFERENCE_STATE
UNSUPPORTED_BY_EVIDENCE
INTERNAL_INCONSISTENCY
```

注意：reference state/GT 可参与离线标注，但不进入 detector text。

### 9.6 Omission

Omission 没有自然 token span，不能随意把前一个 token 标为 omission。Pilot 主 token-localization 集合建议不主动构造纯 omission；如必须保留：

- 只保存 trace/step-level label；或
- 单独定义 `boundary/eos_label` diagnostic protocol；
- 不插入固定 `[MISSING]` token，避免 shortcut。

---

## 10. Char span、Trace AST 和 token projection

### 10.1 Trace AST

```python
@dataclass(frozen=True)
class TraceDocument:
    steps: tuple[StepDocument, ...]
    answer: AnswerDocument

@dataclass(frozen=True)
class RenderedExample:
    detector_text: str
    segment_spans: Mapping[str, CharSpan]
    node_mentions: Mapping[str, tuple[CharSpan, ...]]
    edge_mentions: Mapping[str, tuple[CharSpan, ...]]
```

State 到文本的每次 mention 都在渲染时产生 offset。禁止渲染完成后使用 `text.find(value)` 猜位置，因为同一数字、fragment 或 SMILES 可能重复出现。

### 10.2 两级 span

每个 annotation 同时保存：

- `literal_span`：最小事实值，例如 `21`、`Br`、`+3`；
- `claim_span`：完整原子命题，例如 “the attachment atom is N21”。

主 token mask 默认使用 literal span；span-level evaluation 同时报告 literal 和 claim 两套结果。

错误 SMILES 的可靠最小语义单位通常是整个 SMILES literal，不是与 GT 的字符 diff。可选保存 graph-aligned atom/bond localization 和 `localization_confidence`，但不能以不稳定的字符串差异替代字段级标签。

### 10.3 Char annotation

```python
@dataclass(frozen=True)
class CharAnnotation:
    span_id: str
    component: str
    step_index: int | None
    state_or_edge_id: str
    literal_start: int
    literal_end: int
    claim_start: int
    claim_end: int
    semantic_types: frozenset[HallucinationType]
    edit_subtypes: frozenset[EditErrorSubtype]
    evidence_relations: frozenset[EvidenceRelation]
    causal_role: CausalRole
    root_span_id: str
```

所有区间使用半开 `[start, end)`。

### 10.4 Char-to-token projection

```python
overlap = max(
    0,
    min(token_end, span_end) - max(token_start, span_start),
)
token_positive = overlap > 0
```

同时保存：

```text
error_char_fraction
boundary_ambiguous_mask
```

主 lenient metric 使用 any-overlap；strict metric 可忽略 boundary-ambiguous token。

必须保存 tokenizer fingerprint：

```text
tokenizer_name
tokenizer_revision
tokenizer_vocab_hash
special-token config
normalization config
serialized_text_sha256
offset_mapping
```

如果 tokenizer 不支持可靠 offsets，应使用对应 fast tokenizer/backend tokenizer；禁止逐 token decode 后搜索定位。

### 10.5 TokenLabelSet

```python
@dataclass(frozen=True)
class TokenLabelSet:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    evaluation_mask: tuple[int, ...]
    hallucination_core_mask: tuple[int, ...]
    error_any_mask: tuple[int, ...]
    semantic_type_masks: Mapping[HallucinationType, tuple[int, ...]]
    edit_subtype_masks: Mapping[EditErrorSubtype, tuple[int, ...]]
    causal_role_masks: Mapping[CausalRole, tuple[int, ...]]
    local_falsehood_mask: tuple[int, ...]
    off_task_branch_mask: tuple[int, ...]
    reasoning_mask: tuple[int, ...]
    answer_mask: tuple[int, ...]
```

N controls 的 positive masks 全部为 0，但应保存 `matched_target_span`，用于检查 detector 是否只因为某个 field/step 出现就报错。

---

## 11. 1200 records 的 split 方案

### 11.1 精确配额

| Split | Origins | Records | H | N | 每种 H/N variant |
|---|---:|---:|---:|---:|---:|
| train | 100 | 800 | 400 | 400 | 100 |
| validation | 25 | 200 | 100 | 100 | 25 |
| test | 25 | 200 | 100 | 100 | 25 |

Subtask origin 配额：

| Subtask | train | validation | test | total |
|---|---:|---:|---:|---:|
| addition | 34 | 8 | 8 | 50 |
| deletion | 33 | 9 | 8 | 50 |
| substitution | 33 | 8 | 9 | 50 |
| total | 100 | 25 | 25 | 150 |

对应 records：

| Subtask | train | validation | test |
|---|---:|---:|---:|
| addition | 272 | 64 | 64 |
| deletion | 264 | 72 | 64 |
| substitution | 264 | 64 | 72 |

### 11.2 Leakage group

不能只按 `origin_id` 分组。使用 union-find 合并：

- 同 origin 的全部 variants；
- 去 atom map 后 canonical source 相同的 origins；
- canonical GT 相同的 origins；
- 相同/近重复 Murcko scaffold。

Donor relations 在 split 冻结后才会产生，因此不用于反向决定 split；候选引擎必须从类型上限制 donor 只能来自当前 split。若导入已有 donor artifact，则发现 cross-split donor edge 时直接拒绝该 artifact，而不是重排已经冻结的 split。

当前必须保持同 split 的已知 groups：

```text
mol_edit.add_v2.0279
mol_edit.add_v2.0235

mol_edit.substitute_v2.0270
mol_edit.substitute_v2.0248

mol_edit.substitute_v2.0283
mol_edit.substitute_v2.0136
mol_edit.substitute_v2.0134
```

### 11.3 正确执行顺序

```text
读取和校验 150 origins
→ 计算 canonical source/GT/scaffold/leakage groups
→ group-stratified split
→ 冻结 split_manifest.csv
→ 分 split 建 donor/candidate pools
→ 每 origin 生成完整 8-record bundle
```

禁止 test origin 的 fragment/product 被用作 train decoy。

### 11.4 分层特征

在不做高维笛卡尔 strata 的前提下，平衡：

```text
subtask
rxn_cls
anchor_element
source heavy/ring quantile
heavy/ring delta bin
fragment size bin
mol_complexity quantile
tanimoto quantile
operator availability flags
```

推荐使用 MILP 或 constrained iterative stratification。Seed 从 `SHA256(dataset_version)` 派生，split 输出写入 immutable manifest。

硬约束优先级：

1. leakage group 不拆；
2. origins 总数严格 100/25/25；
3. records 严格 800/200/200；
4. subtask 配额按上表；若 scaffold group 使其数学不可行，最多 ±1 并在 manifest 显式记录，不能静默拆 group。

### 11.5 split manifest

```text
origin_id
anonymous_sample_id
leakage_group_id
subtask
split
canonical_source_hash
canonical_gt_hash
scaffold_hash
split_seed
dataset_version
```

每个 variant 只能继承 origin split，禁止二次 random split。

---

## 12. Deterministic validation gates

### 12.1 Validation chain

1. `InputRecordValidator`
   - raw/process ID 对齐；
   - required fields 完整；
   - source/GT parse；
   - subtask mapping 合法。

2. `ReferenceDAGValidator`
   - DAG 无环；
   - dependencies 完整；
   - clean process product/Answer 与 GT 图等价；
   - natural/FORMAL slots 可 round-trip。

3. `RDKitStructureValidator`
   - strict sanitize、valence、stereo、主片段策略；
   - descriptors 可重算；
   - candidate/ref equivalence 明确。

4. `GraphEditValidator`
   - anchor/group/fragment 邻接；
   - source→product 真实 atom/bond diff；
   - 声称的 addition/deletion/substitution 类型正确；
   - attachment atom/bond 正确实现 candidate state。

5. `HallucinationSemanticValidator`
   - H 的目标 root 确实错误；
   - N 完全 faithful；
   - core variant 恰好一个 independent root；
   - operator ID 与实际 GraphDelta 一致。

6. `PropagationValidator`
   - changed set 满足 STOP/PARTIAL/FULL/TERMINAL；
   - FULL 内部自洽；
   - TERMINAL reasoning state 与 reference 相同，只改变 Answer。

7. `RendererValidator`
   - natural language、FORMAL、state 一致；
   - placeholder/claim spans 完整；
   - 无 `incorrect/corrupted/hallucinated/reference answer` 等 label leakage；
   - detector text 不含 GT/reference-only header。

8. `TokenAlignmentValidator`
   - 每个 positive char span 覆盖至少一个 evaluated token；
   - source/instruction/special token 全部 ignore；
   - type/role masks 是 error mask 的合法子集；
   - token arrays 等长；
   - 没有 pre-token shift。

9. `BundleIntegrityValidator`
   - 每个 origin 恰好八条；
   - 每个 H 有唯一 matched N；
   - bundle/leakage group 在一个 split；
   - H/N candidate source、renderer、长度和 target field 分布匹配。

### 12.2 Structured validation issue

```python
@dataclass(frozen=True)
class ValidationIssue:
    code: str
    severity: Severity
    node_ids: tuple[str, ...]
    message: str
    evidence: Mapping[str, Any]
```

不能只保存 `all_pass: true/false`。所有 rejection 都进入 ledger，用于检查某个 operator/LLM backend 是否产生选择偏差。

### 12.3 H/N phenotype invariants

#### H_LOCAL

```text
一个 root upstream claim 错
Answer ≡ GT
正确 downstream value 不自动标错
```

#### H_PARTIAL

```text
一个 root error
changed descendants 为合法连通子图
cut 后 state 保持 reference
root/propagated spans 可区分
```

#### H_FULL_CF

```text
一个结构 root error
candidate product ≠ GT
candidate Answer ≡ candidate product
所有 downstream count/delta 与 candidate product 一致
candidate chain 内部自洽但违反 instruction/oracle path
```

#### H_TERMINAL

```text
reasoning state 与 reference 一致
candidate Answer ≠ reference product/GT
positive span 只位于 final answer
causal_role = TERMINAL
```

#### 所有 N

```text
所有 semantic state faithful
Answer ≡ GT
positive masks 全 0
仍保留 matched_target_span
```

---

## 13. Anti-shortcut 和 adversarial filtering

### 13.1 H/N matched controls

每个 H/N pair 匹配：

```text
origin/input
target step/field
propagation family
renderer backend/style
rewrite次数
字符/token 长度 bucket
数字/元素/标点分布
SMILES validity bucket
candidate-source mode
LM perplexity bucket
```

### 13.2 SMILES surface control

- Correct process product 与 correct final Answer 使用独立的 equivalent non-canonical serialization。
- H/N pair 使用相同 source/input view。
- 正确性一律使用 isomeric molecular graph equivalence。
- 对 TERMINAL，N 的 reasoning product 与 Answer 也允许字符串不同但图等价，避免裸字符串相等成为标签。
- Canonical RDKit comparator 仍是合法、必须报告的 symbolic baseline；目标是消除字符串 shortcut，不是隐藏真正的化学一致性信号。

### 13.3 值级交叉平衡

- donor fragment/group 只能来自同 split；
- 某个值作为 H replacement 时，应尽量在其他上下文中作为 N truth 出现；
- F/Cl/Br/I 等使用循环平衡；
- numeric distractor 使用真实分布；
- anchor index、fragment size、ring count 不能与 label 单调相关。

### 13.4 生成大候选池后再选 1,200 条

每个 recipe 先产生 3–10 个已验证 candidates，再用 constrained matching 选择最终版本。不要接受“第一个通过”的 candidate，否则不同 backend 的失败率会改变最终分布。

### 13.5 Shortcut attack suite

至少训练/运行：

```text
metadata-only LR
char/word TF-IDF + LR/SVM
target-span-only classifier
reasoning-only classifier（屏蔽 source/instruction）
char-CNN / small MLP
frozen chemical encoder span-only
nearest-neighbor origin/operator retrieval
SMILES parse/validity baseline
reasoning-product vs final-answer graph comparator
full symbolic graph-edit verifier
```

建议 Pilot go/no-go：

```text
metadata-only AUROC       ≤ 0.55
span-only TF-IDF AUROC    ≤ 0.55
reasoning-only shallow    ≤ 0.60
invalid-SMILES 与全局 H 标签无非预期相关
H/N token-length standardized difference < 0.1
```

这些阈值是工程筛查条件，不是论文预设结论。Parser/verifier 对相应可执行 error type 表现很好属于合理能力，应按 type 单独报告，而不是视作 shortcut。

---

## 14. Canonical record schema

### 14.1 Dataset record

```json
{
  "schema_version": "molhallulens.edit.v1",
  "dataset_version": "pilot_v1",
  "record_id": "mol_edit.add_v2.0057__LOCAL__H",
  "origin_id": "mol_edit.add_v2.0057",
  "leakage_group_id": "lg_...",
  "bundle_id": "bundle_...",
  "pair_id": "mol_edit.add_v2.0057__LOCAL",
  "split": "train",
  "family": "mol_edit",
  "subtask": "add",

  "variant": {
    "label": "H",
    "propagation": "LOCAL",
    "matched_record_id": "mol_edit.add_v2.0057__LOCAL__N"
  },

  "detector_input": {
    "indexed_smiles": "...",
    "instruction": "...",
    "reasoning_chain": "...",
    "final_answer": "...",
    "field_order": [
      "indexed_smiles",
      "instruction",
      "reasoning_chain",
      "final_answer"
    ]
  },

  "serialized": {
    "text": "...",
    "sha256": "...",
    "segments": []
  },

  "mutation": {
    "root_state_id": "step1.anchor.idx",
    "operator_id": "mol_edit.add.alternate_anchor",
    "candidate_source": "HYBRID",
    "candidate_id": "candidate_03",
    "donor_origin_id": null,
    "renderer_id": "poe_gpt54mini_style_07",
    "seed": 10291
  },

  "trace_labels": {
    "hallucination_present": true,
    "reasoning_valid": false,
    "answer_correct": true,
    "chemically_valid": true,
    "constraint_satisfied": false,
    "format_valid": true,
    "answer_complete": true
  },

  "spans": [],
  "verification": {
    "rdkit_sanitize": true,
    "graph_edit_verified": true,
    "propagation_verified": true,
    "span_verified": true
  }
}
```

### 14.2 Oracle record

为降低误用风险，`gt_smiles`、reference graph、candidate graph 和完整 LLM provenance 建议保存在以 `record_id` join 的独立 oracle/provenance artifact：

```json
{
  "record_id": "...",
  "gt_smiles": "...",
  "reference_state_graph": {},
  "candidate_state_graph": {},
  "graph_delta": [],
  "build_provenance": {
    "provider": "poe",
    "base_url": "https://api.poe.com/v1",
    "requested_model_id": "gpt-5.4-mini",
    "bot_display_name": "GPT-5.4-Mini",
    "transport": "responses",
    "model_catalog_entry_sha256": "...",
    "response_ids": ["..."],
    "x_request_ids": ["..."],
    "proposal_prompt_sha256": "...",
    "renderer_prompt_sha256": "...",
    "tool_schema_sha256": "...",
    "attempt_count": 2,
    "token_usage": {},
    "cost_points": null,
    "cache_keys": []
  },
  "visible_to_detector": false
}
```

`POE_API_KEY` 和 Authorization header 永远不属于 provenance schema。

### 14.3 ChemDFM-R tokenized derived artifact

```json
{
  "record_id": "...",
  "activation_alignment": "post_token_h_t",
  "tokenizer_fingerprint": "...",
  "input_ids": [],
  "attention_mask": [],
  "offset_mapping": [],
  "segment_ids": [],
  "evaluation_mask": [],
  "hallucination_core_mask": [],
  "error_any_mask": [],
  "semantic_type_ids_per_token": [],
  "edit_subtype_ids_per_token": [],
  "root_mask": [],
  "propagated_false_mask": [],
  "propagated_conditional_mask": [],
  "terminal_mask": [],
  "local_falsehood_mask": [],
  "off_task_branch_mask": []
}
```

多标签建议在文件中保存为 `list[list[int]]`，训练 collator 再展开为 `[T, K]` multi-hot tensor。

---

## 15. 代码目录和配置

### 15.1 建议代码目录

```text
Pilot/molhallulens/
├── config/
│   ├── dataset.yaml
│   ├── operators.yaml
│   ├── labels.yaml
│   ├── llm.yaml
│   └── rendering.yaml
├── domain/
│   ├── enums.py
│   ├── records.py
│   ├── edit_truth.py
│   ├── state_dag.py
│   ├── recipes.py
│   ├── labels.py
│   └── errors.py
├── adapters/
│   ├── base.py
│   └── chemcot_mol_edit.py
├── perturbators/
│   ├── base.py
│   ├── registry.py
│   ├── editing/
│   │   ├── base.py
│   │   ├── addition.py
│   │   ├── deletion.py
│   │   └── substitution.py
│   ├── optimization/base.py
│   ├── understanding/base.py
│   └── reaction_prediction/base.py
├── candidates/
│   ├── base.py
│   ├── rule_source.py
│   ├── rdkit_source.py
│   ├── llm_agent_source.py
│   ├── hybrid_engine.py
│   ├── selector.py
│   └── response_cache.py
├── providers/
│   └── poe/
│       ├── client.py
│       ├── responses_client.py
│       ├── chat_client.py
│       ├── fastapi_poe_client.py
│       ├── model_registry.py
│       ├── schemas.py
│       ├── rate_limiter.py
│       └── usage_ledger.py
├── propagation/
│   ├── base.py
│   └── editing.py
├── rendering/
│   ├── trace_ast.py
│   ├── formal.py
│   ├── natural_rule.py
│   ├── natural_llm.py
│   └── detector_prompt.py
├── validation/
│   ├── base.py
│   ├── reference.py
│   ├── chemistry.py
│   ├── graph_edit.py
│   ├── semantics.py
│   ├── propagation.py
│   ├── rendering.py
│   ├── token_labels.py
│   └── bundle.py
├── annotation/
│   ├── char_spans.py
│   └── token_projection.py
├── builders/
│   ├── bundles.py
│   ├── splitter.py
│   └── writer.py
├── audit/
│   ├── shortcut_features.py
│   ├── attack_models.py
│   └── report.py
├── cli/
│   ├── audit_origins.py
│   ├── make_split.py
│   ├── build_hallucination_dataset.py
│   ├── validate_dataset.py
│   ├── tokenize_dataset.py
│   └── inspect_origin.py
└── tests/
    ├── fixtures/
    ├── unit/
    ├── property/
    ├── integration/
    └── golden/
```

当前 DAG 很小，可以先自行实现 deterministic topological sort 和 reachable descendants，不必立即引入 `networkx`。

### 15.2 dataset.yaml

```yaml
version_name: pilot_v1
global_seed: 20260828

input:
  root: Pilot/Dataset
  family: mol_edit
  subtasks: [add, delete, substitute]

bundle:
  policies: [LOCAL, PARTIAL, FULL_CF, TERMINAL]
  matched_controls: true
  records_per_origin: 8

split:
  train_origins: 100
  validation_origins: 25
  test_origins: 25
  enforce_leakage_groups: true

detector:
  field_order: [indexed_smiles, instruction, reasoning_chain, final_answer]
  include_gt_smiles: false
  activation_alignment: post_token_h_t
```

### 15.3 llm.yaml

```yaml
provider:
  name: poe
  base_url: https://api.poe.com/v1
  api_key_env: POE_API_KEY
  model_id: gpt-5.4-mini
  bot_name: GPT-5.4-Mini
  primary_transport: responses
  fallback_transports: [chat_completions, fastapi_poe]

model_discovery:
  endpoint: /models
  require_model_id: gpt-5.4-mini
  require_endpoints: [/v1/responses, /v1/chat/completions]
  require_features: [tools]
  fail_closed: true
  cache_ttl_hours: 1

proposal:
  model_id: gpt-5.4-mini
  reasoning_effort: medium
  retry_reasoning_effort: high
  temperature: 0.4
  parallel_tool_calls: false
  web_search: false
  max_attempts: 3
  candidates_per_attempt: 5
  max_tool_calls: 6
  structured_output_schema: proposal_v1
  local_schema_validation: true

renderer:
  model_id: gpt-5.4-mini
  reasoning_effort: low
  temperature: 0.2
  label_blind: true
  output_mode: placeholder_template
  local_schema_validation: true

rate_limit:
  requests_per_minute: 450
  max_concurrency: 8
  retry_base_ms: 250
  respect_retry_after: true

usage:
  check_balance_before_build: true
  balance_endpoint: https://api.poe.com/usage/current_balance
  history_endpoint: https://api.poe.com/usage/points_history
  export_private_ledger: true

cache:
  enabled: true
  content_addressed: true
  replay_only_for_release: true
  include_model_catalog_hash: true
```

### 15.4 Poe dependencies 和 secret handling

建议依赖：

```text
openai          # primary Poe Responses/Chat compatibility client
fastapi-poe     # Poe-native smoke/simple streaming adapter
httpx           # model catalog、balance、usage history
pydantic        # request/tool/output 本地强校验
jsonschema      # JSON schema validation/audit
```

具体版本通过项目 lockfile 固定。运行时只从环境变量读取：

```python
poe_api_key = os.environ["POE_API_KEY"]
```

不得提供源码内默认 key，不得把 key 放入 YAML、CLI 参数、异常 traceback、raw HTTP dump 或 notebook output。HTTP logger 必须对 `Authorization` header 做 redaction。

### 15.5 CLI 阶段

```text
1. audit_origins.py
2. make_split.py --version pilot_v1
3. build_hallucination_dataset.py --split train
4. build_hallucination_dataset.py --split validation
5. build_hallucination_dataset.py --split test
6. validate_dataset.py --strict
7. tokenize_dataset.py --model ChemDFM-R-14B
8. run_shortcut_audit.py
```

Release build 应从 frozen caches replay，不在最后一步临时重新请求 API。

---

## 16. 测试计划

### 16.1 Unit tests

- source/GT canonicalization；
- subtask normalization；
- DAG 无环和 topo order；
- graph descendant 查询；
- EditTruth 映射和 symmetry handling；
- 每个 operator 的 root-only patch；
- fragment/group attachment atom；
- molecule equivalence；
- char span builder；
- token offset projection。

### 16.2 Property tests

```text
STOP changed set == root
PARTIAL changed set 是 root downstream 的合法连通子图
FULL changed set 覆盖所有可推导 descendants
TERMINAL changed set == final_answer
H 至少有一个 positive span
N 没有 positive span
所有 candidate product 可按预期 sanitize/拒绝
```

### 16.3 Golden tests

每个 subtask 至少选择 3 个 golden origins：

- 一个标准样本；
- 一个多候选 hard sample；
- 一个边界/异常 sample。

每个 golden origin 完整生成 8 records，并冻结：

- state graph；
- mutation events；
- rendered text；
- char spans；
- token masks；
- validation report。

### 16.4 LLM tests

- mocked valid Structured Output；
- invalid JSON/schema；
- invalid SMILES；
- equivalent-to-reference candidate；
- repeated rejected candidate；
- timeout/rate limit；
- cache hit/replay；
- 三次失败后的 deterministic fallback；
- renderer 修改 locked facts 时拒绝。

### 16.5 Dataset integrity tests

```python
assert n_origins == 150
assert n_records == 1200
assert split_records == {"train": 800, "validation": 200, "test": 200}
assert every_origin_has_exactly_8_records
assert every_pair_has_one_h_and_one_n
assert no_leakage_group_crosses_splits
assert no_cross_split_donor_edge
assert h_n_per_split == {"train": (400, 400), "validation": (100, 100), "test": (100, 100)}
assert propagation_counts_per_split_are_exact
assert detector_text_never_contains_gt_field
assert all_labels_are_post_token_aligned
```

### 16.6 Reproducibility tests

- 同 origin/recipe/seed + frozen cache 输出 byte-identical；
- serialized text SHA 与 tokenized artifact 输入一致；
- manifest hash 与所有 record provenance 对齐；
- dependency versions、RDKit、tokenizer、Poe model ID/catalog hash、prompt hashes 完整记录；
- 构建时没有把 `POE_API_KEY`、Authorization header 写入任何 artifact；
- capability smoke tests 和 frozen cache replay 可在不发新请求时通过。

---

## 17. 评测协议

### 17.1 Primary metrics

不只报告 pooled token AUROC。建议主指标：

1. origin-balanced token AUPRC；
2. span F1，token IoU ≥ 0.5；
3. macro semantic/edit-type AUPRC；
4. matched-pair accuracy：`score(H) > score(N)`；
5. N-control false-positive rate。

### 17.2 必须切片报告

```text
Addition / Deletion / Substitution
LOCAL / PARTIAL / FULL_CF / TERMINAL
ROOT / PROPAGATED_FALSE / PROPAGATED_CONDITIONAL / TERMINAL
Reasoning / Final Answer
每个 semantic type
每个 editing subtype（样本数足够时）
Rule / RDKit / LLM / Hybrid candidate source
```

### 17.3 额外指标

- token AUROC/AUPRC/F1；
- span exact、any-overlap、IoU F1；
- typed span F1；
- localization recall；
- Brier score、ECE；
- local falsehood 与 off-task branch 分开；
- symbolic verifier、RDKit comparator 和 shallow attacks。

### 17.4 阈值和置信区间

- 阈值只在 validation 选择；test 固定。
- 置信区间按 origin/leakage group bootstrap，不能按 token bootstrap。
- Test 有 200 records 但只有 25 origins，细粒度 type 若少于约 5 个独立 origins，只作为 diagnostic，不做 headline conclusion。

---

## 18. 分阶段执行路线

### Phase 0：冻结 schema 和接口

产物：

- 本实施计划审阅通过；
- `dataset.yaml`、`labels.yaml`、`operators.yaml`；
- detector serialization golden string；
- State DAG node/edge IDs；
- post-token alignment unit test。

退出条件：所有人对 GT 不可见、四 policy 等额、8-record bundle、multi-label schema 无歧义。

### Phase 1：Reference DAG 和严格 validator

实现：

- raw/process adapters；
- `SubtaskNormalizer`；
- three editing StateSchemas；
- EditTruth graph diff；
- reference/chemistry/graph-edit validators；
- anomaly registry。

退出条件：150 origins 全部有 validation report；异常样本被显式分类，无 silent fallback。

### Phase 2：OOP skeleton 和 deterministic operators

实现：

- abstract/family/subtask hierarchy；
- operator decorator/registry；
- rule/RDKit candidate sources；
- STOP/PARTIAL/FULL/TERMINAL；
- deterministic FORMAL renderer。

退出条件：每个 subtask 的 golden origin 可在无 LLM 条件下生成完整 8-record bundle。

### Phase 3：Split 和 donor pools

实现：

- canonical source/GT/scaffold hashes；
- union-find leakage groups；
- exact 100/25/25 split；
- frozen split manifest；
- split-local donor pools。

退出条件：所有 leakage、quota 和 donor assertions 通过。

### Phase 4：LLM Agent candidate invention

实现：

- proposal request/response schemas；
- read-only chemistry tools；
- Poe model registry/capability probe；
- Poe Responses API primary client；
- Poe Chat Completions / `fastapi_poe` fallback adapters；
- Poe rate limiter、error classifier 和 usage ledger；
- validate–retry–fallback；
- content-addressed cache；
- provenance ledger。

退出条件：每个 structural operator 至少有 rule、LLM 或 hybrid 的可验证候选；失败率和 reject codes 可统计。

### Phase 5：Label-blind renderer 和 char spans

实现：

- Trace AST；
- placeholder-based natural renderer；
- deterministic FORMAL/Answer；
- node/edge mention offsets；
- leakage phrase scanner。

退出条件：golden texts 的所有 positive claims 都有精确 char span；无全文 diff 标注。

### Phase 6：15-origin dry run

选择每个 subtask 5 origins：

```text
15 origins × 8 = 120 records
```

执行：

- 人工化学审查；
- token projection；
- shallow shortcut audit；
- operator/LLM failure analysis；
- H/N length/style matching。

退出条件：无系统性标签错误，shortcut 指标达到工程阈值，八类 bundle invariants 稳定。

### Phase 7：完整 1,200-record build

分 split 构建，test 最后生成并冻结。任何 single variant 失败都必须重建/补齐整个 matched pair，最终每个 origin 必须八条完整。

### Phase 8：Release QA 和 post-token activation extraction

执行：

- strict dataset validation；
- replay cache reproducibility；
- tokenized artifact；
- ChemDFM-R `h_t` extraction；
- baseline/shortcut report；
- dataset card 和 known limitations。

---

## 19. 发布验收标准

数据只有在以下条件全部满足时才可标记为 `pilot_v1`：

```text
[ ] 150 origins、1,200 records
[ ] train/validation/test = 800/200/200
[ ] 每个 origin 恰好 4 H + 4 matched N
[ ] 每种 propagation 在每个 split 精确平衡
[ ] 无 origin/scaffold/donor leakage
[ ] detector text 顺序严格为 SMILES→instruction→reasoning→answer
[ ] detector text 不包含 gt_smiles/reference-only metadata
[ ] activation_alignment 仅为 post_token_h_t
[ ] H 每条至少一个有效 char/token positive span
[ ] N 所有 hallucination/error masks 为 0
[ ] root/propagated/terminal labels 完整
[ ] semantic type 和 editing subtype 为 multi-label
[ ] FORMAL、natural language、state graph 可互相追溯
[ ] 所有 structural candidates 经过 deterministic RDKit/graph validation
[ ] LLM request、tool run、reject、Poe model catalog hash 和 cache 可审计
[ ] 所有请求使用 Poe `gpt-5.4-mini`，无未声明模型 fallback
[ ] Poe points/usage ledger 已导出，artifact 中不含 `POE_API_KEY`
[ ] shortcut audit 和 symbolic baselines 已运行
[ ] test threshold 未被用于选择 candidate、layer 或 detector threshold
```

---

## 20. 风险与处理

### 风险 1：1,200 records 的表面规模掩盖只有 150 origins

处理：所有 split、bootstrap、显著性和 paired evaluation 以 origin/leakage group 为单位。

### 风险 2：LLM 生成了额外、未标注错误

处理：LLM 只提结构候选或 placeholder prose；所有 locked facts、FORMAL 和 Answer 由程序生成；NL–state validator 加人工抽检。

### 风险 3：FULL_CF 下游数字局部正确，标签定义争议

处理：同时保存 `local_falsehood_mask` 和 `off_task_branch_mask`，不把二者混成单一语义。

### 风险 4：TERMINAL 可被 reasoning-product/Answer verifier 检出

处理：保留为正式、等额类别并报告 symbolic baseline；使用独立 equivalent SMILES serialization 消除裸字符串 shortcut，但不掩盖真正的分子不一致。

### 风险 5：Invalid chemistry 让 parser 成为标签代理

处理：控制 invalid/format 类占比，按 type 分开报告；核心结构错误以 chemically valid near-miss 为主。

### 风险 6：Pilot 来自冻结 benchmark

如果这 150 origins 属于原规划中的 ChemCoTBench-V2 final blind test，那么用其训练 detector、选 layer 或调 threshold 会消耗该 test。处理方案必须二选一并写入 dataset card：

1. 这些 150 条只用于 pipeline/schema smoke test，不进入正式 detector training；或
2. 正式将其定义为开发用 counterfactual split，不再声称对应 ChemCoTBench-V2 slice 是 untouched final test。

### 风险 7：API 结果不可逐字重现

处理：Poe 当前没有为 `gpt-5.4-mini` 暴露可选 upstream snapshot，因此保存 model ID、catalog entry/hash、完整 transcript 和内容寻址 cache；正式 release 使用 replay-only，并由确定性 validator 保证语义不变量。数据卡必须披露这一限制。

### 风险 8：Poe compatibility 参数被静默忽略

处理：不依赖 Chat `response_format`、function `strict`、API `seed` 或 provider metadata；构建前执行 capability probes，所有 tool arguments 和 model outputs 经过本地 schema/chemistry validation。

### 风险 9：Poe points 或 rate limit 中断完整 bundle

处理：full build 前检查 point balance；按 request ID checkpoint；遵守 request-rate headers 和 `Retry-After`；出现 402 时停止新请求但保留全部 cache。恢复后从未完成 recipe 继续，不能留下不完整 H/N bundle。

---

## 21. 推荐立即开始的第一个实现批次

第一批不要先写 GPT prompt，而应依次完成：

1. `domain/enums.py`、`records.py`、`state_dag.py`；
2. `adapters/chemcot_mol_edit.py`；
3. 三个 StateSchema；
4. `EditTruth` 和 graph-edit validator；
5. `Perturbator` / `MoleculeEditingPerturbator` / 三个 subtask classes；
6. 四类 propagation 的 changed-set tests；
7. 一个 add、一个 delete、一个 substitute golden origin 的 deterministic 8-record bundle；
8. 再接 LLM Agent candidate source 和 label-blind renderer。

这个顺序确保 LLM 进入系统时，已经存在可执行的 state schema、候选 contract 和唯一 validator，不会让自然语言生成反过来决定数据标签。
