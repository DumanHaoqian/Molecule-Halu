"""单一、多点、可控的分子推理幻觉生成配置。

这个文件是新生成链的唯一参数入口。每一条生成记录都至少包含一个真实
发生的 mutation。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal


# ---------------------------------------------------------------------------
# 1. 六种明确的 mutation category
# ---------------------------------------------------------------------------

# numeric_integer: COUNT / INTEGER，只做整数偏移。
# numeric_float: FLOAT，做绝对值或比例变化，始终保留 float 类型。
# atom_index: anchor_idx，只替换为输入分子中真实存在的 atom map。
# element_symbol: anchor_element，只替换元素符号。
# molecular_fragment: leaving / remove / add fragment，从语料 pool 中替换。
# smiles_structure: product / final_answer，修改真实分子图后重新生成 SMILES。
MUTATION_CATEGORIES = (
    "numeric_integer",
    "numeric_float",
    "atom_index",
    "element_symbol",
    "molecular_fragment",
    "smiles_structure",
)


# ---------------------------------------------------------------------------
# 2. 一条样本修改几个“语义点”
# ---------------------------------------------------------------------------

# "fixed" 固定修改 FIXED_EDIT_COUNT 处；"range" 在 MIN/MAX 之间采样。
# "maximum" 每题都取互不冲突的 root 编辑点最大数，不使用 FIXED/MIN/MAX。
# 传播产生的下游节点和文本枚举错误不计入 root 数量；不足时直接报错，不降量。
# MAX_EDIT_COUNT=None 表示不设置人为上限，由每条 Reference DAG 的可编辑
# 节点和传播冲突共同计算该样本真正能够支持的最大值。
EDIT_COUNT_MODE: Literal["fixed", "range", "maximum"] = "range"
FIXED_EDIT_COUNT = 2
MIN_EDIT_COUNT = 1
MAX_EDIT_COUNT: int | None = None


# ---------------------------------------------------------------------------
# 3. 哪些位置允许修改
# ---------------------------------------------------------------------------

# reasoning step 默认允许修改。
INCLUDE_REASONING_STEPS = True

# final_answer 与 reasoning node 使用同一个候选、采样和标注流程。
INCLUDE_FINAL_ANSWER = True

# final_answer 在一次 plan 中被主动选中的概率。0.0 表示从不选，1.0 表示
# 只要 edit_count > 0 就优先选它。
FINAL_ANSWER_PROBABILITY = 0.35

# 这里使用语义节点名。DELETE 的 remove_group 同时对应 Step 1 和 Step 2
# 两个 DAG 节点，但只算“一处修改”。
EDITABLE_NODES_BY_SUBTASK: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "add": (
            "anchor_idx",
            "anchor_element",
            "leaving",
            "add_fragment",
            "fragment_heavy",
            "product",
            "source_heavy",
            "product_heavy",
            "source_rings",
            "product_rings",
            "final_answer",
        ),
        "delete": (
            "anchor_idx",
            "anchor_element",
            "remove_group",
            "remove_heavy",
            "product",
            "source_heavy",
            "product_heavy",
            "source_rings",
            "product_rings",
            "final_answer",
        ),
        "substitute": (
            "anchor_idx",
            "anchor_element",
            "remove_group",
            "add_fragment",
            "remove_heavy",
            "add_heavy",
            "product",
            "source_heavy",
            "product_heavy",
            "source_rings",
            "product_rings",
            "final_answer",
        ),
    }
)


# ---------------------------------------------------------------------------
# 4. 数值修改幅度
# ---------------------------------------------------------------------------

# COUNT / INTEGER 使用离散整数偏移；0 被禁止，因为它不会产生幻觉。
INTEGER_DELTAS = (-5, -3, -2, -1, 1, 2, 3, 5)

# COUNT（重原子数、环数等）不能小于 0；普通 INTEGER（delta）允许为负。
COUNT_MIN_VALUE = 0

# FLOAT 与整数分开处理。"absolute" 表示加减绝对值；"relative" 表示
# 按原值比例改变。
FLOAT_MUTATION_MODE: Literal["absolute", "relative"] = "relative"
FLOAT_ABSOLUTE_DELTAS = (-0.20, -0.10, -0.05, 0.05, 0.10, 0.20)
FLOAT_RELATIVE_CHANGES = (-0.20, -0.10, -0.05, 0.05, 0.10, 0.20)
FLOAT_DECIMAL_PLACES = 4


# ---------------------------------------------------------------------------
# 5. 原子编号和元素修改
# ---------------------------------------------------------------------------

# anchor_idx 不是普通整数：候选必须是 source molecule 中真实存在的 atom map。
ATOM_INDEX_PREFER_SAME_ELEMENT = True

# anchor_element 只从这里选择替代元素。
REPLACEMENT_ELEMENTS = ("C", "N", "O", "S", "F", "Cl", "Br", "I", "P")


# ---------------------------------------------------------------------------
# 6. fragment / functional-group pool
# ---------------------------------------------------------------------------

# 候选片段来自整个 Reference DAG corpus，而不是十来个硬编码字符串。
# 当前 150 条 Pilot 数据可以抽取 77 个去重片段；换成更大语料会自动扩展。
FRAGMENT_SIMILARITY_MIN = 0.20
FRAGMENT_SIMILARITY_MAX = 0.95
FRAGMENT_TARGET_SIMILARITY = 0.70
FRAGMENT_REQUIRE_SAME_CHARGE = True
FRAGMENT_MAX_HEAVY_ATOM_DIFFERENCE = 4


# ---------------------------------------------------------------------------
# 7. product / final_answer 的 SMILES 结构修改
# ---------------------------------------------------------------------------

# 新实现当前支持原子替换、键级替换和末端原子删除；所有候选都必须先通过
# RDKit sanitization，且不能与 reference 分子图等价。
SMILES_MUTATION_OPERATORS = (
    "smiles_atom_replacement",
    "smiles_bond_order_change",
    "smiles_terminal_atom_deletion",
)
SMILES_REPLACEMENT_ATOMIC_NUMBERS = (6, 7, 8, 9, 15, 16, 17, 35)
SMILES_SIMILARITY_MIN = 0.50
SMILES_SIMILARITY_MAX = 0.999
REQUIRE_VALID_SMILES = True
REQUIRE_DIFFERENT_FROM_REFERENCE = True


# ---------------------------------------------------------------------------
# 8. 确定性传播与一致性审计
# ---------------------------------------------------------------------------

# 根错误注入后，自动更新可以精确计算的关联节点，例如：
# product_rings -> ring_delta，product -> final_answer / count / delta。
ENABLE_DETERMINISTIC_PROPAGATION = True

# 算术、重复值和 product/final-answer 等硬关系若传播后仍冲突，立即拒绝样本。
# 化学语义关系（例如 instruction 是否真的支持 product）允许作为核心幻觉保留。
FAIL_ON_TRIVIAL_EDGE_VIOLATION = True

# ---------------------------------------------------------------------------
# 9. Poe agent：依据修改后的 FORMAL 最小修改原始 step_text
# ---------------------------------------------------------------------------

# API token 只允许从这个环境变量读取。代码不会接收命令行 token，也不会把
# token 写进 prompt、cache、日志或最终数据。用户在 terminal 里执行：
#   export POE_API_KEY='从 https://poe.com/api/keys 获取的 token'
POE_API_KEY_ENV = "POE_API_KEY"

# Poe bot 名称可以直接在这里替换。最终 record 会保存 bot 名称和 prompt hash。
POE_BOT_NAME = "gpt-5.4-mini"
POE_TEMPERATURE = 0.20

# 每个待生成步骤的总尝试上限。重试只发送失败步骤，已通过步骤不重新生成。
POE_MAX_ATTEMPTS = 2

# 已验证的 response 会缓存。相对路径以 Pilot/ 为根目录；cache 不包含 token。
POE_CACHE_DIRECTORY = "GeneratedDataset/.poe_text_cache"

# 失败诊断与成功缓存分开保存；仅包含白名单字段、脱敏后的相关正文。
# False 只关闭落盘，不关闭校验、异常内诊断或失败统计。
POE_SAVE_DIAGNOSTICS = True
POE_DIAGNOSTIC_DIRECTORY = "GeneratedDataset/.poe_text_diagnostics"
# 每个诊断文本字段（包括反馈给 Poe 的上一版输出）的最大字符数。
POE_DIAGNOSTIC_MAX_CHARACTERS = 12000


# ---------------------------------------------------------------------------
# 10. H/N matched-pair 输出
# ---------------------------------------------------------------------------

# True 时，每条幻觉正样本 H 都同时发布一条真值负样本 N。N 复用 H 的
# Poe 文本骨架，只把经过 marker 验证的 claim 值换回 reference truth；若
# 换回后不能通过 FORMAL / 残留 / 算术检查，则对该步反向调用 Poe。
EMIT_MATCHED_NEGATIVE = True


# ---------------------------------------------------------------------------
# 11. 可复现性
# ---------------------------------------------------------------------------

# 每条记录的最终 seed = GLOBAL_SEED + origin_id + variant_index 的稳定哈希。
GLOBAL_SEED = 20260903


@dataclass(frozen=True, slots=True)
class HallucinationGenerationConfig:
    """经过校验的扁平配置；字段与上面的显式参数一一对应。"""

    edit_count_mode: Literal["fixed", "range", "maximum"]
    fixed_edit_count: int
    min_edit_count: int
    max_edit_count: int | None
    include_reasoning_steps: bool
    include_final_answer: bool
    final_answer_probability: float
    editable_nodes_by_subtask: Mapping[str, tuple[str, ...]]
    integer_deltas: tuple[int, ...]
    count_min_value: int
    float_mutation_mode: Literal["absolute", "relative"]
    float_absolute_deltas: tuple[float, ...]
    float_relative_changes: tuple[float, ...]
    float_decimal_places: int
    atom_index_prefer_same_element: bool
    replacement_elements: tuple[str, ...]
    fragment_similarity_min: float
    fragment_similarity_max: float
    fragment_target_similarity: float
    fragment_require_same_charge: bool
    fragment_max_heavy_atom_difference: int
    smiles_mutation_operators: tuple[str, ...]
    smiles_replacement_atomic_numbers: tuple[int, ...]
    smiles_similarity_min: float
    smiles_similarity_max: float
    require_valid_smiles: bool
    require_different_from_reference: bool
    enable_deterministic_propagation: bool
    fail_on_trivial_edge_violation: bool
    poe_api_key_env: str
    poe_bot_name: str
    poe_temperature: float
    poe_max_attempts: int
    poe_cache_directory: str
    emit_matched_negative: bool
    global_seed: int
    poe_save_diagnostics: bool = POE_SAVE_DIAGNOSTICS
    poe_diagnostic_directory: str = POE_DIAGNOSTIC_DIRECTORY
    poe_diagnostic_max_characters: int = POE_DIAGNOSTIC_MAX_CHARACTERS

    def __post_init__(self) -> None:
        if type(self.poe_save_diagnostics) is not bool:
            raise ValueError("poe_save_diagnostics must be bool")
        if type(self.poe_diagnostic_directory) is not str or not self.poe_diagnostic_directory.strip():
            raise ValueError("poe_diagnostic_directory must be non-empty text")
        if type(self.poe_diagnostic_max_characters) is not int or self.poe_diagnostic_max_characters < 1:
            raise ValueError("poe_diagnostic_max_characters must be a positive integer")
        if self.edit_count_mode not in {"fixed", "range", "maximum"}:
            raise ValueError("edit_count_mode must be 'fixed', 'range', or 'maximum'")
        if min(self.fixed_edit_count, self.min_edit_count) < 1:
            raise ValueError("edit counts must be positive")
        if self.max_edit_count is not None and (
            type(self.max_edit_count) is not int or self.max_edit_count < 1
        ):
            raise ValueError("max_edit_count must be positive or None")
        if (
            self.max_edit_count is not None
            and self.min_edit_count > self.max_edit_count
        ):
            raise ValueError("min_edit_count cannot exceed max_edit_count")
        if not 0.0 <= self.final_answer_probability <= 1.0:
            raise ValueError("final_answer_probability must be in [0, 1]")
        if not self.integer_deltas or 0 in self.integer_deltas:
            raise ValueError("integer_deltas must be non-empty and cannot contain 0")
        if self.float_decimal_places < 0:
            raise ValueError("float_decimal_places cannot be negative")
        if not (
            0.0 <= self.fragment_similarity_min
            <= self.fragment_target_similarity
            <= self.fragment_similarity_max
            <= 1.0
        ):
            raise ValueError("fragment similarity bounds are invalid")
        if not 0.0 <= self.smiles_similarity_min < self.smiles_similarity_max <= 1.0:
            raise ValueError("SMILES similarity bounds are invalid")
        for value, name in (
            (self.enable_deterministic_propagation, "enable_deterministic_propagation"),
            (self.fail_on_trivial_edge_violation, "fail_on_trivial_edge_violation"),
            (self.emit_matched_negative, "emit_matched_negative"),
        ):
            if type(value) is not bool:
                raise TypeError(f"{name} must be bool")
        for value, name in (
            (self.poe_api_key_env, "poe_api_key_env"),
            (self.poe_bot_name, "poe_bot_name"),
            (self.poe_cache_directory, "poe_cache_directory"),
        ):
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        if not 0.0 <= self.poe_temperature <= 2.0:
            raise ValueError("poe_temperature must be in [0, 2]")
        if type(self.poe_max_attempts) is not int or self.poe_max_attempts < 1:
            raise ValueError("poe_max_attempts must be a positive integer")
        if self.global_seed < 0:
            raise ValueError("global_seed cannot be negative")
        expected_subtasks = {"add", "delete", "substitute"}
        if set(self.editable_nodes_by_subtask) != expected_subtasks:
            raise ValueError("editable_nodes_by_subtask must define add/delete/substitute")
        frozen_nodes = {
            name: tuple(nodes) for name, nodes in self.editable_nodes_by_subtask.items()
        }
        if any(len(nodes) != len(set(nodes)) for nodes in frozen_nodes.values()):
            raise ValueError("editable node lists cannot contain duplicates")
        object.__setattr__(
            self,
            "editable_nodes_by_subtask",
            MappingProxyType(frozen_nodes),
        )

    def requested_edit_count(
        self,
        random_source: object,
        *,
        maximum_available: int | None = None,
    ) -> int:
        """Return an edit count capped by the current DAG's real capacity."""

        if self.edit_count_mode == "fixed":
            return self.fixed_edit_count
        if maximum_available is not None and (
            type(maximum_available) is not int or maximum_available < 1
        ):
            raise ValueError("maximum_available must be a positive integer or None")
        if self.edit_count_mode == "maximum":
            if maximum_available is None:
                raise ValueError("maximum mode requires maximum_available")
            return maximum_available
        if self.max_edit_count is None:
            if maximum_available is None:
                raise ValueError(
                    "maximum_available is required when max_edit_count is automatic"
                )
            upper = maximum_available
        elif maximum_available is None:
            upper = self.max_edit_count
        else:
            upper = min(self.max_edit_count, maximum_available)
        if self.min_edit_count > upper:
            raise ValueError("current DAG cannot satisfy the configured minimum edit count")
        randint = getattr(random_source, "randint", None)
        if not callable(randint):
            raise TypeError("random_source must provide randint")
        return randint(self.min_edit_count, upper)


# Demo 的参考 tokenizer，不等于 Poe 或下游化学模型的实际 tokenizer。
# 统计完整 serialized.text；不包含模型自行添加的 BOS/EOS 或 chat template。
DEMO_TOKEN_ENCODING = "cl100k_base"

DEFAULT_HALLUCINATION_CONFIG = HallucinationGenerationConfig(
    edit_count_mode=EDIT_COUNT_MODE,
    fixed_edit_count=FIXED_EDIT_COUNT,
    min_edit_count=MIN_EDIT_COUNT,
    max_edit_count=MAX_EDIT_COUNT,
    include_reasoning_steps=INCLUDE_REASONING_STEPS,
    include_final_answer=INCLUDE_FINAL_ANSWER,
    final_answer_probability=FINAL_ANSWER_PROBABILITY,
    editable_nodes_by_subtask=EDITABLE_NODES_BY_SUBTASK,
    integer_deltas=INTEGER_DELTAS,
    count_min_value=COUNT_MIN_VALUE,
    float_mutation_mode=FLOAT_MUTATION_MODE,
    float_absolute_deltas=FLOAT_ABSOLUTE_DELTAS,
    float_relative_changes=FLOAT_RELATIVE_CHANGES,
    float_decimal_places=FLOAT_DECIMAL_PLACES,
    atom_index_prefer_same_element=ATOM_INDEX_PREFER_SAME_ELEMENT,
    replacement_elements=REPLACEMENT_ELEMENTS,
    fragment_similarity_min=FRAGMENT_SIMILARITY_MIN,
    fragment_similarity_max=FRAGMENT_SIMILARITY_MAX,
    fragment_target_similarity=FRAGMENT_TARGET_SIMILARITY,
    fragment_require_same_charge=FRAGMENT_REQUIRE_SAME_CHARGE,
    fragment_max_heavy_atom_difference=FRAGMENT_MAX_HEAVY_ATOM_DIFFERENCE,
    smiles_mutation_operators=SMILES_MUTATION_OPERATORS,
    smiles_replacement_atomic_numbers=SMILES_REPLACEMENT_ATOMIC_NUMBERS,
    smiles_similarity_min=SMILES_SIMILARITY_MIN,
    smiles_similarity_max=SMILES_SIMILARITY_MAX,
    require_valid_smiles=REQUIRE_VALID_SMILES,
    require_different_from_reference=REQUIRE_DIFFERENT_FROM_REFERENCE,
    enable_deterministic_propagation=ENABLE_DETERMINISTIC_PROPAGATION,
    fail_on_trivial_edge_violation=FAIL_ON_TRIVIAL_EDGE_VIOLATION,
    poe_api_key_env=POE_API_KEY_ENV,
    poe_bot_name=POE_BOT_NAME,
    poe_temperature=POE_TEMPERATURE,
    poe_max_attempts=POE_MAX_ATTEMPTS,
    poe_cache_directory=POE_CACHE_DIRECTORY,
    emit_matched_negative=EMIT_MATCHED_NEGATIVE,
    global_seed=GLOBAL_SEED,
)


__all__ = [
    "DEFAULT_HALLUCINATION_CONFIG",
    "HallucinationGenerationConfig",
    "MUTATION_CATEGORIES",
]
