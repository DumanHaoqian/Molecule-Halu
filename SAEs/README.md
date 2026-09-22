# ChemDFM-R Layer 26 SAE

SAE 的语料、审计、激活提取、训练和评估集中在本目录。原 `Pilot_v2/sae` 已整体迁入；语料调查移到 `audits/corpus_inventory`。Pilot 的标注数据和幻觉探针实验仍在 `../Pilot_v2`。

**当前状态：代码、环境和 CPU 集成测试已准备。尚未抽取真实 ChemDFM 激活，也未启动 GPU SAE 训练或在线 W&B run。** 下列 GPU 命令用于之后启动；本次只运行合成数据的 CPU 测试。使用说明以本文和 `configs/` 为准。

## 目录

```text
data/v1/                        已验证文本语料和固定 train/validation 索引
audits/corpus_inventory/         原始数据调查与重叠检查
configs/                        五种 SAE 的训练 YAML
activation_inputs.py            完整聊天上下文分词与 assistant token 定位
extract_activations.py           第 26 层激活缓存、断点续传和输入校验
activation_store.py              mmap 读取、洗牌、训练集归一化
sae_models.py                    SAE-Lens 架构适配和一致的推断接口
sparsemax_attention_sae.py       适配旧项目的 sparsemax attention SAE
train_sae.py                    训练、验证、W&B、检查点和续训
eval_sae.py                     留出激活上的重建、稀疏度、特征使用率评估
delta_lm.py                     替换激活后的语言模型损失评估
feature_examples.py             指定特征的高激活 token 与文本上下文
activations/                    后续生成的 FP16 激活缓存（目前未生成真实缓存）
runs/                           后续正式训练的检查点和曲线
artifacts/                      CPU 合成测试产物，不能作为科学实验结果
provenance/                     迁移回执、迁移前元数据、环境版本
reference/ChemDFM_SAE_Training/  用户旧仓库的只读参考副本
```

## 数据和激活定义

- 模型：本地 `../chemical_models/ChemDFM-R-14B`，5120 维。
- **Layer 26 = 零索引 block 25 输出**，post-token residual，位于最终 LayerNorm 之前。AutoModel 中是 `layers[25]`；AutoModelForCausalLM 中是 `model.layers[25]`。
- 完整 `messages` 经模型原有聊天模板渲染，以 teacher forcing 获取文本激活。只保存 assistant 文本的有效 token；排除特殊符号、纯空白和 `<think>/<answer>` 控制标签。没有把 provenance、参考答案或标签送入模型。
- 保留完整文档上下文，不拼接不同文档、不移动 token 位置、不截断原有合格文档。目标层捕获后提前结束 forward，避免继续计算后续层和词表 logits。
- 已固定按分子组隔离的 train/validation；不会重新把激活行随机切分为验证集。Pilot 的 600 道题已按既定文本/CID/分子 connectivity 规则排除；不保证排除所有语义近重复或相同骨架。

首轮 `data/v1/selections/pilot_10m`：

| split | 文档 | 有效 assistant tokens |
|---|---:|---:|
| train | 19,482 | 10,000,474 |
| validation | 883 | 443,731 |

两者原始 FP16 激活约 **99.60 GiB**，另需元数据和检查点空间。完整主语料有 136,150,091 个训练 tokens；可选补充和仅题目的候选池见 [语料说明](data/v1/README.md)。当前配置先使用上述 10M 固定子集，不能据此宣称 SAE 已收敛。

迁移后重新全量验证：1,628,920 条保留记录，170 个输出文件的原有校验和保持一致。见 [迁移回执](provenance/migration.json) 和 [数据验证](data/v1/verification.json)。原始构建记录在 `provenance/pre_relocation`；历史日志中的旧路径保留用于追溯。

## SAE 设计

默认使用 `configs/batchtopk.yaml`：输入 5120、字典 16,384、平均目标 k=64、batch=1024、Adam lr=3e-4、100 步 warmup、cosine decay，首轮遍历固定训练子集一次。只用训练集样本拟合向量均值与全局标量 RMS，保存到检查点，评估时做逆变换。没有逐 token 单位化输入。

| 配置 | 稀疏机制 | 需要关注 |
|---|---|---|
| `batchtopk.yaml` | 训练 batch 内共享 TopK 预算 | 推断使用训练阈值，单 token 结果不依赖评估 batch |
| `topk.yaml` | 每个 token 最多 k 个非零特征 | 固定上限，便于建立基线 |
| `jumprelu.yaml` | 可学习阈值 + L0 正则 | k 仅记录比较目标，实际 L0 不保证等于 k |
| `matryoshka.yaml` | 多个嵌套字典宽度联合训练 | 同时评估前缀字典的重建质量 |
| `sparsemax_attention.yaml` | 对字典 key 的 sparsemax 稀疏选择 | 延续旧项目；使用可微稀疏度代理，非固定 TopK |

前四种使用 SAE-Lens 6.51.1 的原生训练实现；TopK 系列保留 decoder norm 缩放和 AuxK 等原生机制，未另加相冲突的单位范数约束。JumpReLU 在更新后保持非负阈值。BatchTopK 尚无有效阈值时，推断临时使用每 token TopK，评估报告会标明。

**不同架构的训练 loss 数值不能直接横比**：SAE-Lens 的重建 loss 按输入维度求和，旧 sparsemax 使用按元素平均；正则项也不同。比较时应看同一验证集上统一定义的 FVU、实际 L0 和语言模型损失，必要时调正则获得相近 L0。

## 环境

```bash
cd /home/haoqian/Data/Molecule/SAEs
.venv/bin/python -m pip check
```

本机 `.venv` 已准备，复用已有 CUDA PyTorch 环境的系统包，新增依赖只安装在本目录虚拟环境内，没有修改共享 conda 环境。直接依赖见 `requirements.txt`，实测版本见 `requirements-lock.txt` 与 `provenance/environment.json`。

如需重建，可运行 `bash scripts/setup_env.sh /path/to/python3.10`。脚本使用 `--system-site-packages`；换机器应先准备匹配 CUDA 的 PyTorch 环境。

## 运行顺序

### 1. CPU 预检查

```bash
.venv/bin/python extract_activations.py --preflight
.venv/bin/python train_sae.py --config configs/batchtopk.yaml --validate-config
```

前者验证语料封印、选样、模型/tokenizer 指纹和来源文件，报告空间预算；后者检查架构配置并估算参数/优化器内存，不分配 SAE 参数、不启动 W&B。缓存尚未生成时会明确报告 `cache_exists: false`。`--dry-run` 还要求真实缓存已完成并验证缓存。

### 2. 获取激活（之后 GPU 空闲时运行）

先用少量文档做真实 GPU 冒烟验证，另存目录：

```bash
.venv/bin/python extract_activations.py --gpus 4,5,6,7 \
  --limit-documents 2 --output activations/layer26/gpu_smoke
```

确认通过后获取完整子集：

```bash
.venv/bin/python extract_activations.py --gpus 4,5,6,7
# 如中断，恢复相同目录和参数：
.venv/bin/python extract_activations.py --gpus 4,5,6,7 --resume
```

加载 BF16 模型，保存 FP16 `.npy` 分片及每个 token 的来源。默认每张可见卡要求至少 37 GiB 空闲；不足时明确退出。可以显式减少 `--gpus` 并调整 `--max-memory-gib`。不会结束其他人的进程，也不会静默改用别的卡或 CPU/disk offload。

缓存每个分片在文档边界提交；中断后从最后已提交文档继续。完整缓存才可训练。已完成分片做校验和验证，配置、数据或模型身份不匹配时拒绝混用。

### 3. 训练和实时 W&B

先完成 W&B 登录；密钥不要写入 YAML：

```bash
.venv/bin/python -m wandb login
CUDA_VISIBLE_DEVICES=4 .venv/bin/python train_sae.py \
  --config configs/batchtopk.yaml --device cuda:0
```

这里 `cuda:0` 是可见物理 GPU 4。若不设置 `CUDA_VISIBLE_DEVICES`，配置的 `cuda:4` 指物理 GPU 4。各架构是独立单 GPU SAE 训练任务，可在 4–7 卡分别运行不同配置；ChemDFM 激活提取支持跨卡分配模型。当前没有实现单个 SAE 的 DDP 分布式训练。

默认在线项目：`chemdfm-layer26-sae`。曲线包含总 loss、重建 MSE、各损失项、L0、学习率、梯度范数、从未激活特征比例、吞吐率，以及周期性 `val/*`。同时写本地 `metrics.jsonl`。可用 `--wandb-mode offline` 或 `disabled`。

```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python train_sae.py \
  --config configs/batchtopk.yaml --device cuda:0 \
  --resume runs/batchtopk_layer26/last.pt
```

`best.pt` 按验证 NMSE 选择，`last.pt` 保存最新已提交训练状态；均附校验回执。保存模型、优化器、随机数、采样游标、归一化、数据指纹、代码哈希和 W&B run ID。`--max-steps N` 是绝对停止步数，不改变原定学习率计划。意外故障恢复到最近检查点；尚未提交的步会重跑。不同设备/库版本可能带来浮点差异。

YAML 相对路径以配置文件所在目录解析。输出目录非空时拒绝覆盖；改架构或优化器设置需使用新运行目录。当前训练器是单遍预算，`max_tokens` 超过缓存规模时会明确报告单遍上限。

### 4. 多种评估

```bash
.venv/bin/python eval_sae.py --checkpoint runs/batchtopk_layer26/best.pt \
  --device cuda:0 --gpus 4 --output runs/batchtopk_layer26/eval_full.json

.venv/bin/python delta_lm.py --checkpoint runs/batchtopk_layer26/best.pt \
  --gpus 4,5,6,7 --max-documents 64 \
  --output runs/batchtopk_layer26/delta_lm.json
```

| 视角 | 指标/产物 | 含义 |
|---|---|---|
| 激活保真度 | MSE、FVU/NMSE、1−FVU、余弦相似度、相对 L2 误差 | 是否保留原始表示 |
| 稀疏性 | L0 均值、标准差、分位数 | 每个 token 实际使用多少特征 |
| 字典使用 | 全验证集不活跃比例、密集特征比例、每特征频率 | 是否出现明显死特征或过度使用 |
| 嵌套字典 | Matryoshka 各宽度的指标 | 较小字典是否仍能有效重建 |
| 语言模型保真度 | clean/reconstruction/mean/zero CE、ΔCE、loss recovery | 用重建替换 Layer 26 后是否影响语言预测 |
| 可解释性检查 | 高激活 token 和完整来源/上下文 | 人工查看特征对应的化学概念或模式 |

FVU 由全验证流的总误差和全局均值方差计算，不取每个 batch FVU 的简单平均。训练期间默认验证前 100k tokens，最终 `eval_sae.py` 默认使用全部 443,731 个验证 tokens。`explained_variance` 另外对残差去均值，与 `1−FVU` 不完全相同。

ΔLM 默认替换 assistant 当前 token 的层输出；对 assistant 目标 token t 使用 `logits[t−1]` 计算 CE，按有效 token 数加权。首个 assistant 目标的预测位置可能未替换。`--patch-scope all` 可评估包含 prompt/control 在内的全序列干预，分布范围更广，应单独报告。恢复率分母近零时不给数值，分母非正会显式标记。

上述指标不能单独证明“检测到了幻觉”。后续应冻结 SAE，在原 Pilot 固定 train/validation/test 上训练特征探针，再与 hidden-state 和 TF-IDF 基线比较 AUPRC/AUROC/F1；本次没有运行或声称完成该研究评估。

查看指定特征的高激活文本（只读取 validation）：

```bash
.venv/bin/python feature_examples.py --checkpoint runs/batchtopk_layer26/best.pt \
  --features 0 7 42 --top-k 10 --output runs/batchtopk_layer26/feature_examples.json
```

它检查缓存、元数据和文本来源，再给出 token 位置、文本区间、上下文、来源和激活值。特征编号仅为查看示例；不能先把某个编号解释成特定化学概念。

## 验证与复现

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python smoke_test.py --output artifacts/my_new_cpu_smoke
.venv/bin/python verify.py data/v1
```

`smoke_test.py` 只在 CPU 上用 16 维合成激活测试五种架构，默认写离线 W&B；不会加载 ChemDFM。测试覆盖冻结 mask、层位置、缓存中断、源数据篡改、架构梯度、重载、归一化、全局指标、因果 CE 对齐和精确续训。测试日志、集成产物及最终核验见 `logs/` 和 `VALIDATION.md`。

实现参考：[用户原仓库](https://github.com/DumanHaoqian/ChemDFM_SAE_Training)，固定 commit `c9b6d55013a321571784943a0c7135d38d6df2ab`。复用了其架构选择与 sparsemax 思路，替换了旧模型路径、token 级 holdout、层编号和缓存接口。W&B 接口依据 [官方记录指标说明](https://docs.wandb.ai/models/track/log)。
