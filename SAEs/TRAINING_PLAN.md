# 当前训练设计

以 [README.md](README.md)、[五种架构配置](configs/) 和 [实施计划](docs/plans/2026-09-22-training.md) 为准。

此前仅 TopK 的草案已被当前多架构实现替代，原文保存在 `provenance/pre_relocation/TRAINING_PLAN.md`。当前默认 BatchTopK 16k / k64；模型仍为 ChemDFM-R-14B Layer26（block index25）post-token 表示。仅 CPU 软件验证已运行，真实激活提取和 GPU 训练待后续启动。
