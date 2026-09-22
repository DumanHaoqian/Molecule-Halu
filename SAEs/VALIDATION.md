# 本次实现验证

验证日期：2026-09-22。机器可读回执：[`provenance/implementation_validation.json`](provenance/implementation_validation.json)。

- **55 项测试全部通过**：冻结上下文/token mask、post-token/block25 位置、缓存中断与篡改拒绝、五种架构梯度/学习/保存重载、训练集归一化、精确续训、重建统计、CE 对齐和高激活 token 排名。日志：[`logs/tests-final.log`](logs/tests-final.log)。6 条警告来自 W&B SDK 内部弃用接口，不影响通过结果。
- 五种架构均完成真实训练循环的 **CPU 合成集成测试**：输入维度16，512个训练/128个验证激活行，每种16步；检查点和完整验证报告均已生成。每种架构均有实际离线 W&B `.wandb` 文件。产物：[`artifacts/smoke_cpu_verified`](artifacts/smoke_cpu_verified)，日志：[`logs/cpu-integration-final.log`](logs/cpu-integration-final.log)。**这些数值不能作为 ChemDFM SAE 质量结果。**
- 独立代码检查未发现实质性正确性问题；额外复跑21个 CPU 测试通过（1个 W&B 测试未选择）。
- 迁移后语料完成全量验证，170个输出文件的原始校验和不变；完成封印和当前代码指纹一致。原元数据归档在 `provenance/pre_relocation`。
- 五种正式配置均通过无参数分配检查。CPU 激活预检查通过：Layer26/block25、5120维、10,000,474/443,731 tokens、原始 FP16 缓存约99.60 GiB。
- `pip check` 通过；代码编译检查和主要 CLI 入口通过，实际运行了独立 `eval_sae.py` CLI。

尚未验证的真实运行：14B 模型 GPU 激活抽取、真实数据 SAE 收敛、真实 LM 的 ΔCE/恢复率、在线 W&B 网络/账号权限，以及 SAE 特征的幻觉预测质量。本次按要求只准备代码和运行配置，未启动这些任务。

当前 `.venv` 复用本机 PyTorch 基础环境，额外依赖安装在项目内。重建说明及未来运行命令见 [`README.md`](README.md)。
