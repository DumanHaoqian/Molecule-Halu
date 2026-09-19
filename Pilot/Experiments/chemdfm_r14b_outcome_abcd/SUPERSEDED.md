# 已由 v2 实验替代

本目录的原始结果存在实验设计问题，不应继续用于判断 hallucinated reasoning 对 outcome 的因果影响：B/C 输入泄漏正确 `PRODUCT_SMILES`，A/D 缺少 plain source SMILES，旧指标标签与实际 benchmark 判定不一致，并存在重复生成和长度截断。

修正后的实验与结论位于：

`/home/haoqian/Data/Molecule/Pilot/Experiments/chemdfm_r14b_outcome_abcd_v2/`

旧文件保留用于追溯，没有删除或覆盖。
