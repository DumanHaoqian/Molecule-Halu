# v10 开发集覆盖记录

日期：2026-09-20。v10 的 18 个预定开发 origin 已全部处理：接受 2 个，拒绝 16 个，接受率为 2/18（11.11%）。未执行目标模型效果评估，因此没有 v10 的 A/B/C/D/E 准确率或 H<A 结论。

拒绝类别按保存的结果记录统计：

| 保存的类别 | origin 数 |
| --- | ---: |
| `local_environment_unavailable` | 3 |
| `reference_semantic_uncertain` | 5 |
| `audit_unresolved_defects` | 8 |

后两类是当时版本中具有否决作用的模型审查结果。这里只记录其返回状态，不把每条判断自动认定为真实化学错误，也不把它们统一判为误报。原始意见与证据保留在各 origin 的审查文件中。

v11 保留程序化化学执行、局部结构构造、确定性渲染、标注和泄漏检查，以这些程序检查作为发布硬条件；模型审查改为保留的诊断意见。故 v11 的“接受”不再表示所有 Agent 一致认可，两个版本接受率的差异不能解释为同一验收标准下的化学质量提升。

v11 同一批 18 个开发 origin 的终态为程序接受 15 个、局部环境构造失败 3 个。15 个已接受记录中，参考审查争议 6 个、构造合规审查争议 11 个，两项同时一致 3 个；盲审泄漏字段为 false 的 15 个。false 表示对应模型未报疑，不证明不存在泄漏。所有争议继续保留，不能因程序接受而改记为已解决。

这些接受/拒绝决定未使用目标模型 A/B/C/D/E 输出筛选 origin。v9 的开发效果门槛未通过；其日志、结果及不进入保留集的决定独立保存。v10 没有保留集评估结果。

来源：

- [v10 终态记录](../Experiments/agent_corruption_v10/development18/summary.json)
- [v11 终态记录](../Experiments/agent_corruption_v11/development18/summary.json)
- [v11 程序验收及模型审查报告](../Experiments/agent_corruption_v11/development18/program_review_report.md)
- [v9 开发门槛记录](../Experiments/agent_corruption_v9/development18/v9briefgate.json)
