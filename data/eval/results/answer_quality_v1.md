# RAG Law V1 端到端评测报告

- 生成时间：2026-10-16T21:05:27+08:00
- 评测集：`C:\Users\issuser\Desktop\jerry\RAG\rag_law\data\eval\questions_v1.yaml`
- 题目数量：70
- 质量门槛口径：`holdout`
- LLM：`deepseek-chat`
- Prompt：`stage5-v1-behavior-calibrated`

## 核心指标

| 指标 | 结果 |
|---|---:|
| Recall@10 | 0.900 |
| MRR | 0.827 |
| nDCG@10 | 0.827 |
| 引用精确率 | 1.000 |
| 核心法条标注精确率 | 0.315 |
| 引用完整率 | 0.788 |
| 证据忠实度 | 1.000 |
| 不可回答题行为准确率 | 0.533 |
| P95 总延迟 | 13957.0 ms |

## 质量门槛

| 门槛 | 目标 | 实际 | 通过 |
|---|---:|---:|:---:|
| retrieval_recall_at_10 | 0.9 | 0.9 | 是 |
| citation_precision | 1.0 | 1.0 | 是 |
| fabricated_citation_rate | 0.0 | 0.0 | 是 |
| unanswerable_behavior_accuracy | 0.9 | 0.5333333333333333 | 否 |
| pipeline_error_count | 0 | 0 | 是 |

## 待处理题目

| ID | Split | 题型 | 预期行为 | 实际行为 | 引用完全匹配 | 错误 |
|---|---|---|---|---|:---:|---|
| q031 | holdout | 直接查询 | answered | answered | 否 | - |
| q032 | holdout | 直接查询 | answered | answered | 否 | - |
| q033 | holdout | 直接查询 | answered | answered | 否 | - |
| q035 | holdout | 直接查询 | answered | answered | 否 | - |
| q036 | holdout | 直接查询 | answered | answered | 否 | - |
| q037 | holdout | 直接查询 | answered | answered | 否 | - |
| q039 | holdout | 直接查询 | answered | answered | 否 | - |
| q040 | holdout | 直接查询 | answered | answered | 否 | - |
| q041 | holdout | 直接查询 | answered | answered | 否 | - |
| q042 | holdout | 直接查询 | answered | answered | 否 | - |
| q043 | holdout | 直接查询 | answered | answered | 否 | - |
| q044 | holdout | 直接查询 | answered | answered | 否 | - |
| q045 | holdout | 直接查询 | answered | answered | 否 | - |
| q046 | holdout | 口语场景 | answered | answered | 否 | - |
| q047 | holdout | 口语场景 | answered | answered | 否 | - |
| q048 | holdout | 口语场景 | answered | answered | 否 | - |
| q049 | holdout | 口语场景 | answered | answered | 否 | - |
| q050 | holdout | 口语场景 | answered | answered | 否 | - |
| q051 | holdout | 口语场景 | answered | answered | 否 | - |
| q052 | holdout | 口语场景 | answered | answered | 否 | - |
| q053 | holdout | 口语场景 | answered | answered | 否 | - |
| q054 | holdout | 口语场景 | answered | answered | 否 | - |
| q055 | holdout | 口语场景 | answered | answered | 否 | - |
| q056 | holdout | 口语场景 | answered | answered | 否 | - |
| q057 | holdout | 口语场景 | answered | answered | 否 | - |
| q058 | holdout | 口语场景 | answered | answered | 否 | - |
| q059 | holdout | 口语场景 | answered | answered | 否 | - |
| q060 | holdout | 口语场景 | answered | answered | 否 | - |
| q061 | holdout | 口语场景 | answered | answered | 否 | - |
| q062 | holdout | 口语场景 | answered | generation_failed | 否 | - |
| q064 | holdout | 口语场景 | answered | needs_clarification | 否 | - |
| q065 | holdout | 口语场景 | answered | answered | 否 | - |
| q066 | holdout | 跨条文 | answered | needs_clarification | 否 | - |
| q067 | holdout | 跨条文 | answered | answered | 否 | - |
| q068 | holdout | 跨法律 | answered | answered | 否 | - |
| q069 | holdout | 跨法律 | answered | answered | 否 | - |
| q070 | holdout | 跨法律 | answered | insufficient_evidence | 否 | - |
| q071 | holdout | 跨条文 | answered | answered | 否 | - |
| q072 | holdout | 跨法律 | answered | answered | 否 | - |
| q073 | holdout | 跨法律 | answered | answered | 否 | - |
| q074 | holdout | 跨法律 | answered | answered | 否 | - |
| q075 | holdout | 跨条文 | answered | answered | 否 | - |
| q076 | holdout | 指定条号 | answered | answered | 否 | - |
| q078 | holdout | 指定条号 | answered | answered | 否 | - |
| q080 | holdout | 指定条号 | answered | answered | 否 | - |
| q081 | holdout | 指定条号 | answered | answered | 否 | - |
| q082 | holdout | 指定条号 | answered | answered | 否 | - |
| q083 | holdout | 指定条号 | answered | generation_failed | 否 | - |
| q085 | holdout | 指定条号 | answered | answered | 否 | - |
| q086 | holdout | 事实不足 | needs_clarification | insufficient_evidence | 是 | - |
| q087 | holdout | 事实不足 | needs_clarification | answered | 否 | - |
| q088 | holdout | 事实不足 | needs_clarification | answered | 否 | - |
| q089 | holdout | 事实不足 | needs_clarification | answered | 否 | - |
| q094 | holdout | 超范围拒答 | insufficient_evidence | answered | 否 | - |
| q099 | holdout | 超范围拒答 | insufficient_evidence | answered | 否 | - |
| q100 | holdout | 超范围拒答 | insufficient_evidence | answered | 否 | - |

最终结论：仍有质量门槛未通过。
