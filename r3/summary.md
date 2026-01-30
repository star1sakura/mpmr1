# 问题三（r3）摘要文档

本摘要用于快速了解 r3 的建模思路、诊断方式与关键输出。具体结果以 `r3/outputs/summary.md` 与 `r3/outputs/interpretation_report.md` 为准。

## 1. 目标与数据
- 解释评审评分与粉丝投票的影响因素，并评估对淘汰风险的综合影响。
- 输入：`r1/outputs/req1_features.csv` 与原始赛季信息（含名人年龄、行业、地区等）。

## 2. 关键处理与特征
- 稳定合并键：`contestant_key = season + celebrity_name + ballroom_partner`
- 地域映射：`homecountry/region + homestate` → International / Northeast / Midwest / South / West / Unknown
- 标准化：`J_z`（season-week 标准化），`age_z` / `week_z`（z-score）
- 粉丝投票：`p_hat` 裁剪到 `(0,1)` 得 `p_hat_clip`

## 3. 模型框架
- Model J（评审评分）：线性混合效应模型（固定效应 + 舞伴/行业随机效应）
- Model F（粉丝投票）：Beta 混合效应模型（bambi/PyMC）
- Model P（综合表现）：Time‑varying Cox + 进入决赛的 Logit

## 4. 诊断与稳健性
- 诊断：LOO/WAIC + Pareto‑k + PPC（分布与均值）
- 模型对比：M0 vs M3
  - M0：无随机效应
  - M3：舞伴 + 行业随机效应
- K‑fold：按 `contestant_key` 分组；稀有区域固定留在训练集以避免新类别错误

常见结论模式（以最新输出为准）：
- LOO 倾向支持 M3（拟合更优）
- K‑fold 可能倾向 M0（泛化收益有限）

## 5. 关键输出
- `r3/outputs/summary.md`：论文式摘要页（自动生成）
- `r3/outputs/interpretation_report.md`：完整解读报告
- `r3/outputs/model_diagnostics.csv`：LOO/WAIC + Pareto‑k 摘要
- `r3/outputs/model_comparison.csv` / `kfold_results.csv`：模型对比与折内结果
- `r3/outputs/beta_fan_effects.csv` / `beta_marginal_effects.csv`
- `r3/outputs/cox_model_summary_coef.csv`

## 6. 运行方式
基础运行：

```bash
python r3/req3_analysis.py
```

仅比较 M0 vs M3（更稳健的 K‑fold）：

```bash
python r3/req3_analysis.py \
  --compare-models --compare-minimal \
  --compare-folds 3 \
  --compare-draws 2000 --compare-tune 2000 \
  --compare-chains 4 --compare-target-accept 0.99
```
