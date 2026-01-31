# 问题四（r4）摘要文档

本摘要用于快速了解 r4 的模拟框架、诊断方式与输出清单。具体结果以 `r4/outputs/interpretation_report.md` 为准。

## 1. 目标
- 评估 AMEI 2.0 相对于 CBR/CBP 的公平性与争议拦截能力
- 检验不同权重与阈值设定对结果的影响

## 2. 关键机制
- 周内分位数阈值：`z_soft_q/z_hard_q`（默认 0.20/0.05）
- 动态权重：`w_start/w_end` 随赛季推进通过 logistic 过渡
- 多目标约束权重搜索：拦截率下限 + EMR/B2CR 跌幅约束；无可行点时启用 soft penalty
- 按赛季类型自适应权重：分别对 JS/非 JS 赛季搜索权重
- Judge Save 规则 B：Bottom‑2 中淘汰 **J_total + 反向排名** 较低者
- Rule C（敏感性）：J_total + 0.5 * 上周排名反向得分

## 3. 诊断与稳健性
- Bootstrap 均值 CI + 差值 CI
- McNemar + Holm 校正
- 争议拦截：分位数敏感性 + 绝对阈值对照
- 争议定义替代：粉丝‑评委排名差距（fan‑judge gap）
- 公平性 Spearman 稳健性

## 4. 结果与稳健性（最新一次运行）
- 权重优化：w_start=0.80, w_end=0.50, k=4.0, t0=0.60, min_vote_weight=0.03；tradeoff_score=0.171，interception_mean=0.571；EMR/B2CR 约束未同时满足（无可行点）
- soft penalty：λ=1 与主选一致；λ>=2 转向 w_start=0.75, w_end=0.20, k=16.0, t0=0.40（tradeoff_score=0.421）但仍不可行；建议默认 λ=1
- 网格范围：w_start 0.55-0.80，w_end 0.20-0.50，k 4-16，t0 0.40-0.60，min_vote_weight 0.03-0.10
- Judge Save 规则：A/B 相同（AMEI 0.375；CBR/CBP 0.393），Rule C(0.5) 略低（AMEI 0.339；CBR 0.357；CBP 0.375）；系数敏感性显示 0.3-0.4 最优（AMEI 0.393，推荐区间）
- 核心指标速览（按周数加权）：

| 指标（范围） | AMEI | CBR | CBP | Δ 相对最优基线 |
|---|---:|---:|---:|---:|
| EMR（无 JS） | 0.347 | 0.504 | 0.508 | -0.161 |
| B2CR（有 JS） | 0.534 | 0.699 | 0.699 | -0.165 |
| JS-EMR（Rule B） | 0.375 | 0.393 | 0.393 | -0.018 |
| Interception（全周） | 0.571 | 0.095 | 0.190 | +0.381 |
| Spearman（全周） | 0.952 | 0.904 | 0.825 | +0.048 |

说明：按周数加权（`r4/outputs/comparison_metrics.csv`），Δ 相对最优基线取 max(CBR, CBP)，正值表示 AMEI 优势。

- 公平/兴奋度损失（越低越好）：Fairness 48.249 vs 79.569/79.569；Excitement 0.044 vs 1.569/0.021
- 指标口径：EMR（无 JS）= 预测淘汰与实际一致比例；B2CR（有 JS）= 实际淘汰是否落入 Bottom-2；JS-EMR（Rule B）= 底部二人套用评审救人规则后的淘汰一致率
- Interception：争议选手落入 Bottom-2 的比例（默认 z/p 分位阈值）；Spearman：评审排名与方法排名的周内相关
- 解读：AMEI 主动牺牲 EMR/B2CR 换取争议拦截与公平‑兴奋度平衡，负向 Δ 属于设计目标
- 争议拦截稳健：分位数网格下 AMEI 平均 0.534（0.261-0.759），绝对阈值下 AMEI 0.846；自动选择阈值 z_q=0.10, p_q=0.75
- 自适应权重：JS/非JS 搜索得到一致参数（w_start=0.80, w_end=0.50, k=4.0, t0=0.60, min_vote_weight=0.03），综合 EMR=0.347，B2CR=0.534，Interception=0.571，Spearman=0.952
- 争议 gap 定义（fan‑judge gap, q=0.8）：AMEI 0.383 vs CBR 0.244 vs CBP 0.277
- 稳定性：Spearman std=0.023（IQR=0.024），排名一致性稳定；Interception std=0.456
- 争议稀疏：17/34 赛季无争议事件，总计 21 起（0.62/赛季），解释 Interception 波动
- LOSO：均值优势 EMR -0.219，B2CR -0.202，Interception +0.353，Spearman +0.032；负向主要出现在 season 19/2/1（EMR）与 season 32/34/28（B2CR），其中多为短赛季或争议稀疏

## 5. 关键输出
- `r4/outputs/interpretation_report.md`
- `r4/outputs/summary.md`
- `r4/outputs/README.md`
- `r4/outputs/weight_grid.csv` / `weight_selected.csv`
- `r4/outputs/weight_grid_js.csv` / `weight_grid_nojs.csv`
- `r4/outputs/weight_selected_js.csv` / `weight_selected_nojs.csv` / `adaptive_weight_metrics.csv`
- `r4/outputs/weight_selected_soft.csv` / `weight_selected_soft_js.csv` / `weight_selected_soft_nojs.csv`
- `r4/outputs/bootstrap_diff_ci.csv`
- `r4/outputs/controversy_sensitivity.csv` / `controversy_alt_thresholds.csv` / `threshold_selected.csv`
- `r4/outputs/controversy_alt_definitions.csv`
- `r4/outputs/loso_summary.csv`
- `r4/outputs/interception_impact.csv`
- `r4/outputs/fairness_spearman.csv`
- `r4/outputs/season_stability.csv` / `season_stability_summary.md` / `season_stage_stability.csv`
- `r4/outputs/judge_save_rule_comparison.csv`
- `r4/outputs/judge_save_coeff_grid.csv`

## 6. 运行方式

```bash
python r4/req4_analysis.py
```

可调参数示例：

```bash
python r4/req4_analysis.py \
  --z-soft-q 0.2 --z-hard-q 0.05 \
  --min-interception 0.4 \
  --tie-seed 42
```
