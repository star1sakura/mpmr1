# AMEI 2.0 Simulation Results Summary

## Overall Performance Comparison

- Threshold mode: quantile (z_soft_q=0.20, z_hard_q=0.05)
- Controversy thresholds: z_q=0.20, p_q=0.80
- Weight optimization constraint: interception >= 0.40
- EMR/B2CR constraints: drop <= 0.05 absolute and <= 10% relative vs CBR/CBP
- Optimized weights: w_start=0.80, w_end=0.50, k=4.0, t0=0.60, min_vote_weight=0.03
- Selected point feasible: False (interception=0.571)

## Core Metrics Snapshot (Weighted by Weeks)

Values are weighted by weeks using `r4/outputs/comparison_metrics.csv`.

| Metric (Scope) | AMEI | CBR | CBP | Δ vs max baseline |
|---|---:|---:|---:|---:|
| EMR (No JS) | 0.347 | 0.504 | 0.508 | -0.161 |
| B2CR (JS) | 0.534 | 0.699 | 0.699 | -0.165 |
| JS-EMR (JS, Rule B) | 0.375 | 0.393 | 0.393 | -0.018 |
| Interception (All) | 0.571 | 0.095 | 0.190 | +0.381 |
| Spearman (All) | 0.952 | 0.904 | 0.825 | +0.048 |

Δ vs max baseline uses max(CBR, CBP); positive favors AMEI.

Lower is better:

- Fairness loss: AMEI 48.249 vs CBR/CBP 79.569
- Excitement loss: AMEI 0.044 vs CBR 1.569 vs CBP 0.021

## Metric Definitions

- EMR (No JS): share of non-Judge Save weeks where predicted elimination matches actual elimination
- B2CR (JS): share of Judge Save weeks where the actual eliminated contestant appears in the model Bottom-2
- JS-EMR (JS): share of Judge Save weeks where the judge-save rule (Rule B) applied to model Bottom-2 matches actual elimination
- Interception: fraction of controversy contestants captured in Bottom-2 (controversy defined by z/p quantiles unless noted)
- Spearman: rank correlation between judge ranking and method ranking within a week

Interpretation: AMEI intentionally trades EMR/B2CR for stronger controversy interception and improved fairness-excitement balance.

### Elimination Match Rate (EMR) - Seasons without Judge Save

| Method | EMR | Δ vs CBR |
|--------|-----|----------|
| AMEI 2.0 | 0.347 | -31.1% |
| CBR | 0.504 | 0.0% |
| CBP | 0.508 | +0.8% |

### Controversy Interception Rate (No Judge Save)

| Method | Interception Rate |
|--------|-------------------|
| AMEI 2.0 | 0.634 |
| CBR | 0.083 |
| CBP | 0.225 |

### Bottom-2 Capture Rate (B2CR) - Seasons with Judge Save

| Method | B2CR | Δ vs CBR |
|--------|------|----------|
| AMEI 2.0 | 0.534 | -23.5% |
| CBR | 0.699 | 0.0% |
| CBP | 0.699 | 0.0% |

### Elimination Match Rate (JS-EMR) - Judge Save Rule

| Method | JS-EMR | Δ vs CBR |
|--------|--------|----------|
| AMEI 2.0 | 0.375 | -4.5% |
| CBR | 0.393 | 0.0% |
| CBP | 0.393 | 0.0% |

- Judge Save rule uses J_total + reverse rank (Rule B).
- Rule comparison summary: judge_save_rule_comparison.csv

### Controversy Interception Rate (Judge Save)

| Method | Interception Rate |
|--------|-------------------|
| AMEI 2.0 | 0.250 |
| CBR | 0.000 |
| CBP | 0.000 |

### Weight Optimization Justification

- Selected weights: w_start=0.80, w_end=0.50, k=4.0, t0=0.60, min_vote_weight=0.03
- tradeoff_score=0.171, fairness_loss=48.249, excitement_loss=0.044
- interception_mean=0.571 (feasible=False)
- EMR constraint (CBR/CBP): >= 0.454 / 0.458
- B2CR constraint (CBR/CBP): >= 0.649 / 0.649
- No grid point satisfies all constraints; soft-penalty selection keeps λ=1 as default
- Stronger penalties (λ>=2) shift to w_start=0.75, w_end=0.20, k=16.0, t0=0.40 (tradeoff_score=0.421) but remain infeasible
- Recommendation: keep λ=1 for interpretability and stability; higher penalties do not recover feasibility and worsen trade-off loss
- Grid bounds: w_start 0.55-0.80, w_end 0.20-0.50, k 4-16, t0 0.40-0.60, min_vote_weight 0.03-0.10; reduced grid balances runtime and coverage

### Adaptive Weights by Season Type (JS vs No JS)

- Per-type grid search converged to the same parameters: w_start=0.80, w_end=0.50, k=4.0, t0=0.60, min_vote_weight=0.03
- No-JS metrics: EMR 0.347, Interception 0.611
- JS metrics: B2CR 0.534, Interception 0.333
- Combined (adaptive_weight_metrics.csv): EMR 0.347, B2CR 0.534, Interception 0.571, Spearman 0.952
- Identical parameters across season types indicate stable tuning; a single global setting is sufficient

### Judge Save Rule Robustness (A vs B vs C)

- AMEI: JS-EMR A=0.375, B=0.375, C=0.339
- CBR: JS-EMR A=0.393, B=0.393, C=0.357
- CBP: JS-EMR A=0.393, B=0.393, C=0.375
- Rule C uses J_total + 0.5 * previous reverse rank.

### Rule C Coefficient Sensitivity

- JS-EMR peaks at coef 0.3-0.4 (AMEI 0.393; CBR/CBP 0.411)
- JS-EMR declines with larger coefficients; at coef 0.8, AMEI 0.321 (CBR 0.339, CBP 0.357)
- Recommended coef: 0.3-0.4 to emphasize recent performance without amplifying noise

### Controversy Threshold Robustness

- Quantile grid (z_q 0.10-0.30, p_q 0.70-0.90):
  AMEI mean=0.534 (range 0.261-0.759),
  CBR mean=0.119, CBP mean=0.187
- Absolute threshold (z<=-1.0, p>=0.15): AMEI=0.846, CBR=0.462, CBP=0.308
- Selected threshold (max advantage): z_q=0.10, p_q=0.75, advantage=0.450

### Alternative Controversy Definition (Fan-Judge Rank Gap)

- Fan-judge rank gap (q=0.8): AMEI 0.383 vs CBR 0.244 vs CBP 0.277

### Controversy Sparsity (All Seasons)

| Item | Value |
|---|---:|
| Seasons with zero controversies | 17/34 |
| Total controversies | 21 |
| Avg per season | 0.62 |

- Sparse events explain higher variance in Interception and motivate threshold sensitivity checks

### Cross-Season Generalization (LOSO)

- Mean advantage vs max baseline: EMR -0.219, B2CR -0.202, Interception +0.353, Spearman +0.032
- LOSO summary saved to loso_summary.csv
- Largest negative deltas occur in seasons 19/2/1 (EMR) and 32/34/28 (B2CR), highlighting season-specific volatility
- These cases include shorter seasons (S1=6, S2=8 weeks) and very sparse controversies (often 0-2), so per-season deltas are noisier

### Expected Corrections per Season

- AMEI: 0.353 controversies corrected/season
- CBR: 0.059 controversies corrected/season
- CBP: 0.118 controversies corrected/season

## Uncertainty Analysis

- AMEI entropy: 1.972 ± 0.289
- CBR entropy: 0.300 ± 0.214
- CBP entropy: 2.803 ± 0.233
- Higher entropy indicates more balanced competition

## Fairness & Excitement Loss

- Fairness loss (AMEI/CBR/CBP): 48.249 / 79.569 / 79.569
- Spearman fairness (AMEI/CBR/CBP): 0.952 / 0.904 / 0.825
- Excitement loss (AMEI/CBR/CBP): 0.044 / 1.569 / 0.021

## Bobby Bones Case Study (Season 27)

### Finale Analysis

| Contestant | J_total | p_hat | z_score | kappa | AMEI Score | Rank |
|------------|---------|-------|---------|-------|------------|------|
| Milo Manheim | 30.0 | 0.468 | 0.67 | 1.000 | 0.4932 | 1 |
| Evanna Lynch | 30.0 | 0.249 | 0.67 | 1.000 | 0.3433 | 2 |
| Alexis Ren | 28.5 | 0.150 | -0.67 | 1.000 | 0.1472 | 3 |
| Bobby Bones | 27.0 | 0.133 | -2.02 | 0.050 | 0.0162 | 4 |

### Bobby Bones Season Summary

- Average z-score: -1.23 (negative indicates below-median judge scores)
- Average kappa (vote weight): 0.703
- Times in AMEI bottom-2: 4/9 weeks
- Under AMEI 2.0, low technical merit would have been penalized via reduced vote weight

## Key Insights

1. **Merit-Weighted Voting**: AMEI 2.0 applies kappa gating to reduce vote weight for contestants with poor technical scores
2. **Dynamic Balance**: Judge weight decreases from 80% to 50% as season progresses, giving audience more influence in later rounds
3. **Soft Penalty**: Rather than hard cutoffs, the hinge gate provides gradual penalty for below-threshold performance
4. **Constraint-Aware Tuning**: Hard constraints are infeasible, so soft penalties select stable weights while preserving interception gains

## Interpretation Notes

- AMEI is designed to alter elimination logic rather than replicate historical outcomes, so a lower EMR vs CBR is expected and not a defect.
- The key target is higher controversy interception and improved fairness-excitement balance.
- Controversies are sparse (17/34 seasons with zero events), which inflates Interception variance; robustness checks address this.

## Figure Takeaways

- Pareto plots: AMEI points cluster toward lower fairness loss with modest excitement loss, supporting the intended trade-off
- Sensitivity heatmaps: AMEI interception remains high across broad threshold ranges; baselines stay low
- Radar comparison: AMEI dominates on interception and Spearman while ceding EMR/B2CR, matching the design goal

## Additional Outputs

- README.md: quick index for outputs
- sensitivity_grid.csv: grid search results for z_soft_q/z_hard_q
- sensitivity_heatmap_nojs.png: EMR heatmap for seasons without Judge Save
- sensitivity_heatmap_js.png: B2CR heatmap for seasons with Judge Save
- weight_grid.csv: logistic weight grid search
- weight_selected.csv: selected weight parameters
- weight_grid_js.csv / weight_grid_nojs.csv: season-type weight grids
- weight_selected_js.csv / weight_selected_nojs.csv: season-type selected weights
- weight_selected_soft.csv / weight_selected_soft_js.csv / weight_selected_soft_nojs.csv: soft-constraint selections
- adaptive_weight_metrics.csv: adaptive-weight aggregate metrics
- controversy_sensitivity.csv: controversy threshold sensitivity
- controversy_heatmap.png: interception sensitivity heatmap
- controversy_alt_thresholds.csv: absolute-threshold interception
- controversy_alt_definitions.csv: fan-judge rank gap definition
- threshold_selected.csv: data-driven threshold selection
- pareto_nojs.png / pareto_js.png: fairness-excitement trade-off plots
- bobby_trajectory.png: Bobby Bones z-score/kappa/rank trend
- mcnemar_summary.md: McNemar significance tests
- fairness_spearman.csv: fairness rank correlation summary
- radar_comparison.png: radar chart across methods
- weight_transition.png: logistic weight transition curve
- summary.md: concise numeric summary
- whatif_bobby_bones.csv / whatif_summary.md: Season 27 what-if analysis
- bootstrap_ci.csv: bootstrap confidence intervals
- bootstrap_diff_ci.csv: bootstrap difference CIs
- bootstrap_diff_summary.md: bootstrap difference summary
- season_stability.csv / season_stability_summary.md: season stability metrics
- season_stage_stability.csv: stage stability metrics
- judge_save_rule_comparison.csv: Rule A vs Rule B (JS-EMR)
- judge_save_coeff_grid.csv: Rule C coefficient sensitivity grid
- loso_summary.csv: leave-one-season-out results
- interception_impact.csv: expected corrections per season
- stage_analysis.csv / stage_comparison.png: stage-wise performance

## Statistical Significance (McNemar)

- No Judge Save AMEI vs CBR: n01=51, n10=10, p=0.0000, p_adj=0.0000
- No Judge Save AMEI vs CBP: n01=43, n10=1, p=0.0000, p_adj=0.0000
- Judge Save (Bottom-2) AMEI vs CBR: n01=13, n10=0, p=0.0009, p_adj=0.0035
- Judge Save (Bottom-2) AMEI vs CBP: n01=13, n10=0, p=0.0009, p_adj=0.0035
- Judge Save (Elimination) AMEI vs CBR: n01=1, n10=0, p=1.0000, p_adj=1.0000
- Judge Save (Elimination) AMEI vs CBP: n01=1, n10=0, p=1.0000, p_adj=1.0000

## Bootstrap Confidence Intervals (95%)

| Metric | Method | Group | Mean | CI Low | CI High |
|--------|--------|-------|------|--------|---------|
| EMR | AMEI | No Judge Save | 0.347 | 0.294 | 0.405 |
| B2CR | AMEI | Judge Save | 0.521 | 0.411 | 0.630 |
| Interception | AMEI | All Weeks | 0.571 | 0.333 | 0.762 |
| EMR | CBR | No Judge Save | 0.504 | 0.443 | 0.561 |
| B2CR | CBR | Judge Save | 0.699 | 0.589 | 0.795 |
| Interception | CBR | All Weeks | 0.095 | 0.000 | 0.238 |
| EMR | CBP | No Judge Save | 0.508 | 0.443 | 0.569 |
| B2CR | CBP | Judge Save | 0.699 | 0.589 | 0.795 |
| Interception | CBP | All Weeks | 0.190 | 0.048 | 0.381 |

## Bootstrap Difference CIs (95%)

| Metric | Group | Compare | Mean Diff | CI Low | CI High |
|--------|-------|---------|-----------|--------|---------|
| EMR | No Judge Save | AMEI-CBR | -0.156 | -0.210 | -0.099 |
| EMR | No Judge Save | AMEI-CBP | -0.160 | -0.206 | -0.118 |
| B2CR | Judge Save | AMEI-CBR | -0.178 | -0.274 | -0.096 |
| B2CR | Judge Save | AMEI-CBP | -0.178 | -0.260 | -0.096 |
| JS_EMR | Judge Save | AMEI-CBR | -0.018 | -0.054 | 0.000 |
| JS_EMR | Judge Save | AMEI-CBP | -0.018 | -0.054 | 0.000 |
| Interception | All Weeks | AMEI-CBR | 0.476 | 0.286 | 0.668 |
| Interception | All Weeks | AMEI-CBP | 0.381 | 0.190 | 0.571 |

Interpretation: CIs that do not cross 0 indicate consistent differences; negative EMR/B2CR deltas reflect the intended trade-off toward interception and balance.

## Season Stability Summary

- EMR (No Judge Save, AMEI): std=0.145, IQR=0.127
- B2CR (Judge Save, AMEI): std=0.176, IQR=0.260
- JS-EMR (Judge Save, AMEI): std=0.174, IQR=0.256
- Interception (All Seasons, AMEI): std=0.456, IQR=1.000
- Spearman (All Seasons, AMEI): std=0.023, IQR=0.024

## Stage Stability (Means)

| Stage | Metric | AMEI | CBR | CBP |
|-------|--------|------|-----|-----|
| Early | EMR | 0.337 | 0.458 | 0.494 |
| Early | B2CR | 0.560 | 0.760 | 0.640 |
| Early | JS_EMR | 0.421 | 0.421 | 0.421 |
| Early | Interception | 0.500 | 0.000 | 0.250 |
| Early | Spearman | 0.980 | 0.937 | 0.906 |

| Mid | EMR | 0.390 | 0.558 | 0.532 |
| Mid | B2CR | 0.476 | 0.762 | 0.857 |
| Mid | JS_EMR | 0.263 | 0.263 | 0.263 |
| Mid | Interception | 0.400 | 0.000 | 0.000 |
| Mid | Spearman | 0.953 | 0.894 | 0.814 |

| Late | EMR | 0.324 | 0.500 | 0.500 |
| Late | B2CR | 0.556 | 0.593 | 0.630 |
| Late | JS_EMR | 0.444 | 0.500 | 0.500 |
| Late | Interception | 0.667 | 0.167 | 0.250 |
| Late | Spearman | 0.929 | 0.884 | 0.767 |
