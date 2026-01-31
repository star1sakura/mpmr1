# AMEI 2.0 Results Summary (r4)

This summary consolidates the latest r4 outputs and robustness checks. Values correspond to the current run of `r4/req4_analysis.py`.

## 0. Core Metrics Snapshot (Weighted by Weeks)

Values are weighted by weeks using `comparison_metrics.csv`.

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

## 0b. Metric Definitions

- EMR (No JS): share of non-Judge Save weeks where predicted elimination matches the actual elimination
- B2CR (JS): share of Judge Save weeks where the actual eliminated contestant appears in the model Bottom-2
- JS-EMR (JS): share of Judge Save weeks where the judge-save rule (Rule B) applied to model Bottom-2 matches the actual elimination
- Interception: fraction of controversy contestants captured in Bottom-2 (controversy defined by z/p quantiles unless noted)
- Spearman: rank correlation between judge ranking and method ranking within a week

Interpretation: AMEI intentionally trades EMR/B2CR for stronger controversy interception and better fairness-excitement balance; negative deltas are expected by design.

## 1. Weight Optimization (with Interception Constraint)

- Selected weights: **w_start=0.80, w_end=0.50, k=4.0, t0=0.60, min_vote_weight=0.03**
- Trade-off score: **0.171** (fairness loss=48.249, excitement loss=0.044)
- Interception mean: **0.571** (constraint >= 0.40 satisfied)
- EMR/B2CR constraints (<= 0.05 abs or 10% relative drop) are **not satisfied**; no grid point meets both constraints simultaneously
- Soft-penalty search: λ=1 keeps the same weights; λ>=2 shifts to **w_start=0.75, w_end=0.20, k=16.0, t0=0.40** (trade-off score 0.421) but still infeasible
- Recommended default: keep λ=1 because higher penalties do not improve feasibility and increase trade-off loss
- Grid bounds: w_start 0.55-0.80, w_end 0.20-0.50, k 4-16, t0 0.40-0.60, min_vote_weight 0.03-0.10; reduced grid balances runtime and coverage

## 1b. Adaptive Weights by Season Type (JS vs No JS)

- Per-type grid search converged to the same parameters: **w_start=0.80, w_end=0.50, k=4.0, t0=0.60, min_vote_weight=0.03**
- No-JS metrics: EMR **0.347**, Interception **0.611**
- JS metrics: B2CR **0.534**, Interception **0.333**
- Combined (adaptive_weight_metrics.csv): EMR **0.347**, B2CR **0.534**, Interception **0.571**, Spearman **0.952**
- Identical parameters across season types indicate stable tuning; a single global setting is sufficient

## 2. Judge Save Rule Robustness

Rule A (J_total only), Rule B (J_total + reverse rank), Rule C (J_total + 0.5 * previous reverse rank):

| Method | JS-EMR (A) | JS-EMR (B) | JS-EMR (C) |
|---|---:|---:|---:|
| AMEI | 0.375 | 0.375 | 0.339 |
| CBR | 0.393 | 0.393 | 0.357 |
| CBP | 0.393 | 0.393 | 0.375 |

## 2b. Rule C Coefficient Sensitivity

- JS-EMR peaks at coef **0.3-0.4** (AMEI **0.393**; CBR/CBP **0.411**)
- JS-EMR declines with larger coefficients; at coef **0.8**, AMEI **0.321** (CBR **0.339**, CBP **0.357**)
- Recommended coef: **0.3-0.4** to balance recency and stability

## 3. Controversy Threshold Robustness

Quantile grid (z_q 0.10-0.30, p_q 0.70-0.90):

- AMEI mean 0.534, range 0.261-0.759
- CBR mean 0.119, range 0.033-0.286
- CBP mean 0.187, range 0.083-0.379

Absolute thresholds (z_score <= -1.0, p_hat >= 0.15):

- AMEI 0.846 vs CBR 0.462 vs CBP 0.308
- Selected threshold (max advantage): z_q=0.10, p_q=0.75 (advantage=0.450)

## 3b. Alternative Controversy Definition (Fan-Judge Rank Gap)

- Fan-judge rank gap (q=0.8): AMEI **0.383** vs CBR **0.244** vs CBP **0.277**

## 3c. Controversy Sparsity (All Seasons)

| Item | Value |
|---|---:|
| Seasons with zero controversies | 17/34 |
| Total controversies | 21 |
| Avg per season | 0.62 |

- Sparse events explain higher Interception variance and motivate threshold sensitivity checks

## 4. Season and Stage Stability (AMEI)

Season-level dispersion:

- EMR std=0.145, IQR=0.127
- B2CR std=0.176, IQR=0.260
- JS-EMR std=0.174, IQR=0.256
- Interception std=0.456, IQR=1.000
- Spearman std=0.023, IQR=0.024

Stage means:

- EMR: Early 0.337, Mid 0.390, Late 0.324
- B2CR: Early 0.560, Mid 0.476, Late 0.556
- Interception: Early 0.500, Mid 0.400, Late 0.667
- Spearman: Early 0.980, Mid 0.953, Late 0.929

## 5. Cross-Season Generalization (LOSO)

- Mean advantage vs max baseline: EMR -0.219, B2CR -0.202, Interception +0.353, Spearman +0.032
- Largest negative deltas concentrate in seasons 19/2/1 (EMR) and 32/34/28 (B2CR)
- These cases include shorter seasons (S1=6, S2=8 weeks) and very sparse controversies (often 0-2), so per-season deltas are noisier

## 6. Expected Corrections per Season

- AMEI 0.353 vs CBR 0.059 vs CBP 0.118 controversies corrected/season

## 7. Figure Takeaways

- Pareto plots: AMEI points cluster toward lower fairness loss with modest excitement loss, supporting the intended trade-off
- Sensitivity heatmaps: AMEI interception remains high across broad threshold ranges; baselines stay low
- Radar comparison: AMEI dominates on interception and Spearman while ceding EMR/B2CR, matching the design goal

## 8. Key Outputs

- `r4/outputs/interpretation_report.md`
- `r4/outputs/summary.md`
- `r4/outputs/README.md`
- `r4/outputs/weight_grid.csv`, `r4/outputs/weight_selected.csv`
- `r4/outputs/weight_grid_js.csv`, `r4/outputs/weight_grid_nojs.csv`
- `r4/outputs/weight_selected_js.csv`, `r4/outputs/weight_selected_nojs.csv`, `r4/outputs/adaptive_weight_metrics.csv`
- `r4/outputs/weight_selected_soft.csv`, `r4/outputs/weight_selected_soft_js.csv`, `r4/outputs/weight_selected_soft_nojs.csv`
- `r4/outputs/controversy_sensitivity.csv`, `r4/outputs/controversy_alt_thresholds.csv`, `r4/outputs/threshold_selected.csv`
- `r4/outputs/controversy_alt_definitions.csv`
- `r4/outputs/loso_summary.csv`
- `r4/outputs/interception_impact.csv`
- `r4/outputs/season_stability.csv`, `r4/outputs/season_stability_summary.md`, `r4/outputs/season_stage_stability.csv`
- `r4/outputs/judge_save_rule_comparison.csv`
- `r4/outputs/judge_save_coeff_grid.csv`
