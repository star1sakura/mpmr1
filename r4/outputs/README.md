# r4 Outputs Guide

If you only read three files:
- `r4/outputs/interpretation_report.md` (full narrative + diagnostics)
- `r4/outputs/summary.md` (concise numeric summary)
- `r4/outputs/weight_selected.csv` (selected weights + constraint status)

Reviewer quick path:
- `r4/outputs/summary.md` (Section 0 core metrics table)
- `r4/outputs/interpretation_report.md` (Weight Optimization + LOSO + Sparsity)
- `r4/outputs/mcnemar_summary.md` and `r4/outputs/bootstrap_diff_summary.md`

Core tables:
- `r4/outputs/comparison_metrics.csv` (season-level metrics)
- `r4/outputs/loso_summary.csv` (LOSO generalization)
- `r4/outputs/interception_impact.csv` (expected corrections/season)
- `r4/outputs/season_stability_summary.md` and `r4/outputs/season_stage_stability.csv` (stability)

Robustness checks:
- `r4/outputs/weight_grid.csv` (weight grid search)
- `r4/outputs/weight_selected_soft.csv` (soft-penalty choices)
- `r4/outputs/controversy_sensitivity.csv` and `r4/outputs/threshold_selected.csv`
- `r4/outputs/controversy_alt_thresholds.csv` and `r4/outputs/controversy_alt_definitions.csv`
- `r4/outputs/judge_save_rule_comparison.csv` and `r4/outputs/judge_save_coeff_grid.csv`

Figures:
- `r4/outputs/pareto_nojs.png`, `r4/outputs/pareto_js.png`
- `r4/outputs/sensitivity_heatmap_nojs.png`, `r4/outputs/sensitivity_heatmap_js.png`
- `r4/outputs/controversy_heatmap.png`, `r4/outputs/radar_comparison.png`
- `r4/outputs/weight_transition.png`, `r4/outputs/bobby_trajectory.png`
