# AMEI 2.0 Simulation Results Summary

## Overall Performance Comparison

### Elimination Match Rate (EMR) - Seasons without Judge Save

| Method | EMR | Improvement vs CBR |
|--------|-----|--------------------|
| AMEI 2.0 | 0.347 | baseline |
| CBR | 0.534 | -18.7% |
| CBP | 0.530 | -18.3% |

### Controversy Interception Rate (No Judge Save)

| Method | Interception Rate |
|--------|-------------------|
| AMEI 2.0 | 0.522 |
| CBR | 0.156 |
| CBP | 0.222 |

### Bottom-2 Capture Rate (B2CR) - Seasons with Judge Save

| Method | B2CR | Improvement vs CBR |
|--------|------|--------------------|
| AMEI 2.0 | 0.548 | baseline |
| CBR | 0.686 | -13.8% |
| CBP | 0.695 | -14.7% |

### Controversy Interception Rate (Judge Save)

| Method | Interception Rate |
|--------|-------------------|
| AMEI 2.0 | 0.750 |
| CBR | 0.000 |
| CBP | 0.000 |

## Uncertainty Analysis

- AMEI entropy: 2.053 ± 0.263
- CBR entropy: 0.300
- CBP entropy: 2.803
- Higher entropy indicates more balanced competition

## Fairness & Excitement Loss

- Fairness loss (AMEI/CBR/CBP): 52.25 / 78.25 / 78.25
- Excitement loss (AMEI/CBR/CBP): 0.036 / 1.555 / 0.022

## Bobby Bones Case Study (Season 27)

### Finale Analysis

| Contestant | J_total | p_hat | z_score | kappa | AMEI Score | Rank |
|------------|---------|-------|---------|-------|------------|------|
| Milo Manheim | 30.0 | 0.468 | 0.67 | 1.000 | 0.4987 | 1 |
| Evanna Lynch | 30.0 | 0.249 | 0.67 | 1.000 | 0.3359 | 2 |
| Alexis Ren | 28.5 | 0.150 | -0.67 | 1.000 | 0.1503 | 3 |
| Bobby Bones | 27.0 | 0.133 | -2.02 | 0.050 | 0.0151 | 4 |

### Bobby Bones Season Summary

- Average z-score: -1.23 (negative indicates below-median judge scores)
- Average kappa (vote weight): 0.613
- Times in AMEI bottom-2: 5/9 weeks
- Under AMEI 2.0, low technical merit would have been penalized via reduced vote weight

## Key Insights

1. **Merit-Weighted Voting**: AMEI 2.0 applies kappa gating to reduce vote weight for contestants with poor technical scores
2. **Dynamic Balance**: Judge weight decreases from 65% to 35% as season progresses, giving audience more influence in later rounds
3. **Soft Penalty**: Rather than hard cutoffs, the hinge gate provides gradual penalty for below-threshold performance

## Interpretation Notes

- AMEI is designed to alter elimination logic rather than replicate historical outcomes, so a lower EMR vs CBR is expected and not a defect.
- The key target is higher controversy interception and improved fairness–excitement balance.

## Additional Outputs

- sensitivity_grid.csv: grid search results for z_soft/z_hard
- sensitivity_heatmap_nojs.png: EMR heatmap for seasons without Judge Save
- sensitivity_heatmap_js.png: B2CR heatmap for seasons with Judge Save
- pareto_nojs.png / pareto_js.png: fairness–excitement trade-off plots
- bobby_trajectory.png: Bobby Bones z-score/kappa/rank trend
- mcnemar_summary.md: McNemar significance tests
- radar_comparison.png: radar chart across methods
- weight_transition.png: logistic weight transition curve
- whatif_bobby_bones.csv / whatif_summary.md: Season 27 what-if analysis
- bootstrap_ci.csv: bootstrap confidence intervals
- stage_analysis.csv / stage_comparison.png: stage-wise performance

## Statistical Significance (McNemar)

- No Judge Save AMEI vs CBR: n01=60, n10=11, p=0.0000
- No Judge Save AMEI vs CBP: n01=49, n10=1, p=0.0000

- Judge Save AMEI vs CBR: n01=10, n10=0, p=0.0044
- Judge Save AMEI vs CBP: n01=11, n10=0, p=0.0026

## Bootstrap Confidence Intervals (95%)

| Metric | Method | Group | Mean | CI Low | CI High |
|--------|--------|-------|------|--------|---------|
| EMR | AMEI | No Judge Save | 0.347 | 0.294 | 0.405 |
| B2CR | AMEI | Judge Save | 0.548 | 0.438 | 0.658 |
| Interception | AMEI | All Weeks | 0.524 | 0.286 | 0.762 |
| EMR | CBR | No Judge Save | 0.534 | 0.473 | 0.588 |
| B2CR | CBR | Judge Save | 0.685 | 0.575 | 0.795 |
| Interception | CBR | All Weeks | 0.143 | 0.000 | 0.333 |
| EMR | CBP | No Judge Save | 0.531 | 0.473 | 0.592 |
| B2CR | CBP | Judge Save | 0.699 | 0.589 | 0.795 |
| Interception | CBP | All Weeks | 0.190 | 0.048 | 0.381 |
