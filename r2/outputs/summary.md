# 要求二结果摘要（基于 r1 新模型）

## 数据与方法
- 输入数据：`r1/outputs/req1_vote_estimates.csv`
- 方法：比较 CBR（排名合并）与 CBP（百分比合并），并使用 contestant_id 作为稳定性 tie-break。

## 总体结果（按淘汰周加权平均）
- 无 judge-save 赛季：CBR_EMR = 0.615，CBP_EMR = 0.615
- judge-save 赛季：CBR_B2CR = 0.875，CBP_B2CR = 0.893
- Flip rate（全体）= 0.322；无 judge-save = 0.341；judge-save = 0.250

## 总体解读（学术表述）
总体而言，CBR 与 CBP 在无 judge-save 赛季的 EMR 接近，说明两种合并机制在常规淘汰周具有相似的一致性上限。然而在 judge-save 赛季，CBP 的 B2CR 略高，表明当规则关注 Bottom-2 覆盖时，基于占比的合并对边缘选手更敏感。Flip rate 处于中等水平，意味着两种规则在相当比例的周次给出不同的淘汰（或 Bottom-2）集合，制度差异具有可观的结构性影响。
补充：CBP 的 B2CR 比 CBR 高约 0.018。

## 翻转来源分解
- judge: 5 / 85（0.059）
- fan: 8 / 85（0.094）
- mixed: 72 / 85（0.847）

## 翻转强度分析
- flip_strength mean = 2.659，median = 3.000（按排名距离）
- no judge-save mean = 3.085，judge-save mean = 0.500

## 稳健性与不确定性
- 分箱采用 uncertainty_width 分位数（low/mid/high）。
- low: flip_rate = 0.223，flip_strength_mean = 1.600
- mid: flip_rate = 0.378，flip_strength_mean = 3.190
- high: flip_rate = 0.161，flip_strength_mean = 2.889
- 高不确定性组 judge-save 周次较少，B2CR 可能缺失或不稳定。

## 分层分析（规模 / 赛季阶段）
- size=small: EMR(CBR=0.656, CBP=0.656), B2CR(CBR=0.846, CBP=0.923), flip_rate=0.320
- size=mid: EMR(CBR=0.521, CBP=0.577), B2CR(CBR=0.789, CBP=1.000), flip_rate=0.378
- size=large: EMR(CBR=0.681, CBP=0.596), B2CR(CBR=0.958, CBP=0.792), flip_rate=0.254
- stage=early: EMR(CBR=0.625, CBP=0.613), B2CR(CBR=0.952, CBP=0.810), flip_rate=0.277
- stage=mid: EMR(CBR=0.559, CBP=0.574), B2CR(CBR=0.737, CBP=0.947), flip_rate=0.448
- stage=late: EMR(CBR=0.667, CBP=0.667), B2CR(CBR=0.938, CBP=0.938), flip_rate=0.237

## tie 阈值敏感性
- threshold=0.03: CBR=10, CBP=13, tie=11 (total=34)
- threshold=0.05: CBR=10, CBP=13, tie=11 (total=34)
- threshold=0.07: CBR=10, CBP=13, tie=11 (total=34)
- 0.03/0.05/0.07 的结果一致，结论对阈值不敏感。

## 典型赛季（5 个）
- S1（no judge-save）：flip_rate = 0.667，EMR_gap = 0.667；CBP 在该赛季 EMR 更高。标签：高翻转/高差异。制度差异显著，规则选择对结果影响大。
- S8（no judge-save）：flip_rate = 0.667，EMR_gap = 0.111；CBR 在该赛季 EMR 更高。标签：高翻转。预测集合分歧较频繁。
- S21（no judge-save）：flip_rate = 0.429，EMR_gap = 0.429；CBP 在该赛季 EMR 更高。标签：高差异。一致性指标差距明显。
- S29（judge-save）：flip_rate = 0.000，B2CR_gap = 0.000；两种方法在该赛季 B2CR 表现相同。标签：低翻转/趋同。两种规则给出接近的淘汰结论。
- S30（judge-save）：flip_rate = 0.375，B2CR_gap = 0.250；CBR 在该赛季 B2CR 更高。标签：中等差异。规则差异存在但不极端。

## 争议选手画像（平均 fan_judge_gap）
- S2 Jerry Rice：mean fan_judge_gap = -0.062（8 周）；粉丝偏好显著低于评审。
- S4 Billy Ray Cyrus：mean fan_judge_gap = -0.058（8 周）；粉丝偏好显著低于评审。
- S11 Bristol Palin：mean fan_judge_gap = -0.095（10 周）；粉丝偏好显著低于评审。
- S27 Bobby Bones：mean fan_judge_gap = -0.033（9 周）；粉丝偏好显著低于评审。

## 输出索引
- 典型赛季与统计表：`/home/chaossora/Projects/USmath/mpmr1/r2/outputs/summary_table.csv`
- 争议选手差异曲线：`/home/chaossora/Projects/USmath/mpmr1/r2/outputs/controversy_gap_trends.png`
- 翻转强度分布：`/home/chaossora/Projects/USmath/mpmr1/r2/outputs/flip_strength.png`
- 赛季建议表（tie 阈值=0.05）：`/home/chaossora/Projects/USmath/mpmr1/r2/outputs/decision_guidance.csv`
- 不确定性-翻转分析：`/home/chaossora/Projects/USmath/mpmr1/r2/outputs/flip_uncertainty_summary.csv`
- 不确定性图（rate/strength）：`/home/chaossora/Projects/USmath/mpmr1/r2/outputs/flip_uncertainty_rate.png` / `/home/chaossora/Projects/USmath/mpmr1/r2/outputs/flip_uncertainty_strength.png`
- 高低不确定性对比：`/home/chaossora/Projects/USmath/mpmr1/r2/outputs/uncertainty_focus.csv` / `/home/chaossora/Projects/USmath/mpmr1/r2/outputs/uncertainty_focus.png`
- tie 阈值敏感性：`/home/chaossora/Projects/USmath/mpmr1/r2/outputs/tie_sensitivity.csv` / `/home/chaossora/Projects/USmath/mpmr1/r2/outputs/tie_sensitivity.png`
- 分层指标表：`/home/chaossora/Projects/USmath/mpmr1/r2/outputs/strata_metrics.csv`
- 分层图（规模/阶段）：`/home/chaossora/Projects/USmath/mpmr1/r2/outputs/strata_by_size.png` / `/home/chaossora/Projects/USmath/mpmr1/r2/outputs/strata_by_stage.png`
- 汇总对比图：`/home/chaossora/Projects/USmath/mpmr1/r2/outputs/summary_metrics.png` / `/home/chaossora/Projects/USmath/mpmr1/r2/outputs/summary_flip_strength.png`

## 备注
- r2 输出基于 r1 最新模型估计的 p_hat，结论与旧模型输出不再可比。
- judge-save 赛季使用 Bottom-2 覆盖率（B2CR）评价方法表现。
- flip_strength 在 judge-save 赛季以最差者为代表，用于比较规则差异强度。