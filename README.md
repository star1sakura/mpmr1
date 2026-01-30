# 项目说明（面向使用者）

本项目包含 2026 MCM Problem C 的 **r1（要求一）投票估计** 与 **r2（要求二）方法对比**，输出后续题目（2–5）可直接使用的特征表、对比表与图表。

适用对象：负责后续题目撰写与分析的项目参与者（直接使用本项目输出数据与图表即可）。

---

## 目录结构

- `2026_MCM_Problem_C_Data.csv`：官方原始数据
- `r1/req1_solve.py`：核心模型（生成投票估计、指标、特征表）
- `r1/req1_summary.py`：汇总图表与表格（赛季指标、争议选手画像、不确定性可视化）
- `r1/tune_params.py`：参数调优脚本（网格搜索，输出推荐参数）
- `r1/outputs/`：r1 运行输出目录（见下方“输出文件详解”）
- `r2/req2_analysis.py`：要求二方法对比分析脚本
- `r2/要求二实现方案.md`：r2 详细说明文档
- `r2/outputs/`：r2 运行输出目录

---

## 环境准备

### Python 版本

建议 Python 3.10+（项目使用 `numpy/pandas/scipy/matplotlib/pulp`）。

### 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 安装依赖

```bash
pip install "pulp<3" numpy pandas scipy matplotlib
```

如需代理（示例）：

```bash
export all_proxy="http://127.0.0.1:7897"
pip install "pulp<3" numpy pandas scipy matplotlib
```

> 说明：`pulp` 是 ILP 求解器接口，用于排名赛季的严格约束求解。没有 `pulp` 时会降级为启发式解法，建议安装。

---

## 快速开始（推荐参数）

推荐使用参数搜索得到的最优组合（可在 `r1/outputs/tuning/top_params.csv` 查看第一行）。

```bash
python r1/req1_solve.py \
  --alpha-pop 0.4 \
  --beta 1.2 \
  --lambda-smooth 0.7 \
  --rho-pop 0.3 \
  --kappa-rank 0.5
```

运行后会生成：

- `r1/outputs/req1_vote_estimates.csv`
- `r1/outputs/req1_features.csv`
- `r1/outputs/req1_metrics.json`

并可再运行汇总图表：

```bash
python r1/req1_summary.py
```

---

## 详细使用

### 1) 基础运行（默认参数）

```bash
python r1/req1_solve.py
```

适合快速生成基础结果。默认参数可用，但推荐使用调优参数。

### 2) 生成不确定性输出（对应题 1“确定性度量”）

```bash
python r1/req1_solve.py --uncertainty
```

会在输出中增加 `uncertainty_width` 字段，并生成不确定性图表（运行 `req1_summary.py` 后）。

### 3) 参数调优（稳健搜索，30–60 分钟）

```bash
python r1/tune_params.py
```

输出：

- `r1/outputs/tuning/stage_a_results.csv`（粗筛）
- `r1/outputs/tuning/stage_b_results.csv`（精筛）
- `r1/outputs/tuning/top_params.csv`（推荐参数）

调优完成后会自动用最优参数重跑模型，并更新 `r1/outputs/` 的最终结果。

### 4) 汇总图表

```bash
python r1/req1_summary.py
```

也可指定输入 CSV：

```bash
python r1/req1_summary.py --votes r1/outputs/req1_vote_estimates.csv
```

---

## 输出文件详解（重点）

### 1) `r1/outputs/req1_vote_estimates.csv`

用途：**最完整的周级估计数据**（后续题的主数据源）。

字段说明：

- `season`：赛季编号
- `week`：周次
- `contestant_id`：选手在原始 CSV 中的行号（可用来与原始表关联）
- `celebrity_name`：明星姓名
- `ballroom_partner`：专业舞者搭档
- `J_total`：该周评审总分（所有评审分之和）
- `p_hat`：估计的粉丝投票份额（0–1，所有在赛选手之和为 1）
- `V_hat`：等效票数 = `p_hat * total_votes`（仅尺度化，默认总票数为 1e7）
- `judge_share`：评审分数份额 `q`（该周评审总分占比）
- `fan_judge_gap`：`p_hat - judge_share`（粉丝偏好相对评审的差值；正值=粉丝更支持）
- `momentum`：`p_hat - prev_p`（相邻周投票份额变化，正值=热度上升）
- `popularity_score`：人气潜变量（对数尺度的平滑人气分数，建议看趋势而非绝对值）
- `eliminated`：是否在该周被淘汰（1=淘汰，0=未淘汰）
- `uncertainty_width`：估计不确定性宽度（仅在 `--uncertainty` 时生成）

使用建议：

- 题 2：用 `fan_judge_gap` 识别争议选手/周次
- 题 4：用 `p_hat` 作为粉丝偏好变量，和 `J_total` 比较建模
- 题 5：用 `momentum`/`fan_judge_gap` 构造新规则

### 2) `r1/outputs/req1_features.csv`

用途：**后续题目（2–5）直接使用的特征表**。字段与 `req1_vote_estimates.csv` 基本一致，便于直接建模/汇总。

建议以此为主表，不需要自行再拼接字段。

### 3) `r1/outputs/req1_metrics.json`

用途：整体一致性指标（验证模型是否复现淘汰结果）。

字段说明：

- `EMR`：淘汰命中率（无 judge-save 赛季）
- `B2CR`：Bottom-2 覆盖率（judge-save 赛季）
- `EMR_denom`：EMR 统计周数
- `B2CR_denom`：B2CR 统计周数

### 4) `r1/outputs/summary/season_metrics.csv`

用途：赛季级别的一致性指标（跨季稳定性分析）。

字段说明：

- `season`：赛季编号
- `rule`：该赛季使用的合并规则（`PERCENT`/`RANK`）
- `judge_save`：是否使用评审在 Bottom-2 中选择
- `EMR`/`B2CR`：赛季指标
- `EMR_denom`/`B2CR_denom`：赛季统计周数

### 5) `r1/outputs/summary/season_metrics.png`

用途：赛季指标可视化（可直接放论文）。

### 6) `r1/outputs/summary/weekly_uncertainty.csv`

用途：不确定性随周次变化的统计（仅在 `--uncertainty` 后生成）。

字段说明：

- `season`：赛季编号
- `week`：周次
- `uncertainty_width`：该赛季/周的平均不确定性

### 7) `r1/outputs/summary/weekly_uncertainty.png`

用途：周次平均不确定性曲线（跨季汇总）。

### 8) `r1/outputs/summary/weekly_uncertainty_heatmap.png`

用途：季 × 周不确定性热力图，直观看出哪类赛季/周更不确定。

### 9) `r1/outputs/summary/controversy_profiles.csv`

用途：争议选手的周度画像数据（题 2 可直接引用）。

字段说明：

- `season`、`week`、`celebrity_name`、`J_total`、`p_hat`、`eliminated`

### 10) `r1/outputs/summary/controversy_profiles.png`

用途：争议选手的“评审表现 vs 粉丝支持”曲线对比图。

### 11) `r1/outputs/tuning/stage_a_results.csv` / `stage_b_results.csv`

用途：参数搜索过程记录（可作为稳健性/敏感性分析材料）。

字段说明：

- `alpha_pop`：人气权重
- `beta`：评审表现权重
- `lambda_smooth`：惯性/平滑权重
- `rho_pop`：人气更新速率
- `kappa_rank`：排名转份额温度
- `EMR`/`B2CR`：一致性指标
- `std_EMR`/`std_B2CR`：跨季波动
- `score`：综合评分

### 12) `r1/outputs/tuning/top_params.csv`

用途：Top 参数组合（默认取第一行作为最终推荐）。

### 13) `r1/outputs/tune_tmp/`

用途：参数调优过程中的临时输出（可删除）。

---

## 要求二（r2）使用

前置条件：需要先生成 `r1/outputs/req1_vote_estimates.csv`（若包含 `uncertainty_width`，r2 会自动做稳健性与分层分析）。

运行方式：

```bash
python r2/req2_analysis.py
```

指定输入文件：

```bash
python r2/req2_analysis.py --votes r1/outputs/req1_vote_estimates.csv
```

关键输出：

- `r2/outputs/summary.md`：自动汇总结论（含不确定性与分层分析）
- `r2/outputs/decision_guidance.csv`：赛季规则建议表（tie 阈值默认 0.05）
- `r2/outputs/summary_table.csv`：结构化指标汇总表

完整说明与输出清单请见：`r2/要求二实现方案.md`

---

## 输出解读常见坑

- `p_hat`/`judge_share` 的归一化范围是**同赛季、同一周、仍在赛的选手**，不要跨周或跨季相加。
- `V_hat` 只是按 `--total-votes` 设定的尺度化票数，**只看相对大小**，不要解读为真实票数。
- `popularity_score` 是对数尺度的平滑人气分数，**建议看趋势**，不建议跨赛季作绝对比较。
- `momentum` 是相邻周变化，首周使用均匀先验，首周的 `momentum` 可作为“初始化效应”。
- `uncertainty_width` 只在 `--uncertainty` 运行时出现；排名赛季使用采样近似，不确定性仅作趋势参考。
- `eliminated` 对应淘汰周；对 `Withdrew` 的选手淘汰周由程序根据分数变化推断，可能与现实存在偏差。

---

## 参数说明（核心）

- `--alpha-pop`：人气权重（越大越强调“长期人气”）
- `--beta`：评审表现权重（越大越强调技术得分）
- `--lambda-smooth`：周与周惯性（越大越“平滑”）
- `--rho-pop`：人气更新速率（越大越依赖近期表现）
- `--kappa-rank`：排名转投票份额的温度参数（排名赛季）
- `--ilp-time-limit`：ILP 单周求解时间上限（秒）
- `--total-votes`：总票数标尺（只影响 `V_hat` 绝对值）

---

## 模型假设与局限

- 解释性优先：先验由“人气 + 评审表现 + 惯性”组成，追求可解释而非指标最大化。
- 票数尺度不可识别：`p_hat` 是可识别的相对份额，`V_hat` 仅是尺度化结果。
- 排名赛季用 ILP 保证 rank-sum 约束，但可行解可能不唯一，最终解受先验影响。
- judge-save 赛季仅约束“淘汰者在 Bottom-2”，评审最终选择未建模。
- 不确定性度量：百分比赛季用可行域宽度；排名赛季用采样近似；judge-save 的可行域不确定性为近似。
- `Withdrew` 选手的淘汰周由分数变化推断，可能与真实退赛时间存在偏差。

---

## 后续题目使用建议

- 题 2（方法对比）：用 `r1/outputs/req1_features.csv` 中 `p_hat`、`judge_share`、`fan_judge_gap` 对两种合并规则做模拟与对比。
- 题 3（建议）：用 `fan_judge_gap` 的分布和 EMR/B2CR 论证“公平 vs 兴奋度”。
- 题 4（因素分析）：以 `p_hat`/`J_total` 为因变量，合并原始数据中的年龄、行业、搭档等做回归或混合效应模型。
- 题 5（新系统）：用 `momentum` 与 `fan_judge_gap` 设计新规则（动态权重或保底机制）。

---

## 常见问题

- 运行慢：ILP 求解较慢，尝试缩短 `--ilp-time-limit` 或减少参数搜索范围。
- 没有 `uncertainty_width`：需加 `--uncertainty` 重新运行。
- `pulp` 未安装：排名赛季会退化为启发式解法，建议安装 `pulp`。
- `V_hat` 的绝对值偏大/偏小：用 `--total-votes` 设定总票标尺；分析主要看 `p_hat`。
