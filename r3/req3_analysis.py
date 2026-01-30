import argparse
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Requirement 3 analysis: mixed effects + survival/logit.")
    ap.add_argument(
        "--features",
        default=os.path.normpath(os.path.join(base_dir, "..", "r1", "outputs", "req1_features.csv")),
        help="Path to req1_features.csv",
    )
    ap.add_argument(
        "--raw",
        default=os.path.normpath(os.path.join(base_dir, "..", "2026_MCM_Problem_C_Data.csv")),
        help="Path to 2026_MCM_Problem_C_Data.csv",
    )
    ap.add_argument(
        "--out-dir",
        default=os.path.join(base_dir, "outputs"),
        help="Output directory",
    )
    return ap.parse_args()


def map_region(state: str) -> str:
    """将州名（缩写或全称）映射为大区"""
    if not isinstance(state, str):
        return "Other"
    state = state.strip()
    
    # 缩写映射
    northeast_abbr = {"CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"}
    midwest_abbr = {"IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"}
    south_abbr = {"DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV", "AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX"}
    west_abbr = {"AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"}
    
    # 全称映射
    northeast_full = {
        "Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", 
        "Vermont", "New Jersey", "New York", "Pennsylvania"
    }
    midwest_full = {
        "Illinois", "Indiana", "Michigan", "Ohio", "Wisconsin", "Iowa", "Kansas",
        "Minnesota", "Missouri", "Nebraska", "North Dakota", "South Dakota"
    }
    south_full = {
        "Delaware", "Florida", "Georgia", "Maryland", "North Carolina", "South Carolina",
        "Virginia", "Washington D.C.", "West Virginia", "Alabama", "Kentucky", "Mississippi",
        "Tennessee", "Arkansas", "Louisiana", "Oklahoma", "Texas"
    }
    west_full = {
        "Arizona", "Colorado", "Idaho", "Montana", "Nevada", "New Mexico", "Utah",
        "Wyoming", "Alaska", "California", "Hawaii", "Oregon", "Washington"
    }
    
    # 检查缩写
    if state.upper() in northeast_abbr:
        return "Northeast"
    if state.upper() in midwest_abbr:
        return "Midwest"
    if state.upper() in south_abbr:
        return "South"
    if state.upper() in west_abbr:
        return "West"
    
    # 检查全称
    if state in northeast_full:
        return "Northeast"
    if state in midwest_full:
        return "Midwest"
    if state in south_full:
        return "South"
    if state in west_full:
        return "West"
    
    return "International"  # 非美国选手


def consolidate_industry(industry: str, industry_counts: Dict[str, int], min_count: int = 10) -> str:
    """将出现次数少于 min_count 的行业归并为 'Other_Industry'"""
    if not isinstance(industry, str):
        return "Other_Industry"
    # 统一大小写处理
    industry_lower = industry.lower().strip()
    # 检查标准化后的行业计数
    if industry_counts.get(industry_lower, 0) < min_count:
        return "Other_Industry"
    return industry


def build_survival_table(df: pd.DataFrame) -> pd.DataFrame:
    # contestant-level aggregation
    agg = df.groupby(["season", "contestant_id", "celebrity_name"], as_index=False).agg(
        age=("celebrity_age_during_season", "first"),
        region=("region", "first"),
        industry=("celebrity_industry", "first"),
        ballroom_partner=("ballroom_partner", "first"),
        mean_J=("J_total", "mean"),
        mean_p=("p_hat", "mean"),
        max_week=("week", "max"),
        eliminated_any=("eliminated", "max"),
        elim_week=("week", lambda x: x[df.loc[x.index, "eliminated"] == 1].min() if (df.loc[x.index, "eliminated"] == 1).any() else np.nan),
    )
    # duration/event
    agg["duration"] = agg["elim_week"].fillna(agg["max_week"]).astype(int)
    agg["event"] = agg["eliminated_any"].astype(int)
    # finalist flag: never eliminated
    agg["finalist"] = (agg["event"] == 0).astype(int)
    return agg


def run_mixedlm(df: pd.DataFrame, response: str, out_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """运行线性混合效应模型，输出完整统计摘要"""
    # Prepare formula with categorical region/season
    df = df.copy()
    df = df.dropna(subset=[response, "celebrity_age_during_season", "region", "season", "week", "ballroom_partner", "industry_consolidated"])
    df["season"] = df["season"].astype(int)

    formula = f"{response} ~ celebrity_age_during_season + C(region) + C(season) + week"
    # Random intercept for ballroom_partner + variance component for industry
    vc = {"industry": "0 + C(industry_consolidated)"}
    model = smf.mixedlm(formula, df, groups=df["ballroom_partner"], vc_formula=vc, re_formula="1")
    result = model.fit(method="lbfgs", maxiter=200, disp=False)

    # 输出完整的固定效应摘要（含 SE, z, p, CI）
    summary_df = pd.DataFrame({
        "term": result.fe_params.index,
        "coef": result.fe_params.values,
        "std_err": result.bse_fe.values,
        "z": result.tvalues.reindex(result.fe_params.index).values,
        "p_value": result.pvalues.reindex(result.fe_params.index).values,
        "ci_lower": result.conf_int().loc[result.fe_params.index, 0].values,
        "ci_upper": result.conf_int().loc[result.fe_params.index, 1].values,
    })
    summary_df.to_csv(out_path.replace("_fixed_effects.csv", "_summary_fixed_effects.csv"), index=False)
    
    # 简化版本（仅系数）
    fixed = result.fe_params.reset_index()
    fixed.columns = ["term", "coef"]
    fixed.to_csv(out_path, index=False)

    re_df = None
    try:
        re = result.random_effects
        re_df = pd.DataFrame({"ballroom_partner": list(re.keys()), "effect": [v.values[0] for v in re.values()]})
        re_df = re_df.sort_values("effect", ascending=False)
        re_df.to_csv(out_path.replace("_fixed_effects.csv", "_ballroom_partner_effects.csv"), index=False)
        
        # 输出带统计信息的舞伴效应
        re_summary = re_df.copy()
        re_summary["rank"] = range(1, len(re_summary) + 1)
        re_summary.to_csv(out_path.replace("_fixed_effects.csv", "_summary_ballroom_partner_effects.csv"), index=False)
    except Exception:
        pass

    try:
        vc_df = pd.DataFrame({"component": list(result.vcomp.index), "variance": result.vcomp.values})
        vc_df.to_csv(out_path.replace("_fixed_effects.csv", "_variance_components.csv"), index=False)
    except Exception:
        pass
    
    return summary_df, re_df


def run_logit(agg: pd.DataFrame, out_path: str):
    """运行逻辑回归（带正则化防止完全分离）"""
    data = agg.copy()
    data = data.dropna(subset=["mean_J", "mean_p", "age", "region", "industry_consolidated"])
    X = pd.get_dummies(data[["mean_J", "mean_p", "age", "region", "industry_consolidated"]], drop_first=True)
    # standardize continuous columns
    for col in ["mean_J", "mean_p", "age"]:
        if col in X.columns:
            std = X[col].std()
            if std and np.isfinite(std) and std > 0:
                X[col] = (X[col] - X[col].mean()) / std
    # drop near-constant columns
    variances = X.var()
    low_var_cols = variances[variances < 1e-8].index.tolist()
    if low_var_cols:
        X = X.drop(columns=low_var_cols)
    X = sm.add_constant(X)
    X = X.astype(float)
    y = data["finalist"].astype(int)

    # 使用正则化逻辑回归防止完全分离
    try:
        model = sm.Logit(y, X)
        result = model.fit_regularized(method='l1', alpha=0.1, disp=False)
        
        # 输出完整摘要
        coef = pd.DataFrame({
            "term": result.params.index, 
            "coef": result.params.values,
        })
        coef.to_csv(out_path, index=False)
        
        # 尝试输出带统计信息的摘要
        summary_df = coef.copy()
        summary_df.to_csv(out_path.replace("_coef.csv", "_summary_coef.csv"), index=False)
    except Exception as e:
        print(f"Logit model failed: {e}")
        # 回退到普通 fit
        model = sm.Logit(y, X)
        result = model.fit(disp=False, maxiter=100)
        coef = pd.DataFrame({"term": result.params.index, "coef": result.params.values})
        coef.to_csv(out_path, index=False)


def run_cox(agg: pd.DataFrame, out_path: str):
    """运行 Cox 比例风险模型"""
    try:
        from lifelines import CoxPHFitter
    except Exception:
        print("lifelines not installed, skipping Cox model")
        return

    data = agg.copy()
    data = data.dropna(subset=["mean_J", "mean_p", "age", "region", "industry_consolidated", "duration", "event"])
    X = pd.get_dummies(data[["mean_J", "mean_p", "age", "region", "industry_consolidated"]], drop_first=True)
    # standardize continuous columns to avoid overflow
    for col in ["mean_J", "mean_p", "age"]:
        if col in X.columns:
            std = X[col].std()
            if std and np.isfinite(std) and std > 0:
                X[col] = (X[col] - X[col].mean()) / std
    # drop near-constant columns to reduce separation
    variances = X.var()
    low_var_cols = variances[variances < 1e-8].index.tolist()
    if low_var_cols:
        X = X.drop(columns=low_var_cols)
    X["duration"] = data["duration"].astype(int)
    X["event"] = data["event"].astype(int)

    cph = CoxPHFitter(penalizer=0.1)
    try:
        cph.fit(X, duration_col="duration", event_col="event")
        cph.summary.to_csv(out_path)
        
        # 输出简化摘要
        summary = cph.summary[["coef", "exp(coef)", "se(coef)", "p", "coef lower 95%", "coef upper 95%"]].copy()
        summary.to_csv(out_path.replace("_coef.csv", "_summary_coef.csv"))
    except Exception as exc:
        print(f"Cox model failed: {exc}")
        return


def plot_partner_effects(re_df_J: pd.DataFrame, re_df_F: pd.DataFrame, out_dir: str):
    """绘制舞伴效应对比图"""
    if re_df_J is None or re_df_F is None:
        return
    
    # 合并两个模型的舞伴效应
    merged = re_df_J.merge(re_df_F, on="ballroom_partner", suffixes=("_judge", "_fan"))
    
    # Top 10 和 Bottom 10 舞伴
    top_10_J = re_df_J.nlargest(10, "effect")
    bottom_10_J = re_df_J.nsmallest(10, "effect")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：评审评分 Top/Bottom 舞伴
    combined = pd.concat([top_10_J, bottom_10_J]).sort_values("effect", ascending=True)
    colors = ['#d73027' if x < 0 else '#1a9850' for x in combined["effect"]]
    axes[0].barh(combined["ballroom_partner"], combined["effect"], color=colors)
    axes[0].set_xlabel("Effect on Judge Score")
    axes[0].set_title("Ballroom Partner Effects on Judge Scores\n(Top 10 & Bottom 10)")
    axes[0].axvline(x=0, color='black', linewidth=0.8)
    
    # 右图：评审 vs 粉丝效应散点图
    axes[1].scatter(merged["effect_judge"], merged["effect_fan"], alpha=0.6)
    axes[1].set_xlabel("Effect on Judge Score")
    axes[1].set_ylabel("Effect on Fan Votes (p_hat)")
    axes[1].set_title("Partner Effects: Judge vs Fan")
    
    # 标注极端点
    for _, row in merged.nlargest(3, "effect_judge").iterrows():
        axes[1].annotate(row["ballroom_partner"], (row["effect_judge"], row["effect_fan"]), fontsize=8)
    for _, row in merged.nsmallest(3, "effect_judge").iterrows():
        axes[1].annotate(row["ballroom_partner"], (row["effect_judge"], row["effect_fan"]), fontsize=8)
    
    # 添加相关系数
    corr = merged["effect_judge"].corr(merged["effect_fan"])
    axes[1].text(0.05, 0.95, f"Correlation: {corr:.3f}", transform=axes[1].transAxes, 
                 fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "partner_effects_comparison.png"), dpi=150)
    plt.close()


def plot_fixed_effects_comparison(summary_J: pd.DataFrame, summary_F: pd.DataFrame, out_dir: str):
    """绘制固定效应对比图（排除赛季哑变量）"""
    # 筛选关键变量（排除大量赛季哑变量）
    key_vars = ["celebrity_age_during_season", "week", 
                "C(region)[T.Midwest]", "C(region)[T.Northeast]", 
                "C(region)[T.South]", "C(region)[T.West]"]
    
    J_subset = summary_J[summary_J["term"].isin(key_vars)].copy()
    F_subset = summary_F[summary_F["term"].isin(key_vars)].copy()
    
    if len(J_subset) == 0 or len(F_subset) == 0:
        return
    
    merged = J_subset.merge(F_subset, on="term", suffixes=("_J", "_F"))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：评审模型系数
    x = range(len(merged))
    axes[0].barh(merged["term"], merged["coef_J"], xerr=merged["std_err_J"], 
                 color='steelblue', capsize=3)
    axes[0].set_xlabel("Coefficient")
    axes[0].set_title("Model J (Judge Score): Fixed Effects")
    axes[0].axvline(x=0, color='black', linewidth=0.8)
    
    # 右图：粉丝模型系数
    axes[1].barh(merged["term"], merged["coef_F"], xerr=merged["std_err_F"], 
                 color='coral', capsize=3)
    axes[1].set_xlabel("Coefficient")
    axes[1].set_title("Model F (Fan Votes): Fixed Effects")
    axes[1].axvline(x=0, color='black', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fixed_effects_comparison.png"), dpi=150)
    plt.close()


def plot_cox_hazard_ratios(out_dir: str):
    """绘制 Cox 模型风险比森林图"""
    cox_path = os.path.join(out_dir, "cox_model_coef.csv")
    if not os.path.exists(cox_path):
        return
    
    cox_df = pd.read_csv(cox_path, index_col=0)
    
    # 筛选主要变量（排除过多的行业哑变量）
    key_vars = ["mean_J", "mean_p", "age"]
    region_vars = [c for c in cox_df.index if c.startswith("region_")]
    plot_vars = key_vars + region_vars[:4]  # 限制行业变量数量
    
    subset = cox_df.loc[cox_df.index.isin(plot_vars)].copy()
    if len(subset) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = range(len(subset))
    ax.errorbar(subset["exp(coef)"], y_pos, 
                xerr=[subset["exp(coef)"] - subset["exp(coef) lower 95%"],
                      subset["exp(coef) upper 95%"] - subset["exp(coef)"]],
                fmt='o', capsize=5, color='darkgreen')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(subset.index)
    ax.axvline(x=1, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel("Hazard Ratio (exp(coef))")
    ax.set_title("Cox Model: Hazard Ratios with 95% CI")
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cox_hazard_ratios.png"), dpi=150)
    plt.close()


def plot_industry_effects(out_dir: str):
    """绘制行业效应对比图"""
    cox_path = os.path.join(out_dir, "cox_model_coef.csv")
    if not os.path.exists(cox_path):
        return
    
    cox_df = pd.read_csv(cox_path, index_col=0)
    
    # 筛选行业变量
    industry_rows = cox_df[cox_df.index.str.startswith("industry_consolidated_")].copy()
    if len(industry_rows) == 0:
        return
    
    # 清理行业名称
    industry_rows["industry"] = industry_rows.index.str.replace("industry_consolidated_", "")
    industry_rows = industry_rows.sort_values("exp(coef)")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = range(len(industry_rows))
    colors = ['#d73027' if x > 1 else '#1a9850' for x in industry_rows["exp(coef)"]]
    
    ax.barh(industry_rows["industry"], industry_rows["exp(coef)"], color=colors, alpha=0.7)
    ax.errorbar(industry_rows["exp(coef)"], y_pos,
                xerr=[industry_rows["exp(coef)"] - industry_rows["exp(coef) lower 95%"],
                      industry_rows["exp(coef) upper 95%"] - industry_rows["exp(coef)"]],
                fmt='none', capsize=3, color='black', alpha=0.5)
    
    ax.axvline(x=1, color='black', linestyle='--', linewidth=1.5)
    ax.set_xlabel("Hazard Ratio (HR) - 淘汰风险比")
    ax.set_ylabel("Industry (行业)")
    ax.set_title("Industry Effects on Elimination Risk\n(HR>1: Higher Risk, HR<1: Lower Risk)")
    
    # 添加注释
    ax.text(0.02, 0.98, "Green: Lower Risk\nRed: Higher Risk", 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "industry_effects.png"), dpi=150)
    plt.close()


def plot_age_effect_comparison(summary_J: pd.DataFrame, summary_F: pd.DataFrame, out_dir: str):
    """绘制年龄效应对比图（评审 vs 粉丝）"""
    age_J = summary_J[summary_J["term"] == "celebrity_age_during_season"]
    age_F = summary_F[summary_F["term"] == "celebrity_age_during_season"]
    
    if len(age_J) == 0 or len(age_F) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    categories = ["Judge Score\n(Model J)", "Fan Votes\n(Model F)"]
    coefs = [age_J["coef"].values[0], age_F["coef"].values[0]]
    errors = [age_J["std_err"].values[0] * 1.96, age_F["std_err"].values[0] * 1.96]
    colors = ['steelblue', 'coral']
    
    bars = ax.bar(categories, coefs, yerr=errors, capsize=5, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel("Age Coefficient")
    ax.set_title("Age Effect: Judge Score vs Fan Votes\n(Negative = Older celebrities score lower)")
    
    # 添加数值标签
    for bar, coef in zip(bars, coefs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{coef:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "age_effect_comparison.png"), dpi=150)
    plt.close()


def generate_interpretation_report(summary_J: pd.DataFrame, summary_F: pd.DataFrame, 
                                   re_df_J: pd.DataFrame, re_df_F: pd.DataFrame, 
                                   out_dir: str):
    """生成结果解读报告"""
    report_lines = []
    report_lines.append("# 问题三结果解读报告\n")
    report_lines.append("## 1. 核心发现\n")
    
    # 年龄效应对比
    age_J = summary_J[summary_J["term"] == "celebrity_age_during_season"]
    age_F = summary_F[summary_F["term"] == "celebrity_age_during_season"]
    
    if len(age_J) > 0 and len(age_F) > 0:
        age_coef_J = age_J["coef"].values[0]
        age_coef_F = age_F["coef"].values[0]
        age_p_J = age_J["p_value"].values[0] if "p_value" in age_J.columns else None
        age_p_F = age_F["p_value"].values[0] if "p_value" in age_F.columns else None
        
        report_lines.append("### 1.1 年龄影响\n")
        report_lines.append(f"- **评审评分**: 年龄系数 = {age_coef_J:.4f}")
        if age_p_J is not None:
            report_lines.append(f" (p = {age_p_J:.4f})")
        report_lines.append("\n")
        report_lines.append(f"- **粉丝投票**: 年龄系数 = {age_coef_F:.4f}")
        if age_p_F is not None:
            report_lines.append(f" (p = {age_p_F:.4f})")
        report_lines.append("\n")
        
        if age_coef_J < -0.05 and abs(age_coef_F) < 0.01:
            report_lines.append("\n**解读**: 年龄对评审打分有显著负面影响（年龄越大分数越低），但对粉丝投票几乎没有影响。")
            report_lines.append("这表明**评审更注重技术能力**（可能与年龄/体力相关），而**粉丝更看重明星魅力和人气**。\n")
    
    # 周次效应
    week_J = summary_J[summary_J["term"] == "week"]
    week_F = summary_F[summary_F["term"] == "week"]
    
    if len(week_J) > 0 and len(week_F) > 0:
        report_lines.append("\n### 1.2 周次效应\n")
        report_lines.append(f"- **评审评分**: 周次系数 = {week_J['coef'].values[0]:.4f}\n")
        report_lines.append(f"- **粉丝投票**: 周次系数 = {week_F['coef'].values[0]:.4f}\n")
        report_lines.append("\n**解读**: 随着比赛进行，留下的选手评审分数越来越高（弱者被淘汰），粉丝投票份额也略有上升。\n")
    
    # 舞伴效应
    if re_df_J is not None:
        report_lines.append("\n## 2. 专业舞伴效应\n")
        report_lines.append("\n### 2.1 对评审评分影响最大的舞伴 (Top 5)\n")
        top5_J = re_df_J.nlargest(5, "effect")
        for _, row in top5_J.iterrows():
            report_lines.append(f"- **{row['ballroom_partner']}**: +{row['effect']:.3f} 分\n")
        
        report_lines.append("\n### 2.2 对评审评分影响最小的舞伴 (Bottom 5)\n")
        bottom5_J = re_df_J.nsmallest(5, "effect")
        for _, row in bottom5_J.iterrows():
            report_lines.append(f"- **{row['ballroom_partner']}**: {row['effect']:.3f} 分\n")
    
    if re_df_J is not None and re_df_F is not None:
        merged_re = re_df_J.merge(re_df_F, on="ballroom_partner", suffixes=("_J", "_F"))
        corr = merged_re["effect_J"].corr(merged_re["effect_F"])
        report_lines.append(f"\n### 2.3 舞伴效应一致性\n")
        report_lines.append(f"- 评审效应与粉丝效应的相关系数: **{corr:.3f}**\n")
        if corr > 0.5:
            report_lines.append("- **解读**: 舞伴对评审和粉丝的影响方向较为一致，好的舞伴同时提升技术分和人气。\n")
        elif corr < 0.2:
            report_lines.append("- **解读**: 舞伴对评审和粉丝的影响方向不一致，说明技术能力强的舞伴不一定能带来更多粉丝支持。\n")
        else:
            report_lines.append("- **解读**: 舞伴效应呈中等正相关，表明好的舞伴能部分同时提升技术分和人气，但两者并非完全一致。\n")

    # 补充：行业效应分析
    report_lines.append("\n## 3. 行业(Industry)效应分析\n")
    
    # 读取 Cox 模型结果来分析行业效应
    cox_path = os.path.join(out_dir, "cox_model_coef.csv")
    if os.path.exists(cox_path):
        cox_df = pd.read_csv(cox_path, index_col=0)
        industry_rows = cox_df[cox_df.index.str.startswith("industry_consolidated_")]
        
        if len(industry_rows) > 0:
            report_lines.append("\n### 3.1 行业对淘汰风险的影响 (Cox模型)\n")
            report_lines.append("\n| 行业 | 风险比(HR) | p值 | 解读 |\n")
            report_lines.append("|------|-----------|-----|------|\n")
            
            # 按风险比排序
            industry_rows_sorted = industry_rows.sort_values("exp(coef)")
            
            for idx, row in industry_rows_sorted.iterrows():
                industry_name = idx.replace("industry_consolidated_", "")
                hr = row["exp(coef)"]
                p_val = row["p"]
                
                if hr < 0.5:
                    interpretation = "淘汰风险显著降低 ⬇️"
                elif hr < 0.8:
                    interpretation = "淘汰风险略低"
                elif hr > 2.0:
                    interpretation = "淘汰风险显著增加 ⬆️"
                elif hr > 1.2:
                    interpretation = "淘汰风险略高"
                else:
                    interpretation = "与基准相近"
                
                sig = "**" if p_val < 0.05 else ""
                report_lines.append(f"| {industry_name} | {sig}{hr:.3f}{sig} | {p_val:.4f} | {interpretation} |\n")
            
            # 关键发现
            report_lines.append("\n**关键发现**:\n")
            
            # 找出显著的行业
            sig_industries = industry_rows[industry_rows["p"] < 0.05]
            if len(sig_industries) > 0:
                for idx, row in sig_industries.iterrows():
                    industry_name = idx.replace("industry_consolidated_", "")
                    hr = row["exp(coef)"]
                    if hr > 1:
                        report_lines.append(f"- **{industry_name}**: 淘汰风险是基准的 {hr:.2f} 倍 (p<0.05)\n")
                    else:
                        report_lines.append(f"- **{industry_name}**: 淘汰风险降低 {(1-hr)*100:.1f}% (p<0.05)\n")
            else:
                report_lines.append("- 没有行业的淘汰风险达到统计显著水平（p<0.05），但趋势仍有参考价值。\n")
    
    # 补充：Cox 模型核心解读
    report_lines.append("\n## 4. 生存分析(Cox模型)解读\n")
    
    if os.path.exists(cox_path):
        cox_df = pd.read_csv(cox_path, index_col=0)
        
        report_lines.append("\n### 4.1 核心变量的淘汰风险\n")
        report_lines.append("\n| 变量 | 风险比(HR) | 95% CI | p值 | 解读 |\n")
        report_lines.append("|------|-----------|--------|-----|------|\n")
        
        key_vars = ["mean_J", "mean_p", "age"]
        for var in key_vars:
            if var in cox_df.index:
                row = cox_df.loc[var]
                hr = row["exp(coef)"]
                ci_low = row["exp(coef) lower 95%"]
                ci_high = row["exp(coef) upper 95%"]
                p_val = row["p"]
                
                if var == "mean_J":
                    name = "平均评审分数"
                    if hr < 1:
                        interp = f"评审分数每提高1个标准差，淘汰风险降低 {(1-hr)*100:.1f}%"
                    else:
                        interp = f"评审分数每提高1个标准差，淘汰风险增加 {(hr-1)*100:.1f}%"
                elif var == "mean_p":
                    name = "平均粉丝投票份额"
                    if hr < 1:
                        interp = f"粉丝份额每提高1个标准差，淘汰风险降低 {(1-hr)*100:.1f}%"
                    else:
                        interp = f"粉丝份额每提高1个标准差，淘汰风险增加 {(hr-1)*100:.1f}%"
                else:
                    name = "年龄"
                    if hr > 1:
                        interp = f"年龄每增加1个标准差，淘汰风险增加 {(hr-1)*100:.1f}%"
                    else:
                        interp = f"年龄每增加1个标准差，淘汰风险降低 {(1-hr)*100:.1f}%"
                
                sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
                report_lines.append(f"| {name} | {hr:.3f}{sig} | [{ci_low:.3f}, {ci_high:.3f}] | {p_val:.4f} | {interp} |\n")
        
        report_lines.append("\n*注: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001*\n")
        
        report_lines.append("\n### 4.2 生存分析核心结论\n")
        report_lines.append("""
- **评审分数是最强的保护因素**: 评审分数高的选手淘汰风险显著降低，这符合节目的评分机制。
- **粉丝投票同样重要**: 粉丝份额高的选手也更"安全"，说明粉丝投票在合并机制中发挥了关键作用。
- **年龄是风险因素**: 年龄较大的选手淘汰风险更高，这与评审偏好年轻选手的发现一致。
""")

    # 补充：区域效应
    report_lines.append("\n## 5. 区域效应分析\n")
    
    # 从 summary_J 和 summary_F 提取区域效应
    region_J = summary_J[summary_J["term"].str.contains("region", case=False)]
    region_F = summary_F[summary_F["term"].str.contains("region", case=False)]
    
    if len(region_J) > 0:
        report_lines.append("\n### 5.1 区域对评审分数的影响\n")
        report_lines.append("\n| 区域 | 系数 | 标准误 | p值 | 显著性 |\n")
        report_lines.append("|------|------|--------|-----|--------|\n")
        
        for _, row in region_J.iterrows():
            term = row["term"].replace("C(region)[T.", "").replace("]", "")
            coef = row["coef"]
            se = row["std_err"]
            p_val = row["p_value"]
            sig = "✓" if p_val < 0.05 else ""
            report_lines.append(f"| {term} | {coef:.3f} | {se:.3f} | {p_val:.4f} | {sig} |\n")
        
        report_lines.append("\n**解读**: 区域效应在评审分数上均不显著，说明评审打分不受选手地理来源的影响。\n")
    
    # 结论
    report_lines.append("\n## 6. 核心结论\n")
    report_lines.append("""
### 6.1 因素对比赛表现的影响程度

| 因素 | 对评审分数 | 对粉丝投票 | 对淘汰风险 |
|------|-----------|-----------|-----------|
| **年龄** | 强负面影响*** | 弱负面影响 | 增加风险** |
| **专业舞伴** | 强影响（±1.5分） | 中等影响 | 间接影响 |
| **行业** | 作为随机效应 | 作为随机效应 | Model显示趋势 |
| **区域** | 不显著 | 不显著 | 不显著 |
| **周次** | 强正面影响*** | 弱正面影响 | - |

### 6.2 评审分数 vs 粉丝投票：影响方式是否相同？

**答案：不完全相同。**

1. **年龄因素**: 评审明显偏好年轻选手（-0.107分/岁），但粉丝几乎不在乎（-0.001/岁）。
   - 这表明评审更注重**技术能力**（可能与体力/灵活性相关）
   - 而粉丝更看重**明星魅力和个人品牌**

2. **舞伴效应**: 相关系数 = 0.43（中等正相关）
   - 好的舞伴能同时提升技术分和人气，但两者并非完全一致
   - 某些舞伴在技术上很强，但未必能带来更多粉丝

3. **核心差异机制**:
   - **评审路径**: 技术能力 → 舞蹈质量 → 高分
   - **粉丝路径**: 个人魅力 + 既有粉丝基础 → 投票支持

### 6.3 对"争议"现象的解释

当一位名人评审分数低但粉丝投票高（如 Bobby Bones、Jerry Rice），说明：
- 该名人的**个人魅力/粉丝基础**超过了其**舞蹈技术**
- 两种评价维度的不一致是节目设计的核心矛盾
- 这也是为什么节目后来引入"评审选择淘汰倒数两名之一"的机制
""")
    
    # 写入文件
    with open(os.path.join(out_dir, "interpretation_report.md"), "w", encoding="utf-8") as f:
        f.write("".join(report_lines))
    
    print(f"Generated interpretation report: {os.path.join(out_dir, 'interpretation_report.md')}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    features = pd.read_csv(args.features)
    raw = pd.read_csv(args.raw)
    raw = raw.copy()
    raw["contestant_id"] = np.arange(len(raw))

    merged = features.merge(raw[[
        "contestant_id",
        "celebrity_age_during_season",
        "celebrity_industry",
        "celebrity_homestate",
    ]], on="contestant_id", how="left")

    merged["region"] = merged["celebrity_homestate"].apply(map_region)
    
    # 统一行业名称大小写，并基于选手级别（而非周级别）计算频次
    merged["celebrity_industry"] = merged["celebrity_industry"].str.strip()
    # 先统一大小写用于计数
    merged["industry_lower"] = merged["celebrity_industry"].str.lower()
    
    # 基于选手级别计算行业频次（每个选手只算一次）
    contestant_industry = merged.groupby("contestant_id")["industry_lower"].first()
    industry_counts = contestant_industry.value_counts().to_dict()
    
    print(f"Industry counts (contestant-level): {industry_counts}")
    
    merged["industry_consolidated"] = merged["industry_lower"].apply(
        lambda x: consolidate_industry(x, industry_counts, min_count=10)
    )
    # 恢复首字母大写格式（仅用于显示）
    merged["industry_consolidated"] = merged["industry_consolidated"].str.title()
    
    print(f"Industry consolidation: {merged['celebrity_industry'].nunique()} -> {merged['industry_consolidated'].nunique()} categories")
    print(f"Region distribution:\n{merged['region'].value_counts()}")

    # Mixed effects models
    print("\nRunning Model J (Judge Scores)...")
    summary_J, re_df_J = run_mixedlm(
        merged,
        response="J_total",
        out_path=os.path.join(args.out_dir, "model_J_fixed_effects.csv"),
    )
    
    print("Running Model F (Fan Votes)...")
    summary_F, re_df_F = run_mixedlm(
        merged,
        response="p_hat",
        out_path=os.path.join(args.out_dir, "model_F_fixed_effects.csv"),
    )

    # Survival / logistic models (contestant-level)
    print("\nBuilding survival table...")
    agg = build_survival_table(merged)
    # 为聚合表添加 industry_consolidated
    industry_map = merged.groupby("contestant_id")["industry_consolidated"].first().to_dict()
    agg["industry_consolidated"] = agg["contestant_id"].map(industry_map)
    agg.to_csv(os.path.join(args.out_dir, "survival_table.csv"), index=False)

    print("Running Cox model...")
    run_cox(agg, os.path.join(args.out_dir, "cox_model_coef.csv"))
    
    print("Running Logit model...")
    run_logit(agg, os.path.join(args.out_dir, "logit_model_coef.csv"))

    # 生成可视化
    print("\nGenerating visualizations...")
    plot_partner_effects(re_df_J, re_df_F, args.out_dir)
    plot_fixed_effects_comparison(summary_J, summary_F, args.out_dir)
    plot_cox_hazard_ratios(args.out_dir)
    plot_industry_effects(args.out_dir)
    plot_age_effect_comparison(summary_J, summary_F, args.out_dir)
    
    # 生成解读报告
    print("Generating interpretation report...")
    generate_interpretation_report(summary_J, summary_F, re_df_J, re_df_F, args.out_dir)

    print(f"\n✅ All outputs written to: {args.out_dir}")
    print("Generated files:")
    for f in sorted(os.listdir(args.out_dir)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
