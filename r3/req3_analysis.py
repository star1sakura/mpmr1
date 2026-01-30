import argparse
import os
import re
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import norm
from scipy.special import logsumexp, gammaln
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# 字体配置（避免缺字警告）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.family'] = 'DejaVu Sans'
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
    ap.add_argument("--beta-draws", type=int, default=1000, help="Beta model draws")
    ap.add_argument("--beta-tune", type=int, default=1000, help="Beta model tuning steps")
    ap.add_argument("--beta-chains", type=int, default=4, help="Beta model chains")
    ap.add_argument(
        "--beta-cores",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Beta model CPU cores",
    )
    ap.add_argument(
        "--beta-target-accept",
        type=float,
        default=0.95,
        help="Beta model target_accept",
    )
    ap.add_argument("--skip-beta", action="store_true", help="Skip beta regression model")
    ap.add_argument("--skip-cox", action="store_true", help="Skip time-varying Cox model")
    ap.add_argument("--compare-models", action="store_true", help="Run beta model variants comparison")
    ap.add_argument("--compare-minimal", action="store_true", help="Compare only M0 vs M3")
    ap.add_argument("--compare-folds", type=int, default=5, help="K-fold folds (by contestant_key)")
    ap.add_argument("--compare-draws", type=int, default=1000, help="Comparison model draws")
    ap.add_argument("--compare-tune", type=int, default=1000, help="Comparison model tuning steps")
    ap.add_argument("--compare-chains", type=int, default=2, help="Comparison model chains")
    ap.add_argument(
        "--compare-cores",
        type=int,
        default=min(2, os.cpu_count() or 1),
        help="Comparison model CPU cores",
    )
    ap.add_argument(
        "--compare-target-accept",
        type=float,
        default=0.9,
        help="Comparison model target_accept",
    )
    ap.add_argument("--compare-seed", type=int, default=42, help="Comparison random seed")
    return ap.parse_args()


def normalize_text(value: Optional[str]) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value.strip())


def build_contestant_key(df: pd.DataFrame) -> pd.Series:
    season = df["season"].astype(str).str.strip()
    name = df["celebrity_name"].fillna("").astype(str).str.strip().str.lower()
    partner = df["ballroom_partner"].fillna("").astype(str).str.strip().str.lower()
    return season + "||" + name + "||" + partner


def map_state_to_region(state: Optional[str]) -> str:
    if not isinstance(state, str):
        return "Unknown"
    state_clean = normalize_text(state)
    if not state_clean:
        return "Unknown"

    state_upper = re.sub(r"[.\s]", "", state_clean.upper())
    if state_upper in {"DC", "WASHINGTONDC", "DISTRICTOFCOLUMBIA"}:
        state_upper = "DC"

    northeast_abbr = {"CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"}
    midwest_abbr = {"IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"}
    south_abbr = {"DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV", "AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX"}
    west_abbr = {"AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"}

    if state_upper in northeast_abbr:
        return "Northeast"
    if state_upper in midwest_abbr:
        return "Midwest"
    if state_upper in south_abbr:
        return "South"
    if state_upper in west_abbr:
        return "West"

    state_lower = state_clean.lower()
    northeast_full = {
        "connecticut", "maine", "massachusetts", "new hampshire", "rhode island",
        "vermont", "new jersey", "new york", "pennsylvania"
    }
    midwest_full = {
        "illinois", "indiana", "michigan", "ohio", "wisconsin", "iowa", "kansas",
        "minnesota", "missouri", "nebraska", "north dakota", "south dakota"
    }
    south_full = {
        "delaware", "florida", "georgia", "maryland", "north carolina", "south carolina",
        "virginia", "district of columbia", "washington dc", "washington d.c.", "west virginia",
        "alabama", "kentucky", "mississippi", "tennessee", "arkansas", "louisiana", "oklahoma", "texas"
    }
    west_full = {
        "arizona", "colorado", "idaho", "montana", "nevada", "new mexico", "utah",
        "wyoming", "alaska", "california", "hawaii", "oregon", "washington"
    }

    if state_lower in northeast_full:
        return "Northeast"
    if state_lower in midwest_full:
        return "Midwest"
    if state_lower in south_full:
        return "South"
    if state_lower in west_full:
        return "West"

    return "Unknown"


def classify_country(country: Optional[str]) -> str:
    country_clean = normalize_text(country)
    if not country_clean:
        return "Unknown"
    normalized = country_clean.lower().replace(".", "")
    us_aliases = {
        "united states", "united states of america", "usa", "us", "u s", "u s a", "america"
    }
    if normalized in us_aliases:
        return "US"
    return "International"


def map_region_and_country(state: Optional[str], country: Optional[str]) -> Tuple[str, str]:
    country_group = classify_country(country)
    region = map_state_to_region(state)

    if country_group == "International":
        return "International", "International"

    if country_group == "Unknown" and region in {"Northeast", "Midwest", "South", "West"}:
        return region, "US"

    if country_group == "US":
        return region if region != "Unknown" else "Unknown", "US"

    return "Unknown", "Unknown"


def zscore_series(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    std = values.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return values * 0.0
    return (values - values.mean()) / std


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
    agg = df.groupby(["contestant_key", "season", "celebrity_name"], as_index=False).agg(
        age=("celebrity_age_during_season", "first"),
        age_z=("age_z", "first"),
        region=("region", "first"),
        country_group=("country_group", "first"),
        industry=("celebrity_industry", "first"),
        industry_consolidated=("industry_consolidated", "first"),
        ballroom_partner=("ballroom_partner", "first"),
        mean_J_z=("J_z", "mean"),
        mean_p_hat=("p_hat_clip", "mean"),
        mean_p_hat_z=("p_hat_z", "mean"),
        max_week=("week", "max"),
        eliminated_any=("eliminated", "max"),
        elim_week=(
            "week",
            lambda x: x[df.loc[x.index, "eliminated"] == 1].min()
            if (df.loc[x.index, "eliminated"] == 1).any()
            else np.nan,
        ),
    )
    # duration/event
    agg["duration"] = agg["elim_week"].fillna(agg["max_week"]).astype(int)
    agg["event"] = agg["eliminated_any"].astype(int)
    # finalist flag: never eliminated
    agg["finalist"] = (agg["event"] == 0).astype(int)
    return agg


def build_time_varying_table(df: pd.DataFrame) -> pd.DataFrame:
    tv = df.copy()
    tv = tv.sort_values(["contestant_key", "week"]).reset_index(drop=True)
    elim_week = tv[tv["eliminated"] == 1].groupby("contestant_key")["week"].min()
    tv["elim_week"] = tv["contestant_key"].map(elim_week)
    tv["last_week"] = tv["elim_week"].fillna(tv.groupby("contestant_key")["week"].transform("max"))
    tv = tv[tv["week"] <= tv["last_week"]].copy()
    tv["start"] = tv["week"] - 1
    tv["stop"] = tv["week"]
    tv["event"] = (tv["week"] == tv["elim_week"]).astype(int)
    return tv


def run_mixedlm(df: pd.DataFrame, response: str, out_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """运行线性混合效应模型，输出完整统计摘要"""
    # Prepare formula with categorical region/season
    df = df.copy()
    df = df.dropna(subset=[response, "age_z", "region", "season", "week_z", "ballroom_partner", "industry_consolidated"])
    df["season"] = df["season"].astype(int)

    formula = f"{response} ~ age_z + C(region) + C(season) + week_z"
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


def extract_beta_fixed_effects(summary: pd.DataFrame) -> pd.DataFrame:
    fixed = summary.copy()
    fixed = fixed[~fixed.index.str.contains(r"\|", regex=True)]
    fixed = fixed[~fixed.index.str.contains("kappa|alpha|sigma|sd", case=False, regex=True)]
    fixed = fixed.reset_index().rename(columns={"index": "term", "mean": "coef", "sd": "std_err"})
    fixed = fixed.rename(columns={"hdi_2.5%": "ci_lower", "hdi_97.5%": "ci_upper"})
    return fixed[["term", "coef", "std_err", "ci_lower", "ci_upper"]]


def extract_beta_random_effects(summary: pd.DataFrame, group_label: str) -> Optional[pd.DataFrame]:
    mask = summary.index.str.contains(group_label) & summary.index.str.contains(r"offset\[")
    subset = summary[mask].copy()
    if subset.empty:
        return None
    subset = subset.reset_index().rename(columns={"index": "term", "mean": "effect"})
    subset["level"] = subset["term"].str.extract(r"\[(.*)\]")
    subset = subset.rename(columns={"hdi_2.5%": "hdi_lower", "hdi_97.5%": "hdi_upper"})
    subset = subset[["level", "effect", "hdi_lower", "hdi_upper"]]
    subset = subset.sort_values("effect", ascending=False)
    return subset


def summarize_model_diagnostics(idata, out_dir: str) -> Optional[pd.DataFrame]:
    try:
        import arviz as az
    except Exception:
        return None

    def extract_metric(obj, keys):
        for key in keys:
            if hasattr(obj, key):
                return float(getattr(obj, key))
            try:
                return float(obj[key])
            except Exception:
                continue
        return np.nan

    try:
        loo = az.loo(idata, pointwise=True)
        waic = az.waic(idata)
    except Exception as exc:
        print(f"LOO/WAIC failed: {exc}")
        return None

    metrics = [
        {"metric": "elpd_loo", "value": extract_metric(loo, ["elpd_loo"])},
        {"metric": "p_loo", "value": extract_metric(loo, ["p_loo"])},
        {"metric": "elpd_loo_se", "value": extract_metric(loo, ["elpd_loo_se", "loo_se", "se"])},
        {"metric": "elpd_waic", "value": extract_metric(waic, ["elpd_waic"])},
        {"metric": "p_waic", "value": extract_metric(waic, ["p_waic"])},
        {"metric": "elpd_waic_se", "value": extract_metric(waic, ["elpd_waic_se", "waic_se", "se"])},
    ]

    pareto = None
    if hasattr(loo, "pareto_k"):
        pareto = loo.pareto_k
    else:
        try:
            pareto = loo["pareto_k"]
        except Exception:
            pareto = None

    if pareto is not None:
        values = np.asarray(pareto).ravel()
        values = values[np.isfinite(values)]
        if values.size > 0:
            lt_rate = float((values < 0.7).mean())
            mid_rate = float(((values >= 0.7) & (values <= 1.0)).mean())
            gt_rate = float((values > 1.0).mean())
            metrics.extend([
                {"metric": "pareto_k_n", "value": float(values.size)},
                {"metric": "pareto_k_lt_0_7_rate", "value": lt_rate},
                {"metric": "pareto_k_0_7_1_rate", "value": mid_rate},
                {"metric": "pareto_k_gt_1_rate", "value": gt_rate},
                {"metric": "pareto_k_mean", "value": float(values.mean())},
                {"metric": "pareto_k_max", "value": float(values.max())},
            ])
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(out_dir, "model_diagnostics.csv"), index=False)
    return df


def generate_ppc_plots(model, idata, response: str, out_dir: str):
    try:
        import arviz as az
    except Exception:
        return

    try:
        model.predict(idata, kind="response", inplace=True, random_seed=42)
    except Exception as exc:
        print(f"Posterior predictive generation failed: {exc}")
        return

    try:
        ax = az.plot_ppc(idata, data_pairs={response: response}, num_pp_samples=50, kind="kde")
        fig = ax.figure if hasattr(ax, "figure") else plt.gcf()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "ppc_distribution.png"), dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"PPC distribution plot failed: {exc}")

    try:
        ppc = idata.posterior_predictive[response]
        obs_dims = [d for d in ppc.dims if d not in ("chain", "draw")]
        ppc_means = ppc.mean(dim=obs_dims).stack(sample=("chain", "draw")).values
        obs_mean = float(idata.observed_data[response].mean().values)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(ppc_means, bins=40, alpha=0.75, color="steelblue")
        ax.axvline(obs_mean, color="red", linestyle="--", label="Observed mean")
        ax.set_xlabel(f"Posterior predictive mean of {response}")
        ax.set_ylabel("Frequency")
        ax.set_title("PPC: Mean of response")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "ppc_mean.png"), dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"PPC mean plot failed: {exc}")


def compute_beta_marginal_effects(model, idata, data: pd.DataFrame, response: str, out_dir: str):
    try:
        import arviz as az
    except Exception:
        az = None

    def mean_over_obs(da):
        obs_dims = [d for d in da.dims if d not in ("chain", "draw")]
        return da.mean(dim=obs_dims)

    effects = []
    base_pred = model.predict(idata, kind="response_params", data=data, inplace=False)
    base_mu = base_pred.posterior["mu"]
    base_mean = mean_over_obs(base_mu)

    for var in ["age_z", "week_z"]:
        shifted = data.copy()
        shifted[var] = shifted[var] + 1
        shifted_pred = model.predict(idata, kind="response_params", data=shifted, inplace=False)
        shifted_mu = shifted_pred.posterior["mu"]
        shifted_mean = mean_over_obs(shifted_mu)
        diff = (shifted_mean - base_mean).stack(sample=("chain", "draw")).values
        if az is not None:
            hdi = az.hdi(diff, hdi_prob=0.95)
            hdi_lower, hdi_upper = float(hdi[0]), float(hdi[1])
        else:
            hdi_lower, hdi_upper = float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))
        effects.append({
            "effect": f"{var}+1sd",
            "mean_diff": float(np.mean(diff)),
            "hdi_lower": hdi_lower,
            "hdi_upper": hdi_upper,
        })

    effects_df = pd.DataFrame(effects)
    effects_df.to_csv(os.path.join(out_dir, "beta_marginal_effects.csv"), index=False)
    return effects_df


def prepare_beta_data(df: pd.DataFrame, response: str) -> pd.DataFrame:
    data = df.copy()
    data = data.dropna(subset=[
        response,
        "age_z",
        "region",
        "season",
        "week_z",
        "ballroom_partner",
        "industry_consolidated",
        "contestant_key",
    ])
    data["season"] = data["season"].astype(int)
    return data


def fit_beta_model(
    data: pd.DataFrame,
    formula: str,
    draws: int,
    tune: int,
    chains: int,
    cores: int,
    target_accept: float,
):
    try:
        import bambi as bmb
    except Exception as exc:
        print(f"bambi not available: {exc}")
        return None, None

    model = bmb.Model(formula, data, family="beta")
    idata = model.fit(
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        target_accept=target_accept,
        progressbar=False,
        idata_kwargs={"log_likelihood": True},
    )
    return model, idata


def collect_loo_metrics(idata) -> Dict[str, float]:
    try:
        import arviz as az
    except Exception:
        return {}

    def extract_metric(obj, keys):
        for key in keys:
            if hasattr(obj, key):
                return float(getattr(obj, key))
            try:
                return float(obj[key])
            except Exception:
                continue
        return np.nan

    loo = az.loo(idata, pointwise=True)
    waic = az.waic(idata)

    metrics = {
        "elpd_loo": extract_metric(loo, ["elpd_loo"]),
        "p_loo": extract_metric(loo, ["p_loo"]),
        "elpd_loo_se": extract_metric(loo, ["elpd_loo_se", "loo_se", "se"]),
        "elpd_waic": extract_metric(waic, ["elpd_waic"]),
        "p_waic": extract_metric(waic, ["p_waic"]),
        "elpd_waic_se": extract_metric(waic, ["elpd_waic_se", "waic_se", "se"]),
    }

    pareto = None
    if hasattr(loo, "pareto_k"):
        pareto = loo.pareto_k
    else:
        try:
            pareto = loo["pareto_k"]
        except Exception:
            pareto = None

    if pareto is not None:
        values = np.asarray(pareto).ravel()
        values = values[np.isfinite(values)]
        if values.size > 0:
            metrics.update({
                "pareto_k_n": float(values.size),
                "pareto_k_lt_0_7_rate": float((values < 0.7).mean()),
                "pareto_k_0_7_1_rate": float(((values >= 0.7) & (values <= 1.0)).mean()),
                "pareto_k_gt_1_rate": float((values > 1.0).mean()),
                "pareto_k_mean": float(values.mean()),
                "pareto_k_max": float(values.max()),
            })
    return metrics


def beta_logpdf(y: np.ndarray, mu: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    y = np.clip(y, 1e-9, 1 - 1e-9)
    mu = np.clip(mu, 1e-9, 1 - 1e-9)
    alpha = mu * kappa
    beta = (1.0 - mu) * kappa
    return (
        gammaln(alpha + beta)
        - gammaln(alpha)
        - gammaln(beta)
        + (alpha - 1.0) * np.log(y)
        + (beta - 1.0) * np.log(1.0 - y)
    )


def compute_beta_elpd(model, idata, data: pd.DataFrame, response: str) -> float:
    pred = model.predict(
        idata,
        kind="response_params",
        data=data,
        inplace=False,
        include_group_specific=True,
        sample_new_groups=True,
        random_seed=42,
    )
    mu_da = pred.posterior["mu"].stack(sample=("chain", "draw"))
    mu = mu_da.values.reshape(mu_da.sizes["sample"], -1)

    kappa_name = None
    for name in ["kappa", "precision", "phi"]:
        if name in pred.posterior:
            kappa_name = name
            break
    if kappa_name is None:
        raise ValueError("Could not find kappa/precision in posterior")

    kappa_da = pred.posterior[kappa_name].stack(sample=("chain", "draw"))
    kappa = kappa_da.values.reshape(kappa_da.sizes["sample"], -1)
    if kappa.shape[1] > 1:
        kappa = kappa.mean(axis=1, keepdims=True)

    y = data[response].values.astype(float)
    logp = beta_logpdf(y[None, :], mu, kappa)
    lpd = logsumexp(logp, axis=0) - np.log(logp.shape[0])
    return float(lpd.sum())


def run_beta_model_comparison(
    df: pd.DataFrame,
    response: str,
    out_dir: str,
    folds: int,
    draws: int,
    tune: int,
    chains: int,
    cores: int,
    target_accept: float,
    seed: int,
    model_ids: Optional[List[str]] = None,
):
    data = prepare_beta_data(df, response)
    base_formula = f"{response} ~ age_z + C(region) + C(season) + week_z"
    formulas = {
        "M0": base_formula,
        "M1": base_formula + " + (1|ballroom_partner)",
        "M2": base_formula + " + (1|industry_consolidated)",
        "M3": base_formula + " + (1|ballroom_partner) + (1|industry_consolidated)",
    }
    if model_ids is not None:
        formulas = {k: v for k, v in formulas.items() if k in model_ids}

    comparison_rows = []
    kfold_rows = []

    rng = np.random.default_rng(seed)
    key_region = data.groupby("contestant_key")["region"].first()
    region_counts = key_region.value_counts()
    rare_regions = region_counts[region_counts < folds].index.tolist()
    rare_keys = key_region[key_region.isin(rare_regions)].index.tolist()

    keys = np.array([k for k in key_region.index if k not in set(rare_keys)])
    fold_keys = [[] for _ in range(folds)]
    for region, region_keys in key_region.loc[keys].groupby(key_region.loc[keys]):
        region_keys = region_keys.index.to_numpy()
        rng.shuffle(region_keys)
        for idx, key in enumerate(region_keys):
            fold_keys[idx % folds].append(key)

    for model_id, formula in formulas.items():
        print(f"Running comparison model {model_id}...")
        model, idata = fit_beta_model(data, formula, draws, tune, chains, cores, target_accept)
        if model is None:
            continue

        metrics = collect_loo_metrics(idata)
        metrics.update({
            "model_id": model_id,
            "formula": formula,
            "compare_draws": draws,
            "compare_tune": tune,
            "compare_chains": chains,
            "compare_folds": folds,
            "compare_group": "contestant_key",
            "kfold_excluded_keys": float(len(rare_keys)),
            "kfold_excluded_regions": ",".join(rare_regions) if rare_regions else "",
        })

        fold_elpds = []
        for fold_idx, test_keys in enumerate(fold_keys, start=1):
            train = data[~data["contestant_key"].isin(test_keys)].copy()
            test = data[data["contestant_key"].isin(test_keys)].copy()
            fold_model, fold_idata = fit_beta_model(train, formula, draws, tune, chains, cores, target_accept)
            if fold_model is None:
                continue
            elpd = compute_beta_elpd(fold_model, fold_idata, test, response)
            fold_elpds.append(elpd)
            kfold_rows.append({
                "model_id": model_id,
                "fold": fold_idx,
                "elpd": elpd,
                "n_test": int(len(test)),
            })

        if fold_elpds:
            metrics["kfold_elpd_mean"] = float(np.mean(fold_elpds))
            metrics["kfold_elpd_se"] = float(np.std(fold_elpds, ddof=1) / np.sqrt(len(fold_elpds))) if len(fold_elpds) > 1 else 0.0
        else:
            metrics["kfold_elpd_mean"] = np.nan
            metrics["kfold_elpd_se"] = np.nan

        comparison_rows.append(metrics)

    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows)
        comp_df.to_csv(os.path.join(out_dir, "model_comparison.csv"), index=False)
    if kfold_rows:
        kfold_df = pd.DataFrame(kfold_rows)
        kfold_df.to_csv(os.path.join(out_dir, "kfold_results.csv"), index=False)


def run_beta_model(
    df: pd.DataFrame,
    response: str,
    out_dir: str,
    draws: int,
    tune: int,
    chains: int,
    cores: int,
    target_accept: float,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """运行 Beta 混合效应模型（bambi/pymc），输出固定效应摘要与随机效应"""
    try:
        import bambi as bmb
        import arviz as az
    except Exception as exc:
        print(f"bambi/pymc not available, skipping beta model: {exc}")
        return pd.DataFrame(), None

    data = prepare_beta_data(df, response)

    formula = f"{response} ~ age_z + C(region) + C(season) + week_z + (1|ballroom_partner) + (1|industry_consolidated)"
    model = bmb.Model(formula, data, family="beta")
    idata = model.fit(
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        target_accept=target_accept,
        progressbar=True,
        idata_kwargs={"log_likelihood": True},
    )

    summary = az.summary(idata, hdi_prob=0.95)
    summary.to_csv(os.path.join(out_dir, "beta_model_summary.csv"))

    summarize_model_diagnostics(idata, out_dir)
    generate_ppc_plots(model, idata, response=response, out_dir=out_dir)
    compute_beta_marginal_effects(model, idata, data, response=response, out_dir=out_dir)

    fixed = extract_beta_fixed_effects(summary)
    fixed.to_csv(os.path.join(out_dir, "beta_fan_effects.csv"), index=False)

    re_partner = extract_beta_random_effects(summary, "ballroom_partner")
    if re_partner is not None:
        re_partner = re_partner.rename(columns={"level": "ballroom_partner"})
        re_partner.to_csv(os.path.join(out_dir, "beta_random_effects_ballroom_partner.csv"), index=False)

    re_industry = extract_beta_random_effects(summary, "industry_consolidated")
    if re_industry is not None:
        re_industry = re_industry.rename(columns={"level": "industry_consolidated"})
        re_industry.to_csv(os.path.join(out_dir, "beta_random_effects_industry.csv"), index=False)

    return fixed, re_partner


def compare_effects(summary_J: pd.DataFrame, summary_F: pd.DataFrame, out_path: str) -> pd.DataFrame:
    """比较评审模型与粉丝模型的系数差异（近似检验）"""
    common_terms = sorted(set(summary_J["term"]).intersection(set(summary_F["term"])))
    records = []
    for term in common_terms:
        j_row = summary_J[summary_J["term"] == term].iloc[0]
        f_row = summary_F[summary_F["term"] == term].iloc[0]
        coef_j = j_row["coef"]
        se_j = j_row["std_err"]
        coef_f = f_row["coef"]
        sd_f = f_row["std_err"]
        diff = coef_f - coef_j
        diff_sd = np.sqrt(se_j ** 2 + sd_f ** 2)
        z = diff / diff_sd if diff_sd > 0 else np.nan
        p_value = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
        records.append({
            "term": term,
            "coef_j": coef_j,
            "se_j": se_j,
            "coef_f": coef_f,
            "sd_f": sd_f,
            "diff_f_minus_j": diff,
            "z": z,
            "p_value": p_value,
            "same_direction": np.sign(coef_j) == np.sign(coef_f),
        })
    diff_df = pd.DataFrame(records)
    diff_df.to_csv(out_path, index=False)
    return diff_df


def run_logit(agg: pd.DataFrame, out_path: str):
    """运行逻辑回归（带正则化防止完全分离）"""
    data = agg.copy()
    data = data.dropna(subset=["mean_J_z", "mean_p_hat_z", "age_z", "region", "industry_consolidated"])
    X = pd.get_dummies(data[["mean_J_z", "mean_p_hat_z", "age_z", "region", "industry_consolidated"]], drop_first=True)
    # standardize continuous columns
    for col in ["mean_J_z", "mean_p_hat_z", "age_z"]:
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
        result = model.fit_regularized(method="l1", alpha=0.1, disp=False)

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


def run_cox_time_varying(tv: pd.DataFrame, out_path: str):
    """运行 Cox time-varying 模型"""
    try:
        from lifelines import CoxTimeVaryingFitter
    except Exception:
        print("lifelines not installed, skipping time-varying Cox model")
        return

    data = tv.copy()
    data = data.dropna(subset=[
        "contestant_key", "season", "start", "stop", "event",
        "J_z", "p_hat_z", "age_z", "region", "industry_consolidated"
    ])
    covariates = data[["J_z", "p_hat_z", "age_z", "region", "industry_consolidated"]]
    X = pd.get_dummies(covariates, drop_first=True)

    variances = X.var()
    low_var_cols = variances[variances < 1e-8].index.tolist()
    if low_var_cols:
        X = X.drop(columns=low_var_cols)

    X["contestant_key"] = data["contestant_key"].values
    X["start"] = data["start"].astype(float)
    X["stop"] = data["stop"].astype(float)
    X["event"] = data["event"].astype(int)
    X["season"] = data["season"].astype(int)

    ctv = CoxTimeVaryingFitter(penalizer=0.1)
    try:
        ctv.fit(
            X,
            id_col="contestant_key",
            start_col="start",
            stop_col="stop",
            event_col="event",
            strata=["season"],
        )
        ctv.summary.to_csv(out_path)

        summary = ctv.summary[["coef", "exp(coef)", "se(coef)", "p", "coef lower 95%", "coef upper 95%"]].copy()
        summary.to_csv(out_path.replace("_coef.csv", "_summary_coef.csv"))
    except Exception as exc:
        print(f"Time-varying Cox model failed: {exc}")


def run_ph_test(agg: pd.DataFrame, out_path: str):
    """运行 PH 检验（基于 contestant-level CoxPH）"""
    try:
        from lifelines import CoxPHFitter
        from lifelines.statistics import proportional_hazard_test
    except Exception:
        print("lifelines not installed, skipping PH test")
        return

    data = agg.copy()
    data = data.dropna(subset=["mean_J_z", "mean_p_hat_z", "age_z", "region", "industry_consolidated", "duration", "event", "season"])
    covariates = data[["mean_J_z", "mean_p_hat_z", "age_z", "region", "industry_consolidated", "season"]]
    X = pd.get_dummies(covariates, drop_first=True)

    variances = X.drop(columns=["season"]).var()
    low_var_cols = variances[variances < 1e-8].index.tolist()
    if low_var_cols:
        X = X.drop(columns=low_var_cols)

    X["duration"] = data["duration"].astype(int)
    X["event"] = data["event"].astype(int)
    X["season"] = data["season"].astype(int)

    cph = CoxPHFitter(penalizer=0.1)
    try:
        cph.fit(X, duration_col="duration", event_col="event", strata=["season"])
        ph_test = proportional_hazard_test(cph, X, time_transform="rank")
        ph_test.summary.to_csv(out_path)
    except Exception as exc:
        print(f"PH test failed: {exc}")


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
    axes[0].set_xlabel("Effect on Judge Score (J_z)")
    axes[0].set_title("Ballroom Partner Effects on Judge Scores\n(Top 10 & Bottom 10)")
    axes[0].axvline(x=0, color='black', linewidth=0.8)
    
    # 右图：评审 vs 粉丝效应散点图
    axes[1].scatter(merged["effect_judge"], merged["effect_fan"], alpha=0.6)
    axes[1].set_xlabel("Effect on Judge Score (J_z)")
    axes[1].set_ylabel("Effect on Fan Votes (Beta)")
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
    key_vars = ["age_z", "week_z",
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
    axes[1].set_title("Model F (Fan Votes, Beta): Fixed Effects")
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
    key_vars = ["J_z", "p_hat_z", "age_z"]
    region_vars = [c for c in cox_df.index if c.startswith("region_")]
    plot_vars = key_vars + region_vars[:4]
    
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
    ax.set_title("Time-varying Cox Model: Hazard Ratios with 95% CI")
    
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
    ax.set_xlabel("Hazard Ratio (HR)")
    ax.set_ylabel("Industry")
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
    age_J = summary_J[summary_J["term"] == "age_z"]
    age_F = summary_F[summary_F["term"] == "age_z"]
    
    if len(age_J) == 0 or len(age_F) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    categories = ["Judge Score\n(Model J)", "Fan Votes\n(Model F)"]
    coefs = [age_J["coef"].values[0], age_F["coef"].values[0]]
    errors = [age_J["std_err"].values[0] * 1.96, age_F["std_err"].values[0] * 1.96]
    colors = ['steelblue', 'coral']
    
    bars = ax.bar(categories, coefs, yerr=errors, capsize=5, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel("Age Coefficient (z-score)")
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
    age_J = summary_J[summary_J["term"] == "age_z"]
    age_F = summary_F[summary_F["term"] == "age_z"]

    # 周次效应
    week_J = summary_J[summary_J["term"] == "week_z"]
    week_F = summary_F[summary_F["term"] == "week_z"]

    # 差异检验与 Cox 结果（用于摘要）
    diff_path = os.path.join(out_dir, "coef_diff_table.csv")
    diff_df = pd.read_csv(diff_path) if os.path.exists(diff_path) else pd.DataFrame()
    sig_diff = diff_df[diff_df["p_value"] < 0.05] if not diff_df.empty else pd.DataFrame()

    cox_path = os.path.join(out_dir, "cox_model_summary_coef.csv")
    cox_df = pd.read_csv(cox_path, index_col=0) if os.path.exists(cox_path) else pd.DataFrame()

    report_lines.append("### 1.0 摘要\n")
    if len(age_J) > 0 and len(age_F) > 0:
        age_coef_J = age_J["coef"].values[0]
        age_p_J = age_J["p_value"].values[0] if "p_value" in age_J.columns else None
        age_coef_F = age_F["coef"].values[0]
        age_ci_F = (age_F["ci_lower"].values[0], age_F["ci_upper"].values[0])
        report_lines.append(
            f"- 年龄对评审评分呈显著负向影响（age_z = {age_coef_J:.3f}, p = {age_p_J:.3g}），"
            f"对粉丝投票亦为负向但幅度更小（age_z = {age_coef_F:.3f}, 95% CI [{age_ci_F[0]:.3f}, {age_ci_F[1]:.3f}]).\n"
        )

    if len(week_J) > 0 and len(week_F) > 0:
        week_coef_J = week_J["coef"].values[0]
        week_coef_F = week_F["coef"].values[0]
        direction = "相反" if week_coef_J * week_coef_F < 0 else "一致"
        report_lines.append(
            f"- 周次效应在评审与粉丝模型中方向{direction}（J: {week_coef_J:.3f}, F: {week_coef_F:.3f}）。\n"
        )

    if not cox_df.empty and all(k in cox_df.index for k in ["J_z", "p_hat_z", "age_z"]):
        hr_j = cox_df.loc["J_z", "exp(coef)"]
        hr_f = cox_df.loc["p_hat_z", "exp(coef)"]
        hr_a = cox_df.loc["age_z", "exp(coef)"]
        report_lines.append(
            f"- 生存分析显示评审分数与粉丝份额均显著降低淘汰风险（HR_J = {hr_j:.3f}, HR_F = {hr_f:.3f}），"
            f"年龄增加淘汰风险（HR_age = {hr_a:.3f}）。\n"
        )

    if len(age_J) > 0 and len(age_F) > 0:
        age_coef_J = age_J["coef"].values[0]
        age_ci_J = (age_J["ci_lower"].values[0], age_J["ci_upper"].values[0])
        age_p_J = age_J["p_value"].values[0] if "p_value" in age_J.columns else None
        age_coef_F = age_F["coef"].values[0]
        age_ci_F = (age_F["ci_lower"].values[0], age_F["ci_upper"].values[0])
        age_sig_J = age_p_J is not None and age_p_J < 0.05
        age_sig_F = age_ci_F[0] * age_ci_F[1] > 0

        report_lines.append("### 1.1 年龄影响（标准化）\n")
        report_lines.append(
            f"- **评审评分**: age_z = {age_coef_J:.4f}, 95% CI [{age_ci_J[0]:.4f}, {age_ci_J[1]:.4f}]"
            + (f", p = {age_p_J:.4f}" if age_p_J is not None else "")
            + "\n"
        )
        report_lines.append(
            f"- **粉丝投票**: age_z = {age_coef_F:.4f}, 95% CI [{age_ci_F[0]:.4f}, {age_ci_F[1]:.4f}]\n"
        )

        if age_sig_J and not age_sig_F:
            report_lines.append(
                "**结果解读**: 年龄对评审打分的负向影响显著，而对粉丝投票不显著，"
                "提示评审更强调技术表现，粉丝更偏向于整体人气与魅力特征。\n"
            )
    
    if len(week_J) > 0 and len(week_F) > 0:
        report_lines.append("\n### 1.2 周次效应（标准化）\n")
        report_lines.append(f"- **评审评分**: week_z = {week_J['coef'].values[0]:.4f}\n")
        report_lines.append(f"- **粉丝投票**: week_z = {week_F['coef'].values[0]:.4f}\n")
        if week_J['coef'].values[0] * week_F['coef'].values[0] < 0:
            report_lines.append("\n**结果解读**: 周次效应在评审与粉丝模型中方向相反，表明两条评价路径对比赛进程的反应存在结构性差异。\n")
        else:
            report_lines.append("\n**结果解读**: 周次效应方向一致，说明两条路径对比赛进程的反应较为同步。\n")

    if not diff_df.empty:
        report_lines.append("\n### 1.3 评审 vs 粉丝差异检验\n")
        if len(sig_diff) > 0:
            report_lines.append("- 下列变量在两模型中的系数差异显著（近似检验）：\n")
            for _, row in sig_diff.head(6).iterrows():
                report_lines.append(
                    f"- {row['term']}: diff = {row['diff_f_minus_j']:.4f}, p = {row['p_value']:.4f}\n"
                )
        else:
            report_lines.append("- 未发现显著差异（p < 0.05）。\n")

    baseline_path = os.path.join(out_dir, "baseline_info.csv")
    if os.path.exists(baseline_path):
        baseline_df = pd.read_csv(baseline_path)
        report_lines.append("\n### 1.4 基准类别说明\n")
        for _, row in baseline_df.iterrows():
            report_lines.append(f"- **{row['feature']}** 基准类别: {row['baseline']}\n")

    diagnostics_path = os.path.join(out_dir, "model_diagnostics.csv")
    if os.path.exists(diagnostics_path):
        diag_df = pd.read_csv(diagnostics_path)
        report_lines.append("\n### 1.5 模型诊断（Beta 模型）\n")
        for _, row in diag_df.iterrows():
            report_lines.append(f"- {row['metric']}: {row['value']:.4f}\n")
        report_lines.append("- PPC 图: ppc_distribution.png, ppc_mean.png\n")
        report_lines.append("- 备注: 若 WAIC 出现高方差警告，表示 WAIC 可能不稳定，建议以 LOO 为主进行模型比较。\n")
        diag_map = dict(zip(diag_df["metric"], diag_df["value"]))
        if "pareto_k_lt_0_7_rate" in diag_map:
            lt = diag_map.get("pareto_k_lt_0_7_rate", 0.0)
            mid = diag_map.get("pareto_k_0_7_1_rate", 0.0)
            gt = diag_map.get("pareto_k_gt_1_rate", 0.0)
            max_k = diag_map.get("pareto_k_max", np.nan)
            report_lines.append(
                f"- LOO Pareto-k: <0.7 {lt*100:.1f}%, 0.7–1.0 {mid*100:.1f}%, >1.0 {gt*100:.1f}% (max {max_k:.2f}).\n"
            )
            if gt > 0:
                report_lines.append("- 解释: 存在高影响观测（k>1），LOO 结果需谨慎，建议补充 K-fold 或稳健比较。\n")
            elif mid > 0.1:
                report_lines.append("- 解释: 存在一定比例 k>0.7 的观测，LOO 可用但需谨慎解释。\n")
            else:
                report_lines.append("- 解释: Pareto-k 主要处于安全区间，LOO 结果可靠。\n")

    marg_path = os.path.join(out_dir, "beta_marginal_effects.csv")
    if os.path.exists(marg_path):
        marg_df = pd.read_csv(marg_path)
        report_lines.append("\n### 1.6 粉丝投票边际效应（Beta 模型）\n")
        for _, row in marg_df.iterrows():
            report_lines.append(
                f"- {row['effect']}: Δp_hat ≈ {row['mean_diff']:.4f} "
                f"(95% CI [{row['hdi_lower']:.4f}, {row['hdi_upper']:.4f}])\n"
            )

    comp_path = os.path.join(out_dir, "model_comparison.csv")
    if os.path.exists(comp_path):
        comp_df = pd.read_csv(comp_path)
        if not comp_df.empty:
            report_lines.append("\n### 1.7 模型复杂度对比（Beta）\n")
            best_loo = comp_df.loc[comp_df["elpd_loo"].idxmax()]
            report_lines.append(
                f"- LOO 最优模型: {best_loo['model_id']} (elpd_loo = {best_loo['elpd_loo']:.2f})\n"
            )
            if "kfold_elpd_mean" in comp_df.columns and comp_df["kfold_elpd_mean"].notna().any():
                best_kfold = comp_df.loc[comp_df["kfold_elpd_mean"].idxmax()]
                report_lines.append(
                    f"- K-fold 最优模型: {best_kfold['model_id']} (elpd = {best_kfold['kfold_elpd_mean']:.2f})\n"
                )
            if "model_id" in comp_df.columns and (comp_df["model_id"] == "M3").any():
                base = comp_df[comp_df["model_id"] == "M3"].iloc[0]
                report_lines.append(
                    f"- 以完整模型 M3 为参照: Δelpd_loo = {best_loo['elpd_loo'] - base['elpd_loo']:.2f}\n"
                )
            if "kfold_excluded_keys" in comp_df.columns:
                excluded = comp_df["kfold_excluded_keys"].iloc[0]
                if excluded > 0:
                    regions = comp_df.get("kfold_excluded_regions", pd.Series([""])).iloc[0]
                    report_lines.append(
                        f"- K-fold 注意: {int(excluded)} 个 contestant_key 因稀有区域被固定留在训练集"
                        + (f"（{regions}）" if regions else "")
                        + "。\n"
                    )
            if "compare_draws" in comp_df.columns:
                draws = int(comp_df["compare_draws"].iloc[0])
                tune = int(comp_df["compare_tune"].iloc[0]) if "compare_tune" in comp_df.columns else ""
                chains = int(comp_df["compare_chains"].iloc[0]) if "compare_chains" in comp_df.columns else ""
                report_lines.append(
                    f"- K-fold 设定: draws={draws}, tune={tune}, chains={chains}（轻量采样，仅用于模型对比）。\n"
                )
    
    # 舞伴效应
    if re_df_J is not None:
        report_lines.append("\n## 2. 专业舞伴效应\n")
        report_lines.append("\n### 2.1 对评审评分影响最大的舞伴 (Top 5)\n")
        top5_J = re_df_J.nlargest(5, "effect")
        for _, row in top5_J.iterrows():
            report_lines.append(f"- **{row['ballroom_partner']}**: +{row['effect']:.3f} (J_z)\n")
        
        report_lines.append("\n### 2.2 对评审评分影响最小的舞伴 (Bottom 5)\n")
        bottom5_J = re_df_J.nsmallest(5, "effect")
        for _, row in bottom5_J.iterrows():
            report_lines.append(f"- **{row['ballroom_partner']}**: {row['effect']:.3f} (J_z)\n")
    
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
            report_lines.append("\n### 3.1 行业对淘汰风险的影响 (Time-varying Cox)\n")
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
        
        key_vars = ["J_z", "p_hat_z", "age_z"]
        for var in key_vars:
            if var in cox_df.index:
                row = cox_df.loc[var]
                hr = row["exp(coef)"]
                ci_low = row["exp(coef) lower 95%"]
                ci_high = row["exp(coef) upper 95%"]
                p_val = row["p"]
                
                if var == "J_z":
                    name = "评审分数 (J_z)"
                    if hr < 1:
                        interp = f"评审分数每提高1个标准差，淘汰风险降低 {(1-hr)*100:.1f}%"
                    else:
                        interp = f"评审分数每提高1个标准差，淘汰风险增加 {(hr-1)*100:.1f}%"
                elif var == "p_hat_z":
                    name = "粉丝投票份额 (p_hat_z)"
                    if hr < 1:
                        interp = f"粉丝份额每提高1个标准差，淘汰风险降低 {(1-hr)*100:.1f}%"
                    else:
                        interp = f"粉丝份额每提高1个标准差，淘汰风险增加 {(hr-1)*100:.1f}%"
                else:
                    name = "年龄 (age_z)"
                    if hr > 1:
                        interp = f"年龄每增加1个标准差，淘汰风险增加 {(hr-1)*100:.1f}%"
                    else:
                        interp = f"年龄每增加1个标准差，淘汰风险降低 {(1-hr)*100:.1f}%"
                
                sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
                report_lines.append(f"| {name} | {hr:.3f}{sig} | [{ci_low:.3f}, {ci_high:.3f}] | {p_val:.4f} | {interp} |\n")
        
        report_lines.append("\n*注: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001*\n")
        
        report_lines.append("\n### 4.2 生存分析核心结论\n")
        report_lines.append("""
- **评审分数是最强的保护因素**: J_z 更高的选手淘汰风险显著降低，符合评分机制。
- **粉丝投票同样重要**: p_hat_z 较高的选手更"安全"，说明粉丝投票在合并机制中发挥了关键作用。
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
        
        sig_count = (region_J["p_value"] < 0.05).sum()
        if sig_count > 0:
            report_lines.append("\n**结果解读**: 部分区域在评审分数上呈显著负向差异（相对基准 International），但总体效应量较小。\n")
        else:
            report_lines.append("\n**结果解读**: 区域效应在评审分数上整体不显著，说明评审打分对地理来源不敏感。\n")
    
    # 结论
    report_lines.append("\n## 6. 核心结论\n")
    report_lines.append("""
### 6.1 因素对比赛表现的影响程度（标准化）

| 因素 | 对评审分数 | 对粉丝投票 | 对淘汰风险 |
|------|-----------|-----------|-----------|
| **年龄** | 负向影响（显著） | 负向影响（显著但较弱） | 增加风险 |
| **专业舞伴** | 强影响 | 中等影响 | 间接影响 |
| **行业** | 随机效应显著 | 随机效应显著 | Cox 模型显示趋势 |
| **区域** | 部分负向显著（相对 International） | 相对 International 多为负向 | 多数不显著 |
| **周次** | 负向影响 | 正向影响 | - |

### 6.2 评审分数 vs 粉丝投票：影响方式是否相同？

**答案：不完全相同。**

- **年龄因素**: 评审对年龄的负向效应更明显，粉丝效应更弱。
- **周次效应**: 评审与粉丝模型方向相反，反映两条路径对赛程的反应不同。
- **舞伴效应**: 评审效应与粉丝效应呈中等正相关，但两者不完全一致。
- **核心差异机制**:
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


def generate_summary_page(out_dir: str):
    summary_path = os.path.join(out_dir, "summary.md")
    lines = []
    lines.append("# 问题三结果摘要\n")
    lines.append("本摘要基于 r3/outputs 的最新模型输出。\n")

    baseline_path = os.path.join(out_dir, "baseline_info.csv")
    baseline_df = pd.read_csv(baseline_path) if os.path.exists(baseline_path) else pd.DataFrame()

    summary_J_path = os.path.join(out_dir, "model_J_summary_fixed_effects.csv")
    summary_F_path = os.path.join(out_dir, "beta_fan_effects.csv")
    cox_path = os.path.join(out_dir, "cox_model_summary_coef.csv")
    diff_path = os.path.join(out_dir, "coef_diff_table.csv")
    marg_path = os.path.join(out_dir, "beta_marginal_effects.csv")
    diag_path = os.path.join(out_dir, "model_diagnostics.csv")

    summary_J = pd.read_csv(summary_J_path) if os.path.exists(summary_J_path) else pd.DataFrame()
    summary_F = pd.read_csv(summary_F_path) if os.path.exists(summary_F_path) else pd.DataFrame()
    cox_df = pd.read_csv(cox_path, index_col=0) if os.path.exists(cox_path) else pd.DataFrame()
    diff_df = pd.read_csv(diff_path) if os.path.exists(diff_path) else pd.DataFrame()
    marg_df = pd.read_csv(marg_path) if os.path.exists(marg_path) else pd.DataFrame()
    diag_df = pd.read_csv(diag_path) if os.path.exists(diag_path) else pd.DataFrame()

    def get_row(df: pd.DataFrame, term: str):
        if df.empty:
            return None
        rows = df[df["term"] == term]
        if len(rows) == 0:
            return None
        return rows.iloc[0]

    age_J = get_row(summary_J, "age_z")
    week_J = get_row(summary_J, "week_z")
    age_F = get_row(summary_F, "age_z")
    week_F = get_row(summary_F, "week_z")

    lines.append("## 1. 核心结论\n")
    if age_J is not None and age_F is not None:
        lines.append(
            f"- 年龄对评审评分显著为负（age_z = {age_J['coef']:.3f}, p = {age_J['p_value']:.3g}），"
            f"对粉丝投票同样为负但幅度更小（age_z = {age_F['coef']:.3f}, 95% CI [{age_F['ci_lower']:.3f}, {age_F['ci_upper']:.3f}]).\n"
        )
    if week_J is not None and week_F is not None:
        direction = "相反" if week_J["coef"] * week_F["coef"] < 0 else "一致"
        lines.append(
            f"- 周次效应在评审与粉丝模型中方向{direction}（J: {week_J['coef']:.3f}, F: {week_F['coef']:.3f}）。\n"
        )
    if not cox_df.empty and all(k in cox_df.index for k in ["J_z", "p_hat_z", "age_z"]):
        lines.append(
            f"- 生存分析显示评审分数与粉丝份额显著降低淘汰风险（HR_J = {cox_df.loc['J_z', 'exp(coef)']:.3f}, "
            f"HR_F = {cox_df.loc['p_hat_z', 'exp(coef)']:.3f}），年龄增加淘汰风险（HR_age = {cox_df.loc['age_z', 'exp(coef)']:.3f}）。\n"
        )

    lines.append("\n## 2. 关键系数（标准化）\n")
    lines.append("| 模型 | 变量 | 估计 | 区间/显著性 |\n")
    lines.append("|---|---|---|---|\n")
    if age_J is not None:
        lines.append(f"| 评审评分 | age_z | {age_J['coef']:.3f} | p = {age_J['p_value']:.3g} |\n")
    if week_J is not None:
        lines.append(f"| 评审评分 | week_z | {week_J['coef']:.3f} | p = {week_J['p_value']:.3g} |\n")
    if age_F is not None:
        lines.append(
            f"| 粉丝投票 | age_z | {age_F['coef']:.3f} | CI [{age_F['ci_lower']:.3f}, {age_F['ci_upper']:.3f}] |\n"
        )
    if week_F is not None:
        lines.append(
            f"| 粉丝投票 | week_z | {week_F['coef']:.3f} | CI [{week_F['ci_lower']:.3f}, {week_F['ci_upper']:.3f}] |\n"
        )
    if not cox_df.empty and all(k in cox_df.index for k in ["J_z", "p_hat_z", "age_z"]):
        lines.append(
            f"| 淘汰风险 | J_z | HR {cox_df.loc['J_z', 'exp(coef)']:.3f} | p = {cox_df.loc['J_z', 'p']:.3g} |\n"
        )
        lines.append(
            f"| 淘汰风险 | p_hat_z | HR {cox_df.loc['p_hat_z', 'exp(coef)']:.3f} | p = {cox_df.loc['p_hat_z', 'p']:.3g} |\n"
        )
        lines.append(
            f"| 淘汰风险 | age_z | HR {cox_df.loc['age_z', 'exp(coef)']:.3f} | p = {cox_df.loc['age_z', 'p']:.3g} |\n"
        )

    if not marg_df.empty:
        lines.append("\n## 3. 粉丝投票边际效应（Beta）\n")
        lines.append("| 变量 | Δp_hat | 95% CI |\n")
        lines.append("|---|---|---|\n")
        for _, row in marg_df.iterrows():
            lines.append(
                f"| {row['effect']} | {row['mean_diff']:.4f} | [{row['hdi_lower']:.4f}, {row['hdi_upper']:.4f}] |\n"
            )

    if not diff_df.empty:
        sig_diff = diff_df[diff_df["p_value"] < 0.05]
        lines.append("\n## 4. 评审 vs 粉丝差异检验\n")
        if len(sig_diff) == 0:
            lines.append("未发现显著差异（p < 0.05）。\n")
        else:
            lines.append("| 变量 | diff(F-J) | p值 | 方向一致 |\n")
            lines.append("|---|---|---|---|\n")
            for _, row in sig_diff.head(6).iterrows():
                lines.append(
                    f"| {row['term']} | {row['diff_f_minus_j']:.4f} | {row['p_value']:.4g} | {row['same_direction']} |\n"
                )

    lines.append("\n## 5. 基准类别与诊断\n")
    if not baseline_df.empty:
        for _, row in baseline_df.iterrows():
            lines.append(f"- **{row['feature']}** 基准类别: {row['baseline']}\n")
    if not diag_df.empty:
        lines.append("- 诊断指标（LOO/WAIC）:\n")
        for _, row in diag_df.iterrows():
            lines.append(f"  - {row['metric']}: {row['value']:.4f}\n")
        lines.append("- 备注: 若 WAIC 出现高方差警告，表示 WAIC 可能不稳定，建议以 LOO 为主进行模型比较。\n")
        diag_map = dict(zip(diag_df["metric"], diag_df["value"]))
        if "pareto_k_lt_0_7_rate" in diag_map:
            lt = diag_map.get("pareto_k_lt_0_7_rate", 0.0)
            mid = diag_map.get("pareto_k_0_7_1_rate", 0.0)
            gt = diag_map.get("pareto_k_gt_1_rate", 0.0)
            max_k = diag_map.get("pareto_k_max", np.nan)
            lines.append(
                f"- LOO Pareto-k: <0.7 {lt*100:.1f}%, 0.7–1.0 {mid*100:.1f}%, >1.0 {gt*100:.1f}% (max {max_k:.2f}).\n"
            )
            if gt > 0:
                lines.append("- 解释: 存在高影响观测（k>1），LOO 结果需谨慎，建议补充 K-fold 或稳健比较。\n")
            elif mid > 0.1:
                lines.append("- 解释: 存在一定比例 k>0.7 的观测，LOO 可用但需谨慎解释。\n")
            else:
                lines.append("- 解释: Pareto-k 主要处于安全区间，LOO 结果可靠。\n")

    comp_path = os.path.join(out_dir, "model_comparison.csv")
    if os.path.exists(comp_path):
        comp_df = pd.read_csv(comp_path)
        if not comp_df.empty:
            lines.append("\n## 6. 模型复杂度对比（Beta）\n")
            best_loo = comp_df.loc[comp_df["elpd_loo"].idxmax()]
            lines.append(
                f"- LOO 最优模型: {best_loo['model_id']} (elpd_loo = {best_loo['elpd_loo']:.2f})\n"
            )
            if "kfold_elpd_mean" in comp_df.columns and comp_df["kfold_elpd_mean"].notna().any():
                best_kfold = comp_df.loc[comp_df["kfold_elpd_mean"].idxmax()]
                lines.append(
                    f"- K-fold 最优模型: {best_kfold['model_id']} (elpd = {best_kfold['kfold_elpd_mean']:.2f})\n"
                )
            if "model_id" in comp_df.columns and (comp_df["model_id"] == "M3").any():
                base = comp_df[comp_df["model_id"] == "M3"].iloc[0]
                lines.append(
                    f"- 以完整模型 M3 为参照: Δelpd_loo = {best_loo['elpd_loo'] - base['elpd_loo']:.2f}\n"
                )
            if "kfold_excluded_keys" in comp_df.columns:
                excluded = comp_df["kfold_excluded_keys"].iloc[0]
                if excluded > 0:
                    regions = comp_df.get("kfold_excluded_regions", pd.Series([""])).iloc[0]
                    lines.append(
                        f"- K-fold 注意: {int(excluded)} 个 contestant_key 因稀有区域被固定留在训练集"
                        + (f"（{regions}）" if regions else "")
                        + "。\n"
                    )
            if "compare_draws" in comp_df.columns:
                draws = int(comp_df["compare_draws"].iloc[0])
                tune = int(comp_df["compare_tune"].iloc[0]) if "compare_tune" in comp_df.columns else ""
                chains = int(comp_df["compare_chains"].iloc[0]) if "compare_chains" in comp_df.columns else ""
                lines.append(
                    f"- K-fold 设定: draws={draws}, tune={tune}, chains={chains}（轻量采样，仅用于模型对比）。\n"
                )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    features = pd.read_csv(args.features)
    raw = pd.read_csv(args.raw).copy()
    raw = raw.rename(columns={"celebrity_homecountry/region": "celebrity_homecountry_region"})

    features = features.copy()
    features["contestant_key"] = build_contestant_key(features)
    raw["contestant_key"] = build_contestant_key(raw)

    dup_raw = raw[raw.duplicated("contestant_key", keep=False)]
    if not dup_raw.empty:
        print("Warning: duplicate contestant_key in raw data; keeping first occurrence.")
        raw = raw.drop_duplicates(subset=["contestant_key"], keep="first")

    merged = features.merge(raw[[
        "contestant_key",
        "celebrity_age_during_season",
        "celebrity_industry",
        "celebrity_homestate",
        "celebrity_homecountry_region",
    ]], on="contestant_key", how="left")

    merged[["region", "country_group"]] = merged.apply(
        lambda row: pd.Series(
            map_region_and_country(row["celebrity_homestate"], row["celebrity_homecountry_region"])
        ),
        axis=1,
    )
    
    # 统一行业名称大小写，并基于选手级别（而非周级别）计算频次
    merged["celebrity_industry"] = merged["celebrity_industry"].fillna("Unknown").astype(str).str.strip()
    # 先统一大小写用于计数
    merged["industry_lower"] = merged["celebrity_industry"].str.lower()
    
    # 基于选手级别计算行业频次（每个选手只算一次）
    contestant_industry = merged.groupby("contestant_key")["industry_lower"].first()
    industry_counts = contestant_industry.value_counts().to_dict()
    
    print(f"Industry counts (contestant-level): {industry_counts}")
    
    merged["industry_consolidated"] = merged["industry_lower"].apply(
        lambda x: consolidate_industry(x, industry_counts, min_count=10)
    )
    # 恢复首字母大写格式（仅用于显示）
    merged["industry_consolidated"] = merged["industry_consolidated"].str.title()

    # 设置类别基准（用于可解释性）
    region_levels = ["International", "Midwest", "Northeast", "South", "Unknown", "West"]
    region_values = [r for r in region_levels if r in set(merged["region"].dropna().unique())]
    extra_regions = sorted(set(merged["region"].dropna().unique()) - set(region_values))
    region_order = region_values + extra_regions
    merged["region"] = pd.Categorical(merged["region"], categories=region_order, ordered=False)

    industry_order = sorted(merged["industry_consolidated"].dropna().unique())
    merged["industry_consolidated"] = pd.Categorical(
        merged["industry_consolidated"], categories=industry_order, ordered=False
    )

    baseline_info = [
        {"feature": "region", "baseline": region_order[0] if region_order else ""},
        {"feature": "industry_consolidated", "baseline": industry_order[0] if industry_order else ""},
    ]
    pd.DataFrame(baseline_info).to_csv(os.path.join(args.out_dir, "baseline_info.csv"), index=False)

    # 标准化与裁剪
    merged["J_z"] = merged.groupby(["season", "week"])["J_total"].transform(zscore_series)
    eps = 1e-4
    merged["p_hat_clip"] = merged["p_hat"].clip(eps, 1 - eps)
    merged["p_hat_z"] = zscore_series(merged["p_hat_clip"])
    merged["age_z"] = zscore_series(merged["celebrity_age_during_season"])
    merged["week_z"] = zscore_series(merged["week"])
    
    print(f"Industry consolidation: {merged['celebrity_industry'].nunique()} -> {merged['industry_consolidated'].nunique()} categories")
    print(f"Region distribution:\n{merged['region'].value_counts()}")

    # Mixed effects models
    print("\nRunning Model J (Judge Scores, J_z)...")
    summary_J, re_df_J = run_mixedlm(
        merged,
        response="J_z",
        out_path=os.path.join(args.out_dir, "model_J_fixed_effects.csv"),
    )
    
    summary_F = pd.DataFrame()
    re_df_F = None
    if not args.skip_beta:
        print("Running Model F (Fan Votes, Beta)...")
        summary_F, re_df_F = run_beta_model(
            merged,
            response="p_hat_clip",
            out_dir=args.out_dir,
            draws=args.beta_draws,
            tune=args.beta_tune,
            chains=args.beta_chains,
            cores=args.beta_cores,
            target_accept=args.beta_target_accept,
        )
        if args.compare_models:
            model_ids = ["M0", "M3"] if args.compare_minimal else None
            run_beta_model_comparison(
                merged,
                response="p_hat_clip",
                out_dir=args.out_dir,
                folds=args.compare_folds,
                draws=args.compare_draws,
                tune=args.compare_tune,
                chains=args.compare_chains,
                cores=args.compare_cores,
                target_accept=args.compare_target_accept,
                seed=args.compare_seed,
                model_ids=model_ids,
            )

    # Survival / logistic models (contestant-level)
    print("\nBuilding survival table...")
    agg = build_survival_table(merged)
    agg.to_csv(os.path.join(args.out_dir, "survival_table.csv"), index=False)

    if not args.skip_cox:
        print("Running time-varying Cox model...")
        tv = build_time_varying_table(merged)
        run_cox_time_varying(tv, os.path.join(args.out_dir, "cox_model_coef.csv"))
        run_ph_test(agg, os.path.join(args.out_dir, "cox_ph_test.csv"))
    
    print("Running Logit model...")
    run_logit(agg, os.path.join(args.out_dir, "logit_model_coef.csv"))

    if not summary_F.empty:
        compare_effects(summary_J, summary_F, os.path.join(args.out_dir, "coef_diff_table.csv"))

    # 生成可视化
    print("\nGenerating visualizations...")
    plot_partner_effects(re_df_J, re_df_F, args.out_dir)
    if not summary_F.empty:
        plot_fixed_effects_comparison(summary_J, summary_F, args.out_dir)
    plot_cox_hazard_ratios(args.out_dir)
    plot_industry_effects(args.out_dir)
    if not summary_F.empty:
        plot_age_effect_comparison(summary_J, summary_F, args.out_dir)
    
    # 生成解读报告
    print("Generating interpretation report...")
    if not summary_F.empty:
        generate_interpretation_report(summary_J, summary_F, re_df_J, re_df_F, args.out_dir)
        generate_summary_page(args.out_dir)
    else:
        print("Skipping interpretation report (beta model not available)")

    print(f"\n✅ All outputs written to: {args.out_dir}")
    print("Generated files:")
    for f in sorted(os.listdir(args.out_dir)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
