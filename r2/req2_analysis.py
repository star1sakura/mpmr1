import argparse
import os
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_votes = os.path.normpath(os.path.join(base_dir, "..", "r1", "outputs", "req1_vote_estimates.csv"))
    default_out = os.path.join(base_dir, "outputs")
    ap = argparse.ArgumentParser(description="Requirement 2 analysis: compare rank vs percent methods.")
    ap.add_argument(
        "--votes",
        default=default_votes,
        help="Path to req1_vote_estimates.csv",
    )
    ap.add_argument("--out-dir", default=default_out, help="Output directory")
    return ap.parse_args()


def weighted_avg(values, weights) -> float:
    mask = values.notna() & weights.notna()
    denom = weights[mask].sum()
    if denom <= 0:
        return float("nan")
    return float((values[mask] * weights[mask]).sum() / denom)


def to_float(val: Any) -> float:
    try:
        out = float(val)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def quantile_bins(values: pd.Series, labels: List[str]) -> pd.Series:
    vals = np.asarray(values.dropna(), dtype=float)
    if vals.size < len(labels):
        return pd.Series([pd.NA] * len(values), index=values.index)
    edges = np.quantile(vals, np.linspace(0, 1, len(labels) + 1))
    if np.unique(edges).size < len(edges):
        return pd.Series([pd.NA] * len(values), index=values.index)
    bins = [-np.inf] + edges[1:-1].tolist() + [np.inf]
    binned = pd.cut(values, bins=bins, labels=labels, include_lowest=True)
    return pd.Series(binned, index=values.index)


def focus_by_uncertainty(values: pd.Series, top_share: float) -> pd.Series:
    vals = np.asarray(values.dropna(), dtype=float)
    if vals.size == 0:
        return pd.Series([pd.NA] * len(values), index=values.index)
    high_cut = np.quantile(vals, 1 - top_share)
    low_cut = np.quantile(vals, top_share)
    def label(v: float) -> str:
        if not np.isfinite(v):
            return "mid"
        if v >= high_cut:
            return "high"
        if v <= low_cut:
            return "low"
        return "mid"
    labels = [label(to_float(v)) for v in values.tolist()]
    return pd.Series(labels, index=values.index)


def select_typical_seasons(metrics_df: pd.DataFrame, count: int = 5) -> List[int]:
    selected: List[int] = []
    season_idx = metrics_df.columns.get_loc("season")
    save_idx = metrics_df.columns.get_loc("judge_save")

    def add_from(df_sorted: pd.DataFrame, limit: int):
        for row in df_sorted.itertuples(index=False, name=None):
            season = int(cast(float, row[season_idx]))
            if season in selected:
                continue
            selected.append(season)
            if len(selected) >= limit:
                break

    high_flip = metrics_df.sort_values("flip_rate", ascending=False)
    high_gap = metrics_df.sort_values("metric_gap", ascending=False)
    low_flip = metrics_df.sort_values("flip_rate", ascending=True)

    add_from(high_flip, min(count, 2))
    add_from(high_gap, min(count, 4))
    add_from(low_flip, count)

    save_count = metrics_df[metrics_df["season"].isin(selected) & metrics_df["judge_save"]].shape[0]
    min_save = 2
    if save_count < min_save:
        save_candidates = metrics_df[metrics_df["judge_save"]].copy()
        metric_gap_vals = np.asarray(save_candidates["metric_gap"], dtype=float)
        flip_vals = np.asarray(save_candidates["flip_rate"], dtype=float)
        save_score = np.nan_to_num(metric_gap_vals, nan=0.0) * 10 + np.nan_to_num(flip_vals, nan=0.0)
        save_candidates = save_candidates.assign(save_score=save_score)
        save_candidates = save_candidates.sort_values("save_score", ascending=False)  # type: ignore[call-overload]
        for row in save_candidates.itertuples(index=False, name=None):
            season = int(cast(float, row[season_idx]))
            if season in selected:
                continue
            selected.append(season)
            save_count += 1
            if save_count >= min_save:
                break

        if len(selected) > count:
            for season in selected[::-1]:
                if len(selected) <= count:
                    break
                row_save = metrics_df.loc[metrics_df["season"] == season, "judge_save"]
                if not row_save.empty and not bool(row_save.iloc[0]):
                    selected.remove(season)
            while len(selected) > count:
                selected.pop()

    return selected[:count]


def has_judge_save(season: int) -> bool:
    return season >= 28


def rank_avg_desc(values: np.ndarray) -> np.ndarray:
    # Average ranks for ties, 1 is best (descending order)
    order = np.argsort(-values, kind="mergesort")
    ranks = np.zeros(len(values), dtype=float)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def pick_bottom_k(scores: np.ndarray, k: int, tie_break: np.ndarray) -> List[int]:
    # deterministic bottom-k: sort by score asc, then tie_break asc
    order = np.lexsort((tie_break, scores))
    return order[:k].tolist()


def pick_top_k(scores: np.ndarray, k: int, tie_break: np.ndarray) -> List[int]:
    # deterministic top-k: sort by score desc, then tie_break asc
    order = np.lexsort((tie_break, -scores))
    return order[:k].tolist()


def apply_cbp(J: np.ndarray, p_hat: np.ndarray, tie_break: np.ndarray) -> Tuple[List[int], List[int]]:
    P_J = J / np.sum(J)
    S = P_J + p_hat
    elim = pick_bottom_k(S, 1, tie_break)
    bottom2 = pick_bottom_k(S, 2, tie_break)
    return elim, bottom2


def apply_cbr(J: np.ndarray, p_hat: np.ndarray, tie_break: np.ndarray) -> Tuple[List[int], List[int]]:
    rJ = rank_avg_desc(J)
    rV = rank_avg_desc(p_hat)
    ranksum = rJ + rV
    # ranksum越大越差，应淘汰最大者/倒数两名
    elim = pick_top_k(ranksum, 1, tie_break)
    bottom2 = pick_top_k(ranksum, 2, tie_break)
    return elim, bottom2


def classify_flip_driver(rJ: np.ndarray, rV: np.ndarray, cbr_idx: int, cbp_idx: int) -> str:
    jd_cbr = rJ[cbr_idx] - rV[cbr_idx]
    jd_cbp = rJ[cbp_idx] - rV[cbp_idx]
    if jd_cbr > 0 and jd_cbp > 0:
        return "judge"
    if jd_cbr < 0 and jd_cbp < 0:
        return "fan"
    return "mixed"


def rank_worst_positions(scores: np.ndarray, tie_break: np.ndarray, higher_worse: bool) -> np.ndarray:
    if higher_worse:
        order = np.lexsort((tie_break, -scores))
    else:
        order = np.lexsort((tie_break, scores))
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def classify_typical_season(flip_rate: float, metric_gap: float, judge_save: bool) -> Tuple[str, str]:
    if flip_rate >= 0.5 and metric_gap >= 0.3:
        return "高翻转/高差异", "制度差异显著，规则选择对结果影响大。"
    if flip_rate <= 0.1 and metric_gap <= 0.1:
        return "低翻转/趋同", "两种规则给出接近的淘汰结论。"
    if metric_gap >= 0.3:
        return "高差异", "一致性指标差距明显。"
    if flip_rate >= 0.4:
        return "高翻转", "预测集合分歧较频繁。"
    return "中等差异", "规则差异存在但不极端。"


def summarize_contestant_exit(df: pd.DataFrame, method_col: str) -> pd.DataFrame:
    # method_col contains list-like string of eliminated index; derive earliest week per contestant
    rows = []
    for group_key, g in df.groupby(["season", "celebrity_name"]):
        season_key, name = cast(Tuple[int, str], group_key)
        g = g.sort_values("week")
        elim_week = g.loc[g[method_col] == 1, "week"]
        week_val = int(cast(float, elim_week.iloc[0])) if not elim_week.empty else np.nan
        rows.append({"season": int(cast(float, season_key)), "celebrity_name": name, f"{method_col}_week": week_val})
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.votes)
    required = {"season", "week", "contestant_id", "celebrity_name", "J_total", "p_hat", "eliminated"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    if "uncertainty_width" not in df.columns:
        df["uncertainty_width"] = np.nan

    group_sum = df.groupby(["season", "week"])["J_total"].transform("sum")
    df["judge_share"] = np.where(group_sum > 0, df["J_total"] / group_sum, np.nan)
    df["fan_judge_gap"] = df["p_hat"] - df["judge_share"]

    records = []
    flip_rows = []
    strength_rows = []

    # Per-season metrics
    metrics_rows = []
    for season_key, sdf in df.groupby("season"):
        season = int(cast(float, season_key))
        save = has_judge_save(season)
        cbr_hit = cbp_hit = 0
        denom = 0
        cbr_elim_match = cbp_elim_match = 0
        flip_count = 0

        for week_key, wdf in sdf.groupby("week"):
            wdf = wdf.sort_values("contestant_id")
            J = wdf["J_total"].to_numpy(dtype=float)
            p_hat = wdf["p_hat"].to_numpy(dtype=float)
            tie_break = wdf["contestant_id"].to_numpy(dtype=int)
            rJ = rank_avg_desc(J)
            rV = rank_avg_desc(p_hat)
            ranksum = rJ + rV
            P_J = J / np.sum(J)
            cbp_score = P_J + p_hat
            cbr_rank = rank_worst_positions(ranksum, tie_break, higher_worse=True)
            cbp_rank = rank_worst_positions(cbp_score, tie_break, higher_worse=False)

            cbr_elim, cbr_bottom2 = apply_cbr(J, p_hat, tie_break)
            cbp_elim, cbp_bottom2 = apply_cbp(J, p_hat, tie_break)

            elim_idx = wdf.index[wdf["eliminated"].to_numpy(dtype=int) == 1].tolist()
            has_elim = len(elim_idx) > 0

            actual_names = wdf.loc[elim_idx, "celebrity_name"].tolist()
            cbr_elim_name = wdf.iloc[cbr_elim]["celebrity_name"].tolist()
            cbp_elim_name = wdf.iloc[cbp_elim]["celebrity_name"].tolist()
            cbr_bottom2_names = wdf.iloc[cbr_bottom2]["celebrity_name"].tolist()
            cbp_bottom2_names = wdf.iloc[cbp_bottom2]["celebrity_name"].tolist()

            if has_elim:
                denom += 1
                if save:
                    if set(actual_names).issubset(set(cbr_bottom2_names)):
                        cbr_hit += 1
                    if set(actual_names).issubset(set(cbp_bottom2_names)):
                        cbp_hit += 1
                else:
                    if set(actual_names) == set(cbr_elim_name):
                        cbr_elim_match += 1
                    if set(actual_names) == set(cbp_elim_name):
                        cbp_elim_match += 1

                if (set(cbr_elim_name) != set(cbp_elim_name)) if not save else (set(cbr_bottom2_names) != set(cbp_bottom2_names)):
                    flip_count += 1
                    driver = classify_flip_driver(rJ, rV, cbr_elim[0], cbp_elim[0])
                    cbr_rank_in_cbp = int(cbp_rank[cbr_elim[0]])
                    cbp_rank_in_cbr = int(cbr_rank[cbp_elim[0]])
                    flip_strength = (cbr_rank_in_cbp - 1) + (cbp_rank_in_cbr - 1)
                    flip_rows.append({
                        "season": season,
                        "week": int(cast(float, week_key)),
                        "actual_eliminated": "; ".join(actual_names),
                        "cbr_elim": "; ".join(cbr_elim_name),
                        "cbp_elim": "; ".join(cbp_elim_name),
                        "cbr_bottom2": "; ".join(cbr_bottom2_names),
                        "cbp_bottom2": "; ".join(cbp_bottom2_names),
                        "judge_save": save,
                        "flip_driver": driver,
                    })
                    strength_rows.append({
                        "season": season,
                        "week": int(cast(float, week_key)),
                        "judge_save": save,
                        "cbr_elim": "; ".join(cbr_elim_name),
                        "cbp_elim": "; ".join(cbp_elim_name),
                        "cbr_elim_rank_in_cbp": cbr_rank_in_cbp,
                        "cbp_elim_rank_in_cbr": cbp_rank_in_cbr,
                        "flip_strength": flip_strength,
                    })

            records.append({
                "season": season,
                "week": int(cast(float, week_key)),
                "judge_save": save,
                "actual_eliminated": "; ".join(actual_names),
                "cbr_elim": "; ".join(cbr_elim_name),
                "cbp_elim": "; ".join(cbp_elim_name),
                "cbr_bottom2": "; ".join(cbr_bottom2_names),
                "cbp_bottom2": "; ".join(cbp_bottom2_names),
                "cbr_hit": int(set(actual_names).issubset(set(cbr_bottom2_names)) if save else set(actual_names) == set(cbr_elim_name)),
                "cbp_hit": int(set(actual_names).issubset(set(cbp_bottom2_names)) if save else set(actual_names) == set(cbp_elim_name)),
            })

        metrics_rows.append({
            "season": season,
            "judge_save": save,
            "weeks_with_elim": denom,
            "CBR_EMR": cbr_elim_match / denom if denom and not save else np.nan,
            "CBP_EMR": cbp_elim_match / denom if denom and not save else np.nan,
            "CBR_B2CR": cbr_hit / denom if denom and save else np.nan,
            "CBP_B2CR": cbp_hit / denom if denom and save else np.nan,
            "flip_rate": flip_count / denom if denom else np.nan,
        })

    comp_df = pd.DataFrame(records)
    comp_df.to_csv(os.path.join(args.out_dir, "method_comparison.csv"), index=False)

    flip_df = pd.DataFrame(flip_rows)
    flip_df.to_csv(os.path.join(args.out_dir, "flip_events.csv"), index=False)

    strength_df = pd.DataFrame(strength_rows)
    strength_df.to_csv(os.path.join(args.out_dir, "flip_strength.csv"), index=False)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("season")
    metrics_df.to_csv(os.path.join(args.out_dir, "season_method_metrics.csv"), index=False)

    weekly_df = (
        df.groupby(["season", "week"], as_index=False)
        .agg(
            N=("contestant_id", "count"),
            uncertainty_mean=("uncertainty_width", "mean"),
        )
    )
    weekly_df = cast(pd.DataFrame, weekly_df)
    weekly_df["judge_save"] = [has_judge_save(int(x)) for x in weekly_df["season"].tolist()]
    flip_mark = flip_df[["season", "week", "flip_driver"]].drop_duplicates()
    weekly_df = weekly_df.merge(flip_mark, on=["season", "week"], how="left")
    weekly_df["flip"] = weekly_df["flip_driver"].notna().astype(int)
    weekly_df = weekly_df.merge(
        strength_df[["season", "week", "flip_strength"]], on=["season", "week"], how="left"
    )
    weekly_df["size_bin"] = quantile_bins(pd.Series(weekly_df["N"]), ["small", "mid", "large"])
    weekly_df["uncertainty_bin"] = quantile_bins(pd.Series(weekly_df["uncertainty_mean"]), ["low", "mid", "high"])
    weekly_df["uncertainty_group"] = focus_by_uncertainty(pd.Series(weekly_df["uncertainty_mean"]), 0.25)
    weekly_df["stage"] = weekly_df.groupby("season")["week"].transform(
        lambda s: quantile_bins(pd.Series(s), ["early", "mid", "late"])
    )
    weekly_df["stage"] = weekly_df["stage"].astype("object").fillna("mid")
    weekly_df.to_csv(os.path.join(args.out_dir, "flip_uncertainty.csv"), index=False)

    bin_order = ["low", "mid", "high"]
    unc_summary = pd.DataFrame()
    if bool(weekly_df["uncertainty_bin"].notna().any()):
        unc_group = weekly_df.dropna(subset=["uncertainty_bin"])
        rate_df = (
            unc_group.groupby("uncertainty_bin")
            .agg(flip_rate=("flip", "mean"), weeks=("flip", "size"))
            .reindex(bin_order)
        )
        strength_df_bin = (
            unc_group.dropna(subset=["flip_strength"])
            .groupby("uncertainty_bin")["flip_strength"]
            .agg(["mean", "median"])
            .reindex(bin_order)
        )
        unc_summary = rate_df.join(strength_df_bin, how="left")
        unc_summary.to_csv(os.path.join(args.out_dir, "flip_uncertainty_summary.csv"))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(unc_summary.index.tolist(), unc_summary["flip_rate"].fillna(0))
        ax.set_ylabel("Flip rate")
        ax.set_title("Flip Rate by Uncertainty Bin")
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "flip_uncertainty_rate.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(unc_summary.index.tolist(), unc_summary["mean"].fillna(0))
        ax.set_ylabel("Mean flip strength")
        ax.set_title("Flip Strength by Uncertainty Bin")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "flip_uncertainty_strength.png"), dpi=150)
        plt.close(fig)

    comp_df = comp_df.copy()
    comp_df["has_elim"] = comp_df["actual_eliminated"].astype(str).str.len() > 0
    comp_df = comp_df[comp_df["has_elim"]]
    comp_df = comp_df.merge(
        weekly_df[["season", "week", "size_bin", "stage", "flip", "uncertainty_group", "uncertainty_mean", "flip_strength"]],
        on=["season", "week"],
        how="left",
    )

    focus_labels = ["low", "high"]
    focus_df = comp_df[comp_df["uncertainty_group"].isin(focus_labels)].copy()
    focus_rows: List[Dict[str, Any]] = []
    for label in focus_labels:
        g = cast(pd.DataFrame, focus_df[focus_df["uncertainty_group"] == label])
        if len(g.index) == 0:
            continue
        no_save = g[g["judge_save"] == False]
        save = g[g["judge_save"] == True]
        strength_vals = np.asarray(g["flip_strength"], dtype=float)
        strength_vals = strength_vals[np.isfinite(strength_vals)]
        focus_rows.append({
            "uncertainty_group": label,
            "weeks_with_elim": int(len(g)),
            "CBR_EMR": to_float(no_save["cbr_hit"].mean()) if len(no_save) else float("nan"),
            "CBP_EMR": to_float(no_save["cbp_hit"].mean()) if len(no_save) else float("nan"),
            "CBR_B2CR": to_float(save["cbr_hit"].mean()) if len(save) else float("nan"),
            "CBP_B2CR": to_float(save["cbp_hit"].mean()) if len(save) else float("nan"),
            "flip_rate": to_float(g["flip"].mean()) if len(g) else float("nan"),
            "flip_strength_mean": to_float(g["flip_strength"].mean()),
            "flip_strength_median": float(np.nanmedian(strength_vals)) if strength_vals.size else float("nan"),
            "uncertainty_mean": to_float(g["uncertainty_mean"].mean()),
        })
    focus_out = cast(pd.DataFrame, pd.DataFrame(focus_rows))
    focus_out.to_csv(os.path.join(args.out_dir, "uncertainty_focus.csv"), index=False)
    high_unc_missing = False
    if len(focus_out.index):
        high_row = cast(pd.DataFrame, focus_out[focus_out["uncertainty_group"] == "high"])
        if len(high_row.index):
            cbr_b2 = to_float(high_row["CBR_B2CR"].iloc[0])
            cbp_b2 = to_float(high_row["CBP_B2CR"].iloc[0])
            if np.isnan(cbr_b2) or np.isnan(cbp_b2):
                high_unc_missing = True

    if len(focus_out.index):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(focus_out["uncertainty_group"], focus_out["flip_rate"].fillna(0))
        axes[0].set_ylim(0, 1)
        axes[0].set_title("Flip Rate (Low vs High Uncertainty)")
        axes[0].grid(True, axis="y", alpha=0.3)

        axes[1].bar(focus_out["uncertainty_group"], focus_out["flip_strength_mean"].fillna(0))
        axes[1].set_title("Flip Strength (Mean)")
        axes[1].grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "uncertainty_focus.png"), dpi=150)
        plt.close(fig)

    def build_strata_metrics(df_in: pd.DataFrame, strata_col: str, strata_type: str, order: List[str]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for label in order:
            g = df_in[df_in[strata_col] == label]
            if g.empty:
                continue
            no_save = g[g["judge_save"] == False]
            save = g[g["judge_save"] == True]
            rows.append({
                "strata_type": strata_type,
                "strata_label": label,
                "weeks_with_elim": int(len(g)),
                "CBR_EMR": to_float(no_save["cbr_hit"].mean()) if len(no_save) else float("nan"),
                "CBP_EMR": to_float(no_save["cbp_hit"].mean()) if len(no_save) else float("nan"),
                "CBR_B2CR": to_float(save["cbr_hit"].mean()) if len(save) else float("nan"),
                "CBP_B2CR": to_float(save["cbp_hit"].mean()) if len(save) else float("nan"),
                "flip_rate": to_float(g["flip"].mean()) if len(g) else float("nan"),
            })
        return rows

    size_order = ["small", "mid", "large"]
    stage_order = ["early", "mid", "late"]
    strata_rows: List[Dict[str, Any]] = []
    strata_rows.extend(build_strata_metrics(comp_df, "size_bin", "size", size_order))
    strata_rows.extend(build_strata_metrics(comp_df, "stage", "stage", stage_order))
    strata_df = pd.DataFrame(strata_rows)
    strata_df.to_csv(os.path.join(args.out_dir, "strata_metrics.csv"), index=False)

    def plot_strata(df_in: pd.DataFrame, order: List[str], title: str, out_path: str):
        if df_in.empty:
            return
        data = cast(pd.DataFrame, df_in.set_index("strata_label").reindex(order))
        x = np.arange(len(order))
        width = 0.35
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(x - width / 2, np.asarray(data["CBR_EMR"], dtype=float), width, label="CBR_EMR")
        axes[0].bar(x + width / 2, np.asarray(data["CBP_EMR"], dtype=float), width, label="CBP_EMR")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(order)
        axes[0].set_ylim(0, 1)
        axes[0].set_title("EMR (no judge-save)")
        axes[0].grid(True, axis="y", alpha=0.3)
        axes[0].legend()

        axes[1].bar(x - width / 2, np.asarray(data["CBR_B2CR"], dtype=float), width, label="CBR_B2CR")
        axes[1].bar(x + width / 2, np.asarray(data["CBP_B2CR"], dtype=float), width, label="CBP_B2CR")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(order)
        axes[1].set_ylim(0, 1)
        axes[1].set_title("B2CR (judge-save)")
        axes[1].grid(True, axis="y", alpha=0.3)
        axes[1].legend()

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    plot_strata(
        cast(pd.DataFrame, strata_df[strata_df["strata_type"] == "size"]),
        size_order,
        "Stratified Metrics by Participant Size",
        os.path.join(args.out_dir, "strata_by_size.png"),
    )
    plot_strata(
        cast(pd.DataFrame, strata_df[strata_df["strata_type"] == "stage"]),
        stage_order,
        "Stratified Metrics by Season Stage",
        os.path.join(args.out_dir, "strata_by_stage.png"),
    )

    # Controversial contestants profile under both methods
    controversial = {
        (2, "Jerry Rice"),
        (4, "Billy Ray Cyrus"),
        (11, "Bristol Palin"),
        (27, "Bobby Bones"),
    }

    prof_rows = []
    for group_key, wdf in df.groupby(["season", "week"]):
        season_key, week_key = cast(Tuple[int, int], group_key)
        wdf = wdf.sort_values("contestant_id")
        wdf_cols = {name: idx for idx, name in enumerate(wdf.columns)}
        J = wdf["J_total"].to_numpy(dtype=float)
        p_hat = wdf["p_hat"].to_numpy(dtype=float)
        tie_break = wdf["contestant_id"].to_numpy(dtype=int)
        cbr_elim, cbr_bottom2 = apply_cbr(J, p_hat, tie_break)
        cbp_elim, cbp_bottom2 = apply_cbp(J, p_hat, tie_break)

        for row in wdf.itertuples(index=False, name=None):
            row_season = row[wdf_cols["season"]]
            row_name = row[wdf_cols["celebrity_name"]]
            key = (int(cast(float, row_season)), row_name)
            if key not in controversial:
                continue
            prof_rows.append({
                "season": int(cast(float, season_key)),
                "week": int(cast(float, week_key)),
                "celebrity_name": row_name,
                "J_total": float(row[wdf_cols["J_total"]]),
                "p_hat": float(row[wdf_cols["p_hat"]]),
                "judge_share": float(row[wdf_cols["judge_share"]]) if pd.notna(row[wdf_cols["judge_share"]]) else np.nan,
                "fan_judge_gap": float(row[wdf_cols["fan_judge_gap"]]) if pd.notna(row[wdf_cols["fan_judge_gap"]]) else np.nan,
                "actual_eliminated": int(row[wdf_cols["eliminated"]]),
                "cbr_elim": int(row_name in wdf.iloc[cbr_elim]["celebrity_name"].tolist()),
                "cbp_elim": int(row_name in wdf.iloc[cbp_elim]["celebrity_name"].tolist()),
                "cbr_bottom2": int(row_name in wdf.iloc[cbr_bottom2]["celebrity_name"].tolist()),
                "cbp_bottom2": int(row_name in wdf.iloc[cbp_bottom2]["celebrity_name"].tolist()),
            })

    controversy_df = pd.DataFrame(prof_rows)
    controversy_df.to_csv(os.path.join(args.out_dir, "controversy_analysis.csv"), index=False)

    # === Controversy gap trends ===
    controversy_list = [
        (2, "Jerry Rice"),
        (4, "Billy Ray Cyrus"),
        (11, "Bristol Palin"),
        (27, "Bobby Bones"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()
    for ax, (season, name) in zip(axes, controversy_list):
        mask = (controversy_df["season"] == season) & (controversy_df["celebrity_name"] == name)
        sub = controversy_df.loc[mask].sort_values("week")  # type: ignore[call-overload]
        ax.plot(sub["week"], sub["fan_judge_gap"], marker="o", label="fan_judge_gap")
        ax.axhline(0.0, color="gray", linewidth=1, alpha=0.4)
        ax.set_title(f"S{season} {name}")
        ax.set_xlabel("Week")
        ax.set_ylabel("Fan-Judge Gap")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "controversy_gap_trends.png"), dpi=150)
    plt.close(fig)

    # === Flip strength plot ===
    if not strength_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        groups = [
            strength_df[strength_df["judge_save"] == False]["flip_strength"],
            strength_df[strength_df["judge_save"] == True]["flip_strength"],
        ]
        ax.boxplot(groups, showfliers=False)
        ax.set_xticklabels(["no judge-save", "judge-save"])
        ax.set_ylabel("Flip strength (rank distance)")
        ax.set_title("Flip Strength by Season Type")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "flip_strength.png"), dpi=150)
        plt.close(fig)

    # === Plots ===
    # Flip rate by season
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(metrics_df["season"], metrics_df["flip_rate"], marker="o")
    ax.set_xlabel("Season")
    ax.set_ylabel("Flip rate")
    ax.set_title("Flip Rate: CBR vs CBP")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "flip_rate_by_season.png"), dpi=150)
    plt.close(fig)

    # EMR (no judge-save seasons)
    emr_df = metrics_df[metrics_df["judge_save"] == False]
    if not emr_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(emr_df["season"], emr_df["CBR_EMR"], marker="o", label="CBR_EMR")
        ax.plot(emr_df["season"], emr_df["CBP_EMR"], marker="s", label="CBP_EMR")
        ax.set_xlabel("Season")
        ax.set_ylabel("EMR")
        ax.set_title("Elimination Match Rate (No Judge-Save Seasons)")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "emr_by_season.png"), dpi=150)
        plt.close(fig)

    # B2CR (judge-save seasons)
    b2_df = metrics_df[metrics_df["judge_save"] == True]
    if not b2_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(b2_df["season"], b2_df["CBR_B2CR"], marker="o", label="CBR_B2CR")
        ax.plot(b2_df["season"], b2_df["CBP_B2CR"], marker="s", label="CBP_B2CR")
        ax.set_xlabel("Season")
        ax.set_ylabel("B2CR")
        ax.set_title("Bottom-2 Coverage (Judge-Save Seasons)")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "b2cr_by_season.png"), dpi=150)
        plt.close(fig)

    # === Summary ===
    metrics_df = metrics_df.copy()
    metrics_df["metric_gap"] = np.where(
        metrics_df["judge_save"],
        (metrics_df["CBR_B2CR"] - metrics_df["CBP_B2CR"]).abs(),
        (metrics_df["CBR_EMR"] - metrics_df["CBP_EMR"]).abs(),
    )

    non_save = metrics_df[metrics_df["judge_save"] == False]
    save_df = metrics_df[metrics_df["judge_save"] == True]

    emr_cbr = weighted_avg(non_save["CBR_EMR"], non_save["weeks_with_elim"])
    emr_cbp = weighted_avg(non_save["CBP_EMR"], non_save["weeks_with_elim"])
    b2_cbr = weighted_avg(save_df["CBR_B2CR"], save_df["weeks_with_elim"])
    b2_cbp = weighted_avg(save_df["CBP_B2CR"], save_df["weeks_with_elim"])
    emr_gap = emr_cbp - emr_cbr if np.isfinite(emr_cbp) and np.isfinite(emr_cbr) else float("nan")
    b2_gap = b2_cbp - b2_cbr if np.isfinite(b2_cbp) and np.isfinite(b2_cbr) else float("nan")

    flip_all = weighted_avg(metrics_df["flip_rate"], metrics_df["weeks_with_elim"])
    flip_non = weighted_avg(non_save["flip_rate"], non_save["weeks_with_elim"])
    flip_save = weighted_avg(save_df["flip_rate"], save_df["weeks_with_elim"])

    flip_summary = {}
    flip_total = len(flip_df)
    if flip_total > 0 and "flip_driver" in flip_df.columns:
        for driver, count in flip_df["flip_driver"].value_counts().items():
            flip_summary[driver] = {
                "count": int(count),
                "share": float(count) / float(flip_total),
            }

    flip_driver_rows = []
    for driver in ["judge", "fan", "mixed"]:
        row = flip_summary.get(driver, {"count": 0, "share": 0.0})
        flip_driver_rows.append({
            "section": "flip_driver_all",
            "metric": driver,
            "value": row["share"],
            "count": row["count"],
        })

    flip_by_save = []
    if flip_total > 0 and "flip_driver" in flip_df.columns:
        for save_flag in [False, True]:
            subset = flip_df[flip_df["judge_save"] == save_flag]
            total = len(subset)
            for driver in ["judge", "fan", "mixed"]:
                count = int((subset["flip_driver"] == driver).sum()) if total else 0
                share = float(count) / float(total) if total else 0.0
                flip_by_save.append({
                    "section": "flip_driver_judge_save" if save_flag else "flip_driver_no_judge_save",
                    "metric": driver,
                    "value": share,
                    "count": count,
                })

    typical_seasons = select_typical_seasons(metrics_df, count=5)
    order_map = {int(season): idx for idx, season in enumerate(typical_seasons)}
    typical_df = metrics_df[metrics_df["season"].isin(typical_seasons)].copy()
    typical_df = typical_df.assign(order=[order_map.get(int(x), 0) for x in typical_df["season"].tolist()])
    typical_df = typical_df.sort_values("order")
    typical_cols = {name: idx for idx, name in enumerate(typical_df.columns)}

    cont_summary = (
        controversy_df.groupby(["season", "celebrity_name"])
        .agg(
            mean_fan_gap=("fan_judge_gap", "mean"),
            mean_p_hat=("p_hat", "mean"),
            mean_judge=("J_total", "mean"),
            weeks=("week", "count"),
        )
        .reset_index()
        .sort_values(["season", "celebrity_name"])
    )

    tie_threshold = 0.05
    decision_rows = []
    season_idx = metrics_df.columns.get_loc("season")
    save_idx = metrics_df.columns.get_loc("judge_save")
    emr_cbr_idx = metrics_df.columns.get_loc("CBR_EMR")
    emr_cbp_idx = metrics_df.columns.get_loc("CBP_EMR")
    b2_cbr_idx = metrics_df.columns.get_loc("CBR_B2CR")
    b2_cbp_idx = metrics_df.columns.get_loc("CBP_B2CR")
    flip_idx = metrics_df.columns.get_loc("flip_rate")
    for row in metrics_df.itertuples(index=False, name=None):
        season = int(cast(float, row[season_idx]))
        save = bool(row[save_idx])
        if save:
            cbr_val = to_float(row[b2_cbr_idx])
            cbp_val = to_float(row[b2_cbp_idx])
            metric_name = "B2CR"
        else:
            cbr_val = to_float(row[emr_cbr_idx])
            cbp_val = to_float(row[emr_cbp_idx])
            metric_name = "EMR"

        if np.isnan(cbr_val) or np.isnan(cbp_val):
            preferred = "na"
            gap = float("nan")
            diff = float("nan")
        else:
            diff = cbr_val - cbp_val
            gap = abs(diff)
            if gap < tie_threshold:
                preferred = "tie"
            elif diff > 0:
                preferred = "CBR"
            else:
                preferred = "CBP"

        decision_rows.append({
            "season": season,
            "judge_save": save,
            "metric": metric_name,
            "CBR": cbr_val,
            "CBP": cbp_val,
            "metric_gap": gap,
            "metric_diff": diff,
            "preferred_method": preferred,
            "flip_rate": to_float(row[flip_idx]),
        })

    decision_df = pd.DataFrame(decision_rows)
    decision_df.to_csv(os.path.join(args.out_dir, "decision_guidance.csv"), index=False)

    tie_thresholds = [0.03, 0.05, 0.07]
    sensitivity_rows: List[Dict[str, Any]] = []
    for threshold in tie_thresholds:
        for segment, seg_df in [
            ("all", decision_df),
            ("no_judge_save", decision_df[decision_df["judge_save"] == False]),
            ("judge_save", decision_df[decision_df["judge_save"] == True]),
        ]:
            if seg_df.empty:
                continue
            preferred = []
            for row in seg_df.itertuples(index=False, name=None):
                gap = to_float(row[seg_df.columns.get_loc("metric_gap")])
                if np.isnan(gap):
                    preferred.append("na")
                elif gap < threshold:
                    preferred.append("tie")
                else:
                    preferred.append(row[seg_df.columns.get_loc("preferred_method")])
            counts = pd.Series(preferred).value_counts()
            cbr_count = counts.get("CBR", 0)
            cbp_count = counts.get("CBP", 0)
            tie_count = counts.get("tie", 0)
            na_count = counts.get("na", 0)
            sensitivity_rows.append({
                "threshold": threshold,
                "segment": segment,
                "CBR_count": int(cbr_count) if cbr_count is not None else 0,
                "CBP_count": int(cbp_count) if cbp_count is not None else 0,
                "tie_count": int(tie_count) if tie_count is not None else 0,
                "na_count": int(na_count) if na_count is not None else 0,
                "total": int(len(preferred)),
            })

    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_df.to_csv(os.path.join(args.out_dir, "tie_sensitivity.csv"), index=False)

    sens_all = cast(pd.DataFrame, sensitivity_df[sensitivity_df["segment"] == "all"])
    tie_stable = False
    if not sens_all.empty:
        uniq = sens_all[["CBR_count", "CBP_count", "tie_count"]].drop_duplicates()
        tie_stable = len(uniq.index) == 1
    if not sens_all.empty:
        x = np.arange(len(tie_thresholds))
        width = 0.25
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x - width, np.asarray(sens_all["CBR_count"], dtype=float), width, label="CBR")
        ax.bar(x, np.asarray(sens_all["CBP_count"], dtype=float), width, label="CBP")
        ax.bar(x + width, np.asarray(sens_all["tie_count"], dtype=float), width, label="tie")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t:.2f}" for t in tie_thresholds])
        ax.set_ylabel("Season count")
        ax.set_title("Tie Threshold Sensitivity (All Seasons)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "tie_sensitivity.png"), dpi=150)
        plt.close(fig)

    strength_vals = np.asarray(strength_df["flip_strength"], dtype=float) if not strength_df.empty else np.asarray([])
    strength_mean = float(np.nanmean(strength_vals)) if strength_vals.size else float("nan")
    strength_median = float(np.nanmedian(strength_vals)) if strength_vals.size else float("nan")
    strength_by_save = {}
    if not strength_df.empty:
        for save_flag in [False, True]:
            subset = strength_df[strength_df["judge_save"] == save_flag]
            subset_vals = np.asarray(subset["flip_strength"], dtype=float) if not subset.empty else np.asarray([])
            strength_by_save[save_flag] = {
                "mean": float(np.nanmean(subset_vals)) if subset_vals.size else float("nan"),
                "median": float(np.nanmedian(subset_vals)) if subset_vals.size else float("nan"),
            }

    summary_table_rows = []
    summary_table_rows.extend([
        {"section": "overall", "metric": "CBR_EMR", "value": emr_cbr},
        {"section": "overall", "metric": "CBP_EMR", "value": emr_cbp},
        {"section": "overall", "metric": "CBR_B2CR", "value": b2_cbr},
        {"section": "overall", "metric": "CBP_B2CR", "value": b2_cbp},
        {"section": "overall", "metric": "flip_all", "value": flip_all},
        {"section": "overall", "metric": "flip_no_judge_save", "value": flip_non},
        {"section": "overall", "metric": "flip_judge_save", "value": flip_save},
    ])
    summary_table_rows.extend(flip_driver_rows)
    summary_table_rows.extend(flip_by_save)
    summary_table_rows.extend([
        {"section": "flip_strength", "metric": "mean", "value": strength_mean},
        {"section": "flip_strength", "metric": "median", "value": strength_median},
        {"section": "flip_strength_no_judge_save", "metric": "mean", "value": strength_by_save.get(False, {}).get("mean", float("nan"))},
        {"section": "flip_strength_no_judge_save", "metric": "median", "value": strength_by_save.get(False, {}).get("median", float("nan"))},
        {"section": "flip_strength_judge_save", "metric": "mean", "value": strength_by_save.get(True, {}).get("mean", float("nan"))},
        {"section": "flip_strength_judge_save", "metric": "median", "value": strength_by_save.get(True, {}).get("median", float("nan"))},
    ])

    for row in focus_out.itertuples(index=False, name=None):
        col_idx = {name: idx for idx, name in enumerate(focus_out.columns)}
        summary_table_rows.append({
            "section": "uncertainty_focus",
            "metric": row[col_idx["uncertainty_group"]],
            "CBR_EMR": to_float(row[col_idx["CBR_EMR"]]),
            "CBP_EMR": to_float(row[col_idx["CBP_EMR"]]),
            "CBR_B2CR": to_float(row[col_idx["CBR_B2CR"]]),
            "CBP_B2CR": to_float(row[col_idx["CBP_B2CR"]]),
            "flip_rate": to_float(row[col_idx["flip_rate"]]),
            "flip_strength_mean": to_float(row[col_idx["flip_strength_mean"]]),
            "flip_strength_median": to_float(row[col_idx["flip_strength_median"]]),
            "uncertainty_mean": to_float(row[col_idx["uncertainty_mean"]]),
            "weeks_with_elim": int(row[col_idx["weeks_with_elim"]]) if pd.notna(row[col_idx["weeks_with_elim"]]) else float("nan"),
        })

    if not sensitivity_df.empty:
        for row in sensitivity_df.itertuples(index=False, name=None):
            sens_cols = {name: idx for idx, name in enumerate(sensitivity_df.columns)}
            if row[sens_cols["segment"]] != "all":
                continue
            summary_table_rows.append({
                "section": "tie_sensitivity",
                "threshold": to_float(row[sens_cols["threshold"]]),
                "CBR_count": int(row[sens_cols["CBR_count"]]),
                "CBP_count": int(row[sens_cols["CBP_count"]]),
                "tie_count": int(row[sens_cols["tie_count"]]),
                "total": int(row[sens_cols["total"]]),
            })

    for row in typical_df.itertuples(index=False, name=None):
        season = int(cast(float, row[typical_cols["season"]]))
        save = bool(row[typical_cols["judge_save"]])
        metric_gap = to_float(row[typical_cols["metric_gap"]])
        flip_rate = to_float(row[typical_cols["flip_rate"]])
        if save:
            cbr_val = to_float(row[typical_cols["CBR_B2CR"]])
            cbp_val = to_float(row[typical_cols["CBP_B2CR"]])
            metric_name = "B2CR"
        else:
            cbr_val = to_float(row[typical_cols["CBR_EMR"]])
            cbp_val = to_float(row[typical_cols["CBP_EMR"]])
            metric_name = "EMR"
        label, note = classify_typical_season(flip_rate, metric_gap, save)
        summary_table_rows.append({
            "section": "typical_season",
            "season": season,
            "judge_save": save,
            "metric": metric_name,
            "CBR": cbr_val,
            "CBP": cbp_val,
            "flip_rate": flip_rate,
            "metric_gap": metric_gap,
            "label": label,
            "note": note,
        })

    summary_table = pd.DataFrame(summary_table_rows)
    summary_table.to_csv(os.path.join(args.out_dir, "summary_table.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["EMR (no judge-save)", "B2CR (judge-save)"]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, [emr_cbr, b2_cbr], width, label="CBR")
    ax.bar(x + width / 2, [emr_cbp, b2_cbp], width, label="CBP")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Overall CBR vs CBP")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "summary_metrics.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["no judge-save", "judge-save"]
    strengths = [
        strength_by_save.get(False, {}).get("mean", float("nan")),
        strength_by_save.get(True, {}).get("mean", float("nan")),
    ]
    ax.bar(labels, strengths)
    ax.set_ylabel("Mean flip strength")
    ax.set_title("Flip Strength by Season Type")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "summary_flip_strength.png"), dpi=150)
    plt.close(fig)

    summary_lines = []
    summary_lines.append("# 要求二结果摘要（基于 r1 新模型）")
    summary_lines.append("")
    summary_lines.append("## 数据与方法")
    summary_lines.append(f"- 输入数据：`{os.path.normpath(args.votes)}`")
    summary_lines.append("- 方法：比较 CBR（排名合并）与 CBP（百分比合并），并使用 contestant_id 作为稳定性 tie-break。")
    summary_lines.append("")

    summary_lines.append("## 总体结果（按淘汰周加权平均）")
    summary_lines.append(f"- 无 judge-save 赛季：CBR_EMR = {emr_cbr:.3f}，CBP_EMR = {emr_cbp:.3f}")
    summary_lines.append(f"- judge-save 赛季：CBR_B2CR = {b2_cbr:.3f}，CBP_B2CR = {b2_cbp:.3f}")
    summary_lines.append(f"- Flip rate（全体）= {flip_all:.3f}；无 judge-save = {flip_non:.3f}；judge-save = {flip_save:.3f}")
    summary_lines.append("")

    summary_lines.append("## 总体解读（学术表述）")
    summary_lines.append(
        "总体而言，CBR 与 CBP 在无 judge-save 赛季的 EMR 接近，说明两种合并机制在常规淘汰周具有相似的一致性上限。"
        "然而在 judge-save 赛季，CBP 的 B2CR 略高，表明当规则关注 Bottom-2 覆盖时，基于占比的合并对边缘选手更敏感。"
        "Flip rate 处于中等水平，意味着两种规则在相当比例的周次给出不同的淘汰（或 Bottom-2）集合，制度差异具有可观的结构性影响。"
    )
    if np.isfinite(b2_gap) and abs(b2_gap) > 1e-6:
        summary_lines.append(f"补充：CBP 的 B2CR 比 CBR 高约 {b2_gap:.3f}。")
    if np.isfinite(emr_gap) and abs(emr_gap) > 1e-6:
        summary_lines.append(f"补充：CBP 的 EMR 比 CBR 高约 {emr_gap:.3f}。")
    summary_lines.append("")

    summary_lines.append("## 翻转来源分解")
    if flip_total > 0:
        for driver in ["judge", "fan", "mixed"]:
            row = flip_summary.get(driver, {"count": 0, "share": 0.0})
            summary_lines.append(
                f"- {driver}: {row['count']} / {flip_total}（{row['share']:.3f}）"
            )
    else:
        summary_lines.append("- 无翻转记录，无法分解来源。")
    summary_lines.append("")

    summary_lines.append("## 翻转强度分析")
    if strength_vals.size:
        summary_lines.append(
            f"- flip_strength mean = {strength_mean:.3f}，median = {strength_median:.3f}（按排名距离）"
        )
        summary_lines.append(
            f"- no judge-save mean = {strength_by_save.get(False, {}).get('mean', float('nan')):.3f}，"
            f"judge-save mean = {strength_by_save.get(True, {}).get('mean', float('nan')):.3f}"
        )
    else:
        summary_lines.append("- 无翻转记录，无法计算强度。")
    summary_lines.append("")

    summary_lines.append("## 稳健性与不确定性")
    summary_lines.append("- 分箱采用 uncertainty_width 分位数（low/mid/high）。")
    if not unc_summary.empty:
        for label in bin_order:
            if label not in unc_summary.index:
                continue
            row = unc_summary.loc[label]
            flip_rate = to_float(row["flip_rate"])
            strength_mean_bin = to_float(row["mean"]) if "mean" in row else float("nan")
            summary_lines.append(
                f"- {label}: flip_rate = {flip_rate:.3f}，flip_strength_mean = {strength_mean_bin:.3f}"
            )
        if high_unc_missing:
            summary_lines.append("- 高不确定性组 judge-save 周次较少，B2CR 可能缺失或不稳定。")
    else:
        summary_lines.append("- uncertainty_width 缺失或分箱失败，未生成稳健性分析。")
    summary_lines.append("")

    summary_lines.append("## 分层分析（规模 / 赛季阶段）")
    if not strata_df.empty:
        size_df = strata_df[strata_df["strata_type"] == "size"].set_index("strata_label")
        for label in size_order:
            if label not in size_df.index:
                continue
            row = size_df.loc[label]
            summary_lines.append(
                f"- size={label}: EMR(CBR={to_float(row['CBR_EMR']):.3f}, CBP={to_float(row['CBP_EMR']):.3f}), "
                f"B2CR(CBR={to_float(row['CBR_B2CR']):.3f}, CBP={to_float(row['CBP_B2CR']):.3f}), "
                f"flip_rate={to_float(row['flip_rate']):.3f}"
            )
        stage_df = strata_df[strata_df["strata_type"] == "stage"].set_index("strata_label")
        for label in stage_order:
            if label not in stage_df.index:
                continue
            row = stage_df.loc[label]
            summary_lines.append(
                f"- stage={label}: EMR(CBR={to_float(row['CBR_EMR']):.3f}, CBP={to_float(row['CBP_EMR']):.3f}), "
                f"B2CR(CBR={to_float(row['CBR_B2CR']):.3f}, CBP={to_float(row['CBP_B2CR']):.3f}), "
                f"flip_rate={to_float(row['flip_rate']):.3f}"
            )
    else:
        summary_lines.append("- 分层数据不足，未生成分层指标。")
    summary_lines.append("")

    summary_lines.append("## tie 阈值敏感性")
    if not sens_all.empty:
        for row in sens_all.itertuples(index=False, name=None):
            cols = {name: idx for idx, name in enumerate(sens_all.columns)}
            threshold = to_float(row[cols["threshold"]])
            cbr_count = int(row[cols["CBR_count"]])
            cbp_count = int(row[cols["CBP_count"]])
            tie_count = int(row[cols["tie_count"]])
            total = int(row[cols["total"]])
            summary_lines.append(
                f"- threshold={threshold:.2f}: CBR={cbr_count}, CBP={cbp_count}, tie={tie_count} (total={total})"
            )
        if tie_stable:
            summary_lines.append("- 0.03/0.05/0.07 的结果一致，结论对阈值不敏感。")
    else:
        summary_lines.append("- 未生成敏感性统计。")
    summary_lines.append("")

    summary_lines.append("## 典型赛季（5 个）")
    typical_cols = {name: idx for idx, name in enumerate(typical_df.columns)}
    for row in typical_df.itertuples(index=False, name=None):
        season = int(cast(float, row[typical_cols["season"]]))
        save = bool(row[typical_cols["judge_save"]])
        flip_rate = to_float(row[typical_cols["flip_rate"]])
        metric_gap = to_float(row[typical_cols["metric_gap"]])
        if save:
            cbr_val = to_float(row[typical_cols["CBR_B2CR"]])
            cbp_val = to_float(row[typical_cols["CBP_B2CR"]])
            metric_name = "B2CR"
        else:
            cbr_val = to_float(row[typical_cols["CBR_EMR"]])
            cbp_val = to_float(row[typical_cols["CBP_EMR"]])
            metric_name = "EMR"
        label, note = classify_typical_season(flip_rate, metric_gap, save)

        if pd.isna(cbr_val) or pd.isna(cbp_val):
            comp = "两种方法表现接近或数据不足。"
        elif cbr_val > cbp_val:
            comp = f"CBR 在该赛季 {metric_name} 更高。"
        elif cbp_val > cbr_val:
            comp = f"CBP 在该赛季 {metric_name} 更高。"
        else:
            comp = f"两种方法在该赛季 {metric_name} 表现相同。"

        summary_lines.append(
            f"- S{season}（{'judge-save' if save else 'no judge-save'}）："
            f"flip_rate = {flip_rate:.3f}，{metric_name}_gap = {metric_gap:.3f}；{comp}"
            f"标签：{label}。{note}"
        )

    summary_lines.append("")
    summary_lines.append("## 争议选手画像（平均 fan_judge_gap）")
    cont_cols = {name: idx for idx, name in enumerate(cont_summary.columns)}
    for row in cont_summary.itertuples(index=False, name=None):
        season = int(cast(float, row[cont_cols["season"]]))
        name = row[cont_cols["celebrity_name"]]
        gap = float(row[cont_cols["mean_fan_gap"]]) if pd.notna(row[cont_cols["mean_fan_gap"]]) else float("nan")
        weeks = int(cast(float, row[cont_cols["weeks"]]))
        if pd.isna(gap):
            gap_desc = "粉丝与评审偏好差异无法估计。"
        elif gap > 0.02:
            gap_desc = "粉丝偏好显著高于评审。"
        elif gap < -0.02:
            gap_desc = "粉丝偏好显著低于评审。"
        else:
            gap_desc = "粉丝与评审偏好接近。"

        summary_lines.append(
            f"- S{season} {name}：mean fan_judge_gap = {gap:.3f}（{weeks} 周）；{gap_desc}"
        )

    summary_lines.append("")
    summary_lines.append("## 输出索引")
    summary_lines.append(f"- 典型赛季与统计表：`{os.path.join(args.out_dir, 'summary_table.csv')}`")
    summary_lines.append(f"- 争议选手差异曲线：`{os.path.join(args.out_dir, 'controversy_gap_trends.png')}`")
    summary_lines.append(f"- 翻转强度分布：`{os.path.join(args.out_dir, 'flip_strength.png')}`")
    summary_lines.append(f"- 赛季建议表（tie 阈值={tie_threshold:.2f}）：`{os.path.join(args.out_dir, 'decision_guidance.csv')}`")
    summary_lines.append(f"- 不确定性-翻转分析：`{os.path.join(args.out_dir, 'flip_uncertainty_summary.csv')}`")
    summary_lines.append(f"- 不确定性图（rate/strength）：`{os.path.join(args.out_dir, 'flip_uncertainty_rate.png')}` / `{os.path.join(args.out_dir, 'flip_uncertainty_strength.png')}`")
    summary_lines.append(f"- 高低不确定性对比：`{os.path.join(args.out_dir, 'uncertainty_focus.csv')}` / `{os.path.join(args.out_dir, 'uncertainty_focus.png')}`")
    summary_lines.append(f"- tie 阈值敏感性：`{os.path.join(args.out_dir, 'tie_sensitivity.csv')}` / `{os.path.join(args.out_dir, 'tie_sensitivity.png')}`")
    summary_lines.append(f"- 分层指标表：`{os.path.join(args.out_dir, 'strata_metrics.csv')}`")
    summary_lines.append(f"- 分层图（规模/阶段）：`{os.path.join(args.out_dir, 'strata_by_size.png')}` / `{os.path.join(args.out_dir, 'strata_by_stage.png')}`")
    summary_lines.append(f"- 汇总对比图：`{os.path.join(args.out_dir, 'summary_metrics.png')}` / `{os.path.join(args.out_dir, 'summary_flip_strength.png')}`")
    summary_lines.append("")

    summary_lines.append("## 备注")
    summary_lines.append("- r2 输出基于 r1 最新模型估计的 p_hat，结论与旧模型输出不再可比。")
    summary_lines.append("- judge-save 赛季使用 Bottom-2 覆盖率（B2CR）评价方法表现。")
    summary_lines.append("- flip_strength 在 judge-save 赛季以最差者为代表，用于比较规则差异强度。")

    summary_path = os.path.join(args.out_dir, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"Wrote outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
