import argparse
import os
from typing import Dict, List, Tuple

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


def summarize_contestant_exit(df: pd.DataFrame, method_col: str) -> pd.DataFrame:
    # method_col contains list-like string of eliminated index; derive earliest week per contestant
    rows = []
    for (season, name), g in df.groupby(["season", "celebrity_name"]):
        g = g.sort_values("week")
        elim_week = g.loc[g[method_col] == 1, "week"]
        week_val = int(elim_week.iloc[0]) if not elim_week.empty else np.nan
        rows.append({"season": int(season), "celebrity_name": name, f"{method_col}_week": week_val})
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

    records = []
    flip_rows = []

    # Per-season metrics
    metrics_rows = []
    for season, sdf in df.groupby("season"):
        season = int(season)
        save = has_judge_save(season)
        cbr_hit = cbp_hit = 0
        denom = 0
        cbr_elim_match = cbp_elim_match = 0
        flip_count = 0

        for week, wdf in sdf.groupby("week"):
            wdf = wdf.sort_values("contestant_id")
            J = wdf["J_total"].to_numpy(dtype=float)
            p_hat = wdf["p_hat"].to_numpy(dtype=float)
            tie_break = wdf["contestant_id"].to_numpy(dtype=int)

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
                    flip_rows.append({
                        "season": season,
                        "week": int(week),
                        "actual_eliminated": "; ".join(actual_names),
                        "cbr_elim": "; ".join(cbr_elim_name),
                        "cbp_elim": "; ".join(cbp_elim_name),
                        "cbr_bottom2": "; ".join(cbr_bottom2_names),
                        "cbp_bottom2": "; ".join(cbp_bottom2_names),
                        "judge_save": save,
                    })

            records.append({
                "season": season,
                "week": int(week),
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

    metrics_df = pd.DataFrame(metrics_rows).sort_values("season")
    metrics_df.to_csv(os.path.join(args.out_dir, "season_method_metrics.csv"), index=False)

    # Controversial contestants profile under both methods
    controversial = {
        (2, "Jerry Rice"),
        (4, "Billy Ray Cyrus"),
        (11, "Bristol Palin"),
        (27, "Bobby Bones"),
    }

    prof_rows = []
    for (season, week), wdf in df.groupby(["season", "week"]):
        wdf = wdf.sort_values("contestant_id")
        J = wdf["J_total"].to_numpy(dtype=float)
        p_hat = wdf["p_hat"].to_numpy(dtype=float)
        tie_break = wdf["contestant_id"].to_numpy(dtype=int)
        cbr_elim, cbr_bottom2 = apply_cbr(J, p_hat, tie_break)
        cbp_elim, cbp_bottom2 = apply_cbp(J, p_hat, tie_break)

        for idx, row in wdf.iterrows():
            key = (int(row["season"]), row["celebrity_name"])
            if key not in controversial:
                continue
            prof_rows.append({
                "season": int(row["season"]),
                "week": int(row["week"]),
                "celebrity_name": row["celebrity_name"],
                "J_total": float(row["J_total"]),
                "p_hat": float(row["p_hat"]),
                "actual_eliminated": int(row["eliminated"]),
                "cbr_elim": int(row["celebrity_name"] in wdf.iloc[cbr_elim]["celebrity_name"].tolist()),
                "cbp_elim": int(row["celebrity_name"] in wdf.iloc[cbp_elim]["celebrity_name"].tolist()),
                "cbr_bottom2": int(row["celebrity_name"] in wdf.iloc[cbr_bottom2]["celebrity_name"].tolist()),
                "cbp_bottom2": int(row["celebrity_name"] in wdf.iloc[cbp_bottom2]["celebrity_name"].tolist()),
            })

    controversy_df = pd.DataFrame(prof_rows)
    controversy_df.to_csv(os.path.join(args.out_dir, "controversy_analysis.csv"), index=False)

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

    print(f"Wrote outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
