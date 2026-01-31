import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar


def parse_args():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_votes = os.path.normpath(os.path.join(base_dir, "..", "r1", "outputs", "req1_vote_estimates.csv"))
    default_out = os.path.join(base_dir, "outputs")
    ap = argparse.ArgumentParser(description="Requirement 4 analysis: AMEI 2.0 simulation and comparison.")
    ap.add_argument("--votes", default=default_votes, help="Path to req1_vote_estimates.csv")
    ap.add_argument("--out-dir", default=default_out, help="Output directory")
    ap.add_argument("--z-soft", type=float, default=-0.84, help="Soft threshold for gating (approx bottom 20%)")
    ap.add_argument("--z-hard", type=float, default=-1.64, help="Hard threshold for gating (approx bottom 5%)")
    ap.add_argument("--min-vote-weight", type=float, default=0.05, help="Minimum vote weight after gating")
    ap.add_argument("--w-start", type=float, default=0.65, help="Judge weight at season start")
    ap.add_argument("--w-end", type=float, default=0.35, help="Judge weight at season end")
    ap.add_argument("--logistic-k", type=float, default=10.0, help="Logistic transition steepness")
    ap.add_argument("--logistic-t0", type=float, default=0.5, help="Logistic transition midpoint (progress)")
    ap.add_argument("--eps", type=float, default=1e-9, help="Numerical epsilon")
    return ap.parse_args()


def has_judge_save(season: int) -> bool:
    return season >= 28


def robust_z_score(values: np.ndarray, eps: float) -> np.ndarray:
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    if not np.isfinite(mad) or mad <= eps:
        std = np.nanstd(values)
        if not np.isfinite(std) or std <= eps:
            return np.zeros_like(values, dtype=float)
        return (values - np.nanmean(values)) / (std + eps)
    return 0.6745 * (values - median) / (mad + eps)


def softmax(values: np.ndarray) -> np.ndarray:
    vals = values - np.nanmax(values)
    expv = np.exp(vals)
    denom = np.sum(expv)
    if denom <= 0:
        return np.full_like(values, 1.0 / len(values), dtype=float)
    return expv / denom


def hinge_gate(z_scores: np.ndarray, z_soft: float, z_hard: float, min_vote_weight: float) -> np.ndarray:
    denom = z_soft - z_hard
    if abs(denom) < 1e-9:
        return np.full_like(z_scores, 1.0, dtype=float)
    raw = (z_scores - z_hard) / denom
    return np.clip(raw, min_vote_weight, 1.0)


def logistic_weights(progress: float, w_start: float, w_end: float, k: float, t0: float) -> Tuple[float, float]:
    w_j = w_end + (w_start - w_end) / (1 + np.exp(k * (progress - t0)))
    w_f = 1.0 - w_j
    return w_j, w_f


def pick_bottom_k(scores: np.ndarray, k: int, tie_break: np.ndarray) -> List[int]:
    order = np.lexsort((tie_break, scores))
    return order[:k].tolist()


def rank_avg_desc(values: np.ndarray, rtol: float = 1e-9) -> np.ndarray:
    """Compute average ranks for descending order with tie handling."""
    order = np.argsort(-values, kind="mergesort")
    ranks = np.zeros(len(values), dtype=float)
    i = 0
    while i < len(order):
        j = i
        # Use relative tolerance for floating point comparison
        while j + 1 < len(order) and np.isclose(values[order[j + 1]], values[order[i]], rtol=rtol):
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


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
    elim = pick_bottom_k(-ranksum, 1, tie_break)
    bottom2 = pick_bottom_k(-ranksum, 2, tie_break)
    return elim, bottom2


def compute_entropy(scores: np.ndarray, eps: float) -> float:
    scores = np.asarray(scores, dtype=float)
    min_val = np.nanmin(scores)
    if not np.isfinite(min_val):
        return float("nan")
    if min_val <= 0:
        scores = scores - min_val + eps
    risk = 1.0 / (scores + eps)
    total = np.sum(risk)
    if total <= 0:
        return float("nan")
    p = risk / total
    return float(-np.sum(p * np.log2(p + eps)))


def safe_mean(values: np.ndarray) -> float:
    """Compute mean safely without empty-slice warnings."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanmean(arr))


def min_max_norm(values: List[float]) -> np.ndarray:
    """Normalize values to [0,1] with stable handling of equal values."""
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(arr).any():
        return np.zeros_like(arr)
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    if np.isclose(vmin, vmax):
        return np.full_like(arr, 0.5)
    return (arr - vmin) / (vmax - vmin)


def compute_fairness_loss(j_scores: np.ndarray, weights: np.ndarray) -> float:
    """Compute fairness loss: sum((J_i - J_bar)^2 * w_i)."""
    if len(j_scores) == 0:
        return float("nan")
    j_scores = np.asarray(j_scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    j_bar = np.nanmean(j_scores)
    return float(np.nansum(((j_scores - j_bar) ** 2) * weights))


def compute_excitement_loss(scores: np.ndarray, bottom2_idx: List[int]) -> float:
    """Compute excitement loss as absolute gap between bottom-2 scores."""
    if len(bottom2_idx) < 2:
        return float("nan")
    s1 = scores[bottom2_idx[0]]
    s2 = scores[bottom2_idx[1]]
    return float(abs(s1 - s2))


def identify_controversy_cases(
    z_scores: np.ndarray,
    p_hat: np.ndarray,
    z_quantile: float = 0.2,
    p_quantile: float = 0.8,
) -> np.ndarray:
    """Identify high-popularity/low-technical contestants."""
    z_thr = np.nanquantile(z_scores, z_quantile)
    p_thr = np.nanquantile(p_hat, p_quantile)
    return (z_scores <= z_thr) & (p_hat >= p_thr)


def bobby_interception_rate(controversy_mask: np.ndarray, bottom2_idx: List[int]) -> float:
    """Interception rate: controversy contestants captured in Bottom-2."""
    if controversy_mask.sum() == 0:
        return float("nan")
    captured = sum(1 for i in bottom2_idx if controversy_mask[i])
    return float(captured / max(controversy_mask.sum(), 1))


def precompute_week_cache(df: pd.DataFrame, eps: float) -> List[Dict[str, object]]:
    """Precompute week-level data to speed up sensitivity analysis."""
    max_week_map: Dict[int, int] = df.groupby("season")["week"].max().to_dict()
    cache: List[Dict[str, object]] = []

    for (season, week), g in df.groupby(["season", "week"], sort=True):
        g = g.sort_values("contestant_id").reset_index(drop=True)
        J = g["J_total"].to_numpy(dtype=float)
        p_hat = g["p_hat"].to_numpy(dtype=float)
        tie_break = g["contestant_id"].to_numpy(dtype=int)
        max_week = max_week_map.get(int(season), int(week))
        if max_week <= 1:
            progress = 0.0
        else:
            progress = (int(week) - 1) / (max_week - 1)

        z_scores = robust_z_score(J, eps)
        merit_score = softmax(z_scores)
        actual_elim_idx = g.index[g["eliminated"].astype(int) == 1].tolist()
        actual_elim_idx = actual_elim_idx[:1] if len(actual_elim_idx) > 0 else []

        cache.append({
            "season": int(season),
            "week": int(week),
            "judge_save": has_judge_save(int(season)),
            "J": J,
            "p_hat": p_hat,
            "tie_break": tie_break,
            "progress": float(progress),
            "z_scores": z_scores,
            "merit_score": merit_score,
            "actual_elim_idx": actual_elim_idx,
        })

    return cache


def evaluate_amei_for_params(
    week_cache: List[Dict[str, object]],
    z_soft: float,
    z_hard: float,
    min_vote_weight: float,
    w_start: float,
    w_end: float,
    logistic_k: float,
    logistic_t0: float,
    eps: float,
) -> Dict[str, float]:
    week_rows: List[Dict[str, float]] = []

    for row in week_cache:
        J = row["J"]
        p_hat = row["p_hat"]
        tie_break = row["tie_break"]
        progress = row["progress"]
        z_scores = row["z_scores"]
        merit_score = row["merit_score"]
        actual_elim_idx = row["actual_elim_idx"]

        kappa = hinge_gate(z_scores, z_soft, z_hard, min_vote_weight)
        effective_votes = p_hat * kappa
        vote_sum = np.sum(effective_votes)
        if vote_sum > 0:
            pop_score = effective_votes / vote_sum
        else:
            pop_sum = np.sum(p_hat)
            pop_score = p_hat / pop_sum if pop_sum > 0 else np.full_like(p_hat, 1.0 / len(p_hat))

        w_j, w_f = logistic_weights(progress, w_start, w_end, logistic_k, logistic_t0)
        final_score = w_j * merit_score + w_f * pop_score

        bottom2_idx = pick_bottom_k(final_score, 2, tie_break)
        elim_idx = bottom2_idx[:1]

        judge_save = bool(row["judge_save"])
        if judge_save:
            hit = 1 if any(i in bottom2_idx for i in actual_elim_idx) else 0
        else:
            hit = 1 if any(i in elim_idx for i in actual_elim_idx) else 0

        entropy = compute_entropy(final_score, eps)
        fairness_loss = compute_fairness_loss(J, kappa)
        excitement_loss = compute_excitement_loss(final_score, bottom2_idx)
        controversy_mask = identify_controversy_cases(z_scores, p_hat)
        intercept = bobby_interception_rate(controversy_mask, bottom2_idx)

        week_rows.append({
            "judge_save": judge_save,
            "hit": hit,
            "entropy": entropy,
            "fairness_loss": fairness_loss,
            "excitement_loss": excitement_loss,
            "interception": intercept,
        })

    week_df = pd.DataFrame.from_records(week_rows)
    js = week_df[week_df["judge_save"] == True]
    no_js = week_df[week_df["judge_save"] == False]

    return {
        "z_soft": float(z_soft),
        "z_hard": float(z_hard),
        "b2cr": safe_mean(js["hit"]) if len(js) else float("nan"),
        "emr": safe_mean(no_js["hit"]) if len(no_js) else float("nan"),
        "entropy_mean": safe_mean(week_df["entropy"]) if len(week_df) else float("nan"),
        "fairness_loss_mean": safe_mean(week_df["fairness_loss"]) if len(week_df) else float("nan"),
        "excitement_loss_mean": safe_mean(week_df["excitement_loss"]) if len(week_df) else float("nan"),
        "interception_mean": safe_mean(week_df["interception"]) if len(week_df) else float("nan"),
        "weeks_js": int(len(js)),
        "weeks_nojs": int(len(no_js)),
    }


def plot_heatmap(df_grid: pd.DataFrame, value_col: str, out_path: str, title: str) -> None:
    import matplotlib.pyplot as plt

    pivot = df_grid.pivot(index="z_hard", columns="z_soft", values=value_col)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="viridis")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])

    ax.set_xlabel("z_soft")
    ax.set_ylabel("z_hard")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Mark best point (max value)
    if np.isfinite(pivot.values).any():
        max_idx = np.nanargmax(pivot.values)
        r, c = np.unravel_index(max_idx, pivot.values.shape)
        ax.scatter([c], [r], s=80, facecolors="none", edgecolors="red", linewidths=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_sensitivity_analysis(df: pd.DataFrame, args, out_dir: str) -> pd.DataFrame:
    week_cache = precompute_week_cache(df, args.eps)
    z_soft_vals = np.round(np.linspace(-0.5, -1.5, 5), 2)
    z_hard_vals = np.round(np.linspace(-1.5, -2.5, 5), 2)

    records: List[Dict[str, float]] = []
    for z_soft in z_soft_vals:
        for z_hard in z_hard_vals:
            if z_hard >= z_soft:
                continue
            records.append(
                evaluate_amei_for_params(
                    week_cache,
                    z_soft=z_soft,
                    z_hard=z_hard,
                    min_vote_weight=args.min_vote_weight,
                    w_start=args.w_start,
                    w_end=args.w_end,
                    logistic_k=args.logistic_k,
                    logistic_t0=args.logistic_t0,
                    eps=args.eps,
                )
            )

    grid_df = pd.DataFrame.from_records(records)
    grid_path = os.path.join(out_dir, "sensitivity_grid.csv")
    grid_df.to_csv(grid_path, index=False, encoding="utf-8-sig")

    if grid_df["weeks_nojs"].sum() > 0:
        plot_heatmap(
            grid_df,
            value_col="emr",
            out_path=os.path.join(out_dir, "sensitivity_heatmap_nojs.png"),
            title="EMR Heatmap (No Judge Save)",
        )

    if grid_df["weeks_js"].sum() > 0:
        plot_heatmap(
            grid_df,
            value_col="b2cr",
            out_path=os.path.join(out_dir, "sensitivity_heatmap_js.png"),
            title="B2CR Heatmap (Judge Save)",
        )

    return grid_df


def plot_pareto_frontier(
    metrics_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    out_dir: str,
) -> None:
    import matplotlib.pyplot as plt

    js = metrics_df[metrics_df["judge_save"] == True]
    no_js = metrics_df[metrics_df["judge_save"] == False]

    def get_pareto_front(points: np.ndarray) -> np.ndarray:
        points = points[np.argsort(points[:, 0])]
        front = []
        best_y = float("inf")
        for x, y in points:
            if y < best_y:
                front.append((x, y))
                best_y = y
        return np.array(front)

    def scatter_plot(title: str, capture_col: str, out_name: str):
        fig, ax = plt.subplots(figsize=(6.5, 5))

        # AMEI grid points
        if capture_col == "emr":
            grid_x = grid_df["excitement_loss_mean"]
            grid_y = grid_df["fairness_loss_mean"]
        else:
            grid_x = grid_df["excitement_loss_mean"]
            grid_y = grid_df["fairness_loss_mean"]
        ax.scatter(grid_x, grid_y, s=20, alpha=0.5, label="AMEI (grid)")

        points = np.column_stack([grid_x.to_numpy(dtype=float), grid_y.to_numpy(dtype=float)])
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) > 0:
            front = get_pareto_front(points)
            if len(front) > 1:
                ax.plot(front[:, 0], front[:, 1], color="red", linewidth=2, label="Pareto frontier")

        # Baseline points
        if capture_col.startswith("emr"):
            base_df = no_js
        else:
            base_df = js

        if len(base_df) > 0:
            amei_x = base_df["excitement_loss_mean_amei"].mean()
            amei_y = base_df["fairness_loss_mean_amei"].mean()
            cbr_x = base_df["excitement_loss_mean_cbr"].mean()
            cbr_y = base_df["fairness_loss_mean_cbr"].mean()
            cbp_x = base_df["excitement_loss_mean_cbp"].mean()
            cbp_y = base_df["fairness_loss_mean_cbp"].mean()

            ax.scatter(amei_x, amei_y, s=80, marker="*", label="AMEI 2.0")
            ax.scatter(cbr_x, cbr_y, s=60, marker="D", label="CBR")
            ax.scatter(cbp_x, cbp_y, s=60, marker="s", label="CBP")

        ax.set_xlabel("Excitement Loss (Bottom-2 gap)")
        ax.set_ylabel("Fairness Loss")
        ax.set_title(title)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, out_name), dpi=200)
        plt.close(fig)

    if len(no_js) > 0:
        scatter_plot(
            title="Pareto Frontier (No Judge Save)",
            capture_col="emr",
            out_name="pareto_nojs.png",
        )

    if len(js) > 0:
        scatter_plot(
            title="Pareto Frontier (Judge Save)",
            capture_col="b2cr",
            out_name="pareto_js.png",
        )


def plot_bobby_trajectory(bobby_df: pd.DataFrame, out_dir: str) -> None:
    import matplotlib.pyplot as plt

    bobby = bobby_df[bobby_df["celebrity_name"].str.contains("Bobby", case=False, na=False)].copy()
    if bobby.empty:
        return

    bobby = bobby.sort_values("week")
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)

    axes[0].plot(bobby["week"], bobby["z_score"], marker="o")
    axes[0].set_ylabel("z-score")
    axes[0].set_title("Bobby Bones Trajectory")

    axes[1].plot(bobby["week"], bobby["kappa"], marker="o", color="orange")
    axes[1].set_ylabel("kappa")

    axes[2].plot(bobby["week"], bobby["amei_rank"], marker="o", color="green")
    axes[2].invert_yaxis()
    axes[2].set_ylabel("AMEI Rank")
    axes[2].set_xlabel("Week")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "bobby_trajectory.png"), dpi=200)
    plt.close(fig)


def plot_weight_transition(args, out_dir: str) -> None:
    """Plot logistic weight transition across season progress."""
    import matplotlib.pyplot as plt

    progress = np.linspace(0, 1, 200)
    w_j = []
    w_f = []
    for p in progress:
        wj, wf = logistic_weights(p, args.w_start, args.w_end, args.logistic_k, args.logistic_t0)
        w_j.append(wj)
        w_f.append(wf)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(progress, w_j, label="w_j (Judge)", color="tab:blue")
    ax.plot(progress, w_f, label="w_f (Audience)", color="tab:orange")

    ax.axvspan(0.0, 0.3, color="#e8f3ff", alpha=0.6, label="Early")
    ax.axvspan(0.3, 0.7, color="#fff3e0", alpha=0.6, label="Mid")
    ax.axvspan(0.7, 1.0, color="#f3e5f5", alpha=0.6, label="Late")

    t0 = args.logistic_t0
    wj_t0, wf_t0 = logistic_weights(t0, args.w_start, args.w_end, args.logistic_k, args.logistic_t0)
    ax.scatter([t0], [wj_t0], color="tab:blue", zorder=5)
    ax.scatter([t0], [wf_t0], color="tab:orange", zorder=5)
    ax.annotate(f"t0={t0:.2f}\nwj={wj_t0:.2f}", (t0, wj_t0), xytext=(8, 10), textcoords="offset points")
    ax.annotate(f"wf={wf_t0:.2f}", (t0, wf_t0), xytext=(8, -15), textcoords="offset points")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Season Progress")
    ax.set_ylabel("Weight")
    ax.set_title("Logistic Weight Transition")
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "weight_transition.png"), dpi=200)
    plt.close(fig)


def plot_radar_comparison(metrics_df: pd.DataFrame, week_df: pd.DataFrame, out_dir: str) -> None:
    """Plot radar chart comparing AMEI/CBR/CBP across multiple metrics."""
    import matplotlib.pyplot as plt

    no_js = week_df[week_df["judge_save"] == False]
    js = week_df[week_df["judge_save"] == True]

    emr = [safe_mean(no_js["amei_hit"]), safe_mean(no_js["cbr_hit"]), safe_mean(no_js["cbp_hit"])]
    b2cr = [safe_mean(js["amei_hit"]), safe_mean(js["cbr_hit"]), safe_mean(js["cbp_hit"])]
    intercept = [
        safe_mean(week_df["intercept_amei"]),
        safe_mean(week_df["intercept_cbr"]),
        safe_mean(week_df["intercept_cbp"]),
    ]
    fairness_loss = [
        safe_mean(week_df["fairness_loss_amei"]),
        safe_mean(week_df["fairness_loss_cbr"]),
        safe_mean(week_df["fairness_loss_cbp"]),
    ]
    excitement_loss = [
        safe_mean(week_df["excitement_loss_amei"]),
        safe_mean(week_df["excitement_loss_cbr"]),
        safe_mean(week_df["excitement_loss_cbp"]),
    ]
    entropy = [
        safe_mean(week_df["entropy"]),
        safe_mean(week_df["entropy_cbr"]),
        safe_mean(week_df["entropy_cbp"]),
    ]

    emr_n = min_max_norm(emr)
    b2cr_n = min_max_norm(b2cr)
    intercept_n = min_max_norm(intercept)
    fairness_n = 1 - min_max_norm(fairness_loss)
    excitement_n = 1 - min_max_norm(excitement_loss)
    entropy_n = min_max_norm(entropy)

    metrics = ["EMR", "B2CR", "Interception", "Fairness", "Excitement", "Entropy"]
    values = {
        "AMEI 2.0": [emr_n[0], b2cr_n[0], intercept_n[0], fairness_n[0], excitement_n[0], entropy_n[0]],
        "CBR": [emr_n[1], b2cr_n[1], intercept_n[1], fairness_n[1], excitement_n[1], entropy_n[1]],
        "CBP": [emr_n[2], b2cr_n[2], intercept_n[2], fairness_n[2], excitement_n[2], entropy_n[2]],
    }

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7, 6))
    ax = plt.subplot(111, polar=True)

    for label, vals in values.items():
        data = np.array(vals, dtype=float)
        data = np.nan_to_num(data, nan=0.0)
        data = np.concatenate([data, data[:1]])
        ax.plot(angles, data, linewidth=2, label=label)
        ax.fill(angles, data, alpha=0.15)

    ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
    ax.set_ylim(0, 1)
    ax.set_title("Radar Comparison of Methods")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "radar_comparison.png"), dpi=200)
    plt.close(fig)


def compute_whatif_scenario(full_df: pd.DataFrame, out_dir: str) -> None:
    """Compute Bobby Bones what-if scenario for Season 27 under AMEI 2.0."""
    df27 = full_df[full_df["season"] == 27].copy()
    if df27.empty:
        return

    results = []
    first_bottom2_week = None
    eliminated_week = None

    for week, g in df27.groupby("week", sort=True):
        g = g.sort_values("contestant_id").reset_index(drop=True)
        scores = g["amei_score"].to_numpy(dtype=float)
        tie_break = g["contestant_id"].to_numpy(dtype=int)
        bottom2_idx = pick_bottom_k(scores, 2, tie_break)
        bottom2 = g.loc[bottom2_idx]

        bobby_rows = g[g["celebrity_name"].str.contains("Bobby", case=False, na=False)]
        if bobby_rows.empty:
            continue
        bobby = bobby_rows.iloc[0]
        bobby_idx = bobby.name

        in_bottom2 = bobby_idx in bottom2_idx
        if in_bottom2 and first_bottom2_week is None:
            first_bottom2_week = int(week)

        # Judge's Save hypothetical: judges save higher J_total in bottom-2
        if in_bottom2:
            other = bottom2[bottom2.index != bobby_idx].iloc[0]
            bobby_elim_js = bobby["J_total"] <= other["J_total"]
        else:
            bobby_elim_js = False

        if bobby_elim_js and eliminated_week is None:
            eliminated_week = int(week)

        results.append({
            "season": 27,
            "week": int(week),
            "celebrity_name": bobby["celebrity_name"],
            "amei_rank": int(bobby["amei_rank"]),
            "bottom2_flag": int(in_bottom2),
            "amei_score": float(bobby["amei_score"]),
            "J_total": float(bobby["J_total"]),
            "p_hat": float(bobby["p_hat"]),
            "eliminated_under_judge_save": int(bobby_elim_js),
        })

    if not results:
        return

    out_csv = os.path.join(out_dir, "whatif_bobby_bones.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8-sig")

    finale_week = int(df27["week"].max())
    reaches_finale = "Yes" if (eliminated_week is None or eliminated_week >= finale_week) else "No"
    summary_lines = [
        "# What-If Scenario: Season 27 with AMEI 2.0",
        "",
        f"- First Bottom-2 week: {first_bottom2_week if first_bottom2_week is not None else 'N/A'}",
        f"- Hypothetical elimination week (Judge's Save active): {eliminated_week if eliminated_week is not None else 'N/A'}",
        f"- Reaches finale week: {reaches_finale}",
        "",
        "Assumption: If Judge's Save is active, judges save the higher J_total among bottom-2.",
    ]
    out_md = os.path.join(out_dir, "whatif_summary.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))


def bootstrap_metrics(week_df: pd.DataFrame, out_dir: str, n_bootstrap: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Compute bootstrap confidence intervals for EMR/B2CR/Interception."""
    rng = np.random.default_rng(seed)

    def bootstrap_ci(values: np.ndarray) -> Tuple[float, float]:
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return float("nan"), float("nan")
        means = []
        for _ in range(n_bootstrap):
            sample = rng.choice(vals, size=len(vals), replace=True)
            means.append(np.mean(sample))
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    records = []
    no_js = week_df[week_df["judge_save"] == False]
    js = week_df[week_df["judge_save"] == True]

    for method in ["amei", "cbr", "cbp"]:
        emr_vals = no_js[f"{method}_hit"].to_numpy(dtype=float)
        b2cr_vals = js[f"{method}_hit"].to_numpy(dtype=float)
        inter_vals = week_df[f"intercept_{method}"].to_numpy(dtype=float)

        emr_ci = bootstrap_ci(emr_vals)
        b2cr_ci = bootstrap_ci(b2cr_vals)
        inter_ci = bootstrap_ci(inter_vals)

        records.extend([
            {
                "metric": "EMR",
                "method": method.upper(),
                "group": "No Judge Save",
                "mean": safe_mean(emr_vals),
                "ci_low": emr_ci[0],
                "ci_high": emr_ci[1],
            },
            {
                "metric": "B2CR",
                "method": method.upper(),
                "group": "Judge Save",
                "mean": safe_mean(b2cr_vals),
                "ci_low": b2cr_ci[0],
                "ci_high": b2cr_ci[1],
            },
            {
                "metric": "Interception",
                "method": method.upper(),
                "group": "All Weeks",
                "mean": safe_mean(inter_vals),
                "ci_low": inter_ci[0],
                "ci_high": inter_ci[1],
            },
        ])

    ci_df = pd.DataFrame.from_records(records)
    ci_path = os.path.join(out_dir, "bootstrap_ci.csv")
    ci_df.to_csv(ci_path, index=False, encoding="utf-8-sig")
    return ci_df


def write_bootstrap_summary(ci_df: pd.DataFrame, out_path: str) -> None:
    """Write bootstrap CI summary to markdown."""
    lines = [
        "## Bootstrap Confidence Intervals (95%)",
        "",
        "| Metric | Method | Group | Mean | CI Low | CI High |",
        "|--------|--------|-------|------|--------|---------|",
    ]
    for _, row in ci_df.iterrows():
        lines.append(
            f"| {row['metric']} | {row['method']} | {row['group']} | {row['mean']:.3f} | {row['ci_low']:.3f} | {row['ci_high']:.3f} |"
        )
    lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def stage_analysis(week_df: pd.DataFrame, out_dir: str) -> None:
    """Analyze EMR/B2CR by season stage and plot grouped bars."""
    def stage_label(p: float) -> str:
        if p < 0.33:
            return "Early"
        if p <= 0.66:
            return "Mid"
        return "Late"

    df = week_df.copy()
    df["stage"] = df["progress"].apply(stage_label)

    records = []
    for stage in ["Early", "Mid", "Late"]:
        stage_df = df[df["stage"] == stage]
        no_js = stage_df[stage_df["judge_save"] == False]
        js = stage_df[stage_df["judge_save"] == True]
        for method in ["amei", "cbr", "cbp"]:
            records.append({
                "stage": stage,
                "metric": "EMR",
                "method": method.upper(),
                "value": safe_mean(no_js[f"{method}_hit"].to_numpy(dtype=float)),
            })
            records.append({
                "stage": stage,
                "metric": "B2CR",
                "method": method.upper(),
                "value": safe_mean(js[f"{method}_hit"].to_numpy(dtype=float)),
            })

    out_df = pd.DataFrame.from_records(records)
    out_df.to_csv(os.path.join(out_dir, "stage_analysis.csv"), index=False, encoding="utf-8-sig")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    metrics = ["EMR", "B2CR"]
    stages = ["Early", "Mid", "Late"]
    methods = ["AMEI", "CBR", "CBP"]
    x = np.arange(len(stages))
    width = 0.25

    for ax, metric in zip(axes, metrics):
        for i, method in enumerate(methods):
            vals = []
            for stage in stages:
                v = out_df[(out_df["stage"] == stage) & (out_df["metric"] == metric) & (out_df["method"] == method)]["value"]
                vals.append(float(v.values[0]) if len(v) else np.nan)
            ax.bar(x + (i - 1) * width, vals, width=width, label=method)
        ax.set_xticks(x)
        ax.set_xticklabels(stages)
        ax.set_ylim(0, 1)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Rate")
    axes[1].legend(loc="best")
    fig.suptitle("Stage Comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stage_comparison.png"), dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.votes)

    needed_cols = {"season", "week", "contestant_id", "celebrity_name", "J_total", "p_hat", "eliminated"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    df["contestant_id"] = df["contestant_id"].astype(int)

    max_week_map: Dict[int, int] = df.groupby("season")["week"].max().to_dict()

    records: List[Dict[str, float]] = []
    week_metrics: List[Dict[str, float]] = []
    bobby_rows: List[Dict[str, float]] = []

    for (season, week), g in df.groupby(["season", "week"], sort=True):
        g = g.sort_values("contestant_id").reset_index(drop=True)
        J = g["J_total"].to_numpy(dtype=float)
        p_hat = g["p_hat"].to_numpy(dtype=float)
        tie_break = g["contestant_id"].to_numpy(dtype=int)

        max_week = max_week_map.get(int(season), int(week))
        if max_week <= 1:
            progress = 0.0
        else:
            progress = (int(week) - 1) / (max_week - 1)

        z_scores = robust_z_score(J, args.eps)
        merit_score = softmax(z_scores)
        kappa = hinge_gate(z_scores, args.z_soft, args.z_hard, args.min_vote_weight)
        effective_votes = p_hat * kappa
        vote_sum = np.sum(effective_votes)
        if vote_sum > 0:
            pop_score = effective_votes / vote_sum
        else:
            pop_sum = np.sum(p_hat)
            pop_score = p_hat / pop_sum if pop_sum > 0 else np.full_like(p_hat, 1.0 / len(p_hat))

        w_j, w_f = logistic_weights(progress, args.w_start, args.w_end, args.logistic_k, args.logistic_t0)
        final_score = w_j * merit_score + w_f * pop_score

        bottom2_idx = pick_bottom_k(final_score, 2, tie_break)
        elim_idx = bottom2_idx[:1]

        cbp_elim, cbp_bottom2 = apply_cbp(J, p_hat, tie_break)
        cbr_elim, cbr_bottom2 = apply_cbr(J, p_hat, tie_break)

        actual_elim_idx = g.index[g["eliminated"].astype(int) == 1].tolist()
        actual_elim_idx = actual_elim_idx[:1] if len(actual_elim_idx) > 0 else []

        judge_save = has_judge_save(int(season))
        if judge_save:
            amei_hit = 1 if any(i in bottom2_idx for i in actual_elim_idx) else 0
            cbr_hit = 1 if any(i in cbr_bottom2 for i in actual_elim_idx) else 0
            cbp_hit = 1 if any(i in cbp_bottom2 for i in actual_elim_idx) else 0
        else:
            amei_hit = 1 if any(i in elim_idx for i in actual_elim_idx) else 0
            cbr_hit = 1 if any(i in cbr_elim for i in actual_elim_idx) else 0
            cbp_hit = 1 if any(i in cbp_elim for i in actual_elim_idx) else 0

        entropy = compute_entropy(final_score, args.eps)
        # Baseline scores
        P_J = J / np.sum(J) if np.sum(J) > 0 else np.full_like(J, 1.0 / len(J))
        cbp_score = P_J + p_hat
        cbr_score = -rank_avg_desc(J) - rank_avg_desc(p_hat)
        entropy_cbp = compute_entropy(cbp_score, args.eps)
        entropy_cbr = compute_entropy(cbr_score, args.eps)

        fairness_loss_amei = compute_fairness_loss(J, kappa)
        fairness_loss_cbr = compute_fairness_loss(J, np.ones_like(J))
        fairness_loss_cbp = compute_fairness_loss(J, np.ones_like(J))
        excitement_loss_amei = compute_excitement_loss(final_score, bottom2_idx)
        excitement_loss_cbr = compute_excitement_loss(cbr_score, cbr_bottom2)
        excitement_loss_cbp = compute_excitement_loss(cbp_score, cbp_bottom2)

        controversy_mask = identify_controversy_cases(z_scores, p_hat)
        amei_intercept = bobby_interception_rate(controversy_mask, bottom2_idx)
        cbr_intercept = bobby_interception_rate(controversy_mask, cbr_bottom2)
        cbp_intercept = bobby_interception_rate(controversy_mask, cbp_bottom2)

        week_metrics.append({
            "season": int(season),
            "week": int(week),
            "progress": float(progress),
            "judge_save": bool(judge_save),
            "amei_hit": amei_hit,
            "cbr_hit": cbr_hit,
            "cbp_hit": cbp_hit,
            "entropy": entropy,
            "entropy_cbr": entropy_cbr,
            "entropy_cbp": entropy_cbp,
            "fairness_loss_amei": fairness_loss_amei,
            "fairness_loss_cbr": fairness_loss_cbr,
            "fairness_loss_cbp": fairness_loss_cbp,
            "excitement_loss_amei": excitement_loss_amei,
            "excitement_loss_cbr": excitement_loss_cbr,
            "excitement_loss_cbp": excitement_loss_cbp,
            "intercept_amei": amei_intercept,
            "intercept_cbr": cbr_intercept,
            "intercept_cbp": cbp_intercept,
        })

        for idx in range(len(g)):
            records.append({
                "season": int(season),
                "week": int(week),
                "contestant_id": int(g.loc[idx, "contestant_id"]),
                "celebrity_name": g.loc[idx, "celebrity_name"],
                "J_total": float(J[idx]),
                "p_hat": float(p_hat[idx]),
                "z_score": float(z_scores[idx]),
                "kappa": float(kappa[idx]),
                "merit_score": float(merit_score[idx]),
                "pop_score": float(pop_score[idx]),
                "w_j": float(w_j),
                "w_f": float(w_f),
                "amei_score": float(final_score[idx]),
                "amei_rank": int(pd.Series(final_score).rank(ascending=False, method="min").iloc[idx]),
                "bottom2_flag": int(idx in bottom2_idx),
                "eliminated": int(g.loc[idx, "eliminated"]),
            })

        if int(season) == 27:
            for idx in range(len(g)):
                bobby_rows.append({
                    "season": int(season),
                    "week": int(week),
                    "contestant_id": int(g.loc[idx, "contestant_id"]),
                    "celebrity_name": g.loc[idx, "celebrity_name"],
                    "J_total": float(J[idx]),
                    "p_hat": float(p_hat[idx]),
                    "z_score": float(z_scores[idx]),
                    "kappa": float(kappa[idx]),
                    "w_j": float(w_j),
                    "w_f": float(w_f),
                    "merit_score": float(merit_score[idx]),
                    "pop_score": float(pop_score[idx]),
                    "amei_score": float(final_score[idx]),
                    "amei_rank": int(pd.Series(final_score).rank(ascending=False, method="min").iloc[idx]),
                    "bottom2_flag": int(idx in bottom2_idx),
                    "eliminated": int(g.loc[idx, "eliminated"]),
                })

    full_df = pd.DataFrame.from_records(records)
    full_path = os.path.join(args.out_dir, "amei_full_simulation.csv")
    full_df.to_csv(full_path, index=False, encoding="utf-8-sig")

    week_df = pd.DataFrame.from_records(week_metrics)
    season_metrics = []
    for season, g in week_df.groupby("season", sort=True):
        judge_save = bool(g["judge_save"].iloc[0])
        metric = {
            "season": int(season),
            "judge_save": judge_save,
            "weeks": int(g.shape[0]),
            "entropy_mean": safe_mean(g["entropy"]),
            "entropy_mean_amei": safe_mean(g["entropy"]),
            "entropy_mean_cbr": safe_mean(g["entropy_cbr"]),
            "entropy_mean_cbp": safe_mean(g["entropy_cbp"]),
            "fairness_loss_mean_amei": safe_mean(g["fairness_loss_amei"]),
            "fairness_loss_mean_cbr": safe_mean(g["fairness_loss_cbr"]),
            "fairness_loss_mean_cbp": safe_mean(g["fairness_loss_cbp"]),
            "excitement_loss_mean_amei": safe_mean(g["excitement_loss_amei"]),
            "excitement_loss_mean_cbr": safe_mean(g["excitement_loss_cbr"]),
            "excitement_loss_mean_cbp": safe_mean(g["excitement_loss_cbp"]),
            "interception_mean_amei": safe_mean(g["intercept_amei"]),
            "interception_mean_cbr": safe_mean(g["intercept_cbr"]),
            "interception_mean_cbp": safe_mean(g["intercept_cbp"]),
        }
        if judge_save:
            metric.update({
                "b2cr_amei": float(np.nanmean(g["amei_hit"])),
                "b2cr_cbr": float(np.nanmean(g["cbr_hit"])),
                "b2cr_cbp": float(np.nanmean(g["cbp_hit"])),
                "emr_amei": float("nan"),
                "emr_cbr": float("nan"),
                "emr_cbp": float("nan"),
            })
        else:
            metric.update({
                "emr_amei": float(np.nanmean(g["amei_hit"])),
                "emr_cbr": float(np.nanmean(g["cbr_hit"])),
                "emr_cbp": float(np.nanmean(g["cbp_hit"])),
                "b2cr_amei": float("nan"),
                "b2cr_cbr": float("nan"),
                "b2cr_cbp": float("nan"),
            })
        season_metrics.append(metric)

    metrics_df = pd.DataFrame(season_metrics)
    metrics_path = os.path.join(args.out_dir, "comparison_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    if bobby_rows:
        bobby_df = pd.DataFrame.from_records(bobby_rows)
        bobby_path = os.path.join(args.out_dir, "bobby_bones_case_study.csv")
        bobby_df.to_csv(bobby_path, index=False, encoding="utf-8-sig")
        plot_bobby_trajectory(bobby_df, args.out_dir)

    # Generate overall summary statistics
    generate_summary_report(metrics_df, full_df, bobby_rows, args.out_dir)
    mcnemar_path = os.path.join(args.out_dir, "mcnemar_summary.md")
    write_mcnemar_summary(week_df, mcnemar_path)
    append_markdown(args.out_dir, "interpretation_report.md", mcnemar_path)
    ci_df = bootstrap_metrics(week_df, args.out_dir)
    bootstrap_md = os.path.join(args.out_dir, "bootstrap_summary.md")
    write_bootstrap_summary(ci_df, bootstrap_md)
    append_markdown(args.out_dir, "interpretation_report.md", bootstrap_md)
    grid_df = run_sensitivity_analysis(df, args, args.out_dir)
    plot_pareto_frontier(metrics_df, grid_df, args.out_dir)
    plot_radar_comparison(metrics_df, week_df, args.out_dir)
    plot_weight_transition(args, args.out_dir)
    compute_whatif_scenario(full_df, args.out_dir)
    stage_analysis(week_df, args.out_dir)


def generate_summary_report(metrics_df: pd.DataFrame, full_df: pd.DataFrame, 
                            bobby_rows: List[Dict], out_dir: str) -> None:
    """Generate interpretation report with overall statistics."""
    
    # Compute overall metrics (separate for judge_save and non-judge_save seasons)
    js_seasons = metrics_df[metrics_df["judge_save"] == True]
    no_js_seasons = metrics_df[metrics_df["judge_save"] == False]
    
    lines = [
        "# AMEI 2.0 Simulation Results Summary",
        "",
        "## Overall Performance Comparison",
        "",
    ]
    
    # EMR for seasons without judge save
    if len(no_js_seasons) > 0:
        emr_amei = no_js_seasons["emr_amei"].mean()
        emr_cbr = no_js_seasons["emr_cbr"].mean()
        emr_cbp = no_js_seasons["emr_cbp"].mean()
        intercept_amei = no_js_seasons["interception_mean_amei"].mean()
        intercept_cbr = no_js_seasons["interception_mean_cbr"].mean()
        intercept_cbp = no_js_seasons["interception_mean_cbp"].mean()
        lines.extend([
            "### Elimination Match Rate (EMR) - Seasons without Judge Save",
            "",
            f"| Method | EMR | Improvement vs CBR |",
            f"|--------|-----|--------------------|",
            f"| AMEI 2.0 | {emr_amei:.3f} | baseline |",
            f"| CBR | {emr_cbr:.3f} | {(emr_amei - emr_cbr)*100:+.1f}% |",
            f"| CBP | {emr_cbp:.3f} | {(emr_amei - emr_cbp)*100:+.1f}% |",
            "",
            "### Controversy Interception Rate (No Judge Save)",
            "",
            f"| Method | Interception Rate |",
            f"|--------|-------------------|",
            f"| AMEI 2.0 | {intercept_amei:.3f} |",
            f"| CBR | {intercept_cbr:.3f} |",
            f"| CBP | {intercept_cbp:.3f} |",
            "",
        ])
    
    # B2CR for seasons with judge save
    if len(js_seasons) > 0:
        b2cr_amei = js_seasons["b2cr_amei"].mean()
        b2cr_cbr = js_seasons["b2cr_cbr"].mean()
        b2cr_cbp = js_seasons["b2cr_cbp"].mean()
        intercept_amei = js_seasons["interception_mean_amei"].mean()
        intercept_cbr = js_seasons["interception_mean_cbr"].mean()
        intercept_cbp = js_seasons["interception_mean_cbp"].mean()
        lines.extend([
            "### Bottom-2 Capture Rate (B2CR) - Seasons with Judge Save",
            "",
            f"| Method | B2CR | Improvement vs CBR |",
            f"|--------|------|--------------------|",
            f"| AMEI 2.0 | {b2cr_amei:.3f} | baseline |",
            f"| CBR | {b2cr_cbr:.3f} | {(b2cr_amei - b2cr_cbr)*100:+.1f}% |",
            f"| CBP | {b2cr_cbp:.3f} | {(b2cr_amei - b2cr_cbp)*100:+.1f}% |",
            "",
            "### Controversy Interception Rate (Judge Save)",
            "",
            f"| Method | Interception Rate |",
            f"|--------|-------------------|",
            f"| AMEI 2.0 | {intercept_amei:.3f} |",
            f"| CBR | {intercept_cbr:.3f} |",
            f"| CBP | {intercept_cbp:.3f} |",
            "",
        ])
    
    # Entropy analysis
    entropy_mean_amei = metrics_df["entropy_mean_amei"].mean()
    entropy_mean_cbr = metrics_df["entropy_mean_cbr"].mean()
    entropy_mean_cbp = metrics_df["entropy_mean_cbp"].mean()
    entropy_std_amei = metrics_df["entropy_mean_amei"].std()
    fairness_amei = metrics_df["fairness_loss_mean_amei"].mean()
    fairness_cbr = metrics_df["fairness_loss_mean_cbr"].mean()
    fairness_cbp = metrics_df["fairness_loss_mean_cbp"].mean()
    excite_amei = metrics_df["excitement_loss_mean_amei"].mean()
    excite_cbr = metrics_df["excitement_loss_mean_cbr"].mean()
    excite_cbp = metrics_df["excitement_loss_mean_cbp"].mean()
    lines.extend([
        "## Uncertainty Analysis",
        "",
        f"- AMEI entropy: {entropy_mean_amei:.3f} ± {entropy_std_amei:.3f}",
        f"- CBR entropy: {entropy_mean_cbr:.3f}",
        f"- CBP entropy: {entropy_mean_cbp:.3f}",
        f"- Higher entropy indicates more balanced competition",
        "",
        "## Fairness & Excitement Loss",
        "",
        f"- Fairness loss (AMEI/CBR/CBP): {fairness_amei:.2f} / {fairness_cbr:.2f} / {fairness_cbp:.2f}",
        f"- Excitement loss (AMEI/CBR/CBP): {excite_amei:.3f} / {excite_cbr:.3f} / {excite_cbp:.3f}",
        "",
    ])
    
    # Bobby Bones case study
    if bobby_rows:
        bobby_df = pd.DataFrame.from_records(bobby_rows)
        # Find Bobby's data (typically contestant with lowest J_total but high p_hat)
        finale_week = bobby_df["week"].max()
        finale_data = bobby_df[bobby_df["week"] == finale_week]
        
        lines.extend([
            "## Bobby Bones Case Study (Season 27)",
            "",
            "### Finale Analysis",
            "",
            "| Contestant | J_total | p_hat | z_score | kappa | AMEI Score | Rank |",
            "|------------|---------|-------|---------|-------|------------|------|",
        ])
        
        for _, row in finale_data.sort_values("amei_rank").iterrows():
            lines.append(
                f"| {row['celebrity_name']} | {row['J_total']:.1f} | {row['p_hat']:.3f} | "
                f"{row['z_score']:.2f} | {row['kappa']:.3f} | {row['amei_score']:.4f} | {int(row['amei_rank'])} |"
            )
        
        # Calculate Bobby's trajectory
        bobby_data = bobby_df[bobby_df["celebrity_name"].str.contains("Bobby", case=False, na=False)]
        if len(bobby_data) > 0:
            avg_z = bobby_data["z_score"].mean()
            avg_kappa = bobby_data["kappa"].mean()
            bottom2_count = bobby_data["bottom2_flag"].sum()
            total_weeks = len(bobby_data)
            
            lines.extend([
                "",
                "### Bobby Bones Season Summary",
                "",
                f"- Average z-score: {avg_z:.2f} (negative indicates below-median judge scores)",
                f"- Average kappa (vote weight): {avg_kappa:.3f}",
                f"- Times in AMEI bottom-2: {int(bottom2_count)}/{total_weeks} weeks",
                f"- Under AMEI 2.0, low technical merit would have been penalized via reduced vote weight",
                "",
            ])
    
    # Key insights
    lines.extend([
        "## Key Insights",
        "",
        "1. **Merit-Weighted Voting**: AMEI 2.0 applies kappa gating to reduce vote weight for contestants with poor technical scores",
        "2. **Dynamic Balance**: Judge weight decreases from 65% to 35% as season progresses, giving audience more influence in later rounds",
        "3. **Soft Penalty**: Rather than hard cutoffs, the hinge gate provides gradual penalty for below-threshold performance",
        "",
        "## Interpretation Notes",
        "",
        "- AMEI is designed to alter elimination logic rather than replicate historical outcomes, so a lower EMR vs CBR is expected and not a defect.",
        "- The key target is higher controversy interception and improved fairness–excitement balance.",
        "",
        "## Additional Outputs",
        "",
        "- sensitivity_grid.csv: grid search results for z_soft/z_hard",
        "- sensitivity_heatmap_nojs.png: EMR heatmap for seasons without Judge Save",
        "- sensitivity_heatmap_js.png: B2CR heatmap for seasons with Judge Save",
        "- pareto_nojs.png / pareto_js.png: fairness–excitement trade-off plots",
        "- bobby_trajectory.png: Bobby Bones z-score/kappa/rank trend",
        "- mcnemar_summary.md: McNemar significance tests",
        "- radar_comparison.png: radar chart across methods",
        "- weight_transition.png: logistic weight transition curve",
        "- whatif_bobby_bones.csv / whatif_summary.md: Season 27 what-if analysis",
        "- bootstrap_ci.csv: bootstrap confidence intervals",
        "- stage_analysis.csv / stage_comparison.png: stage-wise performance",
        "",
    ])
    
    report_path = os.path.join(out_dir, "interpretation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"[INFO] Summary report saved to: {report_path}")


def write_mcnemar_summary(week_df: pd.DataFrame, out_path: str) -> None:
    """Write McNemar test summary comparing AMEI with CBR/CBP."""
    lines = [
        "## Statistical Significance (McNemar)",
        "",
    ]

    def mcnemar_for_group(df_group: pd.DataFrame, label: str) -> None:
        if df_group.empty:
            lines.extend([f"- {label}: no data", ""])
            return
        for method in ["cbr", "cbp"]:
            a = df_group["amei_hit"].astype(int).to_numpy()
            b = df_group[f"{method}_hit"].astype(int).to_numpy()
            n01 = int(np.sum((a == 0) & (b == 1)))
            n10 = int(np.sum((a == 1) & (b == 0)))
            table = [[0, n01], [n10, 0]]
            try:
                res = mcnemar(table, exact=False, correction=True)
                lines.append(
                    f"- {label} AMEI vs {method.upper()}: n01={n01}, n10={n10}, p={res.pvalue:.4f}"
                )
            except Exception:
                lines.append(f"- {label} AMEI vs {method.upper()}: test failed")
        lines.append("")

    mcnemar_for_group(week_df[week_df["judge_save"] == False], "No Judge Save")
    mcnemar_for_group(week_df[week_df["judge_save"] == True], "Judge Save")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def append_markdown(out_dir: str, base_name: str, append_path: str) -> None:
    base_path = os.path.join(out_dir, base_name)
    if not os.path.exists(base_path) or not os.path.exists(append_path):
        return
    with open(base_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write(open(append_path, "r", encoding="utf-8").read())


if __name__ == "__main__":
    main()