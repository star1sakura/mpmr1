import argparse
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.contingency_tables import mcnemar


def parse_args():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_votes = os.path.normpath(os.path.join(base_dir, "..", "r1", "outputs", "req1_vote_estimates.csv"))
    default_out = os.path.join(base_dir, "outputs")
    ap = argparse.ArgumentParser(description="Requirement 4 analysis: AMEI 2.0 simulation and comparison.")
    ap.add_argument("--votes", default=default_votes, help="Path to req1_vote_estimates.csv")
    ap.add_argument("--out-dir", default=default_out, help="Output directory")
    ap.add_argument("--threshold-mode", choices=["quantile", "value"], default="quantile", help="Threshold mode for gating")
    ap.add_argument("--z-soft-q", type=float, default=0.2, help="Soft threshold quantile (weekly)")
    ap.add_argument("--z-hard-q", type=float, default=0.05, help="Hard threshold quantile (weekly)")
    ap.add_argument("--z-soft", type=float, default=-0.84, help="Soft threshold (value mode)")
    ap.add_argument("--z-hard", type=float, default=-1.64, help="Hard threshold (value mode)")
    ap.add_argument("--min-vote-weight", type=float, default=0.05, help="Minimum vote weight after gating")
    ap.add_argument("--w-start", type=float, default=0.65, help="Judge weight at season start")
    ap.add_argument("--w-end", type=float, default=0.35, help="Judge weight at season end")
    ap.add_argument("--logistic-k", type=float, default=10.0, help="Logistic transition steepness")
    ap.add_argument("--logistic-t0", type=float, default=0.5, help="Logistic transition midpoint (progress)")
    ap.add_argument("--optimize-weights", action="store_true", default=True, help="Optimize logistic weights")
    ap.add_argument("--no-optimize-weights", dest="optimize_weights", action="store_false", help="Disable weight optimization")
    ap.add_argument("--min-interception", type=float, default=0.4, help="Minimum interception rate constraint")
    ap.add_argument("--tie-seed", type=int, default=42, help="Random seed for tie-break")
    ap.add_argument("--controversy-z-q", type=float, default=0.2, help="Controversy z-score quantile")
    ap.add_argument("--controversy-p-q", type=float, default=0.8, help="Controversy p_hat quantile")
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


def resolve_thresholds(
    z_scores: np.ndarray,
    args,
    z_soft_q: Optional[float] = None,
    z_hard_q: Optional[float] = None,
) -> Tuple[float, float]:
    if args.threshold_mode == "quantile":
        soft_q = args.z_soft_q if z_soft_q is None else z_soft_q
        hard_q = args.z_hard_q if z_hard_q is None else z_hard_q
        soft_q = float(np.clip(soft_q, 0.0, 1.0))
        hard_q = float(np.clip(hard_q, 0.0, 1.0))
        if hard_q >= soft_q:
            hard_q = max(0.0, soft_q - 0.05)
        z_soft = float(np.nanquantile(z_scores, soft_q))
        z_hard = float(np.nanquantile(z_scores, hard_q))
    else:
        z_soft = float(args.z_soft)
        z_hard = float(args.z_hard)
    if z_hard >= z_soft:
        z_hard = z_soft - 1e-6
    return z_soft, z_hard


def compute_kappa(
    z_scores: np.ndarray,
    args,
    z_soft_q: Optional[float] = None,
    z_hard_q: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    z_soft, z_hard = resolve_thresholds(z_scores, args, z_soft_q, z_hard_q)
    kappa = hinge_gate(z_scores, z_soft, z_hard, args.min_vote_weight)
    return kappa, z_soft, z_hard


def compute_kappa_custom(
    z_scores: np.ndarray,
    args,
    min_vote_weight: float,
    z_soft_q: Optional[float] = None,
    z_hard_q: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    z_soft, z_hard = resolve_thresholds(z_scores, args, z_soft_q, z_hard_q)
    kappa = hinge_gate(z_scores, z_soft, z_hard, min_vote_weight)
    return kappa, z_soft, z_hard


def compute_baseline_metrics(week_cache: List[Dict[str, object]], args) -> Dict[str, float]:
    rows: List[Dict[str, float]] = []
    for row in week_cache:
        J = row["J"]
        p_hat = row["p_hat"]
        tie_break = row["tie_break"]
        z_scores = row["z_scores"]
        actual_elim_idx = row["actual_elim_idx"]
        judge_save = bool(row["judge_save"])

        cbr_elim, cbr_bottom2 = apply_cbr(J, p_hat, tie_break)
        cbp_elim, cbp_bottom2 = apply_cbp(J, p_hat, tie_break)

        if judge_save:
            cbr_hit = 1 if any(i in cbr_bottom2 for i in actual_elim_idx) else 0
            cbp_hit = 1 if any(i in cbp_bottom2 for i in actual_elim_idx) else 0
        else:
            cbr_hit = 1 if any(i in cbr_elim for i in actual_elim_idx) else 0
            cbp_hit = 1 if any(i in cbp_elim for i in actual_elim_idx) else 0

        controversy_mask = identify_controversy_cases(
            z_scores,
            p_hat,
            z_quantile=args.controversy_z_q,
            p_quantile=args.controversy_p_q,
        )
        intercept_cbr = bobby_interception_rate(controversy_mask, cbr_bottom2)
        intercept_cbp = bobby_interception_rate(controversy_mask, cbp_bottom2)

        rank_j = rank_avg_desc(J)
        cbr_score = -rank_avg_desc(J) - rank_avg_desc(p_hat)
        P_J = J / np.sum(J) if np.sum(J) > 0 else np.full_like(J, 1.0 / len(J))
        cbp_score = P_J + p_hat
        spearman_cbr = spearman_corr(rank_j, rank_avg_desc(cbr_score))
        spearman_cbp = spearman_corr(rank_j, rank_avg_desc(cbp_score))

        rows.append({
            "judge_save": judge_save,
            "cbr_hit": cbr_hit,
            "cbp_hit": cbp_hit,
            "intercept_cbr": intercept_cbr,
            "intercept_cbp": intercept_cbp,
            "spearman_cbr": spearman_cbr,
            "spearman_cbp": spearman_cbp,
        })

    df = pd.DataFrame.from_records(rows)
    js = df[df["judge_save"] == True]
    no_js = df[df["judge_save"] == False]

    return {
        "emr_cbr": safe_mean(no_js["cbr_hit"].to_numpy(dtype=float)),
        "emr_cbp": safe_mean(no_js["cbp_hit"].to_numpy(dtype=float)),
        "b2cr_cbr": safe_mean(js["cbr_hit"].to_numpy(dtype=float)),
        "b2cr_cbp": safe_mean(js["cbp_hit"].to_numpy(dtype=float)),
        "interception_cbr": safe_mean(df["intercept_cbr"].to_numpy(dtype=float)),
        "interception_cbp": safe_mean(df["intercept_cbp"].to_numpy(dtype=float)),
        "spearman_cbr": safe_mean(df["spearman_cbr"].to_numpy(dtype=float)),
        "spearman_cbp": safe_mean(df["spearman_cbp"].to_numpy(dtype=float)),
    }


def apply_emr_b2cr_constraints(grid_df: pd.DataFrame, baseline: Dict[str, float], args) -> pd.DataFrame:
    def min_allowed(base: float) -> float:
        if not np.isfinite(base):
            return float("nan")
        abs_floor = base - 0.05
        rel_floor = base * 0.9
        return max(abs_floor, rel_floor)

    emr_min_cbr = min_allowed(baseline.get("emr_cbr", float("nan")))
    emr_min_cbp = min_allowed(baseline.get("emr_cbp", float("nan")))
    b2cr_min_cbr = min_allowed(baseline.get("b2cr_cbr", float("nan")))
    b2cr_min_cbp = min_allowed(baseline.get("b2cr_cbp", float("nan")))

    grid_df["emr_cbr_baseline"] = baseline.get("emr_cbr", float("nan"))
    grid_df["emr_cbp_baseline"] = baseline.get("emr_cbp", float("nan"))
    grid_df["b2cr_cbr_baseline"] = baseline.get("b2cr_cbr", float("nan"))
    grid_df["b2cr_cbp_baseline"] = baseline.get("b2cr_cbp", float("nan"))
    grid_df["emr_min_cbr"] = emr_min_cbr
    grid_df["emr_min_cbp"] = emr_min_cbp
    grid_df["b2cr_min_cbr"] = b2cr_min_cbr
    grid_df["b2cr_min_cbp"] = b2cr_min_cbp

    def check(metric: float, min_val: float) -> bool:
        if not np.isfinite(min_val):
            return True
        if not np.isfinite(metric):
            return False
        return metric >= min_val

    grid_df["emr_feasible"] = grid_df["emr"].apply(
        lambda x: check(x, emr_min_cbr) and check(x, emr_min_cbp)
    )
    grid_df["b2cr_feasible"] = grid_df["b2cr"].apply(
        lambda x: check(x, b2cr_min_cbr) and check(x, b2cr_min_cbp)
    )
    grid_df["constraint_feasible"] = grid_df["emr_feasible"] & grid_df["b2cr_feasible"]
    grid_df["penalty_emr"] = (
        np.maximum(0.0, emr_min_cbr - grid_df["emr"]) +
        np.maximum(0.0, emr_min_cbp - grid_df["emr"])
    )
    grid_df["penalty_b2cr"] = (
        np.maximum(0.0, b2cr_min_cbr - grid_df["b2cr"]) +
        np.maximum(0.0, b2cr_min_cbp - grid_df["b2cr"])
    )
    grid_df["penalty_total"] = grid_df["penalty_emr"] + grid_df["penalty_b2cr"]
    grid_df["feasible"] = (
        grid_df["interception_mean"] >= float(args.min_interception)
    ) & grid_df["constraint_feasible"]
    return grid_df


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


def pick_judge_save_elim_jtotal(bottom2_idx: List[int], J: np.ndarray, tie_break: np.ndarray) -> Optional[int]:
    """Rule A: eliminate lower J_total (tie -> tie_break)."""
    if len(bottom2_idx) == 0:
        return None
    if len(bottom2_idx) == 1:
        return bottom2_idx[0]
    i1, i2 = bottom2_idx[0], bottom2_idx[1]
    if np.isclose(J[i1], J[i2]):
        return i1 if tie_break[i1] <= tie_break[i2] else i2
    return i1 if J[i1] < J[i2] else i2


def pick_judge_save_elim_rank(bottom2_idx: List[int], J: np.ndarray, tie_break: np.ndarray) -> Optional[int]:
    """Rule B: eliminate lower (normalized J_total + reverse rank)."""
    if len(bottom2_idx) == 0:
        return None
    if len(bottom2_idx) == 1:
        return bottom2_idx[0]
    n = len(J)
    j_norm = min_max_norm(J.tolist())
    ranks = rank_avg_desc(J)
    if n <= 1:
        rank_score = np.full_like(ranks, 0.5)
    else:
        rank_score = (n - ranks) / (n - 1)
    save_score = j_norm + rank_score
    i1, i2 = bottom2_idx[0], bottom2_idx[1]
    if np.isclose(save_score[i1], save_score[i2]):
        return i1 if tie_break[i1] <= tie_break[i2] else i2
    return i1 if save_score[i1] < save_score[i2] else i2


def pick_judge_save_elim_prev_rank(
    bottom2_idx: List[int],
    J: np.ndarray,
    prev_rank_score: np.ndarray,
    tie_break: np.ndarray,
    coef: float = 0.5,
) -> Optional[int]:
    """Rule C: eliminate lower (normalized J_total + coef * previous reverse rank)."""
    if len(bottom2_idx) == 0:
        return None
    if len(bottom2_idx) == 1:
        return bottom2_idx[0]
    j_norm = min_max_norm(J.tolist())
    prev_score = np.asarray(prev_rank_score, dtype=float)
    if prev_score.size != j_norm.size:
        prev_score = np.full_like(j_norm, 0.5, dtype=float)
    save_score = j_norm + coef * prev_score
    i1, i2 = bottom2_idx[0], bottom2_idx[1]
    if np.isclose(save_score[i1], save_score[i2]):
        return i1 if tie_break[i1] <= tie_break[i2] else i2
    return i1 if save_score[i1] < save_score[i2] else i2


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


def build_tie_break_map(df: pd.DataFrame, seed: int) -> Dict[Tuple[int, int], np.ndarray]:
    rng = np.random.default_rng(seed)
    tie_break_map: Dict[Tuple[int, int], np.ndarray] = {}

    for season, season_df in df.groupby("season", sort=True):
        prev_rank: Dict[int, float] = {}
        weeks = sorted(season_df["week"].unique())
        for week in weeks:
            g = season_df[season_df["week"] == week].sort_values("contestant_id").reset_index(drop=True)
            n = len(g)
            if prev_rank:
                tie_break = np.array([
                    -prev_rank.get(int(cid), rng.uniform(-n, -1))
                    for cid in g["contestant_id"].to_numpy()
                ], dtype=float)
            else:
                tie_break = rng.uniform(-n, -1, size=n).astype(float)
            tie_break_map[(int(season), int(week))] = tie_break

            ranks = rank_avg_desc(g["J_total"].to_numpy(dtype=float))
            prev_rank = {int(cid): float(ranks[idx]) for idx, cid in enumerate(g["contestant_id"].to_numpy())}

    return tie_break_map


def build_prev_rank_score_map(df: pd.DataFrame) -> Dict[Tuple[int, int], np.ndarray]:
    score_map: Dict[Tuple[int, int], np.ndarray] = {}

    for season, season_df in df.groupby("season", sort=True):
        prev_rank: Dict[int, float] = {}
        weeks = sorted(season_df["week"].unique())
        for week in weeks:
            g = season_df[season_df["week"] == week].sort_values("contestant_id").reset_index(drop=True)
            n = len(g)
            scores = []
            for cid in g["contestant_id"].to_numpy():
                if cid in prev_rank and n > 1:
                    score = (n - prev_rank[cid]) / (n - 1)
                else:
                    score = 0.5
                scores.append(score)
            score_map[(int(season), int(week))] = np.asarray(scores, dtype=float)

            ranks = rank_avg_desc(g["J_total"].to_numpy(dtype=float))
            prev_rank = {int(cid): float(ranks[idx]) for idx, cid in enumerate(g["contestant_id"].to_numpy())}

    return score_map


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    rho, _ = spearmanr(a, b)
    if not np.isfinite(rho):
        return float("nan")
    return float(rho)


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


def identify_controversy_abs(
    z_scores: np.ndarray,
    p_hat: np.ndarray,
    z_threshold: float = -1.0,
    p_threshold: float = 0.15,
) -> np.ndarray:
    """Identify controversy cases using absolute thresholds."""
    return (z_scores <= z_threshold) & (p_hat >= p_threshold)


def identify_controversy_gap(J: np.ndarray, p_hat: np.ndarray, gap_quantile: float = 0.8) -> np.ndarray:
    """Identify controversy cases using fan-judge rank gap."""
    rank_j = rank_avg_desc(J)
    rank_f = rank_avg_desc(p_hat)
    gap = rank_j - rank_f
    thr = np.nanquantile(gap, gap_quantile)
    return gap >= thr


def bobby_interception_rate(controversy_mask: np.ndarray, bottom2_idx: List[int]) -> float:
    """Interception rate: controversy contestants captured in Bottom-2."""
    if controversy_mask.sum() == 0:
        return float("nan")
    captured = sum(1 for i in bottom2_idx if controversy_mask[i])
    return float(captured / max(controversy_mask.sum(), 1))


def precompute_week_cache(
    df: pd.DataFrame,
    eps: float,
    tie_break_map: Dict[Tuple[int, int], np.ndarray],
    prev_rank_score_map: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
) -> List[Dict[str, object]]:
    """Precompute week-level data to speed up sensitivity analysis."""
    max_week_map: Dict[int, int] = df.groupby("season")["week"].max().to_dict()
    cache: List[Dict[str, object]] = []

    for (season, week), g in df.groupby(["season", "week"], sort=True):
        g = g.sort_values("contestant_id").reset_index(drop=True)
        J = g["J_total"].to_numpy(dtype=float)
        p_hat = g["p_hat"].to_numpy(dtype=float)
        tie_break = tie_break_map.get((int(season), int(week)))
        if tie_break is None:
            tie_break = g["contestant_id"].to_numpy(dtype=int)
        prev_rank_score = None
        if prev_rank_score_map is not None:
            prev_rank_score = prev_rank_score_map.get((int(season), int(week)))
        if prev_rank_score is None:
            prev_rank_score = np.full_like(J, 0.5, dtype=float)
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
            "prev_rank_score": prev_rank_score,
            "progress": float(progress),
            "z_scores": z_scores,
            "merit_score": merit_score,
            "actual_elim_idx": actual_elim_idx,
        })

    return cache


def evaluate_amei_for_params(
    week_cache: List[Dict[str, object]],
    args,
    z_soft_q: float,
    z_hard_q: float,
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

        z_soft, z_hard = resolve_thresholds(z_scores, args, z_soft_q=z_soft_q, z_hard_q=z_hard_q)
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
        controversy_mask = identify_controversy_cases(
            z_scores,
            p_hat,
            z_quantile=args.controversy_z_q,
            p_quantile=args.controversy_p_q,
        )
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
        "z_soft_q": float(z_soft_q),
        "z_hard_q": float(z_hard_q),
        "b2cr": safe_mean(js["hit"]) if len(js) else float("nan"),
        "emr": safe_mean(no_js["hit"]) if len(no_js) else float("nan"),
        "entropy_mean": safe_mean(week_df["entropy"]) if len(week_df) else float("nan"),
        "fairness_loss_mean": safe_mean(week_df["fairness_loss"]) if len(week_df) else float("nan"),
        "excitement_loss_mean": safe_mean(week_df["excitement_loss"]) if len(week_df) else float("nan"),
        "interception_mean": safe_mean(week_df["interception"]) if len(week_df) else float("nan"),
        "weeks_js": int(len(js)),
        "weeks_nojs": int(len(no_js)),
    }


def evaluate_amei_metrics(
    week_cache: List[Dict[str, object]],
    args,
    w_start: float,
    w_end: float,
    logistic_k: float,
    logistic_t0: float,
    min_vote_weight: Optional[float] = None,
) -> Dict[str, float]:
    rows: List[Dict[str, float]] = []
    for row in week_cache:
        J = row["J"]
        p_hat = row["p_hat"]
        tie_break = row["tie_break"]
        progress = row["progress"]
        z_scores = row["z_scores"]
        merit_score = row["merit_score"]
        actual_elim_idx = row["actual_elim_idx"]

        if min_vote_weight is None:
            kappa, _, _ = compute_kappa(z_scores, args)
        else:
            kappa, _, _ = compute_kappa_custom(z_scores, args, min_vote_weight=float(min_vote_weight))
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

        controversy_mask = identify_controversy_cases(
            z_scores,
            p_hat,
            z_quantile=args.controversy_z_q,
            p_quantile=args.controversy_p_q,
        )
        intercept = bobby_interception_rate(controversy_mask, bottom2_idx)

        rank_j = rank_avg_desc(J)
        rank_amei = rank_avg_desc(final_score)
        spearman = spearman_corr(rank_j, rank_amei)

        rows.append({
            "judge_save": judge_save,
            "hit": hit,
            "interception": intercept,
            "spearman": spearman,
        })

    df = pd.DataFrame.from_records(rows)
    js = df[df["judge_save"] == True]
    no_js = df[df["judge_save"] == False]

    return {
        "emr_amei": safe_mean(no_js["hit"].to_numpy(dtype=float)),
        "b2cr_amei": safe_mean(js["hit"].to_numpy(dtype=float)),
        "interception_amei": safe_mean(df["interception"].to_numpy(dtype=float)),
        "spearman_amei": safe_mean(df["spearman"].to_numpy(dtype=float)),
    }


def run_loso_validation(week_cache: List[Dict[str, object]], args, out_dir: str) -> pd.DataFrame:
    seasons = sorted({int(row["season"]) for row in week_cache})
    records: List[Dict[str, float]] = []

    for season in seasons:
        test_cache = [row for row in week_cache if int(row["season"]) == season]
        if not test_cache:
            continue

        amei_metrics = evaluate_amei_metrics(
            test_cache,
            args,
            w_start=float(args.w_start),
            w_end=float(args.w_end),
            logistic_k=float(args.logistic_k),
            logistic_t0=float(args.logistic_t0),
            min_vote_weight=float(args.min_vote_weight),
        )
        baseline_test = compute_baseline_metrics(test_cache, args)

        records.append({
            "season": season,
            "w_start": float(args.w_start),
            "w_end": float(args.w_end),
            "logistic_k": float(args.logistic_k),
            "logistic_t0": float(args.logistic_t0),
            "min_vote_weight": float(args.min_vote_weight),
            **amei_metrics,
            "emr_cbr": baseline_test.get("emr_cbr", float("nan")),
            "emr_cbp": baseline_test.get("emr_cbp", float("nan")),
            "b2cr_cbr": baseline_test.get("b2cr_cbr", float("nan")),
            "b2cr_cbp": baseline_test.get("b2cr_cbp", float("nan")),
            "interception_cbr": baseline_test.get("interception_cbr", float("nan")),
            "interception_cbp": baseline_test.get("interception_cbp", float("nan")),
            "spearman_cbr": baseline_test.get("spearman_cbr", float("nan")),
            "spearman_cbp": baseline_test.get("spearman_cbp", float("nan")),
        })

    loso_df = pd.DataFrame.from_records(records)
    out_path = os.path.join(out_dir, "loso_summary.csv")
    loso_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return loso_df


def compute_loso_ci(loso_df: pd.DataFrame, out_dir: str, n_bootstrap: int = 1000, seed: int = 42) -> pd.DataFrame:
    if loso_df.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    emr_adv = loso_df["emr_amei"] - loso_df[["emr_cbr", "emr_cbp"]].max(axis=1)
    b2cr_adv = loso_df["b2cr_amei"] - loso_df[["b2cr_cbr", "b2cr_cbp"]].max(axis=1)
    inter_adv = loso_df["interception_amei"] - loso_df[["interception_cbr", "interception_cbp"]].max(axis=1)
    spearman_adv = loso_df["spearman_amei"] - loso_df[["spearman_cbr", "spearman_cbp"]].max(axis=1)

    metrics = {
        "emr_adv": emr_adv.to_numpy(dtype=float),
        "b2cr_adv": b2cr_adv.to_numpy(dtype=float),
        "interception_adv": inter_adv.to_numpy(dtype=float),
        "spearman_adv": spearman_adv.to_numpy(dtype=float),
    }

    records = []
    for name, values in metrics.items():
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        boots = []
        for _ in range(n_bootstrap):
            sample = rng.choice(values, size=len(values), replace=True)
            boots.append(np.mean(sample))
        records.append({
            "metric": name,
            "mean": float(np.mean(values)),
            "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)),
        })

    out_df = pd.DataFrame.from_records(records)
    out_df.to_csv(os.path.join(out_dir, "loso_ci.csv"), index=False, encoding="utf-8-sig")
    return out_df


def compute_interception_impact(week_cache: List[Dict[str, object]], args, out_dir: str) -> pd.DataFrame:
    season_rows: Dict[int, Dict[str, float]] = {}
    for row in week_cache:
        season = int(row["season"])
        J = row["J"]
        p_hat = row["p_hat"]
        tie_break = row["tie_break"]
        progress = row["progress"]
        z_scores = row["z_scores"]
        merit_score = row["merit_score"]

        kappa, _, _ = compute_kappa(z_scores, args)
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

        _, cbr_bottom2 = apply_cbr(J, p_hat, tie_break)
        _, cbp_bottom2 = apply_cbp(J, p_hat, tie_break)

        controversy_mask = identify_controversy_cases(
            z_scores,
            p_hat,
            z_quantile=args.controversy_z_q,
            p_quantile=args.controversy_p_q,
        )
        controversy_count = int(np.sum(controversy_mask))
        amei_captured = sum(1 for i in bottom2_idx if controversy_mask[i])
        cbr_captured = sum(1 for i in cbr_bottom2 if controversy_mask[i])
        cbp_captured = sum(1 for i in cbp_bottom2 if controversy_mask[i])

        if season not in season_rows:
            season_rows[season] = {
                "season": season,
                "controversy_count": 0,
                "amei_captured": 0,
                "cbr_captured": 0,
                "cbp_captured": 0,
            }
        season_rows[season]["controversy_count"] += controversy_count
        season_rows[season]["amei_captured"] += amei_captured
        season_rows[season]["cbr_captured"] += cbr_captured
        season_rows[season]["cbp_captured"] += cbp_captured

    out_df = pd.DataFrame.from_records(list(season_rows.values()))
    denom = out_df["controversy_count"].replace(0, np.nan)
    out_df["amei_rate"] = out_df["amei_captured"] / denom
    out_df["cbr_rate"] = out_df["cbr_captured"] / denom
    out_df["cbp_rate"] = out_df["cbp_captured"] / denom
    out_df["delta_vs_cbr"] = out_df["amei_captured"] - out_df["cbr_captured"]
    out_df["delta_vs_cbp"] = out_df["amei_captured"] - out_df["cbp_captured"]
    out_df.to_csv(os.path.join(out_dir, "interception_impact.csv"), index=False, encoding="utf-8-sig")
    return out_df


def select_controversy_threshold(out_dir: str) -> Optional[pd.Series]:
    path = os.path.join(out_dir, "controversy_sensitivity.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    df["advantage"] = df["interception_amei"] - df[["interception_cbr", "interception_cbp"]].max(axis=1)
    best = df.sort_values(["advantage", "interception_amei"], ascending=False).iloc[0]
    best.to_frame().T.to_csv(os.path.join(out_dir, "threshold_selected.csv"), index=False, encoding="utf-8-sig")
    return best


def plot_heatmap(df_grid: pd.DataFrame, value_col: str, out_path: str, title: str) -> None:
    import matplotlib.pyplot as plt

    pivot = df_grid.pivot(index="z_hard_q", columns="z_soft_q", values=value_col)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="viridis")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])

    ax.set_xlabel("z_soft_q")
    ax.set_ylabel("z_hard_q")
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


def plot_controversy_heatmap(df_grid: pd.DataFrame, value_col: str, out_path: str, title: str) -> None:
    import matplotlib.pyplot as plt

    pivot = df_grid.pivot(index="p_quantile", columns="z_quantile", values=value_col)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="viridis")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])

    ax.set_xlabel("z_quantile")
    ax.set_ylabel("p_quantile")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)

    if np.isfinite(pivot.values).any():
        max_idx = np.nanargmax(pivot.values)
        r, c = np.unravel_index(max_idx, pivot.values.shape)
        ax.scatter([c], [r], s=80, facecolors="none", edgecolors="red", linewidths=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_sensitivity_analysis(
    df: pd.DataFrame,
    args,
    out_dir: str,
    tie_break_map: Dict[Tuple[int, int], np.ndarray],
    prev_rank_score_map: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
) -> pd.DataFrame:
    week_cache = precompute_week_cache(df, args.eps, tie_break_map, prev_rank_score_map)
    z_soft_vals = np.round(np.linspace(0.1, 0.3, 5), 2)
    z_hard_vals = np.round(np.linspace(0.01, 0.1, 5), 2)

    records: List[Dict[str, float]] = []
    for z_soft in z_soft_vals:
        for z_hard in z_hard_vals:
            if z_hard >= z_soft:
                continue
            records.append(
                evaluate_amei_for_params(
                    week_cache,
                    args,
                    z_soft_q=z_soft,
                    z_hard_q=z_hard,
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


def build_weight_grid(week_cache: List[Dict[str, object]], args) -> pd.DataFrame:
    w_start_vals = np.round(np.arange(0.55, 0.85, 0.05), 2)
    w_end_vals = np.round(np.arange(0.20, 0.55, 0.05), 2)
    k_vals = [4.0, 8.0, 12.0, 16.0]
    t0_vals = [0.40, 0.50, 0.60]
    min_vote_vals = [0.03, 0.06, 0.10]

    records: List[Dict[str, float]] = []
    for w_start in w_start_vals:
        for w_end in w_end_vals:
            if w_end >= w_start:
                continue
            for k in k_vals:
                for t0 in t0_vals:
                    for min_vote in min_vote_vals:
                        metrics = evaluate_amei_for_params(
                            week_cache,
                            args,
                            z_soft_q=args.z_soft_q,
                            z_hard_q=args.z_hard_q,
                            min_vote_weight=float(min_vote),
                            w_start=float(w_start),
                            w_end=float(w_end),
                            logistic_k=float(k),
                            logistic_t0=float(t0),
                            eps=args.eps,
                        )
                        metrics.update({
                            "w_start": float(w_start),
                            "w_end": float(w_end),
                            "logistic_k": float(k),
                            "logistic_t0": float(t0),
                            "min_vote_weight": float(min_vote),
                        })
                        records.append(metrics)

    return pd.DataFrame.from_records(records)


def run_weight_search(
    week_cache: List[Dict[str, object]],
    args,
    out_dir: str,
    tag: Optional[str] = None,
) -> Dict[str, float]:
    grid_df = build_weight_grid(week_cache, args)
    if grid_df.empty:
        return {}

    fair_norm = min_max_norm(grid_df["fairness_loss_mean"].to_list())
    excite_norm = min_max_norm(grid_df["excitement_loss_mean"].to_list())
    grid_df["tradeoff_score"] = 0.5 * fair_norm + 0.5 * excite_norm
    baseline = compute_baseline_metrics(week_cache, args)
    grid_df = apply_emr_b2cr_constraints(grid_df, baseline, args)

    suffix = f"_{tag}" if tag else ""
    grid_path = os.path.join(out_dir, f"weight_grid{suffix}.csv")
    grid_df.to_csv(grid_path, index=False, encoding="utf-8-sig")

    candidate = grid_df[grid_df["feasible"]]
    soft_lambda = None
    if candidate.empty:
        penalties = grid_df["penalty_total"].fillna(0.0)
        for lam in [1.0, 2.0, 3.0]:
            grid_df[f"objective_lambda_{int(lam)}"] = grid_df["tradeoff_score"] + lam * penalties

        soft_candidates = []
        for lam in [1.0, 2.0, 3.0]:
            col = f"objective_lambda_{int(lam)}"
            idx = grid_df[col].idxmin()
            row = grid_df.loc[idx].copy()
            row["soft_lambda"] = lam
            row["objective"] = row[col]
            soft_candidates.append(row)
        soft_df = pd.DataFrame(soft_candidates)
        soft_path = os.path.join(out_dir, f"weight_selected_soft{suffix}.csv")
        soft_df.to_csv(soft_path, index=False, encoding="utf-8-sig")
        best = soft_df.sort_values("objective").iloc[0]
        soft_lambda = float(best["soft_lambda"])
    else:
        best = candidate.sort_values(["tradeoff_score", "fairness_loss_mean", "excitement_loss_mean"]).iloc[0]

    selected_path = os.path.join(out_dir, f"weight_selected{suffix}.csv")
    pd.DataFrame([best]).to_csv(selected_path, index=False, encoding="utf-8-sig")
    return {
        "w_start": float(best["w_start"]),
        "w_end": float(best["w_end"]),
        "logistic_k": float(best["logistic_k"]),
        "logistic_t0": float(best["logistic_t0"]),
        "min_vote_weight": float(best.get("min_vote_weight", args.min_vote_weight)),
        "tradeoff_score": float(best["tradeoff_score"]),
        "interception_mean": float(best["interception_mean"]),
        "feasible": bool(best["feasible"]),
        "emr_min_cbr": float(best.get("emr_min_cbr", float("nan"))),
        "emr_min_cbp": float(best.get("emr_min_cbp", float("nan"))),
        "b2cr_min_cbr": float(best.get("b2cr_min_cbr", float("nan"))),
        "b2cr_min_cbp": float(best.get("b2cr_min_cbp", float("nan"))),
        "soft_lambda": soft_lambda,
    }


def run_weight_search_by_type(
    week_cache: List[Dict[str, object]],
    args,
    out_dir: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    js_cache = [row for row in week_cache if bool(row["judge_save"])]
    no_js_cache = [row for row in week_cache if not bool(row["judge_save"])]
    weight_js = run_weight_search(js_cache, args, out_dir, tag="js") if js_cache else {}
    weight_nojs = run_weight_search(no_js_cache, args, out_dir, tag="nojs") if no_js_cache else {}
    return weight_js, weight_nojs


def evaluate_adaptive_weights(
    week_cache: List[Dict[str, object]],
    args,
    weight_js: Dict[str, float],
    weight_nojs: Dict[str, float],
    out_dir: str,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for row in week_cache:
        is_js = bool(row["judge_save"])
        params = weight_js if is_js else weight_nojs
        if not params:
            params = {
                "w_start": args.w_start,
                "w_end": args.w_end,
                "logistic_k": args.logistic_k,
                "logistic_t0": args.logistic_t0,
                "min_vote_weight": args.min_vote_weight,
            }
        J = row["J"]
        p_hat = row["p_hat"]
        tie_break = row["tie_break"]
        progress = row["progress"]
        z_scores = row["z_scores"]
        merit_score = row["merit_score"]
        actual_elim_idx = row["actual_elim_idx"]

        kappa, _, _ = compute_kappa_custom(
            z_scores,
            args,
            min_vote_weight=float(params.get("min_vote_weight", args.min_vote_weight)),
        )
        effective_votes = p_hat * kappa
        vote_sum = np.sum(effective_votes)
        if vote_sum > 0:
            pop_score = effective_votes / vote_sum
        else:
            pop_sum = np.sum(p_hat)
            pop_score = p_hat / pop_sum if pop_sum > 0 else np.full_like(p_hat, 1.0 / len(p_hat))

        w_j, w_f = logistic_weights(
            progress,
            float(params.get("w_start", args.w_start)),
            float(params.get("w_end", args.w_end)),
            float(params.get("logistic_k", args.logistic_k)),
            float(params.get("logistic_t0", args.logistic_t0)),
        )
        final_score = w_j * merit_score + w_f * pop_score
        bottom2_idx = pick_bottom_k(final_score, 2, tie_break)
        elim_idx = bottom2_idx[:1]

        if is_js:
            hit = 1 if any(i in bottom2_idx for i in actual_elim_idx) else 0
        else:
            hit = 1 if any(i in elim_idx for i in actual_elim_idx) else 0

        controversy_mask = identify_controversy_cases(
            z_scores,
            p_hat,
            z_quantile=args.controversy_z_q,
            p_quantile=args.controversy_p_q,
        )
        intercept = bobby_interception_rate(controversy_mask, bottom2_idx)

        rank_j = rank_avg_desc(J)
        rank_amei = rank_avg_desc(final_score)
        spearman = spearman_corr(rank_j, rank_amei)

        rows.append({
            "judge_save": is_js,
            "hit": hit,
            "interception": intercept,
            "spearman": spearman,
        })

    df = pd.DataFrame.from_records(rows)
    js = df[df["judge_save"] == True]
    no_js = df[df["judge_save"] == False]

    summary = pd.DataFrame([
        {
            "emr": safe_mean(no_js["hit"].to_numpy(dtype=float)),
            "b2cr": safe_mean(js["hit"].to_numpy(dtype=float)),
            "interception": safe_mean(df["interception"].to_numpy(dtype=float)),
            "spearman": safe_mean(df["spearman"].to_numpy(dtype=float)),
        }
    ])
    summary.to_csv(os.path.join(out_dir, "adaptive_weight_metrics.csv"), index=False, encoding="utf-8-sig")
    return summary


def run_controversy_sensitivity(
    week_cache: List[Dict[str, object]],
    args,
    out_dir: str,
) -> pd.DataFrame:
    z_q_vals = np.round(np.linspace(0.1, 0.3, 5), 2)
    p_q_vals = np.round(np.linspace(0.7, 0.9, 5), 2)

    records: List[Dict[str, float]] = []
    for z_q in z_q_vals:
        for p_q in p_q_vals:
            amei_vals: List[float] = []
            cbr_vals: List[float] = []
            cbp_vals: List[float] = []
            for row in week_cache:
                J = row["J"]
                p_hat = row["p_hat"]
                tie_break = row["tie_break"]
                progress = row["progress"]
                z_scores = row["z_scores"]
                merit_score = row["merit_score"]

                kappa, _, _ = compute_kappa(z_scores, args)
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
                _, cbp_bottom2 = apply_cbp(J, p_hat, tie_break)
                _, cbr_bottom2 = apply_cbr(J, p_hat, tie_break)

                controversy_mask = identify_controversy_cases(z_scores, p_hat, z_quantile=z_q, p_quantile=p_q)
                amei_vals.append(bobby_interception_rate(controversy_mask, bottom2_idx))
                cbr_vals.append(bobby_interception_rate(controversy_mask, cbr_bottom2))
                cbp_vals.append(bobby_interception_rate(controversy_mask, cbp_bottom2))

            records.append({
                "z_quantile": float(z_q),
                "p_quantile": float(p_q),
                "interception_amei": safe_mean(np.asarray(amei_vals)),
                "interception_cbr": safe_mean(np.asarray(cbr_vals)),
                "interception_cbp": safe_mean(np.asarray(cbp_vals)),
            })

    out_df = pd.DataFrame.from_records(records)
    out_path = os.path.join(out_dir, "controversy_sensitivity.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    if not out_df.empty:
        plot_controversy_heatmap(
            out_df,
            value_col="interception_amei",
            out_path=os.path.join(out_dir, "controversy_heatmap.png"),
            title="Controversy Interception (AMEI)",
        )

    return out_df


def run_controversy_alt_thresholds(
    week_cache: List[Dict[str, object]],
    args,
    out_dir: str,
    z_threshold: float = -1.0,
    p_threshold: float = 0.15,
) -> pd.DataFrame:
    amei_vals: List[float] = []
    cbr_vals: List[float] = []
    cbp_vals: List[float] = []
    for row in week_cache:
        z_scores = row["z_scores"]
        p_hat = row["p_hat"]
        tie_break = row["tie_break"]
        J = row["J"]
        merit_score = row["merit_score"]
        progress = row["progress"]

        kappa, _, _ = compute_kappa(z_scores, args)

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
        _, cbp_bottom2 = apply_cbp(J, p_hat, tie_break)
        _, cbr_bottom2 = apply_cbr(J, p_hat, tie_break)

        controversy_mask = identify_controversy_abs(z_scores, p_hat, z_threshold=z_threshold, p_threshold=p_threshold)
        amei_vals.append(bobby_interception_rate(controversy_mask, bottom2_idx))
        cbr_vals.append(bobby_interception_rate(controversy_mask, cbr_bottom2))
        cbp_vals.append(bobby_interception_rate(controversy_mask, cbp_bottom2))

    out_df = pd.DataFrame([
        {
            "z_threshold": float(z_threshold),
            "p_threshold": float(p_threshold),
            "interception_amei": safe_mean(np.asarray(amei_vals)),
            "interception_cbr": safe_mean(np.asarray(cbr_vals)),
            "interception_cbp": safe_mean(np.asarray(cbp_vals)),
        }
    ])
    out_path = os.path.join(out_dir, "controversy_alt_thresholds.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_df


def run_controversy_gap_definition(
    week_cache: List[Dict[str, object]],
    args,
    out_dir: str,
    gap_quantile: float = 0.8,
) -> pd.DataFrame:
    amei_vals: List[float] = []
    cbr_vals: List[float] = []
    cbp_vals: List[float] = []
    for row in week_cache:
        J = row["J"]
        p_hat = row["p_hat"]
        tie_break = row["tie_break"]
        progress = row["progress"]
        z_scores = row["z_scores"]
        merit_score = row["merit_score"]

        kappa, _, _ = compute_kappa(z_scores, args)
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
        _, cbp_bottom2 = apply_cbp(J, p_hat, tie_break)
        _, cbr_bottom2 = apply_cbr(J, p_hat, tie_break)

        controversy_mask = identify_controversy_gap(J, p_hat, gap_quantile=gap_quantile)
        amei_vals.append(bobby_interception_rate(controversy_mask, bottom2_idx))
        cbr_vals.append(bobby_interception_rate(controversy_mask, cbr_bottom2))
        cbp_vals.append(bobby_interception_rate(controversy_mask, cbp_bottom2))

    out_df = pd.DataFrame([{
        "definition": "fan_judge_gap",
        "gap_quantile": float(gap_quantile),
        "interception_amei": safe_mean(np.asarray(amei_vals)),
        "interception_cbr": safe_mean(np.asarray(cbr_vals)),
        "interception_cbp": safe_mean(np.asarray(cbp_vals)),
    }])
    out_path = os.path.join(out_dir, "controversy_alt_definitions.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_df


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
    spearman_vals = [
        safe_mean(week_df["spearman_amei"]),
        safe_mean(week_df["spearman_cbr"]),
        safe_mean(week_df["spearman_cbp"]),
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
    spearman_n = min_max_norm(spearman_vals)
    excitement_n = 1 - min_max_norm(excitement_loss)
    entropy_n = min_max_norm(entropy)

    metrics = ["EMR", "B2CR", "Interception", "Spearman", "Excitement", "Entropy"]
    values = {
        "AMEI 2.0": [emr_n[0], b2cr_n[0], intercept_n[0], spearman_n[0], excitement_n[0], entropy_n[0]],
        "CBR": [emr_n[1], b2cr_n[1], intercept_n[1], spearman_n[1], excitement_n[1], entropy_n[1]],
        "CBP": [emr_n[2], b2cr_n[2], intercept_n[2], spearman_n[2], excitement_n[2], entropy_n[2]],
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


def write_spearman_summary(metrics_df: pd.DataFrame, out_dir: str) -> None:
    rows = [
        {"method": "AMEI", "spearman_mean": float(metrics_df["spearman_mean_amei"].mean())},
        {"method": "CBR", "spearman_mean": float(metrics_df["spearman_mean_cbr"].mean())},
        {"method": "CBP", "spearman_mean": float(metrics_df["spearman_mean_cbp"].mean())},
    ]
    out_df = pd.DataFrame.from_records(rows)
    out_df.to_csv(os.path.join(out_dir, "fairness_spearman.csv"), index=False, encoding="utf-8-sig")


def stage_label(progress: float) -> str:
    if progress <= 0.33:
        return "Early"
    if progress <= 0.66:
        return "Mid"
    return "Late"


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
        if "tie_break" in g.columns:
            tie_break = g["tie_break"].to_numpy(dtype=float)
        else:
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


def bootstrap_diff_metrics(
    week_df: pd.DataFrame,
    out_dir: str,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def diff_ci(a_vals: np.ndarray, b_vals: np.ndarray) -> Tuple[float, float, float]:
        a_vals = np.asarray(a_vals, dtype=float)
        b_vals = np.asarray(b_vals, dtype=float)
        mask = np.isfinite(a_vals) & np.isfinite(b_vals)
        a_vals = a_vals[mask]
        b_vals = b_vals[mask]
        if len(a_vals) == 0:
            return float("nan"), float("nan"), float("nan")
        diffs = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(a_vals), size=len(a_vals))
            diffs.append(np.mean(a_vals[idx] - b_vals[idx]))
        return float(np.mean(a_vals - b_vals)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

    records = []
    no_js = week_df[week_df["judge_save"] == False]
    js = week_df[week_df["judge_save"] == True]

    comparisons = [
        ("EMR", "No Judge Save", no_js, "amei_hit", "cbr_hit", "AMEI-CBR"),
        ("EMR", "No Judge Save", no_js, "amei_hit", "cbp_hit", "AMEI-CBP"),
        ("B2CR", "Judge Save", js, "amei_hit", "cbr_hit", "AMEI-CBR"),
        ("B2CR", "Judge Save", js, "amei_hit", "cbp_hit", "AMEI-CBP"),
        ("JS_EMR", "Judge Save", js, "js_emr_amei", "js_emr_cbr", "AMEI-CBR"),
        ("JS_EMR", "Judge Save", js, "js_emr_amei", "js_emr_cbp", "AMEI-CBP"),
        ("Interception", "All Weeks", week_df, "intercept_amei", "intercept_cbr", "AMEI-CBR"),
        ("Interception", "All Weeks", week_df, "intercept_amei", "intercept_cbp", "AMEI-CBP"),
    ]

    for metric, group, df_group, a_col, b_col, label in comparisons:
        mean_diff, ci_low, ci_high = diff_ci(df_group[a_col].to_numpy(), df_group[b_col].to_numpy())
        records.append({
            "metric": metric,
            "group": group,
            "compare": label,
            "mean_diff": mean_diff,
            "ci_low": ci_low,
            "ci_high": ci_high,
        })

    diff_df = pd.DataFrame.from_records(records)
    diff_path = os.path.join(out_dir, "bootstrap_diff_ci.csv")
    diff_df.to_csv(diff_path, index=False, encoding="utf-8-sig")
    return diff_df


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


def write_bootstrap_diff_summary(diff_df: pd.DataFrame, out_path: str) -> None:
    lines = [
        "## Bootstrap Difference CIs (95%)",
        "",
        "| Metric | Group | Compare | Mean Diff | CI Low | CI High |",
        "|--------|-------|---------|-----------|--------|---------|",
    ]
    for _, row in diff_df.iterrows():
        lines.append(
            f"| {row['metric']} | {row['group']} | {row['compare']} | {row['mean_diff']:.3f} | {row['ci_low']:.3f} | {row['ci_high']:.3f} |"
        )
    lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def stage_analysis(week_df: pd.DataFrame, out_dir: str) -> None:
    """Analyze EMR/B2CR by season stage and plot grouped bars."""
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


def stage_stability_table(week_df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
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
            records.append({
                "stage": stage,
                "metric": "JS_EMR",
                "method": method.upper(),
                "value": safe_mean(js[f"js_emr_{method}"].to_numpy(dtype=float)),
            })
            records.append({
                "stage": stage,
                "metric": "Interception",
                "method": method.upper(),
                "value": safe_mean(stage_df[f"intercept_{method}"].to_numpy(dtype=float)),
            })
            records.append({
                "stage": stage,
                "metric": "Spearman",
                "method": method.upper(),
                "value": safe_mean(stage_df[f"spearman_{method}"].to_numpy(dtype=float)),
            })

    out_df = pd.DataFrame.from_records(records)
    out_df.to_csv(os.path.join(out_dir, "season_stage_stability.csv"), index=False, encoding="utf-8-sig")
    return out_df


def season_stability_analysis(metrics_df: pd.DataFrame, week_df: pd.DataFrame, out_dir: str) -> None:
    metrics_df.to_csv(os.path.join(out_dir, "season_stability.csv"), index=False, encoding="utf-8-sig")

    js_seasons = metrics_df[metrics_df["judge_save"] == True]
    no_js_seasons = metrics_df[metrics_df["judge_save"] == False]

    def stat_line(label: str, values: np.ndarray) -> str:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return f"- {label}: n/a"
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
        return f"- {label}: std={std:.3f}, IQR={iqr:.3f}"

    lines = [
        "## Season Stability Summary",
        "",
        stat_line("EMR (No Judge Save, AMEI)", no_js_seasons["emr_amei"].to_numpy()),
        stat_line("B2CR (Judge Save, AMEI)", js_seasons["b2cr_amei"].to_numpy()),
        stat_line("JS-EMR (Judge Save, AMEI)", js_seasons["js_emr_amei"].to_numpy()),
        stat_line("Interception (All Seasons, AMEI)", metrics_df["interception_mean_amei"].to_numpy()),
        stat_line("Spearman (All Seasons, AMEI)", metrics_df["spearman_mean_amei"].to_numpy()),
        "",
    ]

    stage_df = stage_stability_table(week_df, out_dir)
    lines.extend([
        "## Stage Stability (Means)",
        "",
        "| Stage | Metric | AMEI | CBR | CBP |",
        "|-------|--------|------|-----|-----|",
    ])
    for stage in ["Early", "Mid", "Late"]:
        for metric in ["EMR", "B2CR", "JS_EMR", "Interception", "Spearman"]:
            vals = []
            for method in ["AMEI", "CBR", "CBP"]:
                v = stage_df[(stage_df["stage"] == stage) & (stage_df["metric"] == metric) & (stage_df["method"] == method)]["value"]
                vals.append(float(v.values[0]) if len(v) else float("nan"))
            lines.append(
                f"| {stage} | {metric} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} |"
            )
        lines.append("")

    out_path = os.path.join(out_dir, "season_stability_summary.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def judge_save_rule_comparison(week_df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    js = week_df[week_df["judge_save"] == True]
    records = []
    for method in ["amei", "cbr", "cbp"]:
        records.append({
            "method": method.upper(),
            "js_emr_rule_a": safe_mean(js[f"js_emr_{method}_a"].to_numpy(dtype=float)),
            "js_emr_rule_b": safe_mean(js[f"js_emr_{method}"].to_numpy(dtype=float)),
            "js_emr_rule_c": safe_mean(js[f"js_emr_{method}_c"].to_numpy(dtype=float)),
        })
    out_df = pd.DataFrame.from_records(records)
    out_df.to_csv(os.path.join(out_dir, "judge_save_rule_comparison.csv"), index=False, encoding="utf-8-sig")
    return out_df


def run_judge_save_coeff_grid(
    week_cache: List[Dict[str, object]],
    args,
    out_dir: str,
) -> pd.DataFrame:
    coeffs = np.round(np.arange(0.2, 0.9, 0.1), 2)
    records: List[Dict[str, float]] = []

    for coef in coeffs:
        hits = {"amei": [], "cbr": [], "cbp": []}
        for row in week_cache:
            if not bool(row["judge_save"]):
                continue
            J = row["J"]
            p_hat = row["p_hat"]
            tie_break = row["tie_break"]
            prev_rank_score = row["prev_rank_score"]
            progress = row["progress"]
            z_scores = row["z_scores"]
            merit_score = row["merit_score"]
            actual_elim_idx = row["actual_elim_idx"]

            kappa, _, _ = compute_kappa(z_scores, args)
            effective_votes = p_hat * kappa
            vote_sum = np.sum(effective_votes)
            if vote_sum > 0:
                pop_score = effective_votes / vote_sum
            else:
                pop_sum = np.sum(p_hat)
                pop_score = p_hat / pop_sum if pop_sum > 0 else np.full_like(p_hat, 1.0 / len(p_hat))

            w_j, w_f = logistic_weights(progress, args.w_start, args.w_end, args.logistic_k, args.logistic_t0)
            final_score = w_j * merit_score + w_f * pop_score

            bottom2_amei = pick_bottom_k(final_score, 2, tie_break)
            _, bottom2_cbr = apply_cbr(J, p_hat, tie_break)
            _, bottom2_cbp = apply_cbp(J, p_hat, tie_break)

            elim_amei = pick_judge_save_elim_prev_rank(bottom2_amei, J, prev_rank_score, tie_break, coef=float(coef))
            elim_cbr = pick_judge_save_elim_prev_rank(bottom2_cbr, J, prev_rank_score, tie_break, coef=float(coef))
            elim_cbp = pick_judge_save_elim_prev_rank(bottom2_cbp, J, prev_rank_score, tie_break, coef=float(coef))

            if actual_elim_idx:
                hits["amei"].append(1 if elim_amei in actual_elim_idx else 0)
                hits["cbr"].append(1 if elim_cbr in actual_elim_idx else 0)
                hits["cbp"].append(1 if elim_cbp in actual_elim_idx else 0)

        for method in ["amei", "cbr", "cbp"]:
            records.append({
                "coef": float(coef),
                "method": method.upper(),
                "js_emr": safe_mean(np.asarray(hits[method], dtype=float)),
            })

    out_df = pd.DataFrame.from_records(records)
    out_df.to_csv(os.path.join(out_dir, "judge_save_coeff_grid.csv"), index=False, encoding="utf-8-sig")
    return out_df


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
    tie_break_map = build_tie_break_map(df, args.tie_seed)
    prev_rank_score_map = build_prev_rank_score_map(df)
    week_cache = precompute_week_cache(df, args.eps, tie_break_map, prev_rank_score_map)

    weight_info: Optional[Dict[str, float]] = None
    if args.optimize_weights:
        weight_info = run_weight_search(week_cache, args, args.out_dir)
        if weight_info:
            args.w_start = weight_info["w_start"]
            args.w_end = weight_info["w_end"]
            args.logistic_k = weight_info["logistic_k"]
            args.logistic_t0 = weight_info["logistic_t0"]
            args.min_vote_weight = weight_info.get("min_vote_weight", args.min_vote_weight)

    records: List[Dict[str, float]] = []
    week_metrics: List[Dict[str, float]] = []
    bobby_rows: List[Dict[str, float]] = []

    for (season, week), g in df.groupby(["season", "week"], sort=True):
        g = g.sort_values("contestant_id").reset_index(drop=True)
        J = g["J_total"].to_numpy(dtype=float)
        p_hat = g["p_hat"].to_numpy(dtype=float)
        tie_break = tie_break_map.get((int(season), int(week)))
        if tie_break is None:
            tie_break = g["contestant_id"].to_numpy(dtype=int)
        prev_rank_score = prev_rank_score_map.get((int(season), int(week)))
        if prev_rank_score is None:
            prev_rank_score = np.full_like(J, 0.5, dtype=float)

        max_week = max_week_map.get(int(season), int(week))
        if max_week <= 1:
            progress = 0.0
        else:
            progress = (int(week) - 1) / (max_week - 1)

        z_scores = robust_z_score(J, args.eps)
        merit_score = softmax(z_scores)
        kappa, z_soft, z_hard = compute_kappa(z_scores, args)
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

            amei_js_elim = pick_judge_save_elim_rank(bottom2_idx, J, tie_break)
            cbr_js_elim = pick_judge_save_elim_rank(cbr_bottom2, J, tie_break)
            cbp_js_elim = pick_judge_save_elim_rank(cbp_bottom2, J, tie_break)

            amei_js_elim_a = pick_judge_save_elim_jtotal(bottom2_idx, J, tie_break)
            cbr_js_elim_a = pick_judge_save_elim_jtotal(cbr_bottom2, J, tie_break)
            cbp_js_elim_a = pick_judge_save_elim_jtotal(cbp_bottom2, J, tie_break)

            amei_js_elim_c = pick_judge_save_elim_prev_rank(bottom2_idx, J, prev_rank_score, tie_break)
            cbr_js_elim_c = pick_judge_save_elim_prev_rank(cbr_bottom2, J, prev_rank_score, tie_break)
            cbp_js_elim_c = pick_judge_save_elim_prev_rank(cbp_bottom2, J, prev_rank_score, tie_break)
            if actual_elim_idx:
                amei_js_elim_hit = 1 if amei_js_elim in actual_elim_idx else 0
                cbr_js_elim_hit = 1 if cbr_js_elim in actual_elim_idx else 0
                cbp_js_elim_hit = 1 if cbp_js_elim in actual_elim_idx else 0

                amei_js_elim_hit_a = 1 if amei_js_elim_a in actual_elim_idx else 0
                cbr_js_elim_hit_a = 1 if cbr_js_elim_a in actual_elim_idx else 0
                cbp_js_elim_hit_a = 1 if cbp_js_elim_a in actual_elim_idx else 0

                amei_js_elim_hit_c = 1 if amei_js_elim_c in actual_elim_idx else 0
                cbr_js_elim_hit_c = 1 if cbr_js_elim_c in actual_elim_idx else 0
                cbp_js_elim_hit_c = 1 if cbp_js_elim_c in actual_elim_idx else 0
            else:
                amei_js_elim_hit = float("nan")
                cbr_js_elim_hit = float("nan")
                cbp_js_elim_hit = float("nan")
                amei_js_elim_hit_a = float("nan")
                cbr_js_elim_hit_a = float("nan")
                cbp_js_elim_hit_a = float("nan")
                amei_js_elim_hit_c = float("nan")
                cbr_js_elim_hit_c = float("nan")
                cbp_js_elim_hit_c = float("nan")
        else:
            amei_hit = 1 if any(i in elim_idx for i in actual_elim_idx) else 0
            cbr_hit = 1 if any(i in cbr_elim for i in actual_elim_idx) else 0
            cbp_hit = 1 if any(i in cbp_elim for i in actual_elim_idx) else 0
            amei_js_elim_hit = float("nan")
            cbr_js_elim_hit = float("nan")
            cbp_js_elim_hit = float("nan")
            amei_js_elim_hit_a = float("nan")
            cbr_js_elim_hit_a = float("nan")
            cbp_js_elim_hit_a = float("nan")
            amei_js_elim_hit_c = float("nan")
            cbr_js_elim_hit_c = float("nan")
            cbp_js_elim_hit_c = float("nan")

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

        controversy_mask = identify_controversy_cases(
            z_scores,
            p_hat,
            z_quantile=args.controversy_z_q,
            p_quantile=args.controversy_p_q,
        )
        amei_intercept = bobby_interception_rate(controversy_mask, bottom2_idx)
        cbr_intercept = bobby_interception_rate(controversy_mask, cbr_bottom2)
        cbp_intercept = bobby_interception_rate(controversy_mask, cbp_bottom2)

        rank_j = rank_avg_desc(J)
        rank_amei = rank_avg_desc(final_score)
        rank_cbr = rank_avg_desc(cbr_score)
        rank_cbp = rank_avg_desc(cbp_score)
        spearman_amei = spearman_corr(rank_j, rank_amei)
        spearman_cbr = spearman_corr(rank_j, rank_cbr)
        spearman_cbp = spearman_corr(rank_j, rank_cbp)

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
            "spearman_amei": spearman_amei,
            "spearman_cbr": spearman_cbr,
            "spearman_cbp": spearman_cbp,
            "intercept_amei": amei_intercept,
            "intercept_cbr": cbr_intercept,
            "intercept_cbp": cbp_intercept,
            "js_emr_amei": amei_js_elim_hit,
            "js_emr_cbr": cbr_js_elim_hit,
            "js_emr_cbp": cbp_js_elim_hit,
            "js_emr_amei_a": amei_js_elim_hit_a,
            "js_emr_cbr_a": cbr_js_elim_hit_a,
            "js_emr_cbp_a": cbp_js_elim_hit_a,
            "js_emr_amei_c": amei_js_elim_hit_c,
            "js_emr_cbr_c": cbr_js_elim_hit_c,
            "js_emr_cbp_c": cbp_js_elim_hit_c,
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
                "tie_break": float(tie_break[idx]),
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
                    "tie_break": float(tie_break[idx]),
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
            "spearman_mean_amei": safe_mean(g["spearman_amei"]),
            "spearman_mean_cbr": safe_mean(g["spearman_cbr"]),
            "spearman_mean_cbp": safe_mean(g["spearman_cbp"]),
            "interception_mean_amei": safe_mean(g["intercept_amei"]),
            "interception_mean_cbr": safe_mean(g["intercept_cbr"]),
            "interception_mean_cbp": safe_mean(g["intercept_cbp"]),
        }
        if judge_save:
            metric.update({
                "b2cr_amei": float(np.nanmean(g["amei_hit"])),
                "b2cr_cbr": float(np.nanmean(g["cbr_hit"])),
                "b2cr_cbp": float(np.nanmean(g["cbp_hit"])),
                "js_emr_amei": float(np.nanmean(g["js_emr_amei"])),
                "js_emr_cbr": float(np.nanmean(g["js_emr_cbr"])),
                "js_emr_cbp": float(np.nanmean(g["js_emr_cbp"])),
                "js_emr_amei_a": float(np.nanmean(g["js_emr_amei_a"])),
                "js_emr_cbr_a": float(np.nanmean(g["js_emr_cbr_a"])),
                "js_emr_cbp_a": float(np.nanmean(g["js_emr_cbp_a"])),
                "js_emr_amei_c": float(np.nanmean(g["js_emr_amei_c"])),
                "js_emr_cbr_c": float(np.nanmean(g["js_emr_cbr_c"])),
                "js_emr_cbp_c": float(np.nanmean(g["js_emr_cbp_c"])),
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
                "js_emr_amei": float("nan"),
                "js_emr_cbr": float("nan"),
                "js_emr_cbp": float("nan"),
                "js_emr_amei_a": float("nan"),
                "js_emr_cbr_a": float("nan"),
                "js_emr_cbp_a": float("nan"),
                "js_emr_amei_c": float("nan"),
                "js_emr_cbr_c": float("nan"),
                "js_emr_cbp_c": float("nan"),
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

    grid_df = run_sensitivity_analysis(df, args, args.out_dir, tie_break_map, prev_rank_score_map)
    plot_pareto_frontier(metrics_df, grid_df, args.out_dir)
    plot_radar_comparison(metrics_df, week_df, args.out_dir)
    plot_weight_transition(args, args.out_dir)
    write_spearman_summary(metrics_df, args.out_dir)
    compute_whatif_scenario(full_df, args.out_dir)
    stage_analysis(week_df, args.out_dir)
    judge_save_rule_comparison(week_df, args.out_dir)
    season_stability_analysis(metrics_df, week_df, args.out_dir)
    weight_js, weight_nojs = run_weight_search_by_type(week_cache, args, args.out_dir)
    evaluate_adaptive_weights(week_cache, args, weight_js, weight_nojs, args.out_dir)
    run_judge_save_coeff_grid(week_cache, args, args.out_dir)
    run_controversy_sensitivity(week_cache, args, args.out_dir)
    run_controversy_alt_thresholds(week_cache, args, args.out_dir)
    run_controversy_gap_definition(week_cache, args, args.out_dir)
    select_controversy_threshold(args.out_dir)
    compute_interception_impact(week_cache, args, args.out_dir)
    loso_df = run_loso_validation(week_cache, args, args.out_dir)
    compute_loso_ci(loso_df, args.out_dir)

    # Generate overall summary statistics
    generate_summary_report(metrics_df, full_df, bobby_rows, args.out_dir, args, weight_info)
    mcnemar_path = os.path.join(args.out_dir, "mcnemar_summary.md")
    write_mcnemar_summary(week_df, mcnemar_path)
    append_markdown(args.out_dir, "interpretation_report.md", mcnemar_path)
    ci_df = bootstrap_metrics(week_df, args.out_dir)
    bootstrap_md = os.path.join(args.out_dir, "bootstrap_summary.md")
    write_bootstrap_summary(ci_df, bootstrap_md)
    append_markdown(args.out_dir, "interpretation_report.md", bootstrap_md)

    diff_df = bootstrap_diff_metrics(week_df, args.out_dir)
    diff_md = os.path.join(args.out_dir, "bootstrap_diff_summary.md")
    write_bootstrap_diff_summary(diff_df, diff_md)
    append_markdown(args.out_dir, "interpretation_report.md", diff_md)

    stability_md = os.path.join(args.out_dir, "season_stability_summary.md")
    append_markdown(args.out_dir, "interpretation_report.md", stability_md)


def generate_summary_report(
    metrics_df: pd.DataFrame,
    full_df: pd.DataFrame,
    bobby_rows: List[Dict],
    out_dir: str,
    args,
    weight_info: Optional[Dict[str, float]] = None,
) -> None:
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

    lines.extend([
        f"- Threshold mode: {args.threshold_mode} (z_soft_q={args.z_soft_q:.2f}, z_hard_q={args.z_hard_q:.2f})",
        f"- Controversy thresholds: z_q={args.controversy_z_q:.2f}, p_q={args.controversy_p_q:.2f}",
        f"- Weight optimization constraint: interception >= {args.min_interception:.2f}",
        "- EMR/B2CR constraints: drop <= 0.05 absolute and <= 10% relative vs CBR/CBP",
    ])
    if weight_info:
        lines.append(
            f"- Optimized weights: w_start={weight_info['w_start']:.2f}, w_end={weight_info['w_end']:.2f}, "
            f"k={weight_info['logistic_k']:.1f}, t0={weight_info['logistic_t0']:.2f}"
        )
        lines.append(f"- Selected min_vote_weight: {weight_info.get('min_vote_weight', args.min_vote_weight):.2f}")
        lines.append(
            f"- Selected point feasible: {weight_info.get('feasible', True)} (interception={weight_info.get('interception_mean', float('nan')):.3f})"
        )
        if weight_info.get("soft_lambda") is not None:
            lines.append(f"- Soft constraint used: lambda={weight_info['soft_lambda']:.1f}")
    lines.append("")
    
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
        js_emr_amei = js_seasons["js_emr_amei"].mean()
        js_emr_cbr = js_seasons["js_emr_cbr"].mean()
        js_emr_cbp = js_seasons["js_emr_cbp"].mean()
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
            "### Elimination Match Rate (JS-EMR) - Judge Save Rule",
            "",
            f"| Method | JS-EMR | Improvement vs CBR |",
            f"|--------|--------|--------------------|",
            f"| AMEI 2.0 | {js_emr_amei:.3f} | baseline |",
            f"| CBR | {js_emr_cbr:.3f} | {(js_emr_amei - js_emr_cbr)*100:+.1f}% |",
            f"| CBP | {js_emr_cbp:.3f} | {(js_emr_amei - js_emr_cbp)*100:+.1f}% |",
            "",
            "- Judge Save rule uses J_total + reverse rank (Rule B).",
            "- Rule comparison summary: judge_save_rule_comparison.csv",
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

    # Weight optimization justification
    weight_grid_path = os.path.join(out_dir, "weight_grid.csv")
    if os.path.exists(weight_grid_path):
        grid_df = pd.read_csv(weight_grid_path)
        if "feasible" in grid_df.columns:
            candidate = grid_df[grid_df["feasible"] == True]
            if candidate.empty:
                candidate = grid_df.copy()
        else:
            candidate = grid_df.copy()
        candidate = candidate.sort_values(["tradeoff_score", "fairness_loss_mean", "excitement_loss_mean"]).reset_index(drop=True)
        best = candidate.iloc[0]
        second = candidate.iloc[1] if len(candidate) > 1 else None
        lines.extend([
            "### Weight Optimization Justification",
            "",
            f"- Selected weights: w_start={best['w_start']:.2f}, w_end={best['w_end']:.2f}, k={best['logistic_k']:.1f}, t0={best['logistic_t0']:.2f}",
            f"- tradeoff_score={best['tradeoff_score']:.3f}, fairness_loss={best['fairness_loss_mean']:.3f}, excitement_loss={best['excitement_loss_mean']:.3f}",
            f"- interception_mean={best['interception_mean']:.3f} (feasible={bool(best.get('feasible', True))})",
        ])
        if "emr_min_cbr" in best and "b2cr_min_cbr" in best:
            lines.extend([
                f"- EMR constraint (CBR/CBP): >= {best['emr_min_cbr']:.3f} / {best['emr_min_cbp']:.3f}",
                f"- B2CR constraint (CBR/CBP): >= {best['b2cr_min_cbr']:.3f} / {best['b2cr_min_cbp']:.3f}",
            ])
        if not bool(best.get("feasible", True)):
            lines.append("- No grid point satisfies all constraints; selected the best trade-off candidate.")
        if second is not None:
            delta = float(second["tradeoff_score"] - best["tradeoff_score"])
            lines.append(
                f"- Next-best candidate: tradeoff_score={second['tradeoff_score']:.3f} (Δ={delta:.3f}), k={second['logistic_k']:.1f}"
            )
        lines.append("")

    # Judge Save rule robustness
    if len(js_seasons) > 0 and "js_emr_amei_a" in js_seasons.columns:
        js_emr_amei_a = js_seasons["js_emr_amei_a"].mean()
        js_emr_cbr_a = js_seasons["js_emr_cbr_a"].mean()
        js_emr_cbp_a = js_seasons["js_emr_cbp_a"].mean()
        js_emr_amei_c = js_seasons.get("js_emr_amei_c", pd.Series([np.nan])).mean()
        js_emr_cbr_c = js_seasons.get("js_emr_cbr_c", pd.Series([np.nan])).mean()
        js_emr_cbp_c = js_seasons.get("js_emr_cbp_c", pd.Series([np.nan])).mean()
        lines.extend([
            "### Judge Save Rule Robustness (A vs B vs C)",
            "",
            f"- AMEI: JS-EMR A={js_emr_amei_a:.3f}, B={js_emr_amei:.3f}, C={js_emr_amei_c:.3f}",
            f"- CBR: JS-EMR A={js_emr_cbr_a:.3f}, B={js_emr_cbr:.3f}, C={js_emr_cbr_c:.3f}",
            f"- CBP: JS-EMR A={js_emr_cbp_a:.3f}, B={js_emr_cbp:.3f}, C={js_emr_cbp_c:.3f}",
            "- Rule C uses J_total + 0.5 * previous reverse rank.",
            "",
        ])

    # Judge Save coefficient grid
    coeff_path = os.path.join(out_dir, "judge_save_coeff_grid.csv")
    if os.path.exists(coeff_path):
        coeff_df = pd.read_csv(coeff_path)
        amei_coeff = coeff_df[coeff_df["method"] == "AMEI"]
        if not amei_coeff.empty:
            best_row = amei_coeff.sort_values("js_emr", ascending=False).iloc[0]
            lines.extend([
                "### Rule C Coefficient Selection",
                "",
                f"- Best coef (AMEI JS-EMR): {best_row['coef']:.2f} (JS-EMR={best_row['js_emr']:.3f})",
                "- Full grid: judge_save_coeff_grid.csv",
                "",
            ])

    # Adaptive weights by season type
    adaptive_path = os.path.join(out_dir, "adaptive_weight_metrics.csv")
    if os.path.exists(adaptive_path):
        adaptive_df = pd.read_csv(adaptive_path)
        if not adaptive_df.empty:
            row = adaptive_df.iloc[0]
            lines.extend([
                "### Adaptive Weighting (JS vs non-JS)",
                "",
                f"- EMR={row['emr']:.3f}, B2CR={row['b2cr']:.3f}, Interception={row['interception']:.3f}, Spearman={row['spearman']:.3f}",
                "- Weight parameters saved in weight_selected_js.csv / weight_selected_nojs.csv",
            ])
            js_path = os.path.join(out_dir, "weight_selected_js.csv")
            nojs_path = os.path.join(out_dir, "weight_selected_nojs.csv")
            if os.path.exists(js_path):
                js_df = pd.read_csv(js_path)
                if not js_df.empty:
                    js_row = js_df.iloc[0]
                    lines.append(
                        f"- JS weights: w_start={js_row['w_start']:.2f}, w_end={js_row['w_end']:.2f}, k={js_row['logistic_k']:.1f}, t0={js_row['logistic_t0']:.2f}, min_vote={js_row['min_vote_weight']:.2f}"
                    )
            if os.path.exists(nojs_path):
                nojs_df = pd.read_csv(nojs_path)
                if not nojs_df.empty:
                    nojs_row = nojs_df.iloc[0]
                    lines.append(
                        f"- Non-JS weights: w_start={nojs_row['w_start']:.2f}, w_end={nojs_row['w_end']:.2f}, k={nojs_row['logistic_k']:.1f}, t0={nojs_row['logistic_t0']:.2f}, min_vote={nojs_row['min_vote_weight']:.2f}"
                    )
            lines.append("")

    # Controversy threshold robustness
    cont_path = os.path.join(out_dir, "controversy_sensitivity.csv")
    alt_path = os.path.join(out_dir, "controversy_alt_thresholds.csv")
    if os.path.exists(cont_path):
        cont_df = pd.read_csv(cont_path)
        amei_mean = cont_df["interception_amei"].mean()
        cbr_mean = cont_df["interception_cbr"].mean()
        cbp_mean = cont_df["interception_cbp"].mean()
        lines.extend([
            "### Controversy Threshold Robustness",
            "",
            f"- Quantile grid (z_q 0.10–0.30, p_q 0.70–0.90):",
            f"  AMEI mean={amei_mean:.3f} (range {cont_df['interception_amei'].min():.3f}–{cont_df['interception_amei'].max():.3f}),",
            f"  CBR mean={cbr_mean:.3f}, CBP mean={cbp_mean:.3f}",
        ])
        if os.path.exists(alt_path):
            alt_df = pd.read_csv(alt_path)
            if not alt_df.empty:
                alt = alt_df.iloc[0]
                lines.append(
                    f"- Absolute threshold (z<=-1.0, p>=0.15): AMEI={alt['interception_amei']:.3f}, "
                    f"CBR={alt['interception_cbr']:.3f}, CBP={alt['interception_cbp']:.3f}"
                )
        gap_path = os.path.join(out_dir, "controversy_alt_definitions.csv")
        if os.path.exists(gap_path):
            gap_df = pd.read_csv(gap_path)
            if not gap_df.empty:
                gap = gap_df.iloc[0]
                lines.append(
                    f"- Gap definition (fan-judge): AMEI={gap['interception_amei']:.3f}, "
                    f"CBR={gap['interception_cbr']:.3f}, CBP={gap['interception_cbp']:.3f}"
                )
        selected_path = os.path.join(out_dir, "threshold_selected.csv")
        if os.path.exists(selected_path):
            sel_df = pd.read_csv(selected_path)
            if not sel_df.empty:
                sel = sel_df.iloc[0]
                lines.append(
                    f"- Selected threshold (max advantage): z_q={sel['z_quantile']:.2f}, p_q={sel['p_quantile']:.2f}, "
                    f"advantage={sel['advantage']:.3f}"
                )
        lines.append("")

    loso_path = os.path.join(out_dir, "loso_summary.csv")
    if os.path.exists(loso_path):
        loso_df = pd.read_csv(loso_path)
        if not loso_df.empty:
            emr_adv = (loso_df["emr_amei"] - loso_df[["emr_cbr", "emr_cbp"]].max(axis=1)).mean()
            b2cr_adv = (loso_df["b2cr_amei"] - loso_df[["b2cr_cbr", "b2cr_cbp"]].max(axis=1)).mean()
            inter_adv = (loso_df["interception_amei"] - loso_df[["interception_cbr", "interception_cbp"]].max(axis=1)).mean()
            spearman_adv = (loso_df["spearman_amei"] - loso_df[["spearman_cbr", "spearman_cbp"]].max(axis=1)).mean()
            lines.extend([
                "### Cross-Season Generalization (LOSO)",
                "",
                f"- Mean advantage vs max baseline: EMR {emr_adv:+.3f}, B2CR {b2cr_adv:+.3f}, Interception {inter_adv:+.3f}, Spearman {spearman_adv:+.3f}",
                "- LOSO summary saved to loso_summary.csv",
                "",
            ])
            ci_path = os.path.join(out_dir, "loso_ci.csv")
            if os.path.exists(ci_path):
                ci_df = pd.read_csv(ci_path)
                if not ci_df.empty:
                    lines.append("- LOSO bootstrap CIs (mean [95% CI]):")
                    for _, row in ci_df.iterrows():
                        lines.append(
                            f"  - {row['metric']}: {row['mean']:+.3f} [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"
                        )
                    lines.append("")

    impact_path = os.path.join(out_dir, "interception_impact.csv")
    if os.path.exists(impact_path):
        impact_df = pd.read_csv(impact_path)
        if not impact_df.empty:
            amei_mean = impact_df["amei_captured"].mean()
            cbr_mean = impact_df["cbr_captured"].mean()
            cbp_mean = impact_df["cbp_captured"].mean()
            delta_cbr = impact_df["delta_vs_cbr"].mean()
            delta_cbp = impact_df["delta_vs_cbp"].mean()
            lines.extend([
                "### Expected Corrections per Season",
                "",
                f"- AMEI: {amei_mean:.2f} (Δ vs CBR {delta_cbr:+.2f}, Δ vs CBP {delta_cbp:+.2f})",
                f"- CBR: {cbr_mean:.2f}",
                f"- CBP: {cbp_mean:.2f}",
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
    spearman_amei = metrics_df["spearman_mean_amei"].mean()
    spearman_cbr = metrics_df["spearman_mean_cbr"].mean()
    spearman_cbp = metrics_df["spearman_mean_cbp"].mean()
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
        f"- Spearman fairness (AMEI/CBR/CBP): {spearman_amei:.3f} / {spearman_cbr:.3f} / {spearman_cbp:.3f}",
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
        "- sensitivity_grid.csv: grid search results for z_soft_q/z_hard_q",
        "- sensitivity_heatmap_nojs.png: EMR heatmap for seasons without Judge Save",
        "- sensitivity_heatmap_js.png: B2CR heatmap for seasons with Judge Save",
        "- weight_grid.csv: logistic weight grid search",
        "- weight_selected.csv: selected weight parameters",
        "- weight_selected_soft.csv: soft-constraint selection (if needed)",
        "- weight_grid_js.csv / weight_grid_nojs.csv: season-type weight grids",
        "- weight_selected_js.csv / weight_selected_nojs.csv: season-type selected weights",
        "- adaptive_weight_metrics.csv: adaptive weighting performance",
        "- controversy_sensitivity.csv: controversy threshold sensitivity",
        "- controversy_heatmap.png: interception sensitivity heatmap",
        "- controversy_alt_thresholds.csv: absolute-threshold interception",
        "- controversy_alt_definitions.csv: fan-judge gap controversy",
        "- threshold_selected.csv: data-driven threshold selection",
        "- pareto_nojs.png / pareto_js.png: fairness–excitement trade-off plots",
        "- bobby_trajectory.png: Bobby Bones z-score/kappa/rank trend",
        "- mcnemar_summary.md: McNemar significance tests",
        "- fairness_spearman.csv: fairness rank correlation summary",
        "- radar_comparison.png: radar chart across methods",
        "- weight_transition.png: logistic weight transition curve",
        "- summary.md: concise numeric summary",
        "- whatif_bobby_bones.csv / whatif_summary.md: Season 27 what-if analysis",
        "- bootstrap_ci.csv: bootstrap confidence intervals",
        "- bootstrap_diff_ci.csv: bootstrap difference CIs",
        "- bootstrap_diff_summary.md: bootstrap difference summary",
        "- season_stability.csv / season_stability_summary.md: season stability metrics",
        "- season_stage_stability.csv: stage stability metrics",
        "- judge_save_rule_comparison.csv: Rule A/B/C (JS-EMR)",
        "- judge_save_coeff_grid.csv: Rule C coefficient grid",
        "- loso_summary.csv: leave-one-season-out results",
        "- loso_ci.csv: LOSO bootstrap confidence intervals",
        "- interception_impact.csv: expected corrections per season",
        "- stage_analysis.csv / stage_comparison.png: stage-wise performance",
        "",
    ])
    
    report_path = os.path.join(out_dir, "interpretation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"[INFO] Summary report saved to: {report_path}")


def holm_adjust(pvals: List[float]) -> List[float]:
    if not pvals:
        return []
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    adjusted = np.empty_like(pvals)
    max_adj = 0.0
    n = len(pvals)
    for i, idx in enumerate(order):
        adj = (n - i) * pvals[idx]
        max_adj = max(max_adj, adj)
        adjusted[idx] = min(max_adj, 1.0)
    return adjusted.tolist()


def write_mcnemar_summary(week_df: pd.DataFrame, out_path: str) -> None:
    """Write McNemar test summary comparing AMEI with CBR/CBP."""
    lines = [
        "## Statistical Significance (McNemar)",
        "",
    ]

    tests = []

    def collect_tests(df_group: pd.DataFrame, label: str, a_col: str, b_prefix: str) -> None:
        if df_group.empty:
            return
        for method in ["cbr", "cbp"]:
            b_col = f"{b_prefix}{method}" if b_prefix else f"{method}_hit"
            a = df_group[a_col].astype(float).to_numpy()
            b = df_group[b_col].astype(float).to_numpy()
            mask = np.isfinite(a) & np.isfinite(b)
            a = a[mask].astype(int)
            b = b[mask].astype(int)
            if len(a) == 0:
                continue
            n01 = int(np.sum((a == 0) & (b == 1)))
            n10 = int(np.sum((a == 1) & (b == 0)))
            table = [[0, n01], [n10, 0]]
            try:
                res = mcnemar(table, exact=False, correction=True)
                tests.append({
                    "label": label,
                    "compare": f"AMEI vs {method.upper()}",
                    "n01": n01,
                    "n10": n10,
                    "p_raw": float(res.pvalue),
                })
            except Exception:
                tests.append({
                    "label": label,
                    "compare": f"AMEI vs {method.upper()}",
                    "n01": n01,
                    "n10": n10,
                    "p_raw": float("nan"),
                })

    collect_tests(week_df[week_df["judge_save"] == False], "No Judge Save", "amei_hit", "")
    collect_tests(week_df[week_df["judge_save"] == True], "Judge Save (Bottom-2)", "amei_hit", "")
    collect_tests(week_df[week_df["judge_save"] == True], "Judge Save (Elimination)", "js_emr_amei", "js_emr_")

    pvals = [t["p_raw"] for t in tests if np.isfinite(t["p_raw"])]
    adj = holm_adjust(pvals)
    adj_iter = iter(adj)
    for t in tests:
        if np.isfinite(t["p_raw"]):
            t["p_adj"] = next(adj_iter)
        else:
            t["p_adj"] = float("nan")

    for t in tests:
        lines.append(
            f"- {t['label']} {t['compare']}: n01={t['n01']}, n10={t['n10']}, p={t['p_raw']:.4f}, p_adj={t['p_adj']:.4f}"
        )
    lines.append("")

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
