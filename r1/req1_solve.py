import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog
try:
    import pulp
    PULP_AVAILABLE = True
except Exception:
    pulp = None
    PULP_AVAILABLE = False


# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    ap = argparse.ArgumentParser(description="Estimate weekly fan vote shares for MCM Problem C (Requirement 1).")
    ap.add_argument("--data", default=os.path.join(SCRIPT_DIR, "2026_MCM_Problem_C_Data.csv"), help="CSV path")
    ap.add_argument("--out-dir", default=os.path.join(SCRIPT_DIR, "outputs"), help="output directory")
    ap.add_argument("--total-votes", type=float, default=1e7, help="weekly total votes scale")
    ap.add_argument("--alpha-pop", type=float, default=0.4, help="popularity weight in prior")
    ap.add_argument("--beta", type=float, default=1.2, help="judge score weight in prior")
    ap.add_argument("--lambda-smooth", type=float, default=0.7, help="smoothness weight")
    ap.add_argument("--tau-entropy", type=float, default=0.01, help="entropy weight")
    ap.add_argument("--kappa-rank", type=float, default=0.5, help="rank-to-share temperature")
    ap.add_argument("--rho-pop", type=float, default=0.3, help="popularity update rate")
    ap.add_argument("--ilp-time-limit", type=int, default=10, help="ILP time limit seconds")
    ap.add_argument("--eps", type=float, default=1e-9, help="epsilon for logs")
    ap.add_argument("--uncertainty", action="store_true", help="compute uncertainty intervals (percent seasons exact, rank seasons sampled)")
    ap.add_argument("--rank-samples", type=int, default=50, help="samples for rank-season uncertainty")
    return ap.parse_args()


def rule_type(season: int) -> str:
    return "PERCENT" if 3 <= season <= 27 else "RANK"


def has_judge_save(season: int) -> bool:
    return season >= 28


def zscore(x: np.ndarray, eps: float) -> np.ndarray:
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < eps:
        return np.zeros_like(x)
    return (x - mu) / (sd + eps)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    expz = np.exp(z)
    return expz / np.sum(expz)


def rank_desc(vec: np.ndarray) -> np.ndarray:
    # Competition ranking: 1,1,3 for ties. Tie-break by original order for stability.
    order = np.lexsort((np.arange(len(vec)), -vec))
    ranks = np.zeros(len(vec), dtype=int)
    rank = 1
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vec[order[j + 1]] == vec[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        rank += (j - i + 1)
        i = j + 1
    return ranks


def build_week_totals(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
    score_cols = [c for c in df.columns if c.startswith("week") and c.endswith("_score")]
    weeks = sorted({int(c.split("_", 1)[0][4:]) for c in score_cols})
    totals = {}
    for w in weeks:
        cols = [c for c in score_cols if c.startswith(f"week{w}_")]
        vals = df[cols]
        # sum with NaN -> if all NaN, keep NaN
        sums = vals.sum(axis=1, skipna=True)
        all_nan = vals.isna().all(axis=1)
        sums = sums.mask(all_nan, np.nan)
        totals[w] = sums
    totals_df = pd.DataFrame(totals)
    return totals_df, weeks


def parse_elim_week(results: str) -> int:
    if not isinstance(results, str):
        return -1
    if results.startswith("Eliminated Week"):
        try:
            return int(results.split("Eliminated Week")[1].strip())
        except Exception:
            return -1
    return -1


def infer_withdrew_week(totals_row: pd.Series, weeks: List[int]) -> int:
    last_pos = None
    for w in weeks:
        val = totals_row.get(w)
        if pd.notna(val) and val > 0:
            last_pos = w
    if last_pos is None:
        return -1
    # find first zero after last positive; otherwise, next week
    for w in weeks:
        if w > last_pos:
            val = totals_row.get(w)
            if pd.notna(val) and val == 0:
                return w
    return min(last_pos + 1, max(weeks))


def simplex_projection(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def build_popularity_vector(idx_list: List[int], prev_p: np.ndarray, pop_map: Dict[int, float], eps: float) -> np.ndarray:
    pop_vec = np.zeros(len(idx_list), dtype=float)
    for pos, i in enumerate(idx_list):
        if i in pop_map:
            pop_vec[pos] = pop_map[i]
        else:
            pop_vec[pos] = math.log(max(prev_p[pos], eps))
    return pop_vec


def build_prior_share(Jvec: np.ndarray, prev_p: np.ndarray, pop_vec: np.ndarray,
                      alpha_pop: float, beta: float, lambda_smooth: float, eps: float) -> np.ndarray:
    zJ = zscore(Jvec, eps)
    prev_p = np.clip(prev_p, eps, 1.0)
    pop_z = zscore(pop_vec, eps) if pop_vec is not None else np.zeros_like(zJ)
    logit = alpha_pop * pop_z + beta * zJ + lambda_smooth * np.log(prev_p)
    return softmax(logit)


def solve_week_percent(Jvec, prev_p, p_tilde, E_idx, has_save, lambda_smooth, tau, eps):
    n = len(Jvec)
    q = Jvec / np.sum(Jvec)
    prev_p = np.clip(prev_p, eps, 1.0)
    p_tilde = np.clip(p_tilde, eps, 1.0)

    def objective(p):
        p = np.clip(p, eps, 1.0)
        return np.sum((p - p_tilde) ** 2) + lambda_smooth * np.sum((p - prev_p) ** 2) - tau * np.sum(np.log(p))

    bounds = [(eps, 1.0) for _ in range(n)]
    cons = [
        {"type": "eq", "fun": lambda p: np.sum(p) - 1.0}
    ]

    if E_idx:
        survivors = [i for i in range(n) if i not in E_idx]

        def add_constraints_for_bottom(e_idx, b_idx=None):
            cons_local = list(cons)
            if b_idx is None:
                for e in e_idx:
                    for j in survivors:
                        cons_local.append({
                            "type": "ineq",
                            "fun": lambda p, e=e, j=j: (q[j] - q[e]) - (p[e] - p[j])
                        })
            else:
                for j in [k for k in range(n) if k not in {e_idx, b_idx}]:
                    cons_local.append({
                        "type": "ineq",
                        "fun": lambda p, e=e_idx, j=j: (q[j] - q[e]) - (p[e] - p[j])
                    })
                    cons_local.append({
                        "type": "ineq",
                        "fun": lambda p, b=b_idx, j=j: (q[j] - q[b]) - (p[b] - p[j])
                    })
            return cons_local

        if not has_save:
            cons_use = add_constraints_for_bottom(E_idx)
            res = minimize(objective, p_tilde, method="SLSQP", bounds=bounds, constraints=cons_use)
            if not res.success:
                res = minimize(objective, simplex_projection(n), method="SLSQP", bounds=bounds, constraints=cons_use)
            return np.clip(res.x, eps, 1.0)

        # judge save: ensure eliminated in bottom2; enumerate companion
        best = None
        for e in E_idx:
            for b in [i for i in range(n) if i != e]:
                cons_use = add_constraints_for_bottom(e, b)
                res = minimize(objective, p_tilde, method="SLSQP", bounds=bounds, constraints=cons_use)
                if not res.success:
                    continue
                val = objective(res.x)
                if best is None or val < best[0]:
                    best = (val, res.x)
        if best is None:
            res = minimize(objective, p_tilde, method="SLSQP", bounds=bounds, constraints=cons)
            return np.clip(res.x, eps, 1.0)
        return np.clip(best[1], eps, 1.0)

    res = minimize(objective, p_tilde, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        res = minimize(objective, simplex_projection(n), method="SLSQP", bounds=bounds, constraints=cons)
    return np.clip(res.x, eps, 1.0)


def adjust_ranks_for_constraints(rV, rJ, eliminated, bottom_k):
    n = len(rV)
    if not eliminated:
        return rV
    rV = rV.copy()
    max_iter = n * n

    def ranksum():
        return rV + rJ

    def topk_indices(k):
        rs = ranksum()
        return set(np.argsort(rs)[-k:])

    k = bottom_k
    for _ in range(max_iter):
        topk = topk_indices(k)
        missing = [e for e in eliminated if e not in topk]
        if not missing:
            break
        e = missing[0]
        # swap e with a non-eliminated in topk that has highest ranksum
        rs = ranksum()
        candidates = [i for i in topk if i not in eliminated]
        if not candidates:
            break
        j = max(candidates, key=lambda i: rs[i])
        rV[e], rV[j] = rV[j], rV[e]
    return rV


def solve_week_rank_heuristic(Jvec, p_prior, E_idx, has_save, kappa):
    n = len(Jvec)
    rJ = rank_desc(Jvec)
    r0 = rank_desc(p_prior)

    # initial rV from r0 ordering
    order = np.argsort(r0)
    rV = np.zeros(n, dtype=int)
    for rank, idx in enumerate(order, start=1):
        rV[idx] = rank

    if E_idx:
        bottom_k = 2 if has_save else len(E_idx)
        rV = adjust_ranks_for_constraints(rV, rJ, E_idx, bottom_k)

    p = np.exp(-kappa * rV)
    p = p / np.sum(p)
    return p


def solve_week_rank_ilp(Jvec, p_prior, E_idx, has_save, kappa, time_limit=10):
    if not PULP_AVAILABLE:
        return None
    n = len(Jvec)
    rJ = rank_desc(Jvec)
    r0 = rank_desc(p_prior)

    def build_model(eliminated_set=None, bottom2_pair=None):
        prob = pulp.LpProblem("rank_vote", pulp.LpMinimize)
        x = pulp.LpVariable.dicts("x", (range(n), range(1, n + 1)), lowBound=0, upBound=1, cat="Binary")
        rV = {i: pulp.lpSum(r * x[i][r] for r in range(1, n + 1)) for i in range(n)}

        for i in range(n):
            prob += pulp.lpSum(x[i][r] for r in range(1, n + 1)) == 1
        for r in range(1, n + 1):
            prob += pulp.lpSum(x[i][r] for i in range(n)) == 1

        d = pulp.LpVariable.dicts("d", range(n), lowBound=0, cat="Continuous")
        for i in range(n):
            prob += d[i] >= rV[i] - r0[i]
            prob += d[i] >= -(rV[i] - r0[i])

        if eliminated_set:
            if bottom2_pair is None:
                survivors = [i for i in range(n) if i not in eliminated_set]
                for e in eliminated_set:
                    for s in survivors:
                        prob += (rJ[e] + rV[e]) >= (rJ[s] + rV[s])
            else:
                e, b = bottom2_pair
                others = [i for i in range(n) if i not in {e, b}]
                for j in others:
                    prob += (rJ[e] + rV[e]) >= (rJ[j] + rV[j])
                    prob += (rJ[b] + rV[b]) >= (rJ[j] + rV[j])

        prob += pulp.lpSum(d[i] for i in range(n))
        return prob, x

    def solve_model(eliminated_set=None, bottom2_pair=None):
        prob, x = build_model(eliminated_set, bottom2_pair)
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
        status = prob.solve(solver)
        if pulp.LpStatus.get(status) != "Optimal":
            return None
        rV_sol = np.zeros(n, dtype=float)
        for i in range(n):
            val = 0.0
            for r in range(1, n + 1):
                v = x[i][r].value()
                if v is None:
                    return None
                val += r * v
            rV_sol[i] = val
        p = np.exp(-kappa * rV_sol)
        p = p / np.sum(p)
        return p, pulp.value(prob.objective)

    if not E_idx:
        solved = solve_model()
        return solved[0] if solved else None

    if has_save and len(E_idx) == 1:
        e = E_idx[0]
        best = None
        for b in range(n):
            if b == e:
                continue
            solved = solve_model(eliminated_set=E_idx, bottom2_pair=(e, b))
            if not solved:
                continue
            p, obj = solved
            if best is None or obj < best[0]:
                best = (obj, p)
        return best[1] if best else None

    solved = solve_model(eliminated_set=E_idx)
    return solved[0] if solved else None


def solve_week_rank(Jvec, p_prior, E_idx, has_save, kappa, ilp_time_limit=10):
    p = solve_week_rank_ilp(Jvec, p_prior, E_idx, has_save, kappa, time_limit=ilp_time_limit)
    if p is not None:
        return p
    return solve_week_rank_heuristic(Jvec, p_prior, E_idx, has_save, kappa)


def predict_elimination(rule, has_save, Jvec, p):
    n = len(Jvec)
    if rule == "PERCENT":
        q = Jvec / np.sum(Jvec)
        C = q + p
        if has_save:
            idx = np.argsort(C)[:2]
            return set(idx)
        idx = np.argsort(C)[:1]
        return set(idx)
    # rank
    rJ = rank_desc(Jvec)
    rV = rank_desc(p)
    ranksum = rJ + rV
    if has_save:
        idx = np.argsort(ranksum)[-2:]
        return set(idx)
    idx = np.argsort(ranksum)[-1:]
    return set(idx)


def uncertainty_percent(Jvec, E_idx, has_save):
    n = len(Jvec)
    q = Jvec / np.sum(Jvec)
    A = []
    b = []
    if E_idx:
        survivors = [i for i in range(n) if i not in E_idx]
        if not has_save:
            for e in E_idx:
                for j in survivors:
                    row = np.zeros(n)
                    row[e] = 1
                    row[j] = -1
                    A.append(row)
                    b.append(q[j] - q[e])
        else:
            # use first eliminated only for constraints; caller can interpret as approximation
            e = E_idx[0]
            # pick any companion (first non-e) for constraint; uncertainty in percent with save is approximate
            b_idx = 0 if e != 0 else 1
            for j in [k for k in range(n) if k not in {e, b_idx}]:
                row = np.zeros(n)
                row[e] = 1
                row[j] = -1
                A.append(row)
                b.append(q[j] - q[e])
                row = np.zeros(n)
                row[b_idx] = 1
                row[j] = -1
                A.append(row)
                b.append(q[j] - q[b_idx])

    A = np.array(A) if A else None
    b = np.array(b) if b else None
    bounds = [(0, 1) for _ in range(n)]
    U = np.zeros(n)
    for i in range(n):
        c = np.zeros(n)
        c[i] = 1
        res_min = linprog(c, A_ub=A, b_ub=b, A_eq=[np.ones(n)], b_eq=[1.0], bounds=bounds, method="highs")
        res_max = linprog(-c, A_ub=A, b_ub=b, A_eq=[np.ones(n)], b_eq=[1.0], bounds=bounds, method="highs")
        if res_min.success and res_max.success:
            U[i] = res_max.x[i] - res_min.x[i]
        else:
            U[i] = np.nan
    return U


def uncertainty_rank(Jvec, p_prior, E_idx, has_save, kappa, samples, eps, ilp_time_limit):
    n = len(Jvec)
    Ps = []
    for _ in range(samples):
        noise = np.random.normal(0, 0.01, size=n)
        p0 = np.clip(p_prior + noise, eps, 1.0)
        p0 = p0 / np.sum(p0)
        p = solve_week_rank(Jvec, p0, E_idx, has_save, kappa, ilp_time_limit=ilp_time_limit)
        Ps.append(p)
    Ps = np.array(Ps)
    low = np.quantile(Ps, 0.05, axis=0)
    high = np.quantile(Ps, 0.95, axis=0)
    return high - low


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.data)
    totals_df, weeks = build_week_totals(df)

    df = df.copy()
    df["_id"] = np.arange(len(df))

    elim_week = []
    for idx, row in df.iterrows():
        res = row.get("results")
        w = parse_elim_week(res)
        if w == -1 and isinstance(res, str) and res.strip() == "Withdrew":
            w = infer_withdrew_week(totals_df.loc[idx], weeks)
        elim_week.append(w)
    df["_elim_week"] = elim_week

    # Build season-week contestant sets
    season_groups = df.groupby("season")
    records = []
    metrics = {"EMR": 0, "EMR_denom": 0, "B2CR": 0, "B2CR_denom": 0}

    for season, sdf in season_groups:
        season = int(season)
        rule = rule_type(season)
        save = has_judge_save(season)
        # init prev_p
        prev_p_map: Dict[int, float] = {}
        pop_map: Dict[int, float] = {}
        for w in weeks:
            # active contestants: J not NaN and >0
            Jcol = totals_df.loc[sdf.index, w]
            active_mask = (Jcol.notna()) & (Jcol > 0)
            if not active_mask.any():
                continue
            idx_list = sdf.index[active_mask].tolist()
            Jvec = Jcol[active_mask].to_numpy(dtype=float)
            # elimination set
            E_idx_local = [i for i in idx_list if df.loc[i, "_elim_week"] == w]
            # build prev_p vector
            if prev_p_map:
                prev_p = np.array([prev_p_map.get(i, 1.0 / len(idx_list)) for i in idx_list], dtype=float)
                prev_p = np.clip(prev_p, args.eps, 1.0)
                prev_p = prev_p / np.sum(prev_p)
            else:
                prev_p = simplex_projection(len(idx_list))

            pop_vec = build_popularity_vector(idx_list, prev_p, pop_map, args.eps)
            p_prior = build_prior_share(Jvec, prev_p, pop_vec,
                                        args.alpha_pop, args.beta, args.lambda_smooth, args.eps)

            if rule == "PERCENT":
                p = solve_week_percent(Jvec, prev_p, p_prior, [idx_list.index(i) for i in E_idx_local], save,
                                       args.lambda_smooth, args.tau_entropy, args.eps)
            else:
                p = solve_week_rank(Jvec, p_prior, [idx_list.index(i) for i in E_idx_local], save,
                                    args.kappa_rank, ilp_time_limit=args.ilp_time_limit)

            q = Jvec / np.sum(Jvec)
            momentum = p - prev_p

            # metrics
            if E_idx_local:
                pred = predict_elimination(rule, save, Jvec, p)
                true_local = {idx_list.index(i) for i in E_idx_local}
                if save:
                    metrics["B2CR_denom"] += 1
                    if true_local.issubset(pred):
                        metrics["B2CR"] += 1
                else:
                    metrics["EMR_denom"] += 1
                    if true_local == pred:
                        metrics["EMR"] += 1

            # uncertainty (optional)
            U = None
            if args.uncertainty:
                if rule == "PERCENT":
                    U = uncertainty_percent(Jvec, [idx_list.index(i) for i in E_idx_local], save)
                else:
                    U = uncertainty_rank(Jvec, p_prior, [idx_list.index(i) for i in E_idx_local], save,
                                         args.kappa_rank, args.rank_samples, args.eps, args.ilp_time_limit)

            # write records
            for pos, i in enumerate(idx_list):
                rec = {
                    "season": season,
                    "week": w,
                    "contestant_id": int(i),
                    "celebrity_name": df.loc[i, "celebrity_name"],
                    "ballroom_partner": df.loc[i, "ballroom_partner"],
                    "J_total": float(Jvec[pos]),
                    "p_hat": float(p[pos]),
                    "V_hat": float(p[pos] * args.total_votes),
                    "judge_share": float(q[pos]),
                    "fan_judge_gap": float(p[pos] - q[pos]),
                    "momentum": float(momentum[pos]),
                    "popularity_score": float(pop_vec[pos]),
                    "eliminated": int(i in E_idx_local),
                }
                if U is not None:
                    rec["uncertainty_width"] = float(U[pos])
                records.append(rec)

            # update prev_p map
            prev_p_map = {i: p[idx_list.index(i)] for i in idx_list}
            for pos, i in enumerate(idx_list):
                pop_map[i] = (1.0 - args.rho_pop) * pop_vec[pos] + args.rho_pop * math.log(max(p[pos], args.eps))

    out_df = pd.DataFrame(records)
    out_csv = os.path.join(args.out_dir, "req1_vote_estimates.csv")
    out_df.to_csv(out_csv, index=False)

    features_cols = [
        "season",
        "week",
        "contestant_id",
        "celebrity_name",
        "ballroom_partner",
        "J_total",
        "p_hat",
        "V_hat",
        "judge_share",
        "fan_judge_gap",
        "momentum",
        "popularity_score",
        "eliminated",
    ]
    if "uncertainty_width" in out_df.columns:
        features_cols.append("uncertainty_width")
    features_df = out_df[features_cols]
    features_csv = os.path.join(args.out_dir, "req1_features.csv")
    features_df.to_csv(features_csv, index=False)

    metrics_out = {
        "EMR": metrics["EMR"] / metrics["EMR_denom"] if metrics["EMR_denom"] else None,
        "B2CR": metrics["B2CR"] / metrics["B2CR_denom"] if metrics["B2CR_denom"] else None,
        "EMR_denom": metrics["EMR_denom"],
        "B2CR_denom": metrics["B2CR_denom"],
    }
    with open(os.path.join(args.out_dir, "req1_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {out_csv}")
    print("Metrics:", metrics_out)


if __name__ == "__main__":
    main()
