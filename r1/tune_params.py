import argparse
import json
import os
import subprocess
import sys
import time
from itertools import product

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(description="Parameter tuning for req1_solve.py")
    ap.add_argument("--tmp-out-dir", default="outputs/tune_tmp", help="temporary output directory")
    ap.add_argument("--results-dir", default="outputs/tuning", help="results output directory")
    ap.add_argument("--final-out-dir", default="outputs", help="final output directory")
    ap.add_argument("--ilp-time-limit", type=int, default=10, help="ILP time limit seconds")
    ap.add_argument("--skip-final", action="store_true", help="skip final best run")
    return ap.parse_args()


def rule_type(season: int) -> str:
    return "PERCENT" if 3 <= season <= 27 else "RANK"


def has_judge_save(season: int) -> bool:
    return season >= 28


def rank_desc(vec: np.ndarray) -> np.ndarray:
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


def predict_elimination(rule, save, Jvec, p):
    if rule == "PERCENT":
        q = Jvec / np.sum(Jvec)
        C = q + p
        if save:
            return set(np.argsort(C)[:2])
        return set(np.argsort(C)[:1])
    rJ = rank_desc(Jvec)
    rV = rank_desc(p)
    ranksum = rJ + rV
    if save:
        return set(np.argsort(ranksum)[-2:])
    return set(np.argsort(ranksum)[-1:])


def compute_season_stats(votes_csv):
    df = pd.read_csv(votes_csv)
    emr_by_season = []
    b2_by_season = []

    for season, sdf in df.groupby("season"):
        season = int(season)
        rule = rule_type(season)
        save = has_judge_save(season)
        emr_num = emr_den = 0
        b2_num = b2_den = 0
        for week, wdf in sdf.groupby("week"):
            if wdf["eliminated"].sum() == 0:
                continue
            Jvec = wdf["J_total"].to_numpy(dtype=float)
            p = wdf["p_hat"].to_numpy(dtype=float)
            pred = predict_elimination(rule, save, Jvec, p)
            true_idx = set(np.where(wdf["eliminated"].to_numpy(dtype=int) == 1)[0])
            if save:
                b2_den += 1
                if true_idx.issubset(pred):
                    b2_num += 1
            else:
                emr_den += 1
                if true_idx == pred:
                    emr_num += 1
        if save and b2_den:
            b2_by_season.append(b2_num / b2_den)
        if (not save) and emr_den:
            emr_by_season.append(emr_num / emr_den)

    std_emr = float(np.std(emr_by_season)) if emr_by_season else 0.0
    std_b2 = float(np.std(b2_by_season)) if b2_by_season else 0.0
    return std_emr, std_b2


def run_req1(python_exec, out_dir, alpha_pop, beta, lambda_smooth, rho_pop, kappa_rank, ilp_time_limit):
    cmd = [
        python_exec,
        "req1_solve.py",
        "--out-dir",
        out_dir,
        "--alpha-pop",
        str(alpha_pop),
        "--beta",
        str(beta),
        "--lambda-smooth",
        str(lambda_smooth),
        "--rho-pop",
        str(rho_pop),
        "--kappa-rank",
        str(kappa_rank),
        "--ilp-time-limit",
        str(ilp_time_limit),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def score_result(emr, b2cr, std_emr, std_b2):
    return 0.5 * emr + 0.5 * b2cr - 0.1 * std_emr - 0.1 * std_b2


def main():
    args = parse_args()
    os.makedirs(args.tmp_out_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    python_exec = sys.executable

    grid_alpha = [0.4, 0.6, 0.8, 1.0]
    grid_beta = [0.8, 1.0, 1.2, 1.4]
    grid_lambda = [0.3, 0.5, 0.7]
    grid_rho = [0.1, 0.2, 0.3]
    grid_kappa = [0.3, 0.4, 0.5, 0.6]

    stage_a = []
    start = time.time()
    fixed_kappa = 0.4
    combos = list(product(grid_alpha, grid_beta, grid_lambda, grid_rho))
    for idx, (alpha_pop, beta, lambda_smooth, rho_pop) in enumerate(combos, start=1):
        run_req1(
            python_exec,
            args.tmp_out_dir,
            alpha_pop,
            beta,
            lambda_smooth,
            rho_pop,
            fixed_kappa,
            args.ilp_time_limit,
        )
        metrics_path = os.path.join(args.tmp_out_dir, "req1_metrics.json")
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        std_emr, std_b2 = compute_season_stats(os.path.join(args.tmp_out_dir, "req1_vote_estimates.csv"))
        emr = float(metrics.get("EMR") or 0.0)
        b2cr = float(metrics.get("B2CR") or 0.0)
        score = score_result(emr, b2cr, std_emr, std_b2)
        stage_a.append({
            "alpha_pop": alpha_pop,
            "beta": beta,
            "lambda_smooth": lambda_smooth,
            "rho_pop": rho_pop,
            "kappa_rank": fixed_kappa,
            "EMR": emr,
            "B2CR": b2cr,
            "std_EMR": std_emr,
            "std_B2CR": std_b2,
            "score": score,
        })
        if idx % 10 == 0:
            elapsed = time.time() - start
            print(f"Stage A {idx}/{len(combos)} done ({elapsed:.1f}s)")

    stage_a_df = pd.DataFrame(stage_a).sort_values("score", ascending=False)
    stage_a_path = os.path.join(args.results_dir, "stage_a_results.csv")
    stage_a_df.to_csv(stage_a_path, index=False)
    top10 = stage_a_df.head(10)

    stage_b = []
    for _, row in top10.iterrows():
        alpha_pop = row["alpha_pop"]
        beta = row["beta"]
        lambda_smooth = row["lambda_smooth"]
        rho_pop = row["rho_pop"]
        for kappa in grid_kappa:
            run_req1(
                python_exec,
                args.tmp_out_dir,
                alpha_pop,
                beta,
                lambda_smooth,
                rho_pop,
                kappa,
                args.ilp_time_limit,
            )
            metrics_path = os.path.join(args.tmp_out_dir, "req1_metrics.json")
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            std_emr, std_b2 = compute_season_stats(os.path.join(args.tmp_out_dir, "req1_vote_estimates.csv"))
            emr = float(metrics.get("EMR") or 0.0)
            b2cr = float(metrics.get("B2CR") or 0.0)
            score = score_result(emr, b2cr, std_emr, std_b2)
            stage_b.append({
                "alpha_pop": alpha_pop,
                "beta": beta,
                "lambda_smooth": lambda_smooth,
                "rho_pop": rho_pop,
                "kappa_rank": kappa,
                "EMR": emr,
                "B2CR": b2cr,
                "std_EMR": std_emr,
                "std_B2CR": std_b2,
                "score": score,
            })

    stage_b_df = pd.DataFrame(stage_b).sort_values("score", ascending=False)
    stage_b_path = os.path.join(args.results_dir, "stage_b_results.csv")
    stage_b_df.to_csv(stage_b_path, index=False)

    final_df = stage_b_df if not stage_b_df.empty else stage_a_df
    top_params = final_df.head(5)
    top_path = os.path.join(args.results_dir, "top_params.csv")
    top_params.to_csv(top_path, index=False)

    best = top_params.iloc[0]
    print("Best params:", best.to_dict())

    if not args.skip_final:
        run_req1(
            python_exec,
            args.final_out_dir,
            best["alpha_pop"],
            best["beta"],
            best["lambda_smooth"],
            best["rho_pop"],
            best["kappa_rank"],
            args.ilp_time_limit,
        )
        subprocess.run(
            [python_exec, "req1_summary.py", "--votes", os.path.join(args.final_out_dir, "req1_vote_estimates.csv")],
            check=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    main()
