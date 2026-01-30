import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser(description="Summarize Requirement 1 outputs with tables and plots.")
    ap.add_argument("--votes", default="outputs/req1_vote_estimates.csv", help="vote estimates CSV")
    ap.add_argument("--out-dir", default="outputs/summary", help="output directory")
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


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.votes)

    # Season metrics (EMR/B2CR)
    season_rows = []
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
        season_rows.append({
            "season": season,
            "rule": rule,
            "judge_save": save,
            "EMR": emr_num / emr_den if emr_den else np.nan,
            "EMR_denom": emr_den,
            "B2CR": b2_num / b2_den if b2_den else np.nan,
            "B2CR_denom": b2_den,
        })

    season_metrics = pd.DataFrame(season_rows).sort_values("season")
    season_metrics.to_csv(os.path.join(args.out_dir, "season_metrics.csv"), index=False)

    # Plot season metrics
    fig, ax = plt.subplots(figsize=(10, 4))
    x = season_metrics["season"].to_numpy()
    emr = season_metrics["EMR"].to_numpy()
    b2 = season_metrics["B2CR"].to_numpy()
    ax.plot(x, emr, marker="o", label="EMR (no judge-save)")
    ax.plot(x, b2, marker="s", label="B2CR (judge-save)")
    ax.set_xlabel("Season")
    ax.set_ylabel("Rate")
    ax.set_title("Per-season Consistency Metrics")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "season_metrics.png"), dpi=150)
    plt.close(fig)

    # Weekly uncertainty (if exists)
    if "uncertainty_width" in df.columns:
        weekly_unc = (
            df.groupby(["season", "week"]) ["uncertainty_width"].mean().reset_index()
        )
        weekly_unc.to_csv(os.path.join(args.out_dir, "weekly_uncertainty.csv"), index=False)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(weekly_unc["week"], weekly_unc["uncertainty_width"], alpha=0.7)
        ax.set_xlabel("Week")
        ax.set_ylabel("Avg uncertainty width")
        ax.set_title("Weekly Uncertainty (Average)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "weekly_uncertainty.png"), dpi=150)
        plt.close(fig)

        # Heatmap for clearer season-week pattern
        pivot = weekly_unc.pivot(index="season", columns="week", values="uncertainty_width")
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Week")
        ax.set_ylabel("Season")
        ax.set_title("Weekly Uncertainty Heatmap (Season x Week)")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Avg uncertainty width")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "weekly_uncertainty_heatmap.png"), dpi=150)
        plt.close(fig)
    else:
        with open(os.path.join(args.out_dir, "weekly_uncertainty.txt"), "w", encoding="utf-8") as f:
            f.write("uncertainty_width column not found. Re-run req1_solve.py with --uncertainty to generate it.\n")

    # Controversial contestants profile
    controversial = {
        (2, "Jerry Rice"),
        (4, "Billy Ray Cyrus"),
        (11, "Bristol Palin"),
        (27, "Bobby Bones"),
    }
    prof = df[["season", "week", "celebrity_name", "J_total", "p_hat", "eliminated"]].copy()
    prof = prof[prof.apply(lambda r: (int(r["season"]), r["celebrity_name"]) in controversial, axis=1)]
    prof.to_csv(os.path.join(args.out_dir, "controversy_profiles.csv"), index=False)

    # Plot profiles
    if not prof.empty:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=False)
        axes = axes.flatten()
        for ax, (season, name) in zip(axes, sorted(controversial)):
            sub = prof[(prof["season"] == season) & (prof["celebrity_name"] == name)].sort_values("week")
            if sub.empty:
                ax.axis("off")
                ax.set_title(f"S{season}: {name} (no data)")
                continue
            ax.plot(sub["week"], sub["J_total"], marker="o", label="J_total")
            ax2 = ax.twinx()
            ax2.plot(sub["week"], sub["p_hat"], marker="s", color="tab:orange", label="p_hat")
            ax.set_title(f"S{season}: {name}")
            ax.set_xlabel("Week")
            ax.set_ylabel("J_total")
            ax2.set_ylabel("p_hat")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "controversy_profiles.png"), dpi=150)
        plt.close(fig)

    print(f"Wrote summaries to: {args.out_dir}")


if __name__ == "__main__":
    main()
