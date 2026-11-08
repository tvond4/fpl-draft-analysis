"""
FPL Draft: (1) chance of winning for each points total, and
           (2) which regression explains wins best: position TOTALS or PER-PLAYER averages?

How to run (from this folder, after bench_points.py):
    .venv/bin/python win_models.py

Part 1: win probability by points
---------------------------------
Two estimates:
  a) Head-to-head: logistic regression of win ~ your score, using the real
     matches (86 rows, draws dropped).
  b) "All-play": compare each score with every OTHER manager's score that same
     week (9 comparisons each, 810 in total). This answers "if you had faced a
     random opponent this week, how often would you win?" It removes the luck
     of who you were drawn against and uses ~9x more comparisons.

Part 2: which model is better?
------------------------------
  A. Totals:            points from GKP, DEF, MID, FWD
  B. Per player:        average points per starter at each position
  C. Per player + size: model B plus how many DEF and FWD you started
                        (MID count is then implied: 10 outfield slots in total)

How we judge "better": each model is refit 9 times, each time leaving out one
gameweek, and then predicts the week it never saw. This is "cross-validation":
it rewards models that predict new weeks, not ones that just memorise. Scores:
  log_loss  how surprised the model was by real results (lower is better)
  brier     average squared error of the predicted chance (lower is better)
  accuracy  share of matches called correctly
  AIC       in-sample fit with a penalty for extra variables (lower is better)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from win_factors import POS, build

MODELS = {
    "A_totals": POS,
    "B_per_player": [f"{p}_per_player" for p in POS],
    "C_per_player_plus_counts": [f"{p}_per_player" for p in POS] + ["n_DEF", "n_FWD"],
}


def logit(df, cols, y="win", groups=None):
    X = sm.add_constant(df[cols].astype(float))
    kw = {"cov_type": "cluster", "cov_kwds": {"groups": groups}} if groups is not None else {}
    return sm.Logit(df[y], X).fit(disp=0, **kw)


def win_probability(df, played):
    # a) head-to-head
    h2h = logit(played, ["score"], groups=pd.factorize(played["match"])[0])
    # b) all-play: every score vs every other score that week
    pairs = []
    for gw, wk in df.groupby("gw"):
        for i, a in wk.iterrows():
            for j, b in wk.iterrows():
                if a["manager"] != b["manager"] and a["score"] != b["score"]:
                    pairs.append({"gw": gw, "manager": a["manager"], "score": a["score"],
                                  "win": int(a["score"] > b["score"])})
    pairs = pd.DataFrame(pairs)
    allplay = logit(pairs, ["score"], groups=pd.factorize(pairs["gw"].astype(str) + pairs["manager"])[0])

    grid = pd.DataFrame({"score": range(20, 75, 5)})
    Xg = sm.add_constant(grid[["score"]].astype(float), has_constant="add")
    grid["win_pct_head_to_head"] = h2h.predict(Xg) * 100
    grid["win_pct_all_play"] = allplay.predict(Xg) * 100
    # Raw check: actual all-play win rate for scores within +/-5 of each value
    grid["raw_all_play_pct"] = [pairs.loc[(pairs.score - s).abs() <= 5, "win"].mean() * 100 for s in grid.score]
    grid["raw_n_comparisons"] = [int(((pairs.score - s).abs() <= 5).sum()) for s in grid.score]
    fifty = {"head_to_head": -h2h.params["const"] / h2h.params["score"],
             "all_play": -allplay.params["const"] / allplay.params["score"]}
    return grid, fifty, len(pairs)


def compare(played):
    rows, coefs = [], {}
    for name, cols in MODELS.items():
        # in-sample fit
        m = logit(played, cols, groups=pd.factorize(played["match"])[0])
        # leave-one-gameweek-out predictions
        pred = pd.Series(index=played.index, dtype=float)
        for gw in played["gw"].unique():
            train, test = played[played.gw != gw], played[played.gw == gw]
            mt = logit(train, cols)
            pred[test.index] = mt.predict(sm.add_constant(test[cols].astype(float), has_constant="add"))
        p = pred.clip(1e-6, 1 - 1e-6)
        y = played["win"]
        rows.append({
            "model": name,
            "n_variables": len(cols),
            "cv_log_loss": -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(),
            "cv_brier": ((p - y) ** 2).mean(),
            "cv_accuracy_pct": ((p >= 0.5) == y).mean() * 100,
            "AIC": m.aic,
            "mcfadden_r2": m.prsquared,
        })
        ci = np.exp(m.conf_int())
        coefs[name] = pd.DataFrame({"odds_ratio_per_unit": np.exp(m.params), "low_95": ci[0],
                                    "high_95": ci[1], "p_value": m.pvalues}).drop(index="const")
    return pd.DataFrame(rows).set_index("model"), coefs


def main():
    df, last_done = build()
    played = df[df["result"] != "D"].reset_index(drop=True)

    grid, fifty, n_pairs = win_probability(df, played)
    scores, coefs = compare(played)

    with pd.ExcelWriter("win_models.xlsx") as xw:
        grid.round(1).to_excel(xw, sheet_name="Win chance by points", index=False)
        scores.round(3).to_excel(xw, sheet_name="Model comparison")
        for name, c in coefs.items():
            c.round(3).to_excel(xw, sheet_name=name[:31])

    pd.set_option("display.width", 200)
    print(f"Through GW{last_done}. {len(played)} H2H rows, {n_pairs} all-play comparisons.")
    print("50% break-even score:", {k: round(v, 1) for k, v in fifty.items()}, "\n")
    print(grid.round(1).to_string(index=False), "\n")
    print(scores.round(3), "\n")
    for name, c in coefs.items():
        print(name); print(c.round(3), "\n")


if __name__ == "__main__":
    main()
