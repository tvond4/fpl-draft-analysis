"""
FPL Draft: which factors lead to head-to-head wins? (logistic regression)

How to run (from this folder):
    .venv/bin/python win_factors.py

One row = one manager in one gameweek match. Outcome: won (1) or not (0).
Draws are dropped, since they are neither a win nor a loss.

Why the predictors look the way they do
---------------------------------------
Position is categorical (GKP / DEF / MID / FWD). We bring it in by splitting each
team's weekly score into the points scored by each position group. That is the
same as giving every player's points a position label and adding them up.

We use YOUR points only, not your opponent's. If we used both, total points
would decide every result exactly and the model would break (statisticians
call this "perfect separation"). Using your own points asks a fairer question:
not knowing who you face, which positions' points raise your chance of winning?

Every predictor is "standardised" (converted to a z-score: how many standard
deviations above or below its average a value is). Then the
odds ratios compare like with like: "a typically good week from that position".
A raw point is worth the same anywhere, so per-point effects should be similar.
What differs is how much each position SWINGS from week to week.

Each match shows up twice (once per manager), and those two rows are linked.
Standard errors are therefore "clustered" by match, which is more cautious.

Models
------
1. Positions:  win ~ GKP + DEF + MID + FWD points
2. Decisions:  win ~ points left on bench + starters who didn't play
   Tested on their own. Adding them to model 1 would be meaningless: once you
   know your points by position, bench mistakes are already counted in them.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from draft_grades import LEAGUE_ID, POSITIONS, get, manager_labels

POS = ["GKP", "DEF", "MID", "FWD"]


def build():
    game = get("game")
    last_done = game["current_event"] if game["current_event_finished"] else game["current_event"] - 1
    details = get(f"league/{LEAGUE_ID}/details", refresh=False)
    boot = get("bootstrap-static", refresh=False)
    pos_of = {e["id"]: POSITIONS[e["element_type"]] for e in boot["elements"]}
    entry_of = {e["id"]: e["entry_id"] for e in details["league_entries"]}
    name_of = manager_labels(details)
    weekly = pd.read_excel("bench_points.xlsx", sheet_name="Every week")
    lost = {(r.gw, r.entry_id): r.lost for r in weekly.itertuples()}

    rows = []
    for mt in details["matches"]:
        gw = mt["event"]
        if gw > last_done:
            continue
        live = get(f"event/{gw}/live", refresh=False)["elements"]
        stat = lambda pid, k: live.get(str(pid), {"stats": {k: 0}})["stats"][k]
        sides = [(mt["league_entry_1"], mt["league_entry_1_points"], mt["league_entry_2_points"]),
                 (mt["league_entry_2"], mt["league_entry_2_points"], mt["league_entry_1_points"])]
        for le, score, opp in sides:
            eid = entry_of[le]
            team = get(f"entry/{eid}/event/{gw}", refresh=False)
            starters = {p["element"] for p in team["picks"] if p["position"] <= 11}
            for s in team["subs"]:
                starters.discard(s["element_out"])
                starters.add(s["element_in"])
            row = {"gw": gw, "match": f"{gw}-{min(sides[0][0], sides[1][0])}", "manager": name_of[eid],
                   "score": score, "opp_score": opp,
                   "result": "W" if score > opp else "D" if score == opp else "L"}
            for p in POS:
                row[p] = sum(stat(pid, "total_points") for pid in starters if pos_of[pid] == p)
                row[f"n_{p}"] = sum(1 for pid in starters if pos_of[pid] == p)
                row[f"{p}_per_player"] = row[p] / row[f"n_{p}"]
            row["blank_starters"] = sum(1 for pid in starters if stat(pid, "minutes") == 0)
            row["bench_lost"] = lost[(gw, eid)]
            rows.append(row)
    df = pd.DataFrame(rows)
    df["win"] = (df["result"] == "W").astype(int)
    return df, last_done


def fit(df, cols):
    X = (df[cols] - df[cols].mean()) / df[cols].std()        # z-scores
    X = sm.add_constant(X)
    m = sm.Logit(df["win"], X).fit(disp=0, cov_type="cluster",
                                   cov_kwds={"groups": pd.factorize(df["match"])[0]})
    ci = np.exp(m.conf_int())
    out = pd.DataFrame({
        "1_sd_equals_pts": df[cols].std().reindex(X.columns),
        "odds_ratio": np.exp(m.params),
        "odds_ratio_per_point": np.exp(m.params / df[cols].std().reindex(X.columns)),
        "or_low_95": ci[0], "or_high_95": ci[1],
        "p_value": m.pvalues,
    }).drop(index="const")
    pred = (m.predict(X) >= 0.5).astype(int)
    fitinfo = {"n": int(m.nobs), "mcfadden_r2": m.prsquared,
               "accuracy_pct": (pred == df["win"]).mean() * 100,
               "base_rate_pct": max(df["win"].mean(), 1 - df["win"].mean()) * 100}
    return out, fitinfo, m


def main():
    df, last_done = build()
    played = df[df["result"] != "D"].reset_index(drop=True)

    # Simple picture first: average points by position in wins vs losses
    by_result = played.groupby("result")[POS + ["score"]].mean().T
    by_result["win_minus_loss"] = by_result["W"] - by_result["L"]
    by_result["week_to_week_sd"] = played[POS + ["score"]].std()

    m1, f1, _ = fit(played, POS)
    m2, f2, _ = fit(played, ["bench_lost", "blank_starters"])
    corr = played[POS + ["bench_lost", "blank_starters"]].corr().round(2)

    with pd.ExcelWriter("win_factors.xlsx") as xw:
        by_result.round(2).to_excel(xw, sheet_name="Wins vs losses")
        m1.round(3).to_excel(xw, sheet_name="Model 1 positions")
        m2.round(3).to_excel(xw, sheet_name="Model 2 decisions")
        pd.DataFrame([f1, f2], index=["model_1", "model_2"]).round(3).to_excel(xw, sheet_name="Model fit")
        corr.to_excel(xw, sheet_name="Correlations")
        df.to_excel(xw, sheet_name="Data", index=False)

    pd.set_option("display.width", 200)
    print(f"Through GW{last_done}. {len(played)} rows ({len(df) - len(played)} draw rows dropped).\n")
    print(by_result.round(2), "\n")
    print("MODEL 1", {k: round(v, 3) for k, v in f1.items()}); print(m1.round(3), "\n")
    print("MODEL 2", {k: round(v, 3) for k, v in f2.items()}); print(m2.round(3), "\n")
    print(corr)


if __name__ == "__main__":
    main()
