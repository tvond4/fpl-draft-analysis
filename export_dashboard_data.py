"""
Turn the analysis spreadsheets into small JSON files for the dashboard.

Run after draft_grades.py, bench_points.py, win_factors.py and win_models.py:
    .venv/bin/python export_dashboard_data.py
Files land in ./dashboard_data. Then ask Claude to upload them to the dashboard.
"""

import json
from pathlib import Path

import pandas as pd

from draft_grades import my_name
from win_factors import build

OUT = Path(__file__).parent / "dashboard_data"
OUT.mkdir(exist_ok=True)
first = lambda m: m.split(" (")[0]


def save(name, df):
    rows = json.loads(df.to_json(orient="records"))
    (OUT / f"{name}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print(name, len(rows), "rows")


d = pd.read_excel("draft_grades.xlsx", "Manager summary")
save("draft", pd.DataFrame({
    "manager": d["manager"].map(first), "team": d["manager"].str.extract(r"\((.*)\)")[0],
    "draft_score": d["draft_score"].round(1), "early": d["early_score_r1_5"].round(1),
    "late": d["late_score_r6_15"].round(1), "total_points": d["total_points"],
    "best_pick": d["best_pick"], "worst_pick": d["worst_pick"], "auto_picks": d["auto_picks"]}))

b = pd.read_excel("bench_points.xlsx", "Manager summary")
save("calls", pd.DataFrame({
    "manager": b["manager"].map(first), "accuracy": b["call_accuracy_pct"].round(1),
    "low": b["accuracy_low_90"].round(1), "high": b["accuracy_high_90"].round(1),
    "calls": b["calls"], "right_calls": b["right_calls"], "points_lost": b["points_lost"],
    "record": b["record_W_D_L"], "matches_lost": b["matches_lost_to_lineup"]}))

w = pd.read_excel("win_models.xlsx", "Win chance by points")
save("winchance", pd.DataFrame({
    "score": w["score"], "h2h": w["win_pct_head_to_head"].round(1), "allplay": w["win_pct_all_play"].round(1),
    "raw": w["raw_all_play_pct"].round(1), "raw_n": w["raw_n_comparisons"]}))

o = pd.read_excel("win_factors.xlsx", "Model 1 positions", index_col=0)
save("odds", pd.DataFrame({
    "position": o.index, "sd_pts": o["1_sd_equals_pts"].round(1).values, "odds_ratio": o["odds_ratio"].round(2).values,
    "low": o["or_low_95"].round(2).values, "high": o["or_high_95"].round(2).values,
    "p_value": o["p_value"].round(3).values, "per_point": o["odds_ratio_per_point"].round(2).values}))

m = pd.read_excel("win_models.xlsx", "Model comparison", index_col=0)
labels = {"A_totals": "Totals by position", "B_per_player": "Per player",
          "C_per_player_plus_counts": "Per player + number started"}
save("models", pd.DataFrame({
    "model": [labels[i] for i in m.index], "cv_log_loss": m["cv_log_loss"].round(3).values,
    "cv_accuracy": m["cv_accuracy_pct"].round(1).values, "aic": m["AIC"].round(1).values}))

# Single-row headline facts
df, last_done = build()
played = df[df.result != "D"]
import statsmodels.api as sm
fit = sm.Logit(played["win"], sm.add_constant(played[["score"]].astype(float))).fit(disp=0)
save("summary", pd.DataFrame([{
    "through_gw": last_done,
    "break_even": round(-fit.params["const"] / fit.params["score"], 1),
    "league_accuracy": round(b["right_calls"].sum() / b["calls"].sum() * 100, 1),
    "matches_flipped": int(b["matches_lost_to_lineup"].sum()),
    "matches_played": int(len(df) / 2)}]))

# ---- Free agents, waiver tool and season outlook ----
f = pd.read_excel("fa_war.xlsx", "Manager summary")
save("fa", pd.DataFrame({
    "manager": f["manager"], "moves": f["moves"], "total_war": f["total_war"].round(1),
    "war_per_move": f["war_per_move"].round(1), "hit_rate": f["hit_rate_pct"].round(0),
    "low": f["war_low_90"].round(1), "high": f["war_high_90"].round(1),
    "best_move": f["best_move"], "worst_move": f["worst_move"]}))

wv = pd.read_excel("waiver_tool.xlsx", "Top 5 by position")
save("waivers", pd.DataFrame({
    "player": wv["web_name"], "pos": wv["pos"], "team": wv["team"], "proj_next_4": wv["proj_next_4"].round(1),
    "drop": wv["your_weakest"], "gain": wv["gain_vs_your_weakest"].round(1),
    "gain_if_fit": wv["gain_if_all_fit"].round(1), "status": wv["status"]}))

sc = pd.read_excel("season_sim.xlsx", "Your scenarios")
model_name = {"A_results": "Results so far", "B_squads": "Current squads"}
save("outlook", pd.DataFrame({
    "scenario": [f"{model_name[m]} +{b}" for m, b in zip(sc.strength_model, sc.boost_per_week)],
    "model": sc.strength_model.map(model_name), "boost": sc.boost_per_week,
    "avg_finish": sc.avg_finish.round(1), "p_top2": sc.p_top2.round(1),
    "p_middle6": sc.p_middle6.round(1), "p_bottom2": sc.p_bottom2.round(1)}))

ts = pd.read_excel("season_sim.xlsx", "Team strengths", index_col=0)
s = json.loads((OUT / "summary.json").read_text())[0]
s.update({"my_expected_wins": float(ts.loc[my_name(), "expected_wins"]), "my_actual_wins": int(ts.loc[my_name(), "actual_wins"])})
(OUT / "summary.json").write_text(json.dumps([s], indent=1, ensure_ascii=False))
print("summary updated", s)

# ---- Stash watchlist (injured or barely played, with web research) ----
st = pd.read_json("stash_candidates.json")
src = pd.DataFrame(json.loads(Path("stash_news.json").read_text())).set_index("player")["sources"]
save("stash", pd.DataFrame({
    "player": st.player, "pos": st.pos, "team": st.team, "group": st.group,
    "minutes": st.minutes, "upside_per_week": st.upside_per_week.round(2),
    "weeks_out": st.weeks_out, "role": st.role_outlook, "confidence": st.confidence,
    "stash_value_8wk": st.stash_value_8wk.round(1), "your_weakest": st.your_weakest,
    "gain_vs_weakest_8wk": st.gain_vs_weakest_8wk.round(1), "news": st.news_summary,
    "news_date": st.latest_news_date.astype(str), "source": st.player.map(lambda p: (src.get(p) or [""])[0])}))
