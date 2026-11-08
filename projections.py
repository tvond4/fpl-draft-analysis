"""
FPL Draft: project each player's points, backtest the projection, and build the
waiver tool for your team.

How to run (from this folder, after fa_war.py):
    .venv/bin/python projections.py

The projection
--------------
For one player in one gameweek, points per match are predicted from what was
known before the deadline:
  form_4        average points over the last 4 weeks
  season_avg    average points over all earlier weeks
  minutes_4     average minutes over the last 4 weeks, as a share of 90
  difficulty    FPL's fixture difficulty rating for that match (1 easy, 5 hard)
  home          1 if at home
  position      GKP / DEF / MID / FWD (categorical)
Weekly projection = points per match x number of matches that week (handles
blank and double gameweeks).

The backtest
------------
For each of weeks 6 to 9, the model is fit only on earlier weeks and then
predicts that week. We compare it with two simple rules ("last 4 weeks' form"
and "season average") using the average miss in points (MAE, lower is better)
and the correlation with what actually happened.

The waiver tool
---------------
Every current free agent gets a projection for the next 4 gameweeks, scaled
for injury news (FPL's chance of playing). "Gain" is how many more points he
is projected to score than your weakest player at the same position, the one
you would drop. Results go to waiver_tool.xlsx.
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

from draft_grades import LEAGUE_ID, MY_ENTRY_ID, POSITIONS, get

# Goalkeepers have no defensive-contribution stat, so the per-position fit of
# that variant is rank-deficient. Harmless; it only affects a losing variant.
warnings.filterwarnings("ignore", message="The design matrix is rank-deficient")

ME = MY_ENTRY_ID     # your team, from config.json
HORIZON = 4         # weeks ahead to project
BASE_FEATS = ["form_4", "season_avg", "minutes_4", "difficulty", "home", "DEF", "MID", "FWD"]
# Underlying stats, averaged over the last 4 weeks. These are less noisy than points.
EXTRA_STATS = ["expected_goal_involvements", "bps", "starts", "expected_goals_conceded", "defensive_contribution"]
RICH_FEATS = BASE_FEATS + [f"{k}_4" for k in EXTRA_STATS]
RATE_FEATS = ["rate_x_min", "minutes_4", "difficulty", "home"]
SHRINK_MATCHES = 3      # how many "average" matches each player's rate starts with
FEATS = RATE_FEATS      # default; main() uses whichever variant wins the backtest
EXTRA = {}              # (player, gw) -> {stat: value}


def load():
    game = get("game")
    last_done = game["current_event"] if game["current_event_finished"] else game["current_event"] - 1
    boot = get("bootstrap-static", refresh=False)
    players = pd.DataFrame(boot["elements"]).set_index("id")
    players["pos"] = players["element_type"].map(POSITIONS)
    fixtures = pd.DataFrame(get_classic("fixtures/"))
    pts, mins = {}, {}
    for gw in range(1, last_done + 1):
        for pid, row in get(f"event/{gw}/live", refresh=False)["elements"].items():
            st = row["stats"]
            pts[(int(pid), gw)] = st["total_points"]
            mins[(int(pid), gw)] = st["minutes"]
            EXTRA[(int(pid), gw)] = {k: float(st.get(k) or 0) for k in EXTRA_STATS}
    return game, last_done, players, fixtures, pts, mins


def get_classic(path):
    """Fixture list (with difficulty ratings) from the main FPL game's public feed."""
    import json, urllib.request
    from draft_grades import DATA
    out = DATA / ("classic_" + path.strip("/").replace("/", "_") + ".json")
    req = urllib.request.Request(f"https://fantasy.premierleague.com/api/{path}", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        out.write_bytes(r.read())
    return json.loads(out.read_text())


def team_matches(fixtures):
    """One row per team per match: gw, team, difficulty, home."""
    f = fixtures.dropna(subset=["event"])
    h = pd.DataFrame({"gw": f.event, "team": f.team_h, "difficulty": f.team_h_difficulty, "home": 1})
    a = pd.DataFrame({"gw": f.event, "team": f.team_a, "difficulty": f.team_a_difficulty, "home": 0})
    tm = pd.concat([h, a])
    tm["gw"] = tm["gw"].astype(int)
    return tm


def features(players, pts, mins, tm, gw, known_through):
    """Feature rows for every player for target week `gw`, using weeks <= known_through."""
    weeks = list(range(1, known_through + 1))
    last4 = weeks[-4:]
    rows = []
    agg = tm[tm.gw == gw].groupby("team").agg(n_matches=("home", "size"), difficulty=("difficulty", "mean"), home=("home", "mean"))
    for pid, p in players.iterrows():
        f4 = np.mean([pts.get((pid, g), 0) for g in last4])
        rows.append({
            "pid": pid, "gw": gw, "pos": p.pos,
            "form_4": f4, "season_avg": np.mean([pts.get((pid, g), 0) for g in weeks]),
            "minutes_4": np.mean([mins.get((pid, g), 0) for g in last4]) / 90,
            "n_matches": agg["n_matches"].get(p.team, 0),
            "difficulty": agg["difficulty"].get(p.team, 3), "home": agg["home"].get(p.team, 0.5),
            **{f"{k}_4": np.mean([EXTRA.get((pid, g), {}).get(k, 0) for g in last4]) for k in EXTRA_STATS},
            "season_pts": sum(pts.get((pid, g), 0) for g in weeks),
            "season_mins": sum(mins.get((pid, g), 0) for g in weeks),
            # Minutes share in the weeks he did play: his role when available
            "played_share": np.mean([mins.get((pid, g), 0) for g in weeks if mins.get((pid, g), 0) > 0] or [0]) / 90,
        })
    df = pd.DataFrame(rows)
    # Points per 90 minutes, pulled toward the position average. A player with few
    # minutes is mostly the average; one with many minutes is mostly himself.
    regulars = df[df.season_mins >= 270]
    pos_p90 = regulars.groupby("pos").apply(lambda g: g.season_pts.sum() / g.season_mins.sum() * 90, include_groups=False)
    df["p90_shrunk"] = (df.season_pts + SHRINK_MATCHES * df.pos.map(pos_p90)) / (df.season_mins / 90 + SHRINK_MATCHES)
    df["rate_x_min"] = df.p90_shrunk * df.minutes_4.clip(upper=1)
    for pos in ["DEF", "MID", "FWD"]:
        df[pos] = (df.pos == pos).astype(int)
    return df


def training_rows(players, pts, mins, tm, weeks):
    out = []
    for g in weeks:
        f = features(players, pts, mins, tm, g, g - 1)
        f["actual"] = [pts.get((pid, g), 0) for pid in f.pid]
        out.append(f)
    df = pd.concat(out)
    # Only players who had some role recently; the rest are noise at 0 points
    return df[(df.minutes_4 > 0) & (df.n_matches > 0)]


def fit(train, feats=None, by_position=False):
    """Linear regression of points per match. by_position fits one model per position."""
    feats = feats or FEATS
    def one(df, cols):
        y = df["actual"] / df["n_matches"]
        return sm.OLS(y, sm.add_constant(df[cols].astype(float), has_constant="add")).fit()
    if by_position:
        cols = [c for c in feats if c not in ("DEF", "MID", "FWD")]
        return {"by_position": True, "cols": cols, "models": {p: one(g, cols) for p, g in train.groupby("pos")}}
    return {"by_position": False, "cols": feats, "model": one(train, feats)}


def predict(m, f):
    if m["by_position"]:
        out = pd.Series(0.0, index=f.index)
        for p, g in f.groupby("pos"):
            out[g.index] = m["models"][p].predict(sm.add_constant(g[m["cols"]].astype(float), has_constant="add"))
    else:
        out = m["model"].predict(sm.add_constant(f[m["cols"]].astype(float), has_constant="add"))
    return out.clip(lower=0) * f["n_matches"]


VARIANTS = {"points_only": (BASE_FEATS, False), "points_only_by_pos": (BASE_FEATS, True),
            "with_underlying_stats": (RICH_FEATS, False), "with_underlying_stats_by_pos": (RICH_FEATS, True),
            "rate_x_minutes": (RATE_FEATS, False), "rate_x_minutes_by_pos": (RATE_FEATS, True)}


def backtest(players, pts, mins, tm, last_done):
    rows, preds = [], []
    for test_gw in range(6, last_done + 1):
        train = training_rows(players, pts, mins, tm, range(5, test_gw))
        test = training_rows(players, pts, mins, tm, [test_gw])
        test = test.assign(form_rule=test.form_4 * test.n_matches, season_rule=test.season_avg * test.n_matches)
        for name, (feats, bp) in VARIANTS.items():
            test[name] = predict(fit(train, feats, bp), test)
        preds.append(test)
    allp = pd.concat(preds)
    for name in list(VARIANTS) + ["form_rule", "season_rule"]:
        rows.append({"method": name, "mae": (allp[name] - allp.actual).abs().mean(),
                     "correlation": allp[[name, "actual"]].corr().iloc[0, 1],
                     "rows": len(allp)})
    # Does the model pick good free-agent-type players? Top 30 projected each week
    scores = pd.DataFrame(rows).set_index("method").sort_values("mae")
    best = scores.index[0]
    top = allp.sort_values(best, ascending=False).groupby("gw").head(30)
    resid_sd = (allp[best] - allp.actual).std()
    return scores, top.actual.mean(), allp, resid_sd


def availability(p, week_ahead):
    """Share of a match the player is expected to be available for, from FPL injury news."""
    chance = p.chance_of_playing_next_round
    if week_ahead == 0:
        if pd.notna(chance):
            return chance / 100
        return 1.0 if p.status == "a" else 0.0
    return {"a": 1.0, "d": 0.75, "i": 0.5, "s": 0.5}.get(p.status, 0.0)


def waiver_tool(players, pts, mins, tm, last_done, next_gw, variant):
    feats, bp = VARIANTS[variant]
    model = fit(training_rows(players, pts, mins, tm, range(5, last_done + 1)), feats, bp)
    proj = pd.Series(0.0, index=players.index)
    healthy = pd.Series(0.0, index=players.index)
    by_week = {}
    for k, g in enumerate(range(next_gw, next_gw + HORIZON)):
        f = features(players, pts, mins, tm, g, last_done)
        raw = pd.Series(predict(model, f).values, index=f.pid)
        # An injured player's recent minutes are low, so "healthy" uses his season minutes instead
        fit_f = f.copy()
        # Only players flagged injured, doubtful or suspended get their usual minutes back;
        # a fit player with few recent minutes has lost his place, so he keeps them.
        flagged = f.pid.map(players["status"]).isin(["i", "d", "s"]).values
        fit_f["minutes_4"] = np.where(flagged, np.maximum(f.minutes_4, f.played_share), f.minutes_4)
        fit_f["rate_x_min"] = fit_f.p90_shrunk * fit_f.minutes_4.clip(upper=1)
        healthy += pd.Series(predict(model, fit_f).values, index=f.pid)
        wk = raw * np.array([availability(players.loc[pid], k) for pid in raw.index])
        by_week[f"gw{g}"] = wk
        proj += wk
    status = pd.DataFrame(get(f"league/{LEAGUE_ID}/element-status")["element_status"]).set_index("element")
    t = players[["web_name", "pos", "team", "status", "chance_of_playing_next_round", "news"]].copy()
    t["owner"] = status["owner"]
    t["proj_next_4"] = proj.round(1)
    t["proj_if_fit"] = healthy.round(1)
    for k, v in by_week.items():
        t[k] = v.round(1)
    t["form_4"] = [np.mean([pts.get((pid, g), 0) for g in range(last_done - 3, last_done + 1)]) for pid in t.index]
    teams = {x["id"]: x["short_name"] for x in get("bootstrap-static", refresh=False)["teams"]}
    t["team"] = t["team"].map(teams)

    # Drop candidate = your weakest player at each position when everyone is fit,
    # so a short-term injury to a good player (e.g. Saka) doesn't make him the drop.
    mine = t[t.owner == ME].sort_values(["pos", "proj_if_fit"])
    drop = mine.groupby("pos").head(1).set_index("pos")
    fa = t[t.owner.isna() & players["status"].ne("u")].copy()
    fa["your_weakest"] = fa.pos.map(drop["web_name"])
    fa["weakest_proj"] = fa.pos.map(drop["proj_next_4"])
    fa["gain_vs_your_weakest"] = (fa.proj_next_4 - fa.weakest_proj).round(1)
    # Same comparison with everyone fit: does the upgrade last beyond a short injury?
    fa["gain_if_all_fit"] = (fa.proj_if_fit - fa.pos.map(drop["proj_if_fit"])).round(1)
    # Projected WAR: points above the 5th-best free agent projection at the position
    repl = fa.groupby("pos")["proj_next_4"].apply(lambda s: s.nlargest(5).mean())
    fa["proj_war"] = (fa.proj_next_4 - fa.pos.map(repl)).round(1)
    fa = fa.sort_values("gain_vs_your_weakest", ascending=False)
    return model, mine, fa, t


def main():
    game, last_done, players, fixtures, pts, mins = load()
    tm = team_matches(fixtures)
    scores, top_actual, allp, resid_sd = backtest(players, pts, mins, tm, last_done)
    best = scores.index[0] if scores.index[0] in VARIANTS else [i for i in scores.index if i in VARIANTS][0]
    model, mine, fa, _ = waiver_tool(players, pts, mins, tm, last_done, game["next_event"], best)

    cols = ["web_name", "pos", "team", "proj_next_4"] + [c for c in fa.columns if c.startswith("gw")] + \
           ["proj_if_fit", "form_4", "gain_vs_your_weakest", "gain_if_all_fit", "your_weakest", "proj_war", "status", "news"]
    by_pos = fa.groupby("pos").head(5).sort_values(["pos", "gain_vs_your_weakest"], ascending=[True, False])
    # A single week's projection misses by resid_sd points (1 standard deviation).
    # Over 4 weeks, comparing two players, the miss is about resid_sd * sqrt(4) * sqrt(2).
    gain_noise = resid_sd * 2 * np.sqrt(2)
    with pd.ExcelWriter("waiver_tool.xlsx") as xw:
        fa[cols].head(60).to_excel(xw, sheet_name="Best free agents for you")
        by_pos[cols].to_excel(xw, sheet_name="Top 5 by position")
        mine[["web_name", "pos", "team", "proj_next_4", "proj_if_fit", "form_4", "status", "news"]].to_excel(xw, sheet_name="Your squad")
        scores.round(3).to_excel(xw, sheet_name="Projection backtest")
        coefs = {p: m.params for p, m in model["models"].items()} if model["by_position"] else {"all": model["model"].params}
        pd.DataFrame(coefs).round(3).to_excel(xw, sheet_name="Projection model")

    pd.set_option("display.width", 220)
    print(f"Known through GW{last_done}; projecting GW{game['next_event']}-{game['next_event'] + HORIZON - 1}.\n")
    print(scores.round(3), f"\nBest: {best}. Top-30 projected players averaged {top_actual:.2f} actual pts/week")
    print(f"Weekly miss (1 SD): {resid_sd:.2f} pts. Noise in a 4-week gain between two players: about {gain_noise:.1f} pts\n")
    print("TOP 5 BY POSITION\n", by_pos[["web_name", "pos", "team", "proj_next_4", "proj_if_fit", "gain_vs_your_weakest",
                                        "gain_if_all_fit", "your_weakest", "status"]].to_string(), "\n")
    print("YOUR SQUAD\n", mine[["web_name", "pos", "team", "proj_next_4", "proj_if_fit", "form_4", "status"]].to_string(), "\n")
    print("BEST FREE AGENTS FOR YOU\n", fa[["web_name", "pos", "team", "proj_next_4", "form_4", "gain_vs_your_weakest",
                                          "your_weakest", "proj_war", "status", "news"]].head(20).to_string())


if __name__ == "__main__":
    main()
