"""
FPL Draft: stash candidates. Free agents who are injured or have barely played,
but would score well if they got a regular role.

How to run (from this folder, after projections.py):
    .venv/bin/python potential.py               # shortlist with news from stash_news.json
    .venv/bin/python potential.py --shortlist   # fresh shortlist, ignoring old news

Each week: run with --shortlist, research the players it lists in
stash_candidates.json, save that news (with each player's "id") to
stash_news.json, then run again without the flag.

Who qualifies
-------------
A free agent (in nobody's squad) who is either
  - flagged injured, doubtful or suspended by FPL, or
  - has played under 40% of the available minutes this season, but at least
    90 minutes in total (a single cameo is too little evidence).
Players FPL lists as unavailable ("u", usually left the league or on long loan)
are left out.

How potential is measured
-------------------------
upside_per_week   Projected points per week IF he starts regularly (90% of
                  minutes), from the waiver tool's projection model using
                  blended_p90 (below) as his scoring rate.
xgi_per_90        Expected goals + assists per 90 minutes, also pulled toward
                  the position average. Shows attacking threat behind the points.
preseason_rank    FPL's pre-season draft rank: what was expected of him in August.
expected_p90      Points per 90 a player of his pre-season rank and position
                  usually scores, fitted on this season's regular starters.
blended_p90       His own points per 90 blended with expected_p90, which counts
                  as 4 matches of evidence. A few lucky minutes can't make a
                  star, and an injured star keeps his reputation.
potential_score   Upside per week (uses blended_p90).
stash_value_8wk   After web research: upside per week x weeks fit in the next 8
                  (unknown return = 4 weeks out) x likely role (starter 100%,
                  rotation 60%, bench 30%, unclear 50%).
gain_vs_weakest   Stash value minus what your weakest player at that position
                  is projected to score over the same 8 weeks (fit).

The top candidates are then checked against news by web research (see
stash_news.json), which adds return dates and role outlook.
Results go to stash_watchlist.xlsx.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import projections as pj
from draft_grades import LEAGUE_ID, get

MAX_MINUTES_SHARE = 0.40
STARTER_MINUTES = 0.9
PRIOR_MATCHES = 4       # the pre-season expectation counts as 4 matches of evidence
TOP_N_PER_GROUP = 8
MIN_MINUTES = 90        # "barely played" players need at least one full match's worth
STASH_WEEKS = 8
UNKNOWN_WEEKS_OUT = 4   # if the return date is unknown, assume about a month
ROLE_SHARE = {"starter": 1.0, "rotation": 0.6, "bench": 0.3, "unclear": 0.5}   # share of the starter upside


def main():
    game, last_done, players, fixtures, pts, mins = pj.load()
    tm = pj.team_matches(fixtures)
    variant = pj.chosen_variant(players, pts, mins, tm, last_done)
    feats, by_pos = pj.VARIANTS[variant]
    model = pj.fit(pj.training_rows(players, pts, mins, tm, range(5, last_done + 1)), feats, by_pos)

    base = pj.features(players, pts, mins, tm, game["next_event"], last_done).set_index("pid")

    # Expected goal involvement per 90, pulled toward the position average (3 matches' worth)
    xgi = {pid: sum(pj.EXTRA.get((pid, g), {}).get("expected_goal_involvements", 0) for g in range(1, last_done + 1))
           for pid in players.index}
    base["season_xgi"] = pd.Series(xgi)
    regulars = base[base.season_mins >= 270]
    pos_xgi90 = regulars.groupby("pos").apply(lambda g: g.season_xgi.sum() / g.season_mins.sum() * 90, include_groups=False)
    base["xgi_per_90"] = (base.season_xgi + 3 * base.pos.map(pos_xgi90)) / (base.season_mins / 90 + 3)

    # Talent prior: among regular starters, how do points per 90 relate to pre-season rank?
    # Fit p90 = a + b * log(rank) per position, then blend each player's own p90 with it,
    # counting the prior as PRIOR_MATCHES matches of evidence.
    reg = base[base.season_mins >= 450].join(players[["draft_rank"]])
    reg["p90"] = reg.season_pts / reg.season_mins * 90
    prior_fit = {}
    for pos, g in reg.groupby("pos"):
        b, a = np.polyfit(np.log(g.draft_rank), g.p90, 1)
        prior_fit[pos] = (a, b)
    ranks = players.draft_rank.reindex(base.index)
    base["prior_p90"] = [prior_fit[p][0] + prior_fit[p][1] * np.log(r) for p, r in zip(base.pos, ranks)]
    # Any minutes count as at least one full match, so a 3-minute cameo can't produce a huge rate
    eff_matches = np.where(base.season_mins > 0, base.season_mins.clip(lower=90), 0) / 90
    base["blended_p90"] = (base.season_pts + PRIOR_MATCHES * base.prior_p90) / (eff_matches + PRIOR_MATCHES)
    # Upside over the next 4 weeks if he starts: the projection model with a
    # starter's minutes and the blended rate as his scoring rate
    upside = pd.Series(0.0, index=players.index)
    for g in range(game["next_event"], game["next_event"] + pj.HORIZON):
        f = pj.features(players, pts, mins, tm, g, last_done)
        f["minutes_4"] = STARTER_MINUTES
        f["rate_x_min"] = f.pid.map(base.blended_p90).values * STARTER_MINUTES
        upside += pd.Series(pj.predict(model, f).values, index=f.pid)

    status = pd.DataFrame(get(f"league/{LEAGUE_ID}/element-status")["element_status"]).set_index("element")
    teams = {t["id"]: t["short_name"] for t in get("bootstrap-static", refresh=False)["teams"]}
    t = pd.DataFrame({
        "id": players.index, "player": players.web_name, "full_name": players.first_name + " " + players.second_name,
        "pos": players.pos, "team": players.team.map(teams), "status": players.status,
        "chance_next": players.chance_of_playing_next_round, "fpl_news": players.news,
        "minutes": base.season_mins, "minutes_share_pct": (base.season_mins / (90 * last_done) * 100).round(0),
        "season_points": base.season_pts, "points_per_90": (base.season_pts / base.season_mins.replace(0, np.nan) * 90).round(1),
        "xgi_per_90": base.xgi_per_90.round(2), "preseason_rank": players.draft_rank,
        "expected_p90": base.prior_p90.round(1), "blended_p90": base.blended_p90.round(1),
        "upside_per_week": (upside / pj.HORIZON).round(2), "owner": status.owner,
    })
    flagged = t.status.isin(["i", "d", "s"])
    # A cameo is too little evidence: "few minutes" players need MIN_MINUTES this season
    low_minutes = (t.minutes_share_pct < MAX_MINUTES_SHARE * 100) & (t.minutes >= MIN_MINUTES)
    cand = t[t.owner.isna() & t.status.ne("u") & (flagged | low_minutes)].copy()
    cand["group"] = np.where(flagged.reindex(cand.index), "Injured or doubtful", "Barely played")
    cand["potential_score"] = cand.upside_per_week
    cand = cand.sort_values("potential_score", ascending=False)
    top = cand.groupby("group").head(TOP_N_PER_GROUP)

    news_file = Path("stash_news.json")
    if news_file.exists() and "--shortlist" not in sys.argv:
        # News is matched on FPL's player id: display names repeat (two Hendersons, two Kamaras)
        news = pd.DataFrame(json.loads(news_file.read_text()))
        if "id" not in news.columns:
            raise SystemExit("stash_news.json entries need an 'id' (FPL player id) so news matches the right player.")
        news = news.set_index("id")
        cand = cand.join(news[[c for c in news.columns if c not in cand.columns]], on="id")
        # Stash value: expected points over the next STASH_WEEKS weeks, counting only
        # weeks he should be fit, scaled by how much he is likely to play.
        weeks_fit = (STASH_WEEKS - cand.weeks_out.fillna(UNKNOWN_WEEKS_OUT)).clip(0, STASH_WEEKS)
        role = cand.role_outlook.map(ROLE_SHARE).fillna(ROLE_SHARE["unclear"])
        cand["stash_value_8wk"] = (cand.upside_per_week * weeks_fit * role).round(1)
        cand.loc[cand.news_summary.isna(), "stash_value_8wk"] = np.nan   # not researched
        # Compare with your weakest player at the position over the same 8 weeks (from the waiver tool)
        mine = pd.read_excel("waiver_tool.xlsx", sheet_name="Your squad")
        weakest = mine.sort_values("proj_if_fit").groupby("pos").head(1).set_index("pos")
        cand["your_weakest"] = cand.pos.map(weakest.web_name)
        cand["gain_vs_weakest_8wk"] = (cand.stash_value_8wk - STASH_WEEKS * cand.pos.map(weakest.proj_if_fit) / pj.HORIZON).round(1)
        cand = cand.sort_values(["stash_value_8wk", "potential_score"], ascending=False, na_position="last")
        top = cand[cand.news_summary.notna()]

    top.drop(columns=[c for c in ["sources"] if c in top.columns]).to_json("stash_candidates.json", orient="records", indent=1, force_ascii=False)
    with pd.ExcelWriter("stash_watchlist.xlsx") as xw:
        out = top.copy()
        if "sources" in out.columns:
            out["sources"] = out["sources"].apply(lambda v: " ".join(v) if isinstance(v, list) else v)
        out.to_excel(xw, sheet_name="Stash watchlist", index=False)
        cand.head(60).drop(columns=[c for c in ["sources"] if c in cand.columns]).to_excel(xw, sheet_name="All candidates", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print(f"{len(cand)} candidates. Shortlist of {len(top)}:\n")
    print({k: tuple(round(x, 2) for x in v) for k, v in prior_fit.items()}, "(a, b) in p90 = a + b*log(rank)\n")
    cols = ["player", "pos", "team", "group", "minutes", "blended_p90", "upside_per_week"]
    cols += [c for c in ["weeks_out", "role_outlook", "confidence", "stash_value_8wk", "your_weakest", "gain_vs_weakest_8wk"] if c in top.columns]
    print(top[cols].to_string(index=False))


if __name__ == "__main__":
    main()
