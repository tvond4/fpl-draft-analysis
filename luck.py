"""
FPL Draft: points against and luck, modelled on KenPom's "Luck" rating in
college basketball.

How to run (from this folder):
    .venv/bin/python luck.py

KenPom's idea
-------------
KenPom's Luck is a team's actual winning percentage minus the winning
percentage you would expect from how it played in each game. Every game is
turned into a win probability based on its own scoring, so a 1-point win counts
as about half a "deserved" win and a blowout as nearly a full one. Teams that
keep winning close games come out lucky; teams that keep losing them come out
unlucky. Luck rarely carries over, so it is a good sign of who is likely to
regress.

The FPL version
---------------
For each head-to-head match:
  expected win = P(a team scoring this many points beats one scoring that many)
              = Phi(margin / (SIGMA * sqrt(2)))
where margin is points for minus points against, Phi is the bell-curve
probability, and SIGMA is the league's week-to-week spread in a team's score
(how much one team's score jumps around from week to week). A tie in points
counts as half a win, as FPL draws do.

  luck      = actual win % - expected win %   (KenPom's scale, e.g. +0.050)
  luck_wins = actual wins - expected wins      (same thing, counted in wins)

Two more views of luck that only make sense in FPL, where you can't defend:
  points_against_per_week   what opponents scored against you. You have no
                            control over it, so facing hot opponents is luck.
  pa_vs_league              points against per week minus the league average
                            score. Positive means you faced above-average
                            opponents.
  all_play                  your score against EVERY other team each week. It
                            removes the luck of the draw: the record your
                            scores deserved against an average schedule.

Results go to luck.xlsx.
"""

import numpy as np
import pandas as pd
from math import erf, sqrt

from draft_grades import LEAGUE_ID, get, manager_labels


def phi(z):
    return 0.5 * (1 + erf(z / sqrt(2)))


def main():
    game = get("game")
    last_done = game["current_event"] if game["current_event_finished"] else game["current_event"] - 1
    details = get(f"league/{LEAGUE_ID}/details", refresh=False)
    labels = manager_labels(details)
    name_of = {e["id"]: labels[e["entry_id"]] for e in details["league_entries"]}

    rows = []
    for m in details["matches"]:
        if m["event"] > last_done:
            continue
        a, b = name_of[m["league_entry_1"]], name_of[m["league_entry_2"]]
        sa, sb = m["league_entry_1_points"], m["league_entry_2_points"]
        rows += [{"gw": m["event"], "manager": a, "opponent": b, "pf": sa, "pa": sb},
                 {"gw": m["event"], "manager": b, "opponent": a, "pf": sb, "pa": sa}]
    g = pd.DataFrame(rows)
    g["result"] = np.select([g.pf > g.pa, g.pf == g.pa], ["W", "D"], "L")
    g["win_value"] = g.result.map({"W": 1.0, "D": 0.5, "L": 0.0})

    # Week-to-week spread of one team's score, after removing each team's own average
    sigma = (g.pf - g.groupby("manager").pf.transform("mean")).std(ddof=1)
    g["expected_win"] = [phi((pf - pa) / (sigma * sqrt(2))) for pf, pa in zip(g.pf, g.pa)]
    league_avg = g.pf.mean()

    # All-play: each score against every other score that week (ties count half)
    allplay = []
    for gw, wk in g.groupby("gw"):
        for _, r in wk.iterrows():
            others = wk[wk.manager != r.manager].pf
            allplay.append({"manager": r.manager, "ap_w": (r.pf > others).sum(), "ap_d": (r.pf == others).sum(),
                            "ap_l": (r.pf < others).sum()})
    ap = pd.DataFrame(allplay).groupby("manager").sum()

    s = g.groupby("manager").agg(
        games=("gw", "size"), wins=("result", lambda r: (r == "W").sum()), draws=("result", lambda r: (r == "D").sum()),
        losses=("result", lambda r: (r == "L").sum()), points_for=("pf", "sum"), points_against=("pa", "sum"),
        actual_wins=("win_value", "sum"), expected_wins=("expected_win", "sum"))
    s["record"] = s.wins.astype(str) + "-" + s.draws.astype(str) + "-" + s.losses.astype(str)
    s["pf_per_week"] = s.points_for / s.games
    s["points_against_per_week"] = s.points_against / s.games
    s["pa_vs_league"] = s.points_against_per_week - league_avg
    s["actual_win_pct"] = s.actual_wins / s.games
    s["expected_win_pct"] = s.expected_wins / s.games
    s["luck"] = s.actual_win_pct - s.expected_win_pct
    s["luck_wins"] = s.actual_wins - s.expected_wins
    s = s.join(ap)
    s["all_play_record"] = s.ap_w.astype(str) + "-" + s.ap_d.astype(str) + "-" + s.ap_l.astype(str)
    s["all_play_win_pct"] = (s.ap_w + 0.5 * s.ap_d) / (s.ap_w + s.ap_d + s.ap_l)
    # Wins you'd have against an average schedule, on the same number of games
    s["all_play_wins"] = s.all_play_win_pct * s.games
    # Close games: decided by 5 points or fewer
    close = g[(g.pf - g.pa).abs() <= 5]
    s["close_record"] = close.groupby("manager").result.apply(
        lambda r: f'{(r == "W").sum()}-{(r == "D").sum()}-{(r == "L").sum()}').reindex(s.index).fillna("0-0-0")
    s = s.sort_values("luck", ascending=False)
    s.insert(0, "luck_rank", range(1, len(s) + 1))

    # Check against the official table
    official = {name_of[x["league_entry"]]: (x["matches_won"], x["matches_drawn"], x["matches_lost"], x["points_against"])
                for x in details["standings"]}
    mism = [n for n, (w, d, l, pa) in official.items()
            if (s.loc[n, "wins"], s.loc[n, "draws"], s.loc[n, "losses"]) != (w, d, l) or s.loc[n, "points_against"] != pa]

    with pd.ExcelWriter("luck.xlsx") as xw:
        s.round(3).to_excel(xw, sheet_name="Luck by manager")
        g.round(3).to_excel(xw, sheet_name="Every match", index=False)

    pd.set_option("display.width", 250)
    print(f"Through GW{last_done}. League average {league_avg:.1f} pts/week; week-to-week spread (SIGMA) {sigma:.1f}.")
    print(f"Official table check: {'all match' if not mism else 'MISMATCH ' + str(mism)}\n")
    print(s[["luck_rank", "record", "expected_wins", "luck", "luck_wins", "points_against_per_week", "pa_vs_league",
             "all_play_record", "all_play_win_pct", "close_record"]].round(3).to_string())


if __name__ == "__main__":
    main()
