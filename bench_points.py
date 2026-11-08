"""
FPL Draft: how many points did each manager leave on the bench?

How to run (from this folder):
    .venv/bin/python bench_points.py

For every manager and every finished gameweek:
  actual   = points the manager really scored (starting 11, after auto-subs).
             Checked against the official head-to-head results.
  best     = the most points possible from that week's 15-man squad, using a
             legal formation: 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD (11 players).
  lost     = best - actual. Points left on the bench by picking the wrong 11.
  efficiency = actual / best, as a percentage.

It also asks: in how many head-to-head matches would the best lineup have
turned a loss or draw into a win? Those are "matches lost to lineup choices".
The opponent's score is kept as it really was.

Fairness note: efficiency rewards a weak bench. If your bench players are
injured or never play, there is nothing to get wrong. So we also report
lost_per_live_bench = points lost / number of benched players who actually
played. Lower is better, and it compares managers on the calls that mattered.

call_accuracy_pct (the main ranking): a "call" is one benched player who
actually played, paired with one starter he could legally have replaced. The
call was right if the starter scored at least as many points. Benched players
who never played create no calls, so a dead bench can't inflate the score.
The low/high columns show a 90% range from re-sampling the weeks, to show how
much of the gap between managers could be luck.

Results go to bench_points.xlsx.
"""

import itertools

import numpy as np
import pandas as pd

from draft_grades import LEAGUE_ID, POSITIONS, get

FORMATION_LIMITS = {"GKP": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}


def best_lineup(squad):
    """squad: list of (pos, points). Returns the highest legal 11-player total."""
    by_pos = {p: sorted([pts for pos, pts in squad if pos == p], reverse=True) for p in POSITIONS.values()}
    best = None
    for d, m, f in itertools.product(range(3, 6), range(2, 6), range(1, 4)):
        if d + m + f != 10:
            continue
        need = {"GKP": 1, "DEF": d, "MID": m, "FWD": f}
        if any(len(by_pos[p]) < n for p, n in need.items()):
            continue  # squad doesn't have enough players for this formation
        total = sum(sum(by_pos[p][:n]) for p, n in need.items())
        best = total if best is None else max(best, total)
    return best


def lineup_calls(starters, bench, pos_of, pts):
    """Count the lineup calls that mattered this week.

    A "call" is one benched player who actually played, paired with one starter
    he could legally have replaced (the swap keeps a legal formation). The call
    was right if the starter scored at least as many points.
    Returns (right_calls, total_calls).
    """
    counts = {p: 0 for p in FORMATION_LIMITS}
    for s in starters:
        counts[pos_of[s]] += 1
    right = total = 0
    for b in bench:
        for s in starters:
            after = dict(counts)
            after[pos_of[s]] -= 1
            after[pos_of[b]] += 1
            if all(lo <= after[p] <= hi for p, (lo, hi) in FORMATION_LIMITS.items()):
                total += 1
                right += pts.get(s, 0) >= pts.get(b, 0)
    return right, total


def main():
    game = get("game")
    last_done = game["current_event"] if game["current_event_finished"] else game["current_event"] - 1
    details = get(f"league/{LEAGUE_ID}/details")
    boot = get("bootstrap-static", refresh=False)
    pos_of = {e["id"]: POSITIONS[e["element_type"]] for e in boot["elements"]}

    entries = details["league_entries"]
    name_of = {e["entry_id"]: f'{e["player_first_name"]} ({e["entry_name"]})' for e in entries}
    entry_of_league_id = {e["id"]: e["entry_id"] for e in entries}

    rows = []
    for gw in range(1, last_done + 1):
        live = get(f"event/{gw}/live", refresh=False)
        pts = {int(k): v["stats"]["total_points"] for k, v in live["elements"].items()}
        mins = {int(k): v["stats"]["minutes"] for k, v in live["elements"].items()}
        for e in entries:
            team = get(f"entry/{e['entry_id']}/event/{gw}", refresh=False)
            picks = team["picks"]
            starters = {p["element"] for p in picks if p["position"] <= 11}
            for s in team["subs"]:  # apply the automatic substitutions
                starters.discard(s["element_out"])
                starters.add(s["element_in"])
            actual = sum(pts.get(pid, 0) for pid in starters)
            live_bench = [p["element"] for p in picks
                          if p["element"] not in starters and mins.get(p["element"], 0) > 0]
            right, calls = lineup_calls(starters, live_bench, pos_of, pts)
            squad = [(pos_of[p["element"]], pts.get(p["element"], 0)) for p in picks]
            rows.append({
                "gw": gw,
                "entry_id": e["entry_id"],
                "manager": name_of[e["entry_id"]],
                "actual": actual,
                "best": best_lineup(squad),
                "bench_total": sum(pts.get(p["element"], 0) for p in picks) - actual,
                "auto_subs": len(team["subs"]),
                # A lineup decision only matters if the benched player actually played
                "bench_who_played": len(live_bench),
                "right_calls": right,
                "calls": calls,
            })
    df = pd.DataFrame(rows)
    df["lost"] = df["best"] - df["actual"]

    # Head-to-head: official scores, and would the best lineup have won?
    official, h2h = {}, []
    for mt in details["matches"]:
        if mt["event"] > last_done:
            continue
        a, b = entry_of_league_id[mt["league_entry_1"]], entry_of_league_id[mt["league_entry_2"]]
        sa, sb = mt["league_entry_1_points"], mt["league_entry_2_points"]
        official[(mt["event"], a)], official[(mt["event"], b)] = sa, sb
        h2h += [(mt["event"], a, sa, sb), (mt["event"], b, sb, sa)]
    df["official"] = [official.get((g, e)) for g, e in zip(df["gw"], df["entry_id"])]
    best_of = df.set_index(["gw", "entry_id"])["best"]
    hh = pd.DataFrame(h2h, columns=["gw", "entry_id", "score", "opp_score"])
    hh["result"] = [("W" if s > o else "D" if s == o else "L") for s, o in zip(hh["score"], hh["opp_score"])]
    hh["best"] = [best_of[(g, e)] for g, e in zip(hh["gw"], hh["entry_id"])]
    hh["flipped"] = (hh["result"] != "W") & (hh["best"] > hh["opp_score"])

    mismatches = df[df["official"] != df["actual"]]

    g = df.groupby("manager")
    summary = pd.DataFrame({
        "actual_points": g["actual"].sum(),
        "best_possible": g["best"].sum(),
        "points_lost": g["lost"].sum(),
        "lost_per_week": g["lost"].mean(),
        "efficiency_pct": g["actual"].sum() / g["best"].sum() * 100,
        "perfect_weeks": g["lost"].apply(lambda x: (x == 0).sum()),
        "worst_week_lost": g["lost"].max(),
        "bench_points_scored": g["bench_total"].sum(),
        "bench_who_played": g["bench_who_played"].sum(),
        "lost_per_live_bench": g["lost"].sum() / g["bench_who_played"].sum(),
        "calls": g["calls"].sum(),
        "right_calls": g["right_calls"].sum(),
        "call_accuracy_pct": g["right_calls"].sum() / g["calls"].sum() * 100,
        "auto_subs": g["auto_subs"].sum(),
    })
    # How sure are we? Re-draw the weeks at random 5,000 times and see how much
    # each manager's call accuracy moves. The 90% range is shown.
    rng = np.random.default_rng(0)
    lo, hi = {}, {}
    for m, x in df.groupby("manager"):
        right, calls = x["right_calls"].to_numpy(), x["calls"].to_numpy()
        idx = rng.integers(0, len(x), size=(5000, len(x)))
        sims = right[idx].sum(1) / np.maximum(calls[idx].sum(1), 1) * 100
        lo[m], hi[m] = np.percentile(sims, [5, 95])
    summary["accuracy_low_90"] = pd.Series(lo)
    summary["accuracy_high_90"] = pd.Series(hi)
    hh["manager"] = hh["entry_id"].map(name_of)
    hg = hh.groupby("manager")
    summary["record_W_D_L"] = hg["result"].apply(lambda r: f'{(r=="W").sum()}-{(r=="D").sum()}-{(r=="L").sum()}')
    summary["matches_lost_to_lineup"] = hg["flipped"].sum()
    summary = summary.sort_values("call_accuracy_pct", ascending=False).round(1)
    summary.insert(0, "rank", range(1, len(summary) + 1))

    out = "bench_points.xlsx"
    with pd.ExcelWriter(out) as xw:
        summary.to_excel(xw, sheet_name="Manager summary")
        df.drop(columns="entry_id").to_excel(xw, sheet_name="Every week", index=False)
        hh.drop(columns="entry_id").to_excel(xw, sheet_name="Head to head", index=False)

    print(f"Through GW{last_done}. Score check: {len(df) - len(mismatches)}/{len(df)} manager-weeks match official results.")
    if len(mismatches):
        print(mismatches[["gw", "manager", "actual", "official"]].to_string())
    print(summary.to_string())


if __name__ == "__main__":
    main()
