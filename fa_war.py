"""
FPL Draft: who is best at picking up free agents? (WAR = points above replacement)

How to run (from this folder, after bench_points.py has cached the weekly squads):
    .venv/bin/python fa_war.py

Definitions
-----------
Free agent in week g     A player in no manager's 15-man squad that week.
Replacement level        For each position and week: the average points of the
                         5 free agents with the best recent form (points over the
                         previous 4 weeks, known before the deadline). Week 1 uses
                         FPL's pre-season rank instead of form. Think of it as
                         "what a sensible, low-effort pickup would have scored".
Pickup WAR               Player's points minus replacement level, summed over every
                         finished week the manager owned him after the move.
Swap gain                Player's points minus the dropped player's points over
                         the same weeks. "Did the move beat standing still?"
Started WAR              Like pickup WAR but only weeks the player was in the
                         starting 11. Mixes picking skill with lineup choices.

Moves that take effect in an unfinished week are listed but not graded yet.
Results go to fa_war.xlsx.
"""

import numpy as np
import pandas as pd

from draft_grades import LEAGUE_ID, POSITIONS, get, manager_labels

TOP_N = 5          # free agents that set replacement level
FORM_WEEKS = 4     # weeks of form used to pick them


def load():
    game = get("game")
    last_done = game["current_event"] if game["current_event_finished"] else game["current_event"] - 1
    details = get(f"league/{LEAGUE_ID}/details", refresh=False)
    boot = get("bootstrap-static", refresh=False)
    tx = get(f"draft/league/{LEAGUE_ID}/transactions")["transactions"]
    players = pd.DataFrame(boot["elements"]).set_index("id")
    players["pos"] = players["element_type"].map(POSITIONS)
    entries = manager_labels(details)       # {entry_id: unique label}

    pts = {}      # (player, gw) -> points
    mins = {}
    for gw in range(1, last_done + 1):
        for pid, row in get(f"event/{gw}/live", refresh=False)["elements"].items():
            pts[(int(pid), gw)] = row["stats"]["total_points"]
            mins[(int(pid), gw)] = row["stats"]["minutes"]

    owner, started = {}, set()   # (player, gw) -> entry ; (player, gw, entry) started
    for gw in range(1, last_done + 1):
        for eid in entries:
            team = get(f"entry/{eid}/event/{gw}", refresh=False)
            xi = {p["element"] for p in team["picks"] if p["position"] <= 11}
            for s in team["subs"]:
                xi.discard(s["element_out"]); xi.add(s["element_in"])
            for p in team["picks"]:
                owner[(p["element"], gw)] = eid
            for pid in xi:
                started.add((pid, gw, eid))
    return last_done, players, entries, tx, pts, mins, owner, started


def replacement_levels(last_done, players, pts, owner):
    rows = []
    for gw in range(1, last_done + 1):
        free = [pid for pid in players.index if (pid, gw) not in owner]
        fa = players.loc[free, ["pos", "draft_rank"]].copy()
        if gw == 1:
            fa["form"] = -fa["draft_rank"]          # better rank = better "form"
        else:
            fa["form"] = [sum(pts.get((pid, g), 0) for g in range(max(1, gw - FORM_WEEKS), gw)) for pid in fa.index]
        for pos, grp in fa.groupby("pos"):
            top = grp.sort_values("form", ascending=False).head(TOP_N)
            rows.append({"gw": gw, "pos": pos, "replacement": np.mean([pts.get((pid, gw), 0) for pid in top.index]),
                         "replacement_players": ", ".join(players.loc[top.index, "web_name"])})
    return pd.DataFrame(rows)


def grade(last_done, players, entries, tx, pts, owner, started, repl):
    R = repl.set_index(["gw", "pos"])["replacement"]
    moves = []
    for t in sorted(tx, key=lambda t: (t["event"], t["index"] or 0, t["id"])):
        if t["result"] != "a":
            continue
        pid, out, eid, start = t["element_in"], t["element_out"], t["entry"], t["event"]
        pos = players.at[pid, "pos"]
        weeks = []
        for g in range(start, last_done + 1):          # owned continuously from the move
            if owner.get((pid, g)) != eid:
                break
            weeks.append(g)
        war = sum(pts.get((pid, g), 0) - R[(g, pos)] for g in weeks)
        swap = sum(pts.get((pid, g), 0) - pts.get((out, g), 0) for g in weeks)
        started_war = sum(pts.get((pid, g), 0) - R[(g, pos)] for g in weeks if (pid, g, eid) in started)
        moves.append({
            "gw": start, "manager": entries[eid], "kind": {"w": "waiver", "f": "free agent"}[t["kind"]],
            "player_in": players.at[pid, "web_name"], "pos": pos, "player_out": players.at[out, "web_name"],
            "weeks_owned": len(weeks), "points": sum(pts.get((pid, g), 0) for g in weeks),
            "war": round(war, 1), "swap_gain": swap, "started_war": round(started_war, 1),
            "graded": start <= last_done,
        })
    return pd.DataFrame(moves)


def summarise(moves, entries):
    g = moves[moves["graded"] & (moves["weeks_owned"] > 0)].groupby("manager")
    s = pd.DataFrame({
        "moves": moves[moves["graded"]].groupby("manager").size(),
        "total_war": g["war"].sum(),
        "war_per_move": g["war"].mean(),
        "hit_rate_pct": g["war"].apply(lambda x: (x > 0).mean() * 100),
        "swap_gain": g["swap_gain"].sum(),
        "started_war": g["started_war"].sum(),
        "best_move": g.apply(lambda x: f'{x.loc[x.war.idxmax(), "player_in"]} ({x.war.max():+.0f})', include_groups=False),
        "worst_move": g.apply(lambda x: f'{x.loc[x.war.idxmin(), "player_in"]} ({x.war.min():+.0f})', include_groups=False),
    }).reindex(list(entries.values()))
    s["moves"] = s["moves"].fillna(0).astype(int)
    # How sure are we? Re-sample each manager's moves 5,000 times; 90% range of total WAR.
    rng = np.random.default_rng(0)
    for m, x in g:
        v = x["war"].to_numpy()
        sims = v[rng.integers(0, len(v), size=(5000, len(v)))].sum(1)
        s.loc[m, "war_low_90"], s.loc[m, "war_high_90"] = np.percentile(sims, [5, 95])
    return s.sort_values("total_war", ascending=False).round(1)


def main():
    last_done, players, entries, tx, pts, mins, owner, started = load()
    repl = replacement_levels(last_done, players, pts, owner)
    moves = grade(last_done, players, entries, tx, pts, owner, started, repl)
    summary = summarise(moves, entries)
    with pd.ExcelWriter("fa_war.xlsx") as xw:
        summary.to_excel(xw, sheet_name="Manager summary")
        moves.to_excel(xw, sheet_name="Every move", index=False)
        repl.round(2).to_excel(xw, sheet_name="Replacement level", index=False)
    pd.set_option("display.width", 220)
    print(f"Through GW{last_done}. {int(moves.graded.sum())} graded moves, {int((~moves.graded).sum())} pending.\n")
    print(repl.pivot(index="gw", columns="pos", values="replacement").round(2), "\n")
    print(summary.to_string())


if __name__ == "__main__":
    main()
