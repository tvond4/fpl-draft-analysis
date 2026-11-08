"""
FPL Draft: grade every manager's draft.

How to run (from this folder):
    .venv/bin/python draft_grades.py

What it does:
1. Downloads the league's draft picks and every player's weekly points from the
   public FPL Draft data feed (no login needed). Raw files go in ./data.
2. Scores each pick several ways (explained below).
3. Writes draft_grades.xlsx with a manager summary and a pick-by-pick sheet.

Points are counted only through the last *finished* gameweek, so a half-played
week does not skew the numbers.

Pick metrics
------------
points            Points the drafted player has scored this season (all of them,
                  even if the manager later dropped or traded him).
vs_all_left       Points minus the average of EVERY player still undrafted at that
                  moment. Easy bar to clear, since most of that pool never plays.
vs_next_10        Points minus the average of the next 10 players drafted by
                  anyone. "Did you beat what the league took right after you?"
vs_pos_options    Points minus the average of the 10 best-ranked players still
                  available AT THE SAME POSITION (by FPL's pre-season draft rank).
                  "Among the realistic alternatives at that position, did you
                  pick a good one?"
beat_pct          Share of those 10 same-position options the pick outscored.
rank_value        Overall pick number minus FPL pre-season rank. Positive means
                  you got the player later than his ranking (good value on paper,
                  before any games were played). Summary uses rounds 1-5 only,
                  since late-round rankings are not meaningful.
pick_score        vs_pos_options minus the league's average vs_pos_options in
                  that round. 0 = an average drafter in this league. The
                  manager "draft_score" is the sum over all 15 picks.
"""

import json
import os
import re
import time
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = "https://draft.premierleague.com/api"
HERE = Path(__file__).parent
# Every script reads and writes its files (data, spreadsheets, JSON) in this
# project folder, wherever you run it from.
os.chdir(HERE)

# Your league settings live in config.json, which is kept out of git.
# Copy config.example.json to config.json and fill in your own IDs (see README).
_config_file = HERE / "config.json"
if not _config_file.exists():
    raise SystemExit("Missing config.json. Copy config.example.json to config.json and add your league and team IDs.")
CONFIG = json.loads(_config_file.read_text())
LEAGUE_ID = int(CONFIG["league_id"])
MY_ENTRY_ID = int(CONFIG["my_entry_id"])


def manager_labels(details=None):
    """{entry_id: label} for every manager: their first name, plus their team name
    when two managers share a first name, so labels are always unique."""
    details = details or get(f"league/{LEAGUE_ID}/details", refresh=False)
    entries = details["league_entries"]
    firsts = Counter(e["player_first_name"] for e in entries)
    return {e["entry_id"]: e["player_first_name"] if firsts[e["player_first_name"]] == 1
            else f'{e["player_first_name"]} ({e["entry_name"]})' for e in entries}


def my_name():
    """Your label in results (normally your first name)."""
    return manager_labels()[MY_ENTRY_ID]


DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
N_OPTIONS = 10   # same-position alternatives to compare against
N_NEXT = 10      # following picks to compare against

# How the download cache decides when a saved copy is too old:
#  - Per-gameweek data (scores, team sheets) can still be corrected for a while
#    after a gameweek ends, e.g. when bonus points are confirmed. A saved copy is
#    treated as final only if it was downloaded after the NEXT gameweek's
#    deadline; until then it is downloaded again once it is an hour old.
#  - Everything else (player news, league results, ownership) changes all the
#    time, so it is downloaded again once it is an hour old.
MAX_AGE_SECONDS = 3600
_GAMEWEEK_PATH = re.compile(r"^(?:event/(\d+)/live|entry/\d+/event/(\d+))$")


def _final_after(gw):
    """Time after which a gameweek's data is settled: the next gameweek's deadline."""
    events = get("bootstrap-static", refresh=False)["events"]["data"]
    deadlines = {e["id"]: datetime.fromisoformat(e["deadline_time"].replace("Z", "+00:00")).timestamp() for e in events}
    if gw + 1 in deadlines:
        return deadlines[gw + 1]
    return deadlines.get(gw, 0) + 7 * 24 * 3600     # last gameweek: a week after its deadline


def get(path, refresh=True):
    """Download one API endpoint and cache it in ./data.

    refresh=True always downloads. refresh=False reuses the saved copy unless it
    may be out of date (see MAX_AGE_SECONDS above).
    """
    out = DATA / (path.replace("/", "_") + ".json")
    if not refresh and out.exists():
        saved = out.stat().st_mtime
        old = time.time() - saved > MAX_AGE_SECONDS
        m = _GAMEWEEK_PATH.match(path)
        if m:
            refresh = old and saved < _final_after(int(m.group(1) or m.group(2)))
        else:
            refresh = old
    if refresh or not out.exists():
        req = urllib.request.Request(f"{BASE}/{path}", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            out.write_bytes(r.read())
        time.sleep(0.3)  # be polite to the server
    return json.loads(out.read_text())


def load():
    game = get("game")
    last_done = game["current_event"] if game["current_event_finished"] else game["current_event"] - 1
    details = get(f"league/{LEAGUE_ID}/details")
    choices = get(f"draft/{LEAGUE_ID}/choices")["choices"]
    boot = get("bootstrap-static")

    # Points per player per finished gameweek
    weekly = {}
    for gw in range(1, last_done + 1):
        live = get(f"event/{gw}/live", refresh=False if gw < last_done else True)
        for pid, row in live["elements"].items():
            weekly.setdefault(int(pid), 0)
            weekly[int(pid)] += row["stats"]["total_points"]

    draft_time = details["league"]["drafts"][0]["draft_completed"]
    players = pd.DataFrame(boot["elements"])
    players["pos"] = players["element_type"].map(POSITIONS)
    players["points"] = players["id"].map(weekly).fillna(0).astype(int)
    # Only players who existed when the draft happened could have been picked
    players["in_pool"] = players["added"] < draft_time
    players = players.set_index("id")

    managers = {e["entry_id"]: f'{e["player_first_name"]} ({e["entry_name"]})'
                for e in details["league_entries"]}
    return last_done, players, choices, managers


def grade(players, choices, managers):
    picks = sorted(choices, key=lambda c: c["index"])
    taken = set()
    rows = []
    pool = players[players["in_pool"]]

    for c in picks:
        pid = c["element"]
        p = players.loc[pid]
        left = pool[~pool.index.isin(taken)]                       # still available, incl. this pick
        others = left.drop(index=pid, errors="ignore")
        same_pos = others[others["pos"] == p["pos"]].nsmallest(N_OPTIONS, "draft_rank")

        rows.append({
            "pick": c["index"],
            "round": c["round"],
            "manager": managers[c["entry"]],
            "player": p["web_name"],
            "pos": p["pos"],
            "points": p["points"],
            "preseason_rank": int(p["draft_rank"]),
            "avg_all_left": others["points"].mean(),
            "avg_pos_options": same_pos["points"].mean(),
            "beat_pct": (same_pos["points"] < p["points"]).mean() * 100,
            "options_list": ", ".join(f'{n} ({pts})' for n, pts in zip(same_pos["web_name"], same_pos["points"])),
            "auto_pick": c["was_auto"],
        })
        taken.add(pid)

    df = pd.DataFrame(rows)
    df["avg_next_10"] = [df["points"].iloc[i + 1:i + 1 + N_NEXT].mean() for i in range(len(df))]
    df["vs_all_left"] = df["points"] - df["avg_all_left"]
    df["vs_next_10"] = df["points"] - df["avg_next_10"]
    df["vs_pos_options"] = df["points"] - df["avg_pos_options"]
    df["rank_value"] = df["pick"] - df["preseason_rank"]   # positive = got him later than ranked
    # FPL's pre-season ranking includes players who later left or got injured,
    # so raw vs_pos_options runs positive for almost everyone. Subtracting the
    # league's average for that round makes 0 = "average drafter in this league".
    df["pick_score"] = df["vs_pos_options"] - df.groupby("round")["vs_pos_options"].transform("mean")
    return df


def summarise(df):
    g = df.groupby("manager")
    early = df[df["round"] <= 5].groupby("manager")
    late = df[df["round"] > 5].groupby("manager")
    s = pd.DataFrame({
        "draft_score": g["pick_score"].sum(),
        "early_score_r1_5": early["pick_score"].sum(),
        "late_score_r6_15": late["pick_score"].sum(),
        "total_points": g["points"].sum(),
        "vs_next_10": g["vs_next_10"].sum(),
        "vs_all_left": g["vs_all_left"].sum(),
        "avg_beat_pct": g["beat_pct"].mean(),
        "rank_value_r1_5": early["rank_value"].mean(),
        "best_pick": g.apply(lambda x: x.loc[x["pick_score"].idxmax(), "player"], include_groups=False),
        "worst_pick": g.apply(lambda x: x.loc[x["pick_score"].idxmin(), "player"], include_groups=False),
        "auto_picks": g["auto_pick"].sum(),
    })
    s = s.sort_values("draft_score", ascending=False)
    s.insert(0, "grade_rank", range(1, len(s) + 1))
    return s.round(1)


def main():
    last_done, players, choices, managers = load()
    df = grade(players, choices, managers)
    summary = summarise(df)

    out = HERE / "draft_grades.xlsx"
    with pd.ExcelWriter(out) as xw:
        summary.to_excel(xw, sheet_name="Manager summary")
        df.round(1).to_excel(xw, sheet_name="Every pick", index=False)
    print(f"Points through GW{last_done}. Saved {out.name}\n")
    print(summary.drop(columns=["best_pick", "worst_pick"]).to_string())


if __name__ == "__main__":
    main()
