"""
Is a weekly refresh due? Used by the scheduled task, which checks every few hours.

How to run (from this folder):
    .venv/bin/python next_run.py              # is a refresh due now?
    .venv/bin/python next_run.py --mark-done  # record that the refresh just finished

Prints one line of JSON, for example:
    {"due": true, "gw": 10, "last_refreshed_gw": 9, ...}

A refresh is due when FPL has finished a gameweek that hasn't been refreshed yet
AND at least 24 hours have passed since that gameweek's last match ended. A match
is taken to end MATCH_LENGTH_HOURS after kickoff. Kickoff times come from the main
FPL game's fixture list, downloaded fresh each time, so a match that moves (TV
picks, cup ties, postponements) moves the refresh with it.

The last refreshed gameweek is stored in last_refreshed.json.
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from draft_grades import get
from projections import get_classic

MATCH_LENGTH_HOURS = 2
WAIT_AFTER_LAST_MATCH_HOURS = 24
STATE = Path("last_refreshed.json")


def main():
    game = get("game")
    last_done = game["current_event"] if game["current_event_finished"] else game["current_event"] - 1
    state = json.loads(STATE.read_text()) if STATE.exists() else {"last_refreshed_gw": 0}

    if "--mark-done" in sys.argv:
        state = {"last_refreshed_gw": last_done, "at": datetime.now(timezone.utc).isoformat(timespec="minutes")}
        STATE.write_text(json.dumps(state))
        print(json.dumps({"marked_done": last_done}))
        return

    fixtures = pd.DataFrame(get_classic("fixtures/")).dropna(subset=["event"])
    fixtures["kick"] = pd.to_datetime(fixtures.kickoff_time, utc=True)
    now = datetime.now(timezone.utc)
    out = {"gw": last_done, "last_refreshed_gw": state["last_refreshed_gw"],
           "season_over": bool(fixtures.finished.astype(bool).all() and state["last_refreshed_gw"] >= last_done)}
    if last_done <= state["last_refreshed_gw"]:
        out.update({"due": False, "why": "no newly finished gameweek"})
    else:
        last_kick = fixtures[fixtures.event == last_done].kick.max()
        ready_at = last_kick.to_pydatetime() + timedelta(hours=MATCH_LENGTH_HOURS + WAIT_AFTER_LAST_MATCH_HOURS)
        out.update({"due": now >= ready_at, "ready_at_local": ready_at.astimezone().isoformat(timespec="minutes"),
                    "why": "gameweek finished over 24 hours ago" if now >= ready_at else "waiting for the 24 hours to pass"})
    print(json.dumps(out))


if __name__ == "__main__":
    main()
