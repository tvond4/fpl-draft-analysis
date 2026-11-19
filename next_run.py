"""
Is a refresh due? Used by the scheduled task, which checks every few hours.

How to run (from this folder):
    .venv/bin/python next_run.py              # is a refresh due now? (just looks)
    .venv/bin/python next_run.py --claim      # if due, claim it so no other run starts too
    .venv/bin/python next_run.py --mark-done  # record that the refresh finished, release the claim

Prints one line of JSON, for example:
    {"due": true, "mode": "post_gameweek", "gw": 10, ...}

There are two kinds of refresh ("mode"):

post_gameweek   After a gameweek: FPL has finished a gameweek that hasn't been
                refreshed yet AND at least 16 hours have passed since its last
                match ended (FPL takes hours to settle bonus points). A match is
                taken to end MATCH_LENGTH_HOURS after kickoff. Kickoff times come
                from the main FPL game's fixture list, downloaded fresh each time,
                so a match that moves (TV picks, cup ties, postponements) moves
                the refresh with it.
pre_waiver      Before a waiver deadline: within PRE_WAIVER_HOURS of the next
                gameweek's waiver deadline (and at least STOP_BEFORE_WAIVER_HOURS
                before it), once per gameweek. Points don't change, but injury
                news, waiver picks and the stash list do, especially over an
                international break.
If both are due, post_gameweek wins (it updates everything anyway).

State is stored in last_refreshed.json. A claim (also stored there) stops a
second scheduled check from starting a refresh while the first is still
running. A claim older than CLAIM_HOURS is ignored, so a run that crashed
can't block refreshes forever.
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from draft_grades import get
from projections import get_classic

MATCH_LENGTH_HOURS = 2
WAIT_AFTER_LAST_MATCH_HOURS = 16
PRE_WAIVER_HOURS = 24
STOP_BEFORE_WAIVER_HOURS = 1
STATE = Path("last_refreshed.json")
CLAIM_HOURS = 4


def iso(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def local(dt):
    return dt.astimezone().isoformat(timespec="minutes")


def main():
    now = datetime.now(timezone.utc)
    game = get("game")
    last_done = game["current_event"] if game["current_event_finished"] else game["current_event"] - 1
    next_gw = game["next_event"]
    state = json.loads(STATE.read_text()) if STATE.exists() else {"last_refreshed_gw": 0}
    state.setdefault("pre_waiver_gw", 0)

    if "--mark-done" in sys.argv:
        mode = state.pop("claimed_mode", "post_gameweek")
        if mode == "pre_waiver":
            state["pre_waiver_gw"] = state.pop("claimed_gw", next_gw)
        else:
            state["last_refreshed_gw"] = last_done
            state.pop("claimed_gw", None)
        state.pop("claimed_at", None)
        state["at"] = now.isoformat(timespec="minutes")
        state["last_mode"] = mode
        STATE.write_text(json.dumps(state))
        print(json.dumps({"marked_done": mode, "state": state}))
        return

    fixtures = pd.DataFrame(get_classic("fixtures/")).dropna(subset=["event"])
    fixtures["kick"] = pd.to_datetime(fixtures.kickoff_time, utc=True)
    out = {"gw": last_done, "last_refreshed_gw": state["last_refreshed_gw"],
           "season_over": bool(fixtures.finished.astype(bool).all() and state["last_refreshed_gw"] >= last_done),
           "due": False, "mode": None}

    # 1. After a gameweek
    if last_done > state["last_refreshed_gw"]:
        last_kick = fixtures[fixtures.event == last_done].kick.max()
        ready_at = last_kick.to_pydatetime() + timedelta(hours=MATCH_LENGTH_HOURS + WAIT_AFTER_LAST_MATCH_HOURS)
        out["post_gameweek_ready_local"] = local(ready_at)
        if now >= ready_at:
            out.update({"due": True, "mode": "post_gameweek",
                        "why": f"gameweek {last_done} finished over {WAIT_AFTER_LAST_MATCH_HOURS} hours ago"})

    # 2. Before the next waiver deadline
    events = {e["id"]: e for e in get("bootstrap-static", refresh=False)["events"]["data"]}
    nxt = events.get(next_gw) if next_gw else None
    if nxt and nxt.get("waivers_time"):
        waivers = iso(nxt["waivers_time"])
        opens, closes = waivers - timedelta(hours=PRE_WAIVER_HOURS), waivers - timedelta(hours=STOP_BEFORE_WAIVER_HOURS)
        out.update({"next_gw": next_gw, "waiver_deadline_local": local(waivers), "pre_waiver_opens_local": local(opens)})
        if not out["due"] and state["pre_waiver_gw"] < next_gw and opens <= now <= closes:
            out.update({"due": True, "mode": "pre_waiver", "gw": next_gw,
                        "why": f"waiver deadline for gameweek {next_gw} is within {PRE_WAIVER_HOURS} hours"})

    if not out["due"]:
        out["why"] = "nothing due: no newly finished gameweek and not yet in the pre-waiver window"

    claim = state.get("claimed_at")
    claimed = claim and now - datetime.fromisoformat(claim) < timedelta(hours=CLAIM_HOURS)
    if out["due"] and claimed:
        out.update({"due": False, "why": f"another refresh claimed at {claim} may still be running"})
    elif out["due"] and "--claim" in sys.argv:
        state.update({"claimed_at": now.isoformat(timespec="minutes"), "claimed_mode": out["mode"], "claimed_gw": out["gw"]})
        STATE.write_text(json.dumps(state))
        out["claimed"] = True
    print(json.dumps(out))


if __name__ == "__main__":
    main()
