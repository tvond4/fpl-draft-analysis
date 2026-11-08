"""
FPL Draft: simulate the rest of the season. How likely is each finish, and how
much would better waiver moves help you recover from the draft?

How to run (from this folder, after bench_points.py and projections.py):
    .venv/bin/python simulate_season.py

How it works
------------
1. Standings so far come from the official results (checked against the site).
2. Each team gets a "true strength": its expected weekly score. Two versions:
   A. From results so far. Nine weeks is a small sample, so each team's average
      is pulled toward the league average by exactly as much as the noise in
      the data says (a statistics method called "shrinkage"). We are also unsure
      of each team's strength, so every simulated season draws it fresh.
   B. From current squads. Each team's best legal 11 by projected points (fit
      players), rescaled so the league average matches real scores.
3. Each remaining match (GW10 to GW38) is played by drawing both teams' scores
   from a normal (bell-curve) distribution with that strength and the league's
   week-to-week spread. Win = 3 points, draw = 1. Ties on points are broken by
   total points scored.
4. Repeat 20,000 times and count how often each finish happens, grouped into
   the outcomes that matter here: top 2, middle 6 (3rd to 8th) and bottom 2.
5. Your scenarios add +2, +4 or +6 points a week to your strength from now on,
   to show what better waiver moves would buy.

Results go to season_sim.xlsx.
"""

import numpy as np
import pandas as pd

from bench_points import best_lineup
from draft_grades import LEAGUE_ID, get, manager_labels, my_name
import projections as pj

N_SIMS = 20_000
ME = my_name()      # you, from config.json
BOOSTS = [0, 2, 4, 6]


def standings_so_far(details, name_of, last_done):
    table = {n: {"league_pts": 0, "points_for": 0} for n in name_of.values()}
    for m in details["matches"]:
        if m["event"] > last_done:
            continue
        a, b = name_of[m["league_entry_1"]], name_of[m["league_entry_2"]]
        sa, sb = m["league_entry_1_points"], m["league_entry_2_points"]
        table[a]["points_for"] += sa; table[b]["points_for"] += sb
        if sa > sb: table[a]["league_pts"] += 3
        elif sb > sa: table[b]["league_pts"] += 3
        else: table[a]["league_pts"] += 1; table[b]["league_pts"] += 1
    return pd.DataFrame(table).T


def strength_from_results(weekly):
    """Shrink each team's weekly average toward the league average."""
    g = weekly.groupby("manager")["actual"]
    means, n = g.mean(), g.size()
    within_var = weekly.groupby("manager")["actual"].var().mean()     # week-to-week noise
    league_mean = means.mean()
    between_var = max(means.var() - within_var / n.mean(), 0.0)       # real differences between teams
    weight = between_var / (between_var + within_var / n)             # how much to trust each team's own average
    mu = league_mean + weight * (means - league_mean)
    mu_sd = np.sqrt(1 / (1 / max(between_var, 1e-9) + n / within_var)) if between_var > 0 else pd.Series(0.0, index=means.index)
    return mu, mu_sd, np.sqrt(within_var), weight, between_var, league_mean


def strength_from_squads(league_mean, owner_names):
    game, last_done, players, fixtures, pts, mins = pj.load()
    tm = pj.team_matches(fixtures)
    best = pj.chosen_variant(players, pts, mins, tm, last_done)
    _, _, _, t = pj.waiver_tool(players, pts, mins, tm, last_done, game["next_event"], best)
    t = t[t.owner.notna()]
    xi = {}
    for owner, squad in t.groupby("owner"):
        xi[owner_names[int(owner)]] = best_lineup(list(zip(squad.pos, squad.proj_if_fit / pj.HORIZON)))
    xi = pd.Series(xi)
    return xi - xi.mean() + league_mean, xi


def simulate(details, name_of, last_done, base, mu, mu_sd, sigma, boost=0.0, rng=None):
    names = list(mu.index)
    idx = {n: i for i, n in enumerate(names)}
    remaining = [(m["event"], idx[name_of[m["league_entry_1"]]], idx[name_of[m["league_entry_2"]]])
                 for m in details["matches"] if m["event"] > last_done]
    true_mu = mu.values + rng.standard_normal((N_SIMS, len(names))) * np.asarray(mu_sd.reindex(names) if hasattr(mu_sd, "reindex") else mu_sd)
    true_mu[:, idx[ME]] += boost
    lp = np.tile(base.loc[names, "league_pts"].values.astype(float), (N_SIMS, 1))
    pf = np.tile(base.loc[names, "points_for"].values.astype(float), (N_SIMS, 1))
    for _, a, b in remaining:
        sa = np.round(true_mu[:, a] + sigma * rng.standard_normal(N_SIMS))
        sb = np.round(true_mu[:, b] + sigma * rng.standard_normal(N_SIMS))
        pf[:, a] += sa; pf[:, b] += sb
        lp[:, a] += np.where(sa > sb, 3, np.where(sa == sb, 1, 0))
        lp[:, b] += np.where(sb > sa, 3, np.where(sa == sb, 1, 0))
    key = lp * 10_000 + pf                          # league points first, then points scored
    rank = (-key).argsort(1).argsort(1) + 1         # 1 = first place
    return pd.DataFrame(rank, columns=names), pd.DataFrame(lp, columns=names)


def summarise(rank, lp):
    return pd.DataFrame({
        "avg_finish": rank.mean(),
        # The three outcomes that matter in this league: top 2, middle 6, bottom 2
        "p_top2": (rank <= 2).mean() * 100,
        "p_middle6": ((rank >= 3) & (rank <= 8)).mean() * 100,
        "p_bottom2": (rank >= 9).mean() * 100,
        "p_first": (rank == 1).mean() * 100,
        "p_last": (rank == rank.values.max()).mean() * 100,
        "avg_league_pts": lp.mean(),
    }).sort_values("avg_finish").round(1)


def main():
    game = get("game")
    last_done = game["current_event"] if game["current_event_finished"] else game["current_event"] - 1
    details = get(f"league/{LEAGUE_ID}/details", refresh=False)
    owner_names = manager_labels(details)                                      # entry_id -> unique label
    name_of = {e["id"]: owner_names[e["entry_id"]] for e in details["league_entries"]}  # league entry id -> label
    base = standings_so_far(details, name_of, last_done)

    # Check against the official table
    official = {name_of[s["league_entry"]]: s["total"] for s in details["standings"]}
    bad = [n for n in official if base.loc[n, "league_pts"] != official[n]]
    assert not bad, f"Standings don't match the official table for {bad}. The league data may be mid-update; try again later."

    weekly = pd.read_excel("bench_points.xlsx", sheet_name="Every week")
    weekly["manager"] = weekly["entry_id"].map(owner_names)
    mu_a, mu_sd_a, sigma, weight, between_var, league_mean = strength_from_results(weekly)
    mu_b, xi_raw = strength_from_squads(league_mean, owner_names)

    rng = np.random.default_rng(42)
    out, mine = {}, []
    for label, mu, mu_sd in [("A_results", mu_a, mu_sd_a), ("B_squads", mu_b, pd.Series(0.0, index=mu_b.index))]:
        for boost in BOOSTS:
            rank, lp = simulate(details, name_of, last_done, base, mu, mu_sd, sigma, boost, rng)
            s = summarise(rank, lp)
            out[f"{label}_+{boost}"] = s
            row = s.loc[ME].to_dict()
            row.update({"strength_model": label, "boost_per_week": boost})
            row["finish_dist"] = rank[ME].value_counts(normalize=True).sort_index().mul(100).round(1).to_dict()
            mine.append(row)
    mine = pd.DataFrame(mine)

    # Luck: wins you "deserved" (your score vs every other team that week) vs wins you got
    allplay = weekly.assign(rank_in_week=weekly.groupby("gw")["actual"].rank(method="average") - 1)
    exp_wins = (allplay.groupby("manager")["rank_in_week"].sum() / (weekly.manager.nunique() - 1)).round(1)
    actual_wins = pd.Series({name_of[s["league_entry"]]: s["matches_won"] for s in details["standings"]})
    strengths_luck = pd.DataFrame({"expected_wins": exp_wins, "actual_wins": actual_wins, "luck_wins": (actual_wins - exp_wins).round(1)})

    strengths = pd.DataFrame({"results_avg": weekly.groupby("manager")["actual"].mean().round(1),
                              "A_strength": mu_a.round(1), "A_uncertainty_sd": pd.Series(mu_sd_a).round(1),
                              "B_squad_xi_raw": xi_raw.round(1), "B_strength": mu_b.round(1),
                              "league_pts_now": base["league_pts"], "points_for_now": base["points_for"]})
    with pd.ExcelWriter("season_sim.xlsx") as xw:
        mine.drop(columns="finish_dist").to_excel(xw, sheet_name="Your scenarios", index=False)
        pd.DataFrame({f'{r.strength_model} +{r.boost_per_week}': r.finish_dist for r in mine.itertuples()}).fillna(0) \
            .to_excel(xw, sheet_name="Your finish distribution")
        strengths.join(strengths_luck).to_excel(xw, sheet_name="Team strengths")
        for k, s in out.items():
            s.to_excel(xw, sheet_name=k[:31])

    pd.set_option("display.width", 220)
    print(f"Through GW{last_done}; simulating GW{last_done + 1}-38, {N_SIMS:,} seasons each.")
    print(f"Weekly spread (1 SD) {sigma:.1f} pts; real spread between teams (SD) {np.sqrt(between_var):.1f} pts; "
          f"trust in each team's own average {pd.Series(weight).mean() * 100:.0f}%\n")
    print(strengths.join(strengths_luck).sort_values("A_strength", ascending=False).to_string(), "\n")
    print(out["A_results_+0"], "\n")
    print(out["B_squads_+0"], "\n")
    print(mine.drop(columns="finish_dist").to_string(index=False))


if __name__ == "__main__":
    main()
