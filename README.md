# FPL Draft Analysis

Statistics for a head-to-head [Fantasy Premier League Draft](https://draft.premierleague.com) league. It grades every manager's draft, lineup choices and waiver pickups, finds what wins matches, projects free agents for waiver decisions, and simulates the rest of the season.

Everything comes from FPL Draft's public data feed, so no login is needed. You only need your league's ID.

## What it answers

| Question | Script | Output |
|---|---|---|
| Who drafted best, given what was still available at each pick? | `draft_grades.py` | `draft_grades.xlsx` |
| Who picks the right starting 11? How many points were left on the bench? | `bench_points.py` | `bench_points.xlsx` |
| Which positions' points are most linked to winning? | `win_factors.py` | `win_factors.xlsx` |
| What score gives a 50% win chance? Do totals or per-player models predict better? | `win_models.py` | `win_models.xlsx` |
| Who is best at picking up free agents (points above replacement, or WAR)? | `fa_war.py` | `fa_war.xlsx` |
| Which free agents should I pick up this week, and who should I drop? | `projections.py` | `waiver_tool.xlsx` |
| Which injured or barely used free agents are worth stashing? | `potential.py` | `stash_watchlist.xlsx` |
| Where am I likely to finish, and how much would better waivers help? | `simulate_season.py` | `season_sim.xlsx` |

Every script starts with a plain-English explanation of its method.

## Setup

You need Python 3.11 or newer.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp config.example.json config.json
```

Then put your own IDs in `config.json`:

- `league_id`: sign in at draft.premierleague.com, then open `https://draft.premierleague.com/api/bootstrap-dynamic`. Your league's ID is the `id` under `"leagues"`.
- `my_entry_id`: on the same page, the number under `"entry_set"`.

`config.json` is listed in `.gitignore`, so your IDs are never committed.

## Running it

Run the scripts in this order. Later scripts reuse data that earlier ones download and save in `data/`.

```bash
.venv/bin/python draft_grades.py
.venv/bin/python bench_points.py
.venv/bin/python win_factors.py
.venv/bin/python win_models.py
.venv/bin/python fa_war.py
.venv/bin/python projections.py
.venv/bin/python potential.py
.venv/bin/python simulate_season.py
```

Before each waiver deadline, rerun everything from `bench_points.py` on to refresh the waiver tool, stash list and season outlook.

`export_dashboard_data.py` turns the results into small JSON files for a dashboard.

### Stash news (optional)

`potential.py` can add injury and playing-time news to the stash list. Put a `stash_news.json` file next to the scripts: a list of objects with `player`, `news_summary`, `weeks_out` and `role_outlook` (`starter`, `rotation`, `bench` or `unclear`). This project filled it with an AI research agent that read recent news, but any source works.

## Methods in brief

- **Draft score:** each pick's points minus the average of the 10 best-ranked players still available at the same position, compared with the league average for that round.
- **Lineup call accuracy:** for every benched player who actually played, was each starter he could legally have replaced worth at least as much?
- **WAR for pickups:** points above the 5 free agents in the best recent form at the same position, over the weeks the manager owned the player.
- **Projections:** points per 90 minutes, pulled toward the position average when a player has played little, times expected minutes, adjusted for fixture difficulty and FPL's injury news. They are backtested against simple rules on past gameweeks.
- **Season simulation:** 20,000 simulated seasons, with team strength judged from results so far (pulled toward the league average) or from current squads.

## Caveats

- A few months of results is a small sample. Most scripts report likely ranges, and many gaps between managers are within normal noise.
- Weekly fantasy points are noisy. The projection is only slightly better than a season average, so treat waiver gains as best bets.
- The best possible lineup uses hindsight, which no manager has at the deadline.

## Privacy

The downloaded data and every output name the managers and teams in your league, so `.gitignore` keeps `data/`, `dashboard_data/`, all `.xlsx` files and the stash files out of the repository.
