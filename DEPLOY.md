# Setting up TT Elite Oracle

## 1. Pull the history (once)
GitHub repo → **Actions** tab → **backfill history (run once)** → **Run workflow**.
Leave pages=400 (100 posts per page; it stops when the archive ends). It takes 20–40 minutes. When it finishes it commits
`data/matches.csv`, trains the first model, runs a 60-day backtest and writes the first predictions.

If the run ends with few or no matches, open the run's log: the parser saves the first few raw pages to
`data/raw/` so the HTML can be checked and the parser adjusted.

## 2. Keep it fresh (automatic)
The **refresh** workflow runs every 6 hours: new results and fixtures, DraftKings snapshot, grade the
ledger, re-learn, retrain, predict. It commits to the repo, and Streamlit Cloud redeploys automatically.
You can also run it by hand from the Actions tab.

## 3. Deploy the app
share.streamlit.io → Create app → this repo, branch `main`, main file `app.py`, Python 3.11 → Deploy.

## DraftKings lines
`tt_oracle/dk.py` tries DraftKings' public JSON feeds and saves what it gets to `data/raw/dk_*.json`.
If the first refresh logs show no lines, the feed id needs pinning: add a repository variable
(Settings → Secrets and variables → Actions → Variables) named `DK_LEAGUE_ID` or `DK_EVENT_GROUP_ID`
with the id found in the raw files. Until then, the Pricing desk tab accepts pasted lines.

## Daily use
Open the app on your phone. **Today** shows the sessions with picks highlighted; **Pricing desk** flags
edges vs DraftKings; **Track record** shows how the logged picks have done. Nothing to press.

## Saved picks (My picks tab)
Picks you save are written to `data/my_picks.csv`. To keep them across the app's restarts, give the app a
GitHub token so it can commit that file:
1. github.com → your avatar → Settings → Developer settings → Personal access tokens → **Fine-grained tokens**
   → Generate new token. Repository access: only this repo. Permissions: **Contents: Read and write**. Copy it.
2. share.streamlit.io → the app → Settings → **Secrets** → add a line `GITHUB_TOKEN = "github_pat_..."` → Save.
The app picks it up on the next load. Without it, picks live only until the next redeploy.

## Name merging
The site spells players inconsistently (Radło / Radlo, typos). Merging is on by default. The sidebar checkbox
turns it off for your session; to turn it off everywhere, set `TT_MERGE_NAMES = "0"` in the app's Secrets and as
a repository variable (GitHub → Settings → Secrets and variables → Actions → Variables).

## DraftKings matches on the Today tab

The Today tab has a "DraftKings matches — odds feed" panel with three routes to current lines:

1. **Odds provider with an API key** — currently OddsPapi (oddspapi.io; DraftKings is one of its books). Paste the key
   in the panel (kept for the session) or add `ODDSPAPI_KEY = "..."` to the app's Secrets to keep it. The panel's log
   says what it found at each step (sport → "TT Elite" tournament → DraftKings moneylines); raw responses land in
   `data/raw/provider_*.json` so the parser can be adjusted to the provider's exact format.
2. **DraftKings' own feed** — no key; works only if DraftKings doesn't block the server's IP.
3. **Paste box** on the Pricing desk.

Loaded lines are matched to the cards ("DraftKings: … / market …") and the "Only matches on DraftKings" box filters
the day to those.

### Pushing DraftKings lines from your own computer (the route that works)

DraftKings serves its odds JSON to home connections but not to cloud servers (Streamlit, GitHub). So pull from your
Mac and push to the repo:

```
cd symmetrical-octo-spork            # your clone of the repo
pip install requests pandas          # once
export GITHUB_TOKEN=github_pat_...   # the same fine-grained token as for saved picks (Contents: read/write)
python scripts/push_lines.py         # once;  or:  python scripts/push_lines.py --loop 10   (every 10 min)
```

It commits `data/odds.csv` (and appends `data/odds_history.csv`). The app re-reads it within a minute, matches the
lines to the day's cards, and the next refresh prices its predictions against them. Leave `--loop 10` running in a
terminal on match days.

## Per-game points feed (vendor)

tt-series.com publishes set scores only. Per-game points (11-9, 8-11, …) come from a licensed vendor — BetsAPI
(betsapi.com; table tennis sport id 92, league "TT Elite Series"). Add the token as a repository secret
`BETSAPI_TOKEN`, run the **points backfill** workflow once (a year of history), and the 6-hourly refresh keeps it
current. With ≥1,500 linked matches the model adds: expected total points, P(win | wins game 1), deuce-game
records, points-per-game form. Raw responses are kept in `data/raw/betsapi_*.json` for checking the parser on
the first run; `BETSAPI_LEAGUE_IDS` / `BETSAPI_LEAGUE_RE` override league discovery.

## Market blend

Lines pushed by `scripts/push_lines.py` accumulate in `data/odds_history.csv`; each refresh derives the closing
line per match (`data/closing_lines.csv`) and, once 300 graded predictions have one, fits a model × market
blend (`models/market_blend.json`) shown on the cards as "Model × market blend". The ledger keeps grading the
pure model so the two stay comparable.

## Spreads (games ±1.5, points ±3.5)

Every card, the H2H breakdown and the Pricing desk's "Spreads" table show the chance each side covers
DraftKings' two spread markets, with the fair American price. Games −1.5 (wins 3-0 or 3-1) is a logistic on the
match probability fitted at every retrain; the points spread comes from a winner/loser mixture of the total-points
margin fitted on the vendor's game-by-game points (`spread_model` inside `models/model.joblib`; lines 3.5, 5.5 and
7.5 are precomputed as `p_a_pts_m35` etc.). Both are checked on the walk-forward backtest ("Spreads" on the Track
record tab). The "Check a DraftKings spread price" box turns a DK price into break-even % and EV per $1.

## DraftKings lines without a terminal: the "Copy DK lines" bookmark

DraftKings' odds API refuses everything but its own web page (even Safari typing the API address gets "Access
Denied"), so the lines are read from the page you can see. `scripts/copy_dk_lines.js` is a bookmarklet
(one-line form in `scripts/bookmarklet.txt`, also shown in the app under Pricing desk → paste box): open the TT
Elite Series page on sportsbook.draftkings.com, click the bookmark, every moneyline is copied as
`Player A -150 / Player B +120` lines; paste into the app and press "Use these lines". With `GITHUB_TOKEN` in the
app's Secrets the lines are committed to the repo (odds.csv + odds_history.csv), where the refresh turns them into
closing lines for the model × market blend. Regenerate the one-liner after editing the .js with
`python scripts/make_bookmarklet.py`.
