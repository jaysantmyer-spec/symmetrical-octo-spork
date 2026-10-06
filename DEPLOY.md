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
