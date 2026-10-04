# Setting up TT Elite Oracle

## 1. Pull the history (once)
GitHub repo → **Actions** tab → **backfill history (run once)** → **Run workflow**.
Leave pages=80 (100 posts per page; the archive is ~5,900 posts). It takes a few minutes. When it finishes it commits
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
