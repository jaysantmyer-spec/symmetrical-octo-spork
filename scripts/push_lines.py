"""
Pull TT Elite Series lines from DraftKings on YOUR computer (DraftKings blocks cloud servers, not home
connections) and push them to the repo as data/odds.csv, where the app and the refresh workflow read them.

    export GITHUB_TOKEN=github_pat_...       # fine-grained token, Contents: read/write on this repo
    python scripts/push_lines.py             # once
    python scripts/push_lines.py --loop 10   # every 10 minutes until you stop it (Ctrl-C)
    python scripts/push_lines.py --file ~/Downloads/dk.json   # lines from a JSON you saved in Safari (see below)

Needs: pip install requests pandas curl_cffi      (curl_cffi makes the request look like Chrome; DraftKings 403s without it)

Manual route if DraftKings still refuses: open this in Safari, which always passes --
    https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/208037?format=json
then File > Save As..., format "Page Source", name dk.json, and run with --file.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TT_ORACLE_ROOT", str(ROOT))
from tt_oracle import dk  # noqa: E402

REPO = os.getenv("GITHUB_REPO", "jaysantmyer-spec/symmetrical-octo-spork")
PATHS = {"data/odds.csv": "latest lines", "data/odds_history.csv": "history"}


def gh(method, path, **kw):
    tok = (os.getenv("GITHUB_TOKEN") or "").strip()
    if not tok:
        sys.exit("GITHUB_TOKEN is not set (fine-grained token with Contents: read/write on the repo)")
    r = requests.request(method, f"https://api.github.com/repos/{REPO}/contents/{path}",
                         headers={"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}, timeout=30, **kw)
    return r


def push_file(path: str, content: str, message: str) -> None:
    r = gh("GET", path)
    sha = r.json().get("sha") if r.status_code == 200 else None
    body = {"message": message, "content": base64.b64encode(content.encode()).decode()}
    if sha:
        body["sha"] = sha
    r = gh("PUT", path, json=body)
    r.raise_for_status()


def from_file(path: str) -> pd.DataFrame:
    import json
    payload = json.loads(Path(path).read_text())
    rows = dk._parse_v5(payload) if "eventGroup" in payload or "offerCategories" in payload else dk._parse_nash(payload)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["odds_1"] = pd.to_numeric(df["odds_1"], errors="coerce"); df["odds_2"] = pd.to_numeric(df["odds_2"], errors="coerce")
    df = df.dropna(subset=["odds_1", "odds_2"]); df["bookmaker"] = "DraftKings"; df["fetched_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return df.drop_duplicates(["player_1", "player_2", "commence_time"])


def once(file: str | None = None) -> int:
    df = from_file(file) if file else dk.fetch_dk(save_raw=False)
    if df.empty:
        print("DraftKings returned no TT Elite lines:\n  " + "\n  ".join(dk.LAST_FETCH_LOG))
        return 0
    df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce").dt.tz_convert(None)
    # history: append to what the repo has (keep it bounded)
    r = gh("GET", "data/odds_history.csv")
    hist = pd.DataFrame()
    if r.status_code == 200:
        from io import StringIO
        hist = pd.read_csv(StringIO(base64.b64decode(r.json()["content"]).decode()))
    hist = pd.concat([hist, df], ignore_index=True).tail(100_000) if not hist.empty else df
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M")
    push_file("data/odds.csv", df.to_csv(index=False), f"DraftKings lines {stamp} ({len(df)} matches)")
    push_file("data/odds_history.csv", hist.to_csv(index=False), f"odds history {stamp}")
    print(f"{pd.Timestamp.now():%H:%M}  pushed {len(df)} DraftKings lines")
    for _, x in df.sort_values("commence_time").iterrows():
        print(f"   {pd.Timestamp(x['commence_time']):%d %b %H:%M}  {x['player_1']} {dk.decimal_to_american(x['odds_1'])} / "
              f"{x['player_2']} {dk.decimal_to_american(x['odds_2'])}")
    return len(df)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0, help="repeat every N minutes")
    ap.add_argument("--file", default=None, help="parse a DraftKings JSON saved from the browser instead of fetching")
    a = ap.parse_args()
    while True:
        try:
            once(a.file)
        except Exception as ex:
            print("failed:", ex)
        if not a.loop:
            break
        time.sleep(a.loop * 60)
