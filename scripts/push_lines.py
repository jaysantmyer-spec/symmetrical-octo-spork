"""
Pull TT Elite Series lines from DraftKings on YOUR computer (DraftKings blocks cloud servers, not home
connections) and push them to the repo as data/odds.csv, where the app and the refresh workflow read them.

    export GITHUB_TOKEN=github_pat_...       # fine-grained token, Contents: read/write on this repo
    python scripts/push_lines.py             # once
    python scripts/push_lines.py --loop 10   # every 10 minutes until you stop it (Ctrl-C)

Needs: pip install requests pandas
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


def once() -> int:
    df = dk.fetch_dk(save_raw=False)
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
    a = ap.parse_args()
    while True:
        try:
            once()
        except Exception as ex:
            print("failed:", ex)
        if not a.loop:
            break
        time.sleep(a.loop * 60)
