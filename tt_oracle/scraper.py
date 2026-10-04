"""
Collector for tt-series.com (the TT Elite Series official site).

Every tournament is one WordPress post, numbered sequentially, e.g.
    https://www.tt-series.com/5767-result-2-10-2026-afternoon-tournament-osp/
The post can also be reached by id:  https://www.tt-series.com/?p=5767
Each page holds a 6-player round-robin: a standings table and a schedule table with
time, "1 vs 6", player A, player B, and the set score ("3/2", older pages "3:2").
Pages for tomorrow's tournaments exist before play starts (scores empty) - those are the fixtures.

    python -m tt_oracle backfill --start 1 --end 5900      # one-off history pull
    python -m tt_oracle update                             # new pages since the last one we have
"""
from __future__ import annotations

import logging
import random
import re
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from . import config, store

log = logging.getLogger(__name__)
UA = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
]
TITLE_RX = re.compile(r"(\d+)\.?\s*Results?\s+(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})\s*[–\-]?\s*(\w+)?\s*tournament\s*(\w+)?", re.I)
SCORE_RX = re.compile(r"^\s*(\d)\s*[/:–\-]\s*(\d)\s*$")
TIME_RX = re.compile(r"^\s*(\d{1,2})[:.](\d{2})\s*$")
PAIR_RX = re.compile(r"^\s*(\d)\s*vs?\.?\s*(\d)\s*$", re.I)


# --------------------------------------------------------------------------- HTTP
_session: Optional[requests.Session] = None


def _get(url: str, timeout: int = 25) -> requests.Response:
    global _session
    if _session is None:
        _session = requests.Session()
    time.sleep(random.uniform(0.2, 0.6))  # be polite: one small site, thousands of pages
    return _session.get(url, headers={"User-Agent": random.choice(UA), "Accept-Language": "en,pl;q=0.8"},
                        timeout=timeout, allow_redirects=True)


def fetch_post(post_id: int) -> tuple[int, str, str]:
    """Returns (status_code, final_url, html)."""
    r = _get(f"{config.SITE}/?p={post_id}")
    return r.status_code, r.url, r.text


# --------------------------------------------------------------------------- parsing
def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _norm_name(s: str) -> str:
    """'Jadach, Oskar' -> 'Oskar Jadach'; collapses whitespace and stray punctuation."""
    s = _clean(s).strip(" .")
    if "," in s:
        last, first = [p.strip() for p in s.split(",", 1)]
        s = f"{first} {last}"
    return s


def parse_title(title: str) -> Optional[dict]:
    m = TITLE_RX.search(title or "")
    if not m:
        return None
    pid, d, mth, y, session, venue = m.groups()
    try:
        date = datetime(int(y), int(mth), int(d)).date()
    except ValueError:
        return None
    return {"post_id": int(pid), "date": pd.Timestamp(date), "session": (session or "").lower(),
            "venue": (venue or "").upper()}


def parse_page(html: str, url: str = "") -> tuple[Optional[dict], pd.DataFrame]:
    """Returns (tournament dict or None if this isn't a tournament page, matches DataFrame)."""
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find(["h1", "title"])
    title = _clean(title_tag.get_text(" ")) if title_tag else ""
    meta = parse_title(title)
    if meta is None:
        # some pages put the number in the h1 and the date elsewhere; try any heading
        for h in soup.find_all(["h1", "h2", "h3"]):
            meta = parse_title(_clean(h.get_text(" ")))
            if meta:
                break
    if meta is None:
        return None, pd.DataFrame()
    meta["url"] = url
    meta["title"] = title

    rows = []
    order = 0
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = [_clean(td.get_text(" ")) for td in tr.find_all(["td", "th"])]
            if len(cells) < 4:
                continue
            # schedule rows look like: time | "1 vs 6" | player | player | score   (score may be empty)
            t = next((c for c in cells if TIME_RX.match(c)), None)
            pair = next((c for c in cells if PAIR_RX.match(c)), None)
            if t is None or pair is None:
                continue
            names = [c for c in cells if c not in (t, pair) and not SCORE_RX.match(c) and len(c) > 2
                     and not c.isdigit()]
            if len(names) < 2:
                continue
            score = next((c for c in cells if SCORE_RX.match(c)), "")
            sa, sb = (SCORE_RX.match(score).groups() if score else (None, None))
            hh, mm = TIME_RX.match(t).groups()
            order += 1
            rows.append({
                "post_id": meta["post_id"], "date": meta["date"], "session": meta["session"],
                "venue": meta["venue"], "order": order,
                "start_time": pd.Timestamp(meta["date"]) + pd.Timedelta(hours=int(hh), minutes=int(mm)),
                "player_a": _norm_name(names[0]), "player_b": _norm_name(names[1]),
                "sets_a": int(sa) if sa is not None else None, "sets_b": int(sb) if sb is not None else None,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["match_id"] = df["post_id"].astype(str) + "-" + df["order"].astype(str)
        df["finished"] = df["sets_a"].notna() & df["sets_b"].notna() & ((df["sets_a"] == 3) | (df["sets_b"] == 3))
        df["winner"] = None
        df.loc[df["finished"] & (df["sets_a"] > df["sets_b"]), "winner"] = "a"
        df.loc[df["finished"] & (df["sets_b"] > df["sets_a"]), "winner"] = "b"
    meta["n_matches"] = int(len(df))
    meta["n_finished"] = int(df["finished"].sum()) if not df.empty else 0
    return meta, df


# --------------------------------------------------------------------------- collection
def collect(post_ids, progress=None, keep_raw: int = 0, stop_after_missing: int = 0) -> dict:
    """Fetch and parse the given post ids. keep_raw>0 saves that many raw pages to data/raw for debugging.
    stop_after_missing>0 stops once that many consecutive ids are not tournament pages (end of the archive)."""
    config.ensure_dirs()
    tours, matches, missing_run, saved_raw = [], [], 0, 0
    ids = list(post_ids)
    for k, pid in enumerate(ids):
        try:
            code, url, html = fetch_post(pid)
        except Exception as ex:
            log.warning("fetch %s failed: %s", pid, ex)
            continue
        if keep_raw and saved_raw < keep_raw:
            (config.RAW / f"post_{pid}.html").write_text(html, encoding="utf-8")
            saved_raw += 1
        if code != 200:
            missing_run += 1
        else:
            meta, df = parse_page(html, url)
            if meta is None or df.empty:
                missing_run += 1
                if code == 200 and keep_raw and saved_raw < keep_raw + 3:
                    (config.RAW / f"unparsed_{pid}.html").write_text(html, encoding="utf-8")
                    saved_raw += 1
            else:
                missing_run = 0
                meta["fetched_at"] = pd.Timestamp.utcnow()
                tours.append(meta)
                matches.append(df)
        if progress:
            progress((k + 1) / len(ids), f"page {pid}: {len(tours)} tournaments, {sum(len(m) for m in matches)} matches")
        if stop_after_missing and missing_run >= stop_after_missing:
            log.info("stopping: %d consecutive non-tournament pages", missing_run)
            break
    out = {"pages": len(ids), "tournaments": len(tours), "matches": 0}
    if tours:
        store.upsert(pd.DataFrame(tours), config.TOURNAMENTS_CSV, "post_id")
    if matches:
        allm = pd.concat(matches, ignore_index=True)
        store.upsert(allm, config.MATCHES_CSV, "match_id")
        out["matches"] = int(len(allm))
    return out


def last_post_id() -> int:
    t = store.read(config.TOURNAMENTS_CSV)
    return int(t["post_id"].max()) if not t.empty else config.FIRST_POST_ID - 1


def update(lookahead: int = 40, refresh_unfinished: bool = True, progress=None) -> dict:
    """Re-fetch recent pages that still had unfinished matches, then look ahead for new pages."""
    ids = []
    if refresh_unfinished:
        m = store.read(config.MATCHES_CSV)
        if not m.empty:
            recent = m[(~m["finished"].astype(bool)) & (pd.to_datetime(m["date"]) >= pd.Timestamp.today().normalize() - pd.Timedelta(days=7))]
            ids += sorted(recent["post_id"].unique().tolist())
    start = last_post_id() + 1
    ids += list(range(start, start + lookahead))
    return collect(ids, progress=progress, stop_after_missing=15)


def backfill(start: int, end: int, progress=None, keep_raw: int = 3) -> dict:
    return collect(range(start, end + 1), progress=progress, keep_raw=keep_raw)
