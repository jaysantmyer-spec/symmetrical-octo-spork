"""
Collector for tt-series.com (the TT Elite Series official site, a WordPress blog).

Every tournament is one post. Its content has two tables:
  standings: Number | Name ("Krzyżanowski, Michał") | 1..6 cross-table | Matches | Sets | Ranking
  schedule:  match's hour | "1 vs 6" | PLAYER (surname) | PLAYER (surname) | RESULT ("0:3", empty if not played)
The "1 vs 6" column refers to the standings row numbers, which is how surnames get their full names.
Pages for tomorrow's tournaments exist before play starts (empty results) - those are the fixtures.

Posts are pulled through the WordPress REST API (100 per request, newest first). The WP post id is the
stable key; the "5767." number in newer titles is a separate running count and is kept as `seq`.

    python -m tt_oracle backfill --pages 80       # whole archive (~5,900 posts = ~60 requests)
    python -m tt_oracle update                    # newest pages only
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
API = f"{config.SITE}/wp-json/wp/v2/posts"
TITLE_RX = re.compile(r"(?:(\d+)\.\s*)?Results?\s+(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})\s*[–\-]*\s*([A-Za-z]+)?\s*tournament\s*[–\-]*\s*([A-Za-z]+)?", re.I)
SCORE_RX = re.compile(r"^\s*(\d)\s*[/:–\-]\s*(\d)\s*$")
TIME_RX = re.compile(r"^\s*(\d{1,2})[:.](\d{2})\s*$")
PAIR_RX = re.compile(r"^\s*(\d)\s*vs?\.?\s*(\d)\s*$", re.I)


# --------------------------------------------------------------------------- HTTP
_session: Optional[requests.Session] = None


def _get(url: str, params: dict | None = None, timeout: int = 30) -> requests.Response:
    global _session
    if _session is None:
        _session = requests.Session()
    time.sleep(random.uniform(0.3, 0.8))
    return _session.get(url, params=params, headers={"User-Agent": random.choice(UA), "Accept": "application/json, text/html"},
                        timeout=timeout)


def fetch_posts_page(page: int, per_page: int = 100) -> list[dict]:
    r = _get(API, {"per_page": per_page, "page": page, "orderby": "date", "order": "desc",
                   "_fields": "id,date,link,title,content"})
    if r.status_code == 400:   # past the last page
        return []
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------------- parsing
def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\xa0", " ")).strip()


def _full_name(s: str) -> str:
    """'Krzyżanowski, Michał' -> 'Michał Krzyżanowski'."""
    s = _clean(s).strip(" .")
    if "," in s:
        last, first = [p.strip() for p in s.split(",", 1)]
        return f"{first} {last}".strip()
    return s


def parse_title(title: str) -> Optional[dict]:
    m = TITLE_RX.search(_clean(title))
    if not m:
        return None
    seq, d, mth, y, session, venue = m.groups()
    try:
        date = datetime(int(y), int(mth), int(d)).date()
    except ValueError:
        return None
    session = (session or "").lower()
    if session not in config.SESSIONS:
        session = session or "unknown"
    return {"seq": int(seq) if seq else None, "date": pd.Timestamp(date), "session": session,
            "venue": (venue or "").upper() or "MAIN"}


def _standings(table) -> dict[int, str]:
    seats = {}
    for tr in table.find_all("tr"):
        cells = [_clean(td.get_text(" ")) for td in tr.find_all(["td", "th"])]
        if len(cells) >= 3 and cells[0].isdigit() and 1 <= int(cells[0]) <= 8 and re.search(r"[A-Za-zÀ-ž]", cells[1]):
            seats[int(cells[0])] = _full_name(cells[1])
    return seats


def parse_page(html: str, post_id: int, title: str, url: str = "", post_date: str | None = None) -> tuple[Optional[dict], pd.DataFrame]:
    """Returns (tournament dict or None if this post isn't a tournament, matches DataFrame)."""
    meta = parse_title(title)
    soup = BeautifulSoup(html, "html.parser")
    if meta is None:
        h1 = soup.find("h1")
        meta = parse_title(h1.get_text(" ")) if h1 else None
    if meta is None:
        return None, pd.DataFrame()
    meta.update({"post_id": int(post_id), "url": url, "title": _clean(title)})
    tables = soup.find_all("table")
    seats: dict[int, str] = {}
    for t in tables:
        seats = _standings(t)
        if len(seats) >= 4:
            break
    rows, order = [], 0
    for table in tables:
        for tr in table.find_all("tr"):
            cells = [_clean(td.get_text(" ")) for td in tr.find_all(["td", "th"])]
            if len(cells) < 4:
                continue
            t = next((c for c in cells if TIME_RX.match(c)), None)
            pair = next((c for c in cells if PAIR_RX.match(c)), None)
            if t is None or pair is None:
                continue
            i, j = (int(x) for x in PAIR_RX.match(pair).groups())
            names = [c for c in cells if c not in (t, pair) and not SCORE_RX.match(c) and re.search(r"[A-Za-zÀ-ž]{2}", c)]
            a = seats.get(i) or (_full_name(names[0]) if names else None)
            b = seats.get(j) or (_full_name(names[1]) if len(names) > 1 else None)
            if not a or not b:
                continue
            score = next((c for c in cells if SCORE_RX.match(c)), "")
            sa, sb = (SCORE_RX.match(score).groups() if score else (None, None))
            hh, mm = TIME_RX.match(t).groups()
            order += 1
            rows.append({"post_id": int(post_id), "seq": meta["seq"], "date": meta["date"], "session": meta["session"],
                         "venue": meta["venue"], "order": order,
                         "start_time": pd.Timestamp(meta["date"]) + pd.Timedelta(hours=int(hh), minutes=int(mm)),
                         "player_a": a, "player_b": b,
                         "sets_a": int(sa) if sa is not None else None, "sets_b": int(sb) if sb is not None else None})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["match_id"] = df["post_id"].astype(str) + "-" + df["order"].astype(str)
        df["finished"] = df["sets_a"].notna() & df["sets_b"].notna() & ((df["sets_a"] == 3) | (df["sets_b"] == 3))
        df["winner"] = None
        df.loc[df["finished"] & (df["sets_a"] > df["sets_b"]), "winner"] = "a"
        df.loc[df["finished"] & (df["sets_b"] > df["sets_a"]), "winner"] = "b"
    meta["n_matches"] = int(len(df))
    meta["n_finished"] = int(df["finished"].sum()) if not df.empty else 0
    meta["n_players"] = len(seats)
    return meta, df


# --------------------------------------------------------------------------- collection
def _ingest(posts: list[dict], keep_raw_dir=None, keep_raw: int = 0) -> tuple[list[dict], list[pd.DataFrame]]:
    tours, matches, saved = [], [], 0
    for p in posts:
        title = (p.get("title") or {}).get("rendered", "")
        html = (p.get("content") or {}).get("rendered", "")
        meta, df = parse_page(html, p["id"], BeautifulSoup(title, "html.parser").get_text(), p.get("link", ""), p.get("date"))
        if keep_raw and saved < keep_raw and (meta is None or df.empty):
            (config.RAW / f"post_{p['id']}.html").write_text(f"<!-- {title} -->\n{html}", encoding="utf-8")
            saved += 1
        if meta is None or df.empty:
            continue
        meta["fetched_at"] = pd.Timestamp.utcnow()
        tours.append(meta)
        matches.append(df)
    return tours, matches


def _save(tours, matches) -> dict:
    out = {"tournaments": len(tours), "matches": 0}
    if tours:
        store.upsert(pd.DataFrame(tours), config.TOURNAMENTS_CSV, "post_id")
    if matches:
        allm = pd.concat(matches, ignore_index=True)
        store.upsert(allm, config.MATCHES_CSV, "match_id")
        out["matches"] = int(len(allm))
    return out


def backfill(pages: int = 80, progress=None, keep_raw: int = 3) -> dict:
    """Walk the archive newest-first, 100 posts a page, until it runs out or `pages` is reached."""
    config.ensure_dirs()
    tours, matches = [], []
    for page in range(1, pages + 1):
        try:
            posts = fetch_posts_page(page)
        except Exception as ex:
            log.warning("page %s failed: %s", page, ex)
            break
        if not posts:
            break
        t, m = _ingest(posts, keep_raw=keep_raw if page == 1 else 0)
        tours += t; matches += m
        if progress:
            progress(page / pages, f"page {page}: {len(tours)} tournaments, {sum(len(x) for x in matches)} matches")
        if len(posts) < 100:
            break
    out = _save(tours, matches)
    out["pages"] = page
    return out


def update(pages: int = 2, progress=None) -> dict:
    """Newest `pages` x 100 posts: catches new fixtures, results for recent tournaments, and corrections."""
    return backfill(pages=pages, progress=progress, keep_raw=0)
