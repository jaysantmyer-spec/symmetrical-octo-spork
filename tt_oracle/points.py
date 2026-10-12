"""
Per-game point scores for TT Elite Series matches, from a licensed data vendor (BetsAPI, token in BETSAPI_TOKEN).

tt-series.com publishes only set scores (3:1). Points per game (11-9, 8-11, ...) unlock: total-points markets,
"won game 1 -> won match" statistics, close-game (deuce) records and sharper ratings. This module

  1. pulls ended events for the TT Elite league day by day (data/points.csv, one row per match with the
     game-by-game scores), and
  2. links each to our match_id by date + player names (either order), so the feature builder can use them.

Every raw response is saved to data/raw/betsapi_*.json; the first run on a new account should be checked
against them (the JSON shape is written down below from BetsAPI's docs, but field names are verified on real data).
"""
from __future__ import annotations

import json
import os
import re
import time

import numpy as np
import pandas as pd
import requests

from . import config, store
from .names import fold

API = "https://api.b365api.com"
SPORT_ID = int(os.getenv("BETSAPI_SPORT_ID", "92"))          # table tennis
LEAGUE_RE = os.getenv("BETSAPI_LEAGUE_RE", r"tt\s*elite")
POINTS_CSV = config.DATA / "points.csv"
COLS = ["event_id", "date", "start_time", "league", "player_1", "player_2", "sets_1", "sets_2", "games",
        "n_games", "points_1", "points_2", "first_game_winner", "deuce_games", "match_id"]
LAST_LOG: list[str] = []


def token() -> str | None:
    t = os.getenv("BETSAPI_TOKEN")
    if not t:
        try:
            import streamlit as st
            t = st.secrets.get("BETSAPI_TOKEN")
        except Exception:
            t = None
    return (t or "").strip() or None


def _get(path: str, **params) -> dict:
    tok = token()
    if not tok:
        raise RuntimeError("BETSAPI_TOKEN is not set")
    r = requests.get(f"{API}/{path}", params={"token": tok, **params}, timeout=40)
    config.ensure_dirs()
    tag = re.sub(r"[^A-Za-z0-9]+", "_", f"{path}_{'_'.join(f'{k}{v}' for k, v in params.items())}")
    (config.RAW / f"betsapi_{tag}.json").write_text(r.text[:2_000_000])
    r.raise_for_status()
    d = r.json()
    if d.get("success") not in (1, True, None):
        raise RuntimeError(f"BetsAPI error: {d.get('error', d)}")
    return d


# --------------------------------------------------------------------------- league discovery
def league_ids() -> list[int]:
    ids = os.getenv("BETSAPI_LEAGUE_IDS", "")
    if ids.strip():
        return [int(x) for x in ids.split(",") if x.strip().isdigit()]
    out, page = [], 1
    while page <= 10:
        d = _get("v1/league", sport_id=SPORT_ID, page=page)
        res = d.get("results", [])
        out += [int(l["id"]) for l in res if re.search(LEAGUE_RE, str(l.get("name", "")), re.I)]
        if len(res) < 100:
            break
        page += 1
    return out


# --------------------------------------------------------------------------- parsing
def _parse_event(ev: dict) -> dict | None:
    """BetsAPI ended event: {id, time (unix), league{name}, home{name}, away{name}, ss '3-1',
    scores {'1': {home:'11', away:'9'}, ...}}. Tolerant of missing pieces."""
    try:
        ss = str(ev.get("ss") or "")
        m = re.match(r"(\d+)-(\d+)", ss)
        if not m:
            return None
        s1, s2 = int(m.group(1)), int(m.group(2))
        games = []
        for k in sorted((ev.get("scores") or {}).keys(), key=lambda x: int(x) if str(x).isdigit() else 99):
            g = ev["scores"][k]
            if g is None:
                continue
            h, a = g.get("home"), g.get("away")
            if h in (None, "") or a in (None, ""):
                continue
            games.append((int(h), int(a)))
        games = [g for g in games if max(g) >= 11 or (g[0] + g[1]) >= 20]        # drop empty / aborted games
        if len(games) < 3:
            return None
        start = pd.to_datetime(int(ev["time"]), unit="s", utc=True).tz_convert("Europe/Warsaw").tz_localize(None)
        return {"event_id": str(ev.get("id")), "date": start.normalize(), "start_time": start,
                "league": (ev.get("league") or {}).get("name"),
                "player_1": fold((ev.get("home") or {}).get("name", "")), "player_2": fold((ev.get("away") or {}).get("name", "")),
                "sets_1": s1, "sets_2": s2, "games": json.dumps(games), "n_games": len(games),
                "points_1": sum(g[0] for g in games), "points_2": sum(g[1] for g in games),
                "first_game_winner": 1 if games[0][0] > games[0][1] else 2,
                "deuce_games": sum(1 for g in games if min(g) >= 10), "match_id": None}
    except Exception:
        return None


def fetch_day(day: str, leagues: list[int]) -> pd.DataFrame:
    """All ended TT Elite events on `day` (YYYYMMDD), every page."""
    rows = []
    for lid in leagues:
        page = 1
        while page <= 20:
            d = _get("v3/events/ended", sport_id=SPORT_ID, league_id=lid, day=day, page=page)
            res = d.get("results", [])
            rows += [r for r in (_parse_event(e) for e in res) if r]
            pager = d.get("pager") or {}
            if len(res) < int(pager.get("per_page") or 50) or page * int(pager.get("per_page") or 50) >= int(pager.get("total") or 0):
                break
            page += 1
            time.sleep(0.2)
    return pd.DataFrame(rows, columns=COLS)


# --------------------------------------------------------------------------- linking to our matches
def _key(name: str) -> frozenset:
    return frozenset(t for t in re.split(r"[^a-z]+", fold(name).lower()) if len(t) >= 2)


def link(points: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Fill match_id by same day + both surnames present (either order) + same set score; nearest start time."""
    if points.empty or matches.empty:
        return points
    m = matches[matches["finished"].astype(bool)].copy()
    m["day"] = pd.to_datetime(m["start_time"]).dt.normalize()
    m["ka"], m["kb"] = m["player_a"].map(_key), m["player_b"].map(_key)
    by_day = {d: g for d, g in m.groupby("day")}
    out = points.copy()
    for i, r in out.iterrows():
        if pd.notna(r.get("match_id")):
            continue
        g = by_day.get(pd.Timestamp(r["date"]).normalize())
        if g is None:
            continue
        k1, k2 = _key(r["player_1"]), _key(r["player_2"])
        same = g[((g["ka"].map(lambda k: bool(k & k1))) & (g["kb"].map(lambda k: bool(k & k2))) & (g["sets_a"] == r["sets_1"]) & (g["sets_b"] == r["sets_2"]))]
        flip = g[((g["ka"].map(lambda k: bool(k & k2))) & (g["kb"].map(lambda k: bool(k & k1))) & (g["sets_a"] == r["sets_2"]) & (g["sets_b"] == r["sets_1"]))]
        cand = pd.concat([same.assign(_flip=False), flip.assign(_flip=True)])
        if cand.empty:
            continue
        cand = cand.assign(_dt=(pd.to_datetime(cand["start_time"]) - pd.Timestamp(r["start_time"])).abs()).sort_values("_dt")
        best = cand.iloc[0]
        if best["_dt"] > pd.Timedelta(hours=3):
            continue
        out.at[i, "match_id"] = best["match_id"]
        if best["_flip"]:          # store in our match's A/B order
            out.at[i, "player_1"], out.at[i, "player_2"] = r["player_2"], r["player_1"]
            out.at[i, "sets_1"], out.at[i, "sets_2"] = r["sets_2"], r["sets_1"]
            out.at[i, "points_1"], out.at[i, "points_2"] = r["points_2"], r["points_1"]
            out.at[i, "games"] = json.dumps([(b, a) for a, b in json.loads(r["games"])])
            out.at[i, "first_game_winner"] = 3 - int(r["first_game_winner"])
    return out


# --------------------------------------------------------------------------- jobs
def update(days_back: int = 3, matches: pd.DataFrame | None = None) -> dict:
    global LAST_LOG
    LAST_LOG = []
    leagues = league_ids()
    LAST_LOG.append(f"TT Elite league ids at vendor: {leagues}")
    if not leagues:
        return {"error": "no league matched BETSAPI_LEAGUE_RE", "log": LAST_LOG}
    today = pd.Timestamp.now(tz="Europe/Warsaw").tz_localize(None).normalize()
    new = []
    for d in range(days_back, -1, -1):
        day = (today - pd.Timedelta(days=d)).strftime("%Y%m%d")
        try:
            df = fetch_day(day, leagues)
            LAST_LOG.append(f"{day}: {len(df)} matches with games")
            new.append(df)
        except Exception as ex:
            LAST_LOG.append(f"{day}: {ex}")
    new = pd.concat(new, ignore_index=True) if new else pd.DataFrame(columns=COLS)
    have = store.read(POINTS_CSV)
    allp = pd.concat([have, new], ignore_index=True) if not have.empty else new
    allp = allp.drop_duplicates("event_id", keep="last")
    if matches is None:
        from .pipeline import load_matches
        matches = load_matches()
    allp = link(allp, matches)
    store.write(allp, POINTS_CSV)
    return {"new": int(len(new)), "total": int(len(allp)), "linked": int(allp["match_id"].notna().sum()), "log": LAST_LOG}


def backfill(days: int = 365, matches: pd.DataFrame | None = None, progress=None) -> dict:
    """History, newest day first; safe to re-run (days already on file are skipped)."""
    global LAST_LOG
    LAST_LOG = []
    leagues = league_ids()
    if not leagues:
        return {"error": "no league matched", "log": LAST_LOG}
    have = store.read(POINTS_CSV)
    done_days = set(pd.to_datetime(have["date"]).dt.strftime("%Y%m%d")) if not have.empty else set()
    today = pd.Timestamp.now(tz="Europe/Warsaw").tz_localize(None).normalize()
    frames, pulled = [have] if not have.empty else [], 0
    for d in range(1, days + 1):
        day = (today - pd.Timedelta(days=d)).strftime("%Y%m%d")
        if day in done_days:
            continue
        if progress and d % 10 == 0:
            progress(d / days, day)
        try:
            frames.append(fetch_day(day, leagues)); pulled += 1
        except Exception as ex:
            LAST_LOG.append(f"{day}: {ex}")
            if "limit" in str(ex).lower() or "quota" in str(ex).lower():
                break
    allp = pd.concat(frames, ignore_index=True).drop_duplicates("event_id", keep="last") if frames else pd.DataFrame(columns=COLS)
    if matches is None:
        from .pipeline import load_matches
        matches = load_matches()
    allp = link(allp, matches)
    store.write(allp, POINTS_CSV)
    return {"days_pulled": pulled, "total": int(len(allp)), "linked": int(allp["match_id"].notna().sum()), "log": LAST_LOG[-10:]}


def load_linked() -> pd.DataFrame:
    """Points rows that map to one of our matches, with parsed games."""
    p = store.read(POINTS_CSV)
    if p.empty:
        return p
    p = p[p["match_id"].notna()].copy()
    p["games_list"] = p["games"].map(lambda s: [tuple(g) for g in json.loads(s)])
    p["deuce_w_a"] = p["games_list"].map(lambda gs: sum(1 for g in gs if min(g) >= 10 and g[0] > g[1]))
    p["deuce_w_b"] = p["games_list"].map(lambda gs: sum(1 for g in gs if min(g) >= 10 and g[1] > g[0]))
    return p.rename(columns={"points_1": "points_a", "points_2": "points_b"})[
        ["match_id", "points_a", "points_b", "n_games", "first_game_winner", "deuce_games", "deuce_w_a", "deuce_w_b"]].drop_duplicates("match_id")
