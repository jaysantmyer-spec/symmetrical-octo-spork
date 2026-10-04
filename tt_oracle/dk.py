"""
DraftKings lines for TT Elite Series, and the pricing desk.

DraftKings publishes its sportsbook data as JSON that the website itself reads. The exact endpoint and
the table-tennis group id change over time, so this module tries a list of known shapes, keeps whatever
answers, and can be pinned with environment variables once the right one is known:
    DK_EVENT_GROUP_ID   (v5 "eventgroups" API)
    DK_LEAGUE_ID        (newer "sportscontent" API)
The refresh workflow saves the raw responses to data/raw/dk_*.json so the parser can be adjusted.

If no feed works, the app's Pricing desk accepts pasted lines ("Player A -150 / Player B +120"), which
are stored exactly like a fetched snapshot.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata

import numpy as np
import pandas as pd
import requests

from . import config, store

TT_ELITE_EVENT_GROUP = "208037"   # DraftKings eventGroupId for TT Elite Series (found in their page data, Oct 2026)
V5 = "https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/{gid}?format=json"
V5_NASH = "https://sportsbook-nash.draftkings.com/sites/US-SB/api/v5/eventgroups/{gid}?format=json"
V5_LIST = "https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups?format=json"
NASH = "https://sportsbook-nash.draftkings.com/api/sportscontent/dkusva/v1/leagues/{lid}"
NASH_SPORTS = "https://sportsbook-nash.draftkings.com/api/sportscontent/dkusva/v1/sports"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/126.0 Safari/537.36", "Accept": "application/json", "Accept-Language": "en-US,en;q=0.9"}


# --------------------------------------------------------------------------- odds maths
def _key(name) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", s)


def _key_set(name) -> frozenset:
    """Order-insensitive token key so 'Jadach O.' ~ 'Oskar Jadach' ~ 'Jadach, Oskar'."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    toks = [t for t in re.split(r"[^a-z]+", s) if len(t) >= 2]
    return frozenset(toks)


def _same(k1: frozenset, k2: frozenset) -> bool:
    """'Jadach O.' matches 'Oskar Jadach' (initials dropped, surname shared); unrelated names don't."""
    if not k1 or not k2:
        return False
    return k1 <= k2 or k2 <= k1


def american_to_decimal(a) -> float:
    a = float(str(a).replace("+", "").replace("−", "-"))
    return 1 + a / 100 if a > 0 else 1 + 100 / abs(a)


def decimal_to_american(d: float) -> str:
    d = float(d)
    if d <= 1:
        return "–"
    return f"+{(d - 1) * 100:.0f}" if d >= 2 else f"-{100 / (d - 1):.0f}"


def fair_american(p: float) -> str:
    p = float(min(max(p, 1e-6), 1 - 1e-6))
    return f"-{100 * p / (1 - p):.0f}" if p >= 0.5 else f"+{100 * (1 - p) / p:.0f}"


def kelly(p: float, d: float, fraction: float = 0.25) -> float:
    b = d - 1
    if b <= 0:
        return 0.0
    return max(0.0, (p * b - (1 - p)) / b * fraction)


LAST_FETCH_LOG: list[str] = []


# --------------------------------------------------------------------------- fetching
def _get_json(url: str, timeout: int = 20):
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _parse_v5(payload: dict) -> list[dict]:
    rows = []
    eg = payload.get("eventGroup") or payload
    events = {e.get("eventId"): e for e in eg.get("events", [])}
    for cat in eg.get("offerCategories", []):
        for sub in cat.get("offerSubcategoryDescriptors", []):
            offers = (sub.get("offerSubcategory") or {}).get("offers", [])
            for group in offers:
                for offer in group:
                    if offer.get("isSuspended"):
                        continue
                    outs = offer.get("outcomes", [])
                    if len(outs) != 2:
                        continue
                    label = (offer.get("label") or "").lower()
                    if label and "moneyline" not in label and "winner" not in label:
                        continue
                    ev = events.get(offer.get("eventId"), {})
                    rows.append({"event_id": offer.get("eventId"), "commence_time": ev.get("startDate"),
                                 "player_1": outs[0].get("label"), "odds_1": outs[0].get("oddsDecimal"),
                                 "player_2": outs[1].get("label"), "odds_2": outs[1].get("oddsDecimal"),
                                 "league": eg.get("name")})
    return rows


def _parse_nash(payload: dict) -> list[dict]:
    rows = []
    events = {e.get("id"): e for e in payload.get("events", [])}
    markets = {m.get("id"): m for m in payload.get("markets", [])}
    by_market: dict = {}
    for s in payload.get("selections", []):
        by_market.setdefault(s.get("marketId"), []).append(s)
    for mid, sels in by_market.items():
        mk = markets.get(mid, {})
        name = (mk.get("name") or "").lower()
        if name and "moneyline" not in name and "winner" not in name and "match" not in name:
            continue
        if len(sels) != 2:
            continue
        ev = events.get(mk.get("eventId"), {})
        d = [s.get("displayOdds", {}).get("decimal") or s.get("trueOdds") for s in sels]
        rows.append({"event_id": mk.get("eventId"), "commence_time": ev.get("startEventDate"),
                     "player_1": sels[0].get("label") or sels[0].get("participants", [{}])[0].get("name"),
                     "odds_1": d[0], "player_2": sels[1].get("label") or sels[1].get("participants", [{}])[0].get("name"),
                     "odds_2": d[1], "league": ev.get("leagueName") or ev.get("name")})
    return rows


def fetch_dk(save_raw: bool = True) -> pd.DataFrame:
    """Try the configured / known endpoints; return a tidy table of moneylines (decimal odds)."""
    config.ensure_dirs()
    gid = os.getenv("DK_EVENT_GROUP_ID", TT_ELITE_EVENT_GROUP)
    lid = os.getenv("DK_LEAGUE_ID", TT_ELITE_EVENT_GROUP)
    attempts = [("v5", V5.format(gid=gid), _parse_v5), ("v5nash", V5_NASH.format(gid=gid), _parse_v5),
                ("nash", NASH.format(lid=lid), _parse_nash),
                ("nash-nj", NASH.replace("dkusva", "dkusnj").format(lid=lid), _parse_nash)]
    rows, log_lines = [], []
    for kind, url, parser in attempts:
        if parser is None:
            log_lines.append(url)
            continue
        try:
            payload = _get_json(url)
            if save_raw:
                (config.RAW / f"dk_{kind}_{abs(hash(url)) % 10_000}.json").write_text(json.dumps(payload)[:800000])
            got = parser(payload)
            log_lines.append(f"{kind} {url}: {len(got)} lines")
            rows += got
        except Exception as ex:
            log_lines.append(f"{kind} {url}: {ex}")
    (config.RAW / "dk_fetch_log.txt").write_text("\n".join(log_lines))
    global LAST_FETCH_LOG
    LAST_FETCH_LOG = log_lines
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["league"].astype(str).str.contains("Elite|TT", case=False, na=True)] if "league" in df else df
    df["odds_1"] = pd.to_numeric(df["odds_1"], errors="coerce")
    df["odds_2"] = pd.to_numeric(df["odds_2"], errors="coerce")
    df = df.dropna(subset=["odds_1", "odds_2"])
    df["bookmaker"] = "DraftKings"
    df["fetched_at"] = pd.Timestamp.utcnow()
    return df.drop_duplicates(["player_1", "player_2", "commence_time"])


def snapshot_odds(df: pd.DataFrame | None = None) -> dict:
    df = fetch_dk() if df is None else df
    if df.empty:
        return {"lines": 0}
    store.write(df, config.ODDS_CSV)
    hist = store.read(config.ODDS_HISTORY_CSV)
    hist = pd.concat([hist, df], ignore_index=True) if not hist.empty else df
    store.write(hist, config.ODDS_HISTORY_CSV)
    return {"lines": int(len(df))}


def parse_pasted(text: str) -> pd.DataFrame:
    """'Oskar Jadach -150 / Wojciech Urban +120' per line (also accepts 'vs' and decimal odds)."""
    rows = []
    for line in (text or "").splitlines():
        parts = re.split(r"\s*(?:/|vs\.?|,)\s*", line.strip(), maxsplit=1)
        if len(parts) != 2:
            continue
        m1 = re.match(r"(.+?)\s+([+-−]?\d+(?:\.\d+)?)\s*$", parts[0])
        m2 = re.match(r"(.+?)\s+([+-−]?\d+(?:\.\d+)?)\s*$", parts[1])
        if not (m1 and m2):
            continue
        def dec(s):
            v = float(s.replace("−", "-").replace("+", ""))
            return v if 1.01 <= v <= 20 and "+" not in s and "-" not in s else american_to_decimal(s)
        rows.append({"event_id": None, "commence_time": None, "player_1": m1.group(1).strip(), "odds_1": dec(m1.group(2)),
                     "player_2": m2.group(1).strip(), "odds_2": dec(m2.group(2)), "league": "TT Elite Series (pasted)",
                     "bookmaker": "DraftKings", "fetched_at": pd.Timestamp.utcnow()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- matching and pricing
def attach_market(P: pd.DataFrame, odds: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add DraftKings decimal odds and vig-free market probability for each predicted match."""
    odds = store.read(config.ODDS_CSV) if odds is None else odds
    out = P.copy()
    out["dk_odds_a"], out["dk_odds_b"], out["market_p_a"] = np.nan, np.nan, np.nan
    if odds is None or odds.empty or out.empty:
        return out
    idx = {}
    for _, r in odds.iterrows():
        idx[(_key_set(r["player_1"]), _key_set(r["player_2"]))] = (float(r["odds_1"]), float(r["odds_2"]))
    for i, r in out.iterrows():
        ka, kb = _key_set(r["player_a"]), _key_set(r["player_b"])
        hit = None
        for (k1, k2), (o1, o2) in idx.items():
            if _same(ka, k1) and _same(kb, k2):
                hit = (o1, o2); break
            if _same(ka, k2) and _same(kb, k1):
                hit = (o2, o1); break
        if hit:
            oa, ob = hit
            ia, ib = 1 / oa, 1 / ob
            out.at[i, "dk_odds_a"], out.at[i, "dk_odds_b"], out.at[i, "market_p_a"] = oa, ob, ia / (ia + ib)
    out["edge_a"] = out["p_a"] - out["market_p_a"]
    return out


def price(P: pd.DataFrame, bankroll: float = 1000.0, kelly_fraction: float = 0.25, edge_threshold: float = 0.04) -> pd.DataFrame:
    rows = []
    for _, r in P.iterrows():
        for side, other in (("a", "b"), ("b", "a")):
            p = float(r[f"p_{side}"])
            d = r.get(f"dk_odds_{side}")
            mp = r.get("market_p_a")
            mp = mp if side == "a" else (1 - mp if pd.notna(mp) else np.nan)
            edge = p - mp if pd.notna(mp) else np.nan
            ev = p * (d - 1) - (1 - p) if pd.notna(d) else np.nan
            k = kelly(p, d, kelly_fraction) if pd.notna(d) else 0.0
            rows.append({"match_id": r["match_id"], "start_time": r["start_time"], "player": r[f"player_{side}"],
                         "opponent": r[f"player_{other}"], "p_model": p, "fair": fair_american(p),
                         "p_market": mp, "market": fair_american(mp) if pd.notna(mp) else "–",
                         "dk": decimal_to_american(d) if pd.notna(d) else "–", "dk_dec": d,
                         "edge": edge, "ev": ev, "kelly": k, "stake": round(k * bankroll),
                         "thin": bool(r.get("thin_history", False)),
                         "bet": bool(pd.notna(edge) and edge >= edge_threshold and ev > 0 and not r.get("thin_history", False))})
    return pd.DataFrame(rows)
