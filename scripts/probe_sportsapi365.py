"""Does SportsAPI365's table-tennis API carry TT Elite Series with per-game scores? Output -> data/raw/probe_sapi.txt
(the key is never printed)."""
import json, os, re, sys, requests
from pathlib import Path
import pandas as pd
KEY = (os.getenv("SPORTSAPI365_KEY") or "").strip()
BASE = "https://api.sportsapi365.com/v1/table-tennis"
H = {"X-Gravitee-Api-Key": KEY, "Accept": "application/json"}
out, raw = [], Path("data/raw"); raw.mkdir(parents=True, exist_ok=True)
def scrub(t): return t.replace(KEY, "<key>") if KEY else t
def get(path, **params):
    r = requests.get(f"{BASE}/{path.lstrip('/')}", headers=H, params=params, timeout=40)
    tag = re.sub(r"[^A-Za-z0-9]+", "_", f"{path}_{params}")[:80]
    (raw / f"sapi_{tag}.json").write_text(scrub(r.text[:1_500_000]))
    out.append(f"GET /{path} {params} -> {r.status_code} ({len(r.text)} chars)  {scrub(r.text[:300])!r}")
    try:
        return r.json()
    except Exception:
        return None
def walk(o, want):
    if isinstance(o, dict):
        if want <= set(o): yield o
        for v in o.values(): yield from walk(v, want)
    elif isinstance(o, list):
        for v in o: yield from walk(v, want)

out.append(f"key set: {bool(KEY)} len {len(KEY)}")
today = pd.Timestamp.now(tz="Europe/Warsaw").strftime("%Y-%m-%d")
yday = (pd.Timestamp.now(tz="Europe/Warsaw") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
# 1. search / competitions for TT Elite
for path, params in (("search", {"q": "TT Elite"}), ("search", {"query": "TT Elite"}), ("competitions", {}), ("unique-tournaments", {}),
                     ("competitions", {"q": "elite"})):
    d = get(path, **params)
    if d:
        names = [str(x.get("name")) for x in walk(d, {"name"}) if "elite" in str(x.get("name", "")).lower()]
        if names:
            out.append(f"   TT Elite-like names: {names[:10]}")
            ids = [x for x in walk(d, {"name"}) if "elite" in str(x.get("name", "")).lower()]
            out.append("   first object: " + scrub(json.dumps(ids[0])[:600]))
# 2. events by date
for path, params in (("events", {"date": today}), ("events", {"date": yday}), ("fixtures/today", {}), ("events", {"from": yday, "to": today}),
                     ("events/date/" + yday, {})):
    d = get(path, **params)
    if d:
        evs = list(walk(d, {"homeTeam", "awayTeam"})) or list(walk(d, {"home", "away"}))
        tte = [e for e in evs if "elite" in json.dumps(e).lower()]
        out.append(f"   events: {len(evs)}; mentioning 'elite': {len(tte)}")
        if tte:
            e = tte[0]
            out.append("   sample TT Elite event: " + scrub(json.dumps(e)[:1500]))
            eid = e.get("id") or e.get("eventId") or e.get("customId")
            if eid:
                for p2 in (f"events/{eid}", f"event/{eid}", f"events/{eid}/statistics", f"events/{eid}/periods"):
                    d2 = get(p2)
                    if d2:
                        out.append("   detail: " + scrub(json.dumps(d2)[:1500]))
                        break
        elif evs:
            out.append("   sample event: " + scrub(json.dumps(evs[0])[:800]))
            comps = sorted({str((e.get('tournament') or e.get('competition') or e.get('league') or {}).get('name') if isinstance(e.get('tournament') or e.get('competition') or e.get('league'), dict) else '') for e in evs})
            out.append(f"   competitions seen: {comps[:30]}")
Path("data/raw/probe_sapi.txt").write_text("\n".join(out))
print("\n".join(out)[:6000])
