"""Does SportsAPI365's table-tennis API carry TT Elite Series with per-game scores? Output -> data/raw/probe_sapi.txt
(the key is never printed)."""
import json, os, re, sys, requests
from pathlib import Path
import pandas as pd
KEY = (os.getenv("SPORTSAPI365_KEY") or "").strip()
BASE = "https://api.sportsapi365.com/v1/table-tennis"
H = {"X-API-Key": KEY, "Accept": "application/json"}
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

import atexit
atexit.register(lambda: Path("data/raw/probe_sapi.txt").write_text("\n".join(out)))
out.append(f"key set: {bool(KEY)} len {len(KEY)}")
# auth variants on one cheap endpoint
for label, hdr, params, base in (("X-Gravitee-Api-Key", {"X-Gravitee-Api-Key": KEY}, {}, BASE),
                                 ("X-API-Key", {"X-API-Key": KEY}, {}, BASE),
                                 ("x-api-key", {"x-api-key": KEY}, {}, BASE),
                                 ("Authorization Bearer", {"Authorization": f"Bearer {KEY}"}, {}, BASE),
                                 ("Authorization ApiKey", {"Authorization": f"ApiKey {KEY}"}, {}, BASE),
                                 ("query api-key", {}, {"api-key": KEY}, BASE),
                                 ("query apiKey", {}, {"apiKey": KEY}, BASE),
                                 ("query api_key", {}, {"api_key": KEY}, BASE),
                                 ("gravitee + base /table-tennis/v1", {"X-Gravitee-Api-Key": KEY}, {}, "https://api.sportsapi365.com/table-tennis/v1"),
                                 ("gravitee + base /v1 (no sport)", {"X-Gravitee-Api-Key": KEY}, {}, "https://api.sportsapi365.com/v1"),
                                 ("gravitee + tennis (is the key for another sport?)", {"X-Gravitee-Api-Key": KEY}, {}, "https://api.sportsapi365.com/v1/tennis")):
    try:
        r = requests.get(f"{base}/fixtures/today", headers={**hdr, "Accept": "application/json"}, params=params, timeout=30)
        out.append(f"auth {label:48} -> {r.status_code} {scrub(r.text[:160])!r}")
    except Exception as ex:
        out.append(f"auth {label:48} -> EXC {ex}")
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
        evs = d.get("items") if isinstance(d, dict) and isinstance(d.get("items"), list) else (list(walk(d, {"homeTeam", "awayTeam"})) or list(walk(d, {"home", "away"})))
        if evs:
            out.append("   event keys: " + str(sorted(evs[0].keys()))[:600])
            tn = sorted({str((e.get("tournament") or {}).get("name") or (e.get("uniqueTournament") or {}).get("name") or e.get("tournamentName") or e.get("tournamentId")) for e in evs})
            out.append(f"   tournaments in this list ({len(tn)}): {tn[:40]}")
        tte = [e for e in evs if "elite" in json.dumps(e).lower()]
        out.append(f"   events: {len(evs)}; mentioning 'elite': {len(tte)}")
        if tte:
            e = tte[0]
            out.append("   sample TT Elite event: " + scrub(json.dumps(e)[:1500]))
            eid = e.get("id") or e.get("eventId") or e.get("customId")
            if eid:
                for p2 in (f"events/{eid}", f"event/{eid}", f"events/{eid}/details", f"events/{eid}/statistics", f"events/{eid}/periods", f"matches/{eid}", f"match/{eid}"):
                    d2 = get(p2)
                    if d2:
                        out.append("   detail: " + scrub(json.dumps(d2)[:1500]))
                        break
        elif evs:
            out.append("   sample event: " + scrub(json.dumps(evs[0])[:800]))
            comps = sorted({str((e.get('tournament') or e.get('competition') or e.get('league') or {}).get('name') if isinstance(e.get('tournament') or e.get('competition') or e.get('league'), dict) else '') for e in evs})
            out.append(f"   competitions seen: {comps[:30]}")
# resolve tournament names for the ids seen today
d = get("fixtures/today")
items = d.get("items", []) if isinstance(d, dict) else []
tids = sorted({e.get("tournamentId") for e in items if e.get("tournamentId")})[:40]
out.append(f"tournament ids today: {tids}")
for tid in tids[:25]:
    for path in (f"tournaments/{tid}", f"tournament/{tid}", f"competitions/{tid}", f"unique-tournaments/{tid}"):
        dd = get(path)
        if isinstance(dd, dict) and dd.get("http_status_code") not in (404,) and "message" not in dd:
            nm = dd.get("name") or (dd.get("tournament") or {}).get("name") or (dd.get("uniqueTournament") or {}).get("name") or json.dumps(dd)[:200]
            out.append(f"   tournament {tid}: {nm}")
            break
# per-game scores: take any ended event and dump it fully
def _st(e):
    st = e.get("status"); return str(st.get("type") or st.get("description") or st) if isinstance(st, dict) else str(st)
ended = [e for e in items if _st(e).lower() in ("finished", "ended", "ft")] or items[:1]
if ended:
    out.append("ENDED/sample event full: " + scrub(json.dumps(ended[0])[:3000]))
Path("data/raw/probe_sapi.txt").write_text("\n".join(out))
print("\n".join(out)[:6000])
