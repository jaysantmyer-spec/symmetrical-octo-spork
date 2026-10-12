"""SportsAPI365 table tennis: TT Elite events + per-game scores. Output -> data/raw/probe_sapi.txt (key never printed)."""
import json, os, re, requests, atexit
from pathlib import Path
import pandas as pd
KEY = (os.getenv("SPORTSAPI365_KEY") or "").strip()
BASE = "https://api.sportsapi365.com/v1/table-tennis"
H = {"X-API-Key": KEY, "Accept": "application/json"}
TTE = 57948020
out, raw = [], Path("data/raw"); raw.mkdir(parents=True, exist_ok=True)
atexit.register(lambda: Path("data/raw/probe_sapi.txt").write_text("\n".join(out)))
def scrub(t): return t.replace(KEY, "<key>") if KEY else t
def get(path, **params):
    r = requests.get(f"{BASE}/{path.lstrip('/')}", headers=H, params=params, timeout=40)
    tag = re.sub(r"[^A-Za-z0-9]+", "_", f"{path}_{params}")[:80]
    (raw / f"sapi_{tag}.json").write_text(scrub(r.text[:1_500_000]))
    out.append(f"GET /{path} {params} -> {r.status_code}  {scrub(r.text)[:400]!r}")
    try: return r.json()
    except Exception: return None
yday = (pd.Timestamp.now(tz="Europe/Warsaw") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
d = get(f"unique-tournament/{TTE}")
d = get(f"unique-tournament/{TTE}/seasons")
d = get(f"unique-tournament/{TTE}/scheduled-events/{yday}")
items = (d or {}).get("items") or (d or {}).get("events") or []
meta = {k: v for k, v in (d or {}).items() if k not in ("items", "events")}
out.append(f"   scheduled-events {yday}: {len(items)} items; keys {sorted(items[0].keys()) if items else None}; page info {meta}")
if items:
    out.append("   first: " + scrub(json.dumps(items[0])[:600]))
# pagination on the same endpoint
d2 = get(f"unique-tournament/{TTE}/scheduled-events/{yday}", page=2)
d3 = get(f"unique-tournament/{TTE}/scheduled-events/{yday}", limit=100)
items3 = (d3 or {}).get("items") or []
out.append(f"   with limit=100: {len(items3)} items")
# event detail for a finished TT Elite event
fin = [e for e in (items3 or items) if str(e.get("status")).lower() in ("finished", "ended")]
if fin:
    eid = fin[0]["id"]
    for p in (f"event/{eid}", f"event/{eid}/incidents", f"event/{eid}/statistics", f"event/{eid}/enrichments"):
        dd = get(p)
        if dd: out.append(f"   {p}: " + scrub(json.dumps(dd)[:2500]))
# tournament events (all)
d4 = get(f"unique-tournament/{TTE}/events")
d5 = get(f"unique-tournament/{TTE}/events", page=2)
out.append(f"   events endpoint keys: {list((d4 or {}).keys()) if isinstance(d4, dict) else type(d4)}; n items {len((d4 or {}).get('items', []))}")
# older date for history depth
old = "2025-10-12"
d6 = get(f"unique-tournament/{TTE}/scheduled-events/{old}", limit=100)
out.append(f"   {old}: {len((d6 or {}).get('items', []))} items")
