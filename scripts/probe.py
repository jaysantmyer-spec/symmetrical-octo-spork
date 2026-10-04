"""Diagnostics run on GitHub Actions. Output -> data/raw/probe.txt"""
import re, json, requests
from pathlib import Path
out = []
H = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
     "Accept": "text/html,application/json,*/*", "Accept-Language": "en-US,en;q=0.9"}
def g(url, **kw):
    try:
        return requests.get(url, headers=H, timeout=40, **kw)
    except Exception as ex:
        out.append(f"{url} -> EXC {ex}"); return None
def ctx(txt, needle, n=3, w=250):
    return [txt[max(0, m.start()-w): m.start()+w].replace("\n", " ") for m in list(re.finditer(re.escape(needle), txt))[:n]]
for u in ("https://sportsbook.draftkings.com/leagues/table-tennis", "https://sportsbook.draftkings.com/sports/table-tennis",
          "https://sportsbook.draftkings.com/sites/US-SB/api/v4/featured/sports?format=json"):
    r = g(u)
    if r is None: continue
    t = r.text
    out.append(f"{u}: {r.status_code} {len(t)} chars")
    for c in ctx(t, "TT Elite Series"): out.append("  CTX: " + c)
    for c in ctx(t, "Table Tennis", 2): out.append("  CTX2: " + c)
    ids = set(re.findall(r'"(?:eventGroupId|leagueId|id)"\s*:\s*"?(\d{3,9})"?\s*,\s*"(?:name|displayName|eventGroupName|leagueName)"\s*:\s*"(?:TT Elite Series|Table Tennis)"', t))
    ids |= set(re.findall(r'"(?:name|displayName|eventGroupName|leagueName)"\s*:\s*"(?:TT Elite Series|Table Tennis)"\s*,\s*"(?:eventGroupId|leagueId|id)"\s*:\s*"?(\d{3,9})"?', t))
    out.append(f"  candidate ids: {sorted(ids)}")
    for i in sorted(ids)[:6]:
        for base in ("https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/{i}?format=json",
                     "https://sportsbook.draftkings.com/sites/US-SB/api/v6/eventgroups/{i}?format=json"):
            rr = g(base.format(i=i))
            if rr is not None:
                out.append(f"  try {base.format(i=i)}: {rr.status_code} {rr.headers.get('content-type','')[:30]} {rr.text[:160]!r}")
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/raw/probe.txt").write_text("\n".join(out))
print("\n".join(out))
