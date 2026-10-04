"""Diagnostics run on GitHub Actions. Output -> data/raw/probe.txt (+ dk_league.html)"""
import re, requests
from pathlib import Path
out = []
H = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
     "Accept": "text/html,*/*", "Accept-Language": "en-US,en;q=0.9"}
Path("data/raw").mkdir(parents=True, exist_ok=True)
for u in ("https://sportsbook.draftkings.com/leagues/tabletennis/tt-elite-series",
          "https://sportsbook.draftkings.com/leagues/table-tennis/tt-elite-series?category=game-lines"):
    try:
        r = requests.get(u, headers=H, timeout=40); t = r.text
        out.append(f"{u}: {r.status_code} {len(t)} chars")
        Path("data/raw/dk_league.html").write_text(t)
        for needle in ('"outcomes"', '"oddsAmerican"', '"oddsDecimal"', '"startDate"', '"eventGroupId":208037', '"offers"', '"events"', '"participants"', '"selections"'):
            hits = [m.start() for m in re.finditer(re.escape(needle), t)]
            out.append(f"  {needle}: {len(hits)} hits; first ctx: {t[max(0, hits[0]-300): hits[0]+500] if hits else ''!r}")
        break
    except Exception as ex:
        out.append(f"{u} -> EXC {ex}")
Path("data/raw/probe.txt").write_text("\n".join(out))
print("\n".join(out)[:3000])
