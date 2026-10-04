"""One-off diagnostics run on GitHub Actions (the dev sandbox can't reach these hosts). Output -> data/raw/probe.txt"""
import json, requests, re, sys
from pathlib import Path
out = []
H = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
     "Accept": "application/json, text/html, */*", "Accept-Language": "en-US,en;q=0.9"}
def g(url, **kw):
    try:
        r = requests.get(url, headers=H, timeout=30, **kw)
        return r
    except Exception as ex:
        out.append(f"{url} -> EXC {ex}"); return None
S = "https://www.tt-series.com"
r = g(f"{S}/wp-json/wp/v2/posts?per_page=1")
if r is not None: out.append(f"posts total={r.headers.get('X-WP-Total')} pages={r.headers.get('X-WP-TotalPages')} status={r.status_code}")
r = g(f"{S}/wp-json/wp/v2/posts?per_page=3&after=2022-01-01T00:00:00&before=2023-01-01T00:00:00&_fields=id,date,title")
if r is not None: out.append(f"2022 posts via after/before: {r.status_code} {r.text[:400]}")
r = g(f"{S}/wp-json/wp/v2/posts?per_page=3&orderby=date&order=asc&_fields=id,date,title")
if r is not None: out.append(f"oldest posts: {r.status_code} {r.text[:400]}")
r = g(f"{S}/wp-json/wp/v2/types?_fields=slug,rest_base")
if r is not None: out.append(f"types: {r.status_code} {r.text[:600]}")
for u in (f"{S}/wp-sitemap.xml", f"{S}/sitemap_index.xml", f"{S}/sitemap.xml", f"{S}/post-sitemap.xml"):
    r = g(u)
    if r is not None: out.append(f"{u}: {r.status_code} {len(r.text)} chars; first locs: {re.findall(r'<loc>([^<]+)</loc>', r.text)[:5]}")
r = g(f"{S}/?p=5748")
if r is not None: out.append(f"?p=5748: {r.status_code} {r.url} title={re.search(r'<title>(.*?)</title>', r.text, re.S).group(1)[:80] if '<title>' in r.text else '?'}")
r = g(f"{S}/wp-json/wp/v2/posts/5748?_fields=id,date,title,type,status")
if r is not None: out.append(f"posts/5748: {r.status_code} {r.text[:300]}")
r = g(f"{S}/results-2/")
if r is not None:
    links = re.findall(r'href="(https://www\.tt-series\.com/[^"]*result[^"]*)"', r.text)
    out.append(f"results-2 archive: {r.status_code} {len(links)} result links; sample {links[:3]}; pagination: {re.findall(r'results-2/page/(\d+)', r.text)[:5]}")
# DraftKings
for u in ("https://sportsbook-nash.draftkings.com/api/sportscontent/dkusva/v1/sports",
          "https://sportsbook-nash.draftkings.com/api/sportscontent/dkusnj/v1/sports",
          "https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups?format=json",
          "https://sportsbook-nash.draftkings.com/sites/US-SB/api/v5/eventgroups?format=json",
          "https://sportsbook.draftkings.com/sites/US-SB/api/v4/featured/sports?format=json",
          "https://sportsbook.draftkings.com/leagues/table-tennis/tt-elite-series"):
    r = g(u)
    if r is not None:
        txt = r.text
        hit = [m for m in re.findall(r'"(?:name|displayName)":"([^"]*(?:Table Tennis|TT Elite|Elite Series)[^"]*)"', txt)][:5]
        ids = re.findall(r'"(?:eventGroupId|leagueId|id)":\s*"?(\d+)"?[^}]{0,80}?"(?:name|displayName)":"[^"]*(?:TT Elite|Table Tennis)', txt)[:5]
        out.append(f"DK {u}: {r.status_code} {len(txt)} chars; TT hits {hit}; ids {ids}; head {txt[:150]!r}")
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/raw/probe.txt").write_text("\n".join(out))
print("\n".join(out))
