"""Fetch tt-series.com's H2H page, find its form, submit one lookup, and save everything to data/raw/h2h_*.html
plus a summary in data/raw/probe_h2h.txt -- to see whether per-game point scores exist anywhere on the site."""
import re, requests
from pathlib import Path
from urllib.parse import urljoin
H = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
     "Accept": "text/html,*/*", "Accept-Language": "en-US,en;q=0.9"}
Path("data/raw").mkdir(parents=True, exist_ok=True)
out = []
s = requests.Session(); s.headers.update(H)
r = s.get("https://www.tt-series.com/h2h/", timeout=60); html = r.text
Path("data/raw/h2h_page.html").write_text(html)
out.append(f"GET /h2h/: {r.status_code} {len(html)} chars")
forms = re.findall(r"<form[^>]*>.*?</form>", html, re.S | re.I)
out.append(f"forms: {len(forms)}")
for f in forms[:3]:
    out.append("FORM head: " + re.sub(r"\s+", " ", f[:1500]))
    out.append("  selects: " + str(re.findall(r"<select[^>]*name=[\"']([^\"']+)", f, re.I)))
    out.append("  inputs: " + str(re.findall(r"<input[^>]*name=[\"']([^\"']+)[^>]*>", f, re.I)[:10]))
    out.append("  first options: " + str(re.findall(r"<option[^>]*value=[\"']([^\"']*)[\"'][^>]*>([^<]*)", f, re.I)[:5]))
# scripts that might call an endpoint (AJAX)
for m in re.finditer(r"(admin-ajax\.php|wp-json/[\w/.-]+|ajax[\w/.-]*\.php|/api/[\w/.-]*)", html):
    out.append("endpoint mention: " + html[max(0, m.start()-150): m.end()+150].replace("\n", " "))
# try a lookup with the first form
if forms:
    f = forms[0]
    action = re.search(r"action=[\"']([^\"']*)", f, re.I); method = re.search(r"method=[\"']([^\"']*)", f, re.I)
    action = urljoin("https://www.tt-series.com/h2h/", action.group(1) if action and action.group(1) else "")
    method = (method.group(1) if method else "get").lower()
    sels = re.findall(r"<select[^>]*name=[\"']([^\"']+)[\"'][^>]*>(.*?)</select>", f, re.S | re.I)
    data = {}
    for name, body in sels:
        opts = [v for v in re.findall(r"<option[^>]*value=[\"']([^\"']*)", body, re.I) if v.strip()]
        data[name] = "David Petr" if name == sels[0][0] else "Petas Kacper"
    for name, val in re.findall(r"<input[^>]*name=[\"']([^\"']+)[\"'][^>]*value=[\"']([^\"']*)", f, re.I):
        data.setdefault(name, val)
    out.append(f"submit {method.upper()} {action} {data}")
    r2 = s.post(action, data=data, timeout=60) if method == "post" else s.get(action, params=data, timeout=60)
    Path("data/raw/h2h_result.html").write_text(r2.text)
    txt = re.sub(r"<[^>]+>", " ", r2.text); txt = re.sub(r"\s+", " ", txt)
    PT = re.compile(r"\b1[0-9]:[0-9]{1,2}\b|\b[0-9]{1,2}:1[0-9]\b")
    npts = len(PT.findall(txt))
    out.append(f"result: {r2.status_code} {len(r2.text)} chars; point-like scores (e.g. 11:9) found: {npts}")
    # the part of the page after the player list: that's where results would render
    i = txt.rfind("Show Matches")
    out.append("text after the form: " + txt[i: i + 4000])
    tables = re.findall(r"<table[^>]*>.*?</table>", r2.text, re.S | re.I)
    out.append(f"tables in result: {len(tables)}")
    for t in tables[:3]:
        out.append("TABLE: " + re.sub(r"\s+", " ", t[:2500]))
Path("data/raw/probe_h2h.txt").write_text("\n".join(out))
print("\n".join(out)[:4000])
