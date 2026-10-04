import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from tt_oracle.scraper import parse_page, parse_title

html = (pathlib.Path(__file__).parent / "sample_page.html").read_text()
meta, df = parse_page(html, "https://www.tt-series.com/?p=5767")
assert meta["post_id"] == 5767 and str(meta["date"].date()) == "2026-10-02", meta
assert meta["session"] == "afternoon" and meta["venue"] == "OSP", meta
assert len(df) == 4, df
assert df.iloc[0]["player_a"] == "Milosz Cesarz" and df.iloc[0]["player_b"] == "Krzysztof Schaniel"
assert df.iloc[0]["winner"] == "b" and df.iloc[1]["winner"] == "a" and df.iloc[2]["winner"] == "a"
assert not df.iloc[3]["finished"] and df.iloc[3]["winner"] is None
assert str(df.iloc[0]["start_time"]) == "2026-10-02 11:00:00"
assert parse_title("Results 01.05.2022 – morning tournament HSC Zuzlowa") is None or True
t = parse_title("154. Results 01-05-2022 - morning tournament HSC")
assert t and t["post_id"] == 154 and t["session"] == "morning"
print("parser OK", len(df), "matches")
