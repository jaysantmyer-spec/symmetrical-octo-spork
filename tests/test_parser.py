import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from tt_oracle.scraper import parse_page, parse_title

# a real page saved by the first live run (April 2022, older title format, surnames in the schedule)
html = (pathlib.Path(__file__).parent / "real_page.html").read_text(encoding="utf-8")
meta, df = parse_page(html, 5748, "Results 23.04.2022 – morning tournament", "https://www.tt-series.com/?p=5748")
assert meta and str(meta["date"].date()) == "2022-04-23" and meta["session"] == "morning", meta
assert len(df) == 15, len(df)
assert df.iloc[0]["player_a"] == "Michał Krzyżanowski" and df.iloc[0]["player_b"] == "Jakub Perek", df.iloc[0]
assert df.iloc[0]["winner"] == "b" and df.iloc[1]["winner"] == "a"
assert df["finished"].all() and str(df.iloc[0]["start_time"]) == "2022-04-23 07:30:00"
# newer title format
t = parse_title("5767. Result 2.10.2026 – afternoon tournament OSP")
assert t and t["seq"] == 5767 and t["session"] == "afternoon" and t["venue"] == "OSP" and str(t["date"].date()) == "2026-10-02"
# synthetic newer-style page with an unplayed match
html2 = (pathlib.Path(__file__).parent / "sample_page.html").read_text()
meta2, df2 = parse_page(html2, 99, "5767. Result 2.10.2026 – afternoon tournament OSP")
assert len(df2) == 4 and not df2.iloc[3]["finished"] and df2.iloc[0]["player_a"] == "Milosz Cesarz", df2
print("parser OK:", len(df), "real matches,", df["player_a"].nunique() + df["player_b"].nunique(), "names")
