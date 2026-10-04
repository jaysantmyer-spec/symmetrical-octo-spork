"""
TT Elite Oracle - Streamlit front end.   Run:  streamlit run app.py

The app reads what the GitHub Actions refresh commits to data/: matches, predictions, odds, ledger and
backtest. It never scrapes or retrains on its own, so it stays fast on Streamlit Cloud.
"""
from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd
import streamlit as st

from tt_oracle import config, dk, pipeline, store
from tt_oracle.features import player_snapshot

st.set_page_config(page_title="TT Elite Oracle", page_icon="🏓", layout="wide")
config.ensure_dirs()

ORANGE, NAVY, GOLD = "#D9601A", "#1E3A5F", "#E3B341"
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&display=swap');
html, body, [class*="css"] {{ font-family: 'Barlow', system-ui, sans-serif; }}
h1, h2, h3 {{ font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; }}
.m {{ border: 1px solid #D5DAE1; border-radius: 10px; padding: 12px 16px 10px; margin: 8px 0; background: #fff; }}
.m.hi {{ border: 2px solid {GOLD}; background: #FFFDF5; }}
.m .meta {{ display:flex; justify-content:space-between; color:#5B6470; font-size:.82rem; }}
.m .names {{ display:flex; justify-content:space-between; gap:12px; font-family:'Barlow Condensed',sans-serif; font-size:1.35rem; font-weight:600; }}
.m .a {{ color:{ORANGE}; }} .m .b {{ color:{NAVY}; text-align:right; }}
.m .split {{ display:flex; height:12px; border-radius:6px; overflow:hidden; margin:6px 0 2px; }}
.m .split .ra {{ background:{ORANGE}; }} .m .split .rb {{ background:{NAVY}; }}
.m .pcts {{ display:flex; justify-content:space-between; font-family:'Barlow Condensed',sans-serif; font-size:1.2rem; font-weight:700; }}
.m .line {{ font-size:.9rem; margin-top:4px; }}
.badge {{ font-size:.7rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase; padding:2px 7px; border-radius:4px; background:{GOLD}; color:#1B1D20; }}
.warn {{ background:#F3E6E6; color:#8B2E2E; }}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load(sig):
    return {"pred": store.read(config.PREDICTIONS_CSV), "matches": pipeline.load_matches(),
            "odds": store.read(config.ODDS_CSV), "ledger": store.read(config.LEDGER_CSV),
            "bt": store.read(config.BACKTEST_CSV), "tours": store.read(config.TOURNAMENTS_CSV)}


def _sig():
    return tuple(p.stat().st_mtime if p.exists() else 0 for p in
                 (config.PREDICTIONS_CSV, config.MATCHES_CSV, config.ODDS_CSV, config.LEDGER_CSV, config.BACKTEST_CSV))


D = load(_sig())
pred, matches, odds, ledger, bt = D["pred"], D["matches"], D["odds"], D["ledger"], D["bt"]
state = json.loads(config.STATE_FILE.read_text()) if config.STATE_FILE.exists() else {}


def card_html(r) -> str:
    e = html.escape
    pa = float(r["p_a"])
    hi = r["confidence"] in ("Strong", "Solid") and not bool(r.get("thin_history", False))
    badge = f'<span class="badge">{e(str(r["confidence"]))} pick</span>' if hi else ""
    thin = '<span class="badge warn">thin history</span>' if bool(r.get("thin_history", False)) else ""
    t = pd.Timestamp(r["start_time"])
    mk = ""
    if pd.notna(r.get("market_p_a")):
        edge = pa - r["market_p_a"]
        side = r["player_a"] if edge > 0 else r["player_b"]
        mk = (f'<div class="line">DraftKings: {e(str(r["player_a"]))} {dk.decimal_to_american(r["dk_odds_a"])} / '
              f'{e(str(r["player_b"]))} {dk.decimal_to_american(r["dk_odds_b"])} · market {r["market_p_a"]:.0%} · '
              f'model sees {abs(edge):.0%} more on <b>{e(str(side))}</b></div>')
    return f"""<div class="m{' hi' if hi else ''}"><div class="meta"><span>{t:%a %d %b %H:%M} · {e(str(r['session']))} · {e(str(r['venue']))}</span><span>{thin} {badge}</span></div>
<div class="names"><div class="a">{e(str(r['player_a']))}</div><div class="b">{e(str(r['player_b']))}</div></div>
<div class="split"><div class="ra" style="width:{pa*100:.1f}%"></div><div class="rb" style="width:{(1-pa)*100:.1f}%"></div></div>
<div class="pcts"><span style="color:{ORANGE}">{pa:.0%}</span><span style="color:{NAVY}">{1-pa:.0%}</span></div>
<div class="line"><b>{e(str(r['pick']))}</b> ({str(r['confidence']).lower()}). Elo alone: {r['elo_p_a']:.0%}. H2H matches: {int(r['h2h_n'])}. Today so far: {int(r['A_today_n'])} / {int(r['B_today_n'])}.</div>{mk}</div>"""


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("TT Elite Oracle")
    if matches.empty:
        st.warning("No data yet. The first GitHub Actions backfill hasn't run (see DEPLOY.md).")
    else:
        fin = matches[matches["finished"]]
        st.write(f"**{len(fin):,}** matches, **{matches['player_a'].nunique():,}** players")
        st.write(f"Results through **{fin['start_time'].max():%d %b %Y %H:%M}**")
        st.write(f"**{int((~matches['finished']).sum())}** upcoming on file")
    if config.STATE_FILE.exists():
        st.caption(f"Learners: {', '.join(state.get('learners') or [])}")
    if not odds.empty:
        st.caption(f"DraftKings lines: {len(odds)} ({pd.to_datetime(odds['fetched_at']).max():%d %b %H:%M} UTC)")
    st.divider()
    st.caption("Data refreshes every 6 hours via GitHub Actions. Predictions are logged before each match and graded after.")

tab_today, tab_price, tab_h2h, tab_track, tab_data = st.tabs(["Today", "Pricing desk", "Head to head", "Track record", "Data"])

# --------------------------------------------------------------------------- today
with tab_today:
    if pred.empty:
        st.info("No predictions yet. Once the refresh workflow has run with data, upcoming matches appear here.")
    else:
        P = pred.copy()
        P["start_time"] = pd.to_datetime(P["start_time"])
        P = P.sort_values("start_time")
        c1, c2, c3 = st.columns(3)
        c1.metric("Upcoming matches", len(P))
        hi = P[P["confidence"].isin(["Strong", "Solid"]) & ~P["thin_history"].astype(bool)]
        c2.metric("Strong / solid picks", len(hi))
        c3.metric("With DraftKings line", int(P["market_p_a"].notna().sum()) if "market_p_a" in P else 0)
        day = st.selectbox("Day", sorted(P["start_time"].dt.date.unique()))
        only_hi = st.checkbox("Only strong / solid picks", value=False)
        view = P[P["start_time"].dt.date == day]
        if only_hi:
            view = view[view["confidence"].isin(["Strong", "Solid"]) & ~view["thin_history"].astype(bool)]
        for sess in config.SESSIONS:
            vs = view[view["session"] == sess]
            if vs.empty:
                continue
            st.subheader(f"{sess.title()} session · {len(vs)} matches")
            for _, r in vs.iterrows():
                st.markdown(card_html(r), unsafe_allow_html=True)
        st.download_button("Download predictions (CSV)", P.to_csv(index=False), file_name="tt_predictions.csv")

# --------------------------------------------------------------------------- pricing desk
with tab_price:
    st.write("Model probability as a fair price vs DraftKings. A bet is flagged when the edge beats your threshold, "
             "EV is positive, and both players have enough history.")
    c1, c2, c3 = st.columns(3)
    bankroll = c1.number_input("Bankroll ($)", 50.0, 1_000_000.0, 500.0, step=50.0)
    kf = c2.select_slider("Kelly fraction", options=[0.1, 0.25, 0.5], value=0.25,
                          format_func=lambda v: {0.1: "1/10", 0.25: "Quarter", 0.5: "Half"}[v])
    thr = c3.slider("Edge threshold", 0.0, 0.15, 0.05, 0.01, format="%.2f")
    with st.expander("Paste DraftKings lines (if the automatic feed has nothing)"):
        st.caption("One match per line: `Oskar Jadach -150 / Wojciech Urban +120`. Names can be last name + initial.")
        txt = st.text_area("Lines", height=120, key="paste_odds")
        if st.button("Use these lines"):
            o = dk.parse_pasted(txt)
            if o.empty:
                st.error("Couldn't read any lines.")
            else:
                dk.snapshot_odds(o)
                st.success(f"Saved {len(o)} lines. Re-pricing below.")
                st.cache_data.clear()
                odds = store.read(config.ODDS_CSV)
    if pred.empty:
        st.info("No predictions on file.")
    else:
        P = dk.attach_market(pred, odds) if not odds.empty else pred
        priced = dk.price(P, bankroll, kf, thr)
        flagged = priced[priced["bet"]].sort_values("edge", ascending=False)
        st.subheader(f"Flagged bets: {len(flagged)}")
        fmt = {"p_model": "{:.0%}", "p_market": "{:.0%}", "edge": "{:+.1%}", "ev": "{:+.2f}", "stake": "${:,.0f}"}
        if not flagged.empty:
            st.dataframe(flagged[["start_time", "player", "opponent", "p_model", "fair", "dk", "p_market", "edge", "ev", "stake"]]
                         .style.format(fmt), hide_index=True)
        else:
            st.caption("Nothing clears the bar right now" + ("" if not odds.empty else " — no DraftKings lines loaded."))
        st.subheader("Every side")
        st.dataframe(priced[["start_time", "player", "opponent", "p_model", "fair", "dk", "market", "edge", "ev", "stake", "thin", "bet"]]
                     .style.format(fmt), hide_index=True)

# --------------------------------------------------------------------------- head to head
with tab_h2h:
    if matches.empty:
        st.info("Needs data.")
    else:
        feat = pipeline.features_now()
        snap = player_snapshot(feat)
        names = snap["player"].tolist()
        c1, c2 = st.columns(2)
        a = c1.selectbox("Player A", names, index=0)
        b = c2.selectbox("Player B", names, index=1 if len(names) > 1 else 0)
        fin = matches[matches["finished"]]
        h = fin[((fin["player_a"] == a) & (fin["player_b"] == b)) | ((fin["player_a"] == b) & (fin["player_b"] == a))].sort_values("start_time", ascending=False)
        aw = int(((h["player_a"] == a) & (h["winner"] == "a")).sum() + ((h["player_b"] == a) & (h["winner"] == "b")).sum())
        sa, sb = snap.set_index("player").loc[a], snap.set_index("player").loc[b]
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{a} Elo", f"{sa['elo']:.0f}", f"last 10: {sa['win10']:.0%}")
        c2.metric("Head to head", f"{aw}–{len(h) - aw}")
        c3.metric(f"{b} Elo", f"{sb['elo']:.0f}", f"last 10: {sb['win10']:.0%}")
        if not h.empty:
            st.dataframe(h[["start_time", "session", "venue", "player_a", "sets_a", "sets_b", "player_b"]], hide_index=True)
        st.subheader("Ratings leaderboard")
        st.dataframe(snap.head(40).rename(columns={"n": "matches"}).style.format({"elo": "{:.0f}", "win10": "{:.0%}"}), hide_index=True)

# --------------------------------------------------------------------------- track record
with tab_track:
    st.write("Two views: the **ledger** (real predictions logged before each match, then graded) and the "
             "**walk-forward backtest** (retrained weekly on earlier matches only).")
    g = ledger[ledger["y"].notna()] if not ledger.empty else pd.DataFrame()
    if g.empty:
        st.caption("Ledger: no graded predictions yet.")
    else:
        g = g.copy(); g["correct"] = ((g["p_a"].astype(float) >= 0.5) == (g["y"].astype(float) == 1)).astype(int)
        c1, c2, c3 = st.columns(3)
        c1.metric("Graded predictions", len(g))
        c2.metric("Winners called", f"{g['correct'].mean():.1%}")
        hi = g[g["confidence"].isin(["Strong", "Solid"])]
        c3.metric("Strong / solid hit rate", f"{hi['correct'].mean():.1%}" if len(hi) else "–", f"{len(hi)} picks")
        st.dataframe(g.groupby("confidence")["correct"].agg(picks="size", hit_rate="mean").reindex(["Strong", "Solid", "Lean", "Coin flip"])
                     .style.format({"hit_rate": "{:.1%}"}))
        if "market_p_a" in g and g["market_p_a"].notna().sum() >= 20:
            mk = g[g["market_p_a"].notna()]
            st.caption(f"On {len(mk)} matches with a DraftKings line: model {mk['correct'].mean():.1%} vs market favourite "
                       f"{((mk['market_p_a'].astype(float) >= 0.5) == (mk['y'].astype(float) == 1)).mean():.1%}.")
    st.subheader("Walk-forward backtest")
    if bt.empty:
        st.caption("No backtest on file yet (the refresh workflow runs one weekly).")
    else:
        s = pipeline.summarize(bt)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Matches", s["matches"]); c2.metric("Accuracy", f"{s['accuracy']:.1%}")
        c3.metric("Log loss", f"{s['log_loss']:.4f}"); c4.metric("Elo-only accuracy", f"{s['elo_accuracy']:.1%}")
        st.dataframe(pd.DataFrame(s["by_confidence"]).T.reindex(["Strong", "Solid", "Lean", "Coin flip"]).style.format({"hit_rate": "{:.1%}"}))
        st.dataframe(pipeline.calibration_table(bt).style.format({"predicted": "{:.1%}", "actual": "{:.1%}"}), hide_index=True)

# --------------------------------------------------------------------------- data
with tab_data:
    st.write("Everything here is refreshed by the `refresh` GitHub Actions workflow. See DEPLOY.md to run it by hand.")
    if not D["tours"].empty:
        t = D["tours"].copy(); t["date"] = pd.to_datetime(t["date"])
        st.dataframe(t.sort_values("post_id", ascending=False).head(30)[["post_id", "date", "session", "venue", "n_matches", "n_finished"]], hide_index=True)
    if state:
        st.json({k: v for k, v in state.items() if k != "history"})
        if state.get("history"):
            st.dataframe(pd.DataFrame(state["history"]).tail(10))
