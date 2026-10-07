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

from tt_oracle import config, dk, picks, pipeline, store
from tt_oracle.names import canonicalize, fold, merge_enabled

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


MATCH_COLS = ["match_id", "post_id", "start_time", "session", "venue", "player_a", "player_b", "sets_a", "sets_b", "finished", "winner"]


@st.cache_data(show_spinner=False)
def load(sig, merge_names: bool):
    """Only what the pages show. The full history is ~300k matches; keep the app's memory small."""
    m = pd.read_csv(config.MATCHES_CSV, usecols=MATCH_COLS, parse_dates=["start_time"]) if config.MATCHES_CSV.exists() else pd.DataFrame()
    if not m.empty:
        m, _ = canonicalize(m, enabled=merge_names)
        m["finished"] = m["finished"].astype(bool)
        for c in ("session", "venue", "player_a", "player_b", "winner"):
            m[c] = m[c].astype("category")
    t = pd.read_csv(config.TOURNAMENTS_CSV).sort_values("post_id", ascending=False).head(60) if config.TOURNAMENTS_CSV.exists() else pd.DataFrame()
    return {"pred": store.read(config.PREDICTIONS_CSV), "matches": m, "odds": store.read(config.ODDS_CSV),
            "ledger": store.read(config.LEDGER_CSV), "bt": store.read(config.BACKTEST_CSV), "tours": t,
            "players": store.read(config.PLAYERS_CSV)}


def _sig():
    return tuple(p.stat().st_mtime if p.exists() else 0 for p in
                 (config.PREDICTIONS_CSV, config.MATCHES_CSV, config.ODDS_CSV, config.LEDGER_CSV, config.BACKTEST_CSV))


with st.sidebar:
    merge_names = st.checkbox("Merge name spellings", value=merge_enabled(),
                              help="The site spells players with and without Polish accents (Radło / Radlo) and with typos. "
                                   "On: treated as one player. Off: the old behaviour. Permanent default: TT_MERGE_NAMES setting.")
D = load(_sig(), merge_names)
pred, matches, odds, ledger, bt = D["pred"], D["matches"], D["odds"], D["ledger"], D["bt"]
if not pred.empty and merge_names:
    from tt_oracle.names import fold
    pred = pred.copy(); pred["player_a"] = pred["player_a"].map(fold); pred["player_b"] = pred["player_b"].map(fold); pred["pick"] = pred["pick"].map(fold)
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
        st.write(f"**{len(fin):,}** matches since {fin['start_time'].min():%b %Y}, **{matches['player_a'].nunique():,}** players")
        st.write(f"Results through **{fin['start_time'].max():%d %b %Y %H:%M}**")
        st.write(f"**{int((~matches['finished']).sum())}** upcoming on file")
    if config.STATE_FILE.exists():
        st.caption(f"Learners: {', '.join(state.get('learners') or [])}")
    if not odds.empty:
        st.caption(f"DraftKings lines: {len(odds)} ({pd.to_datetime(odds['fetched_at']).max():%d %b %H:%M} UTC)")
    st.divider()
    st.caption("Data refreshes every 6 hours via GitHub Actions. Predictions are logged before each match and graded after.")

tab_today, tab_price, tab_h2h, tab_mine, tab_track, tab_data = st.tabs(["Today", "Pricing desk", "Head to head", "My picks", "Track record", "Data"])


def save_button(r, source: str, key: str):
    """'Save pick' under a card; r is a prediction row (Series)."""
    if st.button("Save pick", key=key, help="Adds this pick to the My picks tab"):
        side = "a" if str(r["pick"]) == str(r["player_a"]) else "b"
        ok, msg = picks.add({"source": source, "match_id": r.get("match_id") if source == "today" else None,
                             "start_time": pd.Timestamp(r["start_time"]), "session": r.get("session"),
                             "player_a": r["player_a"], "player_b": r["player_b"], "pick": r["pick"],
                             "p_pick": float(r["pick_prob"]), "confidence": r.get("confidence"),
                             "dk_odds": (dk.decimal_to_american(r[f"dk_odds_{side}"]) if pd.notna(r.get(f"dk_odds_{side}")) else None),
                             "note": ""})
        st.toast(f"Saved: {r['pick']} ({msg})" if ok else f"Saved locally only — {msg}", icon="✅" if ok else "⚠️")

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
        ok = ~P["thin_history"].astype(bool)
        c2.metric("75%+ picks", int(((P["pick_prob"] >= 0.75) & ok).sum()), f"{int(((P['pick_prob'] >= 0.85) & ok).sum())} at 85%+")
        c3.metric("With DraftKings line", int(P["market_p_a"].notna().sum()) if "market_p_a" in P else 0)

        # ---- DraftKings matches: a key-based odds provider, DraftKings' own feed, or pasted lines
        with st.expander("DraftKings matches — odds feed", expanded=odds.empty):
            st.caption("Lines come from (1) an odds provider with your API key, (2) DraftKings' own site feed, which works "
                       "only if this server isn't blocked, or (3) pasted lines. Whatever is loaded is matched to today's cards.")
            k1, k2 = st.columns([2, 1])
            provider = k2.selectbox("Provider", list(dk.PROVIDERS), format_func=dk.PROVIDERS.get, key="odds_provider")
            default_key = dk.provider_key(provider) or ""
            key_in = k1.text_input("API key", value=default_key, type="password", key="odds_key_in",
                                   help="Saved for this session only. To keep it, add ODDSPAPI_KEY to the app's Secrets.")
            if key_in.strip():
                st.session_state[f"{provider.upper()}_KEY"] = key_in.strip()
            b1, b2, b3 = st.columns([1, 1, 2])
            if b1.button("Get DraftKings matches", type="primary", key="today_fetch"):
                with st.spinner("Fetching lines…"):
                    try:
                        got = dk.fetch_lines(provider, key_in.strip() or None)
                        if got.empty:
                            st.warning("No TT Elite lines came back. The log below says which step found nothing.")
                        else:
                            dk.snapshot_odds(got)
                            st.cache_data.clear()
                            odds = store.read(config.ODDS_CSV)
                            st.success(f"{len(got)} matches with {got['bookmaker'].iloc[0]} lines loaded.")
                    except Exception as ex:
                        st.error(f"Fetch failed: {ex}")
                    st.code("\n".join(dk.LAST_FETCH_LOG) or "(no log)")
            if b2.button("DraftKings feed only", key="today_fetch_dk", help="Skip the provider; try DraftKings' site feed from this server"):
                with st.spinner("Asking DraftKings…"):
                    got = dk.fetch_dk()
                    if got.empty:
                        st.warning("DraftKings' feed returned nothing from this server (it blocks most cloud IPs).")
                    else:
                        dk.snapshot_odds(got); st.cache_data.clear(); odds = store.read(config.ODDS_CSV)
                        st.success(f"{len(got)} matches loaded from DraftKings.")
                    st.code("\n".join(dk.LAST_FETCH_LOG) or "(no log)")
            b3.caption("The Pricing desk also has a paste box for lines copied from the DraftKings app.")
            if not odds.empty:
                od = odds.copy()
                od["when"] = pd.to_datetime(od["commence_time"], errors="coerce")
                od["line"] = od.apply(lambda r: f"{r['player_1']} {dk.decimal_to_american(r['odds_1'])} / {r['player_2']} {dk.decimal_to_american(r['odds_2'])}", axis=1)
                st.caption(f"{len(od)} lines on file from {od['bookmaker'].iloc[0]} ({pd.to_datetime(od['fetched_at']).max():%d %b %H:%M} UTC)")
                st.dataframe(od[["when", "line"]].sort_values("when"), hide_index=True, height=min(400, 40 + 35 * len(od)))
        if not odds.empty:
            P = dk.attach_market(P, odds)      # re-match with whatever lines are loaded right now
        cd1, cd2, cd3 = st.columns([1, 2, 1])
        day = cd1.selectbox("Day", sorted(P["start_time"].dt.date.unique()))
        only_dk = cd3.checkbox("Only matches on DraftKings", value=False, help="Show just the matches that have a line loaded")
        min_p = cd2.radio("Minimum model probability", [0.0, 0.75, 0.85], index=1, horizontal=True,
                          format_func=lambda v: {0.0: "All matches", 0.75: "75%+ picks", 0.85: "85%+ picks"}[v])
        q = st.text_input("Search a player or match", placeholder="e.g. Jadach, or Jadach Urban", key="today_q")
        view = P[P["start_time"].dt.date == day]
        if only_dk:
            view = view[view["market_p_a"].notna()]
        if min_p > 0:
            view = view[(view["pick_prob"] >= min_p) & ~view["thin_history"].astype(bool)]
        if q.strip():
            # every word typed must appear in one of the two names (accent-insensitive)
            import unicodedata
            def _fold(s): return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
            words = [_fold(w) for w in q.split()]
            both = (view["player_a"].map(_fold) + " " + view["player_b"].map(_fold))
            view = view[both.apply(lambda s: all(w in s for w in words))]
            if view.empty:
                view = P[P["start_time"].dt.date == day]
                both = (view["player_a"].map(_fold) + " " + view["player_b"].map(_fold))
                view = view[both.apply(lambda s: all(w in s for w in words))]
                if not view.empty:
                    st.caption("No match at the chosen probability filter; showing all matches for that search.")
        st.caption(f"{len(view)} matches shown" + (f" at {min_p:.0%}+ (players with thin history excluded)" if min_p and not q.strip() else ""))
        for sess in config.SESSIONS:
            vs = view[view["session"] == sess]
            if vs.empty:
                continue
            st.subheader(f"{sess.title()} session · {len(vs)} matches")
            for _, r in vs.iterrows():
                st.markdown(card_html(r), unsafe_allow_html=True)
                save_button(r, "today", f"save_{r['match_id']}")
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
    min_pp = st.radio("Only price picks at", [0.0, 0.75, 0.85], index=1, horizontal=True, key="price_min",
                      format_func=lambda v: {0.0: "any probability", 0.75: "75%+", 0.85: "85%+"}[v])
    cfa, cfb = st.columns([1, 3])
    if cfa.button("Fetch DraftKings lines now"):
        with st.spinner("Asking DraftKings…"):
            try:
                res = dk.snapshot_odds()
                if res.get("lines"):
                    st.success(f"Got {res['lines']} TT Elite lines from DraftKings.")
                    st.cache_data.clear()
                    odds = store.read(config.ODDS_CSV)
                else:
                    st.warning("DraftKings returned no TT Elite lines from this server. Details below; the paste box still works.")
                    st.code("\n".join(dk.LAST_FETCH_LOG) or "(no log)")
            except Exception as ex:
                st.error(f"Fetch failed: {ex}")
                st.code("\n".join(dk.LAST_FETCH_LOG) or "(no log)")
    cfb.caption("Tries DraftKings' public odds feed for league 208037 (TT Elite Series) from the app's own server.")
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
        if min_pp > 0:
            priced = priced[priced["p_model"] >= min_pp]
        pq = st.text_input("Search a player", placeholder="e.g. Jadach", key="price_q")
        if pq.strip():
            import unicodedata
            def _fold2(s): return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
            words = [_fold2(w) for w in pq.split()]
            both = priced["player"].map(_fold2) + " " + priced["opponent"].map(_fold2)
            priced = priced[both.apply(lambda s: all(w in s for w in words))]
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
        snap = D["players"]
        if snap.empty:
            st.info("Player ratings appear after the next refresh run.")
            st.stop()
        names = snap["player"].tolist()
        folded = {n: fold(n).lower() for n in names}

        def player_picker(col, label, key, default_index):
            """Type any part of a name (accents ignored); the list narrows and the best match is selected at once."""
            q = col.text_input(f"Search {label}", key=f"{key}_q", placeholder="type part of a name, e.g. radlo")
            toks = fold(q).lower().split()
            opts = [n for n in names if all(t in folded[n] for t in toks)] if toks else names
            if not opts:
                col.warning(f"No player matching '{q}'.")
                opts = names
            # best match first: name that starts with the typed text, then by rating (names list is rating-sorted)
            if toks:
                opts = sorted(opts, key=lambda n: 0 if folded[n].startswith(toks[0]) or any(w.startswith(toks[0]) for w in folded[n].split()) else 1)
            return col.selectbox(label, opts, index=0 if toks else min(default_index, len(opts) - 1), key=f"{key}_sel")

        c1, c2 = st.columns(2)
        a = player_picker(c1, "Player A", "h2h_a", 0)
        b = player_picker(c2, "Player B", "h2h_b", 1 if len(names) > 1 else 0)
        fin = matches[matches["finished"]]
        h = fin[((fin["player_a"] == a) & (fin["player_b"] == b)) | ((fin["player_a"] == b) & (fin["player_b"] == a))].sort_values("start_time", ascending=False)
        aw = int(((h["player_a"] == a) & (h["winner"] == "a")).sum() + ((h["player_b"] == a) & (h["winner"] == "b")).sum())
        sa, sb = snap.set_index("player").loc[a], snap.set_index("player").loc[b]
        sess = st.radio("Session", config.SESSIONS, horizontal=True, index=1, key="h2h_sess")
        if a != b:
            @st.cache_resource(show_spinner=False)
            def _model():
                from tt_oracle.model import Predictor
                return Predictor.load()
            mdl = _model()
            if mdl is None:
                st.caption("Model file not available on this server yet.")
            else:
                from tt_oracle.features import matchup_row
                row = matchup_row(sa.rename_axis(None).to_dict() | {"player": a}, sb.to_dict() | {"player": b}, aw, len(h), sess)
                pr = mdl.predict(row).iloc[0]
                pr["elo_p_a"] = row["elo_p_a"].iloc[0]; pr["h2h_n"] = len(h)
                pr["A_today_n"], pr["B_today_n"] = 0, 0
                pr["session"], pr["venue"], pr["start_time"] = sess, "any venue", pd.Timestamp.now()
                st.markdown(card_html(pr), unsafe_allow_html=True)
                save_button(pr, "h2h", f"save_h2h_{a}_{b}_{sess}")
                st.caption("Model prediction for this matchup if it were played in the chosen session today, both players "
                           "fresh. Fatigue and same-day load are set to zero; the live Today tab uses the real schedule.")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{a} Elo", f"{sa['elo']:.0f}", f"last 10: {sa['win10']:.0%}")
        c2.metric("Head to head", f"{aw}–{len(h) - aw}")
        c3.metric(f"{b} Elo", f"{sb['elo']:.0f}", f"last 10: {sb['win10']:.0%}")
        if not h.empty:
            st.dataframe(h[["start_time", "session", "venue", "player_a", "sets_a", "sets_b", "player_b"]], hide_index=True)
        st.subheader("Ratings leaderboard")
        st.dataframe(snap.head(40).rename(columns={"n": "matches"}).style.format({"elo": "{:.0f}", "win10": "{:.0%}"}), hide_index=True)

# --------------------------------------------------------------------------- my picks
with tab_mine:
    mine = picks.grade(picks.load(), matches)
    if picks._token() is None:
        st.warning("Picks are saved on the app's disk only, which resets on every redeploy (about every 6 hours). "
                   "To keep them permanently, add a GitHub token to the app's Secrets (see DEPLOY.md, 'Saved picks').")
    if mine.empty:
        st.info("No saved picks yet. Use the Save pick button under any card on the Today or Head to head tabs.")
    else:
        g = mine[mine["result"].notna()]
        c1, c2, c3 = st.columns(3)
        c1.metric("Saved picks", len(mine))
        c2.metric("Graded", len(g))
        c3.metric("Record", f"{int((g['result'] == 'won').sum())}–{int((g['result'] == 'lost').sum())}" if len(g) else "–",
                  f"{(g['result'] == 'won').mean():.0%} hit" if len(g) else None)
        show = mine.sort_values("start_time", ascending=False).copy()
        show["match"] = show["player_a"].astype(str) + " vs " + show["player_b"].astype(str)
        show["result"] = show["result"].fillna("pending")
        st.dataframe(show[["start_time", "session", "match", "pick", "p_pick", "confidence", "dk_odds", "source", "result", "winner"]]
                     .rename(columns={"p_pick": "model"}).style.format({"model": "{:.0%}"}), hide_index=True)
        st.subheader("Manage")
        labels = {r["pick_id"]: f"{pd.Timestamp(r['start_time']):%d %b %H:%M} · {r['pick']} vs {r['player_b'] if r['pick'] == r['player_a'] else r['player_a']}"
                  for _, r in show.iterrows()}
        sel = st.multiselect("Remove selected picks", options=list(labels), format_func=labels.get)
        cm1, cm2 = st.columns([1, 3])
        if cm1.button("Remove", disabled=not sel):
            ok, msg = picks.remove(sel); st.toast(msg); st.rerun()
        sure = cm2.checkbox("I want to clear all saved picks")
        if cm2.button("Clear all", type="primary", disabled=not sure):
            ok, msg = picks.clear(); st.toast(msg); st.rerun()
        st.download_button("Download my picks (CSV)", show.to_csv(index=False), file_name="my_picks.csv")

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
        t75, t85 = g[g["pick_prob"].astype(float) >= 0.75], g[g["pick_prob"].astype(float) >= 0.85]
        st.caption(f"75%+ picks: {t75['correct'].mean():.1%} of {len(t75)} · 85%+ picks: {t85['correct'].mean():.1%} of {len(t85)}" if len(t75) else "")
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
        pk = bt["pick_prob"] if "pick_prob" in bt else np.maximum(bt["p_a"], 1 - bt["p_a"])
        b75, b85 = bt[pk >= 0.75], bt[pk >= 0.85]
        k1, k2 = st.columns(2)
        k1.metric("75%+ picks hit rate", f"{b75['correct'].mean():.1%}", f"{len(b75):,} picks")
        k2.metric("85%+ picks hit rate", f"{b85['correct'].mean():.1%}" if len(b85) else "–", f"{len(b85):,} picks")
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
