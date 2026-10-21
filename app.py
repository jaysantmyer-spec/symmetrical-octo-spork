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

# Streamlit Cloud reloads app.py on every deploy but keeps the already-imported tt_oracle modules in memory,
# so a new app.py can call a function the old module doesn't have (AttributeError until a manual reboot).
# Drop our own modules when any of their files changed, so they are imported fresh.
def _reload_own_modules():
    import importlib, sys
    from pathlib import Path
    pkg = Path(__file__).parent / "tt_oracle"
    stamp = max((f.stat().st_mtime for f in pkg.glob("*.py")), default=0)
    if st.session_state.get("_pkg_stamp") != stamp:
        for name in [n for n in sys.modules if n == "tt_oracle" or n.startswith("tt_oracle.")]:
            del sys.modules[name]
        importlib.invalidate_caches()
        st.session_state["_pkg_stamp"] = stamp
_reload_own_modules()

from tt_oracle import config, dk, picks, pipeline, store
from tt_oracle import model as M
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
SITE_TZ, HOME_TZ = "Europe/Warsaw", "America/New_York"      # data is in the site's (Polish) clock; shown in US Eastern


def to_local(t):
    """Series or Timestamp in the site's clock -> the viewer's clock (naive), when the sidebar toggle is on."""
    if not st.session_state.get("show_local", True):
        return t
    if isinstance(t, pd.Series):
        return pd.to_datetime(t).dt.tz_localize(SITE_TZ, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert(HOME_TZ).dt.tz_localize(None)
    t = pd.Timestamp(t)
    return t.tz_localize(SITE_TZ, ambiguous="NaT", nonexistent="shift_forward").tz_convert(HOME_TZ).tz_localize(None) if t.tzinfo is None else t.tz_convert(HOME_TZ).tz_localize(None)


def tz_label() -> str:
    return "ET" if st.session_state.get("show_local", True) else "Poland"


@st.cache_data(show_spinner=False)
def load(sig, merge_names: bool):
    """Only what the pages show. The full history is ~300k matches; keep the app's memory small."""
    m = pd.DataFrame()
    for attempt in range(5):                     # Streamlit Cloud may be mid-pull of a refresh commit: file briefly empty
        try:
            m = pd.read_csv(config.MATCHES_CSV, usecols=MATCH_COLS, parse_dates=["start_time"]) if config.MATCHES_CSV.exists() else pd.DataFrame()
            break
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
            import time; time.sleep(1.5)
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
    st.checkbox("Show times in US Eastern", value=True, key="show_local",
                help="Off: the site's own clock (Poland, 6 hours ahead of New York).")
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
    t = to_local(r["start_time"])
    mk = ""
    if pd.notna(r.get("market_p_a")):
        edge = pa - r["market_p_a"]
        side = r["player_a"] if edge > 0 else r["player_b"]
        mk = (f'<div class="line">DraftKings: {e(str(r["player_a"]))} {dk.decimal_to_american(r["dk_odds_a"])} / '
              f'{e(str(r["player_b"]))} {dk.decimal_to_american(r["dk_odds_b"])} · market {r["market_p_a"]:.0%} · '
              f'model sees {abs(edge):.0%} more on <b>{e(str(side))}</b></div>')
    return f"""<div class="m{' hi' if hi else ''}"><div class="meta"><span>{t:%a %d %b %I:%M %p} {tz_label()} · {e(str(r['session']))} · {e(str(r['venue']))}</span><span>{thin} {badge}</span></div>
<div class="names"><div class="a">{e(str(r['player_a']))}</div><div class="b">{e(str(r['player_b']))}</div></div>
<div class="split"><div class="ra" style="width:{pa*100:.1f}%"></div><div class="rb" style="width:{(1-pa)*100:.1f}%"></div></div>
<div class="pcts"><span style="color:{ORANGE}">{pa:.0%}</span><span style="color:{NAVY}">{1-pa:.0%}</span></div>
<div class="line"><b>{e(str(r['pick']))}</b> ({str(r['confidence']).lower()}). Elo alone: {r['elo_p_a']:.0%}. H2H matches: {int(r['h2h_n'])}. Today so far: {int(r['A_today_n'])} / {int(r['B_today_n'])}.</div>{games_line(r)}{spread_line(r)}{points_line(r)}{mk}{blend_line(r)}</div>"""


def fair(p: float) -> str:
    """Fair American price for a probability."""
    return dk.decimal_to_american(1 / max(min(float(p), 0.995), 0.005))


def spread_line(r) -> str:
    """Games spread (±1.5) and points spread (±3.5) - DraftKings' two spread markets - with fair prices."""
    if pd.isna(r.get("p_a_games_m15")):
        return ""
    a, b = html.escape(str(r["player_a"])), html.escape(str(r["player_b"]))
    ga, gb = float(r["p_a_games_m15"]), float(r["p_b_games_m15"])
    fav_a = float(r["p_a"]) >= 0.5
    fav, dog, g_fav, g_dog = (a, b, ga, gb) if fav_a else (b, a, gb, ga)
    out = (f'<div class="line" style="color:#5B6470">Games spread: {fav} −1.5 <b>{g_fav:.0%}</b> ({fair(g_fav)}) · '
           f'{dog} +1.5 <b>{1 - g_fav:.0%}</b> ({fair(1 - g_fav)})</div>')
    if pd.notna(r.get("p_a_pts_m35")):
        pa35, pb35 = float(r["p_a_pts_m35"]), float(r["p_b_pts_m35"])
        p_fav = pa35 if fav_a else pb35
        em = float(r["exp_point_margin"]); em = em if fav_a else -em
        out += (f'<div class="line" style="color:#5B6470">Points spread: {fav} −3.5 <b>{p_fav:.0%}</b> ({fair(p_fav)}) · '
                f'{dog} +3.5 <b>{1 - p_fav:.0%}</b> ({fair(1 - p_fav)}) · expected margin {fav} by <b>{em:.1f}</b> points</div>')
    return out


def blend_line(r) -> str:
    if pd.isna(r.get("p_blend")):
        return ""
    pb = float(r["p_blend"])
    return (f'<div class="line">Model × market blend: <b>{html.escape(str(r["player_a"]))} {pb:.0%}</b> / '
            f'{html.escape(str(r["player_b"]))} {1 - pb:.0%} (learned from graded matches with a closing line)</div>')


def points_line(r) -> str:
    """Total points and game-1 conditionals when the vendor points feed is on."""
    if pd.isna(r.get("exp_total_points")) and pd.isna(r.get("p_a_if_g1")):
        return ""
    a, b = html.escape(str(r["player_a"])), html.escape(str(r["player_b"]))
    parts = []
    if pd.notna(r.get("exp_total_points")):
        parts.append(f"model total points <b>{float(r['exp_total_points']):.0f}</b>")
    if pd.notna(r.get("p_a_if_g1")):
        parts.append(f"if {a} takes game 1: {a} <b>{float(r['p_a_if_g1']):.0%}</b> · if {b} takes game 1: {b} <b>{1 - float(r['p_a_if_lost_g1']):.0%}</b>")
    return '<div class="line" style="color:#5B6470">Points: ' + " · ".join(parts) + '</div>'


def games_line(r) -> str:
    """Number-of-games percentages (DraftKings' 'total games' market) when the model has them."""
    if pd.isna(r.get("p_g3")):
        return ""
    return (f'<div class="line" style="color:#5B6470">Games: 3 <b>{float(r["p_g3"]):.0%}</b> · 4 <b>{float(r["p_g4"]):.0%}</b> · 5 <b>{float(r["p_g5"]):.0%}</b>'
            f' → over 3.5 games {float(r["over35"]):.0%}, over 4.5 games {float(r["over45"]):.0%}.'
            f' Likely score {html.escape(str(r.get("likely_score", "")))} ({float(r.get("likely_score_p", 0)):.0%}).</div>')


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("TT Elite Oracle")
    if matches.empty:
        st.warning("No data yet. The first GitHub Actions backfill hasn't run (see DEPLOY.md).")
    else:
        fin = matches[matches["finished"]]
        st.write(f"**{len(fin):,}** matches since {fin['start_time'].min():%b %Y}, **{matches['player_a'].nunique():,}** players")
        st.write(f"Results through **{to_local(fin['start_time'].max()):%d %b %Y %I:%M %p} {tz_label()}**")
        st.write(f"**{int((~matches['finished']).sum())}** upcoming on file")
    if config.STATE_FILE.exists():
        st.caption(f"Learners: {', '.join(state.get('learners') or [])}")
    if not odds.empty:
        st.caption(f"DraftKings lines: {len(odds)} (updated {pd.to_datetime(odds['fetched_at']).max().tz_localize('UTC').tz_convert(HOME_TZ):%d %b %I:%M %p} ET)")
    st.divider()
    st.caption("Data refreshes every 6 hours via GitHub Actions. Predictions are logged before each match and graded after.")

tab_today, tab_mis, tab_price, tab_live, tab_h2h, tab_mine, tab_track, tab_data = st.tabs(["Today", "Mismatches", "Pricing desk", "Live", "Head to head", "My picks", "Track record", "Data"])


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
        P["start_local"] = to_local(P["start_time"])
        P = P.sort_values("start_time")
        # ---- which picks to show: everything, or only picks at or above a probability you choose
        ok = ~P["thin_history"].astype(bool)
        t1, t2 = st.columns([1, 3])
        mode = t1.radio("Show", ["All picks", "Minimum %"], index=1, horizontal=True, key="today_mode",
                        help="'All picks' lists every upcoming match. 'Minimum %' keeps only picks the model rates at or above the slider.")
        min_p = t2.slider("Minimum model probability for the pick", 50, 95, 75, 5, format="%d%%", key="today_min_pct",
                          disabled=(mode == "All picks")) / 100.0 if mode == "Minimum %" else 0.0
        label = f"{min_p:.0%}+ picks" if min_p else "All picks"
        c1, c2, c3 = st.columns(3)
        c1.metric("Upcoming matches", len(P))
        sel_all = P[(P["pick_prob"] >= min_p) & ok] if min_p else P
        c2.metric(label, len(sel_all), f"{int(((P['pick_prob'] >= min(0.95, min_p + 0.10)) & ok).sum())} at {min(0.95, min_p + 0.10):.0%}+" if min_p else None)
        c3.metric("With DraftKings line", int(P["market_p_a"].notna().sum()) if "market_p_a" in P else 0)

        # ---- where the picks are: every selected pick across all upcoming days, grouped by day and session
        picks75 = sel_all.copy() if min_p else P[(P["pick_prob"] >= 0.75) & ok].copy()
        if not picks75.empty:
            hi_t = min(0.95, (min_p or 0.75) + 0.10)
            picks75["day"] = picks75["start_local"].dt.date
            by_day = picks75.groupby("day").agg(picks=("match_id", "size"), at85=("pick_prob", lambda v: int((v >= hi_t).sum())))
            summary = " · ".join(f"{pd.Timestamp(d):%a %d %b}: {int(r['picks'])} ({int(r['at85'])} at {hi_t:.0%}+)" for d, r in by_day.iterrows())
            with st.expander(f"Where the {len(picks75)} {label.lower() if min_p else '75%+ picks'} are — {summary}", expanded=False):
                tbl = picks75.sort_values("start_time")[["start_local", "session", "player_a", "player_b", "pick", "pick_prob", "confidence"]].copy()
                tbl["when"] = tbl["start_local"].dt.strftime("%a %d %b %I:%M %p")
                tbl["match"] = tbl["player_a"].astype(str) + " vs " + tbl["player_b"].astype(str)
                st.dataframe(tbl[["when", "session", "match", "pick", "pick_prob", "confidence"]].rename(columns={"pick_prob": "model"})
                             .style.format({"model": "{:.0%}"}), hide_index=True, height=min(500, 40 + 35 * len(tbl)))
                st.caption(f"Times in {tz_label()}. Pick the day below to see the full cards; choose 'All days' to list everything.")

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
            b3.caption("DraftKings blocks cloud servers, so the dependable route is `scripts/push_lines.py` run on your own "
                       "computer: it pulls the lines there and pushes them here (see DEPLOY.md). The Pricing desk also has a paste box.")
            if not odds.empty:
                od = odds.copy()
                od["when"] = pd.to_datetime(od["commence_time"], errors="coerce")
                od["line"] = od.apply(lambda r: f"{r['player_1']} {dk.decimal_to_american(r['odds_1'])} / {r['player_2']} {dk.decimal_to_american(r['odds_2'])}", axis=1)
                st.caption(f"{len(od)} lines on file from {od['bookmaker'].iloc[0]} (updated {pd.to_datetime(od['fetched_at']).max().tz_localize('UTC').tz_convert(HOME_TZ):%d %b %I:%M %p} ET)")
                st.dataframe(od[["when", "line"]].sort_values("when"), hide_index=True, height=min(400, 40 + 35 * len(od)))
        if not odds.empty:
            P = dk.attach_market(P, odds)      # re-match with whatever lines are loaded right now
        cd1, cd3 = st.columns([1, 1])
        days = sorted(P["start_local"].dt.date.unique())
        day = cd1.selectbox("Day", ["All days"] + days, index=1 if days else 0)
        only_dk = cd3.checkbox("Only matches on DraftKings", value=False, help="Show just the matches that have a line loaded")
        q = st.text_input("Search a player or match", placeholder="e.g. Jadach, or Jadach Urban", key="today_q")
        view = P if day == "All days" else P[P["start_local"].dt.date == day]
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
                view = P if day == "All days" else P[P["start_local"].dt.date == day]
                both = (view["player_a"].map(_fold) + " " + view["player_b"].map(_fold))
                view = view[both.apply(lambda s: all(w in s for w in words))]
                if not view.empty:
                    st.caption("No match at the chosen probability filter; showing all matches for that search.")
        st.caption(f"{len(view)} matches shown" + (f" — picks at {min_p:.0%}+ (players with thin history excluded)" if min_p and not q.strip() else " — all picks"))
        for d, vd in view.groupby(view["start_local"].dt.date, sort=True):
            if day == "All days":
                st.markdown(f"### {pd.Timestamp(d):%A %d %B} · {len(vd)} matches")
            for sess in config.SESSIONS:
                vs = vd[vd["session"] == sess]
                if vs.empty:
                    continue
                st.subheader(f"{sess.title()} session · {len(vs)} matches")
                for _, r in vs.iterrows():
                    st.markdown(card_html(r), unsafe_allow_html=True)
                    save_button(r, "today", f"save_{r['match_id']}")
        st.download_button("Download predictions (CSV)", P.to_csv(index=False), file_name="tt_predictions.csv")


# --------------------------------------------------------------------------- mismatches (model vs DraftKings)
with tab_mis:
    st.write("Matches where the model and DraftKings disagree. A **flip** is when they favour different players; a **gap** is a "
             "disagreement in degree (same favourite, but the model is much more or less sure). Both need a line loaded — paste "
             "DraftKings' page on the Pricing desk to refresh them.")
    if pred.empty or odds.empty:
        st.info("Needs predictions and DraftKings lines (paste them on the Pricing desk).")
    else:
        Mm = dk.attach_market(pred, odds)
        Mm["start_time"] = pd.to_datetime(Mm["start_time"])
        Mm = Mm[Mm["market_p_a"].notna() & (Mm["start_time"] >= pd.Timestamp.now(tz=SITE_TZ).tz_localize(None) - pd.Timedelta(hours=1))].copy()
        if Mm.empty:
            st.info("No upcoming match has a DraftKings line right now.")
        else:
            Mm["edge_a"] = Mm["p_a"] - Mm["market_p_a"]
            Mm["model_fav"] = np.where(Mm["p_a"] >= 0.5, Mm["player_a"], Mm["player_b"])
            Mm["dk_fav"] = np.where(Mm["market_p_a"] >= 0.5, Mm["player_a"], Mm["player_b"])
            Mm["flip"] = Mm["model_fav"] != Mm["dk_fav"]
            Mm["gap"] = Mm["edge_a"].abs()
            # the side the model likes relative to the book, with its DraftKings price and fair price
            Mm["side"] = np.where(Mm["edge_a"] > 0, Mm["player_a"], Mm["player_b"])
            Mm["side_p"] = np.where(Mm["edge_a"] > 0, Mm["p_a"], 1 - Mm["p_a"])
            Mm["side_mkt"] = np.where(Mm["edge_a"] > 0, Mm["market_p_a"], 1 - Mm["market_p_a"])
            Mm["side_dk"] = np.where(Mm["edge_a"] > 0, Mm["dk_odds_a"], Mm["dk_odds_b"])
            Mm["ev"] = Mm["side_p"] * (Mm["side_dk"] - 1) - (1 - Mm["side_p"])
            c1, c2, c3 = st.columns([1, 1, 2])
            min_gap = c1.slider("Minimum gap", 0, 30, 8, 1, format="%d%%", key="mis_gap") / 100.0
            only_flips = c2.checkbox("Flips only", value=False, help="Only matches where model and DraftKings favour different players")
            hide_thin = c3.checkbox("Hide thin-history players", value=True)
            V = Mm[Mm["gap"] >= min_gap]
            if only_flips:
                V = V[V["flip"]]
            if hide_thin:
                V = V[~V["thin_history"].astype(bool)]
            V = V.sort_values("gap", ascending=False)
            k1, k2, k3 = st.columns(3)
            k1.metric("Matches with a line", len(Mm)); k2.metric("Flips", int(Mm["flip"].sum())); k3.metric("Shown", len(V))
            if V.empty:
                st.caption("Nothing clears the gap you set.")
            tbl = V.assign(when=V["start_time"].map(lambda t: to_local(t).strftime("%a %I:%M %p")),
                           match=V["player_a"] + " v " + V["player_b"],
                           model=V["side_p"], draftkings=V["side_mkt"], dk_price=V["side_dk"].map(dk.decimal_to_american),
                           fair=V["side_p"].map(dk.fair_american), kind=np.where(V["flip"], "FLIP", "gap"))
            st.dataframe(tbl[["when", "match", "kind", "side", "model", "draftkings", "gap", "dk_price", "fair", "ev"]]
                         .rename(columns={"side": "model leans", "model": "model %", "draftkings": "DK %", "dk_price": "DK price", "ev": "EV per $1"})
                         .style.format({"model %": "{:.0%}", "DK %": "{:.0%}", "gap": "{:.0%}", "EV per $1": "{:+.2f}"}), hide_index=True,
                         height=min(600, 40 + 36 * max(len(tbl), 1)))
            st.caption("'model leans' is the side the model rates higher than DraftKings does; 'EV per $1' is the expected profit of $1 on that side at "
                       "DraftKings' price if the model is right. Flips are the strongest disagreements; use the Live tab once a game or two is in.")
            for _, r in V.head(30).iterrows():
                st.markdown(card_html(r), unsafe_allow_html=True)
                save_button(r, "today", f"save_mis_{r['match_id']}")

# --------------------------------------------------------------------------- pricing desk
with tab_price:
    st.write("Model probability as a fair price vs DraftKings. A bet is flagged when the edge beats your threshold, "
             "EV is positive, and both players have enough history.")
    c1, c2, c3 = st.columns(3)
    bankroll = c1.number_input("Bankroll ($)", 50.0, 1_000_000.0, 500.0, step=50.0)
    kf = c2.select_slider("Kelly fraction", options=[0.1, 0.25, 0.5], value=0.25,
                          format_func=lambda v: {0.1: "1/10", 0.25: "Quarter", 0.5: "Half"}[v])
    thr = c3.slider("Edge threshold", 0.0, 0.15, 0.05, 0.01, format="%.2f")
    pm1, pm2 = st.columns([1, 3])
    pmode = pm1.radio("Price", ["All picks", "Minimum %"], index=1, horizontal=True, key="price_mode")
    min_pp = pm2.slider("Only price picks at or above", 50, 95, 75, 5, format="%d%%", key="price_min_pct",
                        disabled=(pmode == "All picks")) / 100.0 if pmode == "Minimum %" else 0.0
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
        st.markdown("**Easiest:** open DraftKings' TT Elite Series page (moneyline view), press **⌘A** then **⌘C** to copy the whole page, "
                    "paste it in the box below and press the button — the names and prices are picked out automatically. "
                    "You can also type lines by hand, one match per line: `Oskar Jadach -150 / Wojciech Urban +120`.")
        st.caption("With a GitHub token in the app's Secrets the lines are stored in the repo, so they also feed the model × market blend.")
        with st.expander("Optional: a 'Copy DK lines' bookmark that copies just the lines"):
            st.markdown("1. In Safari (or Chrome), bookmark any page and name it **Copy DK lines**.\n"
                        "2. Edit that bookmark and replace its address with the whole text below (hover the box, click the copy icon).\n"
                        "3. Open DraftKings' TT Elite Series page, click the bookmark: it copies every match's moneyline. Paste below, press the button.")
            bm = config.ROOT / "scripts" / "bookmarklet.txt"
            st.code(bm.read_text().strip() if bm.exists() else "(bookmarklet file missing)", language=None)
        txt = st.text_area("Lines", height=120, key="paste_odds")
        if st.button("Use these lines"):
            (config.DATA / "last_paste.txt").write_text(txt or "")      # kept with the lines, for checking the parser
            o = dk.parse_pasted(txt)
            n_raw = len(o)
            if not o.empty and not D["players"].empty:
                # keep only pairs where both names are known TT Elite players (guards against copying the wrong page)
                known = [dk._key_set(n) for n in D["players"]["player"].tolist()]
                def _known(name):
                    k = dk._key_set(name); return any(dk._same(k, x) for x in known)
                o = o[o["player_1"].map(_known) & o["player_2"].map(_known)]
            if o.empty:
                st.error("Couldn't read any TT Elite lines." + (f" {n_raw} pairs were found but none are TT Elite players — "
                         "was that DraftKings' TT Elite Series page? (Table Tennis → TT Elite Series, moneyline view.)" if n_raw else ""))
            else:
                if n_raw > len(o):
                    st.caption(f"{n_raw - len(o)} pairs ignored (not TT Elite players).")
                dk.snapshot_odds(o)
                from tt_oracle.market import push_odds_to_github
                okp, msgp = push_odds_to_github()
                st.success(f"Saved {len(o)} lines ({msgp}). Re-pricing below.")
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
            flagged = flagged.assign(start_time=to_local(flagged["start_time"]).dt.strftime("%a %d %b %I:%M %p"))
            st.dataframe(flagged[["start_time", "player", "opponent", "p_model", "fair", "dk", "p_market", "edge", "ev", "stake"]]
                         .style.format(fmt), hide_index=True)
        else:
            st.caption("Nothing clears the bar right now" + ("" if not odds.empty else " — no DraftKings lines loaded."))
        st.subheader("Every side")
        priced = priced.assign(start_time=to_local(priced["start_time"]).dt.strftime("%a %d %b %I:%M %p"))
        st.dataframe(priced[["start_time", "player", "opponent", "p_model", "fair", "dk", "market", "edge", "ev", "stake", "thin", "bet"]]
                     .style.format(fmt), hide_index=True)
        # ---- spreads: the model's cover chance for every upcoming match, and a checker for DraftKings' prices
        if "p_a_pts_m35" in P and P["p_a_pts_m35"].notna().any():
            st.subheader("Spreads")
            st.caption("DraftKings' two spread markets: games (−1.5 = wins 3-0 or 3-1) and points (−3.5 = wins the total "
                       "points by 4 or more). Fair = the price that matches the model's chance; beat it and the bet has value.")
            up = P[~P["finished"].astype(bool)].dropna(subset=["p_a_pts_m35"]).sort_values("start_time")
            if pq.strip():
                both = up["player_a"].map(_fold2) + " " + up["player_b"].map(_fold2)
                up = up[both.apply(lambda s: all(w in s for w in words))]
            rows_sp = []
            for _, r in up.iterrows():
                fav_a = float(r["p_a"]) >= 0.5
                fav, dog = (r["player_a"], r["player_b"]) if fav_a else (r["player_b"], r["player_a"])
                g = float(r["p_a_games_m15"] if fav_a else r["p_b_games_m15"]); p35 = float(r["p_a_pts_m35"] if fav_a else r["p_b_pts_m35"])
                em = float(r["exp_point_margin"]); em = em if fav_a else -em
                rows_sp.append({"start_time": to_local(r["start_time"]).strftime("%a %d %b %I:%M %p"), "favourite": fav, "underdog": dog,
                                "fav −1.5 games": g, "fair": fair(g), "dog +1.5": 1 - g, "fair ": fair(1 - g),
                                "fav −3.5 pts": p35, "fair  ": fair(p35), "dog +3.5": 1 - p35, "fair   ": fair(1 - p35), "exp margin": em})
            if rows_sp:
                st.dataframe(pd.DataFrame(rows_sp).style.format({"fav −1.5 games": "{:.0%}", "dog +1.5": "{:.0%}", "fav −3.5 pts": "{:.0%}",
                                                                 "dog +3.5": "{:.0%}", "exp margin": "{:+.1f}"}), hide_index=True)
                with st.expander("Check a DraftKings spread price"):
                    opts = {f"{r['favourite']} v {r['underdog']} ({r['start_time']})": r for r in rows_sp}
                    pick_sp = st.selectbox("Match", list(opts), key="sp_match")
                    rr = opts[pick_sp]
                    c1, c2 = st.columns(2)
                    mkt = c1.selectbox("Market", [f"{rr['favourite']} −1.5 games", f"{rr['underdog']} +1.5 games",
                                                  f"{rr['favourite']} −3.5 points", f"{rr['underdog']} +3.5 points"], key="sp_mkt")
                    price = c2.text_input("DraftKings price", value="-110", key="sp_price")
                    model_p = {0: rr["fav −1.5 games"], 1: rr["dog +1.5"], 2: rr["fav −3.5 pts"], 3: rr["dog +3.5"]}[
                        [f"{rr['favourite']} −1.5 games", f"{rr['underdog']} +1.5 games", f"{rr['favourite']} −3.5 points", f"{rr['underdog']} +3.5 points"].index(mkt)]
                    try:
                        dec = dk.american_to_decimal(price)
                        ev = model_p * (dec - 1) - (1 - model_p)
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Model chance", f"{model_p:.0%}", f"fair {fair(model_p)}")
                        m2.metric("Break-even at this price", f"{1 / dec:.0%}")
                        m3.metric("EV per $1", f"{ev:+.2f}", "value" if ev > 0.03 else "pass")
                    except Exception:
                        st.caption("Enter an American price like -185 or +125.")


# --------------------------------------------------------------------------- live (in-match)
with tab_live:
    from tt_oracle import live as LV
    lm = LV.load()
    st.write("Enter the games played so far and the model updates the winner, the likely final score and the expected total. "
             "Fitted on a year of game-by-game scores: after one game it calls the winner 74% of the time, after two 76%, after three 80%.")
    if lm is None:
        st.info("The live model appears after the next refresh run.")
    else:
        src = st.radio("Match", ["From today's list", "Any two players (enter pre-match %)"], horizontal=True, key="live_src")
        p_pre, a, b = None, "Player A", "Player B"
        if src == "From today's list" and not pred.empty:
            Pl = pred.copy(); Pl["start_time"] = pd.to_datetime(Pl["start_time"])
            Pl = Pl[Pl["start_time"] >= pd.Timestamp.now(tz=SITE_TZ).tz_localize(None) - pd.Timedelta(hours=2)].sort_values("start_time")
            lq = st.text_input("Search the match", placeholder="e.g. Kosmal", key="live_q")
            Pl = search_filter(Pl, lq, ["player_a", "player_b"]) if lq.strip() else Pl
            if Pl.empty:
                st.caption("No upcoming match matches that search.")
            else:
                opts = {r["match_id"]: f"{to_local(r['start_time']):%a %I:%M %p} · {r['player_a']} v {r['player_b']} ({float(r['p_a']):.0%})" for _, r in Pl.head(80).iterrows()}
                mid = st.selectbox("Pick the match", list(opts), format_func=opts.get, key="live_match")
                r = Pl[Pl["match_id"] == mid].iloc[0]
                p_pre, a, b = float(r["p_a"]), r["player_a"], r["player_b"]
        else:
            c1, c2, c3 = st.columns(3)
            a = c1.text_input("Player A", "Player A", key="live_a"); b = c2.text_input("Player B", "Player B", key="live_b")
            p_pre = c3.slider(f"Pre-match chance for {a}", 5, 95, 60, 1, format="%d%%", key="live_p") / 100.0
        if p_pre is not None:
            st.markdown(f"**{a}** pre-match **{p_pre:.0%}** · **{b}** {1 - p_pre:.0%}")
            st.caption("Points in each game played so far (leave the rest at 0). Example: 11 / 7 means A won the game 11-7.")
            games = []
            cols = st.columns(4)
            for i in range(4):
                with cols[i]:
                    ga = st.number_input(f"G{i + 1} {a[:12]}", 0, 40, 0, key=f"live_g{i}a")
                    gb = st.number_input(f"G{i + 1} {b[:12]}", 0, 40, 0, key=f"live_g{i}b")
                    games.append((ga, gb))
            res = LV.predict(p_pre, games, lm)
            k = len([g for g in games if sum(g) > 0])
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"{a} wins", f"{res['p_a']:.0%}", f"{res['p_a'] - p_pre:+.0%} vs pre-match")
            m2.metric(f"{b} wins", f"{res['p_b']:.0%}")
            m3.metric("Games", f"{res['games_a']}–{res['games_b']}", f"{k} played")
            m4.metric("Expected total points", f"{res['exp_total_points']:.0f}", f"{res['pts_so_far']} so far")
            sc = pd.DataFrame({"final score (A–B)": list(res["scores"]), "probability": list(res["scores"].values())})
            st.dataframe(sc.style.format({"probability": "{:.0%}"}), hide_index=True, height=min(300, 40 + 36 * len(sc)))
            st.caption(f"Fair live price: {a} {dk.fair_american(res['p_a'])} · {b} {dk.fair_american(res['p_b'])}. "
                       "Compare with DraftKings' live line; the model knows nothing about injuries, serve or momentum beyond the points.")
            # ---- live spreads from the same state
            st.markdown("**Live spreads** (chance each side covers from here, with the fair price)")
            extra_line = st.number_input("Add a points line DraftKings is showing (e.g. 9.5)", 0.5, 40.5, 9.5, 1.0, key="live_extra_line")
            sp = LV.spreads(res["p_a"], games, point_lines=tuple(sorted({3.5, 5.5, 7.5, float(extra_line)})))
            cur_margin = sum(ga - gb for ga, gb in games)
            rows_ls = []
            def _side(label, pr):
                rows_ls.append({"side": label, "chance": pr, "fair price": fair(pr)})
            for L in (1.5, 2.5):
                pa_, pb_ = sp["games"][L], 1 - sp["games"][-L]
                _side(f"{a} −{L} games", pa_); _side(f"{b} +{L} games", 1 - pa_)
                _side(f"{b} −{L} games", pb_); _side(f"{a} +{L} games", 1 - pb_)
            for L in sorted(sp["points"]):
                if L <= 0:
                    continue
                pa_, pb_ = sp["points"][L], 1 - sp["points"][-L]
                _side(f"{a} −{L} points", pa_); _side(f"{b} +{L} points", 1 - pa_)
                _side(f"{b} −{L} points", pb_); _side(f"{a} +{L} points", 1 - pb_)
            st.dataframe(pd.DataFrame(rows_ls).style.format({"chance": "{:.0%}"}), hide_index=True, height=min(600, 40 + 36 * len(rows_ls)))
            em = sp["exp_point_margin"]
            st.caption(f"Points so far: {a} {cur_margin:+d}. Expected final points margin: {a if em >= 0 else b} by {abs(em):.1f}. "
                       "−L = that player giving L games/points and still winning by more. Game margins are projected at the pair's "
                       "typical ±3.9 points per game, so the points spread is a rough guide late in a close match.")

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
            def _model(model_sig):
                """Cached per model file version, so a retrain by the refresh workflow is picked up without a reboot."""
                from tt_oracle.model import Predictor
                return Predictor.load()
            mdl = _model((config.MODEL_FILE.stat().st_mtime if config.MODEL_FILE.exists() else 0, st.session_state.get("_pkg_stamp")))
            if mdl is None:
                st.caption("Model file not available on this server yet.")
            else:
                from tt_oracle.features import matchup_row
                row = matchup_row(sa.rename_axis(None).to_dict() | {"player": a}, sb.to_dict() | {"player": b}, aw, len(h), sess)
                pr = mdl.predict(row).iloc[0]
                pr["elo_p_a"] = row["elo_p_a"].iloc[0]; pr["h2h_n"] = len(h)
                pr["A_today_n"], pr["B_today_n"] = 0, 0
                pr["session"], pr["venue"], pr["start_time"] = sess, "any venue", pd.Timestamp.now(tz=SITE_TZ).tz_localize(None)
                st.markdown(card_html(pr), unsafe_allow_html=True)
                save_button(pr, "h2h", f"save_h2h_{a}_{b}_{sess}")
                # the same breakdown the Today cards carry, spelled out
                if pd.notna(pr.get("p_g3")):
                    g1, g2, g3 = st.columns(3)
                    g1.metric("3 games", f"{float(pr['p_g3']):.0%}", f"over 3.5: {float(pr['over35']):.0%}")
                    g2.metric("4 games", f"{float(pr['p_g4']):.0%}")
                    g3.metric("5 games", f"{float(pr['p_g5']):.0%}", f"over 4.5: {float(pr['over45']):.0%}")
                    q = float(pr["p_a"]); G = [float(pr["p_g3"]), float(pr["p_g4"]), float(pr["p_g5"])]
                    s3 = q ** 3 / (q ** 3 + (1 - q) ** 3); s4 = q ** 3 * (1 - q) / (q ** 3 * (1 - q) + (1 - q) ** 3 * q)
                    s5 = q ** 3 * (1 - q) ** 2 / (q ** 3 * (1 - q) ** 2 + (1 - q) ** 3 * q ** 2)
                    scores = {"3-0": G[0] * s3, "0-3": G[0] * (1 - s3), "3-1": G[1] * s4, "1-3": G[1] * (1 - s4), "3-2": G[2] * s5, "2-3": G[2] * (1 - s5)}
                    sc = pd.DataFrame({"score (A-B)": list(scores), "probability": list(scores.values())}).sort_values("probability", ascending=False)
                    st.dataframe(sc.style.format({"probability": "{:.0%}"}), hide_index=True, height=250)
                if pd.notna(pr.get("exp_total_points")):
                    t1, t2, t3 = st.columns(3)
                    t1.metric("Model total points", f"{float(pr['exp_total_points']):.0f}")
                    t2.metric(f"{a} wins if takes game 1", f"{float(pr['p_a_if_g1']):.0%}")
                    t3.metric(f"{b} wins if takes game 1", f"{1 - float(pr['p_a_if_lost_g1']):.0%}")
                if pd.notna(pr.get("p_a_pts_m35")):
                    st.markdown("**Spreads** (chance each side covers, with the fair price)")
                    rows_sp = []
                    def _side2(label, prob):
                        rows_sp.append({"side": label, "chance": float(prob), "fair price": fair(prob)})
                    ga_, gb_ = float(pr["p_a_games_m15"]), float(pr["p_b_games_m15"])
                    _side2(f"{a} −1.5 games (wins 3-0 or 3-1)", ga_); _side2(f"{b} +1.5 games", 1 - ga_)
                    _side2(f"{b} −1.5 games (wins 3-0 or 3-1)", gb_); _side2(f"{a} +1.5 games", 1 - gb_)
                    for L in M.Predictor.POINT_LINES:
                        tag = str(L).replace(".", "")
                        pa_, pb_ = float(pr[f"p_a_pts_m{tag}"]), float(pr[f"p_b_pts_m{tag}"])
                        _side2(f"{a} −{L} points", pa_); _side2(f"{b} +{L} points", 1 - pa_)
                        _side2(f"{b} −{L} points", pb_); _side2(f"{a} +{L} points", 1 - pb_)
                    st.dataframe(pd.DataFrame(rows_sp).style.format({"chance": "{:.0%}"}), hide_index=True, height=40 + 36 * len(rows_sp))
                    em = float(pr["exp_point_margin"])
                    st.caption(f"Each −L column is that player giving L points (or games) and still winning by more. "
                               f"Expected total-points margin: {a if em >= 0 else b} by {abs(em):.1f}.")
                st.caption("Model prediction for this matchup if it were played in the chosen session today, both players "
                           "fresh. Fatigue and same-day load are set to zero; the live Today tab uses the real schedule.")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{a} Elo", f"{sa['elo']:.0f}", f"last 10: {sa['win10']:.0%}")
        c2.metric("Head to head", f"{aw}–{len(h) - aw}")
        c3.metric(f"{b} Elo", f"{sb['elo']:.0f}", f"last 10: {sb['win10']:.0%}")
        if not h.empty:
            hh = h.assign(start_time=to_local(h["start_time"]).dt.strftime("%Y-%m-%d %I:%M %p"))
            st.dataframe(hh[["start_time", "session", "venue", "player_a", "sets_a", "sets_b", "player_b"]], hide_index=True)
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
        show["start_time"] = to_local(show["start_time"]).dt.strftime("%a %d %b %I:%M %p")
        st.dataframe(show[["start_time", "session", "match", "pick", "p_pick", "confidence", "dk_odds", "source", "result", "winner"]]
                     .rename(columns={"p_pick": "model"}).style.format({"model": "{:.0%}"}), hide_index=True)
        st.subheader("Manage")
        labels = {r["pick_id"]: f"{r['start_time']} · {r['pick']} vs {r['player_b'] if r['pick'] == r['player_a'] else r['player_a']}"
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
    from tt_oracle.market import BLEND_FILE
    if BLEND_FILE.exists():
        b = json.loads(BLEND_FILE.read_text())
        st.subheader("Model × market blend")
        if b.get("ready"):
            st.write(f"Fitted on **{b['n']}** graded matches with a DraftKings closing line: model alone {b['acc_model']:.1%}, "
                     f"market favourite {b['acc_market']:.1%}, blend {b['acc_blend_insample']:.1%} (in-sample). "
                     f"Weights: model {b['w_model']:.2f}, market {b['w_market']:.2f} — the larger weight is the one the blend trusts more.")
        else:
            st.caption(f"Collecting closing lines: {b.get('n', 0)} graded matches have one so far (needs 300). Run scripts/push_lines.py on match days to feed it.")
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
        if "points" in s:
            ps = s["points"]
            st.subheader("Points feed: totals and game-1 conditionals")
            q1, q2, q3 = st.columns(3)
            q1.metric("Total points error (avg)", f"{ps['total_points_mae']:.1f}", f"typical total {ps['total_points_mean']:.0f}")
            q2.metric("Accuracy after game 1", f"{ps['after_game1_accuracy']:.1%}", f"{ps['after_game1_accuracy'] - ps['pre_match_accuracy_same']:+.1%} vs pre-match")
            q3.metric("Matches with points", ps["n"])
        if "spreads" in s:
            sp = s["spreads"]
            st.subheader("Spreads (favourite −1.5 games, favourite −3.5 points)")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Fav −1.5 games covers", f"{sp['games_m15_fav_cover_rate']:.1%}", f"model said {sp['games_m15_fav_pred']:.1%}")
            r2.metric("…when model ≥60%", f"{sp['games_m15_hit_when_60plus']:.1%}", f"{sp['games_m15_calls_60plus']:,} calls")
            r3.metric("Fav −3.5 points covers", f"{sp['points_m35_fav_cover_rate']:.1%}", f"model said {sp['points_m35_fav_pred']:.1%}")
            r4.metric("…when model ≥60%", f"{sp['points_m35_hit_when_60plus']:.1%}", f"{sp['points_m35_calls_60plus']:,} calls")
            st.caption(f"Average error on the total-points margin: {sp['point_margin_mae']:.1f} points over {sp['n']:,} matches.")
        if "games" in s:
            gs = s["games"]
            st.subheader("Number of games (total games 3.5 / 4.5)")
            st.caption("How many games a match goes is only weakly predictable: the model's edge over 'assume the long-run split' "
                       "is small, but its percentages are calibrated, so they are useful for spotting a mispriced total, not as a standalone pick.")
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Matches", gs["n"])
            g2.metric("Model log loss", f"{gs['log_loss']:.4f}", f"{gs['prior_log_loss'] - gs['log_loss']:+.4f} vs long-run split", delta_color="normal")
            g3.metric("Over 3.5 when ≥70%", f"{gs['over35_hit_when_called']:.0%}" if gs["over35_calls"] else "–", f"{gs['over35_calls']} calls")
            g4.metric("Under 4.5 when ≤25% five", f"{gs['under45_hit_when_called']:.0%}" if gs["under45_calls"] else "–", f"{gs['under45_calls']} calls")
            st.caption(f"Actual split 3/4/5 games: {' / '.join(f'{x:.0%}' for x in gs['actual_3_4_5'])} · model's average: {' / '.join(f'{x:.0%}' for x in gs['predicted_3_4_5'])}")
            st.dataframe(pipeline.games_calibration(bt).style.format({"pred_3": "{:.1%}", "actual_3": "{:.1%}", "pred_5": "{:.1%}", "actual_5": "{:.1%}"}), hide_index=True)

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
