"""
Leak-free features for TT Elite Series matches.

Everything is computed by walking the matches in start-time order, so a match's features only use
matches that finished before it started. Players appear in both corners, so training rows are
mirrored (A/B swapped) and predictions are averaged over both orders.

Per player, before each match:
  elo, elo_form (Elo change over the last 10 matches), wins/matches over last 10 and last 30 days,
  set ratio over last 30 matches, matches already played today, hours since last match,
  days since last match, career matches, h2h record vs opponent, same-session win rate.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from . import config

DIFF_FEATURES = ["elo", "elo_form", "win10", "win30d", "setr30", "today_n", "hrs_since", "days_since",
                 "career_n", "h2h_net", "session_win", "venue_win", "streak", "elo_set", "bt"]
# elo_set: a second Elo updated set by set (a 3-2 and a 3-0 move it differently and it learns faster)
# bt: rolling Bradley-Terry strength (see add_bt), refit weekly on the trailing 180 days.
# Tested and left out (no out-of-sample gain, Oct 2026): opponent-adjusted form (perf10), strength of schedule
# (opp_elo10), wins / sets lost so far today (today_w, today_sl). They are still computed in _side_feats and
# can be re-added here.
CONTEXT_FEATURES = ["sess_morning", "sess_afternoon", "sess_evening", "sess_night", "both_n_min"]
SESS = config.SESSIONS


def _expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


class _P:
    __slots__ = ("elo", "elo_set", "hist", "elo_hist", "today", "last_time", "n", "sess", "venue", "streak")

    def __init__(self):
        self.elo = 1500.0
        self.elo_set = 1500.0
        self.hist = deque(maxlen=60)       # (time, won, sets_won, sets_lost, expected, opp_elo)
        self.elo_hist = deque(maxlen=11)
        self.today = (None, 0, 0, 0)       # (date, matches played today, wins today, sets lost today)
        self.last_time = None
        self.n = 0
        self.sess = defaultdict(lambda: [0, 0])
        self.venue = defaultdict(lambda: [0, 0])
        self.streak = 0


def _side_feats(p: _P, now: pd.Timestamp, session: str, venue: str, win_mean_default=0.5) -> dict:
    h = list(p.hist)
    last10 = h[-10:]
    cutoff30 = now - pd.Timedelta(days=30)
    last30d = [x for x in h if x[0] >= cutoff30]
    sets = h[-30:]
    sw, sl = sum(x[2] for x in sets), sum(x[3] for x in sets)
    is_today = p.today[0] == now.normalize()
    today_n, today_w, today_sl = (p.today[1], p.today[2], p.today[3]) if is_today else (0, 0, 0)
    hrs = (now - p.last_time).total_seconds() / 3600 if p.last_time is not None else 96.0
    s = p.sess[session]
    v = p.venue[venue]
    return {
        "elo": p.elo,
        "elo_form": (p.elo - p.elo_hist[0]) if len(p.elo_hist) >= 2 else 0.0,
        "win10": np.mean([x[1] for x in last10]) if last10 else win_mean_default,
        "win30d": np.mean([x[1] for x in last30d]) if last30d else win_mean_default,
        "setr30": sw / (sw + sl) if (sw + sl) else 0.5,
        "today_n": today_n,
        "hrs_since": min(hrs, 96.0),
        "days_since": min(hrs / 24.0, 30.0),
        "career_n": p.n,
        "session_win": (s[0] + 1) / (s[1] + 2),
        "venue_win": (v[0] + 1) / (v[1] + 2),
        "streak": p.streak,
        "elo_set": p.elo_set,
        "perf10": np.mean([x[1] - x[4] for x in last10]) if last10 else 0.0,
        "opp_elo10": np.mean([x[5] for x in last10]) if last10 else 1500.0,
        "today_w": today_w,
        "today_sl": today_sl,
    }


def build_features(matches: pd.DataFrame, elo_k: float | None = None, with_bt: bool = True) -> pd.DataFrame:
    """matches: parsed rows (finished or not). Returns one row per match with A/B features, diff features,
    and y (1 if A won, NaN if unfinished)."""
    k = config.DEFAULTS["elo_k"] if elo_k is None else elo_k
    m = matches.copy()
    m["start_time"] = pd.to_datetime(m["start_time"])
    m = m.sort_values(["start_time", "post_id", "order"]).reset_index(drop=True)
    players: dict[str, _P] = defaultdict(_P)
    h2h: dict[tuple, list] = defaultdict(lambda: [0, 0])  # (a,b) sorted -> [wins of first, total]
    rows = []
    for r in m.itertuples(index=False):
        a, b, now = r.player_a, r.player_b, r.start_time
        pa, pb = players[a], players[b]
        sess, venue = str(r.session or ""), str(r.venue or "")
        fa, fb = _side_feats(pa, now, sess, venue), _side_feats(pb, now, sess, venue)
        key = tuple(sorted((a, b)))
        hw, hn = h2h[key]
        a_h2h_w = hw if key[0] == a else hn - hw
        fa["h2h_net"], fb["h2h_net"] = (2 * a_h2h_w - hn), (hn - 2 * a_h2h_w)
        row = {"match_id": r.match_id, "post_id": r.post_id, "date": r.date, "start_time": now,
               "session": sess, "venue": venue, "player_a": a, "player_b": b,
               "finished": bool(r.finished), "sets_a": r.sets_a, "sets_b": r.sets_b,
               "y": (1.0 if r.winner == "a" else 0.0 if r.winner == "b" else np.nan),
               "elo_p_a": _expected(pa.elo, pb.elo), "h2h_n": hn}
        for kf in DIFF_FEATURES:
            if kf == "bt":
                continue                                   # filled by add_bt after the walk
            row[f"A_{kf}"], row[f"B_{kf}"] = fa[kf], fb[kf]
            row[f"d_{kf}"] = fa[kf] - fb[kf]
        for s_ in SESS:
            row[f"sess_{s_}"] = 1.0 if sess == s_ else 0.0
        row["both_n_min"] = min(pa.n, pb.n)
        rows.append(row)

        if r.finished and r.winner in ("a", "b"):
            won_a = r.winner == "a"
            ea = _expected(pa.elo, pb.elo)
            margin = 1.0 + 0.25 * abs(int(r.sets_a) - int(r.sets_b))  # 3-0 moves Elo more than 3-2
            delta = k * margin * ((1.0 if won_a else 0.0) - ea)
            # set-level Elo: every set is a trial at K/4 against the set-Elo expectation
            es = _expected(pa.elo_set, pb.elo_set)
            ds = (k / 4.0) * ((int(r.sets_a) * (1 - es)) - (int(r.sets_b) * es))
            for p, won, sw, sl, d, exp_, opp, dset in ((pa, won_a, r.sets_a, r.sets_b, delta, ea, pb.elo, ds),
                                                     (pb, not won_a, r.sets_b, r.sets_a, -delta, 1 - ea, pa.elo, -ds)):
                p.elo_hist.append(p.elo)
                p.elo += d
                p.elo_set += dset
                p.hist.append((now, 1.0 if won else 0.0, int(sw), int(sl), exp_, opp))
                tw, tsl = (p.today[2], p.today[3]) if p.today[0] == now.normalize() else (0, 0)
                p.today = (now.normalize(), p.today[1] if p.today[0] == now.normalize() else 0, tw + int(won), tsl + int(sl))
                p.n += 1
                p.sess[sess][0] += int(won); p.sess[sess][1] += 1
                p.venue[venue][0] += int(won); p.venue[venue][1] += 1
                p.streak = (p.streak + 1 if p.streak >= 0 else 1) if won else (p.streak - 1 if p.streak <= 0 else -1)
            if key[0] == a:
                h2h[key][0] += int(won_a)
            else:
                h2h[key][0] += int(not won_a)
            h2h[key][1] += 1
        # fatigue counters update at start time (an upcoming match also counts toward today's load)
        for p in (pa, pb):
            same = p.today[0] == now.normalize()
            p.today = (now.normalize(), (p.today[1] + 1) if same else 1, p.today[2] if same else 0, p.today[3] if same else 0)
            p.last_time = now
    out = pd.DataFrame(rows)
    return add_bt(out) if with_bt and len(out) else out


def player_snapshot(feat: pd.DataFrame) -> pd.DataFrame:
    """Latest known rating/form per player (from the most recent row each appears in), with every
    per-side feature so the app can score any hypothetical matchup without rebuilding features."""
    cols = {k: k for k in DIFF_FEATURES}
    a = feat[["player_a", "start_time"] + [f"A_{k}" for k in DIFF_FEATURES]].rename(
        columns={"player_a": "player", **{f"A_{k}": k for k in DIFF_FEATURES}})
    b = feat[["player_b", "start_time"] + [f"B_{k}" for k in DIFF_FEATURES]].rename(
        columns={"player_b": "player", **{f"B_{k}": k for k in DIFF_FEATURES}})
    s = pd.concat([a, b]).sort_values("start_time").groupby("player").last().reset_index()
    s = s.rename(columns={"career_n": "n"})
    return s.sort_values("elo", ascending=False)


def matchup_row(sa: pd.Series, sb: pd.Series, h2h_a_wins: int, h2h_n: int, session: str, now=None) -> pd.DataFrame:
    """One feature row for a hypothetical A vs B, from two player snapshots (as used by the app)."""
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    row = {"match_id": "h2h", "post_id": 0, "date": now.normalize(), "start_time": now, "session": session,
           "venue": "", "player_a": sa["player"], "player_b": sb["player"], "finished": False, "y": np.nan,
           "elo_p_a": _expected(float(sa["elo"]), float(sb["elo"])), "h2h_n": h2h_n}
    for side, s in (("A", sa), ("B", sb)):
        for k in DIFF_FEATURES:
            if k == "career_n":
                v = float(s["n"])
            elif k == "h2h_net":
                v = (2 * h2h_a_wins - h2h_n) if side == "A" else (h2h_n - 2 * h2h_a_wins)
            elif k == "bt":
                v = float(s["bt"]) if "bt" in s and pd.notna(s["bt"]) else 0.0
            elif k in ("today_n", "today_w", "today_sl"):
                v = 0.0
            elif k in ("hrs_since", "days_since"):
                hrs = (now - pd.Timestamp(s["start_time"])).total_seconds() / 3600
                v = min(hrs, 96.0) if k == "hrs_since" else min(hrs / 24.0, 30.0)
            else:
                v = float(s[k]) if k in s and pd.notna(s[k]) else {"elo_set": float(s["elo"]), "opp_elo10": 1500.0}.get(k, 0.0)
            row[f"{side}_{k}"] = v
    for k in DIFF_FEATURES:
        row[f"d_{k}"] = row[f"A_{k}"] - row[f"B_{k}"]
    for s_ in SESS:
        row[f"sess_{s_}"] = 1.0 if session == s_ else 0.0
    row["both_n_min"] = min(float(sa["n"]), float(sb["n"]))
    return pd.DataFrame([row])


# --------------------------------------------------------------------------- Bradley-Terry rating (rolling, leak-free)
def bt_fit(hist: pd.DataFrame, ref: pd.Timestamp, half_life: float = 60.0, C: float = 1.0) -> pd.Series:
    """Ridge-regularised Bradley-Terry strength per player from `hist` (finished matches before `ref`): a logistic
    regression on +1/-1 player indicators with every set as a trial, recency-weighted. Dense round-robin leagues
    reward this over Elo because every match re-fits every player's strength jointly."""
    import scipy.sparse as sp
    from sklearn.linear_model import LogisticRegression
    hist = hist[hist["sets_a"].notna() & hist["sets_b"].notna()]
    if len(hist) < 200:
        return pd.Series(dtype=float)
    players = pd.Index(pd.concat([hist["player_a"], hist["player_b"]]).unique())
    ia, ib = players.get_indexer(hist["player_a"]), players.get_indexer(hist["player_b"])
    n = len(hist)
    X = sp.csr_matrix((np.r_[np.ones(n), -np.ones(n)], (np.r_[np.arange(n), np.arange(n)], np.r_[ia, ib])), shape=(n, len(players)))
    age = (ref - hist["start_time"]).dt.total_seconds().to_numpy() / 86400
    w = 0.5 ** (age / half_life)
    sa, sb = hist["sets_a"].to_numpy(float), hist["sets_b"].to_numpy(float)
    Xs = sp.vstack([X, X]); y = np.r_[np.ones(n), np.zeros(n)]; ww = np.r_[w * sa, w * sb]
    keep = ww > 0
    lr = LogisticRegression(C=C, fit_intercept=False, max_iter=500).fit(Xs[keep], y[keep], sample_weight=ww[keep])
    return pd.Series(lr.coef_[0], index=players)


def add_bt(feat: pd.DataFrame, refit_days: int = 7, window_days: int = 180, progress=None) -> pd.DataFrame:
    """Add A_bt / B_bt / d_bt / bt_p_a to a feature frame. Ratings are refit every `refit_days` on the trailing
    `window_days` of finished matches strictly before the refit date, and applied to the matches that follow it."""
    f = feat.sort_values("start_time").copy()
    f["A_bt"], f["B_bt"] = 0.0, 0.0
    fin = f[f["finished"].astype(bool) & f["y"].notna()]
    t0 = f["start_time"].min().normalize() + pd.Timedelta(days=refit_days)
    edges = list(pd.date_range(t0, f["start_time"].max() + pd.Timedelta(days=refit_days), freq=f"{refit_days}D"))
    for i, lo in enumerate(edges):
        hi = lo + pd.Timedelta(days=refit_days)
        sel = (f["start_time"] >= lo) & (f["start_time"] < hi)
        if not sel.any():
            continue
        if progress and i % 20 == 0:
            progress(i / len(edges), f"Bradley-Terry refit {lo.date()}")
        r = bt_fit(fin[(fin["start_time"] < lo) & (fin["start_time"] >= lo - pd.Timedelta(days=window_days))], lo)
        if r.empty:
            continue
        f.loc[sel, "A_bt"] = f.loc[sel, "player_a"].map(r).fillna(0.0).to_numpy()
        f.loc[sel, "B_bt"] = f.loc[sel, "player_b"].map(r).fillna(0.0).to_numpy()
    f["d_bt"] = f["A_bt"] - f["B_bt"]
    f["bt_p_a"] = 1.0 / (1.0 + np.exp(-f["d_bt"]))
    return f.sort_index()
