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
                 "career_n", "h2h_net", "session_win", "venue_win", "streak"]
CONTEXT_FEATURES = ["sess_morning", "sess_afternoon", "sess_evening", "sess_night", "both_n_min"]
SESS = config.SESSIONS


def _expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


class _P:
    __slots__ = ("elo", "hist", "elo_hist", "today", "last_time", "n", "sess", "venue", "streak")

    def __init__(self):
        self.elo = 1500.0
        self.hist = deque(maxlen=60)       # (time, won, sets_won, sets_lost)
        self.elo_hist = deque(maxlen=11)
        self.today = (None, 0)             # (date, matches played today)
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
    today_n = p.today[1] if p.today[0] == now.normalize() else 0
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
    }


def build_features(matches: pd.DataFrame, elo_k: float | None = None) -> pd.DataFrame:
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
            for p, won, sw, sl, d in ((pa, won_a, r.sets_a, r.sets_b, delta), (pb, not won_a, r.sets_b, r.sets_a, -delta)):
                p.elo_hist.append(p.elo)
                p.elo += d
                p.hist.append((now, 1.0 if won else 0.0, int(sw), int(sl)))
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
            p.today = (now.normalize(), (p.today[1] + 1) if p.today[0] == now.normalize() else 1)
            p.last_time = now
    return pd.DataFrame(rows)


def player_snapshot(feat: pd.DataFrame) -> pd.DataFrame:
    """Latest known rating/form per player (from the most recent row each appears in)."""
    a = feat[["player_a", "start_time", "A_elo", "A_win10", "A_career_n"]].rename(
        columns={"player_a": "player", "A_elo": "elo", "A_win10": "win10", "A_career_n": "n"})
    b = feat[["player_b", "start_time", "B_elo", "B_win10", "B_career_n"]].rename(
        columns={"player_b": "player", "B_elo": "elo", "B_win10": "win10", "B_career_n": "n"})
    s = pd.concat([a, b]).sort_values("start_time").groupby("player").last().reset_index()
    return s.sort_values("elo", ascending=False)
