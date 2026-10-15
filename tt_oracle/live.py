"""
In-match ("live") model: given the pre-match probability and the games played so far (with their point scores),
the updated chance of each player winning, the likely final score, and the expected total points.

Fitted on the vendor's game-by-game points for every match with a pre-match rating. One logistic model per
number of games played (1..4), on:
    z            logit of the calibrated pre-match probability for player A
    lead         games won by A minus games won by B
    net          A's net point margin over the games played so far
    last         A's point margin in the most recent game (momentum)
The final-score distribution is derived from the updated win probability: a per-game probability q is chosen so
that the best-of-5 from the current state gives exactly that match probability, then every path to 3 wins is
enumerated. Expected total points = points so far + expected remaining games x the pair's typical points/game.

Coefficients live in models/live_model.json (small); `fit` is run by the refresh after each points update.
"""
from __future__ import annotations

import json
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import config, store

LIVE_FILE = config.MODELS / "live_model.json"


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _state(games: list, k: int) -> tuple[int, int, int]:
    g = games[:k]
    lead = sum(1 for a, b in g if a > b) - sum(1 for a, b in g if b > a)
    net = sum(a - b for a, b in g)
    last = g[-1][0] - g[-1][1]
    return lead, net, last


def fit(points: pd.DataFrame | None = None, feat: pd.DataFrame | None = None) -> dict:
    """points: data/points.csv rows (linked); feat: feature frame with match_id, elo_p_a, y."""
    points = store.read(config.DATA / "points.csv") if points is None else points
    points = points[points["match_id"].notna()]
    if feat is None:
        from .pipeline import features_now
        feat = features_now()
    d = points.merge(feat[["match_id", "elo_p_a", "y"]], on="match_id").dropna(subset=["elo_p_a", "y"])
    d = d.drop_duplicates("match_id")
    d["g"] = d["games"].map(json.loads)
    d = d[d["g"].map(len).between(3, 5)]
    if len(d) < 2000:
        return {}
    # 1. calibrate the pre-match rating so z is on a trustworthy scale
    cal = LogisticRegression(C=100.0).fit(_logit(d["elo_p_a"]).reshape(-1, 1), d["y"].astype(int))
    z = cal.decision_function(_logit(d["elo_p_a"]).reshape(-1, 1))
    out = {"calibration": {"a": float(cal.coef_[0][0]), "b": float(cal.intercept_[0])}, "n": int(len(d)), "states": {}}
    # 2. one model per number of games played, only matches that were still alive at that point
    for k in (1, 2, 3, 4):
        alive = d[d["g"].map(len) > k]
        if len(alive) < 500:
            continue
        S = np.array([_state(g, k) for g in alive["g"]], float)
        zz = z[d.index.get_indexer(alive.index)]
        X = np.column_stack([zz, S[:, 0], S[:, 1], S[:, 2]])
        lr = LogisticRegression(C=10.0, max_iter=2000).fit(X, alive["y"].astype(int))
        p = lr.predict_proba(X)[:, 1]
        acc = float(((p >= 0.5) == (alive["y"] == 1)).mean())
        out["states"][str(k)] = {"coef": [float(c) for c in lr.coef_[0]], "intercept": float(lr.intercept_[0]),
                                 "n": int(len(alive)), "accuracy": acc}
    # typical points per game, to extend totals
    allg = [ab for g in d["g"] for ab in g]
    out["points_per_game"] = float(np.mean([a + b for a, b in allg]))
    config.ensure_dirs()
    LIVE_FILE.write_text(json.dumps(out, indent=1))
    return out


def load() -> dict | None:
    return json.loads(LIVE_FILE.read_text()) if LIVE_FILE.exists() else None


def _match_prob_from_q(q: float, wa: int, wb: int) -> float:
    """P(A reaches 3 before B) from (wa, wb) with per-game prob q."""
    from functools import lru_cache

    @lru_cache(None)
    def f(a, b):
        if a == 3:
            return 1.0
        if b == 3:
            return 0.0
        return q * f(a + 1, b) + (1 - q) * f(a, b + 1)
    return f(wa, wb)


def _solve_q(p: float, wa: int, wb: int) -> float:
    lo, hi = 0.01, 0.99
    for _ in range(40):
        mid = (lo + hi) / 2
        if _match_prob_from_q(mid, wa, wb) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def score_distribution(p: float, wa: int, wb: int) -> dict:
    """{final score 'x-y' (A-B): probability} from the current game count."""
    q = _solve_q(p, wa, wb)
    out = {}
    need_a, need_b = 3 - wa, 3 - wb
    for extra in range(0, need_b):                          # A wins 3-(wb+extra)
        # A wins last game; before it, need_a-1 wins for A and `extra` for B in any order
        from math import comb
        pr = comb(need_a - 1 + extra, extra) * q ** need_a * (1 - q) ** extra
        out[f"3-{wb + extra}"] = out.get(f"3-{wb + extra}", 0) + pr
    for extra in range(0, need_a):
        from math import comb
        pr = comb(need_b - 1 + extra, extra) * (1 - q) ** need_b * q ** extra
        out[f"{wa + extra}-3"] = out.get(f"{wa + extra}-3", 0) + pr
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def predict(p_pre: float, games: list, model: dict | None = None) -> dict:
    """p_pre: pre-match P(A wins) (the model's p_a); games: [(a_pts, b_pts), ...] played so far (0-4 games)."""
    model = load() if model is None else model
    games = [(int(a), int(b)) for a, b in games if a is not None and b is not None and (a + b) > 0]
    wa = sum(1 for a, b in games if a > b); wb = sum(1 for a, b in games if b > a)
    k = len(games)
    if wa >= 3 or wb >= 3:
        p = 1.0 if wa >= 3 else 0.0
    elif k == 0 or not model or str(k) not in model.get("states", {}):
        p = float(p_pre)
    else:
        st = model["states"][str(k)]
        z = float(_logit(p_pre))                                # the model's p_a is already calibrated
        lead, net, last = _state(games, k)
        x = np.array([z, lead, net, last], float)
        p = float(1 / (1 + np.exp(-(np.dot(st["coef"], x) + st["intercept"]))))
        p = min(max(p, 0.005), 0.995)
    dist = score_distribution(p, wa, wb) if (wa < 3 and wb < 3) else {f"{wa}-{wb}": 1.0}
    exp_games_left = sum((int(s.split("-")[0]) + int(s.split("-")[1]) - k) * pr for s, pr in dist.items())
    ppg = (model or {}).get("points_per_game", 19.5)
    pts_so_far = sum(a + b for a, b in games)
    return {"p_a": p, "p_b": 1 - p, "games_a": wa, "games_b": wb, "scores": dist,
            "exp_total_points": pts_so_far + exp_games_left * ppg, "exp_games": k + exp_games_left,
            "pts_so_far": pts_so_far}
