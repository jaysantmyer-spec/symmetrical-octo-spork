"""
The market as a feature.

Every DraftKings snapshot pushed to data/odds_history.csv (by scripts/push_lines.py on your computer, or by the
app) is matched to our matches here; the LAST line seen before the start is the closing line. Once enough
matches have closing lines (MIN_ROWS), a small logistic blend is fitted on the ledger-style question
"given the model's probability and the market's, who won?" and used for every match that has a line.

Outputs on each prediction with a line: p_blend (model x market), and the blend's learned weights are shown
on the Track record tab. The blend never changes p_a itself, so the ledger and backtest stay comparable.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import config, store
from .dk import _key_set, _same

MIN_ROWS = 300
CLOSING_CSV = config.DATA / "closing_lines.csv"
BLEND_FILE = config.MODELS / "market_blend.json"


def closing_lines(matches: pd.DataFrame, hist: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per match_id: last decimal odds seen before the start (either order of names)."""
    hist = store.read(config.ODDS_HISTORY_CSV) if hist is None else hist
    if hist is None or hist.empty or matches.empty:
        return pd.DataFrame(columns=["match_id", "close_odds_a", "close_odds_b", "close_p_a", "close_fetched_at"])
    h = hist.dropna(subset=["odds_1", "odds_2"]).copy()
    h["fetched_at"] = pd.to_datetime(h["fetched_at"], errors="coerce")
    h["commence_time"] = pd.to_datetime(h["commence_time"], errors="coerce")
    h["k1"], h["k2"] = h["player_1"].map(_key_set), h["player_2"].map(_key_set)
    m = matches.copy()
    m["start_time"] = pd.to_datetime(m["start_time"])
    m["day"] = m["start_time"].dt.normalize()
    h["day"] = h["commence_time"].dt.normalize().fillna(h["fetched_at"].dt.normalize())
    rows = []
    for day, hd in h.groupby("day"):
        md = m[(m["day"] >= day - pd.Timedelta(days=1)) & (m["day"] <= day + pd.Timedelta(days=1))]
        if md.empty:
            continue
        for _, r in md.iterrows():
            ka, kb = _key_set(r["player_a"]), _key_set(r["player_b"])
            same = hd[hd["k1"].map(lambda x: _same(ka, x)) & hd["k2"].map(lambda x: _same(kb, x))]
            flip = hd[hd["k1"].map(lambda x: _same(kb, x)) & hd["k2"].map(lambda x: _same(ka, x))]
            c = pd.concat([same.assign(_oa=same["odds_1"], _ob=same["odds_2"]), flip.assign(_oa=flip["odds_2"], _ob=flip["odds_1"])])
            c = c[c["fetched_at"] <= r["start_time"]]
            if "commence_time" in c and c["commence_time"].notna().any():
                c = c[(c["commence_time"].isna()) | ((c["commence_time"] - r["start_time"]).abs() <= pd.Timedelta(hours=3))]
            if c.empty:
                continue
            last = c.sort_values("fetched_at").iloc[-1]
            ia, ib = 1 / float(last["_oa"]), 1 / float(last["_ob"])
            rows.append({"match_id": r["match_id"], "close_odds_a": float(last["_oa"]), "close_odds_b": float(last["_ob"]),
                         "close_p_a": ia / (ia + ib), "close_fetched_at": last["fetched_at"]})
    out = pd.DataFrame(rows).drop_duplicates("match_id", keep="last")
    store.write(out, CLOSING_CSV)
    return out


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_blend(ledger: pd.DataFrame, closing: pd.DataFrame) -> dict | None:
    """Logistic on [logit(model), logit(market)] over graded ledger rows that have a closing line."""
    if ledger.empty or closing.empty:
        return None
    g = ledger[ledger["y"].notna()].merge(closing, on="match_id")
    if len(g) < MIN_ROWS:
        return {"n": int(len(g)), "ready": False}
    X = np.column_stack([_logit(g["p_a"]), _logit(g["close_p_a"])])
    y = g["y"].astype(int).to_numpy()
    lr = LogisticRegression(C=10.0, max_iter=2000).fit(X, y)
    acc_model = float(((g["p_a"] >= 0.5) == (y == 1)).mean()); acc_mkt = float(((g["close_p_a"] >= 0.5) == (y == 1)).mean())
    pb = lr.predict_proba(X)[:, 1]
    out = {"n": int(len(g)), "ready": True, "w_model": float(lr.coef_[0][0]), "w_market": float(lr.coef_[0][1]),
           "bias": float(lr.intercept_[0]), "acc_model": acc_model, "acc_market": acc_mkt,
           "acc_blend_insample": float(((pb >= 0.5) == (y == 1)).mean()), "fitted_at": pd.Timestamp.now().isoformat(timespec="seconds")}
    config.ensure_dirs()
    BLEND_FILE.write_text(json.dumps(out, indent=1))
    return out


def load_blend() -> dict | None:
    if BLEND_FILE.exists():
        b = json.loads(BLEND_FILE.read_text())
        return b if b.get("ready") else None
    return None


def apply_blend(P: pd.DataFrame, blend: dict | None = None) -> pd.DataFrame:
    """Add p_blend where a market probability is present (market_p_a from attach_market)."""
    blend = load_blend() if blend is None else blend
    out = P.copy()
    out["p_blend"] = np.nan
    if not blend or "market_p_a" not in out:
        return out
    has = out["market_p_a"].notna()
    if has.any():
        z = blend["bias"] + blend["w_model"] * _logit(out.loc[has, "p_a"]) + blend["w_market"] * _logit(out.loc[has, "market_p_a"])
        out.loc[has, "p_blend"] = 1 / (1 + np.exp(-z))
    return out
