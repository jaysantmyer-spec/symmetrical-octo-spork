"""
Pipeline: load data -> features -> train / predict / backtest / ledger -> learning cycle.

    python -m tt_oracle train
    python -m tt_oracle predict            # writes data/predictions.csv for upcoming matches (and logs them)
    python -m tt_oracle backtest --days 60
    python -m tt_oracle cycle              # update data -> grade ledger -> re-learn -> retrain -> predict
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config, model as M, store
from .features import build_features

log = logging.getLogger(__name__)


def _ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def load_matches() -> pd.DataFrame:
    m = store.read(config.MATCHES_CSV)
    if m.empty:
        return m
    m["start_time"] = pd.to_datetime(m["start_time"])
    m["finished"] = m["finished"].astype(bool)
    return m


def features_now() -> pd.DataFrame:
    m = load_matches()
    return build_features(m) if not m.empty else pd.DataFrame()


# --------------------------------------------------------------------------- train / predict
def retrain(state: dict | None = None) -> M.Predictor:
    state = state or M.load_state()
    feat = features_now()
    hardness = _hardness()
    pred = M.train(feat, state.get("learners"), half_life_days=state["half_life_days"], hardness=hardness,
                   hardness_alpha=state["hardness_alpha"], weights=state.get("weights"),
                   temperature=state.get("temperature", 1.0))
    pred.weights = {k: v for k, v in pred.weights.items() if k in pred.models}
    pred.save()
    state["learners"] = pred.learners
    M.save_state(state)
    return pred


def upcoming(feat: pd.DataFrame, horizon_days: int = 2) -> pd.DataFrame:
    now = pd.Timestamp.now()
    up = feat[(~feat["finished"]) & (feat["start_time"] >= now - pd.Timedelta(hours=3))
              & (feat["start_time"] <= now + pd.Timedelta(days=horizon_days))]
    return up.sort_values("start_time")


def predict_upcoming(pred: M.Predictor | None = None, log_to_ledger: bool = True) -> pd.DataFrame:
    pred = pred or M.Predictor.load()
    if pred is None:
        raise RuntimeError("no model; run train first")
    feat = features_now()
    up = upcoming(feat)
    P = pred.predict(up)
    if not P.empty:
        from .dk import attach_market
        P = attach_market(P)
        store.write(P, config.PREDICTIONS_CSV)
        if log_to_ledger:
            log_predictions(P, pred.meta.get("version", ""))
    from .features import player_snapshot
    store.write(player_snapshot(feat), config.PLAYERS_CSV)
    return P


# --------------------------------------------------------------------------- ledger
def log_predictions(P: pd.DataFrame, version: str) -> int:
    """Save predictions for matches that haven't started, once per match (first prediction wins)."""
    led = store.read(config.LEDGER_CSV)
    have = set(led["match_id"]) if not led.empty else set()
    rows = P[~P["match_id"].isin(have) & (P["start_time"] > pd.Timestamp.now())]
    if rows.empty:
        return 0
    cols = ["match_id", "post_id", "date", "start_time", "player_a", "player_b", "p_a", "pick", "pick_prob",
            "confidence", "thin_history"] + [c for c in rows.columns if c.startswith("comp_")] + \
           [c for c in ("market_p_a", "dk_odds_a", "dk_odds_b") if c in rows.columns]
    out = rows[cols].copy()
    out["model_version"] = version
    out["logged_at"] = pd.Timestamp.now()
    out["y"] = np.nan
    out["graded_at"] = pd.NaT
    store.upsert(out, config.LEDGER_CSV, "match_id")
    return int(len(out))


def grade_ledger() -> dict:
    led = store.read(config.LEDGER_CSV)
    if led.empty:
        return {"graded_now": 0, "graded_total": 0}
    m = load_matches()
    res = m[m["finished"]].set_index("match_id")["winner"]
    todo = led["y"].isna() & led["match_id"].isin(res.index)
    led.loc[todo, "y"] = led.loc[todo, "match_id"].map(res).map({"a": 1.0, "b": 0.0})
    led["graded_at"] = pd.to_datetime(led["graded_at"], errors="coerce")
    led.loc[todo, "graded_at"] = pd.Timestamp.now().floor("s")
    store.write(led, config.LEDGER_CSV)
    return {"graded_now": int(todo.sum()), "graded_total": int(led["y"].notna().sum())}


def _hardness() -> dict:
    led = store.read(config.LEDGER_CSV)
    if led.empty:
        return {}
    g = led[led["y"].notna()]
    return dict(zip(g["match_id"], np.abs(g["y"].astype(float) - g["p_a"].astype(float))))


def relearn(state: dict | None = None, min_graded: int = 300) -> dict:
    """Update Hedge weights and temperature from graded, pre-match predictions."""
    state = state or M.load_state()
    led = store.read(config.LEDGER_CSV)
    g = led[led["y"].notna()] if not led.empty else led
    if len(g) < min_graded:
        state.setdefault("history", []).append({"at": str(pd.Timestamp.now()), "action": "update_skipped",
                                                "reason": f"only {len(g)} graded predictions (need {min_graded})"})
        M.save_state(state)
        return {"updated": False, "graded": int(len(g))}
    recent = g.tail(config.DEFAULTS["hedge_window"])
    comp = recent[[c for c in recent.columns if c.startswith("comp_")]].copy()
    comp.columns = [c[5:] for c in comp.columns]
    comp = comp.dropna(axis=1, how="all")
    y = recent["y"].astype(float)
    state["weights"] = M.hedge_weights(comp, y, config.DEFAULTS["hedge_eta"])
    state["temperature"] = M.fit_temperature(M.blend(comp, state["weights"]), y)
    acc = float(((recent["p_a"].astype(float) >= 0.5) == (y == 1)).mean())
    state.setdefault("history", []).append({"at": str(pd.Timestamp.now()), "action": "relearn",
                                            "graded": int(len(g)), "accuracy_recent": round(acc, 4),
                                            "weights": state["weights"], "temperature": state["temperature"]})
    M.save_state(state)
    return {"updated": True, "graded": int(len(g)), "weights": state["weights"], "temperature": state["temperature"]}


def run_cycle(progress=None, scrape: bool = True, odds: bool = True) -> dict:
    out = {}
    if scrape:
        from .scraper import update
        out["scrape"] = update(pages=3, progress=progress)
    if odds:
        try:
            from .dk import snapshot_odds
            out["odds"] = snapshot_odds()
        except Exception as ex:
            out["odds"] = f"failed: {ex}"
    out["grade"] = grade_ledger()
    out["learn"] = relearn()
    pred = retrain()
    out["model"] = pred.meta
    P = predict_upcoming(pred)
    out["predicted"] = int(len(P))
    return out


# --------------------------------------------------------------------------- backtest
def walk_forward(feat: pd.DataFrame, start, end=None, step_days: int = 7, learners=None, adaptive: bool = True,
                 progress=None) -> pd.DataFrame:
    """Retrain every `step_days` using only earlier matches, predict the next window. Honest by construction."""
    data = feat[feat["y"].notna() & (feat["both_n_min"] >= 1)]
    start = pd.Timestamp(start)
    end = pd.Timestamp(end) if end is not None else data["start_time"].max()
    edges = list(pd.date_range(start, end + pd.Timedelta(days=1), freq=f"{step_days}D"))
    if edges[-1] <= end:
        edges.append(end + pd.Timedelta(days=1))
    preds = []
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        rows = data[(data["start_time"] >= lo) & (data["start_time"] < hi)]
        if rows.empty:
            continue
        if progress:
            progress(k / max(len(edges) - 1, 1), f"training before {lo.date()} -> predicting {len(rows)} matches")
        past = pd.concat(preds, ignore_index=True) if preds else None
        weights, temp, hard = {}, 1.0, {}
        if adaptive and past is not None and len(past) >= 300:
            recent = past.tail(config.DEFAULTS["hedge_window"])
            comp = recent[[c for c in recent.columns if c.startswith("comp_")]].copy()
            comp.columns = [c[5:] for c in comp.columns]
            weights = M.hedge_weights(comp, recent["y"], config.DEFAULTS["hedge_eta"])
            temp = M.fit_temperature(M.blend(comp, weights), recent["y"])
            hard = dict(zip(past["match_id"], np.abs(past["y"] - past["p_a"])))
        mdl = M.train(feat, learners, cutoff=lo, hardness=hard,
                      hardness_alpha=config.DEFAULTS["hardness_alpha"] if adaptive else 0.0,
                      weights=weights, temperature=temp)
        P = mdl.predict(rows)
        P["train_cutoff"] = lo
        preds.append(P)
    if progress:
        progress(1.0, "done")
    if not preds:
        return pd.DataFrame()
    bt = pd.concat(preds, ignore_index=True)
    bt["correct"] = ((bt["p_a"] >= 0.5).astype(float) == bt["y"]).astype(int)
    bt["log_loss"] = _ll(bt["p_a"], bt["y"])
    return bt


def summarize(bt: pd.DataFrame) -> dict:
    if bt is None or bt.empty:
        return {}
    y, p = bt["y"].to_numpy(float), bt["p_a"].to_numpy(float)
    s = {"matches": int(len(bt)), "accuracy": float(((p >= 0.5) == y).mean()), "log_loss": float(_ll(p, y).mean()),
         "brier": float(((p - y) ** 2).mean()), "elo_accuracy": float(((bt["elo_p_a"] >= 0.5) == y).mean())}
    try:
        s["auc"] = float(roc_auc_score(np.r_[y, 1 - y], np.r_[p, 1 - p]))
    except ValueError:
        s["auc"] = float("nan")
    s["by_confidence"] = (bt.groupby("confidence")["correct"].agg(["size", "mean"]).rename(
        columns={"size": "picks", "mean": "hit_rate"}).to_dict("index"))
    if "market_p_a" in bt and bt["market_p_a"].notna().sum() >= 30:
        mk = bt[bt["market_p_a"].notna()]
        s["market_accuracy"] = float(((mk["market_p_a"] >= 0.5) == mk["y"]).mean())
        s["model_accuracy_same_matches"] = float(((mk["p_a"] >= 0.5) == mk["y"]).mean())
    return s


def calibration_table(bt: pd.DataFrame) -> pd.DataFrame:
    pick = np.maximum(bt["p_a"], 1 - bt["p_a"])
    b = pd.cut(pick, [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0], include_lowest=True)
    t = pd.DataFrame({"bucket": b.astype(str), "predicted": pick, "correct": bt["correct"]})
    return t.groupby("bucket", observed=True).agg(matches=("correct", "size"), predicted=("predicted", "mean"),
                                                  actual=("correct", "mean")).reset_index()
