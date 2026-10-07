"""
Ensemble win model: logistic regression + histogram gradient boosting, plus XGBoost / LightGBM / CatBoost
when installed. Members are blended with Hedge weights (each model's vote sized by its recent log loss)
and a temperature calibration. Training rows are mirrored so the model is corner-symmetric.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import CONTEXT_FEATURES, DIFF_FEATURES


def _importable(name: str) -> bool:
    """Cheap check (no import) so the Streamlit app never pays for loading the boosting libraries."""
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


HAS_XGB, HAS_LGBM, HAS_CAT = _importable("xgboost"), _importable("lightgbm"), _importable("catboost")
LEARNER_NAMES = {"logreg": "Logistic regression", "hgb": "Hist. gradient boosting", "xgb": "XGBoost",
                 "lgbm": "LightGBM", "cat": "CatBoost"}


def available_learners() -> list[str]:
    out = ["logreg", "hgb"]
    if HAS_XGB:
        out.append("xgb")
    if HAS_LGBM:
        out.append("lgbm")
    if HAS_CAT:
        out.append("cat")
    return out


def make_learner(name: str, seed: int = 42):
    if name == "logreg":
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                         ("clf", LogisticRegression(C=0.1, max_iter=3000))])
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
                                              min_samples_leaf=60, l2_regularization=1.0, random_state=seed)
    if name == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04, subsample=0.8,
                                 colsample_bytree=0.7, min_child_weight=20, reg_lambda=2.0,
                                 eval_metric="logloss", n_jobs=-1, random_state=seed)
    if name == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=15, min_child_samples=60,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
                                  verbose=-1, random_state=seed)
    if name == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=600, depth=5, learning_rate=0.04, l2_leaf_reg=5.0, subsample=0.8,
                                  bootstrap_type="Bernoulli", loss_function="Logloss", allow_writing_files=False,
                                  verbose=0, thread_count=-1, random_seed=seed)
    raise ValueError(name)


def _fit(est, X, y, w):
    if w is None:
        return est.fit(X, y)
    if isinstance(est, Pipeline):
        return est.fit(X, y, clf__sample_weight=w)
    return est.fit(X, y, sample_weight=w)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blend(components: pd.DataFrame, weights: dict | None = None, temperature: float = 1.0) -> np.ndarray:
    names = list(components.columns)
    w = np.array([(weights or {}).get(n, 1.0) for n in names], dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.full(len(names), 1 / len(names))
    p = components.to_numpy(float) @ w
    if temperature != 1.0:
        p = _sigmoid(temperature * _logit(p))
    return np.clip(p, 1e-4, 1 - 1e-4)


def hedge_weights(components: pd.DataFrame, y, eta: float) -> dict:
    y = np.asarray(y, float)
    out = {}
    for c in components.columns:
        p = np.clip(components[c].to_numpy(float), 1e-6, 1 - 1e-6)
        ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
        out[c] = float(np.exp(-eta * ll * 100))
    s = sum(out.values())
    return {k: v / s for k, v in out.items()}


def fit_temperature(p, y) -> float:
    y = np.asarray(y, float)
    z = _logit(np.asarray(p, float))
    best, best_ll = 1.0, np.inf
    for t in np.linspace(0.5, 2.0, 31):
        q = np.clip(_sigmoid(t * z), 1e-6, 1 - 1e-6)
        ll = -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()
        if ll < best_ll:
            best, best_ll = float(t), ll
    return best


GAMES_COLS = ["abs_delo", "abs_dwin10", "abs_dsetr", "sum_setr", "abs_dh2h", "abs_elo_p", "A_today_n", "B_today_n",
              "both_n_min", "A_streak", "B_streak"]


def _games_matrix(df: pd.DataFrame) -> np.ndarray:
    """Symmetric (corner-free) features for how many games a match goes: how lopsided it is, how set-heavy
    both players' recent matches have been, fatigue, streaks."""
    g = pd.DataFrame({"abs_delo": df["d_elo"].abs(), "abs_dwin10": df["d_win10"].abs(), "abs_dsetr": df["d_setr30"].abs(),
                      "sum_setr": df["A_setr30"] + df["B_setr30"], "abs_dh2h": df["d_h2h_net"].abs(),
                      "abs_elo_p": (df["elo_p_a"] - 0.5).abs(), "A_today_n": df["A_today_n"], "B_today_n": df["B_today_n"],
                      "both_n_min": df["both_n_min"], "A_streak": df["A_streak"], "B_streak": df["B_streak"]})
    return g[GAMES_COLS].to_numpy(float)


def _iid_games(p_a: np.ndarray) -> np.ndarray:
    """Fallback: best-of-5 with an independent per-game probability implied by the match probability."""
    qs = np.linspace(0.01, 0.99, 981)
    pm = qs ** 3 * (1 + 3 * (1 - qs) + 6 * (1 - qs) ** 2)
    q = np.interp(np.asarray(p_a, float), pm, qs); r = 1 - q
    return np.column_stack([q ** 3 + r ** 3, 3 * q ** 3 * r + 3 * r ** 3 * q, 6 * q ** 3 * r ** 2 + 6 * r ** 3 * q ** 2])


def _matrix(df: pd.DataFrame, flip: bool) -> np.ndarray:
    D = df[[f"d_{k}" for k in DIFF_FEATURES]].to_numpy(float)
    C = df[CONTEXT_FEATURES].to_numpy(float)
    return np.hstack([(-D if flip else D), C])


@dataclass
class Predictor:
    models: dict
    learners: list
    weights: dict = field(default_factory=dict)
    temperature: float = 1.0
    meta: dict = field(default_factory=dict)
    games_model: object = None          # (logreg, hgb) multinomial on 3 / 4 / 5 games

    def games(self, df: pd.DataFrame, p_a: np.ndarray) -> np.ndarray:
        gm = getattr(self, "games_model", None)
        if not gm:
            return _iid_games(p_a)
        X = _games_matrix(df)
        P = np.mean([m.predict_proba(X) for m in gm], axis=0)
        return P

    def save(self, path=config.MODEL_FILE):
        config.ensure_dirs()
        joblib.dump(self, path)

    @staticmethod
    def load(path=config.MODEL_FILE):
        try:
            return joblib.load(path)
        except FileNotFoundError:
            return None
        except Exception:  # pickle from a different library version
            return None

    def components(self, df: pd.DataFrame) -> pd.DataFrame:
        Xa, Xb = _matrix(df, False), _matrix(df, True)
        return pd.DataFrame({n: 0.5 * (m.predict_proba(Xa)[:, 1] + 1 - m.predict_proba(Xb)[:, 1])
                             for n, m in self.models.items()}, index=df.index)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        comp = self.components(df)
        p = blend(comp, self.weights, self.temperature)
        out = df[["match_id", "post_id", "date", "start_time", "session", "venue", "player_a", "player_b",
                  "finished", "y", "elo_p_a", "h2h_n", "A_career_n", "B_career_n", "A_elo", "B_elo",
                  "A_win10", "B_win10", "A_today_n", "B_today_n"]].copy()
        out["p_a"], out["p_b"] = p, 1 - p
        for c in comp.columns:
            out[f"comp_{c}"] = comp[c].to_numpy()
        out["pick"] = np.where(p >= 0.5, out["player_a"], out["player_b"])
        out["pick_prob"] = np.maximum(p, 1 - p)
        out["confidence"] = pd.cut(out["pick_prob"], [0, 0.55, 0.62, 0.70, 1.01],
                                   labels=["Coin flip", "Lean", "Solid", "Strong"]).astype(str)
        thin = (out["A_career_n"] < config.DEFAULTS["min_history"]) | (out["B_career_n"] < config.DEFAULTS["min_history"])
        out["thin_history"] = thin
        G = self.games(df, p)
        out["p_g3"], out["p_g4"], out["p_g5"] = G[:, 0], G[:, 1], G[:, 2]
        out["over35"], out["over45"] = 1 - G[:, 0], G[:, 2]
        # likely final score: split each game-count between the two players by who is favoured
        q = np.clip(p, 0.02, 0.98)
        share3 = q ** 3 / (q ** 3 + (1 - q) ** 3)
        share4 = q ** 3 * (1 - q) / (q ** 3 * (1 - q) + (1 - q) ** 3 * q)
        share5 = q ** 3 * (1 - q) ** 2 / (q ** 3 * (1 - q) ** 2 + (1 - q) ** 3 * q ** 2)
        scores = {"3-0": G[:, 0] * share3, "0-3": G[:, 0] * (1 - share3), "3-1": G[:, 1] * share4, "1-3": G[:, 1] * (1 - share4),
                  "3-2": G[:, 2] * share5, "2-3": G[:, 2] * (1 - share5)}
        S = np.column_stack(list(scores.values())); names = np.array(list(scores))
        out["likely_score"] = names[S.argmax(axis=1)]
        out["likely_score_p"] = S.max(axis=1)
        return out


def train(feat: pd.DataFrame, learners: list[str] | None = None, cutoff=None, half_life_days: float | None = None,
          hardness: dict | None = None, hardness_alpha: float = 0.0, weights: dict | None = None,
          temperature: float = 1.0, seed: int = 42) -> Predictor:
    hl = config.DEFAULTS["half_life_days"] if half_life_days is None else half_life_days
    tr = feat[feat["y"].notna()]
    if cutoff is not None:
        tr = tr[tr["start_time"] < pd.Timestamp(cutoff)]
    tr = tr[tr["both_n_min"] >= 1]  # at least one prior match each; pure debut rows carry no signal
    if len(tr) < 300:
        raise ValueError(f"only {len(tr)} finished matches with history; need more data")
    ref = pd.Timestamp(cutoff) if cutoff is not None else tr["start_time"].max()
    age = (ref - tr["start_time"]).dt.total_seconds() / 86400.0
    w = 0.5 ** (age / hl)
    if hardness and hardness_alpha > 0:
        h = tr["match_id"].map(hardness).fillna(0.0).to_numpy(float)
        w = w * (1.0 + hardness_alpha * np.clip(h, 0, 1))
    w = w.to_numpy(float) if hasattr(w, "to_numpy") else np.asarray(w, float)
    X = np.vstack([_matrix(tr, False), _matrix(tr, True)])
    y = np.concatenate([tr["y"].to_numpy(int), 1 - tr["y"].to_numpy(int)])
    ww = np.concatenate([w, w])
    learners = learners or available_learners()
    models = {n: _fit(make_learner(n, seed), X, y, ww) for n in learners}
    games_model = None
    if "sets_a" in tr and tr["sets_a"].notna().sum() > 2000:
        g = tr[tr["sets_a"].notna() & tr["sets_b"].notna()]
        ng = (g["sets_a"].astype(float) + g["sets_b"].astype(float)).astype(int)
        g = g[ng.between(3, 5)]; ng = ng[ng.between(3, 5)]
        Xg = _games_matrix(g)
        wg = w[tr.index.get_indexer(g.index)]
        g_lr = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                         ("clf", LogisticRegression(C=0.3, max_iter=2000))]).fit(Xg, ng, clf__sample_weight=wg)
        g_hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=80,
                                               random_state=seed).fit(Xg, ng, sample_weight=wg)
        games_model = (g_lr, g_hgb)
    return Predictor(models, list(learners), weights or {}, temperature, {
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"), "n_train": int(len(tr)),
        "data_through": str(tr["start_time"].max()), "learners": list(learners), "half_life_days": hl,
        "hardness_alpha": hardness_alpha, "version": pd.Timestamp.now().strftime("%Y%m%d-%H%M%S"),
        "games_model": games_model is not None}, games_model)


# --------------------------------------------------------------------------- learning state
def load_state() -> dict:
    if config.STATE_FILE.exists():
        return json.loads(config.STATE_FILE.read_text())
    return {"weights": {}, "temperature": 1.0, "hardness_alpha": config.DEFAULTS["hardness_alpha"],
            "half_life_days": config.DEFAULTS["half_life_days"], "learners": None, "history": []}


def save_state(state: dict) -> None:
    config.ensure_dirs()
    config.STATE_FILE.write_text(json.dumps(state, indent=2, default=str))
