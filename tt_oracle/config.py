from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(os.environ.get("TT_ORACLE_ROOT", Path(__file__).resolve().parent.parent))
DATA = ROOT / "data"
MODELS = ROOT / "models"
RAW = DATA / "raw"

MATCHES_CSV = DATA / "matches.csv"          # every parsed match, finished or not
TOURNAMENTS_CSV = DATA / "tournaments.csv"  # one row per tournament page
ODDS_CSV = DATA / "odds.csv"                # latest DraftKings snapshot
ODDS_HISTORY_CSV = DATA / "odds_history.csv"
LEDGER_CSV = DATA / "ledger.csv"            # predictions made before the match, graded after
PREDICTIONS_CSV = DATA / "predictions.csv"  # precomputed predictions for upcoming matches (read by the app)
BACKTEST_CSV = DATA / "backtest_latest.csv"
MODEL_FILE = MODELS / "predictor.joblib"
STATE_FILE = MODELS / "learning_state.json"

SITE = "https://www.tt-series.com"
SESSIONS = ["morning", "afternoon", "evening", "night"]

DEFAULTS = {
    "elo_k": 32.0,
    "half_life_days": 180.0,     # recency weighting of training matches
    "hardness_alpha": 0.5,
    "hedge_eta": 0.05,
    "hedge_window": 2000,
    "min_history": 5,            # matches a player needs before a prediction is trusted
}


def ensure_dirs() -> None:
    for p in (DATA, MODELS, RAW):
        p.mkdir(parents=True, exist_ok=True)
