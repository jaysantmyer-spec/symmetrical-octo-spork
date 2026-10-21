from __future__ import annotations
import pandas as pd
from pathlib import Path

DATE_COLS = {"date", "start_time", "fetched_at", "logged_at", "graded_at", "commence_time"}


def read(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    df = None
    for attempt in range(5):                     # a deploy may be rewriting the file: retry briefly instead of crashing
        try:
            df = pd.read_csv(path); break
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
            import time; time.sleep(1.5)
    if df is None:
        return pd.DataFrame()
    for c in df.columns:
        if c in DATE_COLS:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def write(df: pd.DataFrame, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def upsert(new: pd.DataFrame, path: Path, key: str | list[str]) -> pd.DataFrame:
    old = read(path)
    if old.empty:
        out = new.copy()
    else:
        out = pd.concat([old, new], ignore_index=True)
    keys = [key] if isinstance(key, str) else key
    out = out.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    write(out, path)
    return out
