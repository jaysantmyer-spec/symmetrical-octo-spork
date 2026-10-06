"""
Player name canonicalization (switchable).

tt-series.com types the same player several ways: with and without Polish diacritics ("Radło" / "Radlo"),
and with occasional typos ("Przewocki" / "Przewlocki", "Zochaniak" / "Zochniak"). Left alone, each
spelling becomes a separate player with its own Elo and history. This module:
  1. folds accents to ASCII (so "Radło" and "Radlo" are the same string), and
  2. merges spellings that share the first name and have a near-identical surname into the most
     common spelling, provided the rarer one is clearly a typo (rare in absolute terms and a small
     share of the pair), not a different player.

Switch: environment variable TT_MERGE_NAMES ("1" on, "0" off; default on). The app's sidebar has a
checkbox that overrides it for the current session. The workflows honour the same variable, so to go back
to the old behaviour everywhere, set TT_MERGE_NAMES=0 (GitHub: Settings -> Secrets and variables ->
Actions -> Variables; Streamlit: app Settings -> Secrets).
"""
from __future__ import annotations

import os
import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd

_SPECIAL = str.maketrans({"ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ß": "ss", "æ": "ae", "Æ": "Ae"})


def merge_enabled() -> bool:
    return os.getenv("TT_MERGE_NAMES", "1").strip().lower() not in ("0", "false", "off", "no")


def fold(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name).translate(_SPECIAL)).encode("ascii", "ignore").decode()
    s = re.sub(r"\s+", " ", s).strip()
    return " ".join(w.capitalize() if w.islower() or w.isupper() else w for w in s.split())


def _parts(n: str) -> tuple[str, str]:
    t = n.lower().split()
    return (t[0] if t else ""), (t[-1] if t else "")


def build_alias_map(counts: pd.Series, min_ratio: float = 0.86, max_share: float = 0.25, max_count: int = 60) -> dict[str, str]:
    """counts: appearances per (already folded) name. Returns {variant: canonical}."""
    names = list(counts.index)
    by_first: dict[str, list[tuple[str, str]]] = {}
    for n in names:
        f, l = _parts(n)
        by_first.setdefault(f, []).append((n, l))
    alias: dict[str, str] = {}
    for f, lst in by_first.items():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                (n1, l1), (n2, l2) = lst[i], lst[j]
                if l1 == l2 or min(len(l1), len(l2)) < 4:
                    continue
                if SequenceMatcher(None, l1, l2).ratio() < min_ratio:
                    continue
                big, small = (n1, n2) if counts[n1] >= counts[n2] else (n2, n1)
                if counts[small] <= max_count and counts[small] / (counts[small] + counts[big]) <= max_share:
                    alias[small] = big
    for k in list(alias):  # resolve chains
        seen = set()
        while alias[k] in alias and alias[k] not in seen:
            seen.add(alias[k]); alias[k] = alias[alias[k]]
    return alias


def canonicalize(matches: pd.DataFrame, enabled: bool | None = None) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fold accents on both player columns, then merge typo variants. No-op when the switch is off."""
    if not (merge_enabled() if enabled is None else enabled):
        return matches, {}
    m = matches.copy()
    for c in ("player_a", "player_b"):
        m[c] = m[c].astype(str).map(fold)
    counts = pd.concat([m["player_a"], m["player_b"]]).value_counts()
    alias = build_alias_map(counts)
    if alias:
        for c in ("player_a", "player_b"):
            m[c] = m[c].map(lambda n: alias.get(n, n))
    return m, alias
