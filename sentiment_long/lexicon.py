"""Loughran-McDonald title polarity. The word lists are frozen by addition year."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

_TOKEN = re.compile(r"[A-Za-z]+")


def _active(values: pd.Series, asof_year: int) -> pd.Series:
    """A positive cell is the year the word was added. A negative cell is a removal year."""
    v = values.astype(float)
    added = (v > 0) & (v <= asof_year)
    removed_later = (v < 0) & (v.abs() > asof_year)
    return added | removed_later


def load_lists(path: Path, asof_year: int = 2010) -> tuple[set[str], set[str]]:
    frame = pd.read_csv(path, usecols=["Word", "Negative", "Positive"])
    words = frame["Word"].astype(str).str.upper()
    pos = set(words[_active(frame["Positive"], asof_year)])
    neg = set(words[_active(frame["Negative"], asof_year)])
    return pos, neg


def score_text(text: str, positive: set[str], negative: set[str]) -> float:
    tokens = [tok.upper() for tok in _TOKEN.findall(text or "")]
    if not tokens:
        return np.nan
    n_pos = sum(tok in positive for tok in tokens)
    n_neg = sum(tok in negative for tok in tokens)
    if n_pos + n_neg == 0:
        return np.nan
    return (n_pos - n_neg) / (n_pos + n_neg)


def daily_polarity(frame: pd.DataFrame, positive: set[str], negative: set[str]) -> pd.Series:
    """frame columns: date, text. Returns the mean title polarity of each day."""
    scores = [score_text(text, positive, negative) for text in frame["text"].astype(str)]
    out = pd.DataFrame({"date": pd.to_datetime(frame["date"]), "score": scores})
    out["date"] = out["date"].dt.tz_localize(None).dt.normalize()
    daily = out.groupby("date")["score"].mean()
    daily.index = pd.DatetimeIndex(daily.index)
    return daily.sort_index()


def fundamental_veto(frame: pd.DataFrame, positive: set[str], negative: set[str]) -> bool:
    """H2: do not buy a day whose fundamental headlines are negative on net.

    Requires at least three rows flagged fundamental. Unflagged text is not a veto.
    """
    if frame.empty or "fundamental" not in frame.columns:
        return False
    flagged = frame[frame["fundamental"].fillna(0).astype(int) == 1]
    if len(flagged) < 3:
        return False
    scores = [score_text(text, positive, negative) for text in flagged["text"].astype(str)]
    finite = [s for s in scores if np.isfinite(s)]
    if len(finite) < 3:
        return False
    return float(np.mean(finite)) < 0
