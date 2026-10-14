"""Causal features. A z-score on date t uses only observations before t."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentiment_long.config import Z_WINDOW


def causal_z(series: pd.Series, window: int = Z_WINDOW) -> pd.Series:
    x = series.astype(float)
    mu = x.shift(1).rolling(window, min_periods=window).mean()
    sd = x.shift(1).rolling(window, min_periods=window).std(ddof=1)
    z = (x - mu) / sd.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan)


def _daily(series: pd.Series) -> pd.Series:
    s = series.copy()
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index)).tz_localize(None).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = s.resample("D").last()
    return s


def align_same_day(series: pd.Series, trading_index: pd.DatetimeIndex, limit: int = 4) -> pd.Series:
    """Last print on or before each trading day, carrying at most `limit` days."""
    daily = _daily(series).ffill(limit=limit)
    out = daily.reindex(pd.DatetimeIndex(trading_index).tz_localize(None).normalize())
    out.index = trading_index
    return out


def align_previous_day(series: pd.Series, trading_index: pd.DatetimeIndex, limit: int = 3) -> pd.Series:
    """Value known before the session: the previous calendar day's print.

    Wikipedia pageviews for calendar date D are complete after that UTC day ends,
    which is after the US cash close. The session on date T may use D = T-1.
    """
    daily = _daily(series).ffill(limit=limit)
    lookup = pd.DatetimeIndex(trading_index).tz_localize(None).normalize() - pd.Timedelta(days=1)
    out = daily.reindex(lookup)
    out.index = trading_index
    return out
