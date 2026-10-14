"""Build the trading-day panel from cached open data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentiment_long.config import WIKI_FEAR_PAGES
from sentiment_long.download import download_all
from sentiment_long.features import align_previous_day, align_same_day, causal_z
from sentiment_long.paths import CACHE


def _series(frame: pd.DataFrame, value: str = "value") -> pd.Series:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    out = out.dropna(subset=["date"]).drop_duplicates("date", keep="last")
    series = out.set_index("date")[value].astype(float).sort_index()
    series.index = pd.DatetimeIndex(series.index)
    return series


def load_prices() -> pd.DataFrame:
    frame = pd.read_csv(CACHE / "prices.csv", parse_dates=["date"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    wide = frame.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    wide.index = pd.DatetimeIndex(wide.index)
    return wide


def build_panel(refresh: bool = False) -> pd.DataFrame:
    download_all(refresh=refresh)
    prices = load_prices()
    if "SPY" not in prices.columns:
        raise RuntimeError("SPY prices missing")
    trading_index = prices.index[prices["SPY"].notna()]
    panel = pd.DataFrame(index=trading_index)
    panel.index.name = "date"
    for symbol in prices.columns:
        name = "VIX" if symbol == "^VIX" else symbol
        panel[f"{name}_close"] = prices[symbol].reindex(trading_index)

    panel["spy_close"] = panel["SPY_close"]
    panel["spy_ret"] = panel["spy_close"].pct_change()
    panel["vix"] = panel["VIX_close"]

    news = _series(pd.read_csv(CACHE / "news_sentiment.csv", parse_dates=["date"]))
    epu = _series(pd.read_csv(CACHE / "epu_daily.csv", parse_dates=["date"]))
    panel["news"] = align_same_day(news, trading_index, limit=4)
    panel["epu"] = align_same_day(epu, trading_index, limit=4)

    wiki = pd.read_csv(CACHE / "wiki_fear.csv", parse_dates=["date"])
    wiki["date"] = pd.to_datetime(wiki["date"]).dt.tz_localize(None).dt.normalize()
    wiki = wiki.set_index("date").sort_index()
    present = [col for col in WIKI_FEAR_PAGES if col in wiki.columns]
    if len(present) < 8:
        present = [col for col in wiki.columns if col != "date"]
    logs = np.log1p(wiki[present].astype(float))
    coverage = logs.notna().sum(axis=1)
    fear = logs.mean(axis=1, skipna=True)
    fear = fear.where(coverage >= 8)
    panel["wiki_fear"] = align_previous_day(fear, trading_index, limit=3)
    panel.attrs["wiki_pages"] = present

    panel["news_shock_z"] = causal_z(panel["news"].diff())
    panel["news_level_z"] = causal_z(panel["news"])
    panel["wiki_shock_z"] = causal_z(panel["wiki_fear"].diff())
    panel["wiki_level_z"] = causal_z(panel["wiki_fear"])
    panel["vix_shock_z"] = causal_z(panel["vix"].diff())
    panel["epu_shock_z"] = causal_z(panel["epu"].diff())
    panel["price_shock_z"] = causal_z(panel["spy_ret"])
    return panel


def sanity_check(panel: pd.DataFrame) -> dict:
    """Known historical facts used only to catch a bad join, not to tune a rule."""
    checks = {}

    def _ret(day: str) -> float:
        stamp = pd.Timestamp(day)
        if stamp not in panel.index:
            raise AssertionError(f"{day} is not a trading day in the panel")
        value = float(panel.loc[stamp, "spy_ret"])
        if not np.isfinite(value):
            raise AssertionError(f"SPY return on {day} is missing")
        return value

    march16 = _ret("2020-03-16")
    march24 = _ret("2020-03-24")
    april4 = _ret("2025-04-04")
    checks["spy_2020_03_16"] = march16
    checks["spy_2020_03_24"] = march24
    checks["spy_2025_04_04"] = april4
    if march16 >= -0.05:
        raise AssertionError(f"expected SPY crash on 2020-03-16, got {march16}")
    if march24 <= 0.05:
        raise AssertionError(f"expected SPY rebound on 2020-03-24, got {march24}")
    if april4 >= -0.03:
        raise AssertionError(f"expected SPY drop on 2025-04-04, got {april4}")

    vix = float(panel.loc[pd.Timestamp("2020-03-16"), "vix"])
    checks["vix_2020_03_16"] = vix
    if not np.isfinite(vix) or vix < 50:
        raise AssertionError(f"expected VIX > 50 on 2020-03-16, got {vix}")

    wiki = panel["wiki_fear"]
    march = float(wiki.loc["2020-03-01":"2020-03-31"].mean())
    january = float(wiki.loc["2020-01-01":"2020-01-31"].mean())
    checks["wiki_march_2020"] = march
    checks["wiki_january_2020"] = january
    if not np.isfinite(march) or not np.isfinite(january) or march <= january * 1.15:
        raise AssertionError(f"wiki fear did not rise in March 2020 ({january} → {march})")
    checks["rows"] = int(len(panel))
    checks["first"] = str(panel.index.min().date())
    checks["last"] = str(panel.index.max().date())
    checks["news_last"] = str(panel.index[panel["news"].notna()].max().date())
    checks["wiki_pages"] = list(panel.attrs.get("wiki_pages", []))
    return checks
