"""Timing tests. These do not use market data and cannot tune a threshold."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentiment_long.engine import (  # noqa: E402
    asset_returns,
    cover_from_signal,
    strategy_returns,
    trade_gross,
)
from sentiment_long.features import causal_z  # noqa: E402
from sentiment_long.lexicon import fundamental_veto, score_text  # noqa: E402
from sentiment_long.scoring import score_source  # noqa: E402


class TimingTests(unittest.TestCase):
    def test_signal_does_not_earn_the_same_bar(self):
        close = np.array([100.0, 50.0, 50.0, 50.0, 80.0])
        signal = np.array([False, True, False, False, False])
        cover = cover_from_signal(signal, hold=2, delay=0)
        self.assertEqual(cover.tolist(), [0.0, 0.0, 1.0, 1.0, 0.0])
        self.assertAlmostEqual(trade_gross(close, 1, 2, 0), 0.0)

    def test_nonoverlapping_hold_does_not_extend_through_a_cluster(self):
        from sentiment_long.engine import cover_nonoverlapping

        signal = np.array([True, True, True, False, False, False, False])
        cover = cover_nonoverlapping(signal, hold=2, delay=0)
        self.assertEqual(cover.tolist(), [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    def test_tetlock_delay_skips_the_next_session(self):
        signal = np.array([True, False, False, False, False, False, False])
        cover = cover_from_signal(signal, hold=4, delay=1)
        self.assertEqual(cover.tolist(), [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0])

    def test_round_trip_cost(self):
        close = np.array([100.0, 100.0, 110.0, 110.0, 110.0])
        signal = np.array([True, False, False, False, False])
        cover = cover_from_signal(signal, hold=2, delay=0)
        rets = asset_returns(close)
        base = strategy_returns(rets, cover, 0.0)
        costly = strategy_returns(rets, cover, 0.01)
        self.assertAlmostEqual(float((base - costly).sum()), 0.02)

    def test_two_day_gross_matches_entry_close(self):
        close = np.array([100.0, 110.0, 121.0])
        self.assertAlmostEqual(trade_gross(close, 0, 2, 0), 0.21)

    def test_causal_z_ignores_the_future(self):
        series = pd.Series(np.arange(80, dtype=float))
        original = causal_z(series, window=20)
        shocked = series.copy()
        shocked.iloc[-1] = 10_000
        altered = causal_z(shocked, window=20)
        self.assertTrue(np.allclose(original.iloc[:-1], altered.iloc[:-1], equal_nan=True))
        self.assertNotEqual(original.iloc[-1], altered.iloc[-1])

    def test_lexicon_polarity_and_fundamental_veto(self):
        positive = {"GOOD", "STRONG"}
        negative = {"LOSS", "WEAK"}
        self.assertAlmostEqual(score_text("strong strong weak", positive, negative), 1 / 3)
        headlines = pd.DataFrame(
            {
                "text": ["loss loss", "weak loss", "loss weak"],
                "fundamental": [1, 1, 1],
            }
        )
        self.assertTrue(fundamental_veto(headlines, positive, negative))
        headlines["fundamental"] = 0
        self.assertFalse(fundamental_veto(headlines, positive, negative))

    def test_score_does_not_see_later_prices(self):
        periods = 80
        index = pd.bdate_range("2020-01-01", periods=periods)
        close = np.full(periods, 100.0)
        for i in range(1, periods):
            close[i] = close[i - 1] * 1.001
        shock = np.zeros(periods)
        shock[10] = -2.0
        panel = pd.DataFrame({"spy_close": close, "news_shock_z": shock}, index=index)
        asof = str(index[30].date())
        first = score_source(panel, "news", asof=asof)
        later = panel.copy()
        crashed = close.copy()
        crashed[-1] = 10.0
        later["spy_close"] = crashed
        second = score_source(later, "news", asof=asof)
        self.assertFalse(first["recommended"])
        self.assertEqual(first["reason"], "no_signal")
        self.assertEqual(first["recommended"], second["recommended"])
        self.assertEqual(first["n_historical_trades"], second["n_historical_trades"])
        self.assertEqual(first["bullish_score"], second["bullish_score"])
        self.assertEqual(first["historical_probability_up"], second["historical_probability_up"])
        self.assertGreaterEqual(first["n_historical_trades"], 1)
        signal_day = str(index[10].date())
        on_signal = score_source(panel, "news", asof=signal_day)
        on_signal_later = score_source(later, "news", asof=signal_day)
        self.assertTrue(on_signal["recommended"])
        self.assertEqual(on_signal["hold_trading_days"], 2)
        self.assertEqual(on_signal["reason"], "literature_prior")
        self.assertEqual(on_signal["bullish_score"], on_signal_later["bullish_score"])
        self.assertEqual(on_signal["n_historical_trades"], on_signal_later["n_historical_trades"])


class ChinaTimingTests(unittest.TestCase):
    def test_newspaper_signal_earns_the_next_month_only(self):
        from sentiment_long.cn_study import run_monthly

        index = pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31", "2020-04-30"])
        table = pd.DataFrame({"ret": [0.0, -0.10, 0.05, 0.02]}, index=index)
        signal = np.array([True, False, False, False])
        stats = run_monthly(table, signal, delay=0, cost=0.0, start="2020-01-01", end="2020-04-30")
        self.assertEqual(stats["n_trades"], 1)
        self.assertAlmostEqual(stats["mean_trade"], -0.10)


if __name__ == "__main__":
    unittest.main()
