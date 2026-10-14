"""Command line: download, backtest, and score one day."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from sentiment_long.download import download_all, download_lexicon
from sentiment_long.lexicon import daily_polarity, fundamental_veto, load_lists
from sentiment_long.panel import build_panel
from sentiment_long.paths import CACHE
from sentiment_long.scoring import score_agent, score_mood_frame, score_source
from sentiment_long.study import run


def _print(payload: dict) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _close_series(path: str | None) -> pd.Series:
    if path:
        frame = pd.read_csv(path, parse_dates=["date"])
        value = "close" if "close" in frame.columns else "value"
        series = frame.set_index("date")[value].astype(float)
    else:
        prices = pd.read_csv(CACHE / "prices.csv", parse_dates=["date"])
        spy = prices[prices["symbol"] == "SPY"]
        series = spy.set_index("date")["close"].astype(float)
    series.index = pd.DatetimeIndex(pd.to_datetime(series.index)).tz_localize(None).normalize()
    return series.sort_index()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Public-opinion long-only research system")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download")
    download.add_argument("--refresh", action="store_true")

    backtest = sub.add_parser("backtest")
    backtest.add_argument("--refresh", action="store_true")

    score = sub.add_parser("score")
    score.add_argument("--source", default="agent", choices=["agent", "news", "wiki", "vix", "price", "epu"])
    score.add_argument("--asof")
    score.add_argument("--mood-csv", help="CSV with date and mood. Higher mood is more bullish.")
    score.add_argument("--price-csv", help="CSV with date and close. Defaults to cached SPY.")
    score.add_argument("--headlines-csv", help="CSV with date and text, optional fundamental 0/1.")
    score.add_argument("--select-direction", action="store_true")
    score.add_argument("--invert-mood", action="store_true", help="Use when the file is a fear index.")

    args = parser.parse_args(argv)
    if args.command == "download":
        download_all(refresh=args.refresh)
        return
    if args.command == "backtest":
        run(refresh=args.refresh)
        return

    if args.headlines_csv or args.mood_csv:
        close = _close_series(args.price_csv)
        veto = False
        if args.headlines_csv:
            headlines = pd.read_csv(args.headlines_csv)
            lexicon = CACHE / "lm_dictionary.csv"
            if not lexicon.exists():
                download_lexicon()
            positive, negative = load_lists(lexicon, asof_year=2010)
            mood = daily_polarity(headlines, positive, negative)
            if args.asof is not None and "fundamental" in headlines.columns:
                day = pd.Timestamp(args.asof)
                subset = headlines.copy()
                subset["date"] = pd.to_datetime(subset["date"])
                veto = fundamental_veto(subset[subset["date"].dt.normalize() == day], positive, negative)
        else:
            frame = pd.read_csv(args.mood_csv, parse_dates=["date"])
            mood = frame.set_index("date")["mood"].astype(float)
            if args.invert_mood:
                mood = -mood
        payload = score_mood_frame(
            mood,
            close,
            asof=args.asof,
            select_direction=args.select_direction,
            veto=veto,
            source_name="headlines" if args.headlines_csv else "custom",
        )
        _print(payload)
        return

    panel = build_panel(refresh=False)
    if args.source == "agent":
        _print(score_agent(panel, args.asof))
    else:
        _print(score_source(panel, args.source, args.asof))


if __name__ == "__main__":
    main()
