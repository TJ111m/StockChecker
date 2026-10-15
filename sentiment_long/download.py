"""Open data pulls. Cached under data/cache. No API keys."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from sentiment_long.config import (
    PRICE_END,
    PRICE_START,
    USER_AGENT,
    WIKI_END,
    WIKI_FEAR_PAGES,
    WIKI_START,
)
from sentiment_long.paths import CACHE, EVENTS

_NEWS_URL = "https://www.frbsf.org/wp-content/uploads/news_sentiment_data.xlsx"
_EPU_URL = "https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv"
_LM_URL = "https://drive.google.com/uc?export=download&id=1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF"


def _get(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504}:
                raise
            time.sleep(2 + attempt * 3)
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(2 + attempt * 3)
    raise RuntimeError(f"failed to fetch {url}") from last_error


def event_symbols() -> list[str]:
    frame = pd.read_csv(EVENTS)
    symbols = set(frame["ticker"].astype(str))
    symbols.update({"SPY", "^VIX"})
    return sorted(symbols)


def download_prices(symbols: list[str] | None = None, refresh: bool = False) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "prices.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["date"])
    symbols = symbols or event_symbols()
    start = int(pd.Timestamp(PRICE_START, tz="UTC").timestamp())
    end = int(pd.Timestamp(PRICE_END, tz="UTC").timestamp())
    frames = []
    for symbol in symbols:
        quoted = urllib.parse.quote(symbol)
        url = (
            "https://query1.finance.yahoo.com/v8/finance/chart/"
            f"{quoted}?interval=1d&period1={start}&period2={end}&events=div%7Csplit"
        )
        payload = json.loads(_get(url))
        result = payload["chart"]["result"][0]
        stamps = result.get("timestamp") or []
        indicators = result["indicators"]
        adj = (indicators.get("adjclose") or [{}])[0].get("adjclose")
        raw = indicators["quote"][0]["close"]
        values = adj if adj is not None else raw
        # Yahoo stamps the session at 13:30 UTC (09:30 New York) or, for VIX, early UTC.
        # The New York calendar date is the trading date.
        dates = (
            pd.to_datetime(stamps, unit="s", utc=True)
            .tz_convert("America/New_York")
            .tz_localize(None)
            .normalize()
        )
        frame = pd.DataFrame({"date": dates, "symbol": symbol, "close": values})
        frame = frame.dropna(subset=["close"])
        frame = frame[frame["close"] > 0]
        frames.append(frame)
        print(f"prices {symbol}: {len(frame)} rows {frame['date'].min().date()} → {frame['date'].max().date()}")
        time.sleep(0.25)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(path, index=False)
    return out


def download_news_sentiment(refresh: bool = False) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "news_sentiment.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["date"])
    raw_path = CACHE / "news_sentiment.xlsx"
    raw_path.write_bytes(_get(_NEWS_URL))
    frame = pd.read_excel(raw_path, sheet_name="Data")
    frame = frame.rename(columns={"News Sentiment": "value"})
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    frame = frame[["date", "value"]].dropna()
    frame.to_csv(path, index=False)
    print(f"news sentiment: {len(frame)} rows {frame['date'].min().date()} → {frame['date'].max().date()}")
    return frame


def download_epu(refresh: bool = False) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "epu_daily.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["date"])
    text = _get(_EPU_URL).decode("utf-8", errors="replace")
    (CACHE / "epu_daily_raw.csv").write_text(text)
    frame = pd.read_csv(CACHE / "epu_daily_raw.csv")
    frame["date"] = pd.to_datetime(dict(year=frame["year"], month=frame["month"], day=frame["day"]))
    frame = frame.rename(columns={"daily_policy_index": "value"})
    frame = frame[["date", "value"]].dropna()
    frame.to_csv(path, index=False)
    print(f"epu: {len(frame)} rows {frame['date'].min().date()} → {frame['date'].max().date()}")
    return frame


def download_wiki(refresh: bool = False) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "wiki_fear.csv"
    manifest_path = CACHE / "wiki_manifest.json"
    if path.exists() and manifest_path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["date"])
    columns = {}
    manifest = {}
    for title in WIKI_FEAR_PAGES:
        quoted = urllib.parse.quote(title, safe="")
        url = (
            "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            f"en.wikipedia/all-access/user/{quoted}/daily/{WIKI_START}/{WIKI_END}"
        )
        try:
            payload = json.loads(_get(url, timeout=40))
        except Exception as exc:  # noqa: BLE001 - record and continue; a missing page is not a strategy choice
            manifest[title] = f"error: {exc}"
            print(f"wiki {title}: FAILED {exc}")
            continue
        items = payload.get("items") or []
        if not items:
            manifest[title] = "empty"
            print(f"wiki {title}: empty")
            continue
        dates = pd.to_datetime([item["timestamp"] for item in items], format="%Y%m%d%H")
        values = [item["views"] for item in items]
        series = pd.Series(values, index=dates, name=title)
        series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize()
        columns[title] = series
        manifest[title] = f"ok:{len(series)}"
        print(f"wiki {title}: {len(series)} days")
        time.sleep(0.2)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    if len(columns) < 8:
        raise RuntimeError(f"only {len(columns)} wiki fear pages downloaded: {manifest}")
    frame = pd.DataFrame(columns).sort_index()
    frame.index.name = "date"
    frame = frame.reset_index()
    frame.to_csv(path, index=False)
    return frame


def download_lexicon(refresh: bool = False) -> pd.DataFrame | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "lm_dictionary.csv"
    if path.exists() and path.stat().st_size > 1_000_000 and not refresh:
        return pd.read_csv(path, usecols=["Word", "Negative", "Positive"])
    try:
        path.write_bytes(_get(_LM_URL, timeout=120))
    except Exception as exc:  # noqa: BLE001
        print(f"LM dictionary download failed: {exc}")
        return None
    print(f"LM dictionary bytes: {path.stat().st_size}")
    return pd.read_csv(path, usecols=["Word", "Negative", "Positive"])


def download_all(refresh: bool = False) -> None:
    download_prices(refresh=refresh)
    download_news_sentiment(refresh=refresh)
    download_epu(refresh=refresh)
    download_wiki(refresh=refresh)
    download_lexicon(refresh=refresh)
