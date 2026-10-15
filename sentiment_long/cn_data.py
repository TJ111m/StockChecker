"""Domestic A-share prices and mainland-newspaper uncertainty. No US series."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

import pandas as pd

from sentiment_long.config import USER_AGENT
from sentiment_long.paths import CACHE

SINA = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
EPU_URL = "https://www.policyuncertainty.com/media/China_Mainland_Paper_EPU.xlsx"
MARGIN_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
INDICES = {
    "csi300": "sh000300",
    "shanghai": "sh000001",
    "chinext": "sz399006",
    "csi500": "sh000905",
}


def _get(url: str, referer: str | None = None, timeout: int = 60) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if referer:
        headers["Referer"] = referer
    request = urllib.request.Request(url, headers=headers)
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(1 + attempt)
    raise RuntimeError(f"failed to fetch {url}") from last_error


def download_index(symbol: str, bars: int = 6000, refresh: bool = False) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"sina_{symbol}.csv"
    if path.exists() and not refresh:
        frame = pd.read_csv(path, parse_dates=["date"])
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
        return frame
    query = urllib.parse.urlencode({"symbol": symbol, "scale": 240, "ma": "no", "datalen": bars})
    payload = json.loads(_get(f"{SINA}?{query}"))
    frame = pd.DataFrame(payload)
    frame["date"] = pd.to_datetime(frame["day"]).dt.tz_localize(None).dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"])
    frame = frame[frame["close"] > 0][["date", "close"]].drop_duplicates("date")
    frame.to_csv(path, index=False)
    print(f"sina {symbol}: {len(frame)} {frame['date'].min().date()} → {frame['date'].max().date()}")
    return frame


def download_indices(refresh: bool = False) -> dict[str, pd.Series]:
    series = {}
    for name, symbol in INDICES.items():
        frame = download_index(symbol, refresh=refresh)
        values = frame.set_index("date")["close"].astype(float).sort_index()
        values = values[~values.index.duplicated(keep="last")]
        series[name] = values
        time.sleep(0.3)
    return series


def download_epu(refresh: bool = False) -> pd.Series:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "cn_mainland_epu.csv"
    if path.exists() and not refresh:
        frame = pd.read_csv(path, parse_dates=["date"])
    else:
        raw = CACHE / "cn_mainland_epu.xlsx"
        raw.write_bytes(_get(EPU_URL))
        frame = pd.read_excel(raw, sheet_name="EPU 2000 onwards")
        frame = frame.rename(columns={"EPU": "value"})
        frame = frame.dropna(subset=["year", "month", "value"])
        frame["date"] = pd.to_datetime(
            {"year": frame["year"].astype(int), "month": frame["month"].astype(int), "day": 1}
        )
        frame = frame[["date", "value"]]
        frame.to_csv(path, index=False)
        print(f"mainland EPU: {len(frame)} {frame['date'].min().date()} → {frame['date'].max().date()}")
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    series = frame.set_index("date")["value"].astype(float).sort_index()
    return series[~series.index.duplicated(keep="last")]


def download_margin(refresh: bool = False) -> pd.Series:
    """Sum of Shanghai and Shenzhen margin balances. Exchange data, not newspaper tone."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "cn_margin.csv"
    if path.exists() and not refresh:
        frame = pd.read_csv(path, parse_dates=["date"])
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
        return frame.set_index("date")["rzye"].astype(float).sort_index()
    rows = []
    page = 1
    while page <= 40:
        query = urllib.parse.urlencode(
            {
                "reportName": "RPTA_WEB_RZRQ_LSSH",
                "columns": "DIM_DATE,RZYE,SCDM",
                "pageSize": 500,
                "pageNumber": page,
                "sortTypes": 1,
                "sortColumns": "DIM_DATE",
            }
        )
        payload = json.loads(_get(f"{MARGIN_URL}?{query}", referer="https://data.eastmoney.com/rzrq/total.html"))
        result = payload.get("result") or {}
        batch = result.get("data") or []
        if not batch:
            break
        rows.extend(batch)
        pages = int(result.get("pages") or page)
        print(f"margin page {page}/{pages} rows {len(batch)}")
        if page >= pages:
            break
        page += 1
        time.sleep(0.2)
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["DIM_DATE"]).dt.tz_localize(None).dt.normalize()
    frame["rzye"] = pd.to_numeric(frame["RZYE"], errors="coerce")
    daily = frame.dropna(subset=["rzye"]).groupby("date")["rzye"].sum().sort_index()
    out = daily.rename("rzye").reset_index()
    out.to_csv(path, index=False)
    print(f"margin: {len(out)} {out['date'].min().date()} → {out['date'].max().date()}")
    return daily
