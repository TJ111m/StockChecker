"""Live retail posts from Eastmoney's Shanghai index bar.

This is the guba tape, not a monthly newspaper index. Timestamps are the
post times published on the page.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from sentiment_long.paths import RESULTS

BJ = timezone(timedelta(hours=8))
BAR = "zssh000001"
COST_RT = 0.001

POSITIVE = (
    "涨停", "利好", "拉升", "暴涨", "大涨", "新高", "突破", "抄底", "反弹", "看多",
    "加仓", "买入", "红盘", "起飞", "主升", "翻红", "止跌", "金坑",
)
NEGATIVE = (
    "跌停", "利空", "暴跌", "大跌", "跳水", "崩盘", "割肉", "看空", "减仓", "破位",
    "恐慌", "出逃", "绿盘", "腰斩", "闪崩", "亏麻", "完了", "跑路", "核按钮", "跌破",
)


def now_bj() -> datetime:
    return datetime.now(BJ)


def score_title(title: str) -> float:
    text = title or ""
    pos = sum(text.count(word) for word in POSITIVE)
    neg = sum(text.count(word) for word in NEGATIVE)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def _fetch_page(code: str, page: int) -> list[dict]:
    url = f"https://guba.eastmoney.com/list,{code},f_{page}.html"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=25) as response:
        html = response.read().decode("utf-8", "replace")
    matched = re.search(r"var article_list=(\{.*?\});", html)
    if not matched:
        return []
    payload = json.loads(matched.group(1))
    rows = []
    for post in payload.get("re") or []:
        if post.get("post_type") == 20:
            continue
        stamp = post.get("post_publish_time")
        title = post.get("post_title") or ""
        if not stamp or not title:
            continue
        rows.append(
            {
                "time": pd.Timestamp(stamp),
                "title": title,
                "clicks": int(post.get("post_click_count") or 0),
                "comments": int(post.get("post_comment_count") or 0),
                "score": score_title(title),
            }
        )
    return rows


def fetch_posts(code: str = BAR, pages: int = 30) -> pd.DataFrame:
    frames = []
    for page in range(1, pages + 1):
        frames.extend(_fetch_page(code, page))
        time.sleep(0.12)
    if not frames:
        return pd.DataFrame(columns=["time", "title", "clicks", "comments", "score"])
    out = pd.DataFrame(frames).drop_duplicates("title")
    return out.sort_values("time")


def fetch_bars(symbol: str = "sh000300", scale: int = 5, bars: int = 120) -> pd.DataFrame:
    url = (
        "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        f"CN_MarketData.getKLineData?symbol={symbol}&scale={scale}&ma=no&datalen={bars}"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=25) as response:
        payload = json.loads(response.read().decode())
    frame = pd.DataFrame(payload)
    frame["time"] = pd.to_datetime(frame["day"])
    frame["close"] = pd.to_numeric(frame["close"])
    return frame.sort_values("time")


def live_pulse(posts: pd.DataFrame, window_minutes: int = 15) -> dict:
    clock = now_bj().replace(tzinfo=None)
    if posts.empty:
        return {"posts": 0, "lag_seconds": None, "stance": "flat"}
    newest = posts["time"].max()
    lag = (clock - newest.to_pydatetime()).total_seconds()
    window = posts[posts["time"] >= newest - pd.Timedelta(minutes=window_minutes)]
    polarity = float(window["score"].mean()) if len(window) else 0.0
    negatives = int((window["score"] < 0).sum())
    positives = int((window["score"] > 0).sum())
    return {
        "source": "东方财富上证指数股吧",
        "url": "https://guba.eastmoney.com/list,zssh000001.html",
        "asof": clock.strftime("%Y-%m-%d %H:%M:%S"),
        "newest_post_time": str(newest),
        "lag_seconds": round(lag, 1),
        "window_minutes": window_minutes,
        "posts_in_window": int(len(window)),
        "scored_negative": negatives,
        "scored_positive": positives,
        "polarity": round(polarity, 3),
        "bullish_score": round(max(polarity, 0.0), 3),
        "bearish_score": round(max(-polarity, 0.0), 3),
        "stance": "long" if polarity >= 0.15 else "flat",
        "sample_titles": window.sort_values("time").tail(5)["title"].tolist(),
    }


def speed_table(posts: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Next 5/15/30/60 minute index return after each 5-minute guba bucket."""
    if posts.empty or bars.empty:
        return pd.DataFrame()
    bucket = posts.set_index("time").sort_index()["score"].resample("5min").mean().dropna()
    counts = posts.set_index("time").sort_index()["score"].resample("5min").size()
    close = bars.set_index("time")["close"].sort_index()
    rows = []
    for stamp, polarity in bucket.items():
        if counts.get(stamp, 0) < 5:
            continue
        after = close[close.index > stamp]
        if after.empty:
            continue
        entry = float(after.iloc[0])
        for minutes in (5, 15, 30, 60):
            future = close[close.index >= stamp + pd.Timedelta(minutes=minutes)]
            if future.empty:
                continue
            gross = float(future.iloc[0]) / entry - 1.0
            rows.append(
                {
                    "bucket": str(stamp),
                    "posts": int(counts.get(stamp, 0)),
                    "polarity": float(polarity),
                    "minutes": minutes,
                    "gross": gross,
                    "net": gross - COST_RT,
                }
            )
    return pd.DataFrame(rows)


def summarize_speed(table: pd.DataFrame) -> list[dict]:
    if table.empty:
        return []
    out = []
    negative = table[table["polarity"] < 0]
    positive = table[table["polarity"] > 0]
    for minutes in (5, 15, 30, 60):
        neg = negative[negative["minutes"] == minutes]["net"]
        pos = positive[positive["minutes"] == minutes]["net"]
        out.append(
            {
                "minutes": minutes,
                "negative_buckets": int(len(neg)),
                "negative_mean_net": None if neg.empty else float(neg.mean()),
                "positive_buckets": int(len(pos)),
                "positive_mean_net": None if pos.empty else float(pos.mean()),
            }
        )
    return out


def run(pages: int = 25) -> dict:
    posts = fetch_posts(BAR, pages=pages)
    bars = fetch_bars()
    pulse = live_pulse(posts)
    table = speed_table(posts, bars)
    pulse["history_from"] = None if posts.empty else str(posts["time"].min())
    pulse["history_to"] = None if posts.empty else str(posts["time"].max())
    pulse["posts_fetched"] = int(len(posts))
    pulse["pages"] = pages
    pulse["forward"] = summarize_speed(table)
    out = RESULTS / "cn"
    out.mkdir(parents=True, exist_ok=True)
    posts.to_csv(out / "guba_posts.csv", index=False)
    (out / "live_latest.json").write_text(json.dumps(pulse, ensure_ascii=False, indent=2))
    if not table.empty:
        table.to_csv(out / "guba_speed.csv", index=False)
    return pulse


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
