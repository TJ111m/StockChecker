#!/usr/bin/env python3
"""Probe keyless HTTP sources for a news-sentiment equity backtest.

Prints success or failure and a few sample rows. It does not download full
price history. Set PROBE_SKIP_DOC=1 to skip the GDELT DOC call (that endpoint
rate-limits this kind of egress hard).
"""

from __future__ import annotations

import gzip
import io
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone

UA = "StockCheckerResearch/1.0 (probe_sources; research@localhost)"
CTX = ssl.create_default_context()
TIMEOUT = 45


def fetch(url: str, timeout: int = TIMEOUT, max_bytes: int | None = None, method: str = "GET", headers: dict | None = None):
    hdrs = {"User-Agent": UA, "Accept": "*/*"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, headers=hdrs, method=method)
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=CTX) as resp:
            body = resp.read() if max_bytes is None else resp.read(max_bytes)
            return {
                "ok": True,
                "status": resp.status,
                "ctype": resp.headers.get("Content-Type"),
                "clen": resp.headers.get("Content-Length"),
                "final": resp.geturl(),
                "elapsed": round(time.time() - started, 2),
                "body": body,
                "headers": resp.headers,
            }
    except urllib.error.HTTPError as exc:
        err_body = exc.read(400) if exc.fp is not None else b""
        return {
            "ok": False,
            "status": exc.code,
            "elapsed": round(time.time() - started, 2),
            "body": err_body,
            "error": str(exc),
            "final": url,
        }
    except Exception as exc:  # noqa: BLE001 - probe should report every failure
        return {
            "ok": False,
            "status": None,
            "elapsed": round(time.time() - started, 2),
            "body": b"",
            "error": f"{type(exc).__name__}: {exc}",
            "final": url,
        }


def line(source: str, ok: bool, detail: str) -> None:
    flag = "OK" if ok else "FAIL"
    print(f"[{flag}] {source}: {detail}")


def yahoo() -> None:
    end = datetime(2026, 10, 15, tzinfo=timezone.utc)
    start = end - timedelta(days=12)
    p1, p2 = int(start.timestamp()), int(end.timestamp())
    for label, symbol in (("SPY", "SPY"), ("QQQ", "QQQ"), ("^VIX", "%5EVIX")):
        url = (
            "https://query1.finance.yahoo.com/v8/finance/chart/"
            f"{symbol}?period1={p1}&period2={p2}&interval=1d&includeAdjustedClose=true"
        )
        got = fetch(url, timeout=30)
        if not got["ok"]:
            line("yahoo " + label, False, f"HTTP {got['status']} {got.get('error', '')[:120]}")
            continue
        try:
            payload = json.loads(got["body"])
            result = payload["chart"]["result"][0]
            stamps = result["timestamp"]
            quote = result["indicators"]["quote"][0]
            adj = result["indicators"]["adjclose"][0]["adjclose"]
            last = len(stamps) - 1
            while last >= 0 and quote["close"][last] is None:
                last -= 1
            when = datetime.fromtimestamp(stamps[last], timezone.utc).date().isoformat()
            line(
                "yahoo " + label,
                True,
                f"n={len(stamps)} last={when} close={quote['close'][last]:.4f} "
                f"adj={adj[last]:.4f} fields=timestamp,open,high,low,close,volume,adjclose",
            )
        except Exception as exc:  # noqa: BLE001
            line("yahoo " + label, False, f"parse {exc}")


def gdelt_doc() -> None:
    if os.environ.get("PROBE_SKIP_DOC") == "1":
        print("[SKIP] gdelt doc: PROBE_SKIP_DOC=1")
        return
    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc?query=stock%20market"
        "&mode=timelinetone&format=json&startdatetime=20200101000000&enddatetime=20200102000000"
    )
    got = fetch(url, timeout=60, headers={"Accept": "application/json,text/plain,*/*"})
    text = got["body"].decode("utf-8", "replace")
    if not got["ok"]:
        line("gdelt doc timelinetone 1d", False, f"HTTP {got['status']} {text[:160].strip()}")
        return
    if not text.lstrip().startswith("{"):
        line("gdelt doc timelinetone 1d", False, f"HTTP {got['status']} non-json {text[:160].strip()}")
        return
    payload = json.loads(text)
    timeline = payload.get("timeline") or []
    points = []
    if timeline and isinstance(timeline[0], dict):
        for value in timeline[0].values():
            if isinstance(value, list):
                points = value
                break
    sample = points[0] if points else {}
    line(
        "gdelt doc timelinetone 1d",
        True,
        f"top={list(payload)} n={len(points)} sample={json.dumps(sample, ensure_ascii=False)[:180]}",
    )


def stooq() -> None:
    url = "https://stooq.com/q/d/l/?s=spy.us&d1=20200102&d2=20200115&i=d"
    got = fetch(url, timeout=25)
    text = got["body"].decode("utf-8", "replace")
    if got["ok"] and text.startswith("Date,"):
        rows = text.splitlines()
        line("stooq spy.us", True, f"rows={len(rows) - 1} header={rows[0]} row1={rows[1] if len(rows) > 1 else ''}")
        return
    snippet = " ".join(text.split())[:140]
    line("stooq spy.us", False, f"HTTP {got['status']} not csv: {snippet}")


def fred() -> None:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS&cosd=2026-10-01&coed=2026-10-14"
    got = fetch(url, timeout=30)
    if not got["ok"]:
        line("fred VIXCLS", False, f"HTTP {got['status']} {got.get('error', '')[:120]}")
        return
    rows = got["body"].decode("utf-8", "replace").splitlines()
    line("fred VIXCLS", rows[:1] == ["observation_date,VIXCLS"], f"header={rows[0] if rows else ''} last={rows[-1] if rows else ''} n={max(len(rows) - 1, 0)}")


def wikipedia() -> None:
    for article in ("Stock_market", "Recession"):
        url = (
            "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            f"en.wikipedia/all-access/user/{article}/daily/2026100100/2026101400"
        )
        got = fetch(url, timeout=30)
        if not got["ok"]:
            line("wiki " + article, False, f"HTTP {got['status']} {got['body'][:120]!r}")
            continue
        items = json.loads(got["body"]).get("items") or []
        sample = {k: items[0][k] for k in ("timestamp", "views", "granularity")} if items else {}
        line("wiki " + article, bool(items), f"n={len(items)} sample={sample}")


def econ_policy_uncertainty() -> None:
    url = "https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv"
    got = fetch(url, timeout=30)
    if not got["ok"]:
        line("epu daily", False, f"HTTP {got['status']}")
        return
    rows = got["body"].decode("utf-8", "replace").splitlines()
    line("epu daily", rows[:1] == ["day,month,year,daily_policy_index"], f"header={rows[0]} first={rows[1] if len(rows) > 1 else ''} last={rows[-1]}")


def frbsf_news() -> None:
    url = "https://www.frbsf.org/wp-content/uploads/news-sentiment-chart-1.csv?2026-10-13"
    got = fetch(url, timeout=30)
    if not got["ok"]:
        line("frbsf news sentiment", False, f"HTTP {got['status']}")
        return
    rows = got["body"].decode("utf-8", "replace").splitlines()
    ok = bool(rows) and rows[0].lower().startswith("date,")
    line("frbsf news sentiment", ok, f"header={rows[0] if rows else ''} first={rows[1] if len(rows) > 1 else ''} last={rows[-1] if rows else ''}")


def loughran_mcdonald() -> None:
    url = "https://drive.google.com/uc?export=download&id=1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF"
    got = fetch(url, timeout=40, max_bytes=500)
    if not got["ok"]:
        line("loughran-mcdonald", False, f"HTTP {got['status']}")
        return
    text = got["body"].decode("utf-8", "replace")
    header = text.splitlines()[0] if text else ""
    line(
        "loughran-mcdonald",
        header.startswith("Word,"),
        f"bytes_read={len(got['body'])} content_length={got.get('clen')} header={header[:180]}",
    )


def vader() -> None:
    url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/sentiment/vader_lexicon.zip"
    got = fetch(url, timeout=30)
    if not got["ok"]:
        line("vader lexicon", False, f"HTTP {got['status']}")
        return
    try:
        with zipfile.ZipFile(io.BytesIO(got["body"])) as archive:
            raw = archive.read("vader_lexicon/vader_lexicon.txt").decode("utf-8")
        rows = raw.splitlines()
        line("vader lexicon", True, f"zip_bytes={len(got['body'])} entries={len(rows)} row0={rows[0][:120]}")
    except Exception as exc:  # noqa: BLE001
        line("vader lexicon", False, str(exc))


def finbert() -> None:
    meta = fetch("https://huggingface.co/ProsusAI/finbert/resolve/main/config.json", timeout=30)
    weights = fetch(
        "https://huggingface.co/ProsusAI/finbert/resolve/main/model.safetensors",
        timeout=30,
        method="HEAD",
    )
    if not meta["ok"]:
        line("finbert", False, f"config HTTP {meta['status']}")
        return
    config = json.loads(meta["body"])
    size = weights.get("clen") if weights["ok"] else None
    line(
        "finbert",
        bool(size),
        f"id2label={config.get('id2label')} safetensors_bytes={size} head_status={weights.get('status')}",
    )


def gdelt_gkg_head() -> None:
    url = "https://data.gdeltproject.org/gdeltv2/20200101000000.gkg.csv.zip"
    got = fetch(url, timeout=30, method="HEAD")
    if not got["ok"]:
        line("gdelt gkg 20200101000000", False, f"HTTP {got['status']}")
        return
    line("gdelt gkg 20200101000000", True, f"content_length={got.get('clen')} type={got.get('ctype')}")


def gdelt_ngram_title() -> None:
    candidates = (
        "https://data.gdeltproject.org/gdeltv5/weblegacy/ngrams/20261014001600.toc.json.gz",
        "https://data.gdeltproject.org/gdeltv5/weblegacy/ngrams/20260630201600.toc.json.gz",
    )
    pattern = re.compile(r"stock market|recession|wall street", re.I)
    for url in candidates:
        got = fetch(url, timeout=40)
        if not got["ok"]:
            continue
        text = gzip.decompress(got["body"]).decode("utf-8", "replace")
        hit = None
        n = 0
        for raw in text.splitlines():
            if not raw.strip():
                continue
            n += 1
            row = json.loads(raw)
            if hit is None and pattern.search(row.get("title") or ""):
                hit = row
        sample = {k: hit.get(k) for k in ("date", "lang", "title")} if hit else {}
        line("gdelt ngram toc", True, f"n={n} sample={json.dumps(sample, ensure_ascii=False)[:220]}")
        return
    line("gdelt ngram toc", False, "both sample timestamps 404")


def main() -> int:
    yahoo()
    fred()
    wikipedia()
    econ_policy_uncertainty()
    frbsf_news()
    stooq()
    loughran_mcdonald()
    vader()
    finbert()
    gdelt_gkg_head()
    gdelt_ngram_title()
    gdelt_doc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
