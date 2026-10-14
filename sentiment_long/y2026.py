"""Search long-only rules on 2026 domestic news dates.

Parameters are chosen on events before 2026-07-01. Events on or after that
date are held out. A rule is preferred only when both slices have positive
mean excess after a 10 bp round trip.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sentiment_long.cn_data import download_indices
from sentiment_long.paths import RESULTS, ROOT

EVENTS = ROOT / "research" / "cn_2026_events.csv"
SPLIT = pd.Timestamp("2026-07-01")
COST_RT = 0.001
HOLDS = (1, 2, 3, 5, 10, 20)
DELAYS = (0, 1)
INDICES = ("csi300", "shanghai", "chinext", "csi500")
MIN_TUNE = 8
MIN_HOLDOUT = 6


def _snap(index: pd.DatetimeIndex, day: pd.Timestamp) -> pd.Timestamp | None:
    if day in index:
        return day
    later = index[index >= day]
    if len(later) == 0 or (later[0] - day).days > 4:
        return None
    return pd.Timestamp(later[0])


def _forward(close: np.ndarray, loc: int, hold: int, delay: int) -> float:
    entry = loc + delay
    last = entry + hold
    if entry < 0 or last >= len(close) or close[entry] <= 0:
        return np.nan
    if not np.isfinite(close[entry]) or not np.isfinite(close[last]):
        return np.nan
    return float(close[last] / close[entry] - 1.0)


def load_events(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.read_csv(EVENTS)
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    snapped = []
    for day in frame["date"]:
        snapped.append(_snap(calendar, pd.Timestamp(day)))
    frame["trade_date"] = snapped
    frame = frame.dropna(subset=["trade_date"]).copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["slice"] = np.where(frame["trade_date"] < SPLIT, "tune", "holdout")
    return frame


def prepare(refresh: bool = False) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    indices = download_indices(refresh=refresh)
    calendar = pd.DatetimeIndex(indices["csi300"].index)
    events = load_events(calendar)
    return indices, events


def _subset(events: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "panic":
        out = events[events["kind"] == "panic"]
    elif name == "euphoria":
        out = events[events["kind"] == "euphoria"]
    elif name == "both":
        out = events[events["kind"].isin(["panic", "euphoria"])]
    elif name == "policy":
        out = events[(events["kind"] == "euphoria") & (events["family"] == "policy")]
    elif name == "geo_panic":
        out = events[(events["kind"] == "panic") & (events["family"] == "geo")]
    elif name == "tech_panic":
        out = events[(events["kind"] == "panic") & (events["family"] == "tech")]
    elif name == "neutral":
        out = events[events["kind"] == "neutral"]
    else:
        raise KeyError(name)
    return out.drop_duplicates("trade_date")


def _stats(gross: np.ndarray, uncond: np.ndarray) -> dict:
    gross = np.asarray(gross, dtype=float)
    gross = gross[np.isfinite(gross)]
    uncond = np.asarray(uncond, dtype=float)
    uncond = uncond[np.isfinite(uncond)]
    n = int(len(gross))
    if n == 0 or len(uncond) == 0:
        return {"n": n, "mean_gross": np.nan, "excess_gross": np.nan, "excess_net": np.nan, "t": np.nan, "hit": np.nan}
    mean_g = float(np.mean(gross))
    excess = mean_g - float(np.mean(uncond))
    if n >= 2:
        se = float(np.std(gross, ddof=1) / np.sqrt(n))
        t_stat = excess / se if se > 0 else np.nan
    else:
        t_stat = np.nan
    return {
        "n": n,
        "mean_gross": mean_g,
        "excess_gross": float(excess),
        "excess_net": float(excess - COST_RT),
        "t": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "hit": float(np.mean(gross > 0)),
    }


def _uncond(close: np.ndarray, dates: pd.DatetimeIndex, hold: int, delay: int, start: str, end: str) -> np.ndarray:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    values = []
    for i in range(len(close)):
        entry = i + delay
        if entry >= len(dates) or dates[entry] < start_ts or dates[entry] > end_ts:
            continue
        value = _forward(close, i, hold, delay)
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def _event_gross(events: pd.DataFrame, prices: pd.Series, hold: int, delay: int) -> np.ndarray:
    index = pd.DatetimeIndex(prices.index)
    close = prices.to_numpy(dtype=float)
    values = []
    for day in events["trade_date"]:
        loc = int(index.get_loc(pd.Timestamp(day)))
        values.append(_forward(close, loc, hold, delay))
    return np.asarray(values, dtype=float)


def search(indices: dict[str, pd.Series], events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subsets = ("panic", "euphoria", "both", "policy", "geo_panic", "tech_panic", "neutral")
    for index_name in INDICES:
        prices = indices[index_name].sort_index()
        dates = pd.DatetimeIndex(prices.index)
        close = prices.to_numpy(dtype=float)
        for hold in HOLDS:
            for delay in DELAYS:
                uncond = _uncond(close, dates, hold, delay, "2026-01-01", "2026-10-14")
                for subset in subsets:
                    chosen = _subset(events, subset)
                    tune = chosen[chosen["slice"] == "tune"]
                    holdout = chosen[chosen["slice"] == "holdout"]
                    tune_stats = _stats(_event_gross(tune, prices, hold, delay), uncond)
                    hold_stats = _stats(_event_gross(holdout, prices, hold, delay), uncond)
                    rows.append(
                        {
                            "market": index_name,
                            "subset": subset,
                            "hold": hold,
                            "delay": delay,
                            "tune_n": tune_stats["n"],
                            "tune_excess_net": tune_stats["excess_net"],
                            "tune_t": tune_stats["t"],
                            "tune_hit": tune_stats["hit"],
                            "holdout_n": hold_stats["n"],
                            "holdout_excess_net": hold_stats["excess_net"],
                            "holdout_t": hold_stats["t"],
                            "holdout_hit": hold_stats["hit"],
                            "holdout_mean_gross": hold_stats["mean_gross"],
                        }
                    )
    return pd.DataFrame(rows)


def select(table: pd.DataFrame) -> pd.DataFrame:
    eligible = table[(table["tune_n"] >= MIN_TUNE) & (table["tune_excess_net"] > 0)].copy()
    eligible = eligible.sort_values(["tune_excess_net", "tune_t"], ascending=False)
    confirmed = eligible[(eligible["holdout_n"] >= MIN_HOLDOUT) & (eligible["holdout_excess_net"] > 0)]
    return confirmed


def render(events: pd.DataFrame, table: pd.DataFrame, confirmed: pd.DataFrame) -> str:
    def bp(value):
        if value is None or not np.isfinite(value):
            return "—"
        return f"{value * 10000:.1f} bp"

    def num(value):
        if value is None or not np.isfinite(value):
            return "—"
        return f"{value:.2f}"

    counts = events.groupby(["slice", "kind"]).size().unstack(fill_value=0)
    lines = [
        "# 2026 年事件上的策略搜索",
        "",
        f"事件 {len(events)} 个，交易日已对齐到沪深 300 日历。7 月 1 日之前是调参集，当天及之后是留出集。",
        "优选条件写在看留出结果之前：调参集至少 8 笔且扣费后超额为正，留出集至少 6 笔且扣费后超额也为正。排序只用调参集。",
        "",
        "```",
        counts.to_string(),
        "```",
        "",
        f"同时满足两边为正的规则有 {len(confirmed)} 条。",
        "",
    ]
    if not confirmed.empty:
        top = confirmed.iloc[0]
        lines.append(
            "按调参集扣费后超额排在第一、并且留出集仍为正的规则是："
            f"{top.market}，子集 {top.subset}，持有 {int(top.hold)} 个交易日，跳过 {int(top.delay)} 天。"
            f"调参超额 {bp(top.tune_excess_net)}，留出超额 {bp(top.holdout_excess_net)}。"
            "留出超额若只比零高几个基点，就不能把这条当成很稳的样本外边缘。"
        )
        lines.append("")
    if confirmed.empty:
        lines.append("这一轮网格里没有规则同时通过调参集和留出集。下表是调参集超额最高、且调参样本够 8 笔的 8 条，留出结果原样附上。")
        show = table[table["tune_n"] >= MIN_TUNE].sort_values("tune_excess_net", ascending=False).head(8)
    else:
        lines.append("下面按调参集扣费后超额排列，只列出通过留出集的规则。")
        show = confirmed.head(12)
    lines.append("")
    lines.append("| 指数 | 事件子集 | 持有 | 跳过 | 调参笔数 | 调参超额 | 调参t | 留出笔数 | 留出超额 | 留出t |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in show.itertuples(index=False):
        lines.append(
            f"| {row.market} | {row.subset} | {row.hold} | {row.delay} | {row.tune_n} | {bp(row.tune_excess_net)} | {num(row.tune_t)} | {row.holdout_n} | {bp(row.holdout_excess_net)} | {num(row.holdout_t)} |"
        )
    lines.extend(
        [
            "",
            "超额是事件窗口收益减去 2026 年所有同长度窗口的平均收益，再减去 10 bp 往返。跳过 1 表示不拿事件收盘后的第一个交易日。",
            "相邻事件的窗口会重叠，t 值偏乐观。这不是投资建议。",
            "",
        ]
    )
    return "\n".join(lines)


def run(refresh: bool = False) -> dict:
    indices, events = prepare(refresh=refresh)
    table = search(indices, events)
    confirmed = select(table)
    out = RESULTS / "cn"
    out.mkdir(parents=True, exist_ok=True)
    events.to_csv(out / "y2026_events.csv", index=False)
    table.to_csv(out / "y2026_search.csv", index=False)
    confirmed.to_csv(out / "y2026_preferred.csv", index=False)
    text = render(events, table, confirmed)
    (out / "y2026_report.md").write_text(text)
    payload = {
        "n_events": int(len(events)),
        "n_tune": int((events["slice"] == "tune").sum()),
        "n_holdout": int((events["slice"] == "holdout").sum()),
        "n_preferred": int(len(confirmed)),
        "preferred": confirmed.head(12).to_dict(orient="records"),
    }
    (out / "y2026_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    print(text)
    return payload


def _applies(subset: str, kind: str) -> bool:
    if subset == "both":
        return kind in {"panic", "euphoria"}
    if subset in {"euphoria", "policy"}:
        return kind == "euphoria"
    if subset in {"panic", "geo_panic", "tech_panic"}:
        return kind == "panic"
    return False


def advise_event(kind: str) -> dict:
    path = RESULTS / "cn" / "y2026_preferred.csv"
    if not path.exists():
        run()
    preferred = pd.read_csv(path)
    chosen = None
    for row in preferred.itertuples(index=False):
        if _applies(row.subset, kind):
            chosen = row
            break
    if chosen is None:
        return {
            "recommended": False,
            "event_kind": kind,
            "reason": "no_preferred_rule_for_this_kind",
            "stance": "flat",
        }
    names = {"csi300": "沪深300", "shanghai": "上证指数", "chinext": "创业板指", "csi500": "中证500"}
    return {
        "recommended": True,
        "stance": "long",
        "event_kind": kind,
        "market": names.get(chosen.market, chosen.market),
        "market_code": chosen.market,
        "subset": chosen.subset,
        "hold_trading_days": int(chosen.hold),
        "skip_sessions": int(chosen.delay),
        "tune_excess_net": float(chosen.tune_excess_net),
        "holdout_excess_net": float(chosen.holdout_excess_net),
        "tune_n": int(chosen.tune_n),
        "holdout_n": int(chosen.holdout_n),
        "rationale": (
            f"2026 年事件搜索里，调参集排第一且留出集仍为正、并适用于{kind}的规则："
            f"做多{names.get(chosen.market, chosen.market)}，持有 {int(chosen.hold)} 个交易日。"
            f"调参扣费超额 {chosen.tune_excess_net * 10000:.1f} bp，"
            f"留出扣费超额 {chosen.holdout_excess_net * 10000:.1f} bp。"
        ),
    }


if __name__ == "__main__":
    run()
