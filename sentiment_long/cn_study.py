"""A-share backtest. The monthly newspaper rule is not fit on these returns."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sentiment_long.cn_data import download_epu, download_indices, download_margin
from sentiment_long.config import COST_PER_SIDE, MIN_TRADES, N_PERM, PRIOR_SCORE_CAP, RANDOM_SEED, SCORE_SCALE
from sentiment_long.engine import excess_tstat, null_pvalue
from sentiment_long.features import causal_z
from sentiment_long.paths import RESULTS, ROOT
from sentiment_long.scoring import _posterior
from sentiment_long.study import evaluate

CN_EVENTS = ROOT / "research" / "cn_events.csv"
EPU_Z_WINDOW = 36
EPU_Z = 1.0
EPU_START = "2005-01-01"
POST_CIRCUIT = "2016-01-01"
MARGIN_START = "2013-01-01"
MARGIN_Z = 1.5


def _jsonify(value):
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if value is None or isinstance(value, str):
        return value
    return str(value)


def month_end_close(prices: pd.Series) -> pd.Series:
    frame = prices.sort_index().to_frame("close")
    frame["ym"] = frame.index.to_period("M")
    last = frame.groupby("ym").tail(1)
    out = last["close"].copy()
    out.index = pd.DatetimeIndex(last.index).tz_localize(None).normalize()
    return out


def monthly_frame(close: pd.Series, epu: pd.Series) -> pd.DataFrame:
    month_end = month_end_close(close)
    table = pd.DataFrame({"close": month_end})
    table["ret"] = table["close"].pct_change()
    epu = epu.copy()
    epu.index = epu.index.to_period("M")
    table["period"] = table.index.to_period("M")
    table["epu"] = table["period"].map(epu)
    table["epu_shock_z"] = causal_z(table["epu"].diff(), EPU_Z_WINDOW)
    table["epu_level_z"] = causal_z(table["epu"], EPU_Z_WINDOW)
    return table


def run_monthly(table: pd.DataFrame, signal: np.ndarray, delay: int, cost: float, start: str, end: str | None) -> dict:
    """Signal on month t earns the return of month t+1+delay. Each month is one non-overlapping bet."""
    dates = pd.DatetimeIndex(table.index)
    rets = table["ret"].to_numpy(dtype=float)
    signal = np.asarray(signal, dtype=bool)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else dates.max()
    n = len(rets)
    position = np.zeros(n)
    trades = []
    for i, flagged in enumerate(signal):
        target = i + 1 + delay
        if not flagged or target >= n or not np.isfinite(rets[target]):
            continue
        if dates[target] < start_ts or dates[target] > end_ts:
            continue
        position[target] = 1.0
        trades.append(rets[target])
    prev = np.concatenate(([0.0], position[:-1]))
    turnover = np.abs(position - prev)
    strat = np.where(np.isfinite(rets), position * rets, 0.0) - turnover * cost
    mask = np.asarray((dates >= start_ts) & (dates <= end_ts))
    daily = strat[mask]
    bench = np.where(np.isfinite(rets), rets, 0.0)[mask]
    gross = np.asarray(trades, dtype=float)
    uncond = rets[mask]
    uncond = uncond[np.isfinite(uncond)]
    stats = excess_tstat(gross, uncond)

    def _stat(candidate: np.ndarray) -> float:
        values = []
        for i, flagged in enumerate(np.asarray(candidate, dtype=bool)):
            target = i + 1 + delay
            if not flagged or target >= n or not np.isfinite(rets[target]):
                continue
            if dates[target] < start_ts or dates[target] > end_ts:
                continue
            values.append(rets[target])
        if len(values) < 2 or len(uncond) == 0:
            return np.nan
        return float(np.mean(values) - np.mean(uncond))

    p_value = null_pvalue(_stat, signal, N_PERM, RANDOM_SEED)
    wealth = np.cumprod(1.0 + daily) if len(daily) else np.array([np.nan])
    years = max(len(daily) / 12.0, 1e-9)
    end_wealth = float(wealth[-1]) if len(wealth) else np.nan
    cagr = end_wealth ** (1.0 / years) - 1.0 if np.isfinite(end_wealth) and end_wealth > 0 else np.nan
    sd = float(np.std(daily, ddof=1)) if len(daily) > 2 else np.nan
    sharpe = float(np.mean(daily) / sd * np.sqrt(12.0)) if sd and sd > 0 else np.nan
    bench_wealth = np.cumprod(1.0 + bench)
    bench_cagr = float(bench_wealth[-1] ** (1.0 / years) - 1.0) if len(bench_wealth) and bench_wealth[-1] > 0 else np.nan
    return {
        "start": str(start_ts.date()),
        "end": str(end_ts.date()),
        "delay_months": delay,
        "n_months": int(mask.sum()),
        "n_trades": int(len(gross)),
        "exposure": float(np.mean(position[mask])) if mask.any() else np.nan,
        "mean_trade": stats["mean_trade"],
        "mean_uncond": stats["mean_uncond"],
        "excess": stats["excess"],
        "t_excess": stats["t_excess"],
        "hit_rate": stats["hit_rate"],
        "sharpe": sharpe,
        "cagr": float(cagr) if np.isfinite(cagr) else np.nan,
        "buyhold_cagr": bench_cagr,
        "null_p": p_value,
        "cost_per_side": cost,
    }


def event_study(prices: pd.Series) -> pd.DataFrame:
    events = pd.read_csv(CN_EVENTS)
    events["date"] = pd.to_datetime(events["date"]).dt.tz_localize(None).dt.normalize()
    index = pd.DatetimeIndex(prices.index)
    close = prices.to_numpy(dtype=float)
    rows = []
    for event in events.itertuples(index=False):
        if event.date not in index:
            later = index[index >= event.date]
            if len(later) == 0 or (later[0] - event.date).days > 4:
                continue
            loc = int(index.get_loc(later[0]))
        else:
            loc = int(index.get_loc(event.date))
        if loc == 0:
            continue
        row = {
            "id": event.id,
            "date": str(index[loc].date()),
            "name": event.name,
            "kind": event.kind,
            "prior_hypothesis": event.prior_hypothesis,
            "hold_days_prior": int(event.hold_days_prior),
            "same_day": float(close[loc] / close[loc - 1] - 1.0),
        }
        for horizon in (1, 2, 5, 10, 20):
            last = loc + horizon
            row[f"fwd_{horizon}"] = (
                float(close[last] / close[loc] - 1.0) if last < len(close) and close[loc] > 0 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _ttest(values: np.ndarray) -> dict:
    sample = np.asarray(values, dtype=float)
    sample = sample[np.isfinite(sample)]
    n = int(len(sample))
    if n < 2:
        return {"n": n, "mean": np.nan, "t": np.nan, "hit": np.nan}
    mean = float(np.mean(sample))
    se = float(np.std(sample, ddof=1) / np.sqrt(n))
    return {"n": n, "mean": mean, "t": mean / se if se > 0 else np.nan, "hit": float(np.mean(sample > 0))}


def sanity(prices: pd.Series, monthly: pd.DataFrame) -> dict:
    ret = prices.pct_change()

    def day(stamp: str) -> float:
        value = float(ret.loc[pd.Timestamp(stamp)])
        if not np.isfinite(value):
            raise AssertionError(f"missing return {stamp}")
        return value

    checks = {
        "circuit_20160104": day("2016-01-04"),
        "covid_20200203": day("2020-02-03"),
        "july_2015": float(monthly.loc[monthly.index.to_period("M") == "2015-07", "ret"].iloc[0]),
        "august_2015": float(monthly.loc[monthly.index.to_period("M") == "2015-08", "ret"].iloc[0]),
        "september_2024": float(monthly.loc[monthly.index.to_period("M") == "2024-09", "ret"].iloc[0]),
        "first": str(prices.index.min().date()),
        "last": str(prices.index.max().date()),
        "epu_last": str(monthly.index[monthly["epu"].notna()].max().date()),
    }
    if checks["circuit_20160104"] >= -0.04:
        raise AssertionError(checks["circuit_20160104"])
    if checks["covid_20200203"] >= -0.04:
        raise AssertionError(checks["covid_20200203"])
    if checks["august_2015"] >= -0.05:
        raise AssertionError(checks["august_2015"])
    if checks["september_2024"] <= 0.05:
        raise AssertionError(checks["september_2024"])
    return checks


def score_newspaper(table: pd.DataFrame, asof: str | None) -> dict:
    dates = pd.DatetimeIndex(table.index)
    asof_ts = pd.Timestamp(asof) if asof else dates.max()
    known = table[table["epu"].notna() & table["epu_shock_z"].notna()]
    usable = known[known.index <= asof_ts]
    if usable.empty:
        raise ValueError("no mainland newspaper observation on or before this date")
    latest = usable.index.max()
    # Fast rule decided at latest month-end applies to the following month.
    hold_month = (latest.to_period("M") + 1).to_timestamp(how="end").normalize()
    shock = float(usable.loc[latest, "epu_shock_z"])
    signal = bool(np.isfinite(shock) and shock >= EPU_Z)
    completed = []
    rets = table["ret"].to_numpy(dtype=float)
    shocks = table["epu_shock_z"].to_numpy(dtype=float)
    for i in range(len(table) - 1):
        target = i + 1
        if dates[target] > asof_ts or latest < pd.Timestamp(EPU_START):
            continue
        if not np.isfinite(shocks[i]) or shocks[i] < EPU_Z or not np.isfinite(rets[target]):
            continue
        if dates[target] < pd.Timestamp(EPU_START):
            continue
        completed.append(float(rets[target]))
    gross = np.asarray(completed, dtype=float)
    window = table[(table.index >= EPU_START) & (table.index <= asof_ts)]["ret"].to_numpy(dtype=float)
    window = window[np.isfinite(window)]
    stats = excess_tstat(gross, window)
    prior = float(np.mean(window > 0)) if len(window) else 0.5
    probability, low, high = _posterior(int(np.sum(gross > 0)) if len(gross) else 0, int(len(gross)), prior)
    n = int(len(gross))
    excess = stats["excess"]
    killed = n >= MIN_TRADES and not (np.isfinite(excess) and excess > 0)
    in_force = signal and asof_ts <= hold_month and not killed
    if n == 0 or not np.isfinite(stats["mean_trade"]) or not np.isfinite(stats["mean_uncond"]):
        shrunk = 0.0
    else:
        shrunk_mean = (n * stats["mean_trade"] + 30 * stats["mean_uncond"]) / (n + 30)
        shrunk = float(shrunk_mean - stats["mean_uncond"])
    if not signal:
        reason = "no_signal"
    elif asof_ts > hold_month:
        reason = "hold_already_finished"
    elif killed:
        reason = "killed_no_edge"
    elif n < MIN_TRADES:
        reason = "literature_prior"
    else:
        reason = "historical_edge"
    recommended = bool(in_force and reason in {"literature_prior", "historical_edge"})
    if reason == "literature_prior" and recommended:
        score = float(min(1.0, max(0.0, shock - EPU_Z) / EPU_Z) * PRIOR_SCORE_CAP)
    elif recommended:
        score = float(np.clip(max(shrunk, 0.0) / SCORE_SCALE, 0.0, 1.0))
    else:
        score = 0.0
    net = None if not recommended or n == 0 else shrunk - 2 * COST_PER_SIDE
    return {
        "asof": str(asof_ts.date()),
        "source": "mainland_newspaper_epu",
        "source_kind": "mainland_newspaper",
        "ticker": "沪深300",
        "signal_month": str(latest.date()),
        "hold_through": str(hold_month.date()),
        "recommended": recommended,
        "stance": "long" if recommended else "flat",
        "reason": reason,
        "bullish_score": score,
        "deterioration_z": shock,
        "probability_up": probability if recommended else None,
        "probability_interval_95": [low, high] if recommended else None,
        "unconditional_hit_rate": prior,
        "expected_excess_return": None if not recommended or n == 0 else shrunk,
        "expected_excess_return_net": net,
        "hold_trading_days": 21 if recommended else 0,
        "policy_hold": "next calendar month",
        "n_historical_trades": n,
        "rationale": _rationale(shock, reason, shrunk if n else None, net, probability if recommended else None),
    }


def _rationale(shock: float, reason: str, excess, net, probability) -> str:
    z_text = "缺失" if not np.isfinite(shock) else f"{shock:.2f}"
    if reason == "no_signal":
        return f"大陆报纸不确定性冲击 z={z_text}，没有达到 1.0，下一月不做多。"
    if reason == "hold_already_finished":
        return f"最近一期报纸冲击 z={z_text} 对应的持有月已经结束，新的月份还没有报纸指数。"
    if reason == "killed_no_edge":
        return f"冲击 z={z_text} 达到阈值，但已完成月份的平均超额不大于零，不做多。"
    prob = "缺失" if probability is None else f"{probability:.1%}"
    excess_text = "缺失" if excess is None else f"{excess * 10000:.1f} bp"
    net_text = "缺失" if net is None else f"{net * 10000:.1f} bp"
    return (
        f"报纸不确定性冲击 z={z_text}。按冻结规则做多下一自然月，收缩后毛超额约 {excess_text}，"
        f"扣掉 10 bp 往返后约 {net_text}，月收益为正的概率 {prob}。"
    )


def render(summary: dict) -> str:
    def bp(value):
        return "—" if value is None else f"{value * 10000:.1f} bp"

    def pct(value):
        return "—" if value is None else f"{value * 100:.2f}%"

    def num(value, digits=2):
        return "—" if value is None else f"{value:.{digits}f}"

    def row(name):
        return next(item for item in summary["strategies"] if item["name"] == name)

    fast = row("epu_fast")
    slow = row("epu_slow")
    level = row("epu_level")
    calm = row("epu_easing")
    late = row("epu_fast_post2016")
    margin = row("margin_drop_2d")
    checks = summary["sanity"]
    events = summary["events"]
    net_fast = None if fast["excess"] is None else fast["excess"] - 2 * fast["cost_per_side"]
    lines = [
        "# A 股舆论做多：回测报告",
        "",
        "只使用国内信息。价格是新浪财经的沪深 300。舆论是 Davis、Liu、Sheng 用大陆报纸编制的经济政策不确定性，上升表示政策报道更不确定。融资余额来自沪深交易所公开数据，只作对照，不叫做舆论。规则见 `RESEARCH_DESIGN_CN.md`，阈值和持有期没有按这次收益修改。",
        "",
        (
            f"2005 年以来，报纸不确定性跳升后的下一月，沪深 300 毛超额 {bp(fast['excess'])}，"
            f"扣掉 10 bp 往返后 {bp(net_fast)}，交易 t = {num(fast['t_excess'])}，"
            f"循环移位 p = {num(fast['null_p'], 3)}，共 {fast['n_trades']} 个月。"
            f"晚一个月再做的毛超额是 {bp(slow['excess'])}，t = {num(slow['t_excess'])}，"
            f"p = {num(slow['null_p'], 3)}，事先注册为稳健性检查，没有拿来替换下一月规则。"
            f"登记的恐慌事件在收盘后两日平均 {bp(events['panic_2d']['mean'])}。"
        ),
        "",
        "## 数据核对",
        "",
        f"沪深 300 从 {checks['first']} 到 {checks['last']}。报纸指数用到 {checks['epu_last']}。",
        f"2016-01-04 收益 {pct(checks['circuit_20160104'])}，2020-02-03 收益 {pct(checks['covid_20200203'])}。",
        f"2015 年 7 月 {pct(checks['july_2015'])}，8 月 {pct(checks['august_2015'])}，2024 年 9 月 {pct(checks['september_2024'])}。",
        "",
        "## 报纸不确定性",
        "",
        f"快速执行，2005 年以来：{fast['n_trades']} 个月，持有月收益 {bp(fast['mean_trade'])}，"
        f"任意月 {bp(fast['mean_uncond'])}，超额 {bp(fast['excess'])}，t = {num(fast['t_excess'])}，"
        f"命中 {pct(fast['hit_rate'])}，仓位 {pct(fast['exposure'])}，年化 {pct(fast['cagr'])}，"
        f"买入持有年化 {pct(fast['buyhold_cagr'])}，p = {num(fast['null_p'], 3)}。",
        "",
        f"2016 年以来的同一规则：{late['n_trades']} 个月，超额 {bp(late['excess'])}，t = {num(late['t_excess'])}，p = {num(late['null_p'], 3)}。",
        "",
        f"慢速执行：{slow['n_trades']} 个月，超额 {bp(slow['excess'])}，t = {num(slow['t_excess'])}，p = {num(slow['null_p'], 3)}。",
        "",
        f"不确定性处于高位而不是只看跳升：超额 {bp(level['excess'])}，t = {num(level['t_excess'])}，{level['n_trades']} 个月。",
        "",
        f"不确定性明显回落就做多：超额 {bp(calm['excess'])}，t = {num(calm['t_excess'])}，{calm['n_trades']} 个月。",
        "",
        f"回归（下一月收益对当月恶化冲击，并控制当月收益）：恶化系数 {bp(summary['regression'].get('deterioration'))}，"
        f"HAC t = {num(summary['regression'].get('deterioration_t'))}，双侧 p = {num(summary['regression'].get('deterioration_p'), 3)}。",
        "",
        "## 融资余额对照",
        "",
        "这不是舆论。融资余额急降后做多两日："
        f"{margin['n_trades']} 笔，毛超额 {bp(margin['excess'])}，t = {num(margin['t_excess'])}，"
        f"p = {num(margin['null_p'], 3)}，扣费后年化 {pct(margin['cagr'])}。",
        "",
        "## 登记事件",
        "",
        f"共 {events['n']} 个，其中恐慌 {events['n_panic']}、利好 {events['n_euphoria']}、混合 {events['n_mixed']}。",
        f"恐慌当天平均 {bp(events['panic_same']['mean'])}，t = {num(events['panic_same']['t'])}。",
        f"恐慌收盘后 2 日平均 {bp(events['panic_2d']['mean'])}，命中 {pct(events['panic_2d']['hit'])}，t = {num(events['panic_2d']['t'])}。",
        f"恐慌收盘后 10 日平均 {bp(events['panic_10d']['mean'])}，t = {num(events['panic_10d']['t'])}。",
        f"利好当天平均 {bp(events['up_same']['mean'])}；收盘后 5 日平均 {bp(events['up_5d']['mean'])}，t = {num(events['up_5d']['t'])}。"
        "利好公告只有 9 个，这项不替换月频规则。",
        "",
        "## 当天怎么读",
        "",
        "输入日期后，系统用已经公布的最近一个报纸月份决定下一自然月是否做多沪深 300，并给出分数、月收益为正的概率和扣费后的超额。报纸月份还没更新到能决定当前月时，输出空仓。",
        "",
    ]
    for item in summary["scores"]:
        prob = "—" if item["probability_up"] is None else f"{item['probability_up']:.1%}"
        lines.append(
            f"- {item['asof']}：{item['stance']}，分数 {item['bullish_score']:.2f}，概率 {prob}，"
            f"持有至 {item['hold_through']}。{item['rationale']}"
        )
    lines.extend(
        [
            "",
            "## 局限",
            "",
            "这不是投资建议。报纸指数是月频的，不能用来做日内或次日的「第一时间」交易。文件由学术网站发布，实盘拿到的时间可能晚于月末。沪深 300 用的是价格指数。融资余额不能解释成新闻语气。事件是人工挑选的公告日。",
            "",
        ]
    )
    return "\n".join(lines)


def _regression(table: pd.DataFrame) -> dict:
    import statsmodels.api as sm

    y = table["ret"].shift(-1)
    frame = pd.DataFrame(
        {
            "y": y,
            "deterioration": table["epu_shock_z"],
            "ret": table["ret"],
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame.loc[frame.index >= EPU_START]
    if len(frame) < 36:
        return {"error": "too few months", "n": int(len(frame))}
    fitted = sm.OLS(frame["y"], sm.add_constant(frame[["deterioration", "ret"]])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 2}
    )
    return {
        "n": int(fitted.nobs),
        "deterioration": float(fitted.params["deterioration"]),
        "deterioration_t": float(fitted.tvalues["deterioration"]),
        "deterioration_p": float(fitted.pvalues["deterioration"]),
    }


def run(refresh: bool = False) -> dict:
    indices = download_indices(refresh=refresh)
    epu = download_epu(refresh=refresh)
    csi = indices["csi300"]
    table = monthly_frame(csi, epu)
    checks = sanity(csi, table)
    print("cn sanity", checks)
    end = str(table.index.max().date())
    shock = table["epu_shock_z"].to_numpy(dtype=float)
    level = table["epu_level_z"].to_numpy(dtype=float)
    specs = {
        "epu_fast": (np.isfinite(shock) & (shock >= EPU_Z), 0, EPU_START),
        "epu_slow": (np.isfinite(shock) & (shock >= EPU_Z), 1, EPU_START),
        "epu_level": (np.isfinite(level) & (level >= EPU_Z), 0, EPU_START),
        "epu_easing": (np.isfinite(shock) & (shock <= -EPU_Z), 0, EPU_START),
        "epu_fast_post2016": (np.isfinite(shock) & (shock >= EPU_Z), 0, POST_CIRCUIT),
    }
    strategies = []
    for name, (signal, delay, start) in specs.items():
        metrics = run_monthly(table, signal, delay, COST_PER_SIDE, start, end)
        strategies.append({"name": name, **metrics})
        print(name, metrics["n_trades"], metrics["excess"])

    margin = download_margin(refresh=refresh)
    daily = pd.DataFrame({"csi_close": csi}).sort_index()
    daily["margin"] = margin.reindex(daily.index).ffill(limit=5)
    daily["spy_ret"] = daily["csi_close"].pct_change()
    daily["margin_shock_z"] = causal_z(np.log(daily["margin"].where(daily["margin"] > 0)).diff())
    margin_signal = (daily["margin_shock_z"] <= -MARGIN_Z).fillna(False).to_numpy(dtype=bool)
    margin_panel = daily.rename(columns={"csi_close": "spy_close"})
    margin_metrics = evaluate(
        margin_panel,
        margin_signal,
        hold=2,
        delay=0,
        cost=COST_PER_SIDE,
        start=MARGIN_START,
        end=str(daily.index.max().date()),
        n_perm=N_PERM,
        seed=RANDOM_SEED,
    )
    strategies.append({"name": "margin_drop_2d", **margin_metrics})
    print("margin", margin_metrics["n_trades"], margin_metrics["excess"])

    events = event_study(csi)
    panic = events[events["kind"] == "panic"]
    euphoria = events[events["kind"] == "euphoria"]
    event_summary = {
        "n": int(len(events)),
        "n_panic": int(len(panic)),
        "n_euphoria": int(len(euphoria)),
        "n_mixed": int((events["kind"] == "mixed").sum()),
        "panic_same": _ttest(panic["same_day"].to_numpy(dtype=float)),
        "panic_2d": _ttest(panic["fwd_2"].to_numpy(dtype=float)),
        "panic_10d": _ttest(panic["fwd_10"].to_numpy(dtype=float)),
        "up_same": _ttest(euphoria["same_day"].to_numpy(dtype=float)),
        "up_5d": _ttest(euphoria["fwd_5"].to_numpy(dtype=float)),
    }
    regression = _regression(table)
    scores = [
        score_newspaper(table, day)
        for day in ("2015-08-24", "2016-01-07", "2020-02-03", "2024-09-24", "2026-08-31", end)
    ]
    summary = {
        "sanity": checks,
        "strategies": strategies,
        "events": event_summary,
        "regression": regression,
        "scores": scores,
        "production_rule": {
            "market": "沪深300",
            "source": "mainland_newspaper_epu",
            "shock_z": EPU_Z,
            "hold": "next calendar month",
            "cost_per_side": COST_PER_SIDE,
        },
    }
    payload = _jsonify(summary)
    out = RESULTS / "cn"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    pd.DataFrame(strategies).to_csv(out / "strategies.csv", index=False)
    events.to_csv(out / "event_study.csv", index=False)
    (out / "report.md").write_text(render(payload))
    (out / "score_examples.json").write_text(json.dumps(_jsonify(scores), indent=2, ensure_ascii=False))
    _plot(table, specs["epu_fast"][0], out)
    print(f"wrote {out}")
    return payload


def _plot(table: pd.DataFrame, signal: np.ndarray, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rets = table["ret"].to_numpy(dtype=float)
    position = np.zeros(len(rets))
    for i, flagged in enumerate(signal):
        if flagged and i + 1 < len(rets):
            position[i + 1] = 1.0
    prev = np.concatenate(([0.0], position[:-1]))
    strat = np.where(np.isfinite(rets), position * rets, 0.0) - np.abs(position - prev) * COST_PER_SIDE
    bench = np.where(np.isfinite(rets), rets, 0.0)
    mask = table.index >= EPU_START
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(table.index[mask], np.cumprod(1 + bench[mask]), label="CSI 300", linewidth=1.2)
    axis.plot(table.index[mask], np.cumprod(1 + strat[mask]), label="Newspaper EPU, next month", linewidth=1.2)
    axis.set_title("CSI 300 vs mainland-newspaper rule, 5 bp per side")
    axis.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    figure.savefig(out / "equity_monthly.png", dpi=120)
    plt.close(figure)


if __name__ == "__main__":
    run()
