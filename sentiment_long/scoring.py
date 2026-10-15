"""Turn a point-in-time opinion series into a long / flat decision.

The hold is the pre-registered two sessions. It is not the horizon with the
best in-sample return. Probabilities are shrunk hit rates of past trades.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta

from sentiment_long.config import (
    COST_PER_SIDE,
    DELAY_FAST,
    EVENT_HORIZONS,
    HOLD_FAST,
    MIN_TRADES,
    NEWS_EVAL_START,
    PRIOR_SCORE_CAP,
    PRIOR_STRENGTH,
    SCORE_SCALE,
    SHOCK_Z,
    WIKI_EVAL_START,
)
from sentiment_long.engine import (
    excess_tstat,
    iter_fixed_trades,
    kill_switch_signal,
    unconditional_gross,
)
from sentiment_long.features import causal_z

SOURCES = {
    "news": ("news_shock_z", -1.0, NEWS_EVAL_START, "news_tone"),
    "wiki": ("wiki_shock_z", 1.0, WIKI_EVAL_START, "attention"),
    "vix": ("vix_shock_z", 1.0, NEWS_EVAL_START, "market_fear_control"),
    "price": ("price_shock_z", -1.0, NEWS_EVAL_START, "price_reversal_control"),
    "epu": ("epu_shock_z", 1.0, NEWS_EVAL_START, "uncertainty"),
}


def snap_date(index: pd.DatetimeIndex, asof: str | pd.Timestamp | None) -> pd.Timestamp:
    index = pd.DatetimeIndex(index)
    if asof is None:
        return pd.Timestamp(index.max())
    stamp = pd.Timestamp(asof)
    eligible = index[index <= stamp]
    if len(eligible) == 0:
        raise ValueError(f"no observation on or before {stamp.date()}")
    return pd.Timestamp(eligible.max())


def _posterior(successes: int, n: int, prior_hit: float) -> tuple[float, float, float]:
    prior_hit = float(np.clip(prior_hit, 0.01, 0.99))
    alpha = prior_hit * PRIOR_STRENGTH
    beta_param = (1.0 - prior_hit) * PRIOR_STRENGTH
    point = (successes + alpha) / (n + PRIOR_STRENGTH)
    low = float(beta.ppf(0.025, successes + alpha, n - successes + beta_param))
    high = float(beta.ppf(0.975, successes + alpha, n - successes + beta_param))
    return float(point), low, high


def _completed(trades: list[dict], asof: pd.Timestamp, start: pd.Timestamp) -> list[dict]:
    return [
        trade
        for trade in trades
        if trade["entry_date"] >= start and trade["exit_date"] <= asof
    ]


def _horizon_profile(dates, close, signal, asof: pd.Timestamp, start: pd.Timestamp) -> list[dict]:
    """Past excess by hold length. Descriptive only; the decision hold stays at 2."""
    rows = []
    for hold in EVENT_HORIZONS:
        trades = _completed(
            iter_fixed_trades(dates, close, signal, hold, DELAY_FAST, start, asof),
            asof,
            start,
        )
        uncond = unconditional_gross(dates, close, hold, start, asof)
        stats = excess_tstat(np.asarray([trade["gross"] for trade in trades], dtype=float), uncond)
        rows.append(
            {
                "hold_trading_days": hold,
                "n_trades": stats["n_trades"],
                "mean_excess_return": stats["excess"],
                "hit_rate": stats["hit_rate"],
                "used_for_decision": hold == HOLD_FAST,
            }
        )
    return rows


def _decision(signal: bool, n_trades: int, unshrunk_excess: float, shrunk_excess: float, deterioration: float, veto: bool) -> dict:
    if veto:
        return {"recommended": False, "reason": "fundamental_veto", "bullish_score": 0.0}
    if not signal:
        return {"recommended": False, "reason": "no_signal", "bullish_score": 0.0}
    if n_trades >= MIN_TRADES and not (np.isfinite(unshrunk_excess) and unshrunk_excess > 0):
        return {"recommended": False, "reason": "killed_no_edge", "bullish_score": 0.0}
    if n_trades < MIN_TRADES:
        intensity = min(1.0, max(0.0, deterioration - SHOCK_Z) / SHOCK_Z)
        return {
            "recommended": True,
            "reason": "literature_prior",
            "bullish_score": float(intensity * PRIOR_SCORE_CAP),
        }
    score = float(np.clip(max(shrunk_excess, 0.0) / SCORE_SCALE, 0.0, 1.0))
    return {"recommended": True, "reason": "historical_edge", "bullish_score": score}


def _pack(
    *,
    asof: pd.Timestamp,
    source: str,
    kind: str,
    deterioration: float,
    raw_z: float,
    signal: bool,
    trades: list[dict],
    uncond: np.ndarray,
    veto: bool,
    profile: list[dict],
) -> dict:
    gross = np.asarray([trade["gross"] for trade in trades], dtype=float)
    stats = excess_tstat(gross, uncond)
    n = stats["n_trades"]
    mean_u = stats["mean_uncond"] if n else (float(np.mean(uncond)) if len(uncond) else np.nan)
    if n == 0 or not np.isfinite(stats["mean_trade"]) or not np.isfinite(mean_u):
        shrunk_excess = 0.0
        shrunk_mean = mean_u
    else:
        weight = 30
        shrunk_mean = (n * stats["mean_trade"] + weight * mean_u) / (n + weight)
        shrunk_excess = float(shrunk_mean - mean_u)
    unshrunk = stats["excess"]
    decision = _decision(signal, n, unshrunk if np.isfinite(unshrunk) else -1.0, shrunk_excess, deterioration, veto)
    prior_hit = float(np.mean(uncond > 0)) if len(uncond) else 0.5
    successes = int(np.sum(gross > 0)) if n else 0
    probability, low, high = _posterior(successes, n, prior_hit)
    recommended = decision["recommended"]
    return {
        "asof": str(asof.date()),
        "source": source,
        "source_kind": kind,
        "ticker": "SPY",
        "recommended": recommended,
        "stance": "long" if recommended else "flat",
        "reason": decision["reason"],
        "bullish_score": decision["bullish_score"],
        "deterioration_z": None if not np.isfinite(deterioration) else float(deterioration),
        "raw_shock_z": None if not np.isfinite(raw_z) else float(raw_z),
        "signal": bool(signal),
        "probability_up": probability if recommended else None,
        "probability_interval_95": [low, high] if recommended else None,
        "historical_probability_up": probability,
        "historical_probability_interval_95": [low, high],
        "unconditional_hit_rate": prior_hit,
        "expected_horizon_return": None if not recommended or n == 0 else float(shrunk_mean),
        "expected_excess_return": None if not recommended or n == 0 else float(shrunk_excess),
        "expected_excess_return_net": None
        if not recommended or n == 0
        else float(shrunk_excess - 2 * COST_PER_SIDE),
        "unshrunk_excess_return": None if not np.isfinite(unshrunk) else float(unshrunk),
        "hold_trading_days": HOLD_FAST if recommended else 0,
        "policy_hold_trading_days": HOLD_FAST,
        "hold_calendar_days_approx": int(round(HOLD_FAST * 7 / 5)) if recommended else 0,
        "n_historical_trades": n,
        "execution": "enter at the signal close, hold the next 2 trading sessions",
        "horizon_profile_not_used_for_decision": profile,
        "bullish_score_scale": "1.0 = 100 bp of shrunk excess over the 2-day hold; literature-prior scores are capped at 0.3",
    }


def score_source(panel: pd.DataFrame, source: str, asof: str | None = None, veto: bool = False) -> dict:
    if source not in SOURCES:
        raise KeyError(f"unknown source {source}")
    column, sign, start, kind = SOURCES[source]
    dates = pd.DatetimeIndex(panel.index)
    asof_ts = snap_date(dates, asof)
    start_ts = pd.Timestamp(start)
    loc = int(dates.get_loc(asof_ts))
    raw = panel[column].to_numpy(dtype=float)
    deterioration = sign * raw
    signal = np.isfinite(deterioration) & (deterioration >= SHOCK_Z)
    close = panel["spy_close"].to_numpy(dtype=float)
    eligible = int(dates.searchsorted(start_ts))
    gated = kill_switch_signal(signal, close, HOLD_FAST, DELAY_FAST, MIN_TRADES, eligible_from=eligible)
    # The boolean `signal` above is the shock. The kill switch may turn it off.
    shock_on = bool(signal[loc])
    killed = shock_on and not bool(gated[loc])
    trades = _completed(
        iter_fixed_trades(dates, close, signal, HOLD_FAST, DELAY_FAST, start_ts, asof_ts),
        asof_ts,
        start_ts,
    )
    uncond = unconditional_gross(dates, close, HOLD_FAST, start_ts, asof_ts)
    profile = _horizon_profile(dates, close, signal, asof_ts, start_ts)
    packed = _pack(
        asof=asof_ts,
        source=source,
        kind=kind,
        deterioration=float(deterioration[loc]) if np.isfinite(deterioration[loc]) else np.nan,
        raw_z=float(raw[loc]) if np.isfinite(raw[loc]) else np.nan,
        signal=shock_on and not killed,
        trades=trades,
        uncond=uncond,
        veto=veto,
        profile=profile,
    )
    if killed and not veto:
        packed["recommended"] = False
        packed["stance"] = "flat"
        packed["reason"] = "killed_no_edge"
        packed["bullish_score"] = 0.0
        packed["probability_up"] = None
        packed["probability_interval_95"] = None
        packed["expected_horizon_return"] = None
        packed["expected_excess_return"] = None
        packed["hold_trading_days"] = 0
        packed["hold_calendar_days_approx"] = 0
        packed["signal"] = True
    packed["rationale"] = _rationale(packed)
    return packed


def _rationale(row: dict) -> str:
    z = row.get("deterioration_z")
    z_text = "缺失" if z is None else f"{z:.2f}"
    reason = row["reason"]
    if reason == "no_signal":
        if z is None:
            return "这一天没有可用的舆论冲击，不做多。"
        return f"恶化冲击 z={z_text}，没有达到事先注册的 {SHOCK_Z:.1f}，不做多。"
    if reason == "killed_no_edge":
        return (
            f"恶化冲击 z={z_text} 已达到阈值，但这套规则在评估样本里已经完成的交易"
            "平均超额不大于零，停止做多。"
        )
    if reason == "fundamental_veto":
        return "公司基本面标题的净语气为负。文献里这是低配而不是抄底，今天不做多。"
    if reason == "no_direction_survived_selection":
        return "在前 60% 样本里，逆向和顺势两个方向的超额都不为正，后面的日子不做多。"
    if reason == "literature_prior":
        return (
            f"恶化冲击 z={z_text}。评估样本里完成的交易还不到 {MIN_TRADES} 笔，"
            f"按文献先验持有 {HOLD_FAST} 个交易日，分数封顶 {PRIOR_SCORE_CAP}。"
        )
    excess = row.get("expected_excess_return")
    excess_text = "缺失" if excess is None else f"{excess * 10000:.1f} bp"
    net = row.get("expected_excess_return_net")
    net_text = "缺失" if net is None else f"{net * 10000:.1f} bp"
    prob = row.get("probability_up")
    prob_text = "缺失" if prob is None else f"{prob:.1%}"
    return (
        f"恶化冲击 z={z_text}。按冻结的两日逆向规则做多，收缩后的毛超额约 {excess_text}，"
        f"扣除事先注册的 10 bp 往返后约 {net_text}，上涨概率 {prob_text}。"
        "概率是以往不重叠交易的命中率，不是模型当场的信心。"
    )


def score_agent(panel: pd.DataFrame, asof: str | None = None) -> dict:
    sources = {name: score_source(panel, name, asof) for name in ("news", "wiki", "vix", "price")}
    primary = sources["news"]
    wiki_long = sources["wiki"]["recommended"]
    news_long = primary["recommended"]
    if news_long and wiki_long:
        agreement = "both_long"
    elif news_long:
        agreement = "news_only"
    elif wiki_long:
        agreement = "wiki_only"
    else:
        agreement = "both_flat"
    return {
        "asof": primary["asof"],
        "recommended": news_long,
        "stance": "long" if news_long else "flat",
        "primary_source": "news",
        "bullish_score": primary["bullish_score"],
        "probability_up": primary["probability_up"],
        "probability_interval_95": primary["probability_interval_95"],
        "expected_excess_return": primary["expected_excess_return"],
        "expected_horizon_return": primary["expected_horizon_return"],
        "hold_trading_days": primary["hold_trading_days"],
        "policy_hold_trading_days": HOLD_FAST,
        "agreement": agreement,
        "rationale": primary["rationale"],
        "sources": sources,
        "vintage_warning": (
            "旧金山联储新闻语气是最新修订版，不是当年当天已经公布的版本。"
            "维基关注度、VIX 和价格是时点数据，用来核对这条新闻语气规则。"
        ),
    }


def score_mood_frame(
    mood: pd.Series,
    close: pd.Series,
    asof: str | None = None,
    select_direction: bool = False,
    veto: bool = False,
    source_name: str = "custom",
) -> dict:
    """Score a user series. Higher `mood` means more bullish opinion.

    Direction stays contrarian unless `select_direction` is set. That option
    picks the direction on the first 60 percent and only quotes trades after the split.
    """
    frame = pd.concat({"mood": mood.astype(float), "close": close.astype(float)}, axis=1).dropna()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None).normalize()
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    if len(frame) < Z_WINDOW_SAFE:
        raise ValueError("need at least a year of overlapping mood and price history")
    shock = causal_z(frame["mood"].diff())
    dates = pd.DatetimeIndex(frame.index)
    asof_ts = snap_date(dates, asof)
    close_arr = frame["close"].to_numpy(dtype=float)
    contrarian = _bool_from_series(shock <= -SHOCK_Z)
    momentum = _bool_from_series(shock >= SHOCK_Z)
    split_loc = int(len(frame) * 0.6)
    split_date = pd.Timestamp(dates[split_loc])
    direction = "contrarian"
    evidence_start = pd.Timestamp(dates[Z_WINDOW_SAFE])
    in_selection = False
    if select_direction:
        in_selection = asof_ts <= split_date
        if in_selection:
            direction = "contrarian"
            evidence_start = pd.Timestamp(dates[Z_WINDOW_SAFE])
        else:
            direction = _choose_direction(dates, close_arr, contrarian, momentum, split_date)
            evidence_start = split_date + pd.Timedelta(days=1)
    if direction is None:
        signal = np.zeros(len(frame), dtype=bool)
    else:
        signal = contrarian if direction == "contrarian" else momentum
    loc = int(dates.get_loc(asof_ts))
    raw = shock.to_numpy(dtype=float)
    deterioration = -raw if direction != "momentum" else raw
    shock_on = bool(signal[loc]) if direction is not None else False
    trades = _completed(
        iter_fixed_trades(dates, close_arr, signal, HOLD_FAST, DELAY_FAST, evidence_start, asof_ts),
        asof_ts,
        evidence_start,
    )
    uncond = unconditional_gross(dates, close_arr, HOLD_FAST, evidence_start, asof_ts)
    profile = _horizon_profile(dates, close_arr, signal, asof_ts, evidence_start) if direction else []
    packed = _pack(
        asof=asof_ts,
        source=source_name,
        kind="custom_mood",
        deterioration=float(deterioration[loc]) if np.isfinite(deterioration[loc]) else np.nan,
        raw_z=float(raw[loc]) if np.isfinite(raw[loc]) else np.nan,
        signal=shock_on and direction is not None,
        trades=trades,
        uncond=uncond,
        veto=veto,
        profile=profile,
    )
    if direction is None:
        packed["recommended"] = False
        packed["stance"] = "flat"
        packed["reason"] = "no_direction_survived_selection"
        packed["bullish_score"] = 0.0
        packed["hold_trading_days"] = 0
        packed["probability_up"] = None
    packed["direction"] = direction or "none"
    packed["select_direction"] = select_direction
    packed["in_selection_sample"] = in_selection
    packed["rationale"] = _rationale(packed)
    return packed


Z_WINDOW_SAFE = 252


def _bool_from_series(series: pd.Series) -> np.ndarray:
    return series.fillna(False).to_numpy(dtype=bool)


def _choose_direction(dates, close, contrarian, momentum, split_date: pd.Timestamp) -> str | None:
    start = pd.Timestamp(dates[Z_WINDOW_SAFE])

    def excess(signal) -> float:
        trades = _completed(
            iter_fixed_trades(dates, close, signal, HOLD_FAST, DELAY_FAST, start, split_date),
            split_date,
            start,
        )
        uncond = unconditional_gross(dates, close, HOLD_FAST, start, split_date)
        stats = excess_tstat(np.asarray([trade["gross"] for trade in trades], dtype=float), uncond)
        return stats["excess"] if stats["n_trades"] >= MIN_TRADES and np.isfinite(stats["excess"]) else -np.inf

    contrarian_excess = excess(contrarian)
    momentum_excess = excess(momentum)
    if contrarian_excess <= 0 and momentum_excess <= 0:
        return None
    return "contrarian" if contrarian_excess >= momentum_excess else "momentum"
