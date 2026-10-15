"""Position construction and performance. Timing is deliberately strict.

A signal on row t may only earn asset returns on rows strictly after t.
`delay` inserts extra sessions on top of that one-session execution lag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cover_nonoverlapping(signal: np.ndarray, hold: int, delay: int) -> np.ndarray:
    """Hold exactly `hold` sessions, then ignore new signals until that hold ends.

    This is the same trade list as `iter_fixed_trades`. A clustered shock does not
    extend the hold past the first signal's window.
    """
    signal = np.asarray(signal, dtype=bool)
    n = len(signal)
    cover = np.zeros(n, dtype=float)
    t = 0
    while t < n:
        if signal[t]:
            start = t + 1 + delay
            if start < n:
                cover[start : min(n, start + hold)] = 1.0
            exit_idx = t + delay + hold
            t = exit_idx + 1 if exit_idx >= t else t + 1
        else:
            t += 1
    return cover


def cover_from_signal(signal: np.ndarray, hold: int, delay: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=bool)
    n = len(signal)
    cover = np.zeros(n, dtype=float)
    if hold < 1 or delay < 0:
        raise ValueError("hold must be >= 1 and delay >= 0")
    for t in range(n):
        if not signal[t]:
            continue
        start = t + 1 + delay
        if start >= n:
            continue
        end = min(n, start + hold)
        cover[start:end] = 1.0
    return cover


def asset_returns(close: np.ndarray) -> np.ndarray:
    close = np.asarray(close, dtype=float)
    out = np.full(len(close), np.nan)
    prev = close[:-1]
    nxt = close[1:]
    ok = np.isfinite(prev) & np.isfinite(nxt) & (prev > 0)
    out[1:] = np.where(ok, nxt / prev - 1.0, np.nan)
    return out


def strategy_returns(asset_ret: np.ndarray, cover: np.ndarray, cost_per_side: float) -> np.ndarray:
    asset_ret = np.asarray(asset_ret, dtype=float)
    cover = np.asarray(cover, dtype=float)
    prev = np.concatenate(([0.0], cover[:-1]))
    turnover = np.abs(cover - prev)
    earned = np.where(np.isfinite(asset_ret), cover * asset_ret, 0.0)
    return earned - turnover * float(cost_per_side)


def trade_gross(close: np.ndarray, t: int, hold: int, delay: int) -> float:
    """Simple return from the entry close to `hold` sessions later.

    Entry close index is t+delay. With delay 0 the trade earns close[t+hold]/close[t]-1,
    which is exactly the asset returns on t+1 ... t+hold.
    """
    close = np.asarray(close, dtype=float)
    entry = t + delay
    last = entry + hold
    if entry < 0 or last >= len(close):
        return np.nan
    px0 = close[entry]
    px1 = close[last]
    if not np.isfinite(px0) or not np.isfinite(px1) or px0 <= 0:
        return np.nan
    return float(px1 / px0 - 1.0)


def iter_fixed_trades(
    dates: pd.DatetimeIndex,
    close: np.ndarray,
    signal: np.ndarray,
    hold: int,
    delay: int,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> list[dict]:
    """Non-overlapping fixed-hold trades. The next signal is eligible only after exit."""
    signal = np.asarray(signal, dtype=bool)
    dates = pd.DatetimeIndex(dates)
    n = len(close)
    trades: list[dict] = []
    t = 0
    while t < n:
        if signal[t]:
            entry = t + delay
            exit_idx = entry + hold
            gross = trade_gross(close, t, hold, delay)
            entry_date = dates[entry] if 0 <= entry < n else None
            exit_date = dates[exit_idx] if 0 <= exit_idx < n else None
            in_window = True
            if start is not None and (entry_date is None or entry_date < start):
                in_window = False
            if end is not None and (entry_date is None or entry_date > end):
                in_window = False
            if end is not None and (exit_date is None or exit_date > end):
                in_window = False
            if in_window and np.isfinite(gross) and entry_date is not None and exit_date is not None:
                trades.append(
                    {
                        "signal_date": dates[t],
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "gross": gross,
                        "signal_index": t,
                        "exit_index": exit_idx,
                    }
                )
            t = exit_idx + 1 if exit_idx >= t else t + 1
        else:
            t += 1
    return trades


def unconditional_gross(
    dates: pd.DatetimeIndex,
    close: np.ndarray,
    hold: int,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> np.ndarray:
    close = np.asarray(close, dtype=float)
    dates = pd.DatetimeIndex(dates)
    vals = []
    for entry in range(len(close) - hold):
        last = entry + hold
        if start is not None and dates[entry] < start:
            continue
        if end is not None and dates[last] > end:
            continue
        px0 = close[entry]
        px1 = close[last]
        if np.isfinite(px0) and np.isfinite(px1) and px0 > 0:
            vals.append(px1 / px0 - 1.0)
    return np.asarray(vals, dtype=float)


def equity_stats(daily: np.ndarray) -> dict:
    daily = np.asarray(daily, dtype=float)
    daily = np.where(np.isfinite(daily), daily, 0.0)
    if len(daily) == 0:
        return {"cagr": np.nan, "sharpe": np.nan, "maxdd": np.nan, "equity_end": np.nan}
    equity = np.cumprod(1.0 + daily)
    years = len(daily) / 252.0
    end = float(equity[-1])
    cagr = end ** (1.0 / years) - 1.0 if years > 0 and end > 0 else np.nan
    sd = float(np.std(daily, ddof=1)) if len(daily) > 2 else np.nan
    mu = float(np.mean(daily))
    sharpe = mu / sd * np.sqrt(252.0) if sd and np.isfinite(sd) and sd > 0 else np.nan
    peak = np.maximum.accumulate(equity)
    maxdd = float(np.min(equity / peak - 1.0))
    return {"cagr": cagr, "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan, "maxdd": maxdd, "equity_end": end}


def excess_tstat(trade_grosses: np.ndarray, uncond: np.ndarray) -> dict:
    g = np.asarray(trade_grosses, dtype=float)
    g = g[np.isfinite(g)]
    u = np.asarray(uncond, dtype=float)
    u = u[np.isfinite(u)]
    if len(g) == 0 or len(u) == 0:
        return {
            "n_trades": int(len(g)),
            "hit_rate": np.nan,
            "mean_trade": np.nan,
            "mean_uncond": np.nan,
            "excess": np.nan,
            "t_excess": np.nan,
        }
    mean_g = float(np.mean(g))
    mean_u = float(np.mean(u))
    excess = mean_g - mean_u
    if len(g) >= 2:
        se = float(np.std(g, ddof=1) / np.sqrt(len(g)))
        t_excess = excess / se if se > 0 else np.nan
    else:
        t_excess = np.nan
    return {
        "n_trades": int(len(g)),
        "hit_rate": float(np.mean(g > 0)),
        "mean_trade": mean_g,
        "mean_uncond": mean_u,
        "excess": float(excess),
        "t_excess": float(t_excess) if np.isfinite(t_excess) else np.nan,
    }


def kill_switch_signal(
    signal: np.ndarray,
    close: np.ndarray,
    hold: int,
    delay: int,
    min_trades: int,
    eligible_from: int = 0,
) -> np.ndarray:
    """Turn the signal off once enough past ungated trades have non-positive excess.

    History is the ungated non-overlapping trade list, so the switch does not
    rewrite its own evidence. Trades are visible only on their exit index.
    """
    signal = np.asarray(signal, dtype=bool)
    close = np.asarray(close, dtype=float)
    n = len(signal)
    gross = np.full(n, np.nan)
    for t in range(n):
        gross[t] = trade_gross(close, t, hold, delay)
    uncond = np.full(n, np.nan)
    for entry in range(n - hold):
        px0 = close[entry]
        px1 = close[entry + hold]
        if np.isfinite(px0) and np.isfinite(px1) and px0 > 0:
            uncond[entry] = px1 / px0 - 1.0

    completed: list[tuple[int, float]] = []
    t = 0
    while t < n:
        if t >= eligible_from and signal[t] and np.isfinite(gross[t]):
            end_idx = t + delay + hold
            completed.append((end_idx, float(gross[t])))
            t = end_idx + 1
        else:
            t += 1

    gated = signal.copy()
    j = 0
    done_sum = 0.0
    done_n = 0
    e_ptr = 0
    u_sum = 0.0
    u_n = 0
    for t in range(n):
        while e_ptr + hold <= t and e_ptr < n:
            if e_ptr >= eligible_from and np.isfinite(uncond[e_ptr]):
                u_sum += float(uncond[e_ptr])
                u_n += 1
            e_ptr += 1
        while j < len(completed) and completed[j][0] <= t:
            done_sum += completed[j][1]
            done_n += 1
            j += 1
        if signal[t] and t >= eligible_from and done_n >= min_trades and u_n > 0:
            excess = done_sum / done_n - u_sum / u_n
            if excess <= 0:
                gated[t] = False
    return gated


def null_pvalue(stat_fn, signal: np.ndarray, n_perm: int, seed: int) -> float | None:
    signal = np.asarray(signal, dtype=bool)
    actual = stat_fn(signal)
    if actual is None or not np.isfinite(actual) or n_perm <= 0:
        return None
    rng = np.random.default_rng(seed)
    n = len(signal)
    if n < 3:
        return None
    beats = 0
    for _ in range(n_perm):
        shifted = np.roll(signal, int(rng.integers(1, n)))
        value = stat_fn(shifted)
        if value is not None and np.isfinite(value) and value >= actual:
            beats += 1
    return float((beats + 1) / (n_perm + 1))
