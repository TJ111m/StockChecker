"""Pre-registered backtest. The production rule is not chosen from these results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sentiment_long.config import (
    COST_PER_SIDE,
    COST_STRESS,
    DELAY_FAST,
    DELAY_SLOW,
    DELAY_TETLOCK,
    EXPLORATORY_HOLDS,
    EXPLORATORY_Z,
    HOLD_FAST,
    HOLD_SLOW,
    HOLD_TETLOCK,
    MIN_TRADES,
    N_PERM,
    NEWS_EVAL_START,
    POST_LETTER,
    RANDOM_SEED,
    SHOCK_Z,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
    WIKI_EVAL_START,
)
from sentiment_long.engine import (
    asset_returns,
    cover_nonoverlapping,
    equity_stats,
    excess_tstat,
    iter_fixed_trades,
    kill_switch_signal,
    null_pvalue,
    strategy_returns,
    trade_gross,
    unconditional_gross,
)
from sentiment_long.events import event_study, summarize_events
from sentiment_long.panel import build_panel, sanity_check
from sentiment_long.paths import RESULTS
from sentiment_long.report import render, save_plots
from sentiment_long.scoring import score_agent


def _bool(series: pd.Series) -> np.ndarray:
    return series.fillna(False).to_numpy(dtype=bool)


def make_signals(panel: pd.DataFrame, shock_z: float = SHOCK_Z) -> dict[str, np.ndarray]:
    signals = {
        "news": _bool(panel["news_shock_z"] <= -shock_z),
        "wiki": _bool(panel["wiki_shock_z"] >= shock_z),
        "vix": _bool(panel["vix_shock_z"] >= shock_z),
        "price": _bool(panel["price_shock_z"] <= -shock_z),
        "epu": _bool(panel["epu_shock_z"] >= shock_z),
        "news_momentum": _bool(panel["news_shock_z"] >= shock_z),
    }
    signals["agreement"] = signals["news"] & signals["wiki"]
    return signals


def evaluate(
    panel: pd.DataFrame,
    signal: np.ndarray,
    hold: int,
    delay: int,
    cost: float,
    start: str,
    end: str | None = None,
    n_perm: int = 0,
    seed: int = RANDOM_SEED,
    close_col: str = "spy_close",
) -> dict:
    dates = pd.DatetimeIndex(panel.index)
    close = panel[close_col].to_numpy(dtype=float)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp(dates.max())
    asset = asset_returns(close)
    cover = cover_nonoverlapping(signal, hold, delay)
    strat = strategy_returns(asset, cover, cost)
    bench = np.where(np.isfinite(asset), asset, 0.0)
    mask = np.asarray((dates >= start_ts) & (dates <= end_ts))
    stats = equity_stats(strat[mask])
    bench_stats = equity_stats(bench[mask])
    trades = iter_fixed_trades(dates, close, signal, hold, delay, start_ts, end_ts)
    grosses = np.asarray([trade["gross"] for trade in trades], dtype=float)
    uncond = unconditional_gross(dates, close, hold, start_ts, end_ts)
    trade_stats = excess_tstat(grosses, uncond)

    def _stat(candidate: np.ndarray) -> float:
        subset = iter_fixed_trades(dates, close, candidate, hold, delay, start_ts, end_ts)
        values = np.asarray([trade["gross"] for trade in subset], dtype=float)
        if len(values) < 2 or len(uncond) == 0:
            return np.nan
        return float(np.mean(values) - np.mean(uncond))

    p_value = null_pvalue(_stat, signal, n_perm, seed) if n_perm else None
    return {
        "start": str(start_ts.date()),
        "end": str(end_ts.date()),
        "hold": hold,
        "delay": delay,
        "cost_per_side": cost,
        "n_days": int(mask.sum()),
        "exposure": float(np.mean(cover[mask])) if mask.any() else np.nan,
        "sharpe": stats["sharpe"],
        "cagr": stats["cagr"],
        "maxdd": stats["maxdd"],
        "equity_end": stats["equity_end"],
        "buyhold_sharpe": bench_stats["sharpe"],
        "buyhold_cagr": bench_stats["cagr"],
        "buyhold_maxdd": bench_stats["maxdd"],
        "null_p": p_value,
        **trade_stats,
    }


def _window_mask(dates: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    return np.asarray((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)))


def run_regression(panel: pd.DataFrame, column: str, sign: float, hold: int, delay: int, start: str, end: str) -> dict:
    import statsmodels.api as sm

    dates = pd.DatetimeIndex(panel.index)
    close = panel["spy_close"].to_numpy(dtype=float)
    y = np.asarray([trade_gross(close, t, hold, delay) for t in range(len(close))], dtype=float)
    deterioration = sign * panel[column].to_numpy(dtype=float)
    frame = pd.DataFrame(
        {
            "y": y,
            "deterioration": deterioration,
            "spy_ret": panel["spy_ret"].to_numpy(dtype=float),
        },
        index=dates,
    )
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame.loc[_window_mask(pd.DatetimeIndex(frame.index), start, end)]
    if len(frame) < 100:
        return {"n": int(len(frame)), "error": "too few rows", "start": start, "end": end}
    fitted = sm.OLS(frame["y"], sm.add_constant(frame[["deterioration", "spy_ret"]])).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": max(hold + delay, 3)},
    )
    return {
        "start": start,
        "end": end,
        "hold": hold,
        "delay": delay,
        "n": int(fitted.nobs),
        "params": {key: float(fitted.params[key]) for key in fitted.params.index},
        "tvalues": {key: float(fitted.tvalues[key]) for key in fitted.tvalues.index},
        "pvalues": {key: float(fitted.pvalues[key]) for key in fitted.pvalues.index},
    }


def _ml_frame(panel: pd.DataFrame) -> pd.DataFrame:
    close = panel["spy_close"].to_numpy(dtype=float)
    forward = np.asarray([trade_gross(close, t, HOLD_FAST, DELAY_FAST) for t in range(len(close))])
    shock = panel["news_shock_z"]
    frame = pd.DataFrame(
        {
            "shock": shock,
            "level": panel["news_level_z"],
            "shock1": shock.shift(1),
            "shock2": shock.shift(2),
            "absshock": shock.abs(),
            "forward": forward,
        },
        index=panel.index,
    )
    frame["up"] = (frame["forward"] > 0).astype(float)
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def _fit_predict(model, train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    features = ["shock", "level", "shock1", "shock2", "absshock"]
    mu = train[features].mean()
    sd = train[features].std(ddof=0).replace(0, 1.0)
    x_train = (train[features] - mu) / sd
    x_target = (target[features] - mu) / sd
    model.fit(x_train, train["up"].astype(int))
    proba = model.predict_proba(x_target)
    classes = list(model.classes_)
    return proba[:, classes.index(1)]


def run_ml(panel: pd.DataFrame) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    frame = _ml_frame(panel)
    train = frame.loc[TRAIN_START:TRAIN_END]
    # Drop training rows whose two-day label lands after the validation boundary.
    train_end = pd.Timestamp(TRAIN_END)
    train = train.loc[train.index <= train_end - pd.offsets.BDay(HOLD_FAST)]
    val = frame.loc[VAL_START:VAL_END]
    test = frame.loc[NEWS_EVAL_START:]
    if min(len(train), len(val), len(test)) < 200:
        return {"error": "not enough rows", "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test))}

    base_rate = float(train["up"].mean())
    models = {
        "logistic": LogisticRegression(max_iter=500, random_state=RANDOM_SEED),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(8,),
            alpha=1e-3,
            max_iter=500,
            random_state=RANDOM_SEED,
            learning_rate_init=1e-3,
        ),
    }
    # Signals live on the full panel index; models only fire where a row exists.
    full_index = pd.DatetimeIndex(panel.index)
    out = {"base_rate": base_rate, "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)), "models": {}}
    for name, model in models.items():
        val_p = _fit_predict(model, train, val)
        # Refit is unnecessary; the same coefficients score the test if we predict test from this fit.
        test_p = model.predict_proba(
            (
                test[["shock", "level", "shock1", "shock2", "absshock"]]
                - train[["shock", "level", "shock1", "shock2", "absshock"]].mean()
            )
            / train[["shock", "level", "shock1", "shock2", "absshock"]].std(ddof=0).replace(0, 1.0)
        )[:, list(model.classes_).index(1)]
        for split, idx, proba in (("val", val.index, val_p), ("test", test.index, test_p)):
            signal = np.zeros(len(full_index), dtype=bool)
            positions = full_index.get_indexer(idx)
            signal[positions] = proba > base_rate
            metrics = evaluate(
                panel,
                signal,
                HOLD_FAST,
                DELAY_FAST,
                COST_PER_SIDE,
                VAL_START if split == "val" else NEWS_EVAL_START,
                VAL_END if split == "val" else None,
                n_perm=0,
            )
            out["models"].setdefault(name, {})[split] = {
                "sharpe": metrics["sharpe"],
                "cagr": metrics["cagr"],
                "n_trades": metrics["n_trades"],
                "excess": metrics["excess"],
                "t_excess": metrics["t_excess"],
                "hit_rate": metrics["hit_rate"],
                "exposure": metrics["exposure"],
            }
        n_iter = getattr(model, "n_iter_", 0)
        if isinstance(n_iter, np.ndarray):
            n_iter = int(n_iter.reshape(-1)[0]) if n_iter.size else 0
        out["models"][name]["n_iter"] = int(n_iter or 0)
    rule_val = evaluate(panel, make_signals(panel)["news"], HOLD_FAST, DELAY_FAST, COST_PER_SIDE, VAL_START, VAL_END)
    rule_test = evaluate(panel, make_signals(panel)["news"], HOLD_FAST, DELAY_FAST, COST_PER_SIDE, NEWS_EVAL_START, None)
    out["rule_val_sharpe"] = rule_val["sharpe"]
    out["rule_test"] = {
        "sharpe": rule_test["sharpe"],
        "excess": rule_test["excess"],
        "t_excess": rule_test["t_excess"],
        "n_trades": rule_test["n_trades"],
    }
    out["note"] = "对照模型不替换生产规则。验证段夏普只用来描述，不用来改阈值。"
    return out


def _specs() -> list[dict]:
    rows = [
        ("news", "fast", DELAY_FAST, HOLD_FAST, "confirmatory", NEWS_EVAL_START, None, N_PERM),
        ("news", "fast", DELAY_FAST, HOLD_FAST, "confirmatory", POST_LETTER, None, N_PERM),
        ("news", "slow", DELAY_SLOW, HOLD_SLOW, "confirmatory", POST_LETTER, None, N_PERM),
        ("news", "tetlock", DELAY_TETLOCK, HOLD_TETLOCK, "confirmatory", POST_LETTER, None, 0),
        ("wiki", "fast", DELAY_FAST, HOLD_FAST, "confirmatory", WIKI_EVAL_START, None, N_PERM),
        ("wiki", "slow", DELAY_SLOW, HOLD_SLOW, "confirmatory", WIKI_EVAL_START, None, 0),
        ("vix", "fast", DELAY_FAST, HOLD_FAST, "control", NEWS_EVAL_START, None, N_PERM),
        ("price", "fast", DELAY_FAST, HOLD_FAST, "control", NEWS_EVAL_START, None, N_PERM),
        ("news", "fast", DELAY_FAST, HOLD_FAST, "horse_race", WIKI_EVAL_START, None, 0),
        ("vix", "fast", DELAY_FAST, HOLD_FAST, "horse_race", WIKI_EVAL_START, None, 0),
        ("price", "fast", DELAY_FAST, HOLD_FAST, "horse_race", WIKI_EVAL_START, None, 0),
        ("news_momentum", "fast", DELAY_FAST, HOLD_FAST, "negative_control", NEWS_EVAL_START, None, 0),
        ("news_momentum", "next_day", DELAY_FAST, 1, "negative_control", NEWS_EVAL_START, None, 0),
        ("epu", "fast", DELAY_FAST, HOLD_FAST, "exploratory", NEWS_EVAL_START, None, 0),
        ("agreement", "fast", DELAY_FAST, HOLD_FAST, "exploratory", WIKI_EVAL_START, None, 0),
    ]
    specs = []
    for source, timing, delay, hold, role, start, end, perms in rows:
        specs.append(
            {
                "source": source,
                "timing": timing,
                "delay": delay,
                "hold": hold,
                "role": role,
                "start": start,
                "end": end,
                "n_perm": perms,
                "shock_z": SHOCK_Z,
                "cost_per_side": COST_PER_SIDE,
            }
        )
    for source in ("news", "wiki"):
        for shock in EXPLORATORY_Z:
            specs.append(
                {
                    "source": source,
                    "timing": f"fast_z{shock}",
                    "delay": DELAY_FAST,
                    "hold": HOLD_FAST,
                    "role": "exploratory",
                    "start": NEWS_EVAL_START if source == "news" else WIKI_EVAL_START,
                    "end": None,
                    "n_perm": 0,
                    "shock_z": shock,
                    "cost_per_side": COST_PER_SIDE,
                }
            )
        for hold in EXPLORATORY_HOLDS:
            specs.append(
                {
                    "source": source,
                    "timing": f"hold{hold}",
                    "delay": DELAY_FAST,
                    "hold": hold,
                    "role": "exploratory",
                    "start": NEWS_EVAL_START if source == "news" else WIKI_EVAL_START,
                    "end": None,
                    "n_perm": 0,
                    "shock_z": SHOCK_Z,
                    "cost_per_side": COST_PER_SIDE,
                }
            )
    for cost, label in ((0.0, "cost0"), (COST_STRESS, "cost10bp_side")):
        for source, start in (("news", NEWS_EVAL_START), ("wiki", WIKI_EVAL_START)):
            specs.append(
                {
                    "source": source,
                    "timing": label,
                    "delay": DELAY_FAST,
                    "hold": HOLD_FAST,
                    "role": "cost_sensitivity",
                    "start": start,
                    "end": None,
                    "n_perm": 0,
                    "shock_z": SHOCK_Z,
                    "cost_per_side": cost,
                }
            )
    return specs


def _jsonify(value):
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if value is None or isinstance(value, str):
        return value
    return str(value)


def run(refresh: bool = False) -> dict:
    panel = build_panel(refresh=refresh)
    checks = sanity_check(panel)
    print("sanity", checks)
    end = str(panel.index.max().date())
    signal_cache: dict[float, dict[str, np.ndarray]] = {}

    def signals_for(shock: float) -> dict[str, np.ndarray]:
        if shock not in signal_cache:
            signal_cache[shock] = make_signals(panel, shock)
        return signal_cache[shock]

    rows = []
    for spec in _specs():
        signal = signals_for(spec["shock_z"])[spec["source"]]
        metrics = evaluate(
            panel,
            signal,
            spec["hold"],
            spec["delay"],
            spec["cost_per_side"],
            spec["start"],
            spec["end"] or end,
            n_perm=spec["n_perm"],
            seed=RANDOM_SEED + len(rows),
        )
        rows.append({**{key: spec[key] for key in ("source", "timing", "role", "shock_z")}, **metrics})
        print(
            f"{spec['source']:16} {spec['timing']:14} {metrics['start']}  "
            f"trades={metrics['n_trades']:4} excess={metrics['excess']}"
        )

    dates = pd.DatetimeIndex(panel.index)
    close = panel["spy_close"].to_numpy(dtype=float)
    base_signals = signals_for(SHOCK_Z)
    for source, start in (("news", NEWS_EVAL_START), ("wiki", WIKI_EVAL_START)):
        eligible = int(np.searchsorted(dates, pd.Timestamp(start)))
        gated = kill_switch_signal(
            base_signals[source], close, HOLD_FAST, DELAY_FAST, MIN_TRADES, eligible_from=eligible
        )
        metrics = evaluate(panel, gated, HOLD_FAST, DELAY_FAST, COST_PER_SIDE, start, end, n_perm=0)
        rows.append({"source": source, "timing": "fast_kill_switch", "role": "overlay", "shock_z": SHOCK_Z, **metrics})

    regressions = [
        {
            "name": "wiki_fast_common",
            "series": "wiki",
            **run_regression(panel, "wiki_shock_z", 1.0, HOLD_FAST, DELAY_FAST, WIKI_EVAL_START, end),
        },
        {
            "name": "news_fast_post_letter",
            "series": "news",
            **run_regression(panel, "news_shock_z", -1.0, HOLD_FAST, DELAY_FAST, POST_LETTER, end),
        },
        {
            "name": "news_slow_post_letter",
            "series": "news",
            **run_regression(panel, "news_shock_z", -1.0, HOLD_SLOW, DELAY_SLOW, POST_LETTER, end),
        },
        {
            "name": "news_fast_from_2010",
            "series": "news",
            **run_regression(panel, "news_shock_z", -1.0, HOLD_FAST, DELAY_FAST, NEWS_EVAL_START, end),
        },
        {
            "name": "vix_fast_from_2010",
            "series": "vix",
            **run_regression(panel, "vix_shock_z", 1.0, HOLD_FAST, DELAY_FAST, NEWS_EVAL_START, end),
        },
        {
            "name": "price_fast_from_2010",
            "series": "price",
            **run_regression(panel, "price_shock_z", -1.0, HOLD_FAST, DELAY_FAST, NEWS_EVAL_START, end),
        },
    ]

    events = event_study(panel)
    event_summary = summarize_events(events)
    print("events", event_summary["n_events"], "panic 2d", event_summary["panic_2d"])
    ml = run_ml(panel)
    print("ml", json.dumps(_jsonify(ml))[:500])

    # Daily returns for the figures.
    asset = asset_returns(close)
    curve = pd.DataFrame({"date": dates, "spy": np.where(np.isfinite(asset), asset, 0.0)})
    for key, delay, hold in (
        ("news_fast", DELAY_FAST, HOLD_FAST),
        ("news_slow", DELAY_SLOW, HOLD_SLOW),
        ("wiki_fast", DELAY_FAST, HOLD_FAST),
        ("vix_fast", DELAY_FAST, HOLD_FAST),
        ("price_fast", DELAY_FAST, HOLD_FAST),
    ):
        source = key.split("_")[0]
        cover = cover_nonoverlapping(base_signals[source], hold, delay)
        curve[key] = strategy_returns(asset, cover, COST_PER_SIDE)

    score_dates = [
        "2018-12-24",
        "2020-03-16",
        "2020-03-23",
        "2024-08-05",
        "2025-01-27",
        "2025-04-03",
        "2025-04-09",
        "2026-06-05",
        end,
    ]
    scores = [score_agent(panel, day) for day in score_dates]

    summary = {
        "sanity": checks,
        "strategies": rows,
        "regressions": regressions,
        "events": event_summary,
        "ml": ml,
        "scores": scores,
        "production_rule": {
            "shock_z": SHOCK_Z,
            "hold": HOLD_FAST,
            "delay": DELAY_FAST,
            "cost_per_side": COST_PER_SIDE,
            "direction": "contrarian_on_deterioration",
            "ml_replaces_rule": False,
        },
    }
    payload = _jsonify(summary)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    pd.DataFrame(rows).to_csv(RESULTS / "strategies.csv", index=False)
    events.to_csv(RESULTS / "event_study.csv", index=False)
    curve.to_csv(RESULTS / "daily_returns.csv", index=False)
    (RESULTS / "score_examples.json").write_text(json.dumps(_jsonify(scores), indent=2, ensure_ascii=False))
    (RESULTS / "report.md").write_text(render(payload))
    save_plots(curve, events, RESULTS)
    print(f"wrote {RESULTS}")
    return payload


if __name__ == "__main__":
    run()
