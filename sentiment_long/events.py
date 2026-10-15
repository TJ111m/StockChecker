"""Event study on the hand-dated opinion episodes. Dates are not chosen from returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentiment_long.config import EVENT_HORIZONS
from sentiment_long.paths import EVENTS

# Episodes that are the same story. Inference uses one observation per cluster.
CLUSTERS = {
    "yuan_devaluation_2015": "china_2015",
    "china_black_monday_2015": "china_2015",
    "china_circuit_breaker_2016": "china_2016",
    "china_circuit_breaker_2016_again": "china_2016",
    "wage_inflation_scare_2018": "vol_2018",
    "volmageddon_2018": "vol_2018",
    "fomc_hike_dec_2018": "dec_2018",
    "christmas_eve_2018": "dec_2018",
    "tariff_hike_threat_2019": "tariff_2019",
    "china_retaliation_2019": "tariff_2019",
    "tariff_tweet_aug_2019": "tariff_2019",
    "yuan_breaks_seven_2019": "tariff_2019",
    "yield_curve_inversion_2019": "tariff_2019",
    "covid_selloff_20200224": "covid_2020",
    "covid_breaker_20200309": "covid_2020",
    "covid_breaker_20200312": "covid_2020",
    "covid_breaker_20200316": "covid_2020",
    "covid_breaker_20200318": "covid_2020",
    "cpi_shock_20220610": "tightening_2022",
    "fed_75bp_repricing_20220613": "tightening_2022",
    "cpi_shock_20220913": "tightening_2022",
    "fomc_hawkish_20220921": "tightening_2022",
    "uk_minibudget_20220923": "tightening_2022",
    "svb_failure_20230310": "banks_2023",
    "regional_bank_monday_20230313": "banks_2023",
    "credit_suisse_20230315": "banks_2023",
    "jobs_miss_20240802": "yen_2024",
    "yen_carry_unwind_20240805": "yen_2024",
    "liberation_day_20250403": "tariff_2025",
    "china_retaliation_20250404": "tariff_2025",
    "china_tariff_threat_20251010": "tariff_2025",
    "north_america_tariffs_20250203": "tariff_2025",
    "iran_hormuz_20260305": "iran_2026",
    "iran_tankers_20260312": "iran_2026",
    "iran_strikes_20260302": "iran_2026",
    "iran_peace_rumor_20260331": "iran_2026",
    "trump_reflation_2016": "election_2016",
    "trump_election_whipsaw_2016": "election_2016",
}

# Forced-selling cascades, as opposed to a scheduled release or a policy announcement.
CASCADE_IDS = {
    "volmageddon_2018",
    "christmas_eve_2018",
    "nasdaq_airpocket_20200903",
    "yen_carry_unwind_20240805",
    "covid_breaker_20200309",
    "covid_breaker_20200312",
    "covid_breaker_20200316",
    "covid_breaker_20200318",
    "china_black_monday_2015",
    "regional_bank_monday_20230313",
}


def load_events() -> pd.DataFrame:
    frame = pd.read_csv(EVENTS)
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    frame["cluster"] = frame["id"].map(lambda item: CLUSTERS.get(item, item))
    frame["cascade"] = frame["id"].isin(CASCADE_IDS)
    return frame


def _locate(index: pd.DatetimeIndex, day: pd.Timestamp) -> int | None:
    if day in index:
        return int(index.get_loc(day))
    later = index[index >= day]
    if len(later) == 0:
        return None
    # Next session only if the catalog date fell on a weekend/holiday within 4 days.
    loc = int(index.get_loc(later[0]))
    if (index[loc] - day).days > 4:
        return None
    return loc


def _forward(close: np.ndarray, loc: int, horizon: int) -> float:
    last = loc + horizon
    if last >= len(close) or loc < 0:
        return np.nan
    px0 = close[loc]
    px1 = close[last]
    if not np.isfinite(px0) or not np.isfinite(px1) or px0 <= 0:
        return np.nan
    return float(px1 / px0 - 1.0)


def event_study(panel: pd.DataFrame) -> pd.DataFrame:
    events = load_events()
    index = pd.DatetimeIndex(panel.index)
    rows = []
    for event in events.itertuples(index=False):
        column = f"{event.ticker}_close"
        if column not in panel.columns:
            continue
        loc = _locate(index, pd.Timestamp(event.date))
        if loc is None or loc == 0:
            continue
        close = panel[column].to_numpy(dtype=float)
        spy = panel["spy_close"].to_numpy(dtype=float)
        row = {
            "id": event.id,
            "date": str(index[loc].date()),
            "catalog_date": str(pd.Timestamp(event.date).date()),
            "name": event.name,
            "kind": event.kind,
            "ticker": event.ticker,
            "prior_hypothesis": event.prior_hypothesis,
            "hold_days_prior": int(event.hold_days_prior),
            "cluster": event.cluster,
            "cascade": bool(event.cascade),
            "same_day": float(close[loc] / close[loc - 1] - 1.0) if close[loc - 1] > 0 else np.nan,
            "spy_same_day": float(spy[loc] / spy[loc - 1] - 1.0) if spy[loc - 1] > 0 else np.nan,
        }
        for horizon in EVENT_HORIZONS:
            row[f"fwd_{horizon}"] = _forward(close, loc, horizon)
            row[f"spy_fwd_{horizon}"] = _forward(spy, loc, horizon)
        hold = int(event.hold_days_prior)
        if hold > 0 and event.prior_hypothesis in {"long_contrarian", "long_momentum"}:
            forward = row.get(f"fwd_{hold}", _forward(close, loc, hold))
            row["prior_return"] = forward
            row["prior_hit"] = bool(forward > 0) if np.isfinite(forward) else None
        else:
            row["prior_return"] = np.nan
            row["prior_hit"] = None
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
    t_stat = mean / se if se > 0 else np.nan
    hit = float(np.mean(sample > 0))
    return {"n": n, "mean": mean, "t": float(t_stat) if np.isfinite(t_stat) else np.nan, "hit": hit}


def summarize_events(frame: pd.DataFrame) -> dict:
    def pack(subset: pd.DataFrame, column: str) -> dict:
        raw = _ttest(subset[column].to_numpy(dtype=float))
        clustered = (
            subset.groupby("cluster")[column].mean().to_numpy(dtype=float)
            if len(subset)
            else np.array([])
        )
        return {"raw": raw, "clustered": _ttest(clustered)}

    panic = frame[frame["kind"] == "panic"]
    euphoria = frame[frame["kind"] == "euphoria"]
    out = {
        "n_events": int(len(frame)),
        "n_panic": int(len(panic)),
        "n_euphoria": int(len(euphoria)),
        "n_mixed": int((frame["kind"] == "mixed").sum()),
        "panic_2d": pack(panic, "fwd_2"),
        "panic_10d": pack(panic, "fwd_10"),
        "panic_same_day": pack(panic, "same_day"),
        "euphoria_1d": pack(euphoria, "fwd_1"),
        "euphoria_5d": pack(euphoria, "fwd_5"),
        "euphoria_same_day": pack(euphoria, "same_day"),
        "cascade_2d": pack(panic[panic["cascade"]], "fwd_2"),
        "policy_2d": pack(panic[~panic["cascade"]], "fwd_2"),
    }
    return out
