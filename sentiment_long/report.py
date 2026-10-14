"""Chinese backtest report and figures. Numbers come only from the result dict."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _missing(value) -> bool:
    return value is None


def bp(value) -> str:
    if _missing(value):
        return "—"
    return f"{value * 10000:.1f} bp"


def pct(value) -> str:
    if _missing(value):
        return "—"
    return f"{value * 100:.2f}%"


def num(value, digits: int = 2) -> str:
    if _missing(value):
        return "—"
    return f"{value:.{digits}f}"


def _find(rows: list[dict], source: str, timing: str, start: str) -> dict | None:
    for row in rows:
        if row["source"] == source and row["timing"] == timing and row["start"] == start:
            return row
    return None


def _reg_sentence(reg: dict) -> str:
    if not reg or reg.get("error"):
        return "这条回归没有足够样本。"
    coef = reg.get("params", {}).get("deterioration")
    t_stat = reg.get("tvalues", {}).get("deterioration")
    p_value = reg.get("pvalues", {}).get("deterioration")
    spy_coef = reg.get("params", {}).get("spy_ret")
    spy_t = reg.get("tvalues", {}).get("spy_ret")
    direction = "为正" if coef is not None and coef > 0 else "不为正"
    return (
        f"恶化冲击的系数 {bp(coef)}（每 1 个标准差），方向{direction}，"
        f"HAC t = {num(t_stat)}，双侧 p = {num(p_value, 3)}，样本 {reg.get('n')} 天。"
        f"控制项当日收益的系数是 {num(spy_coef, 3)}（t = {num(spy_t)}）。"
    )


def _trade_sentence(row: dict | None) -> str:
    if row is None:
        return "没有这行结果。"
    return (
        f"{row['start']} 至 {row['end']}，不重叠交易 {row['n_trades']} 笔，"
        f"持有期毛收益均值 {bp(row['mean_trade'])}，同期任意两日窗口 {bp(row['mean_uncond'])}，"
        f"超额 {bp(row['excess'])}，交易 t = {num(row['t_excess'])}，"
        f"命中率 {pct(row['hit_rate'])}，仓位占比 {pct(row['exposure'])}，"
        f"扣费后夏普 {num(row['sharpe'])}，年化 {pct(row['cagr'])}，最大回撤 {pct(row['maxdd'])}，"
        f"买入持有年化 {pct(row['buyhold_cagr'])}，循环移位 p = {num(row.get('null_p'), 3)}。"
    )


def render(summary: dict) -> str:
    rows = summary["strategies"]
    regs = {item["name"]: item for item in summary["regressions"]}
    events = summary["events"]
    sanity = summary["sanity"]
    wiki = _find(rows, "wiki", "fast", "2016-07-01")
    news_2010 = _find(rows, "news", "fast", "2010-01-01")
    news_2020 = _find(rows, "news", "fast", "2020-04-07")
    news_slow = _find(rows, "news", "slow", "2020-04-07")
    price = _find(rows, "price", "fast", "2010-01-01")
    vix = _find(rows, "vix", "fast", "2010-01-01")
    momentum = _find(rows, "news_momentum", "fast", "2010-01-01")
    lines = [
        "# 舆论做多系统：回测报告",
        "",
        "生产规则在看到这些数字之前已经写死：公众语气或恐慌关注度恶化达到 1.5 个过去标准差时，在信号日收盘做多 SPY，持有随后 2 个交易日，单边成本 5 个基点。机器学习模型只作对照，不替换这条规则。持有期也不是从下面的网格里挑出来的。",
        "",
        "## 数据是否接对",
        "",
        f"价格从 {sanity['first']} 到 {sanity['last']}，共 {sanity['rows']} 个交易日。新闻语气用到 {sanity['news_last']}。",
        f"2020-03-16 标普收益 {pct(sanity['spy_2020_03_16'])}，VIX {num(sanity['vix_2020_03_16'], 1)}；2020-03-24 收益 {pct(sanity['spy_2020_03_24'])}；2025-04-04 收益 {pct(sanity['spy_2025_04_04'])}。",
        f"维基恐慌页 2020 年 3 月的关注度高于 1 月（{num(sanity['wiki_march_2020'], 3)} 对 {num(sanity['wiki_january_2020'], 3)}）。",
        "",
        "## 确认性结果",
        "",
        "### 维基恐慌关注度，2016-07 之后",
        "",
        "这是时点数据：交易日 T 只用日历日 T-1 已经完整的浏览量。它测的是公众在查什么，不是文章的褒贬。",
        "",
        _reg_sentence(regs.get("wiki_fast_common")),
        "",
        _trade_sentence(wiki),
        "",
        "### 旧金山联储日度新闻语气",
        "",
        "这是真正的新闻语气。文件是最新修订版，不是当年已经发布的版本，所以 2010 年以来的整段不能当成干净的实盘记录。公开信在 2020-04-06，下面把 2020-04-07 之后单独列出。快速执行假设当日收盘前能自己把新闻打完分；慢速执行再晚一个交易日。",
        "",
        "2010 年以来的快速执行：" + _trade_sentence(news_2010),
        "",
        "2020-04-07 之后的快速执行：" + _trade_sentence(news_2020),
        "",
        "同一窗口的慢速执行：" + _trade_sentence(news_slow),
        "",
        "快速执行、控制当日涨跌之后的回归：" + _reg_sentence(regs.get("news_fast_post_letter")),
        "",
        "慢速执行的回归：" + _reg_sentence(regs.get("news_slow_post_letter")),
        "",
        "### 两个对照，不是舆论",
        "",
        "价格冲击（当日收益的 z 值低于 -1.5）同样持有两日：" + _trade_sentence(price),
        "",
        "回归：" + _reg_sentence(regs.get("price_fast_from_2010")),
        "",
        "VIX 跳升：" + _trade_sentence(vix),
        "",
        "回归：" + _reg_sentence(regs.get("vix_fast_from_2010")),
        "",
        "如果舆论系数在放进当日收益之后就消失，而价格冲击本身还有预测力，那么舆论只是在重复已经发生的下跌。",
        "",
        "### 应当失败的对照",
        "",
        "新闻语气大幅变好就追多，持有两日：" + _trade_sentence(momentum),
        "",
        "## 70 个已登记事件",
        "",
        _event_section(events),
        "",
        "事件日当天的涨跌不算进策略。表内的 forward 都是从当天收盘之后算起。聚类 t 值把同一轮叙事收成一个观测，避免把五天熔断当成五次独立实验。",
        "",
        "## 机器学习对照",
        "",
        _ml_section(summary["ml"]),
        "",
        "## 系统当天怎么回答",
        "",
        "输入一条语气（越高越乐观）。输出是否做多、看多分数、两日上涨概率和 95% 区间、收缩后的超额收益、持有 2 个交易日。概率是这条规则以往不重叠交易的收缩命中率。评估样本里满 15 笔之后，如果平均超额不大于零，规则继续触发也会被关掉。分数为 1 表示两日超额大约 100 个基点；文献先验阶段分数封顶 0.3。",
        "",
        "下面每个例子都是只使用当天及以前的交易算出来的。",
        "",
    ]
    for score in summary["scores"]:
        lines.append(_score_line(score))
    lines.extend(
        [
            "",
            "## 网格",
            "",
            "阈值 1.0 和 2.0、持有 1/4/5/10 日、零成本和单边 10 个基点，都在 `strategies.csv` 里，角色标成 exploratory 或 cost_sensitivity。它们不改变生产规则。",
            "",
            "## 局限",
            "",
            "这不是投资建议。两日超额若只有几个到十几个基点，扣掉 10 个基点往返后所剩很少，也经常达不到 t > 3 的新发现门槛。新闻语气序列会修订。维基浏览量是关注度，不是褒贬。事件目录是人为列出的叙事，不是随机样本。系统只做多指数，不做个股基本面新闻的反向买入；那种标题按文献应低配。",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _cluster_line(title: str, block: dict) -> str:
    raw = block["raw"]
    clustered = block["clustered"]
    return (
        f"- {title}：{raw['n']} 个事件，平均 {bp(raw['mean'])}，命中 {pct(raw['hit'])}，t = {num(raw['t'])}；"
        f"收成 {clustered['n']} 个簇之后平均 {bp(clustered['mean'])}，t = {num(clustered['t'])}。"
    )


def _event_section(events: dict) -> str:
    return "\n".join(
        [
            f"目录进入价格面板的事件 {events['n_events']} 个，其中恐慌 {events['n_panic']}、狂热 {events['n_euphoria']}、混合 {events['n_mixed']}。",
            _cluster_line("恐慌事件当天", events["panic_same_day"]),
            _cluster_line("恐慌事件之后 2 日", events["panic_2d"]),
            _cluster_line("恐慌事件之后 10 日（目录里的旧持有期）", events["panic_10d"]),
            _cluster_line("其中的抛售连锁", events["cascade_2d"]),
            _cluster_line("其中的政策或数据发布", events["policy_2d"]),
            _cluster_line("狂热事件当天", events["euphoria_same_day"]),
            _cluster_line("狂热事件之后 1 日", events["euphoria_1d"]),
            _cluster_line("狂热事件之后 5 日（目录里的旧持有期）", events["euphoria_5d"]),
        ]
    )


def _ml_section(ml: dict) -> str:
    if ml.get("error"):
        return f"模型没有跑成：{ml['error']}。"
    lines = [
        f"逻辑回归和一层 8 神经元网络只用新闻语气的滞后特征，不用价格、不用 VIX。训练 1995–2004，标签在验证段开始前截断。验证 2005–2009，测试 2010 之后。做多阈值是训练集里的上涨基准率 {pct(ml['base_rate'])}，持有期仍是 2 日。",
        f"阈值规则在验证段的夏普是 {num(ml['rule_val_sharpe'])}，在 2010 之后超额 {bp(ml['rule_test']['excess'])}，交易 t = {num(ml['rule_test']['t_excess'])}，{ml['rule_test']['n_trades']} 笔。",
    ]
    for name, block in ml.get("models", {}).items():
        val = block["val"]
        test = block["test"]
        lines.append(
            f"- {name}：验证夏普 {num(val['sharpe'])}，{val['n_trades']} 笔，超额 {bp(val['excess'])}；"
            f"测试夏普 {num(test['sharpe'])}，{test['n_trades']} 笔，超额 {bp(test['excess'])}，t = {num(test['t_excess'])}。"
        )
    lines.append(ml.get("note", ""))
    return "\n".join(lines)


def _score_line(score: dict) -> str:
    prob = score.get("probability_up")
    prob_text = "—" if prob is None else f"{prob:.1%}"
    excess = score.get("expected_excess_return")
    return (
        f"- {score['asof']}：{score['stance']}，分数 {num(score['bullish_score'])}，"
        f"概率 {prob_text}，超额 {bp(excess)}，持有 {score['hold_trading_days']} 日，"
        f"新闻与维基 {score['agreement']}。{score['rationale']}"
    )


def save_plots(curve: pd.DataFrame, events: pd.DataFrame, results: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = curve.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    window = frame[frame["date"] >= "2016-07-01"]
    figure, axis = plt.subplots(figsize=(10, 5))
    for column, label in (
        ("spy", "SPY buy and hold"),
        ("news_fast", "News tone, 2-day long"),
        ("wiki_fast", "Wiki fear, 2-day long"),
        ("price_fast", "Price shock, 2-day long"),
        ("vix_fast", "VIX shock, 2-day long"),
    ):
        wealth = (1.0 + window[column]).cumprod()
        axis.plot(window["date"], wealth, label=label, linewidth=1.2)
    axis.set_title("Growth of $1 since 2016-07, 5 bp per side")
    axis.legend(loc="upper left", fontsize=8)
    axis.set_ylabel("Wealth")
    figure.tight_layout()
    figure.savefig(results / "equity_2016.png", dpi=120)
    plt.close(figure)

    early = frame[frame["date"] >= "2010-01-01"]
    figure, axis = plt.subplots(figsize=(10, 5))
    for column, label in (
        ("spy", "SPY buy and hold"),
        ("news_fast", "News tone fast"),
        ("news_slow", "News tone slow"),
        ("price_fast", "Price shock"),
    ):
        wealth = (1.0 + early[column]).cumprod()
        axis.plot(early["date"], wealth, label=label, linewidth=1.2)
    axis.set_title("Growth of $1 since 2010-01, 5 bp per side")
    axis.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    figure.savefig(results / "equity_2010.png", dpi=120)
    plt.close(figure)

    panic = events.loc[events["kind"] == "panic", "fwd_2"].dropna()
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.hist(panic * 100, bins=20, color="#4c566a")
    axis.axvline(0, color="black", linewidth=1)
    axis.set_xlabel("Return over the next 2 sessions, percent")
    axis.set_title("Panic episodes, entered at the event close")
    figure.tight_layout()
    figure.savefig(results / "panic_2d_hist.png", dpi=120)
    plt.close(figure)
