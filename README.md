# 舆论做多（A 股）

默认只使用国内信息。价格来自新浪财经的沪深 300。舆论来自大陆报纸编制的经济政策不确定性月度指数。规则写在 `RESEARCH_DESIGN_CN.md`，回测不能反过来改阈值。

生产规则：报纸不确定性的月变化达到过去 36 个月的 1 个标准差时，下一自然月做多沪深 300，单边成本 5 个基点。概率是这些月份收益为正的收缩命中率。融资余额只作对照，不是舆论。

## 运行

实时舆情是东方财富上证指数股吧，不是报纸月报：

```bash
python -m sentiment_long live
```

最新帖子时间和滞后秒数写在 `results/cn/live_latest.json`。

```bash
pip install -r requirements.txt
python -m unittest tests.test_core
python -m sentiment_long backtest
python -m sentiment_long score --asof 2024-09-24
```

结果在 `results/cn/report.md`。2026 年 42 个事件的搜索在 `results/cn/y2026_report.md`。

```bash
python -m sentiment_long score --event-kind euphoria
python -m sentiment_long score --event-kind panic
```

美股版本仍可用 `python -m sentiment_long backtest --market us`，规则在 `RESEARCH_DESIGN.md`。自定义英文标题序列仍走 `--market us --mood-csv`。
