# 舆论做多

输入一条公众舆论序列，输出是否做多、看多分数、两日上涨概率，以及持有两个交易日。规则在 `RESEARCH_DESIGN.md` 里事先写死，回测不能反过来改阈值。

生产规则：舆论恶化达到过去 252 日的 1.5 个标准差时，在收盘做多 SPY，持有随后两个交易日，单边成本 5 个基点。满 15 笔交易后，如果历史超额不大于零，就停止做多。概率是这些交易的收缩命中率，不是语言模型当场的判断。

## 运行

```bash
pip install -r requirements.txt
python -m unittest tests.test_core
python -m sentiment_long backtest
python -m sentiment_long score --source agent --asof 2024-08-05
```

自定义序列（`mood` 越高越乐观）：

```bash
python -m sentiment_long score --mood-csv mood.csv --price-csv spy.csv --asof 2024-08-05
```

标题打分使用 2010 年以前已经收进词表的 Loughran–McDonald 词。`fundamental=1` 且当天净语气为负时，按公司新闻的文献结果不做多。

```bash
python -m sentiment_long score --headlines-csv headlines.csv --asof 2024-08-05
```

## 数据

不需要密钥。价格来自雅虎，新闻语气来自旧金山联储日度新闻情感指数，关注度来自英文维基恐慌页浏览量，不确定性来自 Baker–Bloom–Davis 日度指数。缓存放在 `data/cache/`，不入库。事件目录在 `research/events.csv`。

## 结果

`python -m sentiment_long backtest` 会写出 `results/report.md`、`results/summary.json` 和净值图。确认性读法以报告为准：维基关注度是时点复制，新闻语气文件是最新修订版。

这不是投资建议。低仓位的两日规则经常跑不赢全程买入持有，假设是否成立要看信号之后的超额，而不是净值是否最高。
