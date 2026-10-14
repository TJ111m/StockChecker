# 无密钥数据源实测（2026-10-14）

探测用 Python 3.12 的 `urllib` 直接发 HTTPS，没有注册、没有 API key。价格全历史只在这次核实里拉过一次；`probe_sources.py` 只打短窗口和样例。GDELT DOC 的月度 tone、月度篇数和 2019 整年 tone 都已经拿到 JSON，分辨率是日；artlist 仍是 429。

## 现在能用

| 用途 | 源 | 结果 |
| --- | --- | --- |
| SPY / QQQ / VIX 日线 | Yahoo `query1` chart v8 | 成功，2017-01-03 至 2026-10-13（VIX 另有未收盘的 10-14） |
| 官方 VIX 收盘 | FRED `VIXCLS` CSV | 成功，至 2026-10-12 |
| 现成日频新闻情感 | 旧金山联储 Daily News Sentiment | 成功，1980-01-01 至 2026-10-11 |
| 政策不确定性对照 | Baker–Bloom–Davis 日频 EPU | 成功，1985-01-01 至 2026-10-13 |
| 注意力对照 | 英文维基 pageviews | 成功，日频，无缺口至 2026-10-13 |
| 自建标题分数的词表 | Loughran–McDonald 1993–2025 | 成功，约 8.7MB，一次缓存 |
| 离线通用情感 | NLTK VADER 词表 | 成功，zip 90,486 字节，7,517 行 |
| 自建 tone / 标题 | GDELT v2 GKG 15 分钟 zip | 历史文件成功；单文件约 5–6MB |
| 2026 年中以后的标题 | GDELT v5 weblegacy ngrams | 样例文件成功，覆盖不到 2017 |
| Stooq CSV | `stooq.com/q/d/l` | 失败，JS 校验 |
| GDELT DOC 月度与整年 tone | `api.gdeltproject.org` | 成功。月、年都是日频，结束日包含在内，未见截断 |
| GDELT DOC 月度篇数 | 同上 `timelinevolraw` | 成功，日频，点上多一个整数 `norm` |
| GDELT DOC artlist | 同上 | 四次都是 429，标题字段未拿到 |

## 1. Yahoo Finance 日线

无需 crumb，也未安装 `yfinance`。`chart.error` 为 `null`。

```
https://query1.finance.yahoo.com/v8/finance/chart/SPY?period1=1483228800&period2=1792022400&interval=1d&events=div%7Csplit&includeAdjustedClose=true
https://query1.finance.yahoo.com/v8/finance/chart/QQQ?period1=1483228800&period2=1792022400&interval=1d&events=div%7Csplit&includeAdjustedClose=true
https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?period1=1483228800&period2=1792022400&interval=1d&events=div%7Csplit&includeAdjustedClose=true
```

`period1` 是 2017-01-01 00:00 UTC，`period2` 是 2026-10-15 00:00 UTC。

JSON 路径：

- `chart.result[0].meta`：`symbol`、`currency`、`exchangeTimezoneName`、`instrumentType`、`regularMarketTime`
- `chart.result[0].timestamp`：该日 K 线的 Unix 秒
- `chart.result[0].indicators.quote[0]`：`open`、`high`、`low`、`close`、`volume`
- `chart.result[0].indicators.adjclose[0].adjclose`
- 带 `events=div|split` 时，SPY 另有 `events.dividends`（本次 39 条）。做收益只用 `adjclose`

| 代码 | 根数 | 首根 | 末根 | 末根 close / adjclose |
| --- | --- | --- | --- | --- |
| SPY | 2458 | 2017-01-03 | 2026-10-13 | 778.01 / 778.01 |
| QQQ | 2458 | 2017-01-03 | 2026-10-13 | 743.57 / 743.57 |
| ^VIX | 2552，其中 close 为空 91 根 | 2017-01-03 | 2026-10-14 | 15.34 / 15.34 |

SPY 首根未复权收盘 225.24，复权 193.49。时间戳不是 UTC 零点：2026-10-13 的 SPY 戳是 `2026-10-13T13:30:00Z`（美东 09:30）。用 `America/New_York` 的日历日当作交易日。

^VIX 的 `exchangeTimezoneName` 是 `America/Chicago`，戳常在 07:00 或 08:00 UTC。`volume` 恒为 0。91 个空 close 是休市日（2017-01-16、2017-07-04、2017-12-25 这类），丢掉。2026-10-14 这根是在美股现金开盘前就已经出现的，`regularMarketTime` 当时为 `2026-10-14T11:31:31Z`。回测丢掉晚于最后一根 SPY 的 VIX。

VIX 会在 SPY 没有成交的日子给出数字。2026-09-07（劳动节，周一）SPY 无 K 线，Yahoo ^VIX 和 FRED 都是 15.30。2026-06-19 SPY 无 K 线、Yahoo ^VIX 的 close 为空，FRED 仍有 16.78。交易日历以 SPY 有 `close` 的日期为准。

全区间一次约 260–270KB。三个代码各请求一次，间隔 1 秒。失败再换 `query2.finance.yahoo.com`，路径相同；这次没有用到。

## 2. FRED VIXCLS

```
https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS
https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS&cosd=2017-01-01&coed=2026-10-14
```

HTTP 200，`text/csv` 实际标成 `application/csv`。表头 `observation_date,VIXCLS`。

- 不带日期：1990-01-02 的 17.24 起，末行 2026-10-12 的 15.21，9,595 行数据
- 带 2017-01-01 至 2026-10-14：2,550 行，2,490 个有数字，60 个逗号后为空。空行是假日，例如 `2017-01-16,`
- 周末不出现。2017-01-03 与 Yahoo 同为 12.85；2026-06-18 两边都是 16.40

FRED 比 Yahoo 晚两个交易日左右。主恐惧序列用 FRED，Yahoo ^VIX 只做交叉核对。`api.stlouisfed.org` 的 JSON 要 key，不要走那条。

## 3. 旧金山联储日频新闻情感

这是现在就能对齐到 2017–2026、而且不用自己打分的新闻情感。

页面：<https://www.frbsf.org/research-and-insights/data-and-indicators/daily-news-sentiment-index/>

```
https://www.frbsf.org/wp-content/uploads/news-sentiment-chart-1.csv?2026-10-13
https://www.frbsf.org/wp-content/uploads/news_sentiment_data.xlsx?2026-10-13
```

两个都是 HTTP 200。

CSV：`date,Daily news sentiment`，17,067 行，`01/01/1980,-0.04` 到 `10/11/2026,0.06`。两位小数。

xlsx 有 `Methodology` 和 `Data`。`Data` 表头是 `date`、`News Sentiment`。1980-01-01 为 -0.03784025087952614，2026-10-11 为 0.06280666589736938，行数与 CSV 一致。回测用 xlsx 的全精度。

日历里包含周末（CSV 中周末 4,879 行），但不是完整自然日：1980-01-01 到 2026-10-11 比自然日少 19 天，xlsx 里 1980-01-04 直接跳过。方法页写明，2021 年 4 月起底层改为 Factiva，报纸集合换过。这是序列定义的断点，滚动 z 分数不要跨这个月不假思索地比较。

引用：Shapiro, Sudhof, Wilson，FRBSF Working Paper 2017-01，以及 Buckman、Shapiro、Sudhof、Wilson 的 FRBSF Economic Letter 2020-08。

## 4. 经济政策不确定性（EPU）

```
https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv
```

HTTP 200，`text/csv`，250,359 字节。表头 `day,month,year,daily_policy_index`。首行 `1,1,1985,103.83`，末行 `13,10,2026,216.72`，15,261 行。2017 年 1 月有 31 天，含周日 2017-01-01（235.10）和马丁路德金日 2017-01-16（114.17）。这是自然日，不是交易日。它量的是政策不确定性，不是涨跌语气。

## 5. 维基百科日浏览量

必须带说明身份的 User-Agent。`agent=user` 去掉爬虫。

```
https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/Stock_market/daily/2017010100/2026101400
https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/Recession/daily/2017010100/2026101400
```

两边都是 HTTP 200，顶层只有 `items`。每条字段：`project`、`article`、`granularity`、`timestamp`、`access`、`agent`、`views`。`granularity` 为 `daily`，`timestamp` 形如 `2017010100`。

2017-01-01 至 2026-10-13 各 3,573 条，与自然日天数相同，无缺口。样例：Stock_market 在 2017-01-01 为 1,525，2026-10-13 为 399；Recession 同两日为 442 和 324。一次约 520KB。一篇文章一次请求即可。

浏览量按 UTC 日切。美东收盘时，当天的 UTC 日还没结束。用到交易日 t 的特征，取 UTC 日 t-1 的 `views`。

## 6. GDELT DOC 2.0

接口：

```
https://api.gdeltproject.org/api/v2/doc/doc
```

这台出口上，连续请求会 HTTP 429，正文是纯文本：

```
Please limit requests to one every 5 seconds or contact kalev.leetaru5@gmail.com for larger queries.
```

官方博客把原因写成 legacy search 正在迁 Spanner：<https://blog.gdeltproject.org/using-the-new-web-ngrams-dataset-to-find-relevant-coverage/>（2026-06-30）。实测比「5 秒一次」更严：间隔 8–180 秒仍经常 429；安静大约 8–10 分钟后的下一次，或 429 之后再等约 90 秒的重试，可以打通。不要并行。

### 6.1 月度 timelinetone：成功，日频，未截断

安静约 10 分钟后，HTTP 200，`application/json`，1,684 字节，耗时 17 秒。

```
https://api.gdeltproject.org/api/v2/doc/doc?query=stock%20market&mode=timelinetone&format=json&startdatetime=20200101000000&enddatetime=20200201000000
```

```json
{
  "query_details": {"title": "stock market", "date_resolution": "day"},
  "timeline": [{
    "series": "Average Tone",
    "data": [
      {"date": "20200101T000000Z", "value": 0.4212},
      {"date": "20200102T000000Z", "value": 0.4351}
    ]
  }]
}
```

- 顶层只有 `query_details`、`timeline`
- `query_details` 只有 `title`、`date_resolution`
- `timeline` 长度为 1；`series` 为 `Average Tone`
- 点字段只有 `date`、`value`。`date` 形如 `20200101T000000Z`，`value` 是浮点
- `date_resolution` 为 `day`。32 个点，2020-01-01 到 2020-02-01 都在，间隔全是 86400 秒。结束日包含在内。没有截断
- value 最小 -1.2992，最大 0.4684，均值约 -0.2143

这是日频，不是 15 分钟。更短窗口会不会改成 15 分钟，这次没有第二份成功 JSON。`timespan=15min` 在早一次 200 里只返回纯文本 `Timespan is too short.`

### 6.2 月度 timelinevolraw：成功，日频，多一个 norm

第一次（上一成功之后 10 秒）429。等 90 秒重试，HTTP 200，2,147 字节。

```
https://api.gdeltproject.org/api/v2/doc/doc?query=%28stock%20market%20OR%20recession%20OR%20%22wall%20street%22%29&mode=timelinevolraw&format=json&startdatetime=20200101000000&enddatetime=20200201000000
```

`date_resolution` 仍是 `day`。`series` 为 `Article Count`。点字段是 `date`、`value`、`norm`。`value` 和 `norm` 都是整数。32 天，2020-01-01 至 2020-02-01，间隔全是 1 天，未截断。

样例：`{"date": "20200101T000000Z", "value": 30123, "norm": 262970}`。该月 `value` 从 24,092 到 69,451。响应里没有解释 `norm`。它比 `value` 大一个数量级，不要把它当成情感分，也不要直接当文章数；文章数用 `value`。

### 6.3 整年 timelinetone：成功，仍是日频，没有截断

artlist 在更小的窗口上仍是 429：volraw 成功后 10 秒（`maxrecords=3`，两天）、再等 90 秒、再等 150 秒（`maxrecords=2`），以及整年 tone 成功后再等 160 秒（`maxrecords=1`，一天）。四次都没有 JSON。整年 tone 在第三次 artlist 失败后再等 150 秒，HTTP 200，17,800 字节，耗时 14 秒。

```
https://api.gdeltproject.org/api/v2/doc/doc?query=stock%20market&mode=timelinetone&format=json&startdatetime=20190101000000&enddatetime=20200101000000
```

结构和月度相同：`date_resolution` 为 `day`，`series` 为 `Average Tone`，点只有 `date` 和 `value`。366 个点，2019-01-01 至 2020-01-01，365 个间隔全是 86400 秒。最后一点 `20200101T000000Z` 的 value 是 0.4212，与上面月度序列的第一点相同。整年没有被截成 15 分钟，也没有在中途停住。

整年 `timelinevolraw` 这次没有再打。月度已经证明它比 tone 多一个整数 `norm`，按年拉取时间隔仍用 90 秒以上。

artlist 仍没有 JSON，标题字段不能从 DOC 填写。下面的 GKG `PAGE_TITLE` 和 ngrams `title` 是这次实际拿到的标题。

```
https://api.gdeltproject.org/api/v2/doc/doc?query=%28stock%20market%20OR%20recession%20OR%20%22wall%20street%22%29&mode=artlist&format=json&startdatetime=20200101000000&enddatetime=20200103000000&maxrecords=2&sort=hybridrel
```

拉取：tone 可以按年，一年大约 18KB。两次请求至少隔 90 秒，429 后再等 2–10 分钟只发一个。不要并行。artlist 先别放进日常作业，这次从这台出口没有成功过。

## 7. 能替代 DOC 的 GDELT 文件

### 7.1 v2 GKG：2015 年 2 月至今，tone 和标题都在

清单：

```
https://data.gdeltproject.org/gdeltv2/masterfilelist.txt
https://data.gdeltproject.org/gdeltv2/lastupdate.txt
```

`masterfilelist.txt` 128,536,779 字节。开头是 `20150218230000` 的 export / mentions / gkg，末尾列到 `20261014120000`。每行 `字节数 md5 url`。只下 `gkg.csv.zip`。

实测 HTTP 200：

| 文件 | 压缩大小 |
| --- | --- |
| `https://data.gdeltproject.org/gdeltv2/20170101000000.gkg.csv.zip` | 5,937,338 |
| `https://data.gdeltproject.org/gdeltv2/20200101000000.gkg.csv.zip` | 5,288,568 |
| `https://data.gdeltproject.org/gdeltv2/20261013120000.gkg.csv.zip` | 5,085,209 |
| `https://data.gdeltproject.org/gdeltv2/20261014000000.gkg.csv.zip` | 5,566,883 |

`20261014114500` 和 `20261014120000` 的 gkg 在清单里，HEAD 却是 404；同时间戳的 mentions 曾是 200。最新一批要重试，历史批次是稳的。

`20200101000000.gkg.csv` 解压 16.5MB，1,253 行，每行 27 列，制表符分隔。列名用的是表头文件 <https://raw.githubusercontent.com/linwoodc3/gdelt2HeaderRows/master/schema_csvs/GDELT_2.0_gdeltKnowledgeGraph_Column_Labels_Header_Row_Sep2016.tsv>，和这一行对得上：

| 下标 | 名称 | 这一文件里看到的 |
| --- | --- | --- |
| 0 | GKGRECORDID | `20200101000000-0` |
| 1 | DATE | `20200101000000`，与文件名同一 15 分钟 |
| 2 | SourceCollectionIdentifier | `1`（网页） |
| 3 | SourceCommonName | 域名，如 `wsaz.com` |
| 4 | DocumentIdentifier | 文章 URL |
| 7 | Themes | 分号分隔 |
| 8 | V2Themes | 主题加字符偏移 |
| 15 | V2Tone | 7 个逗号分隔数 |
| 26 | Extras | 含 `<PAGE_TITLE>…</PAGE_TITLE>` |

码本 <https://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf> 把这 7 个数称为 V1.5TONE，并写明最后追加了 Word Count。顺序是 Tone、Positive Score、Negative Score、Polarity、Activity Reference Density、Self/Group Reference Density、Word Count。Tone 等于正分减负分，理论范围 -100 到 +100，常见 -10 到 +10。码本把 Negative Score 误写成正面词，以字段名和「Tone = 正减负」为准。本次看到的第 7 项是整数词数（128、723）。

标题样例（正则打在 `PAGE_TITLE` 上，不是主题）：

- Tone -0.2976：`U.S. stocks see out 2019 with rally`（londonmercury.com）
- Tone 0.8929：`Stocks close out best year since 2013; S&P 500 soars 28.9%`（wsaz.com）

第一则标题是涨，Tone 仍为负。GDELT Tone 不能当成标题极性。该 15 分钟里，标题命中 stock / stocks / wall street / recession / S&P / Nasdaq / Dow Jones 的有 21 篇。主题 `ECON_STOCKMARKET` 有 62 行，但会命中 “New Stock Images”，不能单独当过滤器。

从 2017-01-01 到 2026-10-14 大约 3,400 个自然日、每天 96 个文件、每个约 5MB，传输量大约 1.6TB。流式解析，只落日频表，原始 zip 不留。并发保持 1 或 2。这是 DOC 长期 429 时的补救，不是第一版回测该走的路。

### 7.2 v5 ngrams：有标题，历史只到 2026 年中

博客给出的文件仍在，列也核对过：

```
https://data.gdeltproject.org/gdeltv5/weblegacy/ngrams/20260630201600.toc.json.gz
https://data.gdeltproject.org/gdeltv5/weblegacy/ngrams/20260630201600.ngrams.txt.gz
```

`toc.json.gz` 284,249 字节，解压后 1,977 行 JSON。字段：`ID`、`date`、`img`、`lang`、`title`、`url`。英文 745 篇。样例：

```json
{"ID": 345, "date": "2026-06-30T20:16:00.000Z", "lang": "en", "title": "Why did the stock market soar in the first half of 2026? Experts explain", "url": "https://abcnews.com/Business/stock-market-soar-half-2026-experts-explain/story?id=134336709"}
```

`ngrams.txt.gz` 8,192,280 字节。制表符三列，已读到前两行：`DOCID`、`QUADGRAM`、`COUNT`，例如 `1\t"This is a strategic\t1`。没有全文，也没有情感分。

文件名是 `YYYYMMDDHHMM00`，但不是每一分钟都有。`20260630201600`、`20260630201700`、`20260630211600`、`20261013201600`、`20261014001600`、`20260701001600` 为 200；`20260629001600` 以及 2026-06 之前按整点试的名字为 404。目录列表和 `masterfilelist` 都是 404。能确认的覆盖从 2026-06-30 左右开始，做不了 2017 起的回测。

`20261014001600.toc.json.gz` 在探测脚本里再次成功：2,957 篇，其中一条为 `{"date": "2026-10-14T00:16:00.000Z", "lang": "en", "title": "Japan Stock Market Has Another Weak Lead"}`。

## 8. Stooq

```
https://stooq.com/q/d/l/?s=spy.us&d1=20200102&d2=20200131&i=d
```

HTTP 200，`text/html`，正文要求 JavaScript 做 SHA-256 工作量证明。按页面算法算出 nonce 后，`POST https://stooq.com/__verify` 返回 403，正文 `challenge invalid: ip mismatch`。CSV 列没有拿到。不要放进拉取作业。

## 9. 情感工具

### Loughran–McDonald

页面 <https://sraf.nd.edu/loughranmcdonald-master-dictionary/> 写着 Updated: March, 2026。CSV 不在 nd.edu 域名上，页面链到 Google Drive。直链可用：

```
https://drive.google.com/uc?export=download&id=1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF
```

HTTP 200，`Content-Disposition` 文件名 `Loughran-McDonald_MasterDictionary_1993-2025.csv`，9,093,460 字节。86,553 个词，17 列：

`Word, Seq_num, Word Count, Word Proportion, Average Proportion, Std Dev, Doc Count, Negative, Positive, Uncertainty, Litigious, Strong_Modal, Weak_Modal, Constraining, Complexity, Syllables, Source`

非零是年份，不是权重。`LOSS` 的 Negative 为 2009，Positive 为 0。`GAIN` 的 Positive 为 2009。`PROFIT` 两类都是 0。Negative 非零 2,355 个，其中 10 个是 `-2020`（`BROKEN`、`CLOSED` 等）；Positive 有 7 个 `-2020`（含 `BENEFIT`）。负号表示这一版从该类删除。扫 2017 年文本时，只保留「正年份 ≤ 当年」且删除年还没到的词。

学术使用免费；商用要写信 `loughranmcdonald@gmail.com`。下载一次，本地缓存。

### VADER

```
https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/sentiment/vader_lexicon.zip
```

HTTP 200，90,486 字节。解压路径 `vader_lexicon/vader_lexicon.txt`，7,517 行。列是 token、平均分、标准差、十分制原始列表。样例：`recession	-1.8	1.07703	[-3, -1, -4, -2, -1, -1, -1, -1, -3, -1]`，`gain	2.4	...`。

`pip install nltk` 之后把 zip 放到 `nltk_data/sentiment/vader_lexicon.zip`（不要先解压，NLTK 按这个路径找）。本次离线打分：

- `stocks rally after strong earnings` → compound 0.5106
- `recession fears slam the stock market` → compound -0.802
- `wall street is mixed as investors wait` → compound 0.0

没有密钥，也不下大模型。词表是通用英语，不是 10-K 词表。标题短，适合做对照；否定和标点规则在 `nltk.sentiment.vader`，只拿词表求平均会和 compound 不一致。

### ProsusAI/finbert

`config.json` 的 `id2label` 是 `0 positive / 1 negative / 2 neutral`，架构 `BertForSequenceClassification`。权重：

- `model.safetensors` 437,965,908 字节
- `pytorch_model.bin` 437,992,753 字节

模型卡标签含 `license:cc-by-nc-sa-4.0`。本地推理必须下这一份约 418MB 的权重，另加 tokenizer。`POST https://router.huggingface.co/hf-inference/models/ProsusAI/finbert` 无 token 时 HTTP 401。旧的 `api-inference.huggingface.co` 在这台机器上解析失败。不作为主信号。

## 10. 日频特征

交易日只取 SPY 的 `adjclose` 非空日期。`y_{t+1} = adjclose_{t+1} / adjclose_t - 1`。t 收盘才能使用的信息，用来解释 `y_{t+1}`，不用来解释当天收益。

主信号用旧金山联储 `News Sentiment`：

- `news_t` = 严格早于交易日 t 的最近一个有值日历日。周末的值会落到下一个交易日，当天报纸不会进当天收盘的决策。
- `news_z_t` = `(news_t - mean(news_{t-60..t-1})) / std(news_{t-60..t-1})`。均值和标准差只用过去 60 个交易日。标准差为 0 或不足 60 日，记缺失。
- 2021 年 4 月 Factiva 切换之后重新起算滚动窗口，或者对切换前后分开标准化。

对照，三条都滞后，避免和主信号抢同一天的未来：

- `vix_chg_t`：FRED `VIXCLS` 对齐到 SPY 日（假日空值不前填超过一个交易日），`vix_t - vix_{t-1}`。
- `epu_z_t`：EPU `daily_policy_index` 取严格早于 t 的最近日历日，再做同样的 60 日 z。
- `wiki_rec_z_t`：`log(1 + Recession 的 views)`，用 UTC 日 t-1，再做 60 日 z。`Stock_market` 同样做一条，当作关注度而不是方向。

GDELT 月度 `timelinetone` 可以做第四条对照，不替代旧金山联储。`gdelt_tone_t` 用 `timeline[0].data[].value`，`date` 取 UTC 日。交易日 t 只用严格早于 t 的那个 UTC 日。`timelinevolraw` 的 `value` 是当天文章数，可做覆盖度，`norm` 先不用。这个接口在本机上两次请求要隔 90 秒以上，所以它是月度批量拉下来的序列，不是盘中刷新。标题级 VADER 或 LM 仍要从 GKG / ngrams 来，DOC artlist 这次没有返回。`ECON_STOCKMARKET` 主题会把 “stock images” 算进来。

做多回测看的是 `news_z_t` 对 `y_{t+1}` 的单调性和扣费后的多头收益。`vix_chg_t` 放进同一回归：情感要在控制 VIX 变化之后还在，才不是恐惧的别名。

## 11. 拉取顺序

1. Yahoo 三个代码各一次全样本，间隔 1 秒。落 `date, adjclose`。
2. FRED `VIXCLS` 一次。按 SPY 日期左连接。
3. 旧金山联储 xlsx 一次，读 `Data` 页。CSV 只做核对。
4. EPU CSV 一次。维基两篇文章各一次。
5. LM CSV 和 VADER zip 各一次，提交进缓存，不要放进每日作业。
6. GDELT DOC 的 tone 按年拉，`timelinevolraw` 按月拉，两次至少隔 90 秒，不要和别的 GDELT 请求并行。429 就等几分钟再发一个。artlist 这次没有成功，标题用 GKG 或 ngrams。
7. GKG 只在需要自建 tone 时按 `masterfilelist` 逐个 15 分钟文件流式聚合，按月存日频结果。
8. v5 ngrams 只适合 2026-06-30 之后的标题抽样。没有清单，不能靠猜每一分钟的文件名补历史。
