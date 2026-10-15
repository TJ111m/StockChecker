# 新闻 / 社交情感 → 股票回测：开源版图

检索日：2026-10-14。只读 GitHub README、Hugging Face 模型卡 / 数据集字段、Notre Dame SRAF 页面、GDELT GKG Codebook V2.1、PyPI。未改任何策略代码。

结论先写在前面：日频可交易信号用自己写的「时间戳截断 + Loughran–McDonald 词表 + 下一交易日成交 + 成本」。FinBERT 权重大、许可证是非商用，全文打分不要当主信号。现成的 LSTM / 情感回测仓库要么同日对齐，要么没有成本，要么在合成数据上报夏普，都不要接入。

---

## 1. 自己实现，不要当依赖

这些部分没有可长期依赖的开源实现，应写在自己的回测里：

1. **时钟。** 只用 `timestamp <= 决策时刻` 的文本。默认把日期 t 的新闻放到下一交易日执行（开盘或收盘二选一，写死并做敏感性）。盘中时间戳未知时，用当日收盘到次日收盘就是未来函数。缺新闻写成缺失，不要填 0。
2. **公司归属。** 标题或高精度别名命中才记到该代码。整篇宏观稿的情感不要广播给文中每一个公司名。
3. **去重。** 同一代码、同一日，对标题做规范化哈希；通讯社转载只留一条。重复计数会把覆盖度做成假信号。
4. **词典的时点。** LM 词表的类别列是「加入年份」，负号是「该年版本删除」。回测某年文本时，只用当时已经在表里、且尚未删除的词。直接拿 2026-03 的全表扫 2010 年新闻，等于用未来词表。
5. **执行与检验。** 单边成本至少按 5–10 bp 加压，空头另计借券。截面去均值后做美元中性。报告秩相关 IC、分位是否单调、换手。试过多种平滑窗口时用 Deflated Sharpe，把试验次数算进去。标准化、去重阈值、词表都只在训练窗拟合。
6. **不要训练「预测价格」的 LSTM。** 情感是慢变量，价格水平是非平稳序列。若一定要监督模型，标签用未来收益，切分用带 embargo 的滚动窗，缩放器只在训练折上拟合。

可抄评估设计、不要抄成绩的参考：

- `stefan-jansen/machine-learning-for-trading`（2026-10-13 仍有推送，约 2.1 万 star）。第 10 章 `07_news_return_signals.py` 把 FNSPID 新闻日 t 对到下一交易日收益，并在笔记里写明：他们自己的 news-surprise IC 与 0 无法区分，分位排序也不按构造方向走。借时钟和去重，不借因子。
- `ranafaraz/FinSight`（MIT，2026-06-23，0 star）。结构是对的：截面 z 分数、滞后 1 天、10 bp、purged walk-forward、Deflated Sharpe。README 里的夏普 1.93、DSR 0.997 来自**植入了情感→收益关系的合成市场**，不是行情结果。

---

## 2. 可以直接借鉴的信号公式

符号：`n_pos`、`n_neg` 为命中词数；`n_dict` 为落在 LM 主词表内的 token 数（不是原始空格切词）。标题很短，`n_pos + n_neg = 0` 时记缺失。

**标题极性（García 2013 的形式，词表换成 LM，不要用 Harvard GI）：**

```text
p = (n_pos - n_neg) / (n_pos + n_neg)
```

日频：决策时刻之前、去重后的标题取均值。截面：

```text
z_i = (p_i - mean_i) / std_i
w_i = z_i / sum_j |z_j|
```

`w` 在下一交易日成交。

**长文负向比例（Loughran & McDonald 2011，10-K / 长新闻，正词表预测力弱）：**

```text
FinNeg = n_neg / n_dict
```

长文可选 tf-idf（论文式）：`w_ij = (1 + log tf_ij) / (1 + log a_i) * log(N / df_j)`，文档分是负向词权重之和。标题不要用 tf-idf，文档频率不稳定。

**FinBERT 句分（仅抽样审计，见第 4 节）：**

```text
score = P(positive) - P(negative)    # 落在 [-1, 1]
```

用概率差，不要用 argmax。中性概率会把分数拉向 0，这是应有的行为。Prosus 的类别序是 `0=positive, 1=negative, 2=neutral`。

**GDELT 文档 tone（已算好，见第 6 节）：**

```text
Tone = PositiveScore - NegativeScore
```

两个 Score 都是「词数百分比」，所以 Tone 的单位是百分点，不是概率。

**不要当成已验证 α 的公式**（Jansen 笔记本里的构造，同文结论是样本上没有边）：

```text
surprise_t = cosine_distance(emb_t, mean(emb of prior 20 news-days))
weighted_t = surprise_t * sign(sentiment_t)
```

基线只能用 t 之前的嵌入。即便如此，作者写明加权后的 IC 仍与 0 无法区分。

---

## 3. 不要接入的仓库

| 仓库 | 最后代码推送 | 原因 |
|---|---|---|
| `krishnaik06/Stock-Sentiment-Analysis` | 2024-03-20 | `Data.csv` 是 Reddit r/worldnews 当日 Top25 标题，标签是当日道指涨跌。切分用了 2015-01-01，这一步没问题。它不是公司情感，没有成本，指标是分类准确率。 |
| `shirosaidev/stocksight` | 2023-12-05 | Elasticsearch 5、Tweepy、VADER/TextBlob。没有回测引擎。Twitter API v1 已不可用。 |
| `kaushikjadhav01/Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis` | 2024-02-19 | README 写明代码里带 API key。ARIMA/LSTM 预测未来 7 日价格，推文情感只是事后口播。 |
| `WayneDW/Sentiment-Analysis-in-Event-Driven-Stock-Price-Movement-Prediction` | 2024-07-15 | 爬 2018 年 Reuters 页面，标签是收益方向，贝叶斯 CNN 直接喊买卖。爬虫与页面都已过时，全文再分发有版权问题。882 star 是历史热度。 |
| `harshm2601/News-Based-Stock-Index-Prediction` | 2025-04-27 | `3_news_data_cleaning.py` 把新闻按日历日滤进股价日期，再送 LSTM。这是同日对齐。README 未写下一交易日成交和成本。 |
| `Thirishaa/finbert-lstm-stock-predictor`、`Compilewitharpit/finbert-lstm-stock-predictor` | 2025-05 / 2025-09 | FinBERT 嵌入与 LSTM 价格预测融合，yfinance，样本从 2023 年起。0–3 star。未见成本与时点控制。 |
| `Guannings/stark-finbert-forensics` | 2026-07-31 | 宣称 2009–2026 的 8500 万条已打分标题，parquet 不在仓库里。相似标题的判决使用这些标题之后 1/5/10 日的已实现收益；默认检索没有强制 as-of。0 star。 |
| `GeneralMills/pytrends` | 2024-08-10 | 已 archive。见第 8 节。 |
| 任意 GitHub 上的 `Loughran-McDonald_MasterDictionary.csv` | — | 没有 1993–2025 官方镜像。文件名不带年份的副本无法做时点过滤。 |
| VADER、TextBlob、Harvard General Inquirer | — | LM 2011 的出发点就是这些表把 liability、tax、cost 当成负面。金融主信号不要用。 |

`siddhant-rajhans/bloomberg-sentiment-finbert`（2026-05-31，1 star）可以读，不要交易它的结果。30 只标普 100 成分、2018–2026、Bloomberg `NEWS_SENTIMENT_DAILY_AVG`：1 日截面 IC = +0.004（t = 0.76）；21 日 IC = +0.015（重叠收益，t 被放大）；五分位多空、5 日再平衡、5 bp，FF3 α 约 +10.7%/年，Newey–West t = 1.01。Q4 收益高于 Q5 是事后发现，未预注册。FinBERT 打分约 300 条 Yahoo 标题，与 Bloomberg 的 Spearman = −0.26（N = 41）。数据因终端条款未入库，仓库不能复现。

---

## 4. FinBERT：要用的话，最小依赖与权重

仓库：<https://github.com/ProsusAI/finBERT>（大小写是 `finBERT`）。未 archive。2026-10-12 推送，2253 star。模型卡：<https://huggingface.co/ProsusAI/finbert>，同日更新，累计下载约 428 万。

许可证必须分开看。GitHub 的 Apache-2.0 **只覆盖**仓库里的训练代码和笔记本（许可方 MIH AI B.V.，声明生效日 2026-10-12）。权重是 **CC-BY-NC-SA-4.0**。Financial PhraseBank 另有自己的 CC-BY-NC-SA-3.0。商业交易系统不能把这份权重打进去。

模型是 BERT-base：12 层，隐藏维 768，12 头，词表 30522，`max_position_embeddings` 512，约 1.1 亿参数。`config.json` 的 `id2label`：

```text
0 positive / 1 negative / 2 neutral
```

这与 `yiyanghkust/finbert-tone` 相反（那边是 0 Neutral、1 Positive、2 Negative，且只有 pickle 的 `pytorch_model.bin`，模型卡无 license 字段，2022-10 后再无更新）。两套权重混用会把符号打反。Jansen 笔记本用的是 tone 那一版，并且把 argmax 映成 −1/0/+1，不是概率差。

权重体积（Hugging Face tree API，2026-10-14）：

| 文件 | 大小 |
|---|---|
| `model.safetensors` | 417.7 MiB（约 438 MB） |
| `pytorch_model.bin` | 417.7 MiB，pickle |
| `flax_model.msgpack` | 417.7 MiB |
| `tf_model.h5` | 417.9 MiB |
| `vocab.txt` | 0.2 MiB |

四个格式是同一套权重。仓库合计约 1.2 GiB。最小下载只有 `config.json`、tokenizer 三个小文件、`model.safetensors`，约 420 MiB。推理依赖只有 `transformers` 和 CPU 版 `torch`。不要装仓库的 `environment.yml`，那是训练环境。官方 `scripts/predict.py` 的分数就是 `P(pos) − P(neg)`。训练笔记本 `max_seq_length = 64`，面向句子。

**不要当主信号。** FNSPID 量级是一千五百万篇。BERT 对全文、甚至对全部标题做日更，成本和延迟都不适合日频主信号；截断到 512 token 也不是文档级情感。PhraseBank 的领域是赫尔辛基交易所的句子，不是美股代码模型。Rajhans 的小样本已经显示：Yahoo 标题上的 Prosus FinBERT 复现不了 Bloomberg 情感。若要神经网络，只在去重后的标题抽样上做审计（几千条，不是几百万条），并且接受非商用许可；主分数仍用第 2 节的词表公式。

---

## 5. Loughran–McDonald 主词表

官方页仍是 <https://sraf.nd.edu/loughranmcdonald-master-dictionary/>。2026-10-14 打开，页眉为 **Updated: March, 2026**。

文件名：

- `Loughran-McDonald_MasterDictionary_1993-2025.csv`
- `Loughran-McDonald_MasterDictionary_1993-2025.xlsx`

页面上的 CSV 链接是 Google Drive，不是 nd.edu 上的直链：

`https://drive.google.com/file/d/1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF/view`

说明 PDF：`https://drive.google.com/file/d/1qMZ4WTDv9bq2iYOxmMP5h_JgdLb46z_h/view`

词表加载模块：`https://drive.google.com/file/d/18jbZ3o17PRI_s4xG9UslKnGMpnC1ZoLM/view`（代码页称之为 `MOD_Load_MasterDictionary_v2024.py`，并写明不兼容旧版词表；代码页里的数据文件名少了连字符，以主页文件名为准）。

学术研究免费。商业许可写信 `loughranmcdonald@gmail.com`。引用 Loughran & McDonald, Journal of Finance, 2011。

类别列：Negative、Positive、Uncertainty、Litigious、Strong_Modal、Weak_Modal、Constraining，外加 2024 JFQA 的 Complexity。非零数字是加入年份；负号表示该年版本从该类删除。另有词频、文档频率、音节数，可做第 2 节的 tf-idf，但 `df` 是按 10-K 语料估的，不要直接套到新闻标题上。

没有官方 GitHub 镜像。只从 SRAF 页面下载，本地钉死「2026-03 / 1993–2025」这份，并按第 1 节做年份过滤。

---

## 6. GDELT 的 tone

依据：GKG Codebook V2.1（2015-02-19），字段 `V1.5TONE`。PDF：<https://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf>

逗号分隔，顺序为：

1. **Tone**：全文平均调性。`PositiveScore − NegativeScore`。理论范围 **−100 到 +100**。常见值 **−10 到 +10**，0 为中性。接近 0 有两种情况：情绪词很少，或正负比例差不多。用 Polarity 区分。
2. **Positive Score**：正面词占全文词数的百分比，0 到 100。
3. **Negative Score**：负面词占比，0 到 100。V2.1 正文把这一条误写成 “positive emotional connotation”，与上一条重复，这是笔误；含义是负面词占比。
4. **Polarity**：命中调性词典的词占比。Polarity 高而 Tone 接近 0，是情绪强但方向对冲，不要当成平静中性。
5. Activity Reference Density。
6. Self/Group Reference Density。新闻里通常很低。
7. **Word Count**：v1.5 起追加。前言仍写 “six”，实际是 7 个数。

事件表的 **AvgTone**（Event Codebook V2.0）范围相同，常见值同样是 −10 到 +10。它只平均该事件**第一次出现的那个 15 分钟更新**里的文档，之后的报道不回写。手册写明这是遗留字段，并建议改用 GKG。事件 AvgTone 不要当公司日频情感。

GKG 的机构名不是交易代码，tone 是整篇文档。一家公司出现在一篇负面宏观稿里，会继承整篇的 Tone。手册建议对机构名做直方图，丢掉只出现一两次的名字。公司级信号仍要用自己的别名表，只取标题或高置信提及，再聚合到代码。

GDELT 的价值是 tone 已经算好、历史长、不用自己跑 BERT。它用的是通用情绪词，不是 LM 金融词表。适合做宏观或事件过滤，不适合直接当个股主信号。

---

## 7. 还能看的回测仓库：方法与已知缺陷

真正还在动的是教材仓库，不是「情感选股」产品。

### `stefan-jansen/machine-learning-for-trading`

- 推送 2026-10-13。第 10 章 `07_news_return_signals.py`、`09_filing_text_signals.py`。
- 方法：FNSPID 标题 → 嵌入；`surprise = cosine_distance(今日, 过去 20 个新闻日均值)`；`weighted = surprise × sign(sentiment)`；情感模型是 `yiyanghkust/finbert-tone`，argmax 映到 −1/0/+1。信号日 t 用 `join_asof` 推到下一交易日，对 `adj_close` 的 1/5/20 日前向收益做截面 Spearman IC 和五分位。10-Q 笔记把信号锚在 SEC 接受日，而不是财季结束日。
- 缺陷（作者写在同一文件的 Key takeaways 里）：该子集上 IC 与 0 无法区分，五分位不单调；`surprise × sign` 只是把一个不显著的系数挪到另一个不显著的系数；按「新闻日」回看 20 步会只留下高关注度股票；结果文件里仍有一句把 weighted surprise 称为关键特征，与正文结论互相打架。笔记本默认 `MAX_ARTICLES = 50000`，不是全量。没有交易成本。嵌入模型重，不能当生产依赖。

### `ranafaraz/FinSight`

- 方法：离线词表或可选 FinBERT；合成或 yfinance；截面美元中性；滞后 1 天；10 bp；事件研究 CAR；PSR / DSR；purged + embargo 的 walk-forward。打乱信号后 DSR 掉到 0，这个对照值得抄。
- 缺陷：成绩是合成 α。0 star，不能当信号库。

### `siddhant-rajhans/bloomberg-sentiment-finbert`

- 方法见第 3 节。检验设计（多期限 IC、重叠修正、5 bp、FF3 + Newey–West、分位是否单调）值得抄。
- 缺陷：1 日 IC 不成立，α 不显著，FinBERT 对照只有 41 个点，Bloomberg 数据不能重跑。

### FNSPID 论文附带的实验（`Zdong104/FNSPID_Financial_News_Dataset`）

- 方法：新闻与价格对齐后做深度学习预测；情感分是 ChatGPT 在小子集上打的 1–5 分（论文 Task 3，约 50 只股票的摘要），不是全库字段。
- 缺陷：目标是预测精度而不是扣成本的可交易收益；维护已停（见第 8 节）。不要把论文里的「情感略微提高准确率」当成回测边。

学生项目 `finbert-lstm`、`News-Based-Stock-Index-Prediction` 的共同缺陷：同日合并、没有成本、在很短的 yfinance 窗口上调 LSTM、用准确率或价格水平当目标。见第 3 节，不要复用。

---

## 8. Hugging Face 上能用于回测的免费数据集

标准：免费可下，且同时有**日期**和**公司或代码**。只有句子标签的，列在表后，标明不能回测。

### 可以用（研究回测，先处理许可证和时点）

**`Zihan1004/FNSPID`**  
<https://huggingface.co/datasets/Zihan1004/FNSPID>  
GitHub：<https://github.com/Zdong104/FNSPID_Financial_News_Dataset>  
datasets-server 字段（2026-10-14）：`Date`, `Article_title`, `Stock_symbol`, `Url`, `Publisher`, `Author`, `Article`, `Lsa_summary`, `Luhn_summary`, `Textrank_summary`, `Lexrank_summary`。样例：`2020-06-05 06:30:54 UTC`，代码 `A`。有日期、有代码。  
论文 arXiv:2402.06698：1999–2023，约 1570 万条新闻、2970 万条价格，作者写成 “4,775 S&P 500 companies”。标普 500 没有 4775 家，这个宇宙要当成历史美股名单，有幸存者偏差。  
没有全量情感列。ChatGPT 分数只在小子集。查看器样例里正文经常是空的，可靠字段是标题、时间、代码。  
`LICENSE` 文件仍是 **CC BY-NC 4.0**。README 在 2025-07-24 写 “停止维护，研究与商业权利都放开”，同一篇 README 后文仍禁止商用。以 LICENSE 文件为准：非商用，且项目已停更。Nasdaq 抓取脚本不要当生产采集器。Jansen 用这份数据没有做出可区分的 IC。

**`luckycat37/financial-news-dataset`**  
<https://huggingface.co/datasets/luckycat37/financial-news-dataset>  
51,272 篇 finance.yahoo.com，2017–2023。模型卡字段含 `date_publish`（UTC）、`mentioned_companies`（代码）、`sentiment`（negative/neutral/positive 概率）。有日期、有代码、带模型情感。许可 **CC-BY-NC-SA-4.0**。  
用法：只用 `date_publish` 与 `mentioned_companies`。`related_companies` 是同业扩展，不是文中提及。`curr_day_price_*` 与 `next_day_price_*` 已经写在同一行，拿来当特征就是未来函数；价格用自己的时点行情重接。情感是模型概率，卡上没写模型名，不是人工标签。Yahoo 原文的版权不因上传者声明而消失。下载量约 93，只适合做小样本对照。

### 免费，但不能用于公司回测

| 数据集 | 有什么 | 缺什么 |
|---|---|---|
| `takala/financial_phrasebank`（别名 `financial_phrasebank`） | 句子 + 正/中/负。50% 同意 4846 句，全同意 2264 句。CC-BY-NC-SA-3.0 | 无日期、无代码。来源是 OMX 赫尔辛基，代码已被拿掉。只能做分类器评估。Prosus FinBERT 的监督数据就是它。 |
| `zeroshot/twitter-financial-news-sentiment` | 11,932 条推文，熊/牛/中性。MIT。训练 9938 / 验证 2486 | 无时间戳列。`$BYND` 这类代码只是偶然出现在文本里。 |
| `tasksource/sentfin`（源 `pyRis/SEntFiN` 的 `SEntFiN.csv`） | 10,753 条印度财经标题，实体→正/负/中性。列：`S No.`, `Title`, `Decisions`, `Words` | 无日期。实体是 “Market”“Mid caps” 这种名字，不是交易代码。 |
| `kdave/Indian_Financial_News` | URL、正文、摘要、Sentiment | 无日期列、无代码列。 |
| `ashraq/financial-news-articles` | 306,242 行，title/text/url，来自 Kaggle `jeet2016/us-financial-news-articles` | 无日期、无代码。 |
| `danidanou/Reuters_Financial_News` | 105,359 篇，2006–2013，有 `Date`（如 `Mon Nov 13, 2006`）和正文 | 代码只嵌在正文里，如 `(GE.N)`。上传者标了 Apache-2.0，这不能清除 Reuters 版权。 |
| `yumoxu/stocknet-dataset` | 推文 + 价格，有日期和约 88 个代码 | 最后推送 2019-03-06，样本停在 2014–2016。社交、停更，不要当依赖。 |

---

## 9. pytrends / Google Trends：现在经常失败

是。不要依赖。

- 仓库 <https://github.com/GeneralMills/pytrends> 的 API 显示 `archived: true`。最后一次代码推送 **2024-08-10**。README 原文：只在 Google 不再改后端之前好用，并在找维护者。
- PyPI 上 `pytrends` 最新仍是 **4.9.2**，没有 yanked。包还在，不等于能用。
- 未关闭的 issue：#588、#614、#618 都是 HTTP 429。#638（2025-02-23 开，检索当日仍 open）标题是 Google 改了 Trends 端点，旧 URL 拿不到数据。
- 2025–2026 的使用记录与此一致：未预热 `NID` cookie 时第一次调用就 429；`trending_searches()` 打到已删除页面会 404；数据中心 IP（含云主机）比住宅 IP 更容易被拒。一次 `interest_over_time` 实际是多次 HTTP（token 加各个 widget）。限额按 IP 和 cookie，不公开。
- 这是非官方接口。加重试、换 cookie、上代理，都是在对抗一个会继续改的端点，不构成数据源。

若以后要搜索热度，只能是人工或另行取得的、带 as-of 时间的缓存，缺失保持缺失。它不是主信号，也不要放进每日任务的关键路径。

---

## 10. 落地顺序

1. 从 SRAF 下载 1993–2025 词表，按加入/删除年份过滤，标题用 `(n_pos - n_neg) / (n_pos + n_neg)`，长文用 `n_neg / n_dict`。
2. 研究样本用 FNSPID 的标题 + `Date` + `Stock_symbol`（CC BY-NC，维护已停）。价格自己接，信号用下一交易日。小样本对照可用 `luckycat37` 的 `date_publish` + `mentioned_companies`，丢掉行内的 next-day price。
3. GDELT `V1.5TONE` 的第一列只做宏观或事件过滤；常见幅度按 −10 到 +10 理解，超过这个区间单独检查。个股归属自己做。
4. 评估抄 FinSight / Jansen 的时点、成本、分位单调、试验次数惩罚。不抄他们的夏普数字。
5. Prosus FinBERT 仅在非商用的标题抽样里审计，只下 safetensors（约 420 MiB），分数用 `P(pos)−P(neg)`，类别序按本笔记第 4 节。全文与日频主信号都不用它。
6. pytrends 不接入。

## 来源（均于 2026-10-14 打开）

- <https://github.com/ProsusAI/finBERT>
- <https://huggingface.co/ProsusAI/finbert> 与 `config.json`、tree API
- <https://sraf.nd.edu/loughranmcdonald-master-dictionary/>
- <https://sraf.nd.edu/textual-analysis/code/>
- <https://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf>
- <https://data.gdeltproject.org/documentation/GDELT-Event_Codebook-V2.0.pdf>
- <https://github.com/stefan-jansen/machine-learning-for-trading> `10_text_feature_engineering/07_news_return_signals.py`
- <https://github.com/ranafaraz/FinSight>
- <https://github.com/siddhant-rajhans/bloomberg-sentiment-finbert>
- <https://github.com/Zdong104/FNSPID_Financial_News_Dataset> 与 <https://huggingface.co/datasets/Zihan1004/FNSPID>
- <https://huggingface.co/datasets/luckycat37/financial-news-dataset>
- <https://huggingface.co/datasets/takala/financial_phrasebank>
- <https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment>
- <https://github.com/pyRis/SEntFiN> `SEntFiN.csv` 表头
- <https://github.com/GeneralMills/pytrends> 与 issues #618、#638；PyPI `pytrends` 4.9.2
- Loughran, T. and McDonald, B. (2011). When Is a Liability Not a Liability? *Journal of Finance* 66(1), 35–65.
- García, D. (2013). Sentiment during Recessions. *Journal of Finance* 68(3), 1267–1300.
- Dong, Z., Fan, X., and Peng, Z. (2024). FNSPID. arXiv:2402.06698.
