"""Constants frozen before the market backtest. Do not retune them on results."""

# Shock rule. 1.5 past-only standard deviations, not a searched quantile.
SHOCK_Z = 1.5

# Da, Engelberg, Gao (2015): tradable reversal is the next two sessions.
HOLD_FAST = 2
DELAY_FAST = 0

# Same two-day hold, but one extra session late (published-series lag).
HOLD_SLOW = 2
DELAY_SLOW = 1

# Tetlock (2007): skip the next session, then hold the reversal window.
HOLD_TETLOCK = 4
DELAY_TETLOCK = 1

# 5 bp per side = 10 bp round trip, the cost that wiped out Tetlock-Saar-Macskassy.
COST_PER_SIDE = 0.0005
COST_STRESS = 0.001  # 10 bp per side, sensitivity only

Z_WINDOW = 252
MIN_TRADES = 15
PRIOR_STRENGTH = 20
# Display cap while the sample still has fewer than MIN_TRADES completed bets.
PRIOR_SCORE_CAP = 0.3
# Shrunk excess return that maps to a bullish score of 1. One hundred basis points.
SCORE_SCALE = 0.01

N_PERM = 1000
RANDOM_SEED = 0

TRAIN_START = "1995-01-01"
TRAIN_END = "2004-12-31"
VAL_START = "2005-01-01"
VAL_END = "2009-12-31"
NEWS_EVAL_START = "2010-01-01"
# Day after the FRBSF Economic Letter that documented this news-sentiment series.
POST_LETTER = "2020-04-07"
WIKI_EVAL_START = "2016-07-01"

PRICE_START = "1993-01-01"
PRICE_END = "2026-10-16"
WIKI_START = "20150701"
WIKI_END = "20261014"

# Exploratory grid. Reported, never used to choose the production rule.
EXPLORATORY_Z = (1.0, 2.0)
EXPLORATORY_HOLDS = (1, 4, 5, 10)
EVENT_HORIZONS = (1, 2, 4, 5, 10, 20)

# Pages frozen for meaning, before any return was computed. High views = fear attention.
WIKI_FEAR_PAGES = (
    "Stock_market_crash",
    "Recession",
    "Bear_market",
    "Bankruptcy",
    "Bank_run",
    "Unemployment",
    "Financial_crisis",
    "Inflation",
    "Layoff",
    "Volatility_(finance)",
    "Default_(finance)",
    "Great_Depression",
    "Economic_collapse",
    "Wall_Street",
)

USER_AGENT = "sentiment-long-research/0.1 (academic backtest; local)"
