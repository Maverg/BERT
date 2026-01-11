import os, textwrap, json, datetime, re, math, pathlib

out_path = "/mnt/data/finance_backtest.py"

code = r'''"""
finance_backtest.py
-------------------
End-to-end "finance side" evaluation for a tweet-sentiment model:

1) Loads tweets (CSV) with timestamps + tickers + text
2) Runs inference with a (fine-tuned) BERT/FinBERT sentiment classifier
3) Builds a DAILY sentiment index per ticker (weighted optional)
4) Downloads price data (default: yfinance) and computes next-day returns
5) Evaluates predictive strength:
   - Pearson/Spearman correlation (r) + p-value
   - Optional bootstrap CIs
6) Backtests a simple sentiment-driven strategy and reports:
   - cumulative returns, Sharpe, max drawdown, hit-rate, turnover
   - CSV outputs + plots

- avoids look-ahead bias (uses sentiment on day t to predict return t+1)
- works with any Hugging Face sequence classification model directory


- ticker: e.g. TSLA, AAPL, BTC-USD (yfinance format)
- created_at: timestamp (ISO recommended)
- text: the tweet text

Optional columns for weighting:
- like_count, retweet_count, reply_count, quote_count, followers_count

USAGE EXAMPLE:
python finance_backtest.py \
  --tweets tweets.csv \
  --model_dir ./finbert_twitter \
  --start 2025-03-01 --end 2025-06-15 \
  --out_dir results \
  --threshold 0.10 \
  --tcost_bps 5

NOTES:
- If you're using your current notebook training code, save the fine-tuned model:
    model.save_pretrained("./finbert_twitter")
    tokenizer.save_pretrained("./finbert_twitter")
  Then point --model_dir at that folder.

- yfinance needs internet.

pip install -U pandas numpy torch transformers scipy yfinance matplotlib
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import logging
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Core stats
from scipy import stats

# Model inference
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Prices (online)
try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

# Plotting
import matplotlib.pyplot as plt


# ----------------------------
# Logging / utils
# ----------------------------

def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    logger = logging.getLogger("finance_backtest")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("Logging to %s", log_path)
    return logger


def parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def to_utc_date(ts: pd.Series) -> pd.Series:
    """
    Convert a timestamp series to UTC date.
    If timestamps are naive, we assume they're already UTC.
    """
    t = pd.to_datetime(ts, errors="coerce", utc=True)
    return t.dt.date


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


# ----------------------------
# Model scoring
# ----------------------------

@dataclasses.dataclass
class SentimentScorer:
    model_dir: Path
    device: str = "auto"
    batch_size: int = 64
    max_length: int = 128

    def __post_init__(self) -> None:
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_t = torch.device(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device_t)
        self.model.eval()

        # Expect 3 labels: negative, neutral, positive (0,1,2)
        self.num_labels = int(self.model.config.num_labels)

    @torch.no_grad()
    def score_texts(self, texts: List[str]) -> pd.DataFrame:
        """
        Returns a DataFrame with:
        - p_neg, p_neu, p_pos
        - sentiment_score = p_pos - p_neg (range roughly [-1,1])
        - pred_label (argmax)
        """
        if self.num_labels != 3:
            raise ValueError(f"Expected 3 labels, got {self.num_labels}. "
                             "Update mapping or retrain model with num_labels=3.")

        all_probs: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device_t) for k, v in enc.items()}
            out = self.model(**enc)
            logits = out.logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            all_probs.append(probs)

        probs = np.vstack(all_probs) if all_probs else np.zeros((0, 3))
        df = pd.DataFrame(probs, columns=["p_neg", "p_neu", "p_pos"])
        df["sentiment_score"] = df["p_pos"] - df["p_neg"]
        df["pred_label"] = df[["p_neg", "p_neu", "p_pos"]].to_numpy().argmax(axis=1)
        return df


# Data loading

def load_tweets(path: Path, logger: logging.Logger) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"ticker", "created_at", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Tweets CSV missing required columns: {sorted(missing)}")

    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.replace("$", "", regex=False).str.strip()
    df["date"] = to_utc_date(df["created_at"])
    df = df.dropna(subset=["date", "text", "ticker"]).reset_index(drop=True)

    logger.info("Loaded tweets: %d rows, %d tickers, date range [%s .. %s]",
                len(df), df["ticker"].nunique(), df["date"].min(), df["date"].max())
    return df


def compute_weight(df: pd.DataFrame) -> pd.Series:
    """
    Optional weighting:
    If engagement columns exist, build a soft weight.
    Else returns 1.0 for each tweet.
    """
    cols = ["like_count", "retweet_count", "reply_count", "quote_count", "followers_count"]
    available = [c for c in cols if c in df.columns]
    if not available:
        return pd.Series(np.ones(len(df)), index=df.index)

    w = np.ones(len(df), dtype=float)
    if "like_count" in df.columns:
        w += np.log1p(df["like_count"].fillna(0).to_numpy())
    if "retweet_count" in df.columns:
        w += np.log1p(df["retweet_count"].fillna(0).to_numpy())
    if "reply_count" in df.columns:
        w += 0.5 * np.log1p(df["reply_count"].fillna(0).to_numpy())
    if "quote_count" in df.columns:
        w += 0.5 * np.log1p(df["quote_count"].fillna(0).to_numpy())
    if "followers_count" in df.columns:
        w *= (1.0 + 0.1 * np.log1p(df["followers_count"].fillna(0).to_numpy()))

    return pd.Series(w, index=df.index)


def build_daily_sentiment_index(
    tweets: pd.DataFrame,
    logger: logging.Logger,
    min_tweets_per_day: int = 5,
) -> pd.DataFrame:
    """
    Builds a daily sentiment index per ticker:
      S(ticker, day) = weighted_average(sentiment_score)
    Filters out ticker-days with too few tweets.
    """
    df = tweets.copy()
    df["w"] = compute_weight(df)
    # weighted average
    grp = df.groupby(["ticker", "date"], as_index=False).apply(
        lambda g: pd.Series({
            "sentiment": np.average(g["sentiment_score"], weights=g["w"]) if len(g) else np.nan,
            "tweet_count": len(g),
            "w_sum": float(np.sum(g["w"])),
        })
    ).reset_index(drop=True)

    before = len(grp)
    grp = grp[grp["tweet_count"] >= min_tweets_per_day].copy()
    after = len(grp)

    logger.info("Daily sentiment index: %d ticker-days (filtered %d -> %d by min_tweets_per_day=%d)",
                after, before, after, min_tweets_per_day)
    return grp



# Price data / returns

def load_prices_yfinance(
    tickers: List[str],
    start: dt.date,
    end: dt.date,
    logger: logging.Logger,
) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance is not installed (or failed import). Install with: pip install yfinance")

    # yfinance end is exclusive-ish; add buffer
    start_s = start.isoformat()
    end_s = (end + dt.timedelta(days=3)).isoformat()

    logger.info("Downloading prices from yfinance for %d tickers, %s..%s", len(tickers), start_s, end_s)
    data = yf.download(
        tickers=tickers,
        start=start_s,
        end=end_s,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    # Normalize to long format with (ticker, date)
    rows = []
    if isinstance(data.columns, pd.MultiIndex):
        # Multi-ticker
        for t in tickers:
            if (t, "Close") not in data.columns:
                continue
            tmp = data[t].copy()
            tmp["ticker"] = t
            tmp = tmp.reset_index().rename(columns={"Date": "date"})
            rows.append(tmp[["ticker", "date", "Open", "High", "Low", "Close", "Volume"]])
    else:
        # Single ticker
        tmp = data.copy()
        tmp["ticker"] = tickers[0]
        tmp = tmp.reset_index().rename(columns={"Date": "date"})
        rows.append(tmp[["ticker", "date", "Open", "High", "Low", "Close", "Volume"]])

    px = pd.concat(rows, ignore_index=True)
    px["date"] = pd.to_datetime(px["date"]).dt.date
    px = px.dropna(subset=["Close"]).reset_index(drop=True)

    logger.info("Loaded prices: %d rows", len(px))
    return px


def load_prices_csv(path: Path, logger: logging.Logger) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"ticker", "date", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prices CSV missing required columns: {sorted(missing)} "
                         f"(expected at least ticker,date,Close)")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df = df.dropna(subset=["date", "Close", "ticker"]).reset_index(drop=True)
    logger.info("Loaded prices CSV: %d rows, %d tickers", len(df), df["ticker"].nunique())
    return df


def compute_next_day_returns(px: pd.DataFrame) -> pd.DataFrame:
    """
    Computes next-day close-to-close returns per ticker.
    Return aligned to day t (i.e. return_tplus1 is for t -> next trading day).
    """
    df = px.sort_values(["ticker", "date"]).copy()
    df["close"] = df["Close"].astype(float)
    df["ret_tplus1"] = df.groupby("ticker")["close"].pct_change().shift(-1)
    return df[["ticker", "date", "ret_tplus1", "close"]]



# Evaluation / backtest (##Oct 8, 2025)


def pearsonr_safe(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(x[mask], y[mask])
    return float(r), float(p)


def spearmanr_safe(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan"), float("nan")
    r, p = stats.spearmanr(x[mask], y[mask])
    return float(r), float(p)


def bootstrap_corr_ci(x: np.ndarray, y: np.ndarray, n: int = 2000, seed: int = 42) -> Tuple[float, float]:
    """
    Simple bootstrap CI for Pearson correlation.
    """
    rng = np.random.default_rng(seed)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return float("nan"), float("nan")
    idx = np.arange(len(x))
    rs = []
    for _ in range(n):
        samp = rng.choice(idx, size=len(idx), replace=True)
        r, _ = stats.pearsonr(x[samp], y[samp])
        rs.append(r)
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return float(lo), float(hi)


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def annualized_sharpe(daily_ret: pd.Series, trading_days: int = 252) -> float:
    r = daily_ret.dropna()
    if len(r) < 10:
        return float("nan")
    mu = r.mean()
    sd = r.std(ddof=1)
    return float((mu / sd) * math.sqrt(trading_days)) if sd else float("nan")


def run_strategy(
    df: pd.DataFrame,
    threshold: float,
    tcost_bps: float,
) -> pd.DataFrame:
    """
    df columns:
      - ticker, date, sentiment, ret_tplus1

    Strategy:
      position_t = sign(sentiment_t) if |sentiment_t| >= threshold else 0
      strategy_ret_t = position_t * ret_tplus1 - costs
    Costs:
      - transaction cost applied on position changes: tcost_bps per 1.0 notional change
    """
    out = df.sort_values(["ticker", "date"]).copy()

    # Position by thresholded sentiment
    out["pos"] = 0.0
    out.loc[out["sentiment"] >= threshold, "pos"] = 1.0
    out.loc[out["sentiment"] <= -threshold, "pos"] = -1.0

    # Turnover (abs change in position)
    out["pos_prev"] = out.groupby("ticker")["pos"].shift(1).fillna(0.0)
    out["turnover"] = (out["pos"] - out["pos_prev"]).abs()

    # Transaction cost in decimal return (bps = 1e-4)
    out["cost"] = out["turnover"] * (tcost_bps * 1e-4)

    out["strategy_ret"] = out["pos"] * out["ret_tplus1"] - out["cost"]
    return out


def summarize_results(df: pd.DataFrame) -> Dict[str, float]:
    r = df["strategy_ret"].dropna()
    eq = (1.0 + r).cumprod()

    return {
        "n_obs": float(len(r)),
        "mean_daily_return": float(r.mean()),
        "vol_daily": float(r.std(ddof=1)),
        "sharpe_ann": annualized_sharpe(r),
        "max_drawdown": max_drawdown(eq) if len(eq) else float("nan"),
        "hit_rate": float((r > 0).mean()) if len(r) else float("nan"),
        "avg_turnover": float(df["turnover"].dropna().mean()) if "turnover" in df.columns else float("nan"),
        "total_return": float(eq.iloc[-1] - 1.0) if len(eq) else float("nan"),
    }


def plot_equity_curve(strategy_ret: pd.Series, out_path: Path, title: str) -> None:
    r = strategy_ret.dropna()
    if len(r) < 3:
        return
    eq = (1.0 + r).cumprod()
    plt.figure()
    plt.plot(eq.index, eq.values)
    plt.title(title)
    plt.xlabel("time (rows)")
    plt.ylabel("equity (cumprod)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()



# MAIN

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tweets", type=str, required=True, help="Path to tweets CSV")
    ap.add_argument("--model_dir", type=str, required=True, help="HuggingFace model directory (fine-tuned)")
    ap.add_argument("--out_dir", type=str, default="results", help="Output directory")

    ap.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD (for prices)")
    ap.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD (for prices)")

    ap.add_argument("--prices_csv", type=str, default="", help="Optional: offline price CSV with columns ticker,date,Close")
    ap.add_argument("--min_tweets_per_day", type=int, default=5, help="Filter out ticker-days with fewer tweets")
    ap.add_argument("--threshold", type=float, default=0.10, help="Signal threshold for trading strategy")
    ap.add_argument("--tcost_bps", type=float, default=5.0, help="Transaction cost in basis points per unit turnover")
    ap.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    ap.add_argument("--max_length", type=int, default=128, help="Tokenizer max length")
    ap.add_argument("--bootstrap_ci", action="store_true", help="Compute bootstrap CI for Pearson r")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    logger = setup_logger(out_dir)

    tweets_path = Path(args.tweets)
    model_dir = Path(args.model_dir)

    start = parse_date(args.start)
    end = parse_date(args.end)

    logger.info("Config: %s", vars(args))

    # Load tweets
    tweets = load_tweets(tweets_path, logger)

    # Inference
    scorer = SentimentScorer(
        model_dir=model_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device="auto",
    )
    probs = scorer.score_texts(tweets["text"].astype(str).tolist())
    tweets = pd.concat([tweets.reset_index(drop=True), probs.reset_index(drop=True)], axis=1)

    # Daily sentiment index
    daily = build_daily_sentiment_index(
        tweets=tweets,
        logger=logger,
        min_tweets_per_day=args.min_tweets_per_day,
    )

    tickers = sorted(daily["ticker"].unique().tolist())
    logger.info("Tickers after filtering: %s", tickers)

    # Prices
    if args.prices_csv:
        px = load_prices_csv(Path(args.prices_csv), logger)
    else:
        px = load_prices_yfinance(tickers, start=start, end=end, logger=logger)

    rets = compute_next_day_returns(px)

    # Merge (sentiment day t with return t+1)
    merged = daily.merge(rets, on=["ticker", "date"], how="inner")
    merged = merged.dropna(subset=["sentiment", "ret_tplus1"]).reset_index(drop=True)

    logger.info("Merged dataset: %d rows (ticker-days)", len(merged))
    merged.to_csv(out_dir / "merged_sentiment_returns.csv", index=False)

    # Correlation stats
    x = merged["sentiment"].to_numpy(dtype=float)
    y = merged["ret_tplus1"].to_numpy(dtype=float)

    pear_r, pear_p = pearsonr_safe(x, y)
    spear_r, spear_p = spearmanr_safe(x, y)

    logger.info("Pearson r=%.4f p=%.4g | Spearman r=%.4f p=%.4g",
                pear_r, pear_p, spear_r, spear_p)

    if args.bootstrap_ci:
        lo, hi = bootstrap_corr_ci(x, y)
        logger.info("Bootstrap 95%% CI for Pearson r: [%.4f, %.4f]", lo, hi)

    # Backtest
    bt = run_strategy(merged, threshold=args.threshold, tcost_bps=args.tcost_bps)
    bt.to_csv(out_dir / "backtest_timeseries.csv", index=False)

    summ = summarize_results(bt)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        f.write(pd.Series(summ).to_json(indent=2))

    logger.info("Backtest summary: %s", summ)

    # Plots
    plot_equity_curve(bt["strategy_ret"], out_dir / "equity_curve.png", "Sentiment Strategy Equity Curve")

    # Basic scatter plot for report
    plt.figure()
    plt.scatter(merged["sentiment"], merged["ret_tplus1"], s=8)
    plt.title(f"Sentiment vs Next-Day Return (Pearson r={pear_r:.3f}, p={pear_p:.3g})")
    plt.xlabel("daily sentiment index")
    plt.ylabel("next-day return")
    plt.tight_layout()
    plt.savefig(out_dir / "sentiment_vs_return_scatter.png", dpi=160)
    plt.close()

    logger.info("Done. Outputs in: %s", out_dir.resolve())


if __name__ == "__main__":
    main()
'''

with open(out_path, "w", encoding="utf-8") as f:
    f.write(code)

out_path
