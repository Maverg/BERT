
# Tweet Sentiment + Market Backtesting (BERT)

This project fine-tunes a BERT model to classify financial tweet sentiment (negative / neutral / positive), then evaluates whether aggregated daily sentiment predicts next-day price returns.

## What’s included
- **BERT training + evaluation**: trains a `bert-base-uncased` sequence classifier on a finance sentiment dataset and reports accuracy + confusion matrix.
- **Finance evaluation / backtest** (`finance_backtest.py`): runs inference on tweet data, builds a daily sentiment index, computes correlation (r) vs next-day returns, and backtests a simple sentiment-driven strategy.

---

## Dependencies

### Python version
- Python 3.9+ recommended

### Core packages
- `transformers`
- `torch`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `gdown`
- `tensorflow`
- `keras`

### Finance/backtest script extras
- `scipy`
- `yfinance`

### Install
```bash
pip install -U transformers torch gdown numpy pandas seaborn matplotlib scikit-learn tensorflow keras scipy yfinance
