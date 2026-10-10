# MLTRADER

Evolutionary trading research, backtesting, training visualization, and Alpaca paper-trading tools.

> **Risk notice:** This is research software, not investment advice or a promise of profitability. Keep paper trading enabled; paper results do not guarantee live performance.

## Requirements

- Python 3.11 or newer
- Git
- Historical OHLCV CSV data for training/backtesting
- For paper trading only: an Alpaca paper account and paper API credentials

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` only if using paper trading, then fill in your **Alpaca paper-account** credentials. Never commit `.env` or real credentials.

## Market data format

Each CSV must contain `open`, `high`, `low`, `close`, and `volume` columns and at least 100 valid rows. Column names are case-insensitive. Use consistent bar intervals across instruments and avoid look-ahead leakage.

Keep local/raw market data out of Git. The default training data path is `data/raw/SPY.csv`.

## Train an evolutionary agent

Train on one or more instruments:

```bash
python evolve.py --data data/raw/SPY.csv data/raw/QQQ.csv data/raw/AAPL.csv
```

To launch the training tracker with live fitness/equity plots on a desktop that supports Tk:

```bash
python track_training.py --data data/raw/SPY.csv
```

The tracker launches the evolution process and displays the metrics it receives from training. If running without a graphical desktop, use `python evolve.py` directly.

Review `config/training.yaml` and `config/market.yaml` before long runs. Training writes generated checkpoints and champion files under `models/`, which are intentionally excluded from Git.

## Backtesting and validation

```bash
python run_backtest.py --help
python validate.py --help
```

Use held-out dates/instruments and include realistic transaction costs. A strong in-sample fitness score is not sufficient evidence of a robust strategy.

## Paper trading

1. Create a local `.env` from `.env.example` and use Alpaca **paper** credentials.
2. Train/export a model compatible with the configured observation size (64 OHLCV bars plus 4 portfolio features = 324 inputs).
3. Review `config/trading_universe.yaml`, including symbol scan limits, liquidity filters, position caps, gross exposure, and short-selling settings.
4. Start the paper trader:

   ```bash
   python -m trading.paper_trader
   ```

The trader checks whether the regular market session is open before scanning/submitting orders. It uses Alpaca's paper-trading endpoint and applies configured per-position and gross-exposure caps. Data availability, account permissions, buying power, order support, and broker rules can still prevent orders from being accepted. Do not switch this research project to live trading without a separate, deliberate safety review.

## Tests and code checks

```bash
python -m compileall -q agents backtest environment evolution trading training validation tests
python -m pytest -q
```

The test suite is a smoke check, not proof of profitability or full operational readiness.
