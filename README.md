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

Each CSV must contain `open`, `high`, `low`, `close`, and `volume` columns and at least 100 valid rows. Intraday CSVs must also contain a `timestamp`, `datetime`, `date`, or `time` column. Column names are case-insensitive. Training preserves timestamps, sorts and deduplicates bars, filters timestamped intraday bars to 09:30–16:00 America/New_York, and rejects a bar interval that does not match `config/market.yaml`.

The default configuration now targets five-minute US equity bars (`5m`). Daily OHLCV files such as Yahoo Finance daily exports are not compatible with that configuration. Use timestamped five-minute historical stock bars from a provider such as Alpaca, and use the same bar interval for every instrument. Keep local/raw market data out of Git.

## Train an evolutionary agent

Train on one or more instruments:

```bash
python evolve.py --data data/raw/SPY_5m.csv data/raw/QQQ_5m.csv data/raw/AAPL_5m.csv --reset-champion
```

To launch the training tracker with live fitness/equity plots on a desktop that supports Tk:

```bash
python track_training.py --data data/raw/SPY_5m.csv
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

1. Create a local `.env` from `.env.example` and use Alpaca **paper** credentials only.
2. Train a new model using timestamped five-minute data and `--reset-champion`; old champions may have been trained with different observation/execution semantics.
3. The starter universe is fixed to SPY, QQQ, AAPL, and MSFT; shorting is disabled, per-name exposure is capped at 5%, gross exposure at 15%, and a 2% daily loss stop is configured. Review `config/trading_universe.yaml` before starting.
4. Start the paper trader:

   ```bash
   python -m trading.paper_trader
   ```

The trader checks the regular market session, ignores incomplete or stale five-minute bars, blocks new entries near the close, attempts to flatten before the session ends, and applies a daily loss stop. It uses Alpaca's paper endpoint. The CPU reference evaluator is used during evolution until GPU/CPU semantic parity is proven; training may therefore be slower. Data availability, account permissions, buying power, order support, and broker rules can still prevent orders from being accepted. Do not switch this research project to live trading without a separate, deliberate safety review.

## Tests and code checks

```bash
python -m compileall -q agents backtest environment evolution trading training validation tests
python -m pytest -q
```

The test suite is a smoke check, not proof of profitability or full operational readiness.
