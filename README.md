# MLTRADER

Evolutionary trading research, chronological validation, realistic OHLCV backtesting, and Alpaca paper-trading tools.

> Research software only. Historical results do not establish future performance or profitability.

## Requirements

- Python 3.11 or newer
- Historical OHLCV CSV data
- For paper trading only, Alpaca paper-account credentials

Create an environment and install the declared dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` only for paper trading. Never commit credentials.

## Market data

CSV columns are case-insensitive and must include `open`, `high`, `low`, `close`, and `volume`. Timestamp columns may be named `timestamp`, `datetime`, `date`, or `time`. Timestamped naive values are interpreted in the configured exchange timezone; ambiguous or nonexistent daylight-saving timestamps fail with a row-specific error. A file must contain one instrument. If no `symbol` column is supplied, the CSV filename supplies its identity.

The shared loader sorts rows, rejects duplicate timestamps, non-finite or non-positive prices, inconsistent OHLC values, and negative volume. It does not invent candles or silently drop malformed rows. Intraday loaders apply the configured regular-session filter and detect irregular intervals within each session. Daily data is not filtered by intraday session hours. Keep raw data out of Git.

The default market configuration uses five-minute US equity bars. Daily files require an explicit daily timeframe configuration. All instruments in one training run must use compatible timeframes and observation schemas.

Download completed five-minute bars from Alpaca's historical data API:

```bash
python download_market_data.py --symbols SPY QQQ --start 2024-01-01 --output-dir data/raw
```

The downloader validates each symbol through the shared OHLCV pipeline, keeps regular-session bars, and excludes the currently forming candle. It downloads market data only; it does not submit orders.

## Training

Train on one or more instrument files:

```bash
python evolve.py --data data/raw/SPY_5m.csv data/raw/QQQ_5m.csv
```

`config/training.yaml` sets the reproducibility seed and evaluation episode options. The default evaluates each candidate on eight windows per generation, randomizes start positions and instruments, and samples trailing-volatility regimes. `episode_bars: 0` chooses half of the shortest dataset's available post-lookback bars, leaving valid start choices. Every generation reuses the same sampled windows across candidates for a fair comparison; the next generation receives a new deterministic sample. Aggregate fitness blends the episode mean with the lower-quartile mean to penalize fragile candidates.

Episode selections and per-episode results append to `models/archives/training_episodes.jsonl`. Champion and checkpoint artifacts include observation schema, seed, config hashes, instrument identifiers, and source-file hashes. Use `--seed`, `--generations`, `--checkpoint-dir`, and `--champion-dir` for explicit overrides. The current reference evaluator is serial; `training.parallel.enabled: true` is rejected instead of being silently ignored.

Signals use completed bar data and execute at the next bar's open with half-spread, adverse slippage, and commission applied once. `execute_on_next_bar: false` is rejected because same-bar execution is not modeled.

## Backtesting and validation

Run a cost-aware backtest and same-period baselines:

```bash
python run_backtest.py --data data/holdout/SPY_5m.csv --model models/champions/best_agent.json
```

The report compares cash, buy-and-hold, a transparent moving-average policy, and the model using identical bars, execution timing, and costs. It reports net return, annualized return and volatility, Sharpe, drawdown, turnover, fills, modeled transaction costs, and gross exposure.

Evaluate only the final chronological partition of an untouched dataset:

```bash
python validate.py --mode holdout --data data/untouched/SPY_5m.csv --model models/champions/best_agent.json
```

The default split is 70% train, 15% validation, and 15% test, with a lookback-sized embargo between periods. Only the final test segment is scored; preceding bars are available only as lookback history. The CLI refuses an exact file hash recorded in the model's training provenance. Repeatedly inspecting a holdout and changing the strategy can still overfit it; the code cannot detect analyst-driven reuse.

Run genuine retraining walk-forward validation:

```bash
python validate.py --mode walk-forward --data data/raw/SPY_5m.csv \
  --training-config config/smoke_training.yaml --train-size 600 \
  --validation-size 100 --step-size 100 --window-mode rolling \
  --embargo-bars 64 --generations 1
```

Each fold trains a fresh population using only its prior training window, freezes that model, warms up observations from preceding bars, and scores the following validation window. Choose `--window-mode expanding` for growing training history. Fold seeds, training fitness, model fingerprints, validation metrics, boundaries, and common-period baselines are reported. Validation results never enter the fold's training objective.

The holdout command assumes the supplied model was selected without the final test segment. A positive historical return is not evidence of future profitability.

## Paper trading

`config/live.yaml` is retained at its historical path but configures paper trading only. It must specify Alpaca's paper endpoint; the runtime rejects other modes/endpoints and always constructs the broker client with `paper=True`. There is no live-trading switch.

1. Configure Alpaca paper credentials in `.env`.
2. Review `config/trading_universe.yaml` for symbols, 5% per-name and 15% gross exposure caps, short-sale policy, and the 2% daily loss stop.
3. Train a compatible 64-bar, five-minute model and start the trader:

   ```bash
   python -m trading.paper_trader
   ```

The trader validates completed, fresh bars through the shared OHLCV pipeline and the same observation builder used in training. Model artifacts must match the observation schema and signed target-position output. Before submitting, the broker wrapper refreshes account, position, and open-order state and checks per-name exposure, gross exposure, position count, short restrictions, and buying power. Stable client order IDs make retries idempotent; unresolved orders block conflicting submissions. Broker-confirmed positions are refreshed after order batches.

The daily loss stop persists its baseline and trigger under `logs/paper_risk_state.json`. Corrupt risk state stops startup. A triggered stop blocks new entries, requests cancellation of open orders, and retries closing positions without conflicting with unresolved orders. Cancellation and liquidation can race fills or fail at the broker; the stop is not protection against gaps or losses.

## Checks

```bash
python -m compileall -q agents backtest environment evolution trading training validation tests
python -m pytest --import-mode=importlib -q
```

Tests use synthetic data and mocked broker objects. They do not require credentials and never submit orders. These checks do not establish profitability or operational readiness for unattended deployment.
