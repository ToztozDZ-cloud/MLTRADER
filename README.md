<<<<<<< HEAD
=======
<<<<<<< HEAD


RUN SCRIPT:
python track_training.py --data data\raw\SPY.csv --training-config config\training.yaml --market-config config\market	.yaml --generations 100
=======
>>>>>>> 7ce9a3260bf1e51b5cc30c7b002019690ed69fb8
# MLTRADER

Evolutionary trading research and Alpaca paper-trading tools.

## Train a multi-instrument agent

Prepare one CSV per instrument. Each CSV must contain columns named `open`, `high`, `low`, `close`, and `volume`, with at least 100 valid rows. Use the same bar interval for every instrument, and do not include future data in features.

Example:

```bash
python evolve.py --data data/SPY.csv data/QQQ.csv data/AAPL.csv data/MSFT.csv data/NVDA.csv
```

The evolutionary trainer evaluates each candidate on every supplied instrument and aggregates the results, using mean fitness/returns and worst-instrument drawdown. This discourages optimizing for one ticker only. Supply a broader set of historical instruments for a more representative champion. If only one file is supplied, training remains single-instrument.

## Paper trading

1. Install the project's Python dependencies, including Alpaca's trading/data SDK, NumPy, pandas, PyYAML, and python-dotenv.
2. Put Alpaca paper-account credentials in `.env`:
   `APCA_API_KEY_ID=...`
   `APCA_API_SECRET_KEY=...`
3. Train a model with 324 inputs (64 OHLCV bars plus 4 portfolio features) and confirm `models/champions/best_agent.json` exists.
4. Review `config/trading_universe.yaml`. The default scanner cap is 1,200 symbols, with up to 10 positions, a 10% per-position cap, an 80% target gross-exposure cap, a $5 minimum price, and a $5M average dollar-volume filter.
5. Start the paper-only trader:
   ```bash
   python -m trading.paper_trader
   ```

The trader discovers active/tradable US equity assets from Alpaca, filters for recent data and liquidity, ranks signals, checks pending orders, and observes configured portfolio caps. Short orders are submitted only when shorting is enabled and Alpaca reports the asset as shortable. It intentionally uses Alpaca paper mode.

## Important behavior

- The scanner only evaluates the symbols it can retrieve through the configured Alpaca market-data feed and scan cap; it does not guarantee coverage of every listed security.
- The model emits signed target exposure. A model trained only on one instrument should be retrained on multiple instruments before using its signals across a broad universe.
- The broker can reject orders because of account permissions, buying power, fractional-order support, shortability, market status, or API/data limits.
- The paper trader logs decisions and orders to `logs/paper_trades.csv`.
- This is a research system, not a promise of profitability. Paper results do not guarantee live results.
<<<<<<< HEAD
=======
>>>>>>> origin/master
>>>>>>> 7ce9a3260bf1e51b5cc30c7b002019690ed69fb8
