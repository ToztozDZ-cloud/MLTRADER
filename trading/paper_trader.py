from __future__ import annotations

import math
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from agents.serialization import AgentSerializer
from trading.alpaca_broker import PaperBroker
from trading.config import (
    DECISION_INTERVAL_SECONDS, LOG_PATH, LOOKBACK, MODEL_PATH, load_universe_config
)
from trading.market_data import MarketData
from trading.trade_logger import TradeLogger

FEATURE_COUNT = LOOKBACK * 5 + 4


class PaperTrader:
    """Broad-universe paper trader using one trained policy across many symbols.

    The model emits signed target exposure (-1 short, 0 flat, +1 long). The
    portfolio allocator applies per-name and total gross exposure limits.
    """

    def __init__(self) -> None:
        self.settings = load_universe_config()
        self.max_scan_symbols = max(1, int(self.settings.get("max_scan_symbols", 1200)))
        self.batch_size = max(1, int(self.settings.get("bars_batch_size", 100)))
        self.max_positions = max(1, int(self.settings.get("max_positions", 10)))
        self.min_dollar_volume = float(self.settings.get("min_average_dollar_volume", 5_000_000))
        self.min_price = float(self.settings.get("min_price", 5.0))
        self.min_signal = float(self.settings.get("min_signal_strength", 0.15))
        self.max_position_fraction = float(self.settings.get("max_position_fraction", 0.10))
        self.max_gross_exposure = float(self.settings.get("max_gross_exposure", 0.80))
        self.allow_short = bool(self.settings.get("allow_short", True))
        self.allow_fractional = bool(self.settings.get("allow_fractional", True))
        self.scan_interval_steps = max(1, int(self.settings.get("scan_interval_steps", 12)))

        print(f"Loading trained agent from {MODEL_PATH}...")
        self.agent = AgentSerializer.load(MODEL_PATH)
        self.agent.reset()
        if self.agent.genome.input_size != FEATURE_COUNT:
            raise ValueError(
                f"Model expects {self.agent.genome.input_size} inputs; trader builds {FEATURE_COUNT}. "
                "Retrain/export with matching lookback and ObservationBuilder."
            )
        self.market = MarketData()
        self.broker = PaperBroker()
        self.logger = TradeLogger(LOG_PATH)
        self.symbols: list[str] = []
        self.step_count = 0
        self.last_bar_time: dict[str, object] = {}
        self.last_signals: dict[str, float] = {}
        self.last_prices: dict[str, float] = {}
        self.last_frames: dict[str, pd.DataFrame] = {}
        self._discover_symbols(force=True)

    def _discover_symbols(self, force: bool = False) -> None:
        if self.symbols and not force:
            return
        assets = self.broker.assets(refresh=force)
        # Keep the scan bounded to prevent unbounded API calls on each loop.
        symbols = sorted(assets.keys())[:self.max_scan_symbols]
        if not symbols:
            raise RuntimeError("Alpaca returned no active, tradable US equities.")
        self.symbols = symbols
        print(f"Discovered {len(self.symbols)} active tradable US equity symbols (scan cap={self.max_scan_symbols}).")

    @staticmethod
    def _clean_bars(bars: pd.DataFrame) -> pd.DataFrame:
        frame = bars.copy()
        frame.columns = [str(c).lower() for c in frame.columns]
        columns = ["open", "high", "low", "close", "volume"]
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise ValueError(f"Market data missing columns: {missing}")
        frame = frame[columns].apply(pd.to_numeric, errors="coerce")
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
        if len(frame) < LOOKBACK or (frame["close"] <= 0).any():
            return pd.DataFrame()
        return frame.tail(LOOKBACK)

    def _observation(self, bars: pd.DataFrame, account, position) -> np.ndarray:
        frame = self._clean_bars(bars)
        if frame.empty:
            raise ValueError("Insufficient valid bars.")
        values = frame[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float32)
        price = float(values[-1, 3])
        normalized = values.copy()
        normalized[:, :4] = normalized[:, :4] / price - 1.0
        volume_mean = float(np.mean(normalized[:, 4]))
        normalized[:, 4] = normalized[:, 4] / volume_mean - 1.0 if volume_mean > 0 else 0.0

        equity = float(account.equity)
        cash = float(account.cash)
        quantity = float(position.qty) if position is not None else 0.0
        average_entry = float(position.avg_entry_price) if position is not None else 0.0
        if not math.isfinite(equity) or equity <= 0:
            raise RuntimeError(f"Invalid account equity: {equity}")
        portfolio_state = np.asarray([
            cash / equity,
            quantity * price / equity,
            quantity / max(1.0, abs(quantity)),
            ((price - average_entry) * quantity / equity) if quantity else 0.0,
        ], dtype=np.float32)
        observation = np.concatenate([normalized.flatten(), portfolio_state]).astype(np.float32)
        if observation.shape != (FEATURE_COUNT,) or not np.isfinite(observation).all():
            raise ValueError(f"Invalid observation vector: shape={observation.shape}")
        return observation

    def _scan(self) -> dict[str, tuple[float, float, pd.DataFrame]]:
        self._discover_symbols()
        bars_by_symbol = self.market.get_recent_bars_batched(
            self.symbols, limit=LOOKBACK, batch_size=self.batch_size
        )
        if not bars_by_symbol:
            raise RuntimeError("No recent bars received for the configured universe.")

        account = self.broker.account()
        positions = {p.symbol: p for p in self.broker.positions()}
        candidates = []
        for symbol, raw_frame in bars_by_symbol.items():
            frame = self._clean_bars(raw_frame)
            if frame.empty:
                continue
            price = float(frame["close"].iloc[-1])
            if price < self.min_price:
                continue
            # Use recent mean dollar volume as a practical liquidity filter.
            dollar_volume = float((frame["close"] * frame["volume"]).mean())
            if not math.isfinite(dollar_volume) or dollar_volume < self.min_dollar_volume:
                continue
            position = positions.get(symbol)
            try:
                observation = self._observation(frame, account, position)
                output = float(np.asarray(self.agent.act(observation)).reshape(-1)[0])
                if not math.isfinite(output):
                    continue
                signal = float(np.clip(output, -1.0, 1.0))
                candidates.append((symbol, signal, price, dollar_volume, frame))
            except Exception as exc:
                print(f"SKIP {symbol}: could not evaluate observation: {exc}")

        # Rank by signal magnitude after filtering. Existing positions remain in
        # the candidate list only if fresh bars and valid signals are available.
        candidates.sort(key=lambda item: abs(item[1]), reverse=True)
        selected = [item for item in candidates if abs(item[1]) >= self.min_signal][:self.max_positions]
        equity = float(account.equity)
        # Selected positions receive a target allocation under the total gross
        # exposure budget. Existing positions not selected below are targeted to
        # zero, so their exposure does not consume this new target budget.
        remaining_gross = self.max_gross_exposure
        desired: dict[str, tuple[float, float, pd.DataFrame, float]] = {}
        for symbol, signal, price, _dollar_volume, frame in selected:
            if signal < 0 and not self.allow_short:
                target = 0.0
            else:
                target = math.copysign(min(abs(signal), self.max_position_fraction), signal)
            allocation = min(abs(target), remaining_gross)
            remaining_gross -= allocation
            desired[symbol] = (
                math.copysign(allocation, target) if allocation else 0.0,
                price, frame, signal,
            )

        # Flatten existing positions which are no longer selected. Avoid changing
        # any position for which fresh data was unavailable; that avoids blind orders.
        for symbol, position in positions.items():
            if symbol not in desired and symbol in bars_by_symbol:
                frame = self._clean_bars(bars_by_symbol[symbol])
                if not frame.empty:
                    desired[symbol] = (0.0, float(frame["close"].iloc[-1]), frame, 0.0)
        return desired

    def step(self) -> None:
        self.step_count += 1
        if self.step_count % self.scan_interval_steps == 1:
            self._discover_symbols(force=True)
        desired = self._scan()
        if not desired:
            print("No qualifying liquid symbols/signals this scan.")
            return

        account = self.broker.account()
        positions = {p.symbol: p for p in self.broker.positions()}
        orders_submitted = 0
        for symbol, (target_fraction, price, frame, signal) in desired.items():
            if symbol in self.last_bar_time and self.last_bar_time[symbol] == frame.index[-1]:
                continue
            self.last_bar_time[symbol] = frame.index[-1]
            self.last_prices[symbol] = price
            self.last_signals[symbol] = signal
            position = positions.get(symbol)
            current_qty = float(position.qty) if position is not None else 0.0
            equity = float(account.equity)
            target_qty = equity * target_fraction / price
            delta = target_qty - current_qty
            # Limit the agent's orders to symbols that Alpaca reports as tradable.
            try:
                order = self.broker.submit_delta(symbol, delta, allow_short=self.allow_short)
            except Exception as exc:
                print(f"ORDER FAILED {symbol}: {exc}")
                continue
            if order is not None:
                orders_submitted += 1
                print(f"ORDER {symbol}: target={target_fraction:+.1%}, qty_delta={delta:+.4f}, id={order.id}")
            updated_position = self.broker.position(symbol)
            updated_qty = float(updated_position.qty) if updated_position is not None else 0.0
            updated_account = self.broker.account()
            self.logger.log(
                symbol=symbol, price=price, agent_output=signal,
                target_position=target_fraction, current_position=updated_qty,
                equity=float(updated_account.equity), order_id=str(order.id) if order else None,
            )
        print(f"Scan complete: {len(desired)} target symbols; {orders_submitted} orders submitted.")

    def run(self) -> None:
        print("=" * 60)
        print("MLTRADER MULTI-ASSET PAPER TRADER")
        print("=" * 60)
        print(f"Dynamic universe cap: {self.max_scan_symbols} symbols")
        print(f"Max positions: {self.max_positions} | Per-name cap: {self.max_position_fraction:.1%}")
        print(f"Gross exposure cap: {self.max_gross_exposure:.1%} | Paper mode only")
        while True:
            try:
                self.step()
            except KeyboardInterrupt:
                print("\nPaper trader stopped.")
                break
            except Exception as exc:
                print(f"\nSCAN ERROR: {type(exc).__name__}: {exc}")
            time.sleep(DECISION_INTERVAL_SECONDS)


if __name__ == "__main__":
    PaperTrader().run()
