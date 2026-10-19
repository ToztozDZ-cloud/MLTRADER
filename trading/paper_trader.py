from __future__ import annotations

import json
import hashlib
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from agents.serialization import AgentSerializer
from environment.data import validate_ohlcv
from environment.observation import ObservationBuilder
from trading.alpaca_broker import PaperBroker
from trading.config import (
    DECISION_INTERVAL_SECONDS, LOG_PATH, LOOKBACK, MODEL_PATH, TIMEFRAME_MINUTES,
    load_universe_config,
)
from trading.market_data import MarketData
from trading.trade_logger import TradeLogger

FEATURE_COUNT = ObservationBuilder.input_size(LOOKBACK)


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
        self.scan_interval_steps = max(1, int(self.settings.get("scan_interval_steps", 1)))
        self.flatten_before_close_minutes = max(
            1, int(self.settings.get("flatten_before_close_minutes", 10))
        )
        self.daily_loss_limit_fraction = float(
            self.settings.get("daily_loss_limit_fraction", 0.02)
        )
        if not 0.0 < self.daily_loss_limit_fraction < 1.0:
            raise ValueError("daily_loss_limit_fraction must be between 0 and 1.")
        self.session_date = None
        self.session_start_equity: float | None = None
        self.daily_loss_stop_triggered = False
        self.risk_state_path = Path(LOG_PATH).with_name("paper_risk_state.json")
        self._load_risk_state()

        print(f"Loading trained agent from {MODEL_PATH}...")
        self.agent = AgentSerializer.load(
            MODEL_PATH,
            expected_observation=ObservationBuilder.schema(LOOKBACK),
            expected_timeframe=f"{TIMEFRAME_MINUTES}m",
            require_metadata=True,
        )
        self.agent.reset()
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
        if not bool(self.settings.get("dynamic_asset_discovery", False)):
            requested = self.settings.get("symbols", ["SPY", "QQQ", "AAPL", "MSFT"])
            if not isinstance(requested, list) or not requested:
                raise ValueError("Static universe must contain a non-empty symbols list.")
            unavailable = [symbol for symbol in requested if symbol not in assets]
            if unavailable:
                raise RuntimeError(
                    "Configured symbols are not active/tradable in Alpaca: "
                    + ", ".join(unavailable)
                )
            symbols = list(dict.fromkeys(requested))[:self.max_scan_symbols]
        else:
            priority = [
                "SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "AMZN",
                "GOOGL", "META", "TSLA", "AVGO", "AMD", "BRK.B", "JPM", "LLY",
            ]
            all_symbols = sorted(assets.keys())
            chosen = [symbol for symbol in priority if symbol in assets]
            remaining = [symbol for symbol in all_symbols if symbol not in set(chosen)]
            slots = max(0, self.max_scan_symbols - len(chosen))
            if slots and len(remaining) > slots:
                indices: np.ndarray = np.linspace(0, len(remaining) - 1, num=slots, dtype=int)
                chosen.extend(remaining[index] for index in indices)
            else:
                chosen.extend(remaining[:slots])
            symbols = list(dict.fromkeys(chosen))[:self.max_scan_symbols]
        if not symbols:
            raise RuntimeError("Alpaca returned no active, tradable US equities.")
        self.symbols = symbols
        print(f"Discovered {len(self.symbols)} active tradable US equity symbols (scan cap={self.max_scan_symbols}).")

    @staticmethod
    def _clean_bars(bars: pd.DataFrame) -> pd.DataFrame:
        if bars.empty:
            return pd.DataFrame()
        checked = validate_ohlcv(
            bars,
            source="paper market data",
            timezone="America/New_York",
            regular_session_only=True,
            timeframe=f"{TIMEFRAME_MINUTES}m",
            require_timestamp=True,
        )
        if len(checked) < LOOKBACK:
            return pd.DataFrame()
        return checked.tail(LOOKBACK).set_index("timestamp")

    @staticmethod
    def _bars_are_fresh(bars: pd.DataFrame, max_age_minutes: int = 10) -> bool:
        if bars.empty:
            return False
        latest = pd.Timestamp(bars.index[-1])
        if latest.tzinfo is None:
            latest = latest.tz_localize("UTC")
        else:
            latest = latest.tz_convert("UTC")
        age_seconds = (pd.Timestamp.now(tz="UTC") - latest).total_seconds()
        return -60.0 <= age_seconds <= max_age_minutes * 60.0

    @staticmethod
    def _client_order_id(symbol: str, bar_time, target_quantity: float) -> str:
        identity = f"{symbol}|{pd.Timestamp(bar_time).isoformat()}|{float(target_quantity):.8f}"
        return "mltrader-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]

    def _observation(self, bars: pd.DataFrame, account, position) -> np.ndarray:
        frame = self._clean_bars(bars)
        if frame.empty:
            raise ValueError("Insufficient valid bars.")
        price = float(frame["close"].iloc[-1])
        equity = float(account.equity)
        cash = float(account.cash)
        quantity = float(position.qty) if position is not None else 0.0
        average_entry = float(position.avg_entry_price) if position is not None else 0.0
        if not math.isfinite(equity) or equity <= 0:
            raise RuntimeError(f"Invalid account equity: {equity}")
        portfolio_state: np.ndarray = np.asarray([
            cash / equity,
            quantity * price / equity,
            quantity / max(1.0, abs(quantity)),
            ((price - average_entry) * quantity / equity) if quantity else 0.0,
        ], dtype=np.float32)
        observation = ObservationBuilder.build_from_history(
            frame.reset_index(), portfolio_state, LOOKBACK
        )
        if observation.shape != (FEATURE_COUNT,) or not np.isfinite(observation).all():
            raise ValueError(f"Invalid observation vector: shape={observation.shape}")
        return observation

    def _scan(self) -> dict[str, tuple[float, float, pd.DataFrame, float]]:
        self._discover_symbols()
        bars_by_symbol = self.market.get_recent_bars_batched(
            self.symbols, limit=LOOKBACK, batch_size=self.batch_size
        )
        if not bars_by_symbol:
            raise RuntimeError("No recent bars received for the configured universe.")

        account = self.broker.account()
        positions = {p.symbol: p for p in self.broker.positions()}
        pending_symbols = self.broker.open_order_symbols()
        candidates = []
        for symbol, raw_frame in bars_by_symbol.items():
            if symbol in pending_symbols:
                continue
            if not self._bars_are_fresh(raw_frame):
                print(f"SKIP {symbol}: latest completed bar is stale.")
                continue
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
                self.agent.reset()
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
        reserved_gross = sum(
            abs(float(position.market_value)) / max(equity, 1.0)
            for symbol, position in positions.items()
            if (
                symbol not in bars_by_symbol
                or symbol in pending_symbols
                or not self._bars_are_fresh(bars_by_symbol[symbol])
            )
        )
        # Reserve one per-name allocation for pending orders that do not yet
        # have a position, since a market order may fill after this scan.
        reserved_gross += sum(
            self.max_position_fraction
            for symbol in pending_symbols
            if symbol not in positions
        )
        remaining_gross = max(0.0, self.max_gross_exposure - reserved_gross)
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
            if symbol not in desired and symbol in bars_by_symbol and symbol not in pending_symbols:
                raw_frame = bars_by_symbol[symbol]
                if not self._bars_are_fresh(raw_frame):
                    continue
                frame = self._clean_bars(raw_frame)
                if not frame.empty:
                    desired[symbol] = (0.0, float(frame["close"].iloc[-1]), frame, 0.0)
        return desired

    def _load_risk_state(self) -> None:
        try:
            state = json.loads(self.risk_state_path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                raise ValueError("risk state must be a JSON object")
            required = {"session_date", "session_start_equity", "daily_loss_stop_triggered"}
            if not required.issubset(state):
                raise ValueError("risk state is missing required fields")
            self.session_date = state.get("session_date")
            baseline = state.get("session_start_equity")
            self.session_start_equity = float(baseline) if baseline is not None else None
            self.daily_loss_stop_triggered = bool(state.get("daily_loss_stop_triggered", False))
            if self.session_date is not None:
                if not isinstance(self.session_date, str):
                    raise ValueError("session_date must be a date string")
                pd.Timestamp(self.session_date).date()
            elif self.session_start_equity is not None or self.daily_loss_stop_triggered:
                raise ValueError("risk baseline or stop cannot exist without a session date")
            if self.session_start_equity is not None and (
                not math.isfinite(self.session_start_equity) or self.session_start_equity <= 0
            ):
                raise ValueError("session_start_equity must be finite and positive")
            if not isinstance(state.get("daily_loss_stop_triggered", False), bool):
                raise ValueError("daily_loss_stop_triggered must be boolean")
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not load paper daily risk state; refusing to trade: {exc}") from exc

    def _save_risk_state(self) -> None:
        self.risk_state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "session_date": self.session_date,
            "session_start_equity": self.session_start_equity,
            "daily_loss_stop_triggered": self.daily_loss_stop_triggered,
        }
        temporary = self.risk_state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(self.risk_state_path)

    def _enforce_daily_loss_stop(self, account) -> bool:
        local_now = pd.Timestamp.now(tz="America/New_York")
        today = local_now.date().isoformat()
        account_equity = float(account.equity)
        if not math.isfinite(account_equity) or account_equity <= 0.0:
            raise RuntimeError("Broker equity is invalid; refusing paper trading")
        if self.session_date != today:
            self.session_date = today
            self.session_start_equity = account_equity
            self.daily_loss_stop_triggered = False
            self._save_risk_state()
            print(f"Daily risk baseline reset: equity=${self.session_start_equity:,.2f}")
        if self.daily_loss_stop_triggered:
            # A prior process may have stopped before flatten orders filled.
            self.broker.cancel_open_orders()
            pending = self.broker.open_order_symbols()
            for position in self.broker.positions():
                try:
                    if position.symbol not in pending:
                        self.broker.close_position(position.symbol)
                except Exception as exc:
                    print(f"LOSS-STOP RETRY FAILED {position.symbol}: {exc}")
            return True
        baseline = self.session_start_equity or float(account.equity)
        equity = account_equity
        if baseline > 0 and equity <= baseline * (1.0 - self.daily_loss_limit_fraction):
            self.daily_loss_stop_triggered = True
            self._save_risk_state()
            print(
                f"DAILY LOSS STOP: equity=${equity:,.2f} reached "
                f"{self.daily_loss_limit_fraction:.1%} loss limit; flattening and "
                "blocking further entries for today."
            )
            self.broker.cancel_open_orders()
            pending = self.broker.open_order_symbols()
            for position in self.broker.positions():
                try:
                    if position.symbol not in pending:
                        self.broker.close_position(position.symbol)
                except Exception as exc:
                    print(f"LOSS-STOP CLOSE FAILED {position.symbol}: {exc}")
            return True
        return False

    def step(self) -> None:
        clock = self.broker.clock()
        if not clock.is_open:
            print("Market is closed; waiting for the next regular session.")
            return

        if self._enforce_daily_loss_stop(self.broker.account()):
            return

        # Intraday policy: do not open new risk close to the regular-session
        # close. Flatten existing positions before the session ends.
        next_close = clock.next_close
        now = datetime.now(timezone.utc)
        if next_close.tzinfo is None:
            next_close = next_close.replace(tzinfo=timezone.utc)
        minutes_to_close = (next_close - now).total_seconds() / 60.0
        if minutes_to_close <= self.flatten_before_close_minutes:
            print(
                f"Regular-session close in {max(0.0, minutes_to_close):.1f} min; "
                "flattening paper positions."
            )
            self.broker.cancel_open_orders()
            pending = self.broker.open_order_symbols()
            for position in self.broker.positions():
                try:
                    if position.symbol not in pending:
                        order = self.broker.close_position(position.symbol)
                        if order is not None:
                            print(f"CLOSE {position.symbol}: order_id={order.id}")
                except Exception as exc:
                    print(f"CLOSE FAILED {position.symbol}: {exc}")
            return
        self.step_count += 1
        if self.step_count % self.scan_interval_steps == 1:
            self._discover_symbols(force=True)
        desired = self._scan()
        if not desired:
            print("No qualifying liquid symbols/signals this scan.")
            return

        account = self.broker.account()
        positions = {p.symbol: p for p in self.broker.positions()}
        pending_symbols = self.broker.open_order_symbols()
        orders_submitted = 0
        log_symbols: list[tuple[str, float, float, float, str | None]] = []

        for symbol, (target_fraction, price, frame, signal) in desired.items():
            if symbol in pending_symbols:
                print(f"SKIP {symbol}: an order is already open.")
                continue
            if symbol in self.last_bar_time and self.last_bar_time[symbol] == frame.index[-1]:
                continue

            position = positions.get(symbol)
            current_qty = float(position.qty) if position is not None else 0.0
            equity = float(account.equity)
            target_qty = equity * target_fraction / price
            delta = target_qty - current_qty
            client_order_id = self._client_order_id(symbol, frame.index[-1], target_qty)
            try:
                reference_prices = dict(self.last_prices)
                reference_prices.update({name: values[1] for name, values in desired.items()})
                order = self.broker.submit_delta(
                    symbol,
                    delta,
                    allow_short=self.allow_short,
                    client_order_id=client_order_id,
                    price=price,
                    max_position_fraction=self.max_position_fraction,
                    max_gross_exposure=self.max_gross_exposure,
                    max_positions=self.max_positions,
                    reference_prices=reference_prices,
                )
            except Exception as exc:
                print(f"ORDER FAILED {symbol}: {exc}")
                continue

            self.last_bar_time[symbol] = frame.index[-1]
            self.last_prices[symbol] = price
            self.last_signals[symbol] = signal
            if order is not None:
                orders_submitted += 1
                print(f"ORDER {symbol}: target={target_fraction:+.1%}, qty_delta={delta:+.4f}, id={order.id}")
            log_symbols.append(
                (symbol, price, signal, target_fraction, str(order.id) if order else None)
            )

        # Refresh positions/account once after the order batch rather than once
        # per ticker; this reduces API traffic and logs a consistent snapshot.
        final_positions = {p.symbol: p for p in self.broker.positions()}
        final_account = self.broker.account()
        for symbol, price, signal, target_fraction, order_id in log_symbols:
            final_position = final_positions.get(symbol)
            final_qty = float(final_position.qty) if final_position is not None else 0.0
            self.logger.log(
                symbol=symbol,
                price=price,
                agent_output=signal,
                target_position=target_fraction,
                current_position=final_qty,
                equity=float(final_account.equity),
                order_id=order_id,
            )
        print(f"Scan complete: {len(desired)} target symbols; {orders_submitted} orders submitted.")

    def run(self) -> None:
        print("=" * 60)
        print("MLTRADER MULTI-ASSET PAPER TRADER")
        print("=" * 60)
        print(f"Universe: {self.symbols} | Paper mode only")
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
