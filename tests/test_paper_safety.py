from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from environment.market import MarketEnvironment
from environment.observation import ObservationBuilder
from environment.portfolio import Portfolio
from trading.alpaca_broker import PaperBroker
from trading.config import PAPER_RUNTIME_CONFIG, validate_paper_runtime_config
from trading.paper_trader import PaperTrader


def _bars(rows: int = 64) -> pd.DataFrame:
    index = pd.date_range(
        "2024-01-08 14:30", periods=rows, freq="5min", tz="UTC", name="timestamp"
    )
    close = np.linspace(100.0, 102.0, rows)
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.linspace(1000.0, 2000.0, rows),
        },
        index=index,
    )


def test_paper_and_training_observations_match() -> None:
    trader = PaperTrader.__new__(PaperTrader)
    account = SimpleNamespace(equity=20_100.0, cash=15_000.0)
    position = SimpleNamespace(qty=50.0, avg_entry_price=98.0)
    bars = _bars()

    paper_observation = trader._observation(bars, account, position)
    clean = trader._clean_bars(bars)
    market = MarketEnvironment(clean.reset_index(), initial_cash=20_000.0)
    market.reset(len(market) - 1)
    portfolio = Portfolio(initial_cash=20_000.0, cash=15_000.0)
    portfolio.position = 50.0
    portfolio.average_entry_price = 98.0
    training_observation = ObservationBuilder(lookback=64).build(market, portfolio)

    assert paper_observation.shape == (ObservationBuilder.schema(64)["input_size"],)
    assert np.allclose(paper_observation, training_observation)


def test_paper_bar_validation_rejects_bad_ohlc_instead_of_dropping() -> None:
    bars = _bars()
    bars.iloc[10, bars.columns.get_loc("high")] = 1.0
    with pytest.raises(ValueError, match="inconsistent OHLC"):
        PaperTrader._clean_bars(bars)


def test_stale_bars_never_drive_position_rebalancing() -> None:
    trader = PaperTrader.__new__(PaperTrader)
    trader.market = SimpleNamespace(
        get_recent_bars_batched=lambda *args, **kwargs: {"AAA": _bars()}
    )
    trader.broker = SimpleNamespace(
        account=lambda: SimpleNamespace(equity=10_000.0),
        positions=lambda: [SimpleNamespace(symbol="AAA", market_value=500.0)],
        open_order_symbols=lambda: set(),
    )
    trader.symbols = ["AAA"]
    trader.batch_size = 1
    trader.min_price = 5.0
    trader.min_dollar_volume = 0.0
    trader.max_positions = 2
    trader.max_position_fraction = 0.1
    trader.max_gross_exposure = 0.2
    trader.min_signal = 0.1
    trader.allow_short = False
    trader._discover_symbols = lambda: None

    assert trader._scan() == {}


def test_paper_risk_state_survives_restart_and_corruption_fails_closed(
    tmp_path,
) -> None:
    path = tmp_path / "paper_risk_state.json"
    first = PaperTrader.__new__(PaperTrader)
    first.risk_state_path = path
    first.session_date = "2024-01-08"
    first.session_start_equity = 10_000.0
    first.daily_loss_stop_triggered = True
    first._save_risk_state()

    restarted = PaperTrader.__new__(PaperTrader)
    restarted.risk_state_path = path
    restarted.session_date = None
    restarted.session_start_equity = None
    restarted.daily_loss_stop_triggered = False
    restarted._load_risk_state()
    assert restarted.session_date == "2024-01-08"
    assert restarted.session_start_equity == 10_000.0
    assert restarted.daily_loss_stop_triggered is True

    path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to trade"):
        restarted._load_risk_state()


class _FakeBroker:
    def __init__(self):
        self.cancelled = 0
        self.closed = []
        self._positions = [SimpleNamespace(symbol="AAA")]

    def cancel_open_orders(self):
        self.cancelled += 1

    def open_order_symbols(self):
        return set()

    def positions(self):
        return self._positions

    def close_position(self, symbol):
        self.closed.append(symbol)


def test_daily_loss_stop_cancels_orders_closes_and_persists(tmp_path) -> None:
    trader = PaperTrader.__new__(PaperTrader)
    trader.risk_state_path = tmp_path / "risk.json"
    trader.session_date = pd.Timestamp.now(tz="America/New_York").date().isoformat()
    trader.session_start_equity = 10_000.0
    trader.daily_loss_stop_triggered = False
    trader.daily_loss_limit_fraction = 0.02
    trader.broker = _FakeBroker()

    assert trader._enforce_daily_loss_stop(SimpleNamespace(equity=9_800.0)) is True
    assert trader.broker.cancelled == 1
    assert trader.broker.closed == ["AAA"]
    assert trader.daily_loss_stop_triggered is True
    assert '"daily_loss_stop_triggered": true' in trader.risk_state_path.read_text()


class _FakeClient:
    def __init__(self):
        self.submitted = []

    def submit_order(self, order_data):
        self.submitted.append(order_data)
        return SimpleNamespace(id="paper-order")


def _broker() -> PaperBroker:
    broker = PaperBroker.__new__(PaperBroker)
    broker.client = _FakeClient()
    broker.assets = lambda: {
        "AAA": SimpleNamespace(tradable=True, shortable=False, fractionable=True)
    }
    broker.positions = lambda: []
    broker.open_orders = lambda: []
    broker.account = lambda: SimpleNamespace(equity=1000.0, buying_power=1000.0)
    return broker


def test_broker_enforces_exposure_and_emits_paper_idempotency_key() -> None:
    broker = _broker()
    with pytest.raises(RuntimeError, match="per-symbol exposure"):
        broker.submit_delta(
            "AAA",
            2.0,
            allow_short=False,
            client_order_id="mltrader-test-1",
            price=100.0,
            max_position_fraction=0.10,
            max_gross_exposure=0.20,
            max_positions=2,
        )
    assert not broker.client.submitted

    order = broker.submit_delta(
        "AAA",
        0.5,
        allow_short=False,
        client_order_id="mltrader-test-2",
        price=100.0,
        max_position_fraction=0.10,
        max_gross_exposure=0.20,
        max_positions=2,
    )
    assert order.id == "paper-order"
    assert broker.client.submitted[0].client_order_id == "mltrader-test-2"


def test_unresolved_order_blocks_another_submission() -> None:
    broker = _broker()
    broker.open_orders = lambda: [SimpleNamespace(symbol="AAA")]
    result = broker.submit_delta(
        "AAA",
        0.5,
        allow_short=False,
        client_order_id="mltrader-test-3",
        price=100.0,
        max_position_fraction=0.10,
        max_gross_exposure=0.20,
        max_positions=2,
    )
    assert result is None
    assert not broker.client.submitted


def test_client_order_identity_is_stable_across_restarts() -> None:
    order_id = PaperTrader._client_order_id(
        "AAA", pd.Timestamp("2024-01-02", tz="UTC"), 2.5
    )
    assert order_id == PaperTrader._client_order_id(
        "AAA", pd.Timestamp("2024-01-02", tz="UTC"), 2.5
    )
    assert order_id != PaperTrader._client_order_id(
        "AAA", pd.Timestamp("2024-01-03", tz="UTC"), 2.5
    )


def test_runtime_config_is_paper_only() -> None:
    assert PAPER_RUNTIME_CONFIG["mode"] == "paper"
    assert PAPER_RUNTIME_CONFIG["broker"] == {"provider": "alpaca", "endpoint": "paper"}
    with pytest.raises(RuntimeError, match="Only paper trading"):
        validate_paper_runtime_config(
            {
                "paper": {
                    "mode": "live",
                    "broker": {"provider": "alpaca", "endpoint": "paper"},
                    "market_data": {"timeframe": "5m", "lookback": 64},
                }
            }
        )
