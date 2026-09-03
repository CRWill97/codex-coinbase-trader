from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from coinbase_trader.models import Candle, PortfolioState, Position, ProductSnapshot, TradeAction, TradingDecision
from coinbase_trader.risk import RiskLimits, RiskManager, TradeActivity


def build_snapshot(price: str = "100") -> ProductSnapshot:
    now = datetime.now(UTC)
    candles = [
        Candle(
            start=now - timedelta(hours=index),
            low=Decimal("99"),
            high=Decimal("101"),
            open=Decimal("100"),
            close=Decimal("100"),
            volume=Decimal("10"),
        )
        for index in range(30)
    ]
    return ProductSnapshot(
        product_id="BTC-USD",
        price=Decimal(price),
        best_bid=Decimal("99.9"),
        best_ask=Decimal("100.1"),
        spread_bps=Decimal("20"),
        candles=candles,
    )


def build_portfolio() -> PortfolioState:
    return PortfolioState(
        cash=Decimal("100"),
        position=Position(product_id="BTC-USD", quantity=Decimal("0.1"), average_entry_price=Decimal("95")),
        realized_pnl=Decimal("0"),
        equity=Decimal("110"),
        peak_equity=Decimal("115"),
        daily_realized_pnl=Decimal("0"),
        last_price=Decimal("100"),
    )


def build_manager() -> RiskManager:
    return RiskManager(
        RiskLimits(
            max_notional_per_trade=Decimal("15"),
            max_position_notional=Decimal("40"),
            max_daily_loss=Decimal("10"),
            max_drawdown_pct=0.12,
            max_slippage_bps=Decimal("35"),
            min_confidence=0.62,
            fee_bps=Decimal("60"),
            min_expected_edge_bps=Decimal("20"),
            max_trades_per_day=8,
            min_minutes_between_trades=30,
            volatility_hard_limit=0.08,
        )
    )


def build_activity() -> TradeActivity:
    return TradeActivity(filled_trades_today=0, minutes_since_last_fill=999)


def test_rejects_low_confidence_trade() -> None:
    decision = TradingDecision(
        product_id="BTC-USD",
        action=TradeAction.BUY,
        confidence=0.55,
        size_quote=Decimal("10"),
        thesis="Weak setup.",
        expected_edge_bps=Decimal("200"),
    )
    allowed, reason = build_manager().validate(decision, build_portfolio(), build_snapshot(), build_activity())
    assert allowed is False
    assert "confidence" in reason.lower()


def test_rejects_buy_above_position_limit() -> None:
    portfolio = PortfolioState(
        cash=Decimal("100"),
        position=Position(product_id="BTC-USD", quantity=Decimal("0.30"), average_entry_price=Decimal("95")),
        realized_pnl=Decimal("0"),
        equity=Decimal("130"),
        peak_equity=Decimal("135"),
        daily_realized_pnl=Decimal("0"),
        last_price=Decimal("100"),
    )
    decision = TradingDecision(
        product_id="BTC-USD",
        action=TradeAction.BUY,
        confidence=0.9,
        size_quote=Decimal("15"),
        thesis="Strong setup.",
        expected_edge_bps=Decimal("200"),
    )
    allowed, reason = build_manager().validate(decision, portfolio, build_snapshot(), build_activity())
    assert allowed is False
    assert "position" in reason.lower()


def test_accepts_hold() -> None:
    decision = TradingDecision(
        product_id="BTC-USD",
        action=TradeAction.HOLD,
        confidence=0.1,
        size_quote=Decimal("0"),
        thesis="No edge.",
    )
    allowed, _ = build_manager().validate(decision, build_portfolio(), build_snapshot(), build_activity())
    assert allowed is True


def test_rejects_if_expected_edge_is_below_cost_floor() -> None:
    decision = TradingDecision(
        product_id="BTC-USD",
        action=TradeAction.BUY,
        confidence=0.9,
        size_quote=Decimal("10"),
        thesis="Potential setup.",
        expected_edge_bps=Decimal("80"),
    )
    allowed, reason = build_manager().validate(decision, build_portfolio(), build_snapshot(), build_activity())
    assert allowed is False
    assert "expected edge" in reason.lower()


def test_rejects_if_cooldown_active() -> None:
    decision = TradingDecision(
        product_id="BTC-USD",
        action=TradeAction.BUY,
        confidence=0.9,
        size_quote=Decimal("10"),
        thesis="Potential setup.",
        expected_edge_bps=Decimal("250"),
    )
    activity = TradeActivity(filled_trades_today=1, minutes_since_last_fill=5)
    allowed, reason = build_manager().validate(decision, build_portfolio(), build_snapshot(), activity)
    assert allowed is False
    assert "cooldown" in reason.lower()


def test_force_exit_sell_bypasses_entry_filters() -> None:
    decision = TradingDecision(
        product_id="BTC-USD",
        action=TradeAction.SELL,
        confidence=0.1,
        size_quote=Decimal("20"),
        thesis="Stop loss exit.",
        expected_edge_bps=Decimal("-999"),
        realized_volatility=0.5,
        force_exit=True,
    )
    snapshot = build_snapshot()
    snapshot.spread_bps = Decimal("500")
    activity = TradeActivity(filled_trades_today=8, minutes_since_last_fill=0)
    portfolio = PortfolioState(
        cash=Decimal("100"),
        position=Position(product_id="BTC-USD", quantity=Decimal("0.30"), average_entry_price=Decimal("95")),
        realized_pnl=Decimal("0"),
        equity=Decimal("130"),
        peak_equity=Decimal("135"),
        daily_realized_pnl=Decimal("-20"),
        last_price=Decimal("100"),
    )

    allowed, reason = build_manager().validate(decision, portfolio, snapshot, activity)

    assert allowed is True
    assert reason == "Decision passed risk checks."

