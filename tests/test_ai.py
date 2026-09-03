from __future__ import annotations

from decimal import Decimal

from coinbase_trader.models import TradeAction, TradingDecision


def test_trading_decision_quantizes_size() -> None:
    decision = TradingDecision(
        product_id="BTC-USD",
        action=TradeAction.BUY,
        confidence=0.9,
        size_quote=Decimal("12.345"),
        thesis="Momentum and spread are acceptable.",
    )
    assert decision.size_quote == Decimal("12.34")


def test_trading_decision_rejects_invalid_json() -> None:
    try:
        TradingDecision.model_validate_json('{"product_id":"BTC-USD","action":"buy","confidence":2.0,"size_quote":"10","thesis":"bad"}')
    except Exception as exc:  # noqa: BLE001
        assert "confidence" in str(exc).lower()
    else:
        raise AssertionError("Expected validation failure.")

