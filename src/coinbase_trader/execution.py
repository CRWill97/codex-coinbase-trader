from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any, Protocol

from coinbase.rest import RESTClient

from coinbase_trader.models import ExecutionReport, PortfolioState, ProductSnapshot, TradeAction, TradingDecision
from coinbase_trader.state import StateStore


class Broker(Protocol):
    def execute(
        self,
        decision: TradingDecision,
        portfolio: PortfolioState,
        snapshot: ProductSnapshot,
    ) -> ExecutionReport:
        ...


class PaperBroker:
    def __init__(self, store: StateStore, fee_bps: Decimal) -> None:
        self._store = store
        self._fee_bps = fee_bps

    def execute(
        self,
        decision: TradingDecision,
        portfolio: PortfolioState,
        snapshot: ProductSnapshot,
    ) -> ExecutionReport:
        if decision.action == TradeAction.HOLD:
            return ExecutionReport(product_id=decision.product_id, status="noop", side=decision.action, reason="No trade requested.")

        if decision.action == TradeAction.BUY:
            average_price = snapshot.best_ask
            gross_quote = min(decision.size_quote, portfolio.cash).quantize(Decimal("0.01"))
            fee_quote = _fee_amount(gross_quote, self._fee_bps)
            filled_quote = (gross_quote + fee_quote).quantize(Decimal("0.01"))
            filled_base = (gross_quote / average_price).quantize(Decimal("0.00000001"))
        else:
            average_price = snapshot.best_bid
            max_quote = portfolio.position.quantity * average_price
            gross_quote = min(decision.size_quote, max_quote).quantize(Decimal("0.01"))
            fee_quote = _fee_amount(gross_quote, self._fee_bps)
            filled_quote = (gross_quote - fee_quote).quantize(Decimal("0.01"))
            filled_base = (gross_quote / average_price).quantize(Decimal("0.00000001"))

        report = ExecutionReport(
            product_id=decision.product_id,
            status="filled",
            side=decision.action,
            filled_quote=filled_quote,
            filled_base=filled_base,
            average_price=average_price,
            reason=f"Paper execution completed with fee {self._fee_bps} bps.",
            external_order_id=f"paper-{uuid.uuid4()}",
        )
        self._store.record_execution(decision, report)
        return report


class LiveBroker:
    def __init__(self, api_key: str, api_secret: str, use_sandbox: bool) -> None:
        if use_sandbox:
            self._client = RESTClient(
                api_key=api_key,
                api_secret=api_secret,
                base_url="https://api-public.sandbox.exchange.coinbase.com",
            )
        else:
            # Let the SDK default to Coinbase production host.
            self._client = RESTClient(api_key=api_key, api_secret=api_secret)

    def execute(
        self,
        decision: TradingDecision,
        portfolio: PortfolioState,
        snapshot: ProductSnapshot,
    ) -> ExecutionReport:
        if decision.action == TradeAction.HOLD:
            return ExecutionReport(product_id=decision.product_id, status="noop", side=decision.action, reason="No trade requested.")

        client_order_id = str(uuid.uuid4())
        if decision.action == TradeAction.BUY:
            response = self._client.market_order_buy(
                client_order_id=client_order_id,
                product_id=decision.product_id,
                quote_size=str(decision.size_quote),
            )
        else:
            base_size = (decision.size_quote / snapshot.best_bid).quantize(Decimal("0.00000001"))
            response = self._client.market_order_sell(
                client_order_id=client_order_id,
                product_id=decision.product_id,
                base_size=str(base_size),
            )

        success = bool(_response_get(response, "success", False))
        if not success:
            reason = _extract_error_reason(response)
            return ExecutionReport(product_id=decision.product_id, status="rejected", side=decision.action, reason=str(reason))

        order_id = _extract_order_id(response)
        return ExecutionReport(
            product_id=decision.product_id,
            status="filled",
            side=decision.action,
            filled_quote=decision.size_quote,
            filled_base=(decision.size_quote / snapshot.price).quantize(Decimal("0.00000001")),
            average_price=snapshot.price,
            reason="Live order submitted to Coinbase.",
            external_order_id=order_id,
        )


def _fee_amount(notional_quote: Decimal, fee_bps: Decimal) -> Decimal:
    return (notional_quote * fee_bps / Decimal("10000")).quantize(Decimal("0.01"))


def _response_get(response: Any, key: str, default: Any = None) -> Any:
    if isinstance(response, dict):
        return response.get(key, default)
    if hasattr(response, key):
        return getattr(response, key)
    if hasattr(response, "to_dict"):
        try:
            payload = response.to_dict()
            if isinstance(payload, dict):
                return payload.get(key, default)
        except Exception:  # noqa: BLE001
            pass
    return default


def _extract_order_id(response: Any) -> str | None:
    success_response = _response_get(response, "success_response")
    if success_response is None:
        return _response_get(response, "order_id")
    if isinstance(success_response, dict):
        return success_response.get("order_id")
    if hasattr(success_response, "order_id"):
        return getattr(success_response, "order_id")
    return None


def _extract_error_reason(response: Any) -> str:
    error_response = _response_get(response, "error_response")
    if isinstance(error_response, dict):
        return str(
            error_response.get("message")
            or error_response.get("error_details")
            or error_response.get("error")
            or "Coinbase order rejected."
        )
    if error_response is not None:
        message = getattr(error_response, "message", None)
        details = getattr(error_response, "error_details", None)
        error = getattr(error_response, "error", None)
        if message or details or error:
            return str(message or details or error)
    failure_reason = _response_get(response, "failure_reason")
    if failure_reason is not None:
        return str(failure_reason)
    return "Coinbase order rejected."

