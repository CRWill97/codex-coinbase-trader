from __future__ import annotations

import json
import logging
from decimal import Decimal
from typing import Protocol

from openai import OpenAI

from coinbase_trader.feature_engineering import FeaturePack, build_feature_pack
from coinbase_trader.indicators import realized_volatility, simple_moving_average
from coinbase_trader.models import PortfolioState, ProductSnapshot, TradingDecision
from coinbase_trader.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class DecisionEngine(Protocol):
    def decide(self, snapshot: ProductSnapshot, portfolio: PortfolioState, risk_context: dict[str, str]) -> TradingDecision:
        ...


class OpenAIDecisionEngine:
    def __init__(
        self,
        api_key: str,
        model: str,
        max_notional_per_trade: Decimal,
        volatility_target: float,
        feature_correlation_threshold: float,
        aggressive_mode: bool,
        aggressive_size_multiplier: float,
        aggressive_hold_to_trade_edge_bps: Decimal,
    ) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._max_notional_per_trade = max_notional_per_trade
        self._volatility_target = volatility_target
        self._feature_correlation_threshold = feature_correlation_threshold
        self._aggressive_mode = aggressive_mode
        self._aggressive_size_multiplier = aggressive_size_multiplier
        self._aggressive_hold_to_trade_edge_bps = aggressive_hold_to_trade_edge_bps

    def decide(self, snapshot: ProductSnapshot, portfolio: PortfolioState, risk_context: dict[str, str]) -> TradingDecision:
        closes = [candle.close for candle in snapshot.candles]
        short_ma = simple_moving_average(closes, window=6)
        long_ma = simple_moving_average(closes, window=24)
        vol = realized_volatility(snapshot.candles)
        feature_pack = build_feature_pack(
            snapshot.candles,
            correlation_threshold=self._feature_correlation_threshold,
        )

        user_payload = {
            "product_id": snapshot.product_id,
            "price": str(snapshot.price),
            "best_bid": str(snapshot.best_bid),
            "best_ask": str(snapshot.best_ask),
            "spread_bps": str(snapshot.spread_bps.quantize(Decimal("0.01"))),
            "short_ma": str(short_ma.quantize(Decimal("0.01"))),
            "long_ma": str(long_ma.quantize(Decimal("0.01"))),
            "realized_volatility": str(vol.quantize(Decimal("0.0001"))),
            "portfolio": {
                "cash": str(portfolio.cash),
                "position_qty": str(portfolio.position.quantity),
                "average_entry_price": str(portfolio.position.average_entry_price),
                "equity": str(portfolio.equity),
                "daily_realized_pnl": str(portfolio.daily_realized_pnl),
            },
            "risk_limits": risk_context,
            "max_notional_per_trade": str(self._max_notional_per_trade),
            "selected_features": feature_pack.selected_features,
            "feature_values": {key: str(value) for key, value in feature_pack.scalar_features.items()},
        }
        response = self._client.responses.create(
            model=self._model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
        )
        output_text = response.output_text.strip()
        logger.info("AI raw decision: %s", output_text)
        try:
            decision = TradingDecision.model_validate_json(_normalize_json_payload(output_text))
        except Exception as exc:  # noqa: BLE001
            logger.warning("AI response could not be parsed, defaulting to hold: %s", exc)
            return TradingDecision(
                product_id=snapshot.product_id,
                action="hold",
                confidence=0.0,
                size_quote=Decimal("0"),
                thesis="Model output was invalid JSON; defaulted to hold.",
                risk_notes=["Malformed model output"],
            )
        if decision.product_id != snapshot.product_id:
            decision.product_id = snapshot.product_id
        decision = self._apply_research_overrides(decision=decision, features=feature_pack)
        if decision.size_quote > self._max_notional_per_trade:
            decision.size_quote = self._max_notional_per_trade
        return decision

    def _apply_research_overrides(self, decision: TradingDecision, features: FeaturePack) -> TradingDecision:
        decision.directional_edge_bps = features.directional_edge_bps
        edge_for_action = _edge_for_action(decision.action.value, features.directional_edge_bps)
        decision.expected_edge_bps = edge_for_action
        decision.realized_volatility = float(features.realized_volatility)
        decision.downside_volatility = float(features.downside_volatility)
        decision.sortino_ratio_proxy = float(features.sortino_ratio_proxy)
        decision.regime_consistency = float(features.regime_consistency)
        decision.selected_features = features.selected_features

        if decision.action.value == "hold":
            if self._aggressive_mode and abs(features.directional_edge_bps) >= self._aggressive_hold_to_trade_edge_bps:
                converted_action = "buy" if features.directional_edge_bps > 0 else "sell"
                aggressive_size = (self._max_notional_per_trade * Decimal("0.35")).quantize(Decimal("0.01"))
                return TradingDecision(
                    product_id=decision.product_id,
                    action=converted_action,
                    confidence=max(0.45, decision.confidence),
                    size_quote=aggressive_size,
                    thesis="Aggressive mode converted hold into directional trade on strong edge.",
                    risk_notes=["Aggressive conversion from hold based on edge trigger."],
                    expected_edge_bps=_edge_for_action(converted_action, features.directional_edge_bps),
                    directional_edge_bps=features.directional_edge_bps,
                    realized_volatility=decision.realized_volatility,
                    downside_volatility=decision.downside_volatility,
                    sortino_ratio_proxy=decision.sortino_ratio_proxy,
                    regime_consistency=decision.regime_consistency,
                    selected_features=decision.selected_features,
                )
            return decision

        confidence_after_regime = decision.confidence * float(features.regime_consistency)
        decision.confidence = max(0.0, min(1.0, confidence_after_regime))
        sortino_score = _normalize_sortino(features.sortino_ratio_proxy)
        volatility_score = _volatility_score(float(features.realized_volatility), self._volatility_target)
        quality_score = (decision.confidence * 0.5) + (sortino_score * 0.3) + (volatility_score * 0.2)
        size_multiplier = Decimal(str(quality_score))
        if self._aggressive_mode:
            size_multiplier *= Decimal(str(self._aggressive_size_multiplier))
            quality_score = max(quality_score, 0.30)
        decision.size_quote = (decision.size_quote * size_multiplier).quantize(Decimal("0.01"))
        if quality_score < 0.25 or decision.size_quote < Decimal("1.00"):
            return TradingDecision(
                product_id=decision.product_id,
                action="hold",
                confidence=decision.confidence,
                size_quote=Decimal("0"),
                thesis="Multi-objective quality gate rejected the trade.",
                risk_notes=["Low quality score after regime and downside-risk adjustment."],
                expected_edge_bps=edge_for_action,
                directional_edge_bps=features.directional_edge_bps,
                realized_volatility=decision.realized_volatility,
                downside_volatility=decision.downside_volatility,
                sortino_ratio_proxy=decision.sortino_ratio_proxy,
                regime_consistency=decision.regime_consistency,
                selected_features=decision.selected_features,
            )
        return decision


class StubDecisionEngine:
    def __init__(self, decision: TradingDecision) -> None:
        self._decision = decision

    def decide(self, snapshot: ProductSnapshot, portfolio: PortfolioState, risk_context: dict[str, str]) -> TradingDecision:
        return self._decision


def _normalize_json_payload(payload: str) -> str:
    if payload.startswith("```"):
        lines = [line for line in payload.splitlines() if not line.startswith("```")]
        return "\n".join(lines).strip()
    return payload


def _normalize_sortino(sortino_ratio_proxy: Decimal) -> float:
    clipped = max(Decimal("-1"), min(Decimal("3"), sortino_ratio_proxy))
    return float((clipped + Decimal("1")) / Decimal("4"))


def _volatility_score(realized_volatility_value: float, target: float) -> float:
    if realized_volatility_value <= 0:
        return 1.0
    return max(0.0, min(1.0, target / realized_volatility_value))


def _edge_for_action(action: str, directional_edge_bps: Decimal) -> Decimal:
    if action == "buy":
        return max(Decimal("0"), directional_edge_bps)
    if action == "sell":
        return max(Decimal("0"), -directional_edge_bps)
    return Decimal("0")

