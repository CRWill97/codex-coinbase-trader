from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from coinbase_trader.models import PortfolioState, ProductSnapshot, TradeAction, TradingDecision


@dataclass(slots=True)
class RiskLimits:
    max_notional_per_trade: Decimal
    max_position_notional: Decimal
    max_daily_loss: Decimal
    max_drawdown_pct: float
    max_slippage_bps: Decimal
    min_confidence: float
    fee_bps: Decimal
    min_expected_edge_bps: Decimal
    max_trades_per_day: int
    min_minutes_between_trades: int
    volatility_hard_limit: float


@dataclass(slots=True)
class TradeActivity:
    filled_trades_today: int
    minutes_since_last_fill: int | None


class RiskManager:
    def __init__(self, limits: RiskLimits) -> None:
        self._limits = limits

    def context(self) -> dict[str, str]:
        return {
            "max_notional_per_trade": str(self._limits.max_notional_per_trade),
            "max_position_notional": str(self._limits.max_position_notional),
            "max_daily_loss": str(self._limits.max_daily_loss),
            "max_drawdown_pct": str(self._limits.max_drawdown_pct),
            "max_slippage_bps": str(self._limits.max_slippage_bps),
            "min_confidence": str(self._limits.min_confidence),
            "fee_bps": str(self._limits.fee_bps),
            "min_expected_edge_bps": str(self._limits.min_expected_edge_bps),
            "max_trades_per_day": str(self._limits.max_trades_per_day),
            "min_minutes_between_trades": str(self._limits.min_minutes_between_trades),
            "volatility_hard_limit": str(self._limits.volatility_hard_limit),
        }

    def validate(
        self,
        decision: TradingDecision,
        portfolio: PortfolioState,
        snapshot: ProductSnapshot,
        activity: TradeActivity,
    ) -> tuple[bool, str]:
        if decision.product_id != snapshot.product_id:
            return False, "Decision product does not match market snapshot."
        if decision.action == TradeAction.HOLD:
            return True, "Hold decision accepted."
        force_exit = decision.force_exit and decision.action == TradeAction.SELL
        if not force_exit:
            if decision.confidence < self._limits.min_confidence:
                return False, f"Decision confidence {decision.confidence:.2f} below threshold."
            if snapshot.spread_bps > self._limits.max_slippage_bps:
                return False, f"Spread {snapshot.spread_bps:.2f} bps above threshold."
            if decision.realized_volatility is not None and decision.realized_volatility > self._limits.volatility_hard_limit:
                return False, f"Realized volatility {decision.realized_volatility:.4f} above hard limit."
            if activity.filled_trades_today >= self._limits.max_trades_per_day:
                return False, "Daily trade-count limit reached."
            if activity.minutes_since_last_fill is not None and activity.minutes_since_last_fill < self._limits.min_minutes_between_trades:
                return False, "Trade cooldown still active."
            if portfolio.daily_realized_pnl <= -self._limits.max_daily_loss:
                return False, "Daily loss limit reached."
            if portfolio.peak_equity > 0:
                drawdown = Decimal("1") - (portfolio.equity / portfolio.peak_equity)
                if drawdown >= Decimal(str(self._limits.max_drawdown_pct)):
                    return False, f"Drawdown {drawdown:.2%} above threshold."
        estimated_cost_bps = snapshot.spread_bps + (self._limits.fee_bps * Decimal("2"))
        required_edge_bps = estimated_cost_bps + self._limits.min_expected_edge_bps
        if not decision.probe_trade and not force_exit and decision.expected_edge_bps < required_edge_bps:
            return False, f"Expected edge {decision.expected_edge_bps:.2f} bps is below required {required_edge_bps:.2f} bps."
        if decision.size_quote <= 0:
            return False, "Trade size must be positive."
        if not force_exit and decision.size_quote > self._limits.max_notional_per_trade:
            return False, "Trade exceeds per-trade notional cap."

        current_position_notional = portfolio.position.quantity * snapshot.price
        if decision.action == TradeAction.BUY:
            required_cash = decision.size_quote * (Decimal("1") + (self._limits.fee_bps / Decimal("10000")))
            if required_cash > portfolio.cash:
                return False, "Insufficient cash for proposed buy."
            if current_position_notional + decision.size_quote > self._limits.max_position_notional:
                return False, "Buy exceeds max position notional."
        elif decision.action == TradeAction.SELL:
            if portfolio.position.is_flat:
                return False, "Cannot sell with no position."
            max_sell_quote = portfolio.position.quantity * snapshot.best_bid
            if decision.size_quote > max_sell_quote:
                return False, "Sell size exceeds current position value."
        return True, "Decision passed risk checks."

