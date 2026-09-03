from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TradeAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class TraderMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


@dataclass(slots=True)
class Candle:
    start: datetime
    low: Decimal
    high: Decimal
    open: Decimal
    close: Decimal
    volume: Decimal


@dataclass(slots=True)
class ProductSnapshot:
    product_id: str
    price: Decimal
    best_bid: Decimal
    best_ask: Decimal
    spread_bps: Decimal
    candles: list[Candle]
    as_of: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class Position:
    product_id: str
    quantity: Decimal = Decimal("0")
    average_entry_price: Decimal = Decimal("0")

    @property
    def is_flat(self) -> bool:
        return self.quantity <= 0


@dataclass(slots=True)
class PortfolioState:
    cash: Decimal
    position: Position
    realized_pnl: Decimal
    equity: Decimal
    peak_equity: Decimal
    daily_realized_pnl: Decimal
    last_price: Decimal


class TradingDecision(BaseModel):
    product_id: str = Field(min_length=3, max_length=32)
    action: TradeAction
    confidence: float = Field(ge=0.0, le=1.0)
    size_quote: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    thesis: str = Field(min_length=1, max_length=600)
    risk_notes: list[str] = Field(default_factory=list, max_length=8)
    stop_loss_pct: float | None = Field(default=None, ge=0.0, le=0.5)
    take_profit_pct: float | None = Field(default=None, ge=0.0, le=2.0)
    expected_edge_bps: Decimal = Field(default=Decimal("0"))
    realized_volatility: float | None = Field(default=None, ge=0.0)
    downside_volatility: float | None = Field(default=None, ge=0.0)
    sortino_ratio_proxy: float | None = None
    regime_consistency: float | None = Field(default=None, ge=0.0, le=1.0)
    selected_features: list[str] = Field(default_factory=list, max_length=12)
    directional_edge_bps: Decimal = Field(default=Decimal("0"))
    probe_trade: bool = False
    force_exit: bool = False

    @field_validator("size_quote")
    @classmethod
    def quantize_size(cls, value: Decimal) -> Decimal:
        return value.quantize(Decimal("0.01"))

    @field_validator("expected_edge_bps")
    @classmethod
    def quantize_edge(cls, value: Decimal) -> Decimal:
        return value.quantize(Decimal("0.01"))

    @field_validator("directional_edge_bps")
    @classmethod
    def quantize_directional_edge(cls, value: Decimal) -> Decimal:
        return value.quantize(Decimal("0.01"))


class ExecutionReport(BaseModel):
    product_id: str
    status: Literal["filled", "rejected", "noop"]
    side: TradeAction
    filled_quote: Decimal = Decimal("0")
    filled_base: Decimal = Decimal("0")
    average_price: Decimal = Decimal("0")
    reason: str
    external_order_id: str | None = None

