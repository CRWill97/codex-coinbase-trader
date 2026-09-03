from __future__ import annotations

from decimal import Decimal

from coinbase_trader.models import Candle


def simple_returns(candles: list[Candle]) -> list[Decimal]:
    returns: list[Decimal] = []
    for previous, current in zip(candles, candles[1:], strict=False):
        if previous.close == 0:
            continue
        returns.append((current.close - previous.close) / previous.close)
    return returns


def simple_moving_average(values: list[Decimal], window: int) -> Decimal:
    if len(values) < window:
        raise ValueError(f"Need at least {window} values, received {len(values)}")
    subset = values[-window:]
    return sum(subset, start=Decimal("0")) / Decimal(window)


def realized_volatility(candles: list[Candle], window: int = 24) -> Decimal:
    returns = simple_returns(candles)
    if len(returns) < window:
        raise ValueError(f"Need at least {window + 1} candles, received {len(candles)}")
    subset = returns[-window:]
    mean = sum(subset, start=Decimal("0")) / Decimal(window)
    variance = sum((value - mean) ** 2 for value in subset) / Decimal(window)
    return variance.sqrt()

