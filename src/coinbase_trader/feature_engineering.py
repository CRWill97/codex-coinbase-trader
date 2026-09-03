from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import sqrt

from coinbase_trader.indicators import realized_volatility, simple_returns
from coinbase_trader.models import Candle


@dataclass(slots=True)
class FeaturePack:
    selected_features: list[str]
    directional_edge_bps: Decimal
    realized_volatility: Decimal
    downside_volatility: Decimal
    sortino_ratio_proxy: Decimal
    regime_consistency: Decimal
    scalar_features: dict[str, Decimal]


def build_feature_pack(candles: list[Candle], correlation_threshold: float) -> FeaturePack:
    closes = [candle.close for candle in candles]
    volumes = [candle.volume for candle in candles]
    returns = simple_returns(candles)

    if len(closes) < 30:
        raise ValueError("Need at least 30 candles for robust feature generation.")

    rsi_series = _rsi_series(closes, period=14)
    macd_hist_series = _macd_histogram_series(closes)
    roc_series = _roc_series(closes, period=12)
    cci_series = _cci_series(candles, period=20)
    dx_series = _dx_series(candles, period=14)
    stoch_rsi_series = _stoch_rsi_series(rsi_series, period=14)
    obv_series = _obv_series(candles)
    volume_roc = _rate_of_change_series(volumes, period=5)

    realized_vol = realized_volatility(candles, window=24)
    downside_vol = _downside_volatility(returns, window=24)
    mean_return = _mean_decimal(returns[-24:])
    sortino_proxy = Decimal("0") if downside_vol <= Decimal("0") else mean_return / downside_vol
    short_trend = _price_trend_bps(closes, short_window=6, long_window=24)
    medium_trend = _price_trend_bps(closes, short_window=12, long_window=30)
    directional_edge = (short_trend * Decimal("0.6")) + (medium_trend * Decimal("0.4"))
    regime_consistency = _regime_consistency(short_trend, medium_trend, macd_hist_series[-1])

    scalar_features = {
        "rsi": rsi_series[-1],
        "macd_hist": macd_hist_series[-1],
        "roc": roc_series[-1],
        "cci": cci_series[-1],
        "dx": dx_series[-1],
        "stoch_rsi": stoch_rsi_series[-1],
        "obv": obv_series[-1],
        "volume_roc": volume_roc[-1],
        "realized_volatility": realized_vol,
        "downside_volatility": downside_vol,
        "sortino_proxy": sortino_proxy,
        "short_trend_bps": short_trend,
        "medium_trend_bps": medium_trend,
    }
    series_map = {
        "rsi": rsi_series,
        "macd_hist": macd_hist_series,
        "roc": roc_series,
        "cci": cci_series,
        "dx": dx_series,
        "stoch_rsi": stoch_rsi_series,
        "obv": obv_series,
        "volume_roc": volume_roc,
    }
    selected = _select_low_correlation_features(series_map, threshold=correlation_threshold)
    return FeaturePack(
        selected_features=selected,
        directional_edge_bps=directional_edge,
        realized_volatility=realized_vol,
        downside_volatility=downside_vol,
        sortino_ratio_proxy=sortino_proxy,
        regime_consistency=regime_consistency,
        scalar_features=scalar_features,
    )


def _mean_decimal(values: list[Decimal]) -> Decimal:
    if not values:
        return Decimal("0")
    return sum(values, start=Decimal("0")) / Decimal(len(values))


def _ema_series(values: list[Decimal], period: int) -> list[Decimal]:
    if len(values) < period:
        raise ValueError(f"Need at least {period} values for EMA.")
    multiplier = Decimal("2") / Decimal(period + 1)
    ema_values = [values[period - 1]]
    for value in values[period:]:
        ema_values.append((value - ema_values[-1]) * multiplier + ema_values[-1])
    prefix = [ema_values[0]] * (period - 1)
    return prefix + ema_values


def _rsi_series(closes: list[Decimal], period: int) -> list[Decimal]:
    if len(closes) <= period:
        raise ValueError(f"Need at least {period + 1} closes for RSI.")
    deltas = [curr - prev for prev, curr in zip(closes, closes[1:], strict=False)]
    gains = [max(delta, Decimal("0")) for delta in deltas]
    losses = [max(-delta, Decimal("0")) for delta in deltas]

    avg_gain = _mean_decimal(gains[:period])
    avg_loss = _mean_decimal(losses[:period])
    result: list[Decimal] = [Decimal("50")] * period
    for index in range(period, len(deltas)):
        avg_gain = ((avg_gain * Decimal(period - 1)) + gains[index]) / Decimal(period)
        avg_loss = ((avg_loss * Decimal(period - 1)) + losses[index]) / Decimal(period)
        if avg_loss == 0:
            result.append(Decimal("100"))
            continue
        relative_strength = avg_gain / avg_loss
        result.append(Decimal("100") - (Decimal("100") / (Decimal("1") + relative_strength)))
    result.append(result[-1])
    return result


def _macd_histogram_series(closes: list[Decimal]) -> list[Decimal]:
    ema_fast = _ema_series(closes, period=12)
    ema_slow = _ema_series(closes, period=26)
    macd_line = [fast - slow for fast, slow in zip(ema_fast, ema_slow, strict=False)]
    signal = _ema_series(macd_line, period=9)
    return [line - sig for line, sig in zip(macd_line, signal, strict=False)]


def _roc_series(values: list[Decimal], period: int) -> list[Decimal]:
    if len(values) <= period:
        raise ValueError(f"Need at least {period + 1} values for ROC.")
    out = [Decimal("0")] * period
    for idx in range(period, len(values)):
        previous = values[idx - period]
        if previous == 0:
            out.append(Decimal("0"))
            continue
        out.append((values[idx] - previous) / previous)
    return out


def _rate_of_change_series(values: list[Decimal], period: int) -> list[Decimal]:
    return _roc_series(values, period=period)


def _cci_series(candles: list[Candle], period: int) -> list[Decimal]:
    typical_prices = [(candle.high + candle.low + candle.close) / Decimal("3") for candle in candles]
    if len(typical_prices) < period:
        raise ValueError(f"Need at least {period} candles for CCI.")
    output = [Decimal("0")] * (period - 1)
    for idx in range(period - 1, len(typical_prices)):
        window = typical_prices[idx - period + 1 : idx + 1]
        sma = _mean_decimal(window)
        mean_dev = _mean_decimal([abs(value - sma) for value in window])
        if mean_dev == 0:
            output.append(Decimal("0"))
            continue
        output.append((typical_prices[idx] - sma) / (Decimal("0.015") * mean_dev))
    return output


def _dx_series(candles: list[Candle], period: int) -> list[Decimal]:
    if len(candles) <= period:
        raise ValueError(f"Need at least {period + 1} candles for DX.")
    true_ranges: list[Decimal] = []
    plus_dm: list[Decimal] = []
    minus_dm: list[Decimal] = []
    for prev, curr in zip(candles, candles[1:], strict=False):
        high_move = curr.high - prev.high
        low_move = prev.low - curr.low
        plus_dm.append(high_move if high_move > low_move and high_move > 0 else Decimal("0"))
        minus_dm.append(low_move if low_move > high_move and low_move > 0 else Decimal("0"))
        tr = max(curr.high - curr.low, abs(curr.high - prev.close), abs(curr.low - prev.close))
        true_ranges.append(tr)

    output = [Decimal("0")] * period
    for idx in range(period - 1, len(true_ranges)):
        tr_window = true_ranges[idx - period + 1 : idx + 1]
        plus_window = plus_dm[idx - period + 1 : idx + 1]
        minus_window = minus_dm[idx - period + 1 : idx + 1]
        atr = _mean_decimal(tr_window)
        if atr == 0:
            output.append(Decimal("0"))
            continue
        di_plus = (Decimal("100") * _mean_decimal(plus_window)) / atr
        di_minus = (Decimal("100") * _mean_decimal(minus_window)) / atr
        denom = di_plus + di_minus
        if denom == 0:
            output.append(Decimal("0"))
            continue
        output.append((abs(di_plus - di_minus) / denom) * Decimal("100"))
    output.append(output[-1])
    return output


def _stoch_rsi_series(rsi_values: list[Decimal], period: int) -> list[Decimal]:
    if len(rsi_values) < period:
        raise ValueError(f"Need at least {period} values for stochastic RSI.")
    output = [Decimal("0")] * (period - 1)
    for idx in range(period - 1, len(rsi_values)):
        window = rsi_values[idx - period + 1 : idx + 1]
        lowest = min(window)
        highest = max(window)
        if highest == lowest:
            output.append(Decimal("0.5"))
            continue
        output.append((rsi_values[idx] - lowest) / (highest - lowest))
    return output


def _obv_series(candles: list[Candle]) -> list[Decimal]:
    obv = Decimal("0")
    series: list[Decimal] = [Decimal("0")]
    for prev, curr in zip(candles, candles[1:], strict=False):
        if curr.close > prev.close:
            obv += curr.volume
        elif curr.close < prev.close:
            obv -= curr.volume
        series.append(obv)
    return series


def _downside_volatility(returns: list[Decimal], window: int) -> Decimal:
    if len(returns) < window:
        raise ValueError(f"Need at least {window} return points for downside volatility.")
    subset = returns[-window:]
    downside = [ret for ret in subset if ret < 0]
    if not downside:
        return Decimal("0")
    squares = [value * value for value in downside]
    mean_square = _mean_decimal(squares)
    return mean_square.sqrt()


def _price_trend_bps(closes: list[Decimal], short_window: int, long_window: int) -> Decimal:
    short = _mean_decimal(closes[-short_window:])
    long = _mean_decimal(closes[-long_window:])
    if long == 0:
        return Decimal("0")
    return ((short - long) / long) * Decimal("10000")


def _regime_consistency(short_trend_bps: Decimal, medium_trend_bps: Decimal, macd_hist: Decimal) -> Decimal:
    votes = 0
    if short_trend_bps > 0:
        votes += 1
    if medium_trend_bps > 0:
        votes += 1
    if macd_hist > 0:
        votes += 1
    if votes in (0, 3):
        return Decimal("1")
    if votes in (1, 2):
        return Decimal("0.5")
    return Decimal("0")


def _select_low_correlation_features(series_map: dict[str, list[Decimal]], threshold: float) -> list[str]:
    ordered_names = sorted(series_map.keys())
    selected: list[str] = []
    for name in ordered_names:
        candidate = series_map[name]
        if not selected:
            selected.append(name)
            continue
        correlations = [abs(_pearson_corr(candidate, series_map[chosen])) for chosen in selected]
        if all(corr <= threshold for corr in correlations):
            selected.append(name)
    return selected


def _pearson_corr(left: list[Decimal], right: list[Decimal]) -> float:
    length = min(len(left), len(right))
    if length < 3:
        return 0.0
    left_slice = [float(value) for value in left[-length:]]
    right_slice = [float(value) for value in right[-length:]]
    mean_left = sum(left_slice) / length
    mean_right = sum(right_slice) / length
    cov = sum((l - mean_left) * (r - mean_right) for l, r in zip(left_slice, right_slice, strict=False))
    var_left = sum((l - mean_left) ** 2 for l in left_slice)
    var_right = sum((r - mean_right) ** 2 for r in right_slice)
    if var_left == 0 or var_right == 0:
        return 0.0
    return cov / sqrt(var_left * var_right)

