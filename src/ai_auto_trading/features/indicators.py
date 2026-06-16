from __future__ import annotations

from typing import Iterable


def ema(values: list[float], period: int) -> float | None:
    if period <= 0:
        raise ValueError("period must be > 0")
    if len(values) < period:
        return None

    multiplier = 2.0 / (period + 1)
    current = sum(values[:period]) / period
    for value in values[period:]:
        current = (value - current) * multiplier + current
    return current


def rsi(values: list[float], period: int) -> float | None:
    if period <= 0:
        raise ValueError("period must be > 0")
    if len(values) < period + 1:
        return None

    gains: list[float] = []
    losses: list[float] = []
    for prev, cur in zip(values, values[1:]):
        delta = cur - prev
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for gain, loss in zip(gains[period:], losses[period:]):
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def atr(candles: list[dict[str, float]], period: int) -> float | None:
    if period <= 0:
        raise ValueError("period must be > 0")
    if len(candles) < period + 1:
        return None

    true_ranges: list[float] = []
    for prev, cur in zip(candles, candles[1:]):
        high = float(cur["high"])
        low = float(cur["low"])
        prev_close = float(prev["close"])
        true_ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))

    current = sum(true_ranges[:period]) / period
    for tr in true_ranges[period:]:
        current = ((current * (period - 1)) + tr) / period
    return current


def cumulative_vwap(candles: list[dict[str, float]]) -> float | None:
    if not candles:
        return None
    total_volume = 0.0
    total_value = 0.0
    for candle in candles:
        high = float(candle["high"])
        low = float(candle["low"])
        close = float(candle["close"])
        volume = float(candle["volume"])
        typical_price = (high + low + close) / 3.0
        total_value += typical_price * volume
        total_volume += volume
    if total_volume == 0:
        return None
    return total_value / total_volume


def swing_high(candles: list[dict[str, float]], lookback: int) -> float | None:
    if lookback <= 0:
        raise ValueError("lookback must be > 0")
    if len(candles) < lookback:
        return None
    return max(float(candle["high"]) for candle in candles[-lookback:])


def swing_low(candles: list[dict[str, float]], lookback: int) -> float | None:
    if lookback <= 0:
        raise ValueError("lookback must be > 0")
    if len(candles) < lookback:
        return None
    return min(float(candle["low"]) for candle in candles[-lookback:])


def rate_of_change(values: list[float], period: int) -> float | None:
    if period <= 0:
        raise ValueError("period must be > 0")
    if len(values) <= period:
        return None
    base = values[-(period + 1)]
    if base == 0:
        return None
    return ((values[-1] - base) / base) * 100.0


def simple_moving_average(values: list[float], period: int) -> float | None:
    if period <= 0:
        raise ValueError("period must be > 0")
    if len(values) < period:
        return None
    window = values[-period:]
    return sum(window) / period


def closes(candles: Iterable[dict[str, float]]) -> list[float]:
    return [float(candle["close"]) for candle in candles]
