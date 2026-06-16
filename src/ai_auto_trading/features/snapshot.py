from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .indicators import (
    atr,
    closes,
    cumulative_vwap,
    ema,
    rate_of_change,
    rsi,
    simple_moving_average,
    swing_high,
    swing_low,
)


FEATURE_SCHEMA_VERSION = "features_v1"


@dataclass(frozen=True)
class TimeframeFeatureSnapshot:
    timeframe: str
    last_open_time: int
    candle_count: int
    last_close: float
    current_volume: float
    taker_buy_ratio: float | None
    ema_fast_9: float | None
    ema_slow_21: float | None
    rsi_14: float | None
    atr_14: float | None
    cumulative_vwap: float | None
    swing_high_20: float | None
    swing_low_20: float | None
    roc_5: float | None
    volume_sma_20: float | None
    volume_ratio_20: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureSnapshot:
    feature_schema_version: str
    symbol: str
    generated_from: str
    timeframes: dict[str, TimeframeFeatureSnapshot]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timeframes"] = {
            key: value.to_dict() for key, value in self.timeframes.items()
        }
        return payload


class FeatureBuilder:
    def __init__(self, feature_schema_version: str = FEATURE_SCHEMA_VERSION) -> None:
        self.feature_schema_version = feature_schema_version

    def build_timeframe_snapshot(
        self, *, timeframe: str, candles: list[dict[str, Any]]
    ) -> TimeframeFeatureSnapshot:
        if not candles:
            raise ValueError("candles must not be empty")
        ordered = sorted(candles, key=lambda row: int(row["open_time"]))
        close_values = closes(ordered)
        volume_values = [float(candle["volume"]) for candle in ordered]
        last = ordered[-1]
        volume_sma_20 = simple_moving_average(volume_values, 20)
        current_volume = float(last["volume"])
        taker_buy_volume = float(last.get("taker_buy_base_asset_volume", 0.0))
        return TimeframeFeatureSnapshot(
            timeframe=timeframe,
            last_open_time=int(last["open_time"]),
            candle_count=len(ordered),
            last_close=float(last["close"]),
            current_volume=current_volume,
            taker_buy_ratio=(
                (taker_buy_volume / current_volume)
                if current_volume > 0.0
                else None
            ),
            ema_fast_9=ema(close_values, 9),
            ema_slow_21=ema(close_values, 21),
            rsi_14=rsi(close_values, 14),
            atr_14=atr(ordered, 14),
            cumulative_vwap=cumulative_vwap(ordered),
            swing_high_20=swing_high(ordered, 20),
            swing_low_20=swing_low(ordered, 20),
            roc_5=rate_of_change(close_values, 5),
            volume_sma_20=volume_sma_20,
            volume_ratio_20=(
                (current_volume / volume_sma_20)
                if volume_sma_20 not in (None, 0.0)
                else None
            ),
        )

    def build_multi_timeframe_snapshot(
        self,
        *,
        symbol: str,
        candles_by_timeframe: dict[str, list[dict[str, Any]]],
        generated_from: str = "contract_price",
    ) -> FeatureSnapshot:
        if not candles_by_timeframe:
            raise ValueError("candles_by_timeframe must not be empty")
        snapshots = {
            timeframe: self.build_timeframe_snapshot(
                timeframe=timeframe, candles=candles
            )
            for timeframe, candles in sorted(candles_by_timeframe.items())
        }
        return FeatureSnapshot(
            feature_schema_version=self.feature_schema_version,
            symbol=symbol,
            generated_from=generated_from,
            timeframes=snapshots,
        )
