from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ai_auto_trading.ai.gate import AIGateConfig, AITradeAssistant, apply_ai_entry_gate
from ai_auto_trading.features.snapshot import FeatureSnapshot, TimeframeFeatureSnapshot
from ai_auto_trading.models import PolicyVersionInfo, PositionState, TradeRecord
from ai_auto_trading.schema_validation import validate_trade_record
from ai_auto_trading.strategy.rule_based import (
    RuleStrategyContext,
    RuleStrategyParameters,
    evaluate_rule_signal,
)
from ai_auto_trading.strategy.trade_management import (
    atr_trail_activation_reached,
    atr_trail_price,
)

_TIMEFRAME_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


@dataclass(frozen=True)
class BacktestConfig:
    symbol: str = "BTCUSDT"
    execution_timeframe: str = "3m"
    confirmation_timeframe: str = "15m"
    macro_timeframe: str = "1h"
    lower_mark_timeframe: str = "1m"
    leverage_at_entry: float = 2.0
    entry_notional_usdt: float = 1000.0
    fee_bps: float = 4.0
    slippage_bps: float = 2.0
    atr_trailing_multiplier: float = 2.5
    atr_trail_activation_profit_r: float = 0.5
    atr_trail_min_bars: int = 2
    exit_policy: str = "atr_trail"
    break_even_activation_profit_r: float = 0.5
    break_even_min_bars: int = 2
    fixed_take_profit_r: float = 1.0
    partial_take_profit_r: float = 1.0
    partial_take_profit_fraction: float = 0.5
    early_scratch_min_bars: int = 0
    early_scratch_min_mfe_r: float = 0.0
    early_scratch_max_adverse_r: float = 0.0
    feature_lookback_candles: int = 120
    max_holding_bars: int = 8
    policy_version: str = "policy_v1"
    strategy_version: str = "strategy_v1"
    feature_schema_version: str = "features_v1"
    model_base: str = "rule_only"
    adapter_version: str | None = None
    dataset_version: str | None = None
    min_trades_for_decision: int = 1
    min_profit_factor: float = 1.0
    max_allowed_drawdown_usdt: float = 100.0


@dataclass
class OpenTradeState:
    position: PositionState
    opened_at_ms: int
    signal_reason_codes: list[str]
    model_base: str
    adapter_version: str | None
    ai_snapshot: dict[str, Any] | None
    highest_high: float
    lowest_low: float
    initial_risk_usdt: float
    bars_held: int = 0
    max_favorable_excursion_usdt: float = 0.0
    max_adverse_excursion_usdt: float = 0.0
    stop_override_price_mark: float | None = None
    partial_take_profit_taken: bool = False
    atr_trail_history: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.atr_trail_history is None:
            self.atr_trail_history = []


@dataclass(frozen=True)
class BacktestMetrics:
    trades: int
    wins: int
    losses: int
    win_rate: float
    total_realized_pnl_usdt: float
    total_fees_usdt: float
    total_slippage_usdt: float
    total_pnl_after_fees_usdt: float
    profit_factor: float
    max_drawdown_usdt: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionExplanationReport:
    recommendation: str
    reasons: list[str]
    metrics: dict[str, Any]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BacktestResult:
    trade_records: list[TradeRecord]
    metrics: BacktestMetrics
    decision_report: DecisionExplanationReport


@dataclass(frozen=True)
class LocalExitEvaluation:
    trade_records: list[TradeRecord]
    updated_open_trade: OpenTradeState | None


class _RollingEma:
    def __init__(self, period: int) -> None:
        self.period = period
        self.multiplier = 2.0 / (period + 1)
        self._seed: list[float] = []
        self.current: float | None = None

    def update(self, value: float) -> float | None:
        if self.current is None:
            self._seed.append(value)
            if len(self._seed) == self.period:
                self.current = sum(self._seed) / self.period
            return self.current
        self.current = (value - self.current) * self.multiplier + self.current
        return self.current


class _RollingRsi:
    def __init__(self, period: int) -> None:
        self.period = period
        self.prev_close: float | None = None
        self._gains: list[float] = []
        self._losses: list[float] = []
        self.avg_gain: float | None = None
        self.avg_loss: float | None = None

    def update(self, close: float) -> float | None:
        if self.prev_close is None:
            self.prev_close = close
            return None
        delta = close - self.prev_close
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        if self.avg_gain is None or self.avg_loss is None:
            self._gains.append(gain)
            self._losses.append(loss)
            if len(self._gains) == self.period:
                self.avg_gain = sum(self._gains) / self.period
                self.avg_loss = sum(self._losses) / self.period
        else:
            self.avg_gain = ((self.avg_gain * (self.period - 1)) + gain) / self.period
            self.avg_loss = ((self.avg_loss * (self.period - 1)) + loss) / self.period
        self.prev_close = close
        if self.avg_gain is None or self.avg_loss is None:
            return None
        if self.avg_loss == 0:
            return 100.0
        rs = self.avg_gain / self.avg_loss
        return 100.0 - (100.0 / (1.0 + rs))


class _RollingAtr:
    def __init__(self, period: int) -> None:
        self.period = period
        self.prev_close: float | None = None
        self._true_ranges: list[float] = []
        self.current: float | None = None

    def update(self, high: float, low: float, close: float) -> float | None:
        if self.prev_close is None:
            self.prev_close = close
            return None
        tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))
        if self.current is None:
            self._true_ranges.append(tr)
            if len(self._true_ranges) == self.period:
                self.current = sum(self._true_ranges) / self.period
        else:
            self.current = ((self.current * (self.period - 1)) + tr) / self.period
        self.prev_close = close
        return self.current


class _RollingVwap:
    def __init__(self, lookback: int | None = None) -> None:
        self.lookback = lookback
        self.rows: deque[tuple[float, float]] = deque()
        self.total_volume = 0.0
        self.total_value = 0.0

    def update(self, high: float, low: float, close: float, volume: float) -> float | None:
        typical_price = (high + low + close) / 3.0
        weighted_value = typical_price * volume
        self.rows.append((weighted_value, volume))
        self.total_value += weighted_value
        self.total_volume += volume
        if self.lookback is not None:
            while len(self.rows) > self.lookback:
                old_value, old_volume = self.rows.popleft()
                self.total_value -= old_value
                self.total_volume -= old_volume
        if self.total_volume == 0.0:
            return None
        return self.total_value / self.total_volume


class _RollingExtrema:
    def __init__(self, lookback: int, *, mode: str) -> None:
        self.lookback = lookback
        self.mode = mode
        self.index = -1
        self.values: deque[tuple[int, float]] = deque()

    def update(self, value: float) -> float | None:
        self.index += 1
        if self.mode == "max":
            while self.values and self.values[-1][1] <= value:
                self.values.pop()
        else:
            while self.values and self.values[-1][1] >= value:
                self.values.pop()
        self.values.append((self.index, value))
        cutoff = self.index - self.lookback
        while self.values and self.values[0][0] <= cutoff:
            self.values.popleft()
        if self.index + 1 < self.lookback:
            return None
        return self.values[0][1]


class _RollingRoc:
    def __init__(self, period: int) -> None:
        self.period = period
        self.values: deque[float] = deque()

    def update(self, value: float) -> float | None:
        self.values.append(value)
        if len(self.values) > self.period + 1:
            self.values.popleft()
        if len(self.values) <= self.period:
            return None
        base = self.values[0]
        if base == 0.0:
            return None
        return ((value - base) / base) * 100.0


class _RollingSma:
    def __init__(self, period: int) -> None:
        self.period = period
        self.values: deque[float] = deque()
        self.total = 0.0

    def update(self, value: float) -> float | None:
        self.values.append(value)
        self.total += value
        if len(self.values) > self.period:
            self.total -= self.values.popleft()
        if len(self.values) < self.period:
            return None
        return self.total / self.period


class _RollingSnapshotBuilder:
    def __init__(self, timeframe: str, *, feature_lookback_candles: int | None = None) -> None:
        self.timeframe = timeframe
        self.count = 0
        self.ema_fast = _RollingEma(9)
        self.ema_slow = _RollingEma(21)
        self.rsi = _RollingRsi(14)
        self.atr = _RollingAtr(14)
        self.vwap = _RollingVwap(feature_lookback_candles)
        self.swing_high = _RollingExtrema(20, mode="max")
        self.swing_low = _RollingExtrema(20, mode="min")
        self.roc = _RollingRoc(5)
        self.volume_sma = _RollingSma(20)

    def update(self, candle: dict[str, Any]):
        self.count += 1
        close = float(candle["close"])
        high = float(candle["high"])
        low = float(candle["low"])
        volume = float(candle.get("volume", 0.0))
        taker_buy_volume = float(candle.get("taker_buy_base_asset_volume", 0.0))
        volume_sma_20 = self.volume_sma.update(volume)
        return {
            "timeframe": self.timeframe,
            "last_open_time": int(candle["open_time"]),
            "candle_count": self.count,
            "last_close": close,
            "current_volume": volume,
            "taker_buy_ratio": (
                (taker_buy_volume / volume)
                if volume > 0.0
                else None
            ),
            "ema_fast_9": self.ema_fast.update(close),
            "ema_slow_21": self.ema_slow.update(close),
            "rsi_14": self.rsi.update(close),
            "atr_14": self.atr.update(high, low, close),
            "cumulative_vwap": self.vwap.update(high, low, close, volume),
            "swing_high_20": self.swing_high.update(high),
            "swing_low_20": self.swing_low.update(low),
            "roc_5": self.roc.update(close),
            "volume_sma_20": volume_sma_20,
            "volume_ratio_20": (
                (volume / volume_sma_20)
                if volume_sma_20 not in (None, 0.0)
                else None
            ),
        }


def load_kline_parquet(
    path: Path,
    *,
    symbol: str | None = None,
    interval: str | None = None,
    start_time: int | None = None,
    end_time: int | None = None,
) -> list[dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if symbol is not None and str(row.get("symbol")) != symbol:
            continue
        if interval is not None and str(row.get("interval")) != interval:
            continue
        open_time = int(row["open_time"])
        if start_time is not None and open_time < start_time:
            continue
        if end_time is not None and open_time > end_time:
            continue
        filtered.append(_normalize_kline_row(row))
    return sorted(filtered, key=lambda item: int(item["open_time"]))


def load_funding_rate_parquet(
    path: Path,
    *,
    symbol: str | None = None,
    start_time: int | None = None,
    end_time: int | None = None,
) -> list[dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if symbol is not None and str(row.get("symbol")) != symbol:
            continue
        funding_time = int(row["funding_time"])
        if start_time is not None and funding_time < start_time:
            continue
        if end_time is not None and funding_time > end_time:
            continue
        filtered.append(
            {
                "dataset": "funding_rate",
                "symbol": str(row.get("symbol", symbol or "BTCUSDT")),
                "funding_time": funding_time,
                "funding_rate": float(row["funding_rate"]),
                "mark_price": (
                    None
                    if row.get("mark_price") in (None, "")
                    else float(row["mark_price"])
                ),
            }
        )
    return sorted(filtered, key=lambda item: int(item["funding_time"]))


def resample_klines(
    candles: list[dict[str, Any]],
    *,
    target_interval: str,
    source_interval: str | None = None,
) -> list[dict[str, Any]]:
    if not candles:
        return []

    resolved_source = source_interval or str(candles[0].get("interval") or "")
    source_ms = _timeframe_ms(resolved_source)
    target_ms = _timeframe_ms(target_interval)
    if target_ms < source_ms:
        raise ValueError("target_interval must be >= source_interval")
    if target_ms % source_ms != 0:
        raise ValueError("target_interval must be an exact multiple of source_interval")
    if target_ms == source_ms:
        return [
            {
                **_normalize_kline_row(row),
                "interval": target_interval,
            }
            for row in sorted(candles, key=lambda item: int(item["open_time"]))
        ]

    expected_rows = target_ms // source_ms
    ordered = sorted(candles, key=lambda item: int(item["open_time"]))
    output: list[dict[str, Any]] = []
    current_bucket_start: int | None = None
    bucket_rows: list[dict[str, Any]] = []

    def flush_bucket() -> None:
        nonlocal bucket_rows, current_bucket_start
        if current_bucket_start is None or not bucket_rows:
            bucket_rows = []
            return
        if len(bucket_rows) != expected_rows:
            bucket_rows = []
            return
        expected_open_times = [
            current_bucket_start + (index * source_ms)
            for index in range(expected_rows)
        ]
        actual_open_times = [int(row["open_time"]) for row in bucket_rows]
        if actual_open_times != expected_open_times:
            bucket_rows = []
            return
        first = bucket_rows[0]
        last = bucket_rows[-1]
        output.append(
            {
                "dataset": first.get("dataset", "contract_klines"),
                "symbol": first.get("symbol", "BTCUSDT"),
                "interval": target_interval,
                "open_time": current_bucket_start,
                "open": float(first["open"]),
                "high": max(float(row["high"]) for row in bucket_rows),
                "low": min(float(row["low"]) for row in bucket_rows),
                "close": float(last["close"]),
                "volume": sum(float(row.get("volume", 0.0)) for row in bucket_rows),
                "close_time": current_bucket_start + target_ms - 1,
                "quote_asset_volume": sum(
                    float(row.get("quote_asset_volume", 0.0)) for row in bucket_rows
                ),
                "number_of_trades": sum(
                    int(row.get("number_of_trades", 0)) for row in bucket_rows
                ),
                "taker_buy_base_asset_volume": sum(
                    float(row.get("taker_buy_base_asset_volume", 0.0))
                    for row in bucket_rows
                ),
                "taker_buy_quote_asset_volume": sum(
                    float(row.get("taker_buy_quote_asset_volume", 0.0))
                    for row in bucket_rows
                ),
                "ignore": str(last.get("ignore", "0")),
                "collected_at_ms": max(
                    int(row.get("collected_at_ms", 0)) for row in bucket_rows
                ),
            }
        )
        bucket_rows = []

    for row in ordered:
        open_time = int(row["open_time"])
        bucket_start = open_time - (open_time % target_ms)
        if current_bucket_start is None:
            current_bucket_start = bucket_start
        if bucket_start != current_bucket_start:
            flush_bucket()
            current_bucket_start = bucket_start
        bucket_rows.append(_normalize_kline_row(row))
    flush_bucket()
    return output


def run_hybrid_backtest(
    *,
    contract_candles_by_timeframe: dict[str, list[dict[str, Any]]],
    lower_mark_price_candles: list[dict[str, Any]],
    funding_rate_rows: list[dict[str, Any]] | None = None,
    config: BacktestConfig | None = None,
    strategy_params: RuleStrategyParameters | None = None,
    ai_trade_assistant: AITradeAssistant | None = None,
    ai_gate_config: AIGateConfig | None = None,
) -> BacktestResult:
    config = config or BacktestConfig()
    strategy_params = strategy_params or RuleStrategyParameters(
        execution_timeframe=config.execution_timeframe,
        confirmation_timeframe=config.confirmation_timeframe,
        macro_timeframe=config.macro_timeframe,
    )
    ai_gate_config = ai_gate_config or AIGateConfig()

    execution_candles = sorted(
        contract_candles_by_timeframe[config.execution_timeframe],
        key=lambda row: int(row["open_time"]),
    )
    lower_mark_price_candles = sorted(
        lower_mark_price_candles, key=lambda row: int(row["open_time"])
    )
    snapshot_cache = _build_snapshot_cache(
        contract_candles_by_timeframe=contract_candles_by_timeframe,
        feature_lookback_candles=config.feature_lookback_candles,
    )
    snapshot_cursor_positions = {
        timeframe: -1 for timeframe in snapshot_cache.keys()
    }
    funding_rate_rows = sorted(
        funding_rate_rows or [], key=lambda row: int(row["funding_time"])
    )

    trade_records: list[TradeRecord] = []
    open_trade: OpenTradeState | None = None
    pending_entry: dict[str, Any] | None = None
    mark_start = 0
    mark_end = 0
    funding_cursor = -1

    for index, candle in enumerate(execution_candles):
        candle_open_time = int(candle["open_time"])
        candle_close_time = int(candle["close_time"])
        candle_open = float(candle["open"])
        candle_high = float(candle["high"])
        candle_low = float(candle["low"])
        candle_close = float(candle["close"])

        while mark_start < len(lower_mark_price_candles) and int(lower_mark_price_candles[mark_start]["open_time"]) < candle_open_time:
            mark_start += 1
        if mark_end < mark_start:
            mark_end = mark_start
        while mark_end < len(lower_mark_price_candles) and int(lower_mark_price_candles[mark_end]["open_time"]) <= candle_close_time:
            mark_end += 1
        window_mark_candles = lower_mark_price_candles[mark_start:mark_end]
        while (
            funding_cursor + 1 < len(funding_rate_rows)
            and int(funding_rate_rows[funding_cursor + 1]["funding_time"]) <= candle_close_time
        ):
            funding_cursor += 1
        latest_funding_rate = (
            float(funding_rate_rows[funding_cursor]["funding_rate"])
            if funding_cursor >= 0
            else None
        )
        snapshot = _snapshot_at_or_before(
            snapshot_cache=snapshot_cache,
            snapshot_cursor_positions=snapshot_cursor_positions,
            symbol=config.symbol,
            feature_schema_version=config.feature_schema_version,
            end_open_time=candle_open_time,
        )

        if pending_entry is not None:
            mark_reference = (
                float(window_mark_candles[0]["open"])
                if window_mark_candles
                else candle_open
            )
            quantity = (config.entry_notional_usdt * pending_entry["size_multiplier"]) / candle_open
            position = PositionState(
                side=pending_entry["side"],
                quantity=quantity,
                leverage_at_entry=config.leverage_at_entry,
                entry_contract_price_avg=candle_open,
                entry_mark_price=mark_reference,
                symbol=config.symbol,
            )
            open_trade = OpenTradeState(
                position=position,
                opened_at_ms=candle_open_time,
                signal_reason_codes=pending_entry["reason_codes"],
                model_base=pending_entry["model_base"],
                adapter_version=pending_entry["adapter_version"],
                ai_snapshot=pending_entry["ai_snapshot"],
                highest_high=candle_high,
                lowest_low=candle_low,
                initial_risk_usdt=position.hard_stop_trigger_loss_usdt,
            )
            pending_entry = None

        if open_trade is not None:
            _update_excursions(open_trade, candle_high=candle_high, candle_low=candle_low)
            hard_stop_record = _maybe_hard_stop_exit(
                open_trade=open_trade,
                execution_candle=candle,
                mark_window=window_mark_candles,
                config=config,
            )
            if hard_stop_record is not None:
                trade_records.append(hard_stop_record)
                open_trade = None
                continue

            local_exit = _maybe_local_strategy_exit(
                open_trade=open_trade,
                execution_snapshot=(snapshot.timeframes[config.execution_timeframe] if snapshot is not None else None),
                candle_high=candle_high,
                candle_low=candle_low,
                candle_close=candle_close,
                candle_close_time=candle_close_time,
                config=config,
            )
            if local_exit.trade_records:
                trade_records.extend(local_exit.trade_records)
                open_trade = local_exit.updated_open_trade
                continue

        if open_trade is None and pending_entry is None and index < len(execution_candles) - 1:
            if snapshot is None:
                continue
            rule_decision = evaluate_rule_signal(
                RuleStrategyContext(
                    snapshot=snapshot,
                    latest_funding_rate=latest_funding_rate,
                ),
                params=strategy_params,
            )
            gated_decision = apply_ai_entry_gate(
                rule_decision=rule_decision,
                snapshot=snapshot,
                assistant=ai_trade_assistant,
                gate_config=ai_gate_config,
            )
            if gated_decision.action in {"LONG", "SHORT"}:
                pending_entry = {
                    "side": gated_decision.action,
                    "reason_codes": gated_decision.reason_codes,
                    "size_multiplier": gated_decision.size_multiplier,
                    "model_base": gated_decision.model_base,
                    "adapter_version": gated_decision.adapter_version,
                    "ai_snapshot": gated_decision.ai_snapshot,
                }

    metrics = _compute_metrics(trade_records)
    decision_report = _build_decision_report(metrics, config)
    return BacktestResult(
        trade_records=trade_records,
        metrics=metrics,
        decision_report=decision_report,
    )


def write_trade_logs_jsonl(trade_records: list[TradeRecord], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in trade_records:
            payload = record.to_dict()
            validate_trade_record(payload)
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")
    return output_path


def write_decision_report_json(
    report: DecisionExplanationReport, output_path: Path
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return output_path


def _update_excursions(
    open_trade: OpenTradeState, *, candle_high: float, candle_low: float
) -> None:
    open_trade.highest_high = max(open_trade.highest_high, candle_high)
    open_trade.lowest_low = min(open_trade.lowest_low, candle_low)
    entry_price = open_trade.position.entry_contract_price_avg
    qty = open_trade.position.quantity
    if open_trade.position.side == "LONG":
        favorable = max(0.0, (candle_high - entry_price) * qty)
        adverse = max(0.0, (entry_price - candle_low) * qty)
    else:
        favorable = max(0.0, (entry_price - candle_low) * qty)
        adverse = max(0.0, (candle_high - entry_price) * qty)
    open_trade.max_favorable_excursion_usdt = max(
        open_trade.max_favorable_excursion_usdt, favorable
    )
    open_trade.max_adverse_excursion_usdt = max(
        open_trade.max_adverse_excursion_usdt, adverse
    )


def _maybe_hard_stop_exit(
    *,
    open_trade: OpenTradeState,
    execution_candle: dict[str, Any],
    mark_window: list[dict[str, Any]],
    config: BacktestConfig,
) -> TradeRecord | None:
    position = open_trade.position
    threshold_price = open_trade.stop_override_price_mark or position.hard_stop_trigger_price_mark
    breach_mark_price: float | None = None

    for row in mark_window:
        if position.side == "LONG" and float(row["low"]) <= threshold_price:
            breach_mark_price = min(threshold_price, float(row["low"]))
            break
        if position.side == "SHORT" and float(row["high"]) >= threshold_price:
            breach_mark_price = max(threshold_price, float(row["high"]))
            break

    if breach_mark_price is None:
        return None

    adverse_reference = _worse_contract_exit_price(
        side=position.side,
        threshold_price=threshold_price,
        execution_close=float(execution_candle["close"]),
    )
    exit_price, slippage_usdt = _apply_adverse_slippage(
        side=position.side,
        base_price=adverse_reference,
        quantity=position.quantity,
        slippage_bps=config.slippage_bps,
    )
    exit_reason = (
        "BREAK_EVEN_STOP_EXIT"
        if open_trade.stop_override_price_mark is not None
        else "HARD_STOP_MARK_PRICE"
    )

    return _build_trade_record(
        open_trade=open_trade,
        config=config,
        exit_reason=exit_reason,
        exit_contract_price=exit_price,
        exit_mark_price=breach_mark_price,
        closed_at_ms=int(execution_candle["close_time"]),
        fees_usdt=_fees_for_round_trip(
            entry_notional=position.filled_entry_notional,
            exit_notional=exit_price * position.quantity,
            fee_bps=config.fee_bps,
        ),
        slippage_usdt=slippage_usdt,
    )


def _maybe_local_strategy_exit(
    *,
    open_trade: OpenTradeState,
    execution_snapshot: TimeframeFeatureSnapshot | None,
    candle_high: float,
    candle_low: float,
    candle_close: float,
    candle_close_time: int,
    config: BacktestConfig,
) -> LocalExitEvaluation:
    position = open_trade.position
    open_trade.bars_held += 1

    if execution_snapshot is None:
        return LocalExitEvaluation(trade_records=[], updated_open_trade=open_trade)

    risk_unit = max(open_trade.initial_risk_usdt, 1e-9)
    current_unrealized = (
        (candle_close - position.entry_contract_price_avg) * position.quantity
        if position.side == "LONG"
        else (position.entry_contract_price_avg - candle_close) * position.quantity
    )
    if (
        config.early_scratch_min_bars > 0
        and open_trade.bars_held >= config.early_scratch_min_bars
        and (open_trade.max_favorable_excursion_usdt / risk_unit) < config.early_scratch_min_mfe_r
        and (current_unrealized / risk_unit) <= -config.early_scratch_max_adverse_r
    ):
        exit_price, slippage_usdt = _apply_adverse_slippage(
            side=position.side,
            base_price=candle_close,
            quantity=position.quantity,
            slippage_bps=config.slippage_bps,
        )
        return LocalExitEvaluation(
            trade_records=[
                _build_trade_record(
                    open_trade=open_trade,
                    config=config,
                    exit_reason="EARLY_FAIL_EXIT",
                    exit_contract_price=exit_price,
                    exit_mark_price=candle_close,
                    closed_at_ms=candle_close_time,
                    fees_usdt=_fees_for_round_trip(
                        entry_notional=position.filled_entry_notional,
                        exit_notional=exit_price * position.quantity,
                        fee_bps=config.fee_bps,
                    ),
                    slippage_usdt=slippage_usdt,
                )
            ],
            updated_open_trade=None,
        )

    if config.exit_policy == "break_even_time_stop":
        if (
            open_trade.stop_override_price_mark is None
            and atr_trail_activation_reached(
                side=position.side,
                entry_contract_price_avg=position.entry_contract_price_avg,
                quantity=position.quantity,
                leverage_at_entry=position.leverage_at_entry,
                highest_high=open_trade.highest_high,
                lowest_low=open_trade.lowest_low,
                bars_held=open_trade.bars_held,
                min_bars=config.break_even_min_bars,
                min_profit_r=config.break_even_activation_profit_r,
            )
        ):
            open_trade.stop_override_price_mark = position.entry_contract_price_avg
            if current_unrealized <= 0.0:
                exit_price, slippage_usdt = _apply_adverse_slippage(
                    side=position.side,
                    base_price=candle_close,
                    quantity=position.quantity,
                    slippage_bps=config.slippage_bps,
                )
                return LocalExitEvaluation(
                    trade_records=[
                        _build_trade_record(
                            open_trade=open_trade,
                            config=config,
                            exit_reason="BREAK_EVEN_STOP_EXIT",
                            exit_contract_price=exit_price,
                            exit_mark_price=candle_close,
                            closed_at_ms=candle_close_time,
                            fees_usdt=_fees_for_round_trip(
                                entry_notional=position.filled_entry_notional,
                                exit_notional=exit_price * position.quantity,
                                fee_bps=config.fee_bps,
                            ),
                            slippage_usdt=slippage_usdt,
                        )
                    ],
                    updated_open_trade=None,
                )

    if config.exit_policy == "fixed_tp_time_stop" and (current_unrealized / risk_unit) >= config.fixed_take_profit_r:
        exit_price, slippage_usdt = _apply_adverse_slippage(
            side=position.side,
            base_price=candle_close,
            quantity=position.quantity,
            slippage_bps=config.slippage_bps,
        )
        return LocalExitEvaluation(
            trade_records=[
                _build_trade_record(
                    open_trade=open_trade,
                    config=config,
                    exit_reason="FIXED_TAKE_PROFIT",
                    exit_contract_price=exit_price,
                    exit_mark_price=candle_close,
                    closed_at_ms=candle_close_time,
                    fees_usdt=_fees_for_round_trip(
                        entry_notional=position.filled_entry_notional,
                        exit_notional=exit_price * position.quantity,
                        fee_bps=config.fee_bps,
                    ),
                    slippage_usdt=slippage_usdt,
                )
            ],
            updated_open_trade=None,
        )

    partial_records: list[TradeRecord] = []
    if (
        config.exit_policy == "partial_tp_runner"
        and not open_trade.partial_take_profit_taken
        and (current_unrealized / risk_unit) >= config.partial_take_profit_r
    ):
        partial_fraction = min(max(config.partial_take_profit_fraction, 0.0), 1.0)
        if partial_fraction >= 1.0:
            partial_fraction = 1.0
        if partial_fraction > 0.0:
            partial_exit, remaining_trade = _partial_take_profit(
                open_trade=open_trade,
                config=config,
                candle_close=candle_close,
                candle_close_time=candle_close_time,
                candle_high=candle_high,
                candle_low=candle_low,
                fraction=partial_fraction,
            )
            partial_records.append(partial_exit)
            if remaining_trade is None:
                return LocalExitEvaluation(trade_records=partial_records, updated_open_trade=None)
            open_trade = remaining_trade
            position = open_trade.position
            risk_unit = max(open_trade.initial_risk_usdt, 1e-9)
            current_unrealized = (
                (candle_close - position.entry_contract_price_avg) * position.quantity
                if position.side == "LONG"
                else (position.entry_contract_price_avg - candle_close) * position.quantity
            )

    if config.exit_policy == "atr_trail":
        atr_value = execution_snapshot.atr_14
        if atr_value is not None and atr_trail_activation_reached(
            side=position.side,
            entry_contract_price_avg=position.entry_contract_price_avg,
            quantity=position.quantity,
            leverage_at_entry=position.leverage_at_entry,
            highest_high=open_trade.highest_high,
            lowest_low=open_trade.lowest_low,
            bars_held=open_trade.bars_held,
            min_bars=config.atr_trail_min_bars,
            min_profit_r=config.atr_trail_activation_profit_r,
        ):
            trail = atr_trail_price(
                side=position.side,
                highest_high=open_trade.highest_high,
                lowest_low=open_trade.lowest_low,
                atr_value=float(atr_value),
                atr_trailing_multiplier=config.atr_trailing_multiplier,
            )
            open_trade.atr_trail_history.append(
                {"ts": _iso_ms(candle_close_time), "trail_price_contract": trail}
            )
            if (
                (position.side == "LONG" and candle_close <= trail)
                or (position.side == "SHORT" and candle_close >= trail)
            ):
                exit_price, slippage_usdt = _apply_adverse_slippage(
                    side=position.side,
                    base_price=candle_close,
                    quantity=position.quantity,
                    slippage_bps=config.slippage_bps,
                )
                return LocalExitEvaluation(
                    trade_records=partial_records
                    + [
                        _build_trade_record(
                            open_trade=open_trade,
                            config=config,
                            exit_reason="ATR_TRAIL_EXIT",
                            exit_contract_price=exit_price,
                            exit_mark_price=candle_close,
                            closed_at_ms=candle_close_time,
                            fees_usdt=_fees_for_round_trip(
                                entry_notional=position.filled_entry_notional,
                                exit_notional=exit_price * position.quantity,
                                fee_bps=config.fee_bps,
                            ),
                            slippage_usdt=slippage_usdt,
                        )
                    ],
                    updated_open_trade=None,
                )

    if open_trade.bars_held >= config.max_holding_bars:
        exit_price, slippage_usdt = _apply_adverse_slippage(
            side=position.side,
            base_price=candle_close,
            quantity=position.quantity,
            slippage_bps=config.slippage_bps,
        )
        return LocalExitEvaluation(
            trade_records=partial_records
            + [
                _build_trade_record(
                    open_trade=open_trade,
                    config=config,
                    exit_reason="TIME_STOP",
                    exit_contract_price=exit_price,
                    exit_mark_price=candle_close,
                    closed_at_ms=candle_close_time,
                    fees_usdt=_fees_for_round_trip(
                        entry_notional=position.filled_entry_notional,
                        exit_notional=exit_price * position.quantity,
                        fee_bps=config.fee_bps,
                    ),
                    slippage_usdt=slippage_usdt,
                )
            ],
            updated_open_trade=None,
        )

    return LocalExitEvaluation(trade_records=partial_records, updated_open_trade=open_trade)


def _build_trade_record(
    *,
    open_trade: OpenTradeState,
    config: BacktestConfig,
    exit_reason: str,
    exit_contract_price: float,
    exit_mark_price: float,
    closed_at_ms: int,
    fees_usdt: float,
    slippage_usdt: float,
    trade_id_suffix: str = "",
) -> TradeRecord:
    policy = PolicyVersionInfo(
        policy_version=config.policy_version,
        strategy_version=config.strategy_version,
        feature_schema_version=config.feature_schema_version,
        model_base=open_trade.model_base,
        adapter_version=open_trade.adapter_version,
        dataset_version=config.dataset_version,
    )
    record = TradeRecord.from_closed_position(
        trade_id=f"bt-{open_trade.opened_at_ms}-{closed_at_ms}{trade_id_suffix}",
        opened_at=_dt_ms(open_trade.opened_at_ms),
        closed_at=_dt_ms(closed_at_ms),
        position=open_trade.position,
        policy=policy,
        exit_reason=exit_reason,
        exit_contract_price_avg=exit_contract_price,
        exit_mark_price=exit_mark_price,
        max_favorable_excursion_usdt=open_trade.max_favorable_excursion_usdt,
        max_adverse_excursion_usdt=open_trade.max_adverse_excursion_usdt,
        fees_usdt=fees_usdt,
        slippage_usdt=slippage_usdt,
        signal_reason_codes=open_trade.signal_reason_codes,
        ai_snapshot=open_trade.ai_snapshot,
        atr_trail_history=open_trade.atr_trail_history or [],
        notes="backtest_trade",
    )
    validate_trade_record(record.to_dict())
    return record


def _partial_take_profit(
    *,
    open_trade: OpenTradeState,
    config: BacktestConfig,
    candle_close: float,
    candle_close_time: int,
    candle_high: float,
    candle_low: float,
    fraction: float,
) -> tuple[TradeRecord, OpenTradeState | None]:
    close_fraction = min(max(fraction, 0.0), 1.0)
    remaining_fraction = 1.0 - close_fraction
    exit_price, slippage_usdt = _apply_adverse_slippage(
        side=open_trade.position.side,
        base_price=candle_close,
        quantity=open_trade.position.quantity * close_fraction,
        slippage_bps=config.slippage_bps,
    )
    partial_position = replace(
        open_trade.position,
        quantity=open_trade.position.quantity * close_fraction,
    )
    partial_trade = _build_trade_record(
        open_trade=replace(
            open_trade,
            position=partial_position,
            max_favorable_excursion_usdt=open_trade.max_favorable_excursion_usdt * close_fraction,
            max_adverse_excursion_usdt=open_trade.max_adverse_excursion_usdt * close_fraction,
        ),
        config=config,
        exit_reason="PARTIAL_TAKE_PROFIT",
        exit_contract_price=exit_price,
        exit_mark_price=candle_close,
        closed_at_ms=candle_close_time,
        fees_usdt=_fees_for_round_trip(
            entry_notional=partial_position.filled_entry_notional,
            exit_notional=exit_price * partial_position.quantity,
            fee_bps=config.fee_bps,
        ),
        slippage_usdt=slippage_usdt,
        trade_id_suffix="-partial",
    )
    if remaining_fraction <= 0.0:
        return partial_trade, None

    remaining_position = replace(
        open_trade.position,
        quantity=open_trade.position.quantity * remaining_fraction,
    )
    remaining_trade = replace(
        open_trade,
        position=remaining_position,
        highest_high=candle_high,
        lowest_low=candle_low,
        initial_risk_usdt=open_trade.initial_risk_usdt * remaining_fraction,
        max_favorable_excursion_usdt=open_trade.max_favorable_excursion_usdt * remaining_fraction,
        max_adverse_excursion_usdt=open_trade.max_adverse_excursion_usdt * remaining_fraction,
        partial_take_profit_taken=True,
    )
    return partial_trade, remaining_trade


def _fees_for_round_trip(*, entry_notional: float, exit_notional: float, fee_bps: float) -> float:
    return (entry_notional * fee_bps / 10000.0) + (exit_notional * fee_bps / 10000.0)


def _worse_contract_exit_price(*, side: str, threshold_price: float, execution_close: float) -> float:
    if side == "LONG":
        return min(threshold_price, execution_close)
    return max(threshold_price, execution_close)


def _apply_adverse_slippage(
    *,
    side: str,
    base_price: float,
    quantity: float,
    slippage_bps: float,
) -> tuple[float, float]:
    if side == "LONG":
        slipped = base_price * (1 - (slippage_bps / 10000.0))
    else:
        slipped = base_price * (1 + (slippage_bps / 10000.0))
    slippage_usdt = abs(base_price - slipped) * quantity
    return slipped, slippage_usdt


def _compute_metrics(trade_records: list[TradeRecord]) -> BacktestMetrics:
    trades = len(trade_records)
    wins = 0
    losses = 0
    positive = 0.0
    negative = 0.0
    total_realized = 0.0
    total_fees = 0.0
    total_slippage = 0.0
    total_after_fees = 0.0
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0

    for trade in trade_records:
        pnl_after_fees = trade.realized_pnl_after_fees_usdt or (
            trade.realized_pnl_usdt - trade.fees_usdt
        )
        total_realized += trade.realized_pnl_usdt
        total_fees += trade.fees_usdt
        total_slippage += trade.slippage_usdt
        total_after_fees += pnl_after_fees
        if pnl_after_fees > 0:
            wins += 1
            positive += pnl_after_fees
        elif pnl_after_fees < 0:
            losses += 1
            negative += abs(pnl_after_fees)
        equity += pnl_after_fees
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    profit_factor = positive / negative if negative > 0 else (positive if positive > 0 else 0.0)
    win_rate = (wins / trades) if trades else 0.0
    return BacktestMetrics(
        trades=trades,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        total_realized_pnl_usdt=total_realized,
        total_fees_usdt=total_fees,
        total_slippage_usdt=total_slippage,
        total_pnl_after_fees_usdt=total_after_fees,
        profit_factor=profit_factor,
        max_drawdown_usdt=max_drawdown,
    )


def _build_decision_report(
    metrics: BacktestMetrics, config: BacktestConfig
) -> DecisionExplanationReport:
    reasons: list[str] = []
    recommendation = "ACCEPT"

    if metrics.trades < config.min_trades_for_decision:
        recommendation = "REJECT"
        reasons.append("rejected_too_few_trades")
    if metrics.total_pnl_after_fees_usdt <= 0:
        recommendation = "REJECT"
        reasons.append("rejected_negative_total_pnl")
    if metrics.profit_factor < config.min_profit_factor:
        recommendation = "REJECT"
        reasons.append("rejected_profit_factor_below_min")
    if metrics.max_drawdown_usdt > config.max_allowed_drawdown_usdt:
        recommendation = "REJECT"
        reasons.append("rejected_drawdown_too_high")

    if recommendation == "ACCEPT":
        reasons.extend(
            [
                "accepted_positive_total_pnl",
                "accepted_profit_factor",
                "accepted_drawdown_ok",
            ]
        )

    explanation = (
        f"Recommendation={recommendation}; trades={metrics.trades}; "
        f"pnl_after_fees={metrics.total_pnl_after_fees_usdt:.4f}; "
        f"profit_factor={metrics.profit_factor:.4f}; "
        f"max_drawdown={metrics.max_drawdown_usdt:.4f}; "
        f"reasons={','.join(reasons)}"
    )
    return DecisionExplanationReport(
        recommendation=recommendation,
        reasons=reasons,
        metrics=metrics.to_dict(),
        explanation=explanation,
    )


def _dt_ms(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)


def _iso_ms(value: int) -> str:
    return _dt_ms(value).isoformat()


def _normalize_kline_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": row.get("dataset", "contract_klines"),
        "symbol": row.get("symbol", "BTCUSDT"),
        "interval": row.get("interval"),
        "open_time": int(row["open_time"]),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row.get("volume", 0.0)),
        "close_time": int(row["close_time"]),
        "quote_asset_volume": float(row.get("quote_asset_volume", 0.0)),
        "number_of_trades": int(row.get("number_of_trades", 0)),
        "taker_buy_base_asset_volume": float(row.get("taker_buy_base_asset_volume", 0.0)),
        "taker_buy_quote_asset_volume": float(row.get("taker_buy_quote_asset_volume", 0.0)),
        "ignore": str(row.get("ignore", "0")),
        "collected_at_ms": int(row.get("collected_at_ms", 0)),
    }


def _timeframe_ms(value: str) -> int:
    if value not in _TIMEFRAME_MS:
        raise ValueError(f"unsupported timeframe: {value}")
    return _TIMEFRAME_MS[value]


def _build_snapshot_cache(
    *,
    contract_candles_by_timeframe: dict[str, list[dict[str, Any]]],
    feature_lookback_candles: int,
) -> dict[str, list[TimeframeFeatureSnapshot]]:
    cache: dict[str, list[TimeframeFeatureSnapshot]] = {}
    for timeframe, candles in contract_candles_by_timeframe.items():
        builder = _RollingSnapshotBuilder(
            timeframe,
            feature_lookback_candles=feature_lookback_candles,
        )
        snapshots: list[TimeframeFeatureSnapshot] = []
        for candle in sorted(candles, key=lambda row: int(row["open_time"])):
            values = builder.update(candle)
            snapshots.append(
                TimeframeFeatureSnapshot(
                    timeframe=timeframe,
                    last_open_time=values["last_open_time"],
                    candle_count=values["candle_count"],
                    last_close=values["last_close"],
                    current_volume=values["current_volume"],
                    taker_buy_ratio=values["taker_buy_ratio"],
                    ema_fast_9=values["ema_fast_9"],
                    ema_slow_21=values["ema_slow_21"],
                    rsi_14=values["rsi_14"],
                    atr_14=values["atr_14"],
                    cumulative_vwap=values["cumulative_vwap"],
                    swing_high_20=values["swing_high_20"],
                    swing_low_20=values["swing_low_20"],
                    roc_5=values["roc_5"],
                    volume_sma_20=values["volume_sma_20"],
                    volume_ratio_20=values["volume_ratio_20"],
                )
            )
        cache[timeframe] = snapshots
    return cache


def _snapshot_at_or_before(
    *,
    snapshot_cache: dict[str, list[TimeframeFeatureSnapshot]],
    snapshot_cursor_positions: dict[str, int],
    symbol: str,
    feature_schema_version: str,
    end_open_time: int,
) -> FeatureSnapshot | None:
    snapshots_by_timeframe: dict[str, TimeframeFeatureSnapshot] = {}
    for timeframe, snapshots in snapshot_cache.items():
        cursor = snapshot_cursor_positions.get(timeframe, -1)
        while cursor + 1 < len(snapshots) and snapshots[cursor + 1].last_open_time <= end_open_time:
            cursor += 1
        snapshot_cursor_positions[timeframe] = cursor
        if cursor < 0:
            return None
        snapshots_by_timeframe[timeframe] = snapshots[cursor]
    return FeatureSnapshot(
        feature_schema_version=feature_schema_version,
        symbol=symbol,
        generated_from="contract_price",
        timeframes=snapshots_by_timeframe,
    )
