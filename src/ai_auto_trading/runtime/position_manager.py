from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_auto_trading.features.snapshot import FeatureBuilder
from ai_auto_trading.runtime.testnet import ManagedTradeState
from ai_auto_trading.strategy.trade_management import (
    atr_trail_activation_reached,
    initial_risk_usdt,
    atr_trail_price,
)


@dataclass(frozen=True)
class ManagedExitDecision:
    action: str
    exit_reason: str | None = None
    exit_contract_price: float | None = None
    updated_trade_state: ManagedTradeState | None = None


def evaluate_managed_trade_exit(
    *,
    trade_state: ManagedTradeState,
    execution_candles: list[dict[str, Any]],
    latest_allowed_close_time_ms: int | None = None,
    atr_trail_activation_profit_r: float = 0.5,
    atr_trail_min_bars: int = 2,
) -> ManagedExitDecision:
    if not execution_candles:
        return ManagedExitDecision(action="NO_ACTION", updated_trade_state=trade_state)

    ordered = sorted(execution_candles, key=lambda row: int(row["open_time"]))
    builder = FeatureBuilder()
    current = trade_state
    for candle in ordered:
        close_time = int(candle["close_time"])
        if (
            latest_allowed_close_time_ms is not None
            and close_time > latest_allowed_close_time_ms
        ):
            continue
        if close_time <= current.opened_at_ms:
            continue
        if (
            current.last_processed_candle_close_time_ms is not None
            and close_time <= current.last_processed_candle_close_time_ms
        ):
            continue
        current = _advance_trade_state(current, candle)
        snapshot = builder.build_timeframe_snapshot(
            timeframe=current.execution_timeframe,
            candles=[row for row in ordered if int(row["close_time"]) <= close_time],
        )
        risk_unit = max(
            initial_risk_usdt(
                entry_contract_price_avg=current.entry_contract_price_avg,
                quantity=current.quantity,
                leverage_at_entry=current.leverage_at_entry,
            ),
            1e-9,
        )
        current_unrealized = (
            (float(candle["close"]) - current.entry_contract_price_avg) * current.quantity
            if current.side == "LONG"
            else (current.entry_contract_price_avg - float(candle["close"])) * current.quantity
        )
        if (
            current.exit_policy == "fixed_tp_time_stop"
            and (current_unrealized / risk_unit) >= current.fixed_take_profit_r
        ):
            return ManagedExitDecision(
                action="EXIT",
                exit_reason="FIXED_TAKE_PROFIT",
                exit_contract_price=float(candle["close"]),
                updated_trade_state=current,
            )
        if (
            current.exit_policy == "atr_trail"
            and snapshot.atr_14 is not None
            and atr_trail_activation_reached(
            side=current.side,
            entry_contract_price_avg=current.entry_contract_price_avg,
            quantity=current.quantity,
            leverage_at_entry=current.leverage_at_entry,
            highest_high=current.highest_high,
            lowest_low=current.lowest_low,
            bars_held=current.bars_held,
            min_bars=atr_trail_min_bars,
            min_profit_r=atr_trail_activation_profit_r,
            )
        ):
            trail = _trail_price(current, atr_value=float(snapshot.atr_14))
            atr_history = list(current.atr_trail_history)
            atr_history.append(
                {"ts": close_time, "trail_price_contract": trail}
            )
            current = ManagedTradeState(
                symbol=current.symbol,
                side=current.side,
                quantity=current.quantity,
                leverage_at_entry=current.leverage_at_entry,
                entry_contract_price_avg=current.entry_contract_price_avg,
                entry_mark_price=current.entry_mark_price,
                execution_timeframe=current.execution_timeframe,
                atr_trailing_multiplier=current.atr_trailing_multiplier,
                max_holding_bars=current.max_holding_bars,
                opened_at_ms=current.opened_at_ms,
                signal_reason_codes=current.signal_reason_codes,
                model_base=current.model_base,
                adapter_version=current.adapter_version,
                ai_snapshot=current.ai_snapshot,
                bars_held=current.bars_held,
                highest_high=current.highest_high,
                lowest_low=current.lowest_low,
                last_processed_candle_close_time_ms=current.last_processed_candle_close_time_ms,
                atr_trail_history=atr_history,
                exit_policy=current.exit_policy,
                fixed_take_profit_r=current.fixed_take_profit_r,
            )
            close_price = float(candle["close"])
            if current.side == "LONG" and close_price <= trail:
                return ManagedExitDecision(
                    action="EXIT",
                    exit_reason="ATR_TRAIL_EXIT",
                    exit_contract_price=close_price,
                    updated_trade_state=current,
                )
            if current.side == "SHORT" and close_price >= trail:
                return ManagedExitDecision(
                    action="EXIT",
                    exit_reason="ATR_TRAIL_EXIT",
                    exit_contract_price=close_price,
                    updated_trade_state=current,
                )

        if current.bars_held >= current.max_holding_bars:
            return ManagedExitDecision(
                action="EXIT",
                exit_reason="TIME_STOP",
                exit_contract_price=float(candle["close"]),
                updated_trade_state=current,
            )

    return ManagedExitDecision(action="NO_ACTION", updated_trade_state=current)


def _advance_trade_state(
    trade_state: ManagedTradeState,
    candle: dict[str, Any],
) -> ManagedTradeState:
    return ManagedTradeState(
        symbol=trade_state.symbol,
        side=trade_state.side,
        quantity=trade_state.quantity,
        leverage_at_entry=trade_state.leverage_at_entry,
        entry_contract_price_avg=trade_state.entry_contract_price_avg,
        entry_mark_price=trade_state.entry_mark_price,
        execution_timeframe=trade_state.execution_timeframe,
        atr_trailing_multiplier=trade_state.atr_trailing_multiplier,
        max_holding_bars=trade_state.max_holding_bars,
        opened_at_ms=trade_state.opened_at_ms,
        signal_reason_codes=trade_state.signal_reason_codes,
        model_base=trade_state.model_base,
        adapter_version=trade_state.adapter_version,
        ai_snapshot=trade_state.ai_snapshot,
        bars_held=trade_state.bars_held + 1,
        highest_high=max(trade_state.highest_high, float(candle["high"])),
        lowest_low=min(trade_state.lowest_low, float(candle["low"])),
        last_processed_candle_close_time_ms=int(candle["close_time"]),
        atr_trail_history=list(trade_state.atr_trail_history),
        exit_policy=trade_state.exit_policy,
        fixed_take_profit_r=trade_state.fixed_take_profit_r,
    )


def _trail_price(trade_state: ManagedTradeState, *, atr_value: float) -> float:
    return atr_trail_price(
        side=trade_state.side,
        highest_high=trade_state.highest_high,
        lowest_low=trade_state.lowest_low,
        atr_value=atr_value,
        atr_trailing_multiplier=trade_state.atr_trailing_multiplier,
    )
