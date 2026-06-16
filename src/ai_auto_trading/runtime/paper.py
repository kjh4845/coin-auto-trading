from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ai_auto_trading.ai.gate import AIGateConfig, AITradeAssistant, apply_ai_entry_gate
from ai_auto_trading.backtest.replay import write_trade_logs_jsonl
from ai_auto_trading.features.snapshot import FeatureSnapshot, TimeframeFeatureSnapshot
from ai_auto_trading.models import PolicyVersionInfo, PositionState, TradeRecord
from ai_auto_trading.risk.hard_stop import evaluate_hard_stop
from ai_auto_trading.strategy.rule_based import (
    RuleStrategyContext,
    RuleStrategyParameters,
    evaluate_rule_signal,
)
from ai_auto_trading.strategy.trade_management import (
    atr_trail_activation_reached,
    atr_trail_price,
)


@dataclass(frozen=True)
class PaperRuntimeConfig:
    symbol: str = "BTCUSDT"
    execution_timeframe: str = "3m"
    leverage_at_entry: float = 2.0
    entry_notional_usdt: float = 1000.0
    atr_trailing_multiplier: float = 2.5
    atr_trail_activation_profit_r: float = 0.5
    atr_trail_min_bars: int = 2
    max_holding_bars: int = 8
    policy_version: str = "policy_v1"
    strategy_version: str = "strategy_v1"
    feature_schema_version: str = "features_v1"
    model_base: str = "rule_only"
    adapter_version: str | None = None
    dataset_version: str | None = None


@dataclass(frozen=True)
class PaperFeedEvent:
    event_time_ms: int
    snapshot: FeatureSnapshot
    execution_candle: dict[str, Any]
    current_mark_price: float


@dataclass
class OpenPaperTrade:
    position: PositionState
    opened_at_ms: int
    signal_reason_codes: list[str]
    model_base: str = "rule_only"
    adapter_version: str | None = None
    ai_snapshot: dict[str, Any] | None = None
    bars_held: int = 0
    highest_high: float = 0.0
    lowest_low: float = 0.0
    max_favorable_excursion_usdt: float = 0.0
    max_adverse_excursion_usdt: float = 0.0
    atr_trail_history: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.atr_trail_history is None:
            self.atr_trail_history = []


@dataclass(frozen=True)
class PaperRuntimeResult:
    trade_records: list[TradeRecord]
    open_position: bool


class PaperTradingRuntime:
    def __init__(
        self,
        config: PaperRuntimeConfig | None = None,
        strategy_params: RuleStrategyParameters | None = None,
        ai_trade_assistant: AITradeAssistant | None = None,
        ai_gate_config: AIGateConfig | None = None,
    ) -> None:
        self.config = config or PaperRuntimeConfig()
        self.strategy_params = strategy_params or RuleStrategyParameters(
            execution_timeframe=self.config.execution_timeframe,
        )
        self.ai_trade_assistant = ai_trade_assistant
        self.ai_gate_config = ai_gate_config or AIGateConfig()
        self._open_trade: OpenPaperTrade | None = None
        self.trade_records: list[TradeRecord] = []

    def process_event(self, event: PaperFeedEvent) -> str:
        action = "NO_ACTION"
        if self._open_trade is not None:
            self._update_excursions(self._open_trade, event.execution_candle)
            hard_stop_record = self._maybe_hard_stop_exit(event)
            if hard_stop_record is not None:
                self.trade_records.append(hard_stop_record)
                self._open_trade = None
                return "HARD_STOP_EXIT"

            local_exit_record = self._maybe_local_exit(event)
            if local_exit_record is not None:
                self.trade_records.append(local_exit_record)
                self._open_trade = None
                return local_exit_record.exit_reason

        if self._open_trade is None:
            rule_decision = evaluate_rule_signal(
                RuleStrategyContext(snapshot=event.snapshot),
                params=self.strategy_params,
            )
            gated_decision = apply_ai_entry_gate(
                rule_decision=rule_decision,
                snapshot=event.snapshot,
                assistant=self.ai_trade_assistant,
                gate_config=self.ai_gate_config,
            )
            if gated_decision.action in {"LONG", "SHORT"}:
                self._open_trade = self._open_new_trade(event, gated_decision)
                action = f"OPEN_{gated_decision.action}"
        return action

    def run_feed(self, events: Iterable[PaperFeedEvent]) -> PaperRuntimeResult:
        for event in events:
            self.process_event(event)
        return PaperRuntimeResult(
            trade_records=list(self.trade_records),
            open_position=self._open_trade is not None,
        )

    def export_trade_logs(self, output_path: Path) -> Path:
        return write_trade_logs_jsonl(self.trade_records, output_path)

    def _open_new_trade(
        self,
        event: PaperFeedEvent,
        decision,
    ) -> OpenPaperTrade:
        execution_close = float(event.execution_candle["close"])
        quantity = (self.config.entry_notional_usdt * decision.size_multiplier) / execution_close
        position = PositionState(
            side=decision.action,
            quantity=quantity,
            leverage_at_entry=self.config.leverage_at_entry,
            entry_contract_price_avg=execution_close,
            entry_mark_price=event.current_mark_price,
            symbol=self.config.symbol,
        )
        return OpenPaperTrade(
            position=position,
            opened_at_ms=event.event_time_ms,
            signal_reason_codes=decision.reason_codes,
            model_base=decision.model_base,
            adapter_version=decision.adapter_version,
            ai_snapshot=decision.ai_snapshot,
            highest_high=float(event.execution_candle["high"]),
            lowest_low=float(event.execution_candle["low"]),
        )

    def _maybe_hard_stop_exit(self, event: PaperFeedEvent) -> TradeRecord | None:
        assert self._open_trade is not None
        evaluation = evaluate_hard_stop(
            self._open_trade.position, current_mark_price=event.current_mark_price
        )
        if not evaluation.triggered:
            return None
        return self._build_trade_record(
            exit_reason="HARD_STOP_MARK_PRICE",
            exit_contract_price=float(event.execution_candle["close"]),
            exit_mark_price=event.current_mark_price,
            closed_at_ms=event.event_time_ms,
        )

    def _maybe_local_exit(self, event: PaperFeedEvent) -> TradeRecord | None:
        assert self._open_trade is not None
        self._open_trade.bars_held += 1
        execution_features = event.snapshot.timeframes[self.config.execution_timeframe]
        atr_exit = self._atr_exit_price(execution_features)
        if atr_exit is not None:
            execution_close = float(event.execution_candle["close"])
            if self._open_trade.position.side == "LONG" and execution_close <= atr_exit:
                return self._build_trade_record(
                    exit_reason="ATR_TRAIL_EXIT",
                    exit_contract_price=execution_close,
                    exit_mark_price=event.current_mark_price,
                    closed_at_ms=event.event_time_ms,
                )
            if self._open_trade.position.side == "SHORT" and execution_close >= atr_exit:
                return self._build_trade_record(
                    exit_reason="ATR_TRAIL_EXIT",
                    exit_contract_price=execution_close,
                    exit_mark_price=event.current_mark_price,
                    closed_at_ms=event.event_time_ms,
                )

        if self._open_trade.bars_held >= self.config.max_holding_bars:
            return self._build_trade_record(
                exit_reason="TIME_STOP",
                exit_contract_price=float(event.execution_candle["close"]),
                exit_mark_price=event.current_mark_price,
                closed_at_ms=event.event_time_ms,
            )
        return None

    def _atr_exit_price(
        self, execution_features: TimeframeFeatureSnapshot
    ) -> float | None:
        assert self._open_trade is not None
        if execution_features.atr_14 is None:
            return None
        if not atr_trail_activation_reached(
            side=self._open_trade.position.side,
            entry_contract_price_avg=self._open_trade.position.entry_contract_price_avg,
            quantity=self._open_trade.position.quantity,
            leverage_at_entry=self._open_trade.position.leverage_at_entry,
            highest_high=self._open_trade.highest_high,
            lowest_low=self._open_trade.lowest_low,
            bars_held=self._open_trade.bars_held,
            min_bars=self.config.atr_trail_min_bars,
            min_profit_r=self.config.atr_trail_activation_profit_r,
        ):
            return None
        trail = atr_trail_price(
            side=self._open_trade.position.side,
            highest_high=self._open_trade.highest_high,
            lowest_low=self._open_trade.lowest_low,
            atr_value=execution_features.atr_14,
            atr_trailing_multiplier=self.config.atr_trailing_multiplier,
        )
        self._open_trade.atr_trail_history.append(
            {"ts": _iso_ms(execution_features.last_open_time), "trail_price_contract": trail}
        )
        return trail

    def _update_excursions(
        self, trade: OpenPaperTrade, candle: dict[str, Any]
    ) -> None:
        high = float(candle["high"])
        low = float(candle["low"])
        trade.highest_high = max(trade.highest_high, high)
        trade.lowest_low = min(trade.lowest_low, low)
        entry_price = trade.position.entry_contract_price_avg
        quantity = trade.position.quantity
        if trade.position.side == "LONG":
            favorable = max(0.0, (high - entry_price) * quantity)
            adverse = max(0.0, (entry_price - low) * quantity)
        else:
            favorable = max(0.0, (entry_price - low) * quantity)
            adverse = max(0.0, (high - entry_price) * quantity)
        trade.max_favorable_excursion_usdt = max(
            trade.max_favorable_excursion_usdt, favorable
        )
        trade.max_adverse_excursion_usdt = max(
            trade.max_adverse_excursion_usdt, adverse
        )

    def _build_trade_record(
        self,
        *,
        exit_reason: str,
        exit_contract_price: float,
        exit_mark_price: float,
        closed_at_ms: int,
    ) -> TradeRecord:
        assert self._open_trade is not None
        policy = PolicyVersionInfo(
            policy_version=self.config.policy_version,
            strategy_version=self.config.strategy_version,
            feature_schema_version=self.config.feature_schema_version,
            model_base=self._open_trade.model_base,
            adapter_version=self._open_trade.adapter_version,
            dataset_version=self.config.dataset_version,
        )
        fees = 0.0
        slippage = 0.0
        return TradeRecord.from_closed_position(
            trade_id=f"paper-{self._open_trade.opened_at_ms}-{closed_at_ms}",
            opened_at=_dt_ms(self._open_trade.opened_at_ms),
            closed_at=_dt_ms(closed_at_ms),
            position=self._open_trade.position,
            policy=policy,
            exit_reason=exit_reason,
            exit_contract_price_avg=exit_contract_price,
            exit_mark_price=exit_mark_price,
            max_favorable_excursion_usdt=self._open_trade.max_favorable_excursion_usdt,
            max_adverse_excursion_usdt=self._open_trade.max_adverse_excursion_usdt,
            fees_usdt=fees,
            slippage_usdt=slippage,
            signal_reason_codes=self._open_trade.signal_reason_codes,
            ai_snapshot=self._open_trade.ai_snapshot,
            atr_trail_history=self._open_trade.atr_trail_history or [],
            notes="paper_trade",
        )


def _dt_ms(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)


def _iso_ms(value: int) -> str:
    return _dt_ms(value).isoformat()
