from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


def _require_non_empty(name: str, value: str) -> None:
    if not value:
        raise ValueError(f"{name} must not be empty")


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _require_choice(name: str, value: str, allowed: set[str]) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}")


@dataclass(frozen=True)
class PolicyVersionInfo:
    policy_version: str
    strategy_version: str
    feature_schema_version: str
    model_base: str = "rule_only"
    adapter_version: str | None = None
    dataset_version: str | None = None
    runtime_version: str | None = None
    review_labeler_version: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty("policy_version", self.policy_version)
        _require_non_empty("strategy_version", self.strategy_version)
        _require_non_empty("feature_schema_version", self.feature_schema_version)
        _require_non_empty("model_base", self.model_base)


@dataclass(frozen=True)
class OrderIntent:
    side: str
    quantity: float
    order_type: str
    symbol: str = "BTCUSDT"
    reduce_only: bool = False
    working_type: str | None = None
    stop_price: float | None = None

    def __post_init__(self) -> None:
        _require_choice("side", self.side, {"BUY", "SELL"})
        _require_choice(
            "order_type",
            self.order_type,
            {"MARKET", "LIMIT", "STOP_MARKET"},
        )
        _require_positive("quantity", self.quantity)


@dataclass(frozen=True)
class PositionState:
    side: str
    quantity: float
    leverage_at_entry: float
    entry_contract_price_avg: float
    entry_mark_price: float
    symbol: str = "BTCUSDT"
    position_mode: str = "ONE_WAY"
    margin_mode: str = "ISOLATED"

    def __post_init__(self) -> None:
        _require_choice("side", self.side, {"LONG", "SHORT"})
        _require_positive("quantity", self.quantity)
        _require_positive("leverage_at_entry", self.leverage_at_entry)
        _require_positive("entry_contract_price_avg", self.entry_contract_price_avg)
        _require_positive("entry_mark_price", self.entry_mark_price)

    @property
    def filled_entry_notional(self) -> float:
        return self.entry_contract_price_avg * self.quantity

    @property
    def entry_initial_margin_fixed(self) -> float:
        return self.filled_entry_notional / self.leverage_at_entry

    @property
    def hard_stop_trigger_loss_usdt(self) -> float:
        return self.entry_initial_margin_fixed * 0.05

    @property
    def hard_stop_trigger_price_mark(self) -> float:
        price_delta = self.hard_stop_trigger_loss_usdt / self.quantity
        if self.side == "LONG":
            return self.entry_contract_price_avg - price_delta
        return self.entry_contract_price_avg + price_delta


@dataclass(frozen=True)
class TradeRecord:
    schema_version: str
    trade_id: str
    symbol: str
    side: str
    position_mode: str
    margin_mode: str
    opened_at: str
    closed_at: str
    entry_order_type: str
    exit_reason: str
    policy_version: str
    strategy_version: str
    feature_schema_version: str
    model_base: str
    adapter_version: str | None
    dataset_version: str | None
    entry_contract_price_avg: float
    entry_mark_price: float
    exit_contract_price_avg: float
    exit_mark_price: float
    filled_quantity: float
    leverage_at_entry: float
    filled_entry_notional: float
    entry_initial_margin_fixed: float
    hard_stop_trigger_loss_usdt: float
    hard_stop_trigger_price_mark: float
    hard_stop_working_type: str
    realized_pnl_usdt: float
    fees_usdt: float
    slippage_usdt: float
    max_favorable_excursion_usdt: float
    max_adverse_excursion_usdt: float
    runtime_version: str | None = None
    review_labeler_version: str | None = None
    realized_pnl_after_fees_usdt: float | None = None
    slippage_bps: float | None = None
    signal_reason_codes: list[str] = field(default_factory=list)
    ai_snapshot: dict[str, Any] | None = None
    atr_trail_history: list[dict[str, Any]] = field(default_factory=list)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_closed_position(
        cls,
        *,
        trade_id: str,
        opened_at: datetime,
        closed_at: datetime,
        position: PositionState,
        policy: PolicyVersionInfo,
        exit_reason: str,
        exit_contract_price_avg: float,
        exit_mark_price: float,
        max_favorable_excursion_usdt: float,
        max_adverse_excursion_usdt: float,
        entry_order_type: str = "MARKET",
        fees_usdt: float = 0.0,
        slippage_usdt: float = 0.0,
        signal_reason_codes: list[str] | None = None,
        ai_snapshot: dict[str, Any] | None = None,
        atr_trail_history: list[dict[str, Any]] | None = None,
        notes: str | None = None,
    ) -> "TradeRecord":
        _require_non_empty("trade_id", trade_id)
        _require_positive("exit_contract_price_avg", exit_contract_price_avg)
        _require_positive("exit_mark_price", exit_mark_price)
        _require_choice(
            "exit_reason",
            exit_reason,
            {
                "HARD_STOP_MARK_PRICE",
                "BREAK_EVEN_STOP_EXIT",
                "ATR_TRAIL_EXIT",
                "EARLY_FAIL_EXIT",
                "FIXED_TAKE_PROFIT",
                "PARTIAL_TAKE_PROFIT",
                "TIME_STOP",
                "AI_DEFENSIVE_EXIT",
                "MANUAL_EMERGENCY_EXIT",
                "SYSTEM_FAILSAFE_EXIT",
                "OTHER",
            },
        )

        direction = 1.0 if position.side == "LONG" else -1.0
        realized_pnl_usdt = (
            (exit_contract_price_avg - position.entry_contract_price_avg)
            * position.quantity
            * direction
        )
        realized_pnl_after_fees_usdt = realized_pnl_usdt - fees_usdt
        slippage_bps = (
            (slippage_usdt / position.filled_entry_notional) * 10000.0
            if position.filled_entry_notional > 0
            else None
        )

        return cls(
            schema_version="trade_log_v1",
            trade_id=trade_id,
            symbol=position.symbol,
            side=position.side,
            position_mode=position.position_mode,
            margin_mode=position.margin_mode,
            opened_at=opened_at.isoformat(),
            closed_at=closed_at.isoformat(),
            entry_order_type=entry_order_type,
            exit_reason=exit_reason,
            policy_version=policy.policy_version,
            strategy_version=policy.strategy_version,
            feature_schema_version=policy.feature_schema_version,
            model_base=policy.model_base,
            adapter_version=policy.adapter_version,
            dataset_version=policy.dataset_version,
            entry_contract_price_avg=position.entry_contract_price_avg,
            entry_mark_price=position.entry_mark_price,
            exit_contract_price_avg=exit_contract_price_avg,
            exit_mark_price=exit_mark_price,
            filled_quantity=position.quantity,
            leverage_at_entry=position.leverage_at_entry,
            filled_entry_notional=position.filled_entry_notional,
            entry_initial_margin_fixed=position.entry_initial_margin_fixed,
            hard_stop_trigger_loss_usdt=position.hard_stop_trigger_loss_usdt,
            hard_stop_trigger_price_mark=position.hard_stop_trigger_price_mark,
            hard_stop_working_type="MARK_PRICE",
            realized_pnl_usdt=realized_pnl_usdt,
            fees_usdt=fees_usdt,
            slippage_usdt=slippage_usdt,
            max_favorable_excursion_usdt=max_favorable_excursion_usdt,
            max_adverse_excursion_usdt=max_adverse_excursion_usdt,
            runtime_version=policy.runtime_version,
            review_labeler_version=policy.review_labeler_version,
            realized_pnl_after_fees_usdt=realized_pnl_after_fees_usdt,
            slippage_bps=slippage_bps,
            signal_reason_codes=signal_reason_codes or [],
            ai_snapshot=ai_snapshot,
            atr_trail_history=atr_trail_history or [],
            notes=notes,
        )
