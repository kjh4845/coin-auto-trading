from __future__ import annotations

from dataclasses import dataclass

from ai_auto_trading.models import OrderIntent, PositionState


@dataclass(frozen=True)
class HardStopEvaluation:
    symbol: str
    side: str
    current_mark_price: float
    unrealized_pnl_usdt: float
    trigger_loss_usdt: float
    triggered: bool


def mark_price_unrealized_pnl(
    position: PositionState, *, current_mark_price: float
) -> float:
    if current_mark_price <= 0:
        raise ValueError("current_mark_price must be > 0")
    direction = 1.0 if position.side == "LONG" else -1.0
    return (
        (current_mark_price - position.entry_contract_price_avg)
        * position.quantity
        * direction
    )


def evaluate_hard_stop(
    position: PositionState, *, current_mark_price: float
) -> HardStopEvaluation:
    unrealized = mark_price_unrealized_pnl(
        position, current_mark_price=current_mark_price
    )
    trigger_loss = position.hard_stop_trigger_loss_usdt
    return HardStopEvaluation(
        symbol=position.symbol,
        side=position.side,
        current_mark_price=current_mark_price,
        unrealized_pnl_usdt=unrealized,
        trigger_loss_usdt=trigger_loss,
        triggered=unrealized <= -trigger_loss,
    )


def build_exchange_hard_stop_intent(position: PositionState) -> OrderIntent:
    exit_side = "SELL" if position.side == "LONG" else "BUY"
    return OrderIntent(
        side=exit_side,
        quantity=position.quantity,
        order_type="STOP_MARKET",
        reduce_only=True,
        working_type="MARK_PRICE",
        stop_price=position.hard_stop_trigger_price_mark,
        symbol=position.symbol,
    )


def build_failsafe_market_exit_intent(position: PositionState) -> OrderIntent:
    exit_side = "SELL" if position.side == "LONG" else "BUY"
    return OrderIntent(
        side=exit_side,
        quantity=position.quantity,
        order_type="MARKET",
        reduce_only=True,
        symbol=position.symbol,
    )

