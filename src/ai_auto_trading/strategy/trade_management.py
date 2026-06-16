from __future__ import annotations


def initial_risk_usdt(
    *,
    entry_contract_price_avg: float,
    quantity: float,
    leverage_at_entry: float,
) -> float:
    entry_notional = entry_contract_price_avg * quantity
    entry_initial_margin = entry_notional / leverage_at_entry
    return entry_initial_margin * 0.05


def atr_trail_activation_reached(
    *,
    side: str,
    entry_contract_price_avg: float,
    quantity: float,
    leverage_at_entry: float,
    highest_high: float,
    lowest_low: float,
    bars_held: int,
    min_bars: int,
    min_profit_r: float,
) -> bool:
    if bars_held < min_bars:
        return False
    risk_usdt = initial_risk_usdt(
        entry_contract_price_avg=entry_contract_price_avg,
        quantity=quantity,
        leverage_at_entry=leverage_at_entry,
    )
    if risk_usdt <= 0.0:
        return False
    if side == "LONG":
        favorable_usdt = max(0.0, (highest_high - entry_contract_price_avg) * quantity)
    else:
        favorable_usdt = max(0.0, (entry_contract_price_avg - lowest_low) * quantity)
    return favorable_usdt >= (risk_usdt * min_profit_r)


def atr_trail_price(
    *,
    side: str,
    highest_high: float,
    lowest_low: float,
    atr_value: float,
    atr_trailing_multiplier: float,
) -> float:
    if side == "LONG":
        return highest_high - (atr_value * atr_trailing_multiplier)
    return lowest_low + (atr_value * atr_trailing_multiplier)
