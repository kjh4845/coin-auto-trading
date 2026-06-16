from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_auto_trading.settings import Settings
from ai_auto_trading.strategy.rule_based import RuleStrategyParameters


@dataclass(frozen=True)
class RuntimeEntryProfile:
    name: str
    priority: int
    params: RuleStrategyParameters


def build_runtime_entry_profiles(settings: Settings) -> list[RuntimeEntryProfile]:
    if settings.strategy_mode == "best_pair_v1":
        return _best_pair_btc_profiles()
    if settings.strategy_mode == "multi_symbol_priority_v1":
        return _best_pair_btc_profiles()
    return [
        RuntimeEntryProfile(
            name="single_profile",
            priority=1,
            params=RuleStrategyParameters(
                execution_timeframe=settings.execution_timeframe,
                micro_timeframe=settings.micro_timeframe,
                confirmation_timeframe=settings.confirmation_timeframe,
                macro_timeframe=settings.macro_timeframe,
                regime_timeframe=settings.regime_timeframe,
                anchor_timeframe=settings.anchor_timeframe,
                max_long_funding_rate=settings.max_long_funding_rate,
                min_short_funding_rate=settings.min_short_funding_rate,
                min_long_taker_buy_ratio=settings.min_long_taker_buy_ratio,
                max_short_taker_buy_ratio=settings.max_short_taker_buy_ratio,
                allow_long_entries=settings.allow_long_entries,
                allow_short_entries=settings.allow_short_entries,
                min_micro_long_ema_spread_pct=settings.min_micro_long_ema_spread_pct,
                min_micro_short_ema_spread_pct=settings.min_micro_short_ema_spread_pct,
                min_higher_tf_ema_spread_pct=settings.min_higher_tf_ema_spread_pct,
                min_volume_ratio_20=settings.min_volume_ratio_20,
            ),
        )
    ]


def build_runtime_entry_profiles_for_symbol(
    settings: Settings,
    symbol: str,
) -> list[RuntimeEntryProfile]:
    if settings.strategy_mode != "multi_symbol_priority_v1":
        return build_runtime_entry_profiles(settings)
    normalized_symbol = symbol.upper()
    if normalized_symbol == "BTCUSDT":
        return _best_pair_btc_profiles()
    if normalized_symbol == "ETHUSDT":
        return [
            RuntimeEntryProfile(
                name="eth_medium_15m_both_pullback_time_stop_only",
                priority=1,
                params=RuleStrategyParameters(
                    execution_timeframe="15m",
                    micro_timeframe="30m",
                    confirmation_timeframe="1h",
                    macro_timeframe="4h",
                    regime_timeframe="1d",
                    anchor_timeframe=None,
                    use_confirmation=True,
                    use_macro=True,
                    use_regime=False,
                    use_anchor=False,
                    use_micro=True,
                    require_vwap=True,
                    require_rsi=False,
                    require_price_reclaim=True,
                    require_roc=False,
                    allow_long_entries=True,
                    allow_short_entries=True,
                    min_higher_tf_ema_spread_pct=0.0,
                    min_volume_ratio_20=0.0,
                    min_micro_long_ema_spread_pct=0.0015,
                    min_micro_short_ema_spread_pct=0.0017,
                    max_long_funding_rate=None,
                    min_short_funding_rate=None,
                    min_long_taker_buy_ratio=None,
                    max_short_taker_buy_ratio=None,
                ),
            )
        ]
    if normalized_symbol == "SOLUSDT":
        return [
            RuntimeEntryProfile(
                name="sol_medium_15m_both_momentum_time_stop_only",
                priority=1,
                params=RuleStrategyParameters(
                    execution_timeframe="15m",
                    micro_timeframe="30m",
                    confirmation_timeframe="1h",
                    macro_timeframe="4h",
                    regime_timeframe="1d",
                    anchor_timeframe=None,
                    use_confirmation=True,
                    use_macro=True,
                    use_regime=False,
                    use_anchor=False,
                    use_micro=True,
                    require_vwap=True,
                    require_rsi=True,
                    require_price_reclaim=True,
                    require_roc=True,
                    allow_long_entries=True,
                    allow_short_entries=True,
                    min_higher_tf_ema_spread_pct=0.0,
                    min_volume_ratio_20=0.0,
                    min_micro_long_ema_spread_pct=0.0005,
                    min_micro_short_ema_spread_pct=0.0011,
                    max_long_funding_rate=None,
                    min_short_funding_rate=None,
                    min_long_taker_buy_ratio=None,
                    max_short_taker_buy_ratio=None,
                ),
            )
        ]
    raise ValueError(f"unsupported multi-symbol priority symbol: {symbol}")


def describe_runtime_entry_profiles(settings: Settings) -> list[dict[str, Any]]:
    return [
        {
            "name": profile.name,
            "priority": profile.priority,
            "execution_timeframe": profile.params.execution_timeframe,
            "micro_timeframe": (
                profile.params.micro_timeframe
                if profile.params.use_micro
                else None
            ),
            "confirmation_timeframe": (
                profile.params.confirmation_timeframe
                if profile.params.use_confirmation
                else None
            ),
            "macro_timeframe": (
                profile.params.macro_timeframe
                if profile.params.use_macro
                else None
            ),
            "regime_timeframe": (
                profile.params.regime_timeframe
                if profile.params.use_regime
                else None
            ),
            "anchor_timeframe": (
                profile.params.anchor_timeframe
                if profile.params.use_anchor
                else None
            ),
            "allow_long_entries": profile.params.allow_long_entries,
            "allow_short_entries": profile.params.allow_short_entries,
            "active_conditions": _active_conditions(profile.params),
            "active_condition_count": len(_active_conditions(profile.params)),
        }
        for profile in build_runtime_entry_profiles(settings)
    ]


def runtime_strategy_context(settings: Settings, symbols: list[str] | None = None) -> dict[str, Any]:
    if symbols:
        return {
            "strategy_mode": settings.strategy_mode,
            "symbols": symbols,
            "profiles_by_symbol": {
                symbol: [
                    _describe_profile(profile)
                    for profile in build_runtime_entry_profiles_for_symbol(settings, symbol)
                ]
                for symbol in symbols
            },
            "runtime_stream_interval_by_symbol": {
                symbol: runtime_stream_interval(settings, symbol=symbol)
                for symbol in symbols
            },
        }
    profiles = describe_runtime_entry_profiles(settings)
    execution_timeframes = [profile["execution_timeframe"] for profile in profiles]
    higher_timeframes: list[str] = []
    for profile in profiles:
        for timeframe in (
            profile["micro_timeframe"],
            profile["confirmation_timeframe"],
            profile["macro_timeframe"],
            profile["regime_timeframe"],
            profile["anchor_timeframe"],
        ):
            if timeframe and timeframe not in higher_timeframes:
                higher_timeframes.append(timeframe)
    return {
        "strategy_mode": settings.strategy_mode,
        "runtime_stream_interval": runtime_stream_interval(settings),
        "execution_timeframes": execution_timeframes,
        "higher_timeframes": higher_timeframes,
        "profiles": profiles,
    }


def required_runtime_timeframes(settings: Settings, symbol: str | None = None) -> list[str]:
    output: list[str] = []
    profiles = (
        build_runtime_entry_profiles_for_symbol(settings, symbol)
        if symbol is not None
        else build_runtime_entry_profiles(settings)
    )
    for profile in profiles:
        for timeframe in profile.params.required_timeframes():
            if timeframe not in output:
                output.append(timeframe)
    return output


def runtime_stream_interval(settings: Settings, symbol: str | None = None) -> str:
    profiles = (
        build_runtime_entry_profiles_for_symbol(settings, symbol)
        if symbol is not None
        else build_runtime_entry_profiles(settings)
    )
    timeframes = [profile.params.execution_timeframe for profile in profiles]
    return min(timeframes, key=_timeframe_sort_key)


def _best_pair_btc_profiles() -> list[RuntimeEntryProfile]:
    return [
        RuntimeEntryProfile(
            name="short_best_5m",
            priority=2,
            params=RuleStrategyParameters(
                execution_timeframe="5m",
                micro_timeframe="15m",
                confirmation_timeframe="1h",
                macro_timeframe="4h",
                regime_timeframe="1d",
                anchor_timeframe=None,
                use_confirmation=True,
                use_macro=True,
                use_regime=True,
                use_anchor=False,
                use_micro=True,
                require_vwap=True,
                require_rsi=True,
                require_price_reclaim=True,
                require_roc=True,
                allow_long_entries=False,
                allow_short_entries=True,
                min_higher_tf_ema_spread_pct=0.0005,
                min_volume_ratio_20=1.0,
                min_micro_long_ema_spread_pct=0.0,
                min_micro_short_ema_spread_pct=0.0011,
                max_long_funding_rate=None,
                min_short_funding_rate=None,
                min_long_taker_buy_ratio=None,
                max_short_taker_buy_ratio=None,
            ),
        ),
        RuntimeEntryProfile(
            name="long_best_30m",
            priority=1,
            params=RuleStrategyParameters(
                execution_timeframe="30m",
                micro_timeframe="1h",
                confirmation_timeframe="4h",
                macro_timeframe="1d",
                regime_timeframe=None,
                anchor_timeframe=None,
                use_confirmation=True,
                use_macro=False,
                use_regime=False,
                use_anchor=False,
                use_micro=True,
                require_vwap=True,
                require_rsi=False,
                require_price_reclaim=True,
                require_roc=False,
                allow_long_entries=True,
                allow_short_entries=False,
                min_higher_tf_ema_spread_pct=0.0,
                min_volume_ratio_20=0.0,
                min_micro_long_ema_spread_pct=0.0005,
                min_micro_short_ema_spread_pct=0.0,
                max_long_funding_rate=None,
                min_short_funding_rate=None,
                min_long_taker_buy_ratio=None,
                max_short_taker_buy_ratio=None,
            ),
        ),
    ]


def _describe_profile(profile: RuntimeEntryProfile) -> dict[str, Any]:
    params = profile.params
    return {
        "name": profile.name,
        "priority": profile.priority,
        "execution_timeframe": params.execution_timeframe,
        "micro_timeframe": params.micro_timeframe if params.use_micro else None,
        "confirmation_timeframe": params.confirmation_timeframe if params.use_confirmation else None,
        "macro_timeframe": params.macro_timeframe if params.use_macro else None,
        "regime_timeframe": params.regime_timeframe if params.use_regime else None,
        "anchor_timeframe": params.anchor_timeframe if params.use_anchor else None,
        "allow_long_entries": params.allow_long_entries,
        "allow_short_entries": params.allow_short_entries,
        "active_conditions": _active_conditions(params),
        "active_condition_count": len(_active_conditions(params)),
    }


def _timeframe_sort_key(timeframe: str) -> tuple[int, int]:
    unit = timeframe[-1]
    magnitude = int(timeframe[:-1])
    unit_order = {"m": 0, "h": 1, "d": 2}
    return (unit_order.get(unit, 99), magnitude)


def _active_conditions(params: RuleStrategyParameters) -> list[str]:
    conditions: list[str] = []
    if params.use_confirmation and params.confirmation_timeframe:
        conditions.append("confirmation_trend")
    if params.use_macro and params.macro_timeframe:
        conditions.append("macro_trend")
    if params.use_regime and params.regime_timeframe:
        conditions.append("regime_trend")
    if params.use_anchor and params.anchor_timeframe:
        conditions.append("anchor_trend")
    if params.use_micro and params.micro_timeframe:
        conditions.append("micro_trend")
    if params.require_vwap:
        conditions.append("vwap_pullback")
    if params.require_rsi:
        conditions.append("rsi_reclaim")
    if params.require_price_reclaim:
        conditions.append("price_reclaim")
    if params.require_roc:
        conditions.append("roc_direction")
    if params.min_volume_ratio_20 > 0:
        conditions.append("volume_participation")
    if (
        params.max_long_funding_rate is not None
        or params.min_short_funding_rate is not None
    ):
        conditions.append("funding_filter")
    if (
        params.min_long_taker_buy_ratio is not None
        or params.max_short_taker_buy_ratio is not None
    ):
        conditions.append("taker_flow")
    return conditions
