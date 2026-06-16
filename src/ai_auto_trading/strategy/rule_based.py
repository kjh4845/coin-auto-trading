from __future__ import annotations

from dataclasses import dataclass, field

from ai_auto_trading.features.snapshot import FeatureSnapshot, TimeframeFeatureSnapshot


@dataclass(frozen=True)
class RuleStrategyParameters:
    execution_timeframe: str = "3m"
    micro_timeframe: str | None = None
    confirmation_timeframe: str = "15m"
    macro_timeframe: str = "1h"
    regime_timeframe: str | None = None
    anchor_timeframe: str | None = None
    use_confirmation: bool = True
    use_macro: bool = True
    use_regime: bool = True
    use_anchor: bool = True
    use_micro: bool = True
    require_vwap: bool = True
    require_rsi: bool = True
    require_price_reclaim: bool = True
    require_roc: bool = True
    atr_pct_min: float = 0.0005
    atr_pct_max: float = 0.02
    max_spread_bps: float = 5.0
    pullback_vwap_distance_pct: float = 0.003
    long_rsi_min: float = 52.0
    short_rsi_max: float = 48.0
    long_roc_min: float = 0.0
    short_roc_max: float = 0.0
    min_higher_tf_ema_spread_pct: float = 0.0
    min_volume_ratio_20: float = 0.0
    min_micro_long_ema_spread_pct: float = 0.0
    min_micro_short_ema_spread_pct: float = 0.0
    max_long_funding_rate: float | None = None
    min_short_funding_rate: float | None = None
    min_long_taker_buy_ratio: float | None = None
    max_short_taker_buy_ratio: float | None = None
    allow_long_entries: bool = True
    allow_short_entries: bool = True

    def required_timeframes(self) -> list[str]:
        ordered = [
            self.execution_timeframe,
            self.micro_timeframe if self.use_micro else None,
            self.confirmation_timeframe if self.use_confirmation else None,
            self.macro_timeframe if self.use_macro else None,
            self.regime_timeframe if self.use_regime else None,
            self.anchor_timeframe if self.use_anchor else None,
        ]
        output: list[str] = []
        for timeframe in ordered:
            if timeframe and timeframe not in output:
                output.append(timeframe)
        return output


@dataclass(frozen=True)
class RuleStrategyContext:
    snapshot: FeatureSnapshot
    cooldown_active: bool = False
    position_open: bool = False
    current_spread_bps: float = 0.0
    latest_funding_rate: float | None = None


@dataclass(frozen=True)
class SignalDecision:
    action: str
    reason_codes: list[str] = field(default_factory=list)


def evaluate_rule_signal(
    context: RuleStrategyContext,
    params: RuleStrategyParameters | None = None,
) -> SignalDecision:
    params = params or RuleStrategyParameters()
    blockers = _global_blockers(context, params)
    if blockers:
        return SignalDecision(action="NO_TRADE", reason_codes=blockers)

    try:
        execution = _required_snapshot(context.snapshot, params.execution_timeframe)
        confirmation = (
            _required_snapshot(context.snapshot, params.confirmation_timeframe)
            if params.use_confirmation
            else None
        )
        micro = (
            _required_snapshot(context.snapshot, params.micro_timeframe)
            if params.use_micro and params.micro_timeframe
            else None
        )
        macro = (
            _required_snapshot(context.snapshot, params.macro_timeframe)
            if params.use_macro
            else None
        )
        regime = (
            _required_snapshot(context.snapshot, params.regime_timeframe)
            if params.use_regime and params.regime_timeframe
            else None
        )
        anchor = (
            _required_snapshot(context.snapshot, params.anchor_timeframe)
            if params.use_anchor and params.anchor_timeframe
            else None
        )
    except KeyError as exc:
        return SignalDecision(action="NO_TRADE", reason_codes=["blocked_missing_timeframe", str(exc)])

    missing_reason = _missing_feature_reason(
        execution,
        micro,
        confirmation,
        macro,
        regime=regime,
        anchor=anchor,
        params=params,
        latest_funding_rate=context.latest_funding_rate,
    )
    if missing_reason:
        return SignalDecision(action="NO_TRADE", reason_codes=[missing_reason])

    atr_blocker = _atr_blocker(execution, params)
    if atr_blocker:
        return SignalDecision(action="NO_TRADE", reason_codes=[atr_blocker])

    long_reasons: list[str] = []
    if params.allow_long_entries:
        long_reasons = _long_reasons(
            execution,
            micro,
            confirmation,
            macro,
            regime,
            anchor,
            params,
            latest_funding_rate=context.latest_funding_rate,
        )
        if _long_pass(long_reasons, params=params):
            return SignalDecision(action="LONG", reason_codes=long_reasons)

    short_reasons: list[str] = []
    if params.allow_short_entries:
        short_reasons = _short_reasons(
            execution,
            micro,
            confirmation,
            macro,
            regime,
            anchor,
            params,
            latest_funding_rate=context.latest_funding_rate,
        )
        if _short_pass(short_reasons, params=params):
            return SignalDecision(action="SHORT", reason_codes=short_reasons)

    return SignalDecision(
        action="NO_TRADE",
        reason_codes=sorted(set(long_reasons + short_reasons + ["blocked_no_valid_setup"])),
    )


def _global_blockers(
    context: RuleStrategyContext, params: RuleStrategyParameters
) -> list[str]:
    reasons: list[str] = []
    if context.position_open:
        reasons.append("blocked_position_open")
    if context.cooldown_active:
        reasons.append("blocked_cooldown_active")
    if context.current_spread_bps > params.max_spread_bps:
        reasons.append("blocked_spread_too_wide")
    return reasons


def _required_snapshot(snapshot: FeatureSnapshot, timeframe: str) -> TimeframeFeatureSnapshot:
    if timeframe not in snapshot.timeframes:
        raise KeyError(timeframe)
    return snapshot.timeframes[timeframe]


def _missing_feature_reason(
    execution: TimeframeFeatureSnapshot,
    micro: TimeframeFeatureSnapshot | None,
    confirmation: TimeframeFeatureSnapshot | None,
    macro: TimeframeFeatureSnapshot | None,
    *,
    regime: TimeframeFeatureSnapshot | None,
    anchor: TimeframeFeatureSnapshot | None,
    params: RuleStrategyParameters,
    latest_funding_rate: float | None,
) -> str | None:
    required_values = [execution.atr_14]
    if params.require_rsi:
        required_values.append(execution.rsi_14)
    if params.require_vwap:
        required_values.append(execution.cumulative_vwap)
    if params.require_price_reclaim:
        required_values.append(execution.ema_fast_9)
    if params.require_roc:
        required_values.append(execution.roc_5)
    if confirmation is not None:
        required_values.extend([confirmation.ema_fast_9, confirmation.ema_slow_21])
    if macro is not None:
        required_values.extend([macro.ema_fast_9, macro.ema_slow_21])
    if micro is not None:
        required_values.extend([micro.ema_fast_9, micro.ema_slow_21])
    if regime is not None:
        required_values.extend([regime.ema_fast_9, regime.ema_slow_21])
    if anchor is not None:
        required_values.extend([anchor.ema_fast_9, anchor.ema_slow_21])
    if params.min_volume_ratio_20 > 0:
        required_values.append(execution.volume_ratio_20)
    if params.min_long_taker_buy_ratio is not None or params.max_short_taker_buy_ratio is not None:
        required_values.append(execution.taker_buy_ratio)
    if any(value is None for value in required_values):
        return "blocked_insufficient_features"
    if (
        params.max_long_funding_rate is not None
        or params.min_short_funding_rate is not None
    ) and latest_funding_rate is None:
        return "blocked_missing_funding_rate"
    return None


def _atr_blocker(
    execution: TimeframeFeatureSnapshot, params: RuleStrategyParameters
) -> str | None:
    assert execution.atr_14 is not None
    atr_pct = execution.atr_14 / execution.last_close
    if atr_pct < params.atr_pct_min:
        return "blocked_atr_too_low"
    if atr_pct > params.atr_pct_max:
        return "blocked_atr_too_high"
    return None


def _near_vwap(
    execution: TimeframeFeatureSnapshot, params: RuleStrategyParameters
) -> bool:
    assert execution.cumulative_vwap is not None
    distance_pct = abs(execution.last_close - execution.cumulative_vwap) / execution.cumulative_vwap
    return distance_pct <= params.pullback_vwap_distance_pct


def _long_reasons(
    execution: TimeframeFeatureSnapshot,
    micro: TimeframeFeatureSnapshot | None,
    confirmation: TimeframeFeatureSnapshot | None,
    macro: TimeframeFeatureSnapshot | None,
    regime: TimeframeFeatureSnapshot | None,
    anchor: TimeframeFeatureSnapshot | None,
    params: RuleStrategyParameters,
    latest_funding_rate: float | None,
) -> list[str]:
    reasons: list[str] = []
    if confirmation is not None:
        assert confirmation.ema_fast_9 is not None
        assert confirmation.ema_slow_21 is not None
    if macro is not None:
        assert macro.ema_fast_9 is not None
        assert macro.ema_slow_21 is not None
    if params.require_rsi:
        assert execution.rsi_14 is not None
    if params.require_price_reclaim:
        assert execution.ema_fast_9 is not None
    if params.require_roc:
        assert execution.roc_5 is not None

    if confirmation is not None and confirmation.ema_fast_9 > confirmation.ema_slow_21:
        reasons.append("long_trend_alignment")
    if micro is not None and _bullish_micro_trend_strength(micro, params):
        reasons.append("long_micro_trend_alignment")
    if macro is not None and macro.ema_fast_9 >= macro.ema_slow_21:
        reasons.append("long_macro_alignment")
    if params.require_vwap and _near_vwap(execution, params):
        reasons.append("long_pullback_near_vwap")
    if params.require_rsi and execution.rsi_14 >= params.long_rsi_min:
        reasons.append("long_rsi_reclaim")
    if params.require_price_reclaim and execution.last_close >= execution.ema_fast_9:
        reasons.append("long_price_reclaim")
    if params.require_roc and execution.roc_5 >= params.long_roc_min:
        reasons.append("long_roc_positive")
    if params.min_volume_ratio_20 > 0 and (
        execution.volume_ratio_20 is not None
        and execution.volume_ratio_20 >= params.min_volume_ratio_20
    ):
        reasons.append("long_volume_participation")
    if regime is not None and _bullish_trend_strength(regime, params):
        reasons.append("long_regime_alignment")
    if anchor is not None and _bullish_anchor_alignment(anchor):
        reasons.append("long_anchor_alignment")
    if (
        params.max_long_funding_rate is not None
        and latest_funding_rate is not None
        and latest_funding_rate <= params.max_long_funding_rate
    ):
        reasons.append("long_funding_not_overheated")
    if (
        params.min_long_taker_buy_ratio is not None
        and execution.taker_buy_ratio is not None
        and execution.taker_buy_ratio >= params.min_long_taker_buy_ratio
    ):
        reasons.append("long_taker_flow_alignment")
    return reasons


def _short_reasons(
    execution: TimeframeFeatureSnapshot,
    micro: TimeframeFeatureSnapshot | None,
    confirmation: TimeframeFeatureSnapshot | None,
    macro: TimeframeFeatureSnapshot | None,
    regime: TimeframeFeatureSnapshot | None,
    anchor: TimeframeFeatureSnapshot | None,
    params: RuleStrategyParameters,
    latest_funding_rate: float | None,
) -> list[str]:
    reasons: list[str] = []
    if confirmation is not None:
        assert confirmation.ema_fast_9 is not None
        assert confirmation.ema_slow_21 is not None
    if macro is not None:
        assert macro.ema_fast_9 is not None
        assert macro.ema_slow_21 is not None
    if params.require_rsi:
        assert execution.rsi_14 is not None
    if params.require_price_reclaim:
        assert execution.ema_fast_9 is not None
    if params.require_roc:
        assert execution.roc_5 is not None

    if confirmation is not None and confirmation.ema_fast_9 < confirmation.ema_slow_21:
        reasons.append("short_trend_alignment")
    if micro is not None and _bearish_micro_trend_strength(micro, params):
        reasons.append("short_micro_trend_alignment")
    if macro is not None and macro.ema_fast_9 <= macro.ema_slow_21:
        reasons.append("short_macro_alignment")
    if params.require_vwap and _near_vwap(execution, params):
        reasons.append("short_pullback_near_vwap")
    if params.require_rsi and execution.rsi_14 <= params.short_rsi_max:
        reasons.append("short_rsi_reclaim")
    if params.require_price_reclaim and execution.last_close <= execution.ema_fast_9:
        reasons.append("short_price_reclaim")
    if params.require_roc and execution.roc_5 <= params.short_roc_max:
        reasons.append("short_roc_negative")
    if params.min_volume_ratio_20 > 0 and (
        execution.volume_ratio_20 is not None
        and execution.volume_ratio_20 >= params.min_volume_ratio_20
    ):
        reasons.append("short_volume_participation")
    if regime is not None and _bearish_trend_strength(regime, params):
        reasons.append("short_regime_alignment")
    if anchor is not None and _bearish_anchor_alignment(anchor):
        reasons.append("short_anchor_alignment")
    if (
        params.min_short_funding_rate is not None
        and latest_funding_rate is not None
        and latest_funding_rate >= params.min_short_funding_rate
    ):
        reasons.append("short_funding_not_crowded")
    if (
        params.max_short_taker_buy_ratio is not None
        and execution.taker_buy_ratio is not None
        and execution.taker_buy_ratio <= params.max_short_taker_buy_ratio
    ):
        reasons.append("short_taker_flow_alignment")
    return reasons


def _required_long_reasons(params: RuleStrategyParameters) -> set[str]:
    required: set[str] = set()
    if params.use_confirmation:
        required.add("long_trend_alignment")
    if params.use_macro:
        required.add("long_macro_alignment")
    if params.require_vwap:
        required.add("long_pullback_near_vwap")
    if params.require_rsi:
        required.add("long_rsi_reclaim")
    if params.require_price_reclaim:
        required.add("long_price_reclaim")
    if params.require_roc:
        required.add("long_roc_positive")
    if params.use_micro and params.micro_timeframe:
        required.add("long_micro_trend_alignment")
    if params.min_volume_ratio_20 > 0:
        required.add("long_volume_participation")
    if params.use_regime and params.regime_timeframe:
        required.add("long_regime_alignment")
    if params.use_anchor and params.anchor_timeframe:
        required.add("long_anchor_alignment")
    if params.max_long_funding_rate is not None:
        required.add("long_funding_not_overheated")
    if params.min_long_taker_buy_ratio is not None:
        required.add("long_taker_flow_alignment")
    return required


def _required_short_reasons(params: RuleStrategyParameters) -> set[str]:
    required: set[str] = set()
    if params.use_confirmation:
        required.add("short_trend_alignment")
    if params.use_macro:
        required.add("short_macro_alignment")
    if params.require_vwap:
        required.add("short_pullback_near_vwap")
    if params.require_rsi:
        required.add("short_rsi_reclaim")
    if params.require_price_reclaim:
        required.add("short_price_reclaim")
    if params.require_roc:
        required.add("short_roc_negative")
    if params.use_micro and params.micro_timeframe:
        required.add("short_micro_trend_alignment")
    if params.min_volume_ratio_20 > 0:
        required.add("short_volume_participation")
    if params.use_regime and params.regime_timeframe:
        required.add("short_regime_alignment")
    if params.use_anchor and params.anchor_timeframe:
        required.add("short_anchor_alignment")
    if params.min_short_funding_rate is not None:
        required.add("short_funding_not_crowded")
    if params.max_short_taker_buy_ratio is not None:
        required.add("short_taker_flow_alignment")
    return required


def _long_pass(reasons: list[str], *, params: RuleStrategyParameters) -> bool:
    return _required_long_reasons(params).issubset(set(reasons))


def _short_pass(reasons: list[str], *, params: RuleStrategyParameters) -> bool:
    return _required_short_reasons(params).issubset(set(reasons))


def _ema_spread_pct(snapshot: TimeframeFeatureSnapshot) -> float:
    assert snapshot.ema_fast_9 is not None
    assert snapshot.ema_slow_21 is not None
    base = snapshot.last_close or snapshot.ema_slow_21
    if base == 0:
        return 0.0
    return abs(snapshot.ema_fast_9 - snapshot.ema_slow_21) / base


def _bullish_trend_strength(
    snapshot: TimeframeFeatureSnapshot, params: RuleStrategyParameters
) -> bool:
    assert snapshot.ema_fast_9 is not None
    assert snapshot.ema_slow_21 is not None
    return (
        snapshot.ema_fast_9 > snapshot.ema_slow_21
        and _ema_spread_pct(snapshot) >= params.min_higher_tf_ema_spread_pct
    )


def _bearish_trend_strength(
    snapshot: TimeframeFeatureSnapshot, params: RuleStrategyParameters
) -> bool:
    assert snapshot.ema_fast_9 is not None
    assert snapshot.ema_slow_21 is not None
    return (
        snapshot.ema_fast_9 < snapshot.ema_slow_21
        and _ema_spread_pct(snapshot) >= params.min_higher_tf_ema_spread_pct
    )


def _bullish_micro_trend_strength(
    snapshot: TimeframeFeatureSnapshot, params: RuleStrategyParameters
) -> bool:
    assert snapshot.ema_fast_9 is not None
    assert snapshot.ema_slow_21 is not None
    return (
        snapshot.ema_fast_9 > snapshot.ema_slow_21
        and _ema_spread_pct(snapshot) >= params.min_micro_long_ema_spread_pct
    )


def _bearish_micro_trend_strength(
    snapshot: TimeframeFeatureSnapshot, params: RuleStrategyParameters
) -> bool:
    assert snapshot.ema_fast_9 is not None
    assert snapshot.ema_slow_21 is not None
    return (
        snapshot.ema_fast_9 < snapshot.ema_slow_21
        and _ema_spread_pct(snapshot) >= params.min_micro_short_ema_spread_pct
    )


def _bullish_anchor_alignment(snapshot: TimeframeFeatureSnapshot) -> bool:
    assert snapshot.ema_fast_9 is not None
    assert snapshot.ema_slow_21 is not None
    return snapshot.ema_fast_9 >= snapshot.ema_slow_21 and snapshot.last_close >= snapshot.ema_slow_21


def _bearish_anchor_alignment(snapshot: TimeframeFeatureSnapshot) -> bool:
    assert snapshot.ema_fast_9 is not None
    assert snapshot.ema_slow_21 is not None
    return snapshot.ema_fast_9 <= snapshot.ema_slow_21 and snapshot.last_close <= snapshot.ema_slow_21
