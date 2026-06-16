from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Protocol

from ai_auto_trading.features.snapshot import FeatureSnapshot
from ai_auto_trading.settings import Settings, load_settings
from ai_auto_trading.strategy.rule_based import SignalDecision

from .inference import AIDecision


class AITradeAssistant(Protocol):
    model_base: str

    def review_entry(
        self,
        *,
        snapshot: FeatureSnapshot,
        rule_decision: SignalDecision,
    ) -> AIDecision: ...


@dataclass(frozen=True)
class AIGateConfig:
    min_setup_quality: float = 0.55
    reduce_size_fraction: float = 0.5
    fail_open: bool = False
    max_latency_ms: int = 20_000

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "AIGateConfig":
        resolved = settings or load_settings()
        return cls(
            min_setup_quality=resolved.ai_gate_min_setup_quality,
            reduce_size_fraction=resolved.ai_reduce_size_fraction,
            fail_open=resolved.ai_gate_fail_open,
            max_latency_ms=resolved.ai_max_latency_ms,
        )


@dataclass(frozen=True)
class AIGatedSignalDecision:
    action: str
    reason_codes: list[str]
    size_multiplier: float
    model_base: str
    adapter_version: str | None
    ai_snapshot: dict[str, Any] | None
    fallback_used: bool = False


def apply_ai_entry_gate(
    *,
    rule_decision: SignalDecision,
    snapshot: FeatureSnapshot,
    assistant: AITradeAssistant | None,
    gate_config: AIGateConfig | None = None,
) -> AIGatedSignalDecision:
    if rule_decision.action not in {"LONG", "SHORT"} or assistant is None:
        return AIGatedSignalDecision(
            action=rule_decision.action,
            reason_codes=list(rule_decision.reason_codes),
            size_multiplier=1.0,
            model_base="rule_only",
            adapter_version=None,
            ai_snapshot=None,
        )

    config = gate_config or AIGateConfig()
    started_at = time.perf_counter()
    try:
        ai_decision = assistant.review_entry(snapshot=snapshot, rule_decision=rule_decision)
    except Exception as exc:
        if config.fail_open:
            return AIGatedSignalDecision(
                action=rule_decision.action,
                reason_codes=list(rule_decision.reason_codes),
                size_multiplier=1.0,
                model_base="rule_only",
                adapter_version=None,
                ai_snapshot=None,
                fallback_used=True,
            )
        return AIGatedSignalDecision(
            action="NO_TRADE",
            reason_codes=list(rule_decision.reason_codes) + ["ai_runtime_error_veto"],
            size_multiplier=1.0,
            model_base=assistant.model_base,
            adapter_version=None,
            ai_snapshot={"error": str(exc), "mode": "fail_closed"},
            fallback_used=True,
        )
    latency_ms = int((time.perf_counter() - started_at) * 1000)

    ai_snapshot = ai_decision.to_dict()
    ai_snapshot["latency_ms"] = latency_ms
    if config.max_latency_ms > 0 and latency_ms > config.max_latency_ms:
        ai_snapshot["latency_limit_ms"] = config.max_latency_ms
        return AIGatedSignalDecision(
            action="NO_TRADE",
            reason_codes=list(rule_decision.reason_codes) + ["ai_latency_veto"] + _prefixed_ai_reasons(ai_decision.reason_codes),
            size_multiplier=1.0,
            model_base=assistant.model_base,
            adapter_version=None,
            ai_snapshot=ai_snapshot,
        )
    ai_reason_codes = _prefixed_ai_reasons(ai_decision.reason_codes)
    allowed_regime = _regime_allows_entry(rule_decision.action, ai_decision.regime)
    if not allowed_regime:
        return AIGatedSignalDecision(
            action="NO_TRADE",
            reason_codes=list(rule_decision.reason_codes) + ["ai_regime_veto"] + ai_reason_codes,
            size_multiplier=1.0,
            model_base=assistant.model_base,
            adapter_version=None,
            ai_snapshot=ai_snapshot,
        )
    if ai_decision.setup_quality < config.min_setup_quality:
        return AIGatedSignalDecision(
            action="NO_TRADE",
            reason_codes=list(rule_decision.reason_codes) + ["ai_setup_quality_veto"] + ai_reason_codes,
            size_multiplier=1.0,
            model_base=assistant.model_base,
            adapter_version=None,
            ai_snapshot=ai_snapshot,
        )
    if ai_decision.entry_action == "veto":
        return AIGatedSignalDecision(
            action="NO_TRADE",
            reason_codes=list(rule_decision.reason_codes) + ["ai_entry_action_veto"] + ai_reason_codes,
            size_multiplier=1.0,
            model_base=assistant.model_base,
            adapter_version=None,
            ai_snapshot=ai_snapshot,
        )
    if ai_decision.exit_action == "full_exit":
        return AIGatedSignalDecision(
            action="NO_TRADE",
            reason_codes=list(rule_decision.reason_codes) + ["ai_exit_action_veto"] + ai_reason_codes,
            size_multiplier=1.0,
            model_base=assistant.model_base,
            adapter_version=None,
            ai_snapshot=ai_snapshot,
        )

    reduce_size = ai_decision.entry_action == "reduce_size" or ai_decision.exit_action == "tighten_stop"
    size_multiplier = config.reduce_size_fraction if reduce_size else 1.0
    extra_reasons = []
    if ai_decision.entry_action == "reduce_size":
        extra_reasons.append("ai_entry_reduce_size")
    else:
        extra_reasons.append("ai_entry_allow")
    if ai_decision.exit_action == "tighten_stop":
        extra_reasons.append("ai_exit_tighten_stop")
    return AIGatedSignalDecision(
        action=rule_decision.action,
        reason_codes=list(rule_decision.reason_codes) + extra_reasons + ai_reason_codes,
        size_multiplier=size_multiplier,
        model_base=assistant.model_base,
        adapter_version=None,
        ai_snapshot=ai_snapshot,
    )


def _regime_allows_entry(action: str, regime: str) -> bool:
    if action == "LONG":
        return regime == "trend_up"
    if action == "SHORT":
        return regime == "trend_down"
    return False


def _prefixed_ai_reasons(reason_codes: list[str]) -> list[str]:
    return [f"ai_{item}" for item in reason_codes]
