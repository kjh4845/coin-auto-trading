from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Protocol
from urllib.request import Request, urlopen

from ai_auto_trading.features.snapshot import FeatureSnapshot
from ai_auto_trading.settings import Settings, load_settings
from ai_auto_trading.strategy.rule_based import SignalDecision


@dataclass(frozen=True)
class AIDecision:
    regime: str
    setup_quality: float
    entry_action: str
    exit_action: str
    confidence: float
    reason_codes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AIInferenceError(RuntimeError):
    pass


class HttpJsonTransport(Protocol):
    def request(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes,
        timeout: float,
    ) -> Any: ...


class UrllibJsonTransport:
    def request(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes,
        timeout: float,
    ) -> Any:
        request = Request(url=url, data=body, method=method, headers=headers)
        with urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload) if payload else {}


class LocalInferenceTradeAssistant:
    def __init__(
        self,
        *,
        model_id: str,
        model_base: str,
        model_path: str | None,
        endpoint: str,
        timeout_seconds: float = 20.0,
        max_tokens: int = 128,
        transport: HttpJsonTransport | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_base = model_base
        self.model_path = model_path or ""
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.transport = transport or UrllibJsonTransport()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "LocalInferenceTradeAssistant":
        resolved = settings or load_settings()
        return cls(
            model_id=resolved.local_model_id,
            model_base=resolved.local_model_base,
            model_path=resolved.local_model_path,
            endpoint=resolved.local_model_endpoint,
            timeout_seconds=resolved.ai_request_timeout_seconds,
            max_tokens=resolved.ai_max_tokens,
        )

    def review_entry(
        self,
        *,
        snapshot: FeatureSnapshot,
        rule_decision: SignalDecision,
    ) -> AIDecision:
        payload = {
            "model": self.model_id,
            "messages": _entry_messages(snapshot=snapshot, rule_decision=rule_decision),
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        if self.model_path:
            payload["model_path"] = self.model_path
        response = self.transport.request(
            method="POST",
            url=_chat_completions_url(self.endpoint),
            headers={"Content-Type": "application/json"},
            body=json.dumps(payload).encode("utf-8"),
            timeout=self.timeout_seconds,
        )
        return parse_ai_decision(_extract_message_content(response))


def parse_ai_decision(raw_text: str) -> AIDecision:
    candidate = _extract_json_object(raw_text)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise AIInferenceError(f"invalid ai json output: {exc}") from exc

    required = {"regime", "setup_quality", "entry_action", "exit_action", "confidence", "reason_codes"}
    missing = sorted(required - payload.keys())
    if missing:
        raise AIInferenceError(f"ai json missing required fields: {','.join(missing)}")

    regime = _normalize_regime(payload["regime"])
    entry_action = _normalize_entry_action(payload["entry_action"])
    exit_action = _normalize_exit_action(payload["exit_action"])
    setup_quality = _bounded_float("setup_quality", payload["setup_quality"])
    confidence = _bounded_float("confidence", payload["confidence"])
    reason_codes = payload["reason_codes"]
    if isinstance(reason_codes, str):
        reason_codes = [reason_codes]
    if not isinstance(reason_codes, list) or any(not isinstance(item, str) for item in reason_codes):
        raise AIInferenceError("reason_codes must be a list of strings")

    return AIDecision(
        regime=regime,
        setup_quality=setup_quality,
        entry_action=entry_action,
        exit_action=exit_action,
        confidence=confidence,
        reason_codes=list(reason_codes),
    )


def _bounded_float(name: str, value: Any) -> float:
    if isinstance(value, str):
        normalized_text = value.strip().lower()
        qualitative = {
            "very_poor": 0.1,
            "poor": 0.2,
            "weak": 0.25,
            "fair": 0.4,
            "very_low": 0.1,
            "low": 0.25,
            "medium": 0.5,
            "mid": 0.5,
            "moderate": 0.5,
            "good": 0.7,
            "strong": 0.8,
            "high": 0.75,
            "excellent": 0.9,
            "very_high": 0.9,
        }
        if normalized_text in qualitative:
            return qualitative[normalized_text]
        if normalized_text.endswith("%"):
            normalized_text = normalized_text[:-1].strip()
        value = normalized_text
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise AIInferenceError(f"{name} must be numeric") from exc
    if 1.0 < normalized <= 5.0:
        normalized = normalized / 5.0
    elif 1.0 < normalized <= 100.0:
        normalized = normalized / 100.0
    if not 0.0 <= normalized <= 1.0:
        raise AIInferenceError(f"{name} must be between 0.0 and 1.0")
    return normalized


def _normalize_regime(value: Any) -> str:
    normalized = str(value).strip().lower()
    mapping = {
        "trend_up": "trend_up",
        "uptrend": "trend_up",
        "bullish": "trend_up",
        "trend_down": "trend_down",
        "downtrend": "trend_down",
        "bearish": "trend_down",
        "range": "range",
        "sideways": "range",
        "choppy": "range",
        "high_volatility": "high_volatility",
        "volatile": "high_volatility",
        "unclear": "unclear",
        "neutral": "unclear",
    }
    if normalized in mapping:
        return mapping[normalized]
    if "up" in normalized or "bull" in normalized:
        return "trend_up"
    if "down" in normalized or "bear" in normalized:
        return "trend_down"
    if "range" in normalized or "side" in normalized or "chop" in normalized:
        return "range"
    if "vol" in normalized:
        return "high_volatility"
    if "trend" in normalized:
        return "unclear"
    raise AIInferenceError(f"unsupported regime value: {value}")


def _normalize_entry_action(value: Any) -> str:
    normalized = str(value).strip().lower()
    mapping = {
        "allow": "allow",
        "approve": "allow",
        "enter": "allow",
        "go": "allow",
        "long": "allow",
        "short": "allow",
        "buy": "allow",
        "sell": "allow",
        "execute_long": "allow",
        "execute_short": "allow",
        "veto": "veto",
        "reject": "veto",
        "skip": "veto",
        "no_trade": "veto",
        "avoid": "veto",
        "reduce_size": "reduce_size",
        "reduce": "reduce_size",
        "smaller_size": "reduce_size",
        "trim": "reduce_size",
        "cautious": "reduce_size",
    }
    if normalized in mapping:
        return mapping[normalized]
    if "reduce" in normalized or "small" in normalized or "trim" in normalized:
        return "reduce_size"
    if "veto" in normalized or "reject" in normalized or "skip" in normalized or "avoid" in normalized:
        return "veto"
    if "allow" in normalized or "enter" in normalized or "execute" in normalized or "go" in normalized:
        return "allow"
    raise AIInferenceError(f"unsupported entry_action value: {value}")


def _normalize_exit_action(value: Any) -> str:
    normalized = str(value).strip().lower()
    mapping = {
        "hold": "hold",
        "none": "hold",
        "stay": "hold",
        "tighten_stop": "tighten_stop",
        "tighten": "tighten_stop",
        "tight_stop": "tighten_stop",
        "reduce_risk": "tighten_stop",
        "full_exit": "full_exit",
        "exit": "full_exit",
        "close": "full_exit",
        "take_profit": "full_exit",
        "tp": "full_exit",
        "buy": "full_exit",
        "sell": "full_exit",
        "long": "full_exit",
        "short": "full_exit",
    }
    if normalized in mapping:
        return mapping[normalized]
    if "tight" in normalized or "risk" in normalized or "stop" in normalized:
        return "tighten_stop"
    if "profit" in normalized:
        return "full_exit"
    if "exit" in normalized or "close" in normalized:
        return "full_exit"
    if "hold" in normalized or "none" in normalized or "stay" in normalized:
        return "hold"
    raise AIInferenceError(f"unsupported exit_action value: {value}")


def _entry_messages(*, snapshot: FeatureSnapshot, rule_decision: SignalDecision) -> list[dict[str, str]]:
    system_prompt = (
        "BTCUSDT futures gate. "
        "Return JSON only with keys regime,setup_quality,entry_action,exit_action,confidence,reason_codes. "
        "No prose or markdown."
    )
    user_payload = {
        "task": "entry_gate",
        "sym": snapshot.symbol,
        "rule": {
            "action": rule_decision.action,
            "reasons": rule_decision.reason_codes,
        },
        "tfs": {
            key: {
                "close": value.last_close,
                "ema9": value.ema_fast_9,
                "ema21": value.ema_slow_21,
                "rsi": value.rsi_14,
                "atr": value.atr_14,
                "vwap": value.cumulative_vwap,
                "sh20": value.swing_high_20,
                "sl20": value.swing_low_20,
                "roc5": value.roc_5,
            }
            for key, value in snapshot.timeframes.items()
        },
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, sort_keys=True, separators=(",", ":"))},
    ]


def _chat_completions_url(endpoint: str) -> str:
    if endpoint.endswith("/v1/chat/completions"):
        return endpoint
    if endpoint.endswith("/v1"):
        return f"{endpoint}/chat/completions"
    return f"{endpoint}/v1/chat/completions"


def _extract_message_content(payload: Any) -> str:
    if isinstance(payload, dict) and "choices" in payload:
        choices = payload["choices"]
        if not isinstance(choices, list) or not choices:
            raise AIInferenceError("chat completion response has no choices")
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            fragments: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    fragments.append(str(item.get("text", "")))
            joined = "".join(fragments).strip()
            if joined:
                return joined
    raise AIInferenceError("unable to extract model text from response payload")


def _extract_json_object(raw_text: str) -> str:
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise AIInferenceError("model response does not contain a JSON object")
    return stripped[start : end + 1]
