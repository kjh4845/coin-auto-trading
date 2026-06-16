from __future__ import annotations

from dataclasses import dataclass

from ai_auto_trading.execution.testnet import (
    ReconciliationResult,
    RemoteExecutionState,
    _matches_expected_stop_order,
)
from ai_auto_trading.settings import Settings
from ai_auto_trading.runtime.testnet import TestnetExecutionRuntime


@dataclass(frozen=True)
class RuntimeRiskConfig:
    user_stream_stale_ms: int = 120_000
    mark_price_stale_ms: int = 15_000
    api_error_window_ms: int = 300_000
    max_error_incidents_in_window: int = 3
    daily_loss_window_ms: int = 86_400_000
    max_daily_loss_r: float = 3.0
    max_consecutive_losses: int = 3
    cooldown_after_loss_ms: int = 30 * 60_000

    @classmethod
    def from_settings(cls, settings: Settings) -> "RuntimeRiskConfig":
        return cls(
            max_daily_loss_r=settings.max_daily_loss_r,
            max_consecutive_losses=settings.max_consecutive_losses,
            cooldown_after_loss_ms=settings.cooldown_after_loss_minutes * 60_000,
        )


class TestnetRiskManager:
    def __init__(self, config: RuntimeRiskConfig | None = None) -> None:
        self.config = config or RuntimeRiskConfig()

    def evaluate(
        self,
        *,
        runtime: TestnetExecutionRuntime,
        symbol: str,
        remote_state: RemoteExecutionState,
        reconciliation: ReconciliationResult | None,
        now_ms: int,
    ) -> list[str]:
        active_codes: set[str] = set()
        status = runtime.load_runtime_status(symbol)
        expected_state = runtime.load_expected_state(symbol)

        if expected_state is not None and _has_open_position(remote_state):
            has_stop = any(
                _matches_expected_stop_order(
                    order=order,
                    symbol=symbol,
                    expected_stop_price_mark=expected_state.expected_stop_price_mark,
                )
                for order in remote_state.open_orders + remote_state.open_algo_orders
            )
            if not has_stop:
                runtime.acquire_lockout(
                    symbol=symbol,
                    code="missing_protective_stop",
                    reason="open position without expected protective stop",
                    details={"expected_stop_price_mark": expected_state.expected_stop_price_mark},
                )
                active_codes.add("missing_protective_stop")
            else:
                runtime.release_lockout(
                    symbol=symbol,
                    code="missing_protective_stop",
                    reason="expected protective stop present again",
                )
        else:
            runtime.release_lockout(
                symbol=symbol,
                code="missing_protective_stop",
                reason="no protected position expected",
            )

        if reconciliation is not None and not reconciliation.ok:
            runtime.acquire_lockout(
                symbol=symbol,
                code="reconciliation_mismatch",
                reason="exchange state does not match expected runtime state",
                details={"mismatches": reconciliation.mismatches},
            )
            active_codes.add("reconciliation_mismatch")
        else:
            runtime.release_lockout(
                symbol=symbol,
                code="reconciliation_mismatch",
                reason="reconciliation is clean",
            )

        requires_user_stream = expected_state is not None or _has_open_position(remote_state)
        if status is not None and status.active_listen_key and requires_user_stream:
            heartbeat_reference_ms = status.last_user_stream_event_ms
            if heartbeat_reference_ms is None or (
                now_ms - heartbeat_reference_ms > self.config.user_stream_stale_ms
            ):
                runtime.acquire_lockout(
                    symbol=symbol,
                    code="user_stream_stale",
                    reason="user data stream has not produced events within the allowed window",
                    details={
                        "last_user_stream_event_ms": status.last_user_stream_event_ms,
                        "heartbeat_reference_ms": heartbeat_reference_ms,
                        "now_ms": now_ms,
                    },
                )
                active_codes.add("user_stream_stale")
            else:
                runtime.release_lockout(
                    symbol=symbol,
                    code="user_stream_stale",
                    reason="user data stream heartbeat is fresh",
                )
        else:
            runtime.release_lockout(
                symbol=symbol,
                code="user_stream_stale",
                reason="no protected position requiring user stream monitoring",
            )

        if _has_open_position(remote_state):
            heartbeat_reference_ms = status.last_mark_price_event_ms if status is not None else None
            if heartbeat_reference_ms is None or (
                now_ms - heartbeat_reference_ms > self.config.mark_price_stale_ms
            ):
                runtime.acquire_lockout(
                    symbol=symbol,
                    code="mark_price_stale",
                    reason="mark price heartbeat is stale while a position is open",
                    details={
                        "last_mark_price_event_ms": heartbeat_reference_ms,
                        "now_ms": now_ms,
                    },
                )
                active_codes.add("mark_price_stale")
            else:
                runtime.release_lockout(
                    symbol=symbol,
                    code="mark_price_stale",
                    reason="mark price heartbeat is fresh",
                )
        else:
            runtime.release_lockout(
                symbol=symbol,
                code="mark_price_stale",
                reason="no open position requiring mark price heartbeat",
            )

        error_count = len(
            runtime.recent_incidents(
                limit=self.config.max_error_incidents_in_window,
                level="ERROR",
                since_ms=now_ms - self.config.api_error_window_ms,
            )
        )
        if error_count >= self.config.max_error_incidents_in_window:
            runtime.acquire_lockout(
                symbol=symbol,
                code="api_error_threshold",
                reason="too many recent execution errors",
                details={"error_count": error_count},
            )
            active_codes.add("api_error_threshold")
        else:
            runtime.release_lockout(
                symbol=symbol,
                code="api_error_threshold",
                reason="recent API error count is below threshold",
            )

        account_risk = runtime.account_risk_overview(symbol, now_ms=now_ms)
        daily_limit_r = self.config.max_daily_loss_r
        if daily_limit_r > 0 and account_risk["recent_24h_realized_r"] <= -daily_limit_r:
            runtime.acquire_lockout(
                symbol=symbol,
                code="daily_loss_limit_breached",
                reason="recent 24h realized loss exceeded the configured daily R cap",
                details=account_risk,
            )
            active_codes.add("daily_loss_limit_breached")
        else:
            runtime.release_lockout(
                symbol=symbol,
                code="daily_loss_limit_breached",
                reason="recent 24h realized loss is below the configured daily R cap",
            )

        streak_limit = self.config.max_consecutive_losses
        if streak_limit > 0 and account_risk["consecutive_losses"] >= streak_limit:
            runtime.acquire_lockout(
                symbol=symbol,
                code="consecutive_loss_limit",
                reason="consecutive realized losses exceeded the configured streak cap",
                details=account_risk,
            )
            active_codes.add("consecutive_loss_limit")
        else:
            runtime.release_lockout(
                symbol=symbol,
                code="consecutive_loss_limit",
                reason="consecutive realized losses are below the configured streak cap",
            )

        cooldown_ms = self.config.cooldown_after_loss_ms
        if cooldown_ms > 0 and account_risk["cooldown_remaining_ms"] > 0:
            runtime.acquire_lockout(
                symbol=symbol,
                code="recent_loss_cooldown",
                reason="the runtime is inside the cooldown window after the latest realized loss",
                details=account_risk,
            )
            active_codes.add("recent_loss_cooldown")
        else:
            runtime.release_lockout(
                symbol=symbol,
                code="recent_loss_cooldown",
                reason="loss cooldown window is not active",
            )

        return [lockout.code for lockout in runtime.active_lockouts(symbol)]


def _has_open_position(remote_state: RemoteExecutionState) -> bool:
    for position in remote_state.positions:
        if abs(float(position.get("positionAmt", "0"))) > 1e-9:
            return True
    return False
