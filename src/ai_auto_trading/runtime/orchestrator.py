from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import time
from typing import Any, Callable

from ai_auto_trading.ai.gate import AIGateConfig, AITradeAssistant, apply_ai_entry_gate
from ai_auto_trading.ai.inference import LocalInferenceTradeAssistant
from ai_auto_trading.execution.testnet import (
    ExecutionConstraintError,
    ReconciliationResult,
    TestnetExecutionEngine,
    _has_active_state,
)
from ai_auto_trading.features.snapshot import FeatureBuilder
from ai_auto_trading.models import OrderIntent, TradeRecord
from ai_auto_trading.models import PositionState
from ai_auto_trading.risk.hard_stop import build_exchange_hard_stop_intent, evaluate_hard_stop
from ai_auto_trading.runtime.kline_stream import KlineCloseEvent, TestnetKlineStreamService
from ai_auto_trading.runtime.mark_price_stream import MarkPriceEvent, TestnetMarkPriceStreamService
from ai_auto_trading.runtime.position_manager import ManagedExitDecision, evaluate_managed_trade_exit
from ai_auto_trading.runtime.risk_manager import RuntimeRiskConfig, TestnetRiskManager
from ai_auto_trading.runtime.testnet import (
    BundlePlacementResult,
    ExpectedRuntimeState,
    ManagedTradeState,
    TestnetExecutionRuntime,
    build_runtime_trade_record,
)
from ai_auto_trading.runtime.user_stream import TestnetUserStreamService, UserDataStreamSession
from ai_auto_trading.settings import validate_leverage
from ai_auto_trading.strategy.runtime_profiles import (
    RuntimeEntryProfile,
    build_runtime_entry_profiles,
    build_runtime_entry_profiles_for_symbol,
    required_runtime_timeframes,
    runtime_stream_interval,
)
from ai_auto_trading.strategy.rule_based import RuleStrategyContext, evaluate_rule_signal


@dataclass(frozen=True)
class MonitorResult:
    symbol: str
    reconciliation: ReconciliationResult | None
    active_lockouts: list[str]
    runtime_state: str
    safety_action: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ManagedPositionResult:
    symbol: str
    action: str
    runtime_state: str
    exit_reason: str | None = None
    exit_order: dict[str, Any] | None = None
    managed_trade_state: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EntryAttemptResult:
    symbol: str
    action: str
    signal_action: str
    reason_codes: list[str]
    bundle_result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TestnetRuntimeOrchestrator:
    def __init__(
        self,
        engine: TestnetExecutionEngine,
        runtime: TestnetExecutionRuntime,
        risk_manager: TestnetRiskManager | None = None,
        user_stream_service: TestnetUserStreamService | None = None,
        kline_stream_service: TestnetKlineStreamService | None = None,
        mark_price_stream_service: TestnetMarkPriceStreamService | None = None,
        ai_trade_assistant: AITradeAssistant | None = None,
        ai_gate_config: AIGateConfig | None = None,
    ) -> None:
        self.engine = engine
        self.runtime = runtime
        self.risk_manager = risk_manager or TestnetRiskManager(
            RuntimeRiskConfig.from_settings(runtime.settings)
        )
        self.user_stream_service = user_stream_service or TestnetUserStreamService(
            engine.client,
            runtime,
        )
        self.kline_stream_service = kline_stream_service or TestnetKlineStreamService(
            engine.client,
            runtime,
        )
        self.mark_price_stream_service = mark_price_stream_service or TestnetMarkPriceStreamService(
            engine.client,
            runtime,
        )
        if ai_trade_assistant is None and runtime.settings.ai_gate_enabled:
            ai_trade_assistant = LocalInferenceTradeAssistant.from_settings(runtime.settings)
        self.ai_trade_assistant = ai_trade_assistant
        self.ai_gate_config = ai_gate_config or AIGateConfig.from_settings(runtime.settings)

    def start_user_stream(self, *, symbol: str, timestamp: int) -> UserDataStreamSession:
        session = self.user_stream_service.start_session(symbol=symbol, timestamp=timestamp)
        self.runtime.update_runtime_status(symbol=symbol, state="READY")
        return session

    def keepalive_user_stream(self, *, symbol: str, timestamp: int) -> UserDataStreamSession:
        return self.user_stream_service.keepalive_session(symbol=symbol, timestamp=timestamp)

    def ensure_user_stream_heartbeat(
        self,
        *,
        symbol: str,
        timestamp: int,
        refresh_after_ms: int = 60_000,
    ) -> UserDataStreamSession | None:
        status = self.runtime.load_runtime_status(symbol)
        if status is None or not status.active_listen_key:
            return self.start_user_stream(symbol=symbol, timestamp=timestamp)
        if (
            status.last_user_stream_event_ms is not None
            and timestamp - status.last_user_stream_event_ms < refresh_after_ms
        ):
            return None
        try:
            return self.keepalive_user_stream(symbol=symbol, timestamp=timestamp)
        except Exception as exc:
            self.runtime.record_incident(
                level="WARNING",
                event_type="user_stream_keepalive_failed",
                message="user data stream keepalive failed; starting a fresh listen key",
                details={"symbol": symbol, "error": str(exc)},
            )
            self.runtime.update_runtime_status(
                symbol=symbol,
                active_listen_key=None,
                last_user_stream_event_ms=None,
            )
            return self.start_user_stream(symbol=symbol, timestamp=timestamp)

    def close_user_stream(self, *, symbol: str) -> None:
        self.user_stream_service.close_session(symbol=symbol)
        expected_state = self.runtime.load_expected_state(symbol)
        managed_state = self.runtime.load_managed_trade_state(symbol)
        self.runtime.update_runtime_status(
            symbol=symbol,
            state="PAUSED" if expected_state is not None or managed_state is not None else "IDLE",
        )

    def ingest_user_stream_payload(self, *, symbol: str, payload: dict[str, Any]) -> None:
        self.user_stream_service.ingest_payload(symbol=symbol, payload=payload)

    def record_mark_price_heartbeat(self, *, symbol: str, event_time_ms: int) -> None:
        self.runtime.update_runtime_status(
            symbol=symbol,
            last_mark_price_event_ms=event_time_ms,
        )

    def _record_watchdog_failure(
        self,
        *,
        symbol: str,
        code: str,
        event_type: str,
        message: str,
        details: dict[str, Any],
    ) -> None:
        self.runtime.acquire_lockout(
            symbol=symbol,
            code=code,
            reason=message,
            details=details,
        )
        self.runtime.update_runtime_status(
            symbol=symbol,
            state="ERROR",
            last_error=message,
        )
        self.runtime.record_incident(
            level="ERROR",
            event_type=event_type,
            message=message,
            details=details,
        )

    async def _monitor_watchdog_loop(
        self,
        *,
        symbol: str,
        stale_after_seconds: float,
        check_interval_seconds: float,
        duration_seconds: float | None,
    ) -> None:
        deadline = time.monotonic() + duration_seconds if duration_seconds else None
        stale_after_ms = int(stale_after_seconds * 1000)
        startup_grace_until = time.monotonic() + stale_after_seconds
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                break
            sleep_for = check_interval_seconds
            if deadline is not None:
                sleep_for = min(sleep_for, max(0.1, deadline - time.monotonic()))
            await asyncio.sleep(max(0.1, sleep_for))
            status = self.runtime.load_runtime_status(symbol)
            now_ms = int(time.time() * 1000)
            last_reconcile_ms = status.last_reconcile_ms if status is not None else None
            if last_reconcile_ms is not None and now_ms - last_reconcile_ms <= stale_after_ms:
                self.runtime.release_lockout(
                    symbol=symbol,
                    code="runtime_monitor_stale",
                    reason="runtime monitor tick is fresh",
                )
                continue
            if last_reconcile_ms is None and time.monotonic() < startup_grace_until:
                continue
            message = "runtime monitor tick is stale"
            details = {
                "symbol": symbol,
                "last_reconcile_ms": last_reconcile_ms,
                "now_ms": now_ms,
                "stale_after_seconds": stale_after_seconds,
            }
            self._record_watchdog_failure(
                symbol=symbol,
                code="runtime_monitor_stale",
                event_type="runtime_monitor_stale",
                message=message,
                details=details,
            )
            raise RuntimeError(message)

    def _position_from_trade_state(self, trade_state: ManagedTradeState) -> PositionState:
        return PositionState(
            side=trade_state.side,
            quantity=trade_state.quantity,
            leverage_at_entry=trade_state.leverage_at_entry,
            entry_contract_price_avg=trade_state.entry_contract_price_avg,
            entry_mark_price=trade_state.entry_mark_price,
            symbol=trade_state.symbol,
        )

    def _require_manual_review(
        self,
        *,
        symbol: str,
        reason: str,
        safety_action: str,
    ) -> None:
        self.runtime.acquire_lockout(
            symbol=symbol,
            code="manual_review_required",
            reason="manual resume is required after a runtime safety intervention",
            details={
                "symbol": symbol,
                "reason": reason,
                "safety_action": safety_action,
            },
        )

    def _remote_reduce_only_exit(self, remote_state) -> OrderIntent | None:
        for position in remote_state.positions:
            amount = float(position.get("positionAmt", "0"))
            if abs(amount) <= 1e-9:
                continue
            exit_side = "SELL" if amount > 0 else "BUY"
            return OrderIntent(
                side=exit_side,
                quantity=abs(amount),
                order_type="MARKET",
                symbol=str(position.get("symbol") or self.runtime.settings.trading_symbol),
                reduce_only=True,
            )
        return None

    def _cancel_open_risk_orders(self, *, symbol: str, timestamp: int) -> None:
        cleanup_errors: list[str] = []
        try:
            self.engine.client.cancel_all_open_orders(symbol=symbol, timestamp=timestamp)
        except Exception as exc:
            cleanup_errors.append(f"open_orders={exc}")
        try:
            self.engine.client.cancel_all_open_algo_orders(symbol=symbol, timestamp=timestamp)
        except Exception as exc:
            cleanup_errors.append(f"algo_orders={exc}")
        if cleanup_errors:
            self.runtime.record_incident(
                level="WARNING",
                event_type="risk_order_cleanup_error",
                message="one or more open-order cleanup steps failed",
                details={"symbol": symbol, "errors": cleanup_errors},
            )

    def _persist_closed_trade_record(
        self,
        *,
        trade_record: TradeRecord,
        closed_at_ms: int,
        opened_at_ms: int,
    ) -> None:
        self.runtime.persist_closed_trade_record(
            trade_record=trade_record,
            closed_at_ms=closed_at_ms,
        )
        self._sync_exchange_income_after_close(
            trade_record=trade_record,
            opened_at_ms=opened_at_ms,
            closed_at_ms=closed_at_ms,
        )

    def _sync_exchange_income_after_close(
        self,
        *,
        trade_record: TradeRecord,
        opened_at_ms: int,
        closed_at_ms: int,
    ) -> None:
        start_time_ms = max(0, min(opened_at_ms, closed_at_ms) - 60_000)
        rows: list[dict[str, Any]] = []
        synced_at_ms = closed_at_ms
        try:
            for attempt in range(3):
                synced_at_ms = self.engine.server_timestamp()
                fetched = self.engine.client.query_income_history(
                    symbol=trade_record.symbol,
                    start_time=start_time_ms,
                    end_time=synced_at_ms,
                    limit=1000,
                    timestamp=synced_at_ms,
                )
                if isinstance(fetched, list):
                    rows = [row for row in fetched if isinstance(row, dict)]
                stored = self.runtime.persist_exchange_income_records(
                    rows,
                    synced_at_ms=synced_at_ms,
                    related_trade_id=trade_record.trade_id,
                )
                if _income_rows_include_realized_pnl(stored, symbol=trade_record.symbol):
                    self.runtime.record_incident(
                        level="INFO",
                        event_type="exchange_income_synced",
                        message="exchange income was synced after trade close",
                        details={
                            "symbol": trade_record.symbol,
                            "trade_id": trade_record.trade_id,
                            "row_count": len(stored),
                            "start_time_ms": start_time_ms,
                            "end_time_ms": synced_at_ms,
                        },
                    )
                    return
                if attempt < 2:
                    time.sleep(1.0)
            self.runtime.record_incident(
                level="WARNING",
                event_type="exchange_income_sync_incomplete",
                message="exchange income sync did not find realized pnl after trade close",
                details={
                    "symbol": trade_record.symbol,
                    "trade_id": trade_record.trade_id,
                    "row_count": len(rows),
                    "start_time_ms": start_time_ms,
                    "end_time_ms": synced_at_ms,
                },
            )
        except Exception as exc:
            self.runtime.record_incident(
                level="WARNING",
                event_type="exchange_income_sync_failed",
                message="exchange income sync failed after trade close",
                details={
                    "symbol": trade_record.symbol,
                    "trade_id": trade_record.trade_id,
                    "error": str(exc),
                    "start_time_ms": start_time_ms,
                    "end_time_ms": synced_at_ms,
                },
            )

    def _exit_managed_trade(
        self,
        *,
        trade_state: ManagedTradeState,
        timestamp: int,
        current_mark_price: float,
        exit_reason: str,
        exit_contract_price: float,
        notes: str,
        incident_event_type: str,
        incident_message: str,
        incident_details: dict[str, Any],
    ) -> ManagedPositionResult:
        self._cancel_open_risk_orders(symbol=trade_state.symbol, timestamp=timestamp)
        exit_side = "SELL" if trade_state.side == "LONG" else "BUY"
        exit_order = self.engine.place_failsafe_exit(
            exit_intent=OrderIntent(
                side=exit_side,
                quantity=trade_state.quantity,
                order_type="MARKET",
                symbol=trade_state.symbol,
                reduce_only=True,
            ),
            timestamp=timestamp,
        )
        self.runtime.record_incident(
            level="WARNING" if exit_reason in {"HARD_STOP_MARK_PRICE", "SYSTEM_FAILSAFE_EXIT"} else "INFO",
            event_type=incident_event_type,
            message=incident_message,
            details=incident_details,
        )
        trade_record = build_runtime_trade_record(
            trade_state=trade_state,
            exit_reason=exit_reason,
            exit_contract_price=exit_contract_price,
            exit_mark_price=current_mark_price,
            closed_at_ms=timestamp,
            notes=notes,
        )
        self._persist_closed_trade_record(
            trade_record=trade_record,
            closed_at_ms=timestamp,
            opened_at_ms=trade_state.opened_at_ms,
        )
        self.runtime.clear_expected_state(trade_state.symbol)
        self.runtime.clear_managed_trade_state(trade_state.symbol)
        self.runtime.update_runtime_status(symbol=trade_state.symbol, state="READY", last_error=None)
        return ManagedPositionResult(
            symbol=trade_state.symbol,
            action="EXIT",
            runtime_state="READY",
            exit_reason=exit_reason,
            exit_order=exit_order,
            managed_trade_state=trade_state.to_dict(),
        )

    def _safety_flatten_remote_state(
        self,
        *,
        symbol: str,
        timestamp: int,
        remote_state,
        reason: str,
        trade_state: ManagedTradeState | None,
    ) -> str | None:
        has_remote_activity = _has_active_state(remote_state)
        if not has_remote_activity:
            return None

        exit_intent = self._remote_reduce_only_exit(remote_state)
        self._cancel_open_risk_orders(symbol=symbol, timestamp=timestamp)
        current_mark_price = float(self.engine.client.get_mark_price(symbol=symbol)["markPrice"])

        if exit_intent is not None:
            self.engine.place_failsafe_exit(exit_intent=exit_intent, timestamp=timestamp)
            if trade_state is not None:
                trade_record = build_runtime_trade_record(
                    trade_state=trade_state,
                    exit_reason="SYSTEM_FAILSAFE_EXIT",
                    exit_contract_price=current_mark_price,
                    exit_mark_price=current_mark_price,
                    closed_at_ms=timestamp,
                    notes=reason,
                )
                self._persist_closed_trade_record(
                    trade_record=trade_record,
                    closed_at_ms=timestamp,
                    opened_at_ms=trade_state.opened_at_ms,
                )
            self.runtime.record_incident(
                level="WARNING",
                event_type="system_failsafe_exit",
                message="remote state was force-flattened by safety logic",
                details={
                    "symbol": symbol,
                    "reason": reason,
                    "mark_price": current_mark_price,
                    "trade_state_present": trade_state is not None,
                },
            )
            self._require_manual_review(
                symbol=symbol,
                reason=reason,
                safety_action="system_failsafe_exit",
            )
            self.runtime.clear_expected_state(symbol)
            self.runtime.clear_managed_trade_state(symbol)
            self.runtime.update_runtime_status(symbol=symbol, state="PAUSED", last_error=reason)
            return "system_failsafe_exit"

        if trade_state is not None:
            exit_reason, exit_contract_price, exit_mark_price, closed_at_ms, notes = self._infer_exchange_flat_exit(
                symbol=symbol,
                trade_state=trade_state,
                timestamp=timestamp,
            )
            trade_record = build_runtime_trade_record(
                trade_state=trade_state,
                exit_reason=exit_reason,
                exit_contract_price=exit_contract_price,
                exit_mark_price=exit_mark_price,
                closed_at_ms=closed_at_ms,
                notes=f"{notes};{reason}",
            )
            self._persist_closed_trade_record(
                trade_record=trade_record,
                closed_at_ms=closed_at_ms,
                opened_at_ms=trade_state.opened_at_ms,
            )
            self.runtime.clear_expected_state(symbol)
            self.runtime.clear_managed_trade_state(symbol)

        self.runtime.record_incident(
            level="WARNING",
            event_type="system_failsafe_clear_orders",
            message="orphan remote orders were canceled by safety logic",
            details={"symbol": symbol, "reason": reason},
        )
        self._require_manual_review(
            symbol=symbol,
            reason=reason,
            safety_action="system_failsafe_clear_orders",
        )
        self.runtime.update_runtime_status(symbol=symbol, state="PAUSED", last_error=reason)
        return "system_failsafe_clear_orders"

    def _apply_monitor_safety_action(
        self,
        *,
        symbol: str,
        timestamp: int,
        reconciliation: ReconciliationResult,
        active_lockouts: list[str],
    ) -> str | None:
        trade_state = self.runtime.load_managed_trade_state(symbol)
        reasons: list[str] = []
        if "missing_protective_stop" in active_lockouts:
            reasons.append("missing_protective_stop")
        if "mark_price_stale" in active_lockouts:
            reasons.append("mark_price_stale")
        if "user_stream_stale" in active_lockouts:
            reasons.append("user_stream_stale")
        if "api_error_threshold" in active_lockouts:
            reasons.append("api_error_threshold")
        if "orphan_remote_state" in reconciliation.mismatches:
            reasons.append("orphan_remote_state")
        if reconciliation.mismatches and _has_active_state(reconciliation.remote_state):
            reasons.append(",".join(reconciliation.mismatches))
        if not reasons:
            return None
        return self._safety_flatten_remote_state(
            symbol=symbol,
            timestamp=timestamp,
            remote_state=reconciliation.remote_state,
            reason=";".join(dict.fromkeys(reasons)),
            trade_state=trade_state,
        )

    def _reconcile_remote_state(self, *, symbol: str, timestamp: int) -> tuple[ExpectedRuntimeState | None, ReconciliationResult]:
        expected_state = self.runtime.load_expected_state(symbol)
        if expected_state is not None:
            reconciliation = self.engine.reconcile_state(
                symbol=symbol,
                expected_position_qty=expected_state.expected_position_qty,
                expected_stop_price_mark=expected_state.expected_stop_price_mark,
                expected_leverage=expected_state.expected_leverage,
                expected_margin_mode=expected_state.expected_margin_mode,
                expected_position_mode=expected_state.expected_position_mode,
                timestamp=timestamp,
            )
            return expected_state, reconciliation

        remote_state = self.engine.fetch_remote_state(symbol=symbol, timestamp=timestamp)
        return expected_state, ReconciliationResult(
            ok=not _has_active_state(remote_state),
            mismatches=["orphan_remote_state"] if _has_active_state(remote_state) else [],
            remote_state=remote_state,
        )

    def reconcile_once(self, *, symbol: str, timestamp: int) -> MonitorResult:
        self.runtime.update_runtime_status(symbol=symbol, state="RECONCILING", last_reconcile_ms=timestamp)
        expected_state, reconciliation = self._reconcile_remote_state(symbol=symbol, timestamp=timestamp)

        self.runtime.record_account_snapshot(
            symbol=symbol,
            snapshot_type="monitor_once",
            payload={
                "ok": reconciliation.ok,
                "mismatches": reconciliation.mismatches,
                "remote_state": reconciliation.remote_state,
            },
            recorded_at_ms=timestamp,
        )
        self.runtime.release_lockout(
            symbol=symbol,
            code="runtime_monitor_stale",
            reason="runtime monitor tick completed",
        )
        self.runtime.release_lockout(
            symbol=symbol,
            code="runtime_background_task_failed",
            reason="runtime monitor tick completed",
        )
        active_lockouts = self.risk_manager.evaluate(
            runtime=self.runtime,
            symbol=symbol,
            remote_state=reconciliation.remote_state,
            reconciliation=reconciliation,
            now_ms=timestamp,
        )
        safety_action = self._apply_monitor_safety_action(
            symbol=symbol,
            timestamp=timestamp,
            reconciliation=reconciliation,
            active_lockouts=active_lockouts,
        )
        if safety_action is not None:
            expected_state, reconciliation = self._reconcile_remote_state(symbol=symbol, timestamp=timestamp)
            active_lockouts = self.risk_manager.evaluate(
                runtime=self.runtime,
                symbol=symbol,
                remote_state=reconciliation.remote_state,
                reconciliation=reconciliation,
                now_ms=timestamp,
            )
        protected_state = expected_state is not None or self.runtime.load_managed_trade_state(symbol) is not None
        state = "PROTECTED" if protected_state else ("PAUSED" if active_lockouts else "READY")
        self.runtime.update_runtime_status(
            symbol=symbol,
            state=state,
            last_reconcile_ms=timestamp,
            last_error=None if reconciliation.ok else ",".join(reconciliation.mismatches),
        )
        self.runtime.record_incident(
            level="INFO" if not active_lockouts else "WARNING",
            event_type="monitor_once",
            message="runtime monitor tick completed",
            details={
                "symbol": symbol,
                "runtime_state": state,
                "active_lockouts": active_lockouts,
                "reconciliation_ok": reconciliation.ok,
            },
        )
        return MonitorResult(
            symbol=symbol,
            reconciliation=reconciliation,
            active_lockouts=active_lockouts,
            runtime_state=state,
            safety_action=safety_action,
        )

    def guarded_place_entry_bundle(
        self,
        *,
        symbol: str,
        timestamp: int,
        place_fn,
    ) -> BundlePlacementResult:
        active_lockouts = [lockout.code for lockout in self.runtime.active_lockouts(symbol)]
        if active_lockouts:
            self.runtime.record_incident(
                level="WARNING",
                event_type="entry_blocked_by_lockout",
                message="entry bundle blocked by active risk lockout",
                details={"symbol": symbol, "active_lockouts": active_lockouts},
            )
            raise ValueError(f"entry blocked by active lockouts: {','.join(active_lockouts)}")

        self.runtime.update_runtime_status(symbol=symbol, state="ENTRY_PENDING")
        result = place_fn()
        self.runtime.update_runtime_status(symbol=symbol, state="PROTECTED")
        return result

    def attempt_rule_entry_once(
        self,
        *,
        symbol: str,
        timestamp: int,
        entry_notional_usdt: float,
        leverage: int,
        candle_limit: int = 120,
        trigger_close_time_ms: int | None = None,
    ) -> EntryAttemptResult:
        remote_state = self.engine.fetch_remote_state(symbol=symbol, timestamp=timestamp)
        self.risk_manager.evaluate(
            runtime=self.runtime,
            symbol=symbol,
            remote_state=remote_state,
            reconciliation=None,
            now_ms=timestamp,
        )
        if any(abs(float(position.get("positionAmt", "0"))) > 1e-9 for position in remote_state.positions):
            return EntryAttemptResult(
                symbol=symbol,
                action="SKIP_OPEN_POSITION",
                signal_action="NO_TRADE",
                reason_codes=["blocked_position_open"],
            )

        active_lockouts = [lockout.code for lockout in self.runtime.active_lockouts(symbol)]
        if active_lockouts:
            return EntryAttemptResult(
                symbol=symbol,
                action="SKIP_LOCKOUT",
                signal_action="NO_TRADE",
                reason_codes=active_lockouts,
            )

        leverage = validate_leverage(leverage, self.runtime.settings)
        builder = FeatureBuilder()
        settings = self.runtime.settings
        strategy_profiles = self._strategy_profiles(settings, symbol=symbol)
        candles_by_timeframe = self._closed_candles_by_timeframe(
            symbol=symbol,
            end_close_time=timestamp,
            candle_limit=candle_limit,
            timeframes=required_runtime_timeframes(settings, symbol=symbol),
        )
        candle_close_times = {
            timeframe: int(rows[-1]["close_time"])
            for timeframe, rows in candles_by_timeframe.items()
            if rows
        }
        snapshot = builder.build_multi_timeframe_snapshot(
            symbol=symbol,
            candles_by_timeframe=candles_by_timeframe,
        )
        premium_index = self.engine.client.get_mark_price(symbol=symbol)
        self.runtime.update_runtime_status(
            symbol=symbol,
            last_mark_price_event_ms=timestamp,
        )
        latest_funding_rate = (
            float(premium_index["lastFundingRate"])
            if premium_index.get("lastFundingRate") not in (None, "")
            else None
        )
        profile_diagnostics: list[dict[str, Any]] = []
        selected_profile: RuntimeEntryProfile | None = None
        decision = None
        for profile in strategy_profiles:
            latest_profile_close_time = candle_close_times.get(profile.params.execution_timeframe)
            if trigger_close_time_ms is not None and latest_profile_close_time != trigger_close_time_ms:
                profile_diagnostics.append(
                    {
                        "profile": profile.name,
                        "execution_timeframe": profile.params.execution_timeframe,
                        "status": "skipped_stale_candle",
                        "latest_close_time_ms": latest_profile_close_time,
                        "trigger_close_time_ms": trigger_close_time_ms,
                    }
                )
                continue
            rule_decision = evaluate_rule_signal(
                RuleStrategyContext(
                    snapshot=snapshot,
                    latest_funding_rate=latest_funding_rate,
                ),
                params=profile.params,
            )
            gated_decision = apply_ai_entry_gate(
                rule_decision=rule_decision,
                snapshot=snapshot,
                assistant=self.ai_trade_assistant,
                gate_config=self.ai_gate_config,
            )
            if gated_decision.fallback_used:
                self.runtime.record_incident(
                    level="WARNING",
                    event_type="ai_entry_fallback_rule_only",
                    message="ai entry gate failed and rule-only fallback was applied",
                    details={"symbol": symbol, "profile": profile.name},
                )
            profile_diagnostics.append(
                {
                    "profile": profile.name,
                    "execution_timeframe": profile.params.execution_timeframe,
                    "rule_action": rule_decision.action,
                    "final_action": gated_decision.action,
                    "reason_codes": gated_decision.reason_codes,
                    "ai_snapshot": gated_decision.ai_snapshot,
                    "fallback_used": gated_decision.fallback_used,
                }
            )
            if gated_decision.action in {"LONG", "SHORT"}:
                selected_profile = profile
                decision = gated_decision
                break
        if selected_profile is None or decision is None:
            combined_reasons: list[str] = []
            for diagnostic in profile_diagnostics:
                for reason in diagnostic.get("reason_codes", []):
                    if reason not in combined_reasons:
                        combined_reasons.append(reason)
            self.runtime.record_incident(
                level="INFO",
                event_type="signal_no_trade",
                message="configured runtime profiles did not produce a tradeable setup",
                details={
                    "symbol": symbol,
                    "reason_codes": combined_reasons,
                    "profiles": profile_diagnostics,
                },
            )
            return EntryAttemptResult(
                symbol=symbol,
                action="NO_TRADE",
                signal_action="NO_TRADE",
                reason_codes=combined_reasons,
            )

        effective_entry_notional_usdt = entry_notional_usdt * decision.size_multiplier
        margin_check = self.engine.check_entry_margin_available(
            entry_notional_usdt=effective_entry_notional_usdt,
            leverage=leverage,
            timestamp=timestamp,
        )
        if not margin_check.ok:
            self.runtime.record_incident(
                level="WARNING",
                event_type="signal_entry_skipped_insufficient_margin",
                message="runtime profile produced a setup but available futures balance was too low",
                details={
                    "symbol": symbol,
                    "profile": selected_profile.name,
                    "execution_timeframe": selected_profile.params.execution_timeframe,
                    "signal_action": decision.action,
                    "reason_codes": decision.reason_codes,
                    "margin_check": margin_check.to_dict(),
                },
            )
            return EntryAttemptResult(
                symbol=symbol,
                action="SKIP_INSUFFICIENT_MARGIN",
                signal_action=decision.action,
                reason_codes=["blocked_insufficient_available_balance", *decision.reason_codes],
                bundle_result=margin_check.to_dict(),
            )

        self.ensure_user_stream_heartbeat(symbol=symbol, timestamp=timestamp)

        execution_snapshot = snapshot.timeframes[selected_profile.params.execution_timeframe]
        contract_price = execution_snapshot.last_close
        mark_price = float(premium_index["markPrice"])
        quantity = effective_entry_notional_usdt / contract_price
        side = "BUY" if decision.action == "LONG" else "SELL"
        position = PositionState(
            side=decision.action,
            quantity=quantity,
            leverage_at_entry=float(leverage),
            entry_contract_price_avg=contract_price,
            entry_mark_price=mark_price,
            symbol=symbol,
        )
        hard_stop_intent = build_exchange_hard_stop_intent(position)
        entry_intent = OrderIntent(
            side=side,
            quantity=quantity,
            order_type="MARKET",
            symbol=symbol,
        )

        bundle_result = self.guarded_place_entry_bundle(
            symbol=symbol,
            timestamp=timestamp,
            place_fn=lambda: self.runtime.place_entry_and_protection(
                entry_intent=entry_intent,
                hard_stop_intent=hard_stop_intent,
                timestamp=timestamp,
                expected_leverage=leverage,
                expected_margin_mode="ISOLATED",
                expected_position_mode="ONE_WAY",
                skip_preflight=False,
                signal_reason_codes=[f"profile:{selected_profile.name}", *decision.reason_codes],
                model_base=decision.model_base,
                adapter_version=decision.adapter_version,
                ai_snapshot=decision.ai_snapshot,
                execution_timeframe=selected_profile.params.execution_timeframe,
            ),
        )
        self.runtime.record_incident(
            level="INFO",
            event_type="signal_entry_placed",
            message="runtime profile produced a setup and bundle was placed",
            details={
                "symbol": symbol,
                "profile": selected_profile.name,
                "execution_timeframe": selected_profile.params.execution_timeframe,
                "signal_action": decision.action,
                "reason_codes": decision.reason_codes,
                "entry_notional_usdt": entry_notional_usdt,
                "size_multiplier": decision.size_multiplier,
                "ai_snapshot": decision.ai_snapshot,
                "model_base": decision.model_base,
            },
        )
        return EntryAttemptResult(
            symbol=symbol,
            action="PLACED",
            signal_action=decision.action,
            reason_codes=decision.reason_codes,
            bundle_result=bundle_result.expected_state.to_dict(),
        )

    def manage_open_position_once(
        self,
        *,
        symbol: str,
        timestamp: int,
        candle_limit: int = 120,
    ) -> ManagedPositionResult:
        trade_state = self.runtime.load_managed_trade_state(symbol)
        if trade_state is None:
            self.runtime.record_incident(
                level="INFO",
                event_type="managed_position_idle",
                message="no managed trade state present",
                details={"symbol": symbol},
            )
            self.runtime.update_runtime_status(symbol=symbol, state="READY")
            return ManagedPositionResult(
                symbol=symbol,
                action="NO_POSITION",
                runtime_state="READY",
            )

        remote_state = self.engine.fetch_remote_state(symbol=symbol, timestamp=timestamp)
        has_open_position = any(
            abs(float(position.get("positionAmt", "0"))) > 1e-9
            for position in remote_state.positions
        )
        if not has_open_position:
            exit_reason, exit_contract_price, exit_mark_price, closed_at_ms, notes = self._infer_exchange_flat_exit(
                symbol=symbol,
                trade_state=trade_state,
                timestamp=timestamp,
            )
            trade_record = build_runtime_trade_record(
                trade_state=trade_state,
                exit_reason=exit_reason,
                exit_contract_price=exit_contract_price,
                exit_mark_price=exit_mark_price,
                closed_at_ms=closed_at_ms,
                notes=notes,
            )
            self._persist_closed_trade_record(
                trade_record=trade_record,
                closed_at_ms=closed_at_ms,
                opened_at_ms=trade_state.opened_at_ms,
            )
            self.runtime.record_incident(
                level="WARNING",
                event_type="managed_state_cleared_no_position",
                message="managed trade state cleared because no open exchange position exists",
                details={
                    "symbol": symbol,
                    "exit_reason": exit_reason,
                    "trade_id": trade_record.trade_id,
                },
            )
            self.runtime.clear_expected_state(symbol)
            self.runtime.clear_managed_trade_state(symbol)
            self.runtime.update_runtime_status(symbol=symbol, state="READY")
            return ManagedPositionResult(
                symbol=symbol,
                action="NO_POSITION",
                runtime_state="READY",
            )

        current_mark_price = float(self.engine.client.get_mark_price(symbol=symbol)["markPrice"])
        self.runtime.update_runtime_status(
            symbol=symbol,
            last_mark_price_event_ms=timestamp,
        )
        hard_stop = evaluate_hard_stop(
            self._position_from_trade_state(trade_state),
            current_mark_price=current_mark_price,
        )
        if hard_stop.triggered:
            return self._exit_managed_trade(
                trade_state=trade_state,
                timestamp=timestamp,
                current_mark_price=current_mark_price,
                exit_reason="HARD_STOP_MARK_PRICE",
                exit_contract_price=current_mark_price,
                notes="local_mark_price_failsafe_exit",
                incident_event_type="managed_position_hard_stop_exit",
                incident_message="managed position exited by local mark-price hard stop",
                incident_details={
                    "symbol": symbol,
                    "current_mark_price": current_mark_price,
                    "trigger_loss_usdt": hard_stop.trigger_loss_usdt,
                    "unrealized_pnl_usdt": hard_stop.unrealized_pnl_usdt,
                },
            )

        execution_candles = self.engine.client.fetch_contract_klines(
            symbol=symbol,
            interval=trade_state.execution_timeframe,
            limit=candle_limit,
            end_time=timestamp,
        )
        decision = evaluate_managed_trade_exit(
            trade_state=trade_state,
            execution_candles=execution_candles,
            latest_allowed_close_time_ms=timestamp,
            atr_trail_activation_profit_r=self.runtime.settings.atr_trail_activation_profit_r,
            atr_trail_min_bars=self.runtime.settings.atr_trail_min_bars,
        )
        updated_trade_state = decision.updated_trade_state or trade_state
        self.runtime.save_managed_trade_state(updated_trade_state)

        if decision.action != "EXIT":
            self.runtime.record_incident(
                level="INFO",
                event_type="managed_position_heartbeat",
                message="managed position reviewed without local exit",
                details={
                    "symbol": symbol,
                    "bars_held": updated_trade_state.bars_held,
                    "current_mark_price": current_mark_price,
                },
            )
            self.runtime.update_runtime_status(symbol=symbol, state="PROTECTED")
            return ManagedPositionResult(
                symbol=symbol,
                action="HOLD",
                runtime_state="PROTECTED",
                managed_trade_state=updated_trade_state.to_dict(),
            )

        return self._exit_managed_trade(
            trade_state=updated_trade_state,
            timestamp=timestamp,
            current_mark_price=current_mark_price,
            exit_reason=decision.exit_reason or "OTHER",
            exit_contract_price=decision.exit_contract_price or current_mark_price,
            notes="event_driven_runtime_exit",
            incident_event_type="managed_position_exit",
            incident_message="managed position exited by local exit logic",
            incident_details={
                "symbol": symbol,
                "exit_reason": decision.exit_reason,
                "exit_contract_price": decision.exit_contract_price,
                "current_mark_price": current_mark_price,
            },
        )

    def _closed_candles_by_timeframe(
        self,
        *,
        symbol: str,
        end_close_time: int,
        candle_limit: int,
        timeframes: list[str] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        settings = self.runtime.settings
        requested = timeframes or required_runtime_timeframes(settings)
        candles_by_timeframe: dict[str, list[dict[str, Any]]] = {}
        for timeframe in requested:
            rows = self.engine.client.fetch_contract_klines(
                symbol=symbol,
                interval=timeframe,
                limit=candle_limit,
                end_time=end_close_time,
            )
            closed_rows = [
                row for row in rows
                if int(row["close_time"]) <= end_close_time
            ]
            if not closed_rows:
                raise ExecutionConstraintError(
                    f"no closed candles available for timeframe {timeframe} at {end_close_time}"
                )
            candles_by_timeframe[timeframe] = closed_rows
        return candles_by_timeframe

    def latest_runtime_trigger_close_time(self, *, symbol: str, timestamp: int) -> int:
        interval = runtime_stream_interval(self.runtime.settings, symbol=symbol)
        rows = self.engine.client.fetch_contract_klines(
            symbol=symbol,
            interval=interval,
            limit=2,
            end_time=timestamp,
        )
        closed_rows = [row for row in rows if int(row["close_time"]) <= timestamp]
        if not closed_rows:
            raise ExecutionConstraintError(
                f"no closed candles available for runtime interval {interval} at {timestamp}"
            )
        close_time = int(closed_rows[-1]["close_time"])
        self.runtime.update_runtime_status(
            symbol=symbol,
            last_execution_kline_close_ms=close_time,
        )
        return close_time

    @staticmethod
    def _strategy_profiles(settings, *, symbol: str | None = None) -> list[RuntimeEntryProfile]:
        return sorted(
            (
                build_runtime_entry_profiles_for_symbol(settings, symbol)
                if symbol is not None
                else build_runtime_entry_profiles(settings)
            ),
            key=lambda profile: (-profile.priority, profile.name),
        )

    def _infer_exchange_flat_exit(
        self,
        *,
        symbol: str,
        trade_state: ManagedTradeState,
        timestamp: int,
    ) -> tuple[str, float, float, int, str]:
        recent_orders = self.runtime.recent_order_events(symbol, limit=20)
        for order in recent_orders:
            if not order["reduce_only"]:
                continue
            if order["status"] != "FILLED":
                continue
            if int(order["event_time_ms"]) < trade_state.opened_at_ms:
                continue
            if order["order_type"] == "STOP_MARKET":
                price = (
                    order.get("avg_price")
                    or order.get("stop_price")
                    or order.get("price")
                    or float(self.engine.client.get_mark_price(symbol=symbol)["markPrice"])
                )
                return (
                    "HARD_STOP_MARK_PRICE",
                    float(price),
                    float(price),
                    int(order["event_time_ms"]),
                    "exchange_side_stop_fill_detected",
                )
            price = (
                order.get("avg_price")
                or order.get("price")
                or float(self.engine.client.get_mark_price(symbol=symbol)["markPrice"])
            )
            return (
                "OTHER",
                float(price),
                float(price),
                int(order["event_time_ms"]),
                "exchange_side_flat_detected_from_order_event",
            )

        current_mark_price = float(self.engine.client.get_mark_price(symbol=symbol)["markPrice"])
        return (
            "OTHER",
            current_mark_price,
            current_mark_price,
            timestamp,
            "exchange_side_flat_detected_without_fill_event",
        )

    async def handle_mark_price_tick(self, *, event: MarkPriceEvent) -> dict[str, Any] | None:
        trade_state = self.runtime.load_managed_trade_state(event.symbol)
        if trade_state is None:
            return None

        evaluation = evaluate_hard_stop(
            self._position_from_trade_state(trade_state),
            current_mark_price=event.mark_price,
        )
        if not evaluation.triggered:
            return None

        remote_state = self.engine.fetch_remote_state(symbol=event.symbol, timestamp=event.event_time_ms)
        if not any(abs(float(position.get("positionAmt", "0"))) > 1e-9 for position in remote_state.positions):
            result = self.manage_open_position_once(
                symbol=event.symbol,
                timestamp=event.event_time_ms,
            )
            return {
                "trigger": "mark_price_tick",
                "event": event.to_dict(),
                "result": result.to_dict(),
            }

        result = self._exit_managed_trade(
            trade_state=trade_state,
            timestamp=event.event_time_ms,
            current_mark_price=event.mark_price,
            exit_reason="HARD_STOP_MARK_PRICE",
            exit_contract_price=event.mark_price,
            notes="mark_price_stream_failsafe_exit",
            incident_event_type="mark_price_failsafe_exit",
            incident_message="managed position exited by real-time mark price hard stop",
            incident_details={
                "symbol": event.symbol,
                "event_time_ms": event.event_time_ms,
                "mark_price": event.mark_price,
                "trigger_loss_usdt": evaluation.trigger_loss_usdt,
                "unrealized_pnl_usdt": evaluation.unrealized_pnl_usdt,
            },
        )
        return {
            "trigger": "mark_price_tick",
            "event": event.to_dict(),
            "result": result.to_dict(),
        }

    async def handle_execution_kline_close(
        self,
        *,
        event: KlineCloseEvent,
        entry_notional_usdt: float | Callable[[], float],
        leverage: int,
        candle_limit: int = 120,
    ) -> dict[str, Any]:
        monitor = self.reconcile_once(symbol=event.symbol, timestamp=event.close_time)
        managed_state = self.runtime.load_managed_trade_state(event.symbol)
        if monitor.safety_action is not None:
            return {
                "trigger": "execution_kline_close",
                "monitor": monitor.to_dict(),
                "result": None,
            }
        if managed_state is not None:
            result = self.manage_open_position_once(
                symbol=event.symbol,
                timestamp=event.close_time,
                candle_limit=candle_limit,
            )
        else:
            if monitor.active_lockouts:
                return {
                    "trigger": "execution_kline_close",
                    "monitor": monitor.to_dict(),
                    "result": None,
                }
            resolved_entry_notional_usdt = (
                entry_notional_usdt()
                if callable(entry_notional_usdt)
                else entry_notional_usdt
            )
            result = self.attempt_rule_entry_once(
                symbol=event.symbol,
                timestamp=event.close_time,
                entry_notional_usdt=resolved_entry_notional_usdt,
                leverage=leverage,
                candle_limit=candle_limit,
                trigger_close_time_ms=event.close_time,
            )
        return {
            "trigger": "execution_kline_close",
            "monitor": monitor.to_dict(),
            "result": _maybe_to_dict(result),
        }

    async def _monitor_runtime_loop(
        self,
        *,
        symbol: str,
        monitor_interval_seconds: float,
        duration_seconds: float | None,
        cycle_results: list[dict[str, Any]],
    ) -> None:
        deadline = time.monotonic() + duration_seconds if duration_seconds else None
        while True:
            try:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                timestamp = self.engine.server_timestamp()
                result = self.reconcile_once(symbol=symbol, timestamp=timestamp)
                cycle_results.append(
                    {
                        "trigger": "monitor_tick",
                        "monitor": result.to_dict(),
                        "result": None,
                    }
                )
                sleep_for = monitor_interval_seconds
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    sleep_for = min(sleep_for, remaining)
                await asyncio.sleep(max(0.1, sleep_for))
            except Exception as exc:
                message = "runtime monitor task failed"
                self._record_watchdog_failure(
                    symbol=symbol,
                    code="runtime_background_task_failed",
                    event_type="runtime_background_task_failed",
                    message=message,
                    details={
                        "symbol": symbol,
                        "task": "runtime_monitor",
                        "error": str(exc),
                    },
                )
                raise

    async def run_event_driven_cycle(
        self,
        *,
        symbol: str,
        entry_notional_usdt: float | Callable[[], float],
        leverage: int,
        candle_limit: int = 120,
        duration_seconds: float | None = None,
        max_closed_candles: int | None = None,
        reconnect_delay_seconds: float = 1.0,
        monitor_interval_seconds: float = 5.0,
        stream_no_message_error_seconds: float = 90.0,
        monitor_stale_seconds: float = 90.0,
    ) -> dict[str, Any]:
        start_ts = self.engine.server_timestamp()
        user_task = asyncio.create_task(
            self.user_stream_service.run_stream(
                symbol=symbol,
                timestamp=start_ts,
                duration_seconds=duration_seconds,
                reconnect_delay_seconds=reconnect_delay_seconds,
                close_on_exit=False,
            )
        )
        cycle_results: list[dict[str, Any]] = []
        monitor_task = asyncio.create_task(
            self._monitor_runtime_loop(
                symbol=symbol,
                monitor_interval_seconds=monitor_interval_seconds,
                duration_seconds=duration_seconds,
                cycle_results=cycle_results,
            )
        )
        monitor_watchdog_task = asyncio.create_task(
            self._monitor_watchdog_loop(
                symbol=symbol,
                stale_after_seconds=monitor_stale_seconds,
                check_interval_seconds=max(5.0, min(15.0, monitor_stale_seconds / 3)),
                duration_seconds=duration_seconds,
            )
        )

        async def on_close(event: KlineCloseEvent) -> None:
            result = await self.handle_execution_kline_close(
                event=event,
                entry_notional_usdt=entry_notional_usdt,
                leverage=leverage,
                candle_limit=candle_limit,
            )
            cycle_results.append(result)

        async def on_mark_price(event: MarkPriceEvent) -> None:
            result = await self.handle_mark_price_tick(event=event)
            if result is not None:
                cycle_results.append(result)

        mark_task = asyncio.create_task(
            self.mark_price_stream_service.run_stream(
                symbol=symbol,
                on_event=on_mark_price,
                duration_seconds=duration_seconds,
                reconnect_delay_seconds=reconnect_delay_seconds,
                no_message_error_seconds=stream_no_message_error_seconds,
            )
        )
        kline_task = asyncio.create_task(
            self.kline_stream_service.run_stream(
                symbol=symbol,
                interval=runtime_stream_interval(self.runtime.settings),
                on_close=on_close,
                max_closed_events=max_closed_candles,
                duration_seconds=duration_seconds,
                reconnect_delay_seconds=reconnect_delay_seconds,
                no_message_error_seconds=stream_no_message_error_seconds,
            )
        )
        task_names = {
            user_task: "user_stream",
            mark_task: "mark_price_stream",
            monitor_task: "runtime_monitor",
            monitor_watchdog_task: "runtime_monitor_watchdog",
            kline_task: "kline_stream",
        }
        kline_result: dict[str, Any] | None = None
        terminal_error: str | None = None

        try:
            pending_tasks = set(task_names)
            while pending_tasks:
                done, pending_tasks = await asyncio.wait(
                    pending_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    task_name = task_names[task]
                    exc = task.exception()
                    if exc is not None:
                        message = "runtime background task failed"
                        self._record_watchdog_failure(
                            symbol=symbol,
                            code="runtime_background_task_failed",
                            event_type="runtime_background_task_failed",
                            message=message,
                            details={
                                "symbol": symbol,
                                "task": task_name,
                                "error": str(exc),
                            },
                        )
                        raise RuntimeError(f"{task_name} failed: {exc}") from exc
                    result = task.result()
                    if task is kline_task:
                        kline_result = result
                        pending_tasks.clear()
                        break
                    if duration_seconds is None:
                        message = "runtime background task completed unexpectedly"
                        self._record_watchdog_failure(
                            symbol=symbol,
                            code="runtime_background_task_failed",
                            event_type="runtime_background_task_failed",
                            message=message,
                            details={"symbol": symbol, "task": task_name},
                        )
                        raise RuntimeError(f"{task_name} completed unexpectedly")
        except Exception as exc:
            terminal_error = str(exc)
            raise
        finally:
            self.close_user_stream(symbol=symbol)
            if terminal_error is not None:
                self.runtime.update_runtime_status(
                    symbol=symbol,
                    state="ERROR",
                    last_error=terminal_error,
                )
            for task in (user_task, mark_task, monitor_task, monitor_watchdog_task, kline_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(
                user_task,
                mark_task,
                monitor_task,
                monitor_watchdog_task,
                kline_task,
                return_exceptions=True,
            )

        return {
            "symbol": symbol,
            "kline_stream": kline_result or {
                "symbol": symbol,
                "interval": runtime_stream_interval(self.runtime.settings),
                "closed_events": 0,
            },
            "cycle_results": cycle_results,
        }


def _maybe_to_dict(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return value


def _income_rows_include_realized_pnl(rows: list[dict[str, Any]], *, symbol: str) -> bool:
    target_symbol = symbol.upper()
    return any(
        str(row.get("symbol") or "").upper() == target_symbol
        and str(row.get("income_type") or row.get("incomeType") or "").upper() == "REALIZED_PNL"
        for row in rows
    )
