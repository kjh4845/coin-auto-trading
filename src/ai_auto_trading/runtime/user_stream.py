from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import json
import time
from typing import Any

import websockets

from ai_auto_trading.execution.testnet import BinanceFuturesTestnetClient
from ai_auto_trading.runtime.testnet import TestnetExecutionRuntime


@dataclass(frozen=True)
class UserDataStreamSession:
    symbol: str
    listen_key: str
    stream_url: str
    started_at_ms: int
    last_keepalive_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TestnetUserStreamService:
    def __init__(
        self,
        client: BinanceFuturesTestnetClient,
        runtime: TestnetExecutionRuntime,
    ) -> None:
        self.client = client
        self.runtime = runtime

    def start_session(self, *, symbol: str, timestamp: int) -> UserDataStreamSession:
        payload = self.client.start_user_data_stream()
        listen_key = str(payload["listenKey"])
        session = UserDataStreamSession(
            symbol=symbol,
            listen_key=listen_key,
            stream_url=self.client.user_data_stream_url(listen_key=listen_key),
            started_at_ms=timestamp,
        )
        self.runtime.update_runtime_status(
            symbol=symbol,
            active_listen_key=listen_key,
            state="READY",
            last_user_stream_event_ms=timestamp,
            last_error="",
        )
        self.runtime.release_lockout(
            symbol=symbol,
            code="runtime_background_task_failed",
            reason="user data stream session started",
        )
        self.runtime.record_incident(
            level="INFO",
            event_type="user_stream_started",
            message="user data stream started",
            details=session.to_dict(),
        )
        return session

    def keepalive_session(self, *, symbol: str, timestamp: int) -> UserDataStreamSession:
        status = self.runtime.load_runtime_status(symbol)
        if status is None or not status.active_listen_key:
            raise ValueError("no active listen key for symbol")
        self.client.keepalive_user_data_stream(listen_key=status.active_listen_key)
        session = UserDataStreamSession(
            symbol=symbol,
            listen_key=status.active_listen_key,
            stream_url=self.client.user_data_stream_url(listen_key=status.active_listen_key),
            started_at_ms=status.updated_at_ms,
            last_keepalive_ms=timestamp,
        )
        self.runtime.update_runtime_status(
            symbol=symbol,
            last_user_stream_event_ms=timestamp,
        )
        self.runtime.record_incident(
            level="INFO",
            event_type="user_stream_keepalive",
            message="user data stream keepalive sent",
            details=session.to_dict(),
        )
        return session

    def close_session(self, *, symbol: str) -> None:
        status = self.runtime.load_runtime_status(symbol)
        if status is None or not status.active_listen_key:
            return
        self.client.close_user_data_stream(listen_key=status.active_listen_key)
        self.runtime.update_runtime_status(
            symbol=symbol,
            active_listen_key=None,
        )
        self.runtime.record_incident(
            level="INFO",
            event_type="user_stream_closed",
            message="user data stream closed",
            details={"symbol": symbol, "listen_key": status.active_listen_key},
        )

    def ingest_payload(self, *, symbol: str, payload: dict[str, Any]) -> None:
        event_time_ms = int(payload.get("E") or payload.get("T") or 0)
        self.runtime.update_runtime_status(
            symbol=symbol,
            last_user_stream_event_ms=event_time_ms,
        )
        event_type = str(payload.get("e", "UNKNOWN"))
        if event_type == "ORDER_TRADE_UPDATE":
            self.runtime.record_order_event(
                symbol=symbol,
                payload=payload,
                event_time_ms=event_time_ms,
            )
            self.runtime.record_incident(
                level="INFO",
                event_type="user_stream_order_update",
                message="order update received from user data stream",
                details={"symbol": symbol, "order_event_type": payload.get("o", {}).get("X")},
            )
            return

        if event_type == "ACCOUNT_UPDATE":
            self.runtime.record_account_snapshot(
                symbol=symbol,
                snapshot_type="user_stream_account_update",
                payload=payload,
                recorded_at_ms=event_time_ms,
            )
            account_update = payload.get("a", {})
            for position in account_update.get("P", []):
                if position.get("s") != symbol:
                    continue
                self.runtime.record_position_event(
                    symbol=symbol,
                    payload=position,
                    event_time_ms=event_time_ms,
                )
            self.runtime.record_incident(
                level="INFO",
                event_type="user_stream_account_update",
                message="account update received from user data stream",
                details={"symbol": symbol, "reason": account_update.get("m")},
            )
            return

        self.runtime.record_incident(
            level="INFO",
            event_type="user_stream_event",
            message="user data stream event received",
            details={"symbol": symbol, "event_type": event_type},
        )

    async def run_stream(
        self,
        *,
        symbol: str,
        timestamp: int,
        max_messages: int | None = None,
        duration_seconds: float | None = None,
        reconnect_delay_seconds: float = 1.0,
        recv_timeout_seconds: float = 30.0,
        keepalive_interval_seconds: float = 60.0,
        close_on_exit: bool = True,
    ) -> dict[str, Any]:
        session = self.start_session(symbol=symbol, timestamp=timestamp)
        messages_processed = 0
        deadline = time.monotonic() + duration_seconds if duration_seconds else None
        next_keepalive_monotonic = time.monotonic() + keepalive_interval_seconds
        try:
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                if max_messages is not None and messages_processed >= max_messages:
                    break
                try:
                    async with websockets.connect(session.stream_url) as websocket:
                        self.runtime.record_incident(
                            level="INFO",
                            event_type="user_stream_websocket_connected",
                            message="connected to user data stream websocket",
                            details={"symbol": symbol},
                        )
                        while True:
                            if deadline is not None and time.monotonic() >= deadline:
                                break
                            if max_messages is not None and messages_processed >= max_messages:
                                break
                            timeout = recv_timeout_seconds
                            if deadline is not None:
                                timeout = max(0.1, min(timeout, deadline - time.monotonic()))
                            try:
                                raw = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                            except asyncio.TimeoutError:
                                if time.monotonic() >= next_keepalive_monotonic:
                                    keepalive_ts = int(time.time() * 1000)
                                    self.keepalive_session(symbol=symbol, timestamp=keepalive_ts)
                                    next_keepalive_monotonic = time.monotonic() + keepalive_interval_seconds
                                continue
                            payload = json.loads(raw)
                            self.ingest_payload(symbol=symbol, payload=payload)
                            messages_processed += 1
                        if (deadline is not None and time.monotonic() >= deadline) or (
                            max_messages is not None and messages_processed >= max_messages
                        ):
                            break
                except Exception as exc:
                    self.runtime.record_incident(
                        level="WARNING",
                        event_type="user_stream_websocket_reconnect",
                        message="user data stream websocket reconnect scheduled",
                        details={"symbol": symbol, "error": str(exc)},
                    )
                    await asyncio.sleep(reconnect_delay_seconds)
        finally:
            if close_on_exit:
                self.close_session(symbol=symbol)
                expected_state = self.runtime.load_expected_state(symbol)
                managed_state = self.runtime.load_managed_trade_state(symbol)
                self.runtime.update_runtime_status(
                    symbol=symbol,
                    state="PAUSED" if expected_state is not None or managed_state is not None else "IDLE",
                )
        return {
            "symbol": symbol,
            "listen_key": session.listen_key,
            "messages_processed": messages_processed,
            "closed_on_exit": close_on_exit,
        }
