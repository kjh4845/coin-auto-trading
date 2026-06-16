from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import json
import time
from typing import Any, Awaitable, Callable

import websockets

from ai_auto_trading.execution.testnet import BinanceFuturesTestnetClient
from ai_auto_trading.runtime.testnet import TestnetExecutionRuntime


@dataclass(frozen=True)
class MarkPriceEvent:
    symbol: str
    event_time_ms: int
    mark_price: float
    index_price: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TestnetMarkPriceStreamService:
    def __init__(
        self,
        client: BinanceFuturesTestnetClient,
        runtime: TestnetExecutionRuntime,
    ) -> None:
        self.client = client
        self.runtime = runtime

    def _raise_no_message_watchdog(
        self,
        *,
        symbol: str,
        elapsed_seconds: float,
        threshold_seconds: float,
    ) -> None:
        reason = "mark price websocket produced no messages within the watchdog window"
        details = {
            "symbol": symbol,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "threshold_seconds": threshold_seconds,
        }
        self.runtime.acquire_lockout(
            symbol=symbol,
            code="mark_price_stream_stale",
            reason=reason,
            details=details,
        )
        self.runtime.update_runtime_status(
            symbol=symbol,
            state="ERROR",
            last_error=reason,
        )
        self.runtime.record_incident(
            level="ERROR",
            event_type="mark_price_stream_stale",
            message=reason,
            details=details,
        )
        raise RuntimeError(reason)

    def ingest_payload(self, *, symbol: str, payload: dict[str, Any]) -> MarkPriceEvent | None:
        event_time_ms = int(payload.get("E", 0))
        mark_price = payload.get("p", payload.get("markPrice"))
        if mark_price in (None, ""):
            return None
        index_price = payload.get("i", payload.get("indexPrice"))
        event = MarkPriceEvent(
            symbol=symbol,
            event_time_ms=event_time_ms,
            mark_price=float(mark_price),
            index_price=float(index_price) if index_price not in (None, "") else None,
        )
        self.runtime.update_runtime_status(
            symbol=symbol,
            last_mark_price_event_ms=event_time_ms,
        )
        return event

    async def run_stream(
        self,
        *,
        symbol: str,
        speed: str = "1s",
        on_event: Callable[[MarkPriceEvent], Awaitable[None]] | None = None,
        max_messages: int | None = None,
        duration_seconds: float | None = None,
        reconnect_delay_seconds: float = 1.0,
        recv_timeout_seconds: float = 30.0,
        no_message_error_seconds: float = 90.0,
    ) -> dict[str, Any]:
        url = self.client.mark_price_stream_url(symbol=symbol, speed=speed)
        messages_processed = 0
        deadline = time.monotonic() + duration_seconds if duration_seconds else None
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                break
            if max_messages is not None and messages_processed >= max_messages:
                break
            try:
                async with websockets.connect(url) as websocket:
                    last_message_monotonic = time.monotonic()
                    stale_lockout_cleared = False
                    self.runtime.record_incident(
                        level="INFO",
                        event_type="mark_price_websocket_connected",
                        message="connected to mark price websocket",
                        details={"symbol": symbol, "speed": speed},
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
                            elapsed = time.monotonic() - last_message_monotonic
                            if elapsed >= no_message_error_seconds:
                                self._raise_no_message_watchdog(
                                    symbol=symbol,
                                    elapsed_seconds=elapsed,
                                    threshold_seconds=no_message_error_seconds,
                                )
                            continue
                        last_message_monotonic = time.monotonic()
                        if not stale_lockout_cleared:
                            self.runtime.release_lockout(
                                symbol=symbol,
                                code="mark_price_stream_stale",
                                reason="mark price websocket messages resumed",
                            )
                            self.runtime.release_lockout(
                                symbol=symbol,
                                code="runtime_background_task_failed",
                                reason="mark price websocket messages resumed",
                            )
                            stale_lockout_cleared = True
                        payload = json.loads(raw)
                        event = self.ingest_payload(symbol=symbol, payload=payload)
                        if event is None:
                            continue
                        if on_event is not None:
                            await on_event(event)
                        messages_processed += 1
                    if (deadline is not None and time.monotonic() >= deadline) or (
                        max_messages is not None and messages_processed >= max_messages
                    ):
                        break
            except asyncio.TimeoutError:
                continue
            except RuntimeError:
                raise
            except Exception as exc:
                self.runtime.record_incident(
                    level="WARNING",
                    event_type="mark_price_websocket_reconnect",
                    message="mark price websocket reconnect scheduled",
                    details={"symbol": symbol, "error": str(exc)},
                )
                await asyncio.sleep(reconnect_delay_seconds)
        return {
            "symbol": symbol,
            "speed": speed,
            "messages_processed": messages_processed,
        }
