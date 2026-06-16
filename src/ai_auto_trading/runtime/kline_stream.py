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
class KlineCloseEvent:
    symbol: str
    interval: str
    open_time: int
    close_time: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TestnetKlineStreamService:
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
        interval: str,
        elapsed_seconds: float,
        threshold_seconds: float,
    ) -> None:
        reason = (
            "execution kline websocket produced no messages within the watchdog window"
        )
        details = {
            "symbol": symbol,
            "interval": interval,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "threshold_seconds": threshold_seconds,
        }
        self.runtime.acquire_lockout(
            symbol=symbol,
            code="kline_stream_stale",
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
            event_type="kline_stream_stale",
            message=reason,
            details=details,
        )
        raise RuntimeError(reason)

    def ingest_payload(self, *, symbol: str, payload: dict[str, Any]) -> KlineCloseEvent | None:
        event_type = str(payload.get("e", ""))
        if event_type != "kline":
            return None
        kline = payload.get("k", {})
        if not bool(kline.get("x")):
            return None
        event = KlineCloseEvent(
            symbol=symbol,
            interval=str(kline["i"]),
            open_time=int(kline["t"]),
            close_time=int(kline["T"]),
            open_price=float(kline["o"]),
            high_price=float(kline["h"]),
            low_price=float(kline["l"]),
            close_price=float(kline["c"]),
            volume=float(kline["v"]),
        )
        self.runtime.update_runtime_status(
            symbol=symbol,
            last_execution_kline_close_ms=event.close_time,
        )
        self.runtime.record_incident(
            level="INFO",
            event_type="execution_kline_closed",
            message="execution kline close event received",
            details=event.to_dict(),
        )
        self.runtime.record_account_snapshot(
            symbol=symbol,
            snapshot_type="execution_kline_close",
            payload=event.to_dict(),
            recorded_at_ms=event.close_time,
        )
        return event

    async def run_stream(
        self,
        *,
        symbol: str,
        interval: str,
        on_close: Callable[[KlineCloseEvent], Awaitable[None]],
        max_closed_events: int | None = None,
        duration_seconds: float | None = None,
        reconnect_delay_seconds: float = 1.0,
        recv_timeout_seconds: float = 30.0,
        no_message_error_seconds: float = 90.0,
    ) -> dict[str, Any]:
        url = self.client.kline_stream_url(symbol=symbol, interval=interval)
        closed_events = 0
        deadline = time.monotonic() + duration_seconds if duration_seconds else None
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                break
            if max_closed_events is not None and closed_events >= max_closed_events:
                break
            try:
                async with websockets.connect(url) as websocket:
                    last_message_monotonic = time.monotonic()
                    stale_lockout_cleared = False
                    self.runtime.record_incident(
                        level="INFO",
                        event_type="kline_websocket_connected",
                        message="connected to execution kline websocket",
                        details={"symbol": symbol, "interval": interval},
                    )
                    while True:
                        if deadline is not None and time.monotonic() >= deadline:
                            break
                        if max_closed_events is not None and closed_events >= max_closed_events:
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
                                    interval=interval,
                                    elapsed_seconds=elapsed,
                                    threshold_seconds=no_message_error_seconds,
                                )
                            continue
                        last_message_monotonic = time.monotonic()
                        if not stale_lockout_cleared:
                            self.runtime.release_lockout(
                                symbol=symbol,
                                code="kline_stream_stale",
                                reason="execution kline websocket messages resumed",
                            )
                            self.runtime.release_lockout(
                                symbol=symbol,
                                code="runtime_background_task_failed",
                                reason="execution kline websocket messages resumed",
                            )
                            stale_lockout_cleared = True
                        payload = json.loads(raw)
                        event = self.ingest_payload(symbol=symbol, payload=payload)
                        if event is None:
                            continue
                        await on_close(event)
                        closed_events += 1
                    if (deadline is not None and time.monotonic() >= deadline) or (
                        max_closed_events is not None and closed_events >= max_closed_events
                    ):
                        break
            except asyncio.TimeoutError:
                continue
            except RuntimeError:
                raise
            except Exception as exc:
                self.runtime.record_incident(
                    level="WARNING",
                    event_type="kline_websocket_reconnect",
                    message="execution kline websocket reconnect scheduled",
                    details={"symbol": symbol, "interval": interval, "error": str(exc)},
                )
                await asyncio.sleep(reconnect_delay_seconds)
        return {
            "symbol": symbol,
            "interval": interval,
            "closed_events": closed_events,
        }
