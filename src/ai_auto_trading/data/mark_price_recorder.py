from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterable

import websockets

from ai_auto_trading.settings import Settings, load_settings


def _utc_now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def _iso_from_ms(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()


@dataclass(frozen=True)
class MarkPriceEvent:
    event_type: str
    event_time_ms: int
    event_time_iso: str
    symbol: str
    mark_price: float
    mark_price_avg: float
    index_price: float
    estimated_settle_price: float
    funding_rate: float
    next_funding_time_ms: int
    next_funding_time_iso: str
    collected_at_ms: int
    collected_at_iso: str


@dataclass(frozen=True)
class RecorderHealthEvent:
    level: str
    event_type: str
    message: str
    details_json: str | None
    recorded_at_ms: int


@dataclass(frozen=True)
class RecordRunResult:
    symbol: str
    messages_recorded: int
    health_events_recorded: int
    database_path: Path


class MarkPriceRecorder:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        symbol: str | None = None,
        stream_speed: str = "1s",
        database_path: Path | None = None,
        reconnect_delay_seconds: float = 1.0,
    ) -> None:
        self.settings = settings or load_settings()
        self.symbol = (symbol or self.settings.trading_symbol).upper()
        self.stream_speed = stream_speed
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.database_path = (
            database_path
            or self.settings.data_dir
            / "runtime"
            / "mark_price"
            / f"{self.symbol.lower()}_mark_price.sqlite3"
        )

    @property
    def stream_name(self) -> str:
        if self.stream_speed == "1s":
            return f"{self.symbol.lower()}@markPrice@1s"
        return f"{self.symbol.lower()}@markPrice"

    @property
    def stream_url(self) -> str:
        base_url = self.settings.binance_futures_ws_base_url.rstrip("/")
        if "fstream.binance.com" in base_url:
            for known_route in ("/public", "/market", "/private"):
                if base_url.endswith(known_route):
                    base_url = base_url[: -len(known_route)]
                    break
            base_url = f"{base_url}/market"
        return f"{base_url}/ws/{self.stream_name}"

    @property
    def expected_interval_ms(self) -> int:
        return 1000 if self.stream_speed == "1s" else 3000

    def initialize_storage(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mark_price_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    event_time_ms INTEGER NOT NULL,
                    event_time_iso TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    mark_price REAL NOT NULL,
                    mark_price_avg REAL NOT NULL,
                    index_price REAL NOT NULL,
                    estimated_settle_price REAL NOT NULL,
                    funding_rate REAL NOT NULL,
                    next_funding_time_ms INTEGER NOT NULL,
                    next_funding_time_iso TEXT NOT NULL,
                    collected_at_ms INTEGER NOT NULL,
                    collected_at_iso TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recorder_health_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details_json TEXT,
                    recorded_at_ms INTEGER NOT NULL
                )
                """
            )
            conn.commit()

    def normalize_message(self, payload: dict[str, Any]) -> MarkPriceEvent:
        event_time_ms = int(payload["E"])
        next_funding_time_ms = int(payload["T"])
        collected_at_ms = _utc_now_ms()
        return MarkPriceEvent(
            event_type=str(payload["e"]),
            event_time_ms=event_time_ms,
            event_time_iso=_iso_from_ms(event_time_ms),
            symbol=str(payload["s"]),
            mark_price=float(payload["p"]),
            mark_price_avg=float(payload["ap"]),
            index_price=float(payload["i"]),
            estimated_settle_price=float(payload["P"]),
            funding_rate=float(payload["r"]),
            next_funding_time_ms=next_funding_time_ms,
            next_funding_time_iso=_iso_from_ms(next_funding_time_ms),
            collected_at_ms=collected_at_ms,
            collected_at_iso=_iso_from_ms(collected_at_ms),
        )

    def persist_event(self, event: MarkPriceEvent) -> None:
        with sqlite3.connect(self.database_path) as conn:
            conn.execute(
                """
                INSERT INTO mark_price_events (
                    event_type, event_time_ms, event_time_iso, symbol,
                    mark_price, mark_price_avg, index_price, estimated_settle_price,
                    funding_rate, next_funding_time_ms, next_funding_time_iso,
                    collected_at_ms, collected_at_iso
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_type,
                    event.event_time_ms,
                    event.event_time_iso,
                    event.symbol,
                    event.mark_price,
                    event.mark_price_avg,
                    event.index_price,
                    event.estimated_settle_price,
                    event.funding_rate,
                    event.next_funding_time_ms,
                    event.next_funding_time_iso,
                    event.collected_at_ms,
                    event.collected_at_iso,
                ),
            )
            conn.commit()

    def persist_health_event(
        self,
        *,
        level: str,
        event_type: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        record = RecorderHealthEvent(
            level=level,
            event_type=event_type,
            message=message,
            details_json=json.dumps(details, sort_keys=True) if details else None,
            recorded_at_ms=_utc_now_ms(),
        )
        with sqlite3.connect(self.database_path) as conn:
            conn.execute(
                """
                INSERT INTO recorder_health_events (
                    level, event_type, message, details_json, recorded_at_ms
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record.level,
                    record.event_type,
                    record.message,
                    record.details_json,
                    record.recorded_at_ms,
                ),
            )
            conn.commit()

    def persist_payload(
        self,
        payload: dict[str, Any],
        *,
        previous_event_time_ms: int | None,
    ) -> tuple[MarkPriceEvent, int]:
        event = self.normalize_message(payload)
        health_events = 0
        gap_details = self._gap_details(
            previous_event_time_ms=previous_event_time_ms,
            current_event_time_ms=event.event_time_ms,
        )
        if gap_details is not None:
            self.persist_health_event(
                level="WARNING",
                event_type="mark_price_gap_detected",
                message="mark price stream gap detected",
                details=gap_details,
            )
            health_events += 1
        self.persist_event(event)
        return event, health_events

    def record_payloads(self, payloads: Iterable[dict[str, Any]]) -> RecordRunResult:
        self.initialize_storage()
        messages = 0
        health = 0
        previous_event_time_ms: int | None = None
        self.persist_health_event(
            level="INFO",
            event_type="recorder_start",
            message="record_payloads started",
            details={"symbol": self.symbol},
        )
        health += 1
        for payload in payloads:
            event, gap_health_events = self.persist_payload(
                payload,
                previous_event_time_ms=previous_event_time_ms,
            )
            previous_event_time_ms = event.event_time_ms
            messages += 1
            health += gap_health_events
        self.persist_health_event(
            level="INFO",
            event_type="recorder_stop",
            message="record_payloads completed",
            details={"symbol": self.symbol, "messages_recorded": messages},
        )
        health += 1
        return RecordRunResult(
            symbol=self.symbol,
            messages_recorded=messages,
            health_events_recorded=health,
            database_path=self.database_path,
        )

    async def run(
        self,
        *,
        max_messages: int | None = None,
        duration_seconds: float | None = None,
    ) -> RecordRunResult:
        self.initialize_storage()
        self.persist_health_event(
            level="INFO",
            event_type="recorder_start",
            message="mark price recorder started",
            details={"symbol": self.symbol, "stream_url": self.stream_url},
        )
        health_events = 1
        messages_recorded = 0
        deadline = time.monotonic() + duration_seconds if duration_seconds else None
        previous_event_time_ms: int | None = None

        while True:
            if deadline is not None and time.monotonic() >= deadline:
                break
            if max_messages is not None and messages_recorded >= max_messages:
                break

            try:
                async with websockets.connect(self.stream_url) as websocket:
                    self.persist_health_event(
                        level="INFO",
                        event_type="websocket_connected",
                        message="mark price websocket connected",
                        details={"symbol": self.symbol},
                    )
                    health_events += 1
                    while True:
                        if deadline is not None and time.monotonic() >= deadline:
                            break
                        if max_messages is not None and messages_recorded >= max_messages:
                            break
                        raw = await asyncio.wait_for(websocket.recv(), timeout=5)
                        payload = json.loads(raw)
                        event, gap_health_events = self.persist_payload(
                            payload,
                            previous_event_time_ms=previous_event_time_ms,
                        )
                        previous_event_time_ms = event.event_time_ms
                        health_events += gap_health_events
                        messages_recorded += 1
                    if (deadline is not None and time.monotonic() >= deadline) or (
                        max_messages is not None and messages_recorded >= max_messages
                    ):
                        break
            except Exception as exc:
                self.persist_health_event(
                    level="WARNING",
                    event_type="websocket_reconnect",
                    message="mark price websocket reconnect scheduled",
                    details={"symbol": self.symbol, "error": str(exc)},
                )
                health_events += 1
                await asyncio.sleep(self.reconnect_delay_seconds)

        self.persist_health_event(
            level="INFO",
            event_type="recorder_stop",
            message="mark price recorder stopped",
            details={"symbol": self.symbol, "messages_recorded": messages_recorded},
        )
        health_events += 1
        return RecordRunResult(
            symbol=self.symbol,
            messages_recorded=messages_recorded,
            health_events_recorded=health_events,
            database_path=self.database_path,
        )

    def _gap_details(
        self,
        *,
        previous_event_time_ms: int | None,
        current_event_time_ms: int,
    ) -> dict[str, Any] | None:
        if previous_event_time_ms is None:
            return None

        gap_ms = current_event_time_ms - previous_event_time_ms
        threshold_ms = self.expected_interval_ms * 2
        if gap_ms < threshold_ms:
            return None

        missed_updates = max(1, (gap_ms // self.expected_interval_ms) - 1)
        return {
            "symbol": self.symbol,
            "stream_speed": self.stream_speed,
            "previous_event_time_ms": previous_event_time_ms,
            "current_event_time_ms": current_event_time_ms,
            "gap_ms": gap_ms,
            "expected_interval_ms": self.expected_interval_ms,
            "missed_updates_estimate": missed_updates,
        }
