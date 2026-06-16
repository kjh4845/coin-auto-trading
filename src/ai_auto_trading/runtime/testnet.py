from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from contextlib import contextmanager
import json
import sqlite3
import time
from typing import Any, Iterator

from ai_auto_trading.execution.testnet import ReconciliationResult, TestnetExecutionEngine
from ai_auto_trading.models import OrderIntent, PolicyVersionInfo, PositionState, TradeRecord
from ai_auto_trading.settings import Settings, load_settings
from ai_auto_trading.strategy.runtime_profiles import runtime_strategy_context

_MISSING = object()
_SQLITE_BUSY_TIMEOUT_MS = 30_000


@contextmanager
def _runtime_db_connection(database_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(database_path, timeout=_SQLITE_BUSY_TIMEOUT_MS / 1000.0)
    try:
        conn.execute(f"PRAGMA busy_timeout = {_SQLITE_BUSY_TIMEOUT_MS}")
        yield conn
    finally:
        conn.close()


def _utc_now_ms() -> int:
    return int(time.time() * 1000)


@dataclass(frozen=True)
class ExecutionIncident:
    level: str
    event_type: str
    message: str
    details: dict[str, Any] | None
    recorded_at_ms: int

    def to_row(self) -> tuple[str, str, str, str | None, int]:
        return (
            self.level,
            self.event_type,
            self.message,
            json.dumps(self.details, sort_keys=True) if self.details is not None else None,
            self.recorded_at_ms,
        )


@dataclass(frozen=True)
class ExpectedRuntimeState:
    symbol: str
    expected_position_qty: float
    expected_stop_price_mark: float
    expected_leverage: int
    expected_margin_mode: str
    expected_position_mode: str
    updated_at_ms: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BundlePlacementResult:
    entry_order: dict[str, Any]
    hard_stop_order: dict[str, Any]
    expected_state: ExpectedRuntimeState


@dataclass(frozen=True)
class RuntimeRecoveryResult:
    ok: bool
    stored_state: ExpectedRuntimeState | None
    reconciliation: ReconciliationResult | None
    reason: str | None = None


@dataclass(frozen=True)
class RuntimeStatus:
    symbol: str
    state: str
    active_listen_key: str | None
    last_user_stream_event_ms: int | None
    last_mark_price_event_ms: int | None
    last_execution_kline_close_ms: int | None
    last_account_snapshot_ms: int | None
    last_reconcile_ms: int | None
    last_error: str | None
    updated_at_ms: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ManagedTradeState:
    symbol: str
    side: str
    quantity: float
    leverage_at_entry: float
    entry_contract_price_avg: float
    entry_mark_price: float
    execution_timeframe: str
    atr_trailing_multiplier: float
    max_holding_bars: int
    opened_at_ms: int
    signal_reason_codes: list[str]
    model_base: str
    adapter_version: str | None
    ai_snapshot: dict[str, Any] | None
    bars_held: int
    highest_high: float
    lowest_low: float
    last_processed_candle_close_time_ms: int | None
    atr_trail_history: list[dict[str, Any]]
    exit_policy: str = "atr_trail"
    fixed_take_profit_r: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RiskLockout:
    symbol: str
    code: str
    reason: str
    details_json: str | None
    activated_at_ms: int
    updated_at_ms: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["details"] = json.loads(self.details_json) if self.details_json else None
        return payload


@dataclass(frozen=True)
class TradeReview:
    trade_id: str
    symbol: str
    review_version: str
    closed_at_ms: int
    outcome: str
    primary_cause: str
    market_pattern: str
    explanation: str
    action_items: list[str]
    rule_change_candidates: list[dict[str, Any]]
    handling_decision: dict[str, Any]
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TestnetExecutionRuntime:
    def __init__(
        self,
        engine: TestnetExecutionEngine | None,
        *,
        settings: Settings | None = None,
        database_path: Path | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.engine = engine
        self.database_path = (
            database_path
            or self.settings.data_dir / "runtime" / "execution" / "testnet_execution.sqlite3"
        )

    def _require_engine(self) -> TestnetExecutionEngine:
        if self.engine is None:
            raise ValueError("execution engine is required for this runtime operation")
        return self.engine

    def initialize_storage(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with _runtime_db_connection(self.database_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details_json TEXT,
                    recorded_at_ms INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS expected_runtime_state (
                    symbol TEXT PRIMARY KEY,
                    expected_position_qty REAL NOT NULL,
                    expected_stop_price_mark REAL NOT NULL,
                    expected_leverage INTEGER NOT NULL,
                    expected_margin_mode TEXT NOT NULL,
                    expected_position_mode TEXT NOT NULL,
                    updated_at_ms INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runtime_status (
                    symbol TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    active_listen_key TEXT,
                    last_user_stream_event_ms INTEGER,
                    last_mark_price_event_ms INTEGER,
                    last_execution_kline_close_ms INTEGER,
                    last_account_snapshot_ms INTEGER,
                    last_reconcile_ms INTEGER,
                    last_error TEXT,
                    updated_at_ms INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_lockouts (
                    symbol TEXT NOT NULL,
                    code TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    details_json TEXT,
                    activated_at_ms INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL,
                    PRIMARY KEY(symbol, code)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS account_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    snapshot_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    recorded_at_ms INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS order_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    order_id TEXT,
                    client_order_id TEXT,
                    side TEXT,
                    order_type TEXT,
                    status TEXT,
                    reduce_only INTEGER NOT NULL,
                    quantity REAL,
                    price REAL,
                    stop_price REAL,
                    payload_json TEXT NOT NULL,
                    event_time_ms INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS position_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    position_side TEXT,
                    position_amount REAL,
                    entry_price REAL,
                    unrealized_pnl REAL,
                    margin_type TEXT,
                    payload_json TEXT NOT NULL,
                    event_time_ms INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS managed_trade_state (
                    symbol TEXT PRIMARY KEY,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    leverage_at_entry REAL NOT NULL,
                    entry_contract_price_avg REAL NOT NULL,
                    entry_mark_price REAL NOT NULL,
                    execution_timeframe TEXT NOT NULL,
                    atr_trailing_multiplier REAL NOT NULL,
                    max_holding_bars INTEGER NOT NULL,
                    opened_at_ms INTEGER NOT NULL,
                    signal_reason_codes_json TEXT NOT NULL,
                    model_base TEXT NOT NULL,
                    adapter_version TEXT,
                    ai_snapshot_json TEXT,
                    bars_held INTEGER NOT NULL,
                    highest_high REAL NOT NULL,
                    lowest_low REAL NOT NULL,
                    last_processed_candle_close_time_ms INTEGER,
                    atr_trail_history_json TEXT NOT NULL,
                    exit_policy TEXT NOT NULL DEFAULT 'atr_trail',
                    fixed_take_profit_r REAL NOT NULL DEFAULT 1.0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS closed_trade_records (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    exit_reason TEXT NOT NULL,
                    closed_at_ms INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_reviews (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    closed_at_ms INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS exchange_income_records (
                    income_key TEXT PRIMARY KEY,
                    related_trade_id TEXT,
                    symbol TEXT NOT NULL,
                    income_type TEXT NOT NULL,
                    income_usdt REAL NOT NULL,
                    asset TEXT NOT NULL,
                    info TEXT,
                    time_ms INTEGER NOT NULL,
                    tran_id TEXT,
                    trade_id TEXT,
                    payload_json TEXT NOT NULL,
                    synced_at_ms INTEGER NOT NULL
                )
                """
            )
            _ensure_column(conn, "runtime_status", "last_execution_kline_close_ms", "INTEGER")
            _ensure_column(conn, "order_events", "avg_price", "REAL")
            _ensure_column(conn, "managed_trade_state", "signal_reason_codes_json", "TEXT NOT NULL DEFAULT '[]'")
            _ensure_column(conn, "managed_trade_state", "model_base", "TEXT NOT NULL DEFAULT 'rule_only'")
            _ensure_column(conn, "managed_trade_state", "adapter_version", "TEXT")
            _ensure_column(conn, "managed_trade_state", "ai_snapshot_json", "TEXT")
            _ensure_column(conn, "managed_trade_state", "exit_policy", "TEXT NOT NULL DEFAULT 'atr_trail'")
            _ensure_column(conn, "managed_trade_state", "fixed_take_profit_r", "REAL NOT NULL DEFAULT 1.0")
            conn.commit()

    def record_incident(
        self,
        *,
        level: str,
        event_type: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> ExecutionIncident:
        self.initialize_storage()
        incident = ExecutionIncident(
            level=level,
            event_type=event_type,
            message=message,
            details=_json_ready(details) if details is not None else None,
            recorded_at_ms=_utc_now_ms(),
        )
        with _runtime_db_connection(self.database_path) as conn:
            conn.execute(
                """
                INSERT INTO execution_incidents (
                    level, event_type, message, details_json, recorded_at_ms
                ) VALUES (?, ?, ?, ?, ?)
                """,
                incident.to_row(),
            )
            conn.commit()
        return incident

    def save_expected_state(self, state: ExpectedRuntimeState) -> ExpectedRuntimeState:
        self.initialize_storage()
        with _runtime_db_connection(self.database_path) as conn:
            conn.execute(
                """
                INSERT INTO expected_runtime_state (
                    symbol,
                    expected_position_qty,
                    expected_stop_price_mark,
                    expected_leverage,
                    expected_margin_mode,
                    expected_position_mode,
                    updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    expected_position_qty=excluded.expected_position_qty,
                    expected_stop_price_mark=excluded.expected_stop_price_mark,
                    expected_leverage=excluded.expected_leverage,
                    expected_margin_mode=excluded.expected_margin_mode,
                    expected_position_mode=excluded.expected_position_mode,
                    updated_at_ms=excluded.updated_at_ms
                """,
                (
                    state.symbol,
                    state.expected_position_qty,
                    state.expected_stop_price_mark,
                    state.expected_leverage,
                    state.expected_margin_mode,
                    state.expected_position_mode,
                    state.updated_at_ms,
                ),
            )
            conn.commit()
        return state

    def load_expected_state(self, symbol: str) -> ExpectedRuntimeState | None:
        self.initialize_storage()
        with _runtime_db_connection(self.database_path) as conn:
            row = conn.execute(
                """
                SELECT
                    symbol,
                    expected_position_qty,
                    expected_stop_price_mark,
                    expected_leverage,
                    expected_margin_mode,
                    expected_position_mode,
                    updated_at_ms
                FROM expected_runtime_state
                WHERE symbol = ?
                """,
                (symbol,),
            ).fetchone()
        if row is None:
            return None
        return ExpectedRuntimeState(
            symbol=str(row[0]),
            expected_position_qty=float(row[1]),
            expected_stop_price_mark=float(row[2]),
            expected_leverage=int(row[3]),
            expected_margin_mode=str(row[4]),
            expected_position_mode=str(row[5]),
            updated_at_ms=int(row[6]),
        )

    def load_runtime_status(self, symbol: str) -> RuntimeStatus | None:
        self.initialize_storage()
        with _runtime_db_connection(self.database_path) as conn:
            row = conn.execute(
                """
                SELECT
                    symbol,
                    state,
                    active_listen_key,
                    last_user_stream_event_ms,
                    last_mark_price_event_ms,
                    last_execution_kline_close_ms,
                    last_account_snapshot_ms,
                    last_reconcile_ms,
                    last_error,
                    updated_at_ms
                FROM runtime_status
                WHERE symbol = ?
                """,
                (symbol,),
            ).fetchone()
        if row is None:
            return None
        return RuntimeStatus(
            symbol=str(row[0]),
            state=str(row[1]),
            active_listen_key=row[2],
            last_user_stream_event_ms=_int_or_none(row[3]),
            last_mark_price_event_ms=_int_or_none(row[4]),
            last_execution_kline_close_ms=_int_or_none(row[5]),
            last_account_snapshot_ms=_int_or_none(row[6]),
            last_reconcile_ms=_int_or_none(row[7]),
            last_error=row[8],
            updated_at_ms=int(row[9]),
        )

    def save_managed_trade_state(self, state: ManagedTradeState) -> ManagedTradeState:
        self.initialize_storage()
        with _runtime_db_connection(self.database_path) as conn:
            conn.execute(
                """
                INSERT INTO managed_trade_state (
                    symbol, side, quantity, leverage_at_entry, entry_contract_price_avg,
                    entry_mark_price, execution_timeframe, atr_trailing_multiplier,
                    max_holding_bars, opened_at_ms, signal_reason_codes_json, model_base,
                    adapter_version, ai_snapshot_json, bars_held, highest_high, lowest_low,
                    last_processed_candle_close_time_ms, atr_trail_history_json,
                    exit_policy, fixed_take_profit_r
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    side=excluded.side,
                    quantity=excluded.quantity,
                    leverage_at_entry=excluded.leverage_at_entry,
                    entry_contract_price_avg=excluded.entry_contract_price_avg,
                    entry_mark_price=excluded.entry_mark_price,
                    execution_timeframe=excluded.execution_timeframe,
                    atr_trailing_multiplier=excluded.atr_trailing_multiplier,
                    max_holding_bars=excluded.max_holding_bars,
                    opened_at_ms=excluded.opened_at_ms,
                    signal_reason_codes_json=excluded.signal_reason_codes_json,
                    model_base=excluded.model_base,
                    adapter_version=excluded.adapter_version,
                    ai_snapshot_json=excluded.ai_snapshot_json,
                    bars_held=excluded.bars_held,
                    highest_high=excluded.highest_high,
                    lowest_low=excluded.lowest_low,
                    last_processed_candle_close_time_ms=excluded.last_processed_candle_close_time_ms,
                    atr_trail_history_json=excluded.atr_trail_history_json,
                    exit_policy=excluded.exit_policy,
                    fixed_take_profit_r=excluded.fixed_take_profit_r
                """,
                (
                    state.symbol,
                    state.side,
                    state.quantity,
                    state.leverage_at_entry,
                    state.entry_contract_price_avg,
                    state.entry_mark_price,
                    state.execution_timeframe,
                    state.atr_trailing_multiplier,
                    state.max_holding_bars,
                    state.opened_at_ms,
                    json.dumps(_json_ready(state.signal_reason_codes), sort_keys=True),
                    state.model_base,
                    state.adapter_version,
                    json.dumps(_json_ready(state.ai_snapshot), sort_keys=True) if state.ai_snapshot is not None else None,
                    state.bars_held,
                    state.highest_high,
                    state.lowest_low,
                    state.last_processed_candle_close_time_ms,
                    json.dumps(_json_ready(state.atr_trail_history), sort_keys=True),
                    state.exit_policy,
                    state.fixed_take_profit_r,
                ),
            )
            conn.commit()
        return state

    def load_managed_trade_state(self, symbol: str) -> ManagedTradeState | None:
        self.initialize_storage()
        with _runtime_db_connection(self.database_path) as conn:
            row = conn.execute(
                """
                SELECT
                    symbol, side, quantity, leverage_at_entry, entry_contract_price_avg,
                    entry_mark_price, execution_timeframe, atr_trailing_multiplier,
                    max_holding_bars, opened_at_ms, signal_reason_codes_json, model_base,
                    adapter_version, ai_snapshot_json, bars_held, highest_high, lowest_low,
                    last_processed_candle_close_time_ms, atr_trail_history_json,
                    exit_policy, fixed_take_profit_r
                FROM managed_trade_state
                WHERE symbol = ?
                """,
                (symbol,),
            ).fetchone()
        if row is None:
            return None
        return ManagedTradeState(
            symbol=str(row[0]),
            side=str(row[1]),
            quantity=float(row[2]),
            leverage_at_entry=float(row[3]),
            entry_contract_price_avg=float(row[4]),
            entry_mark_price=float(row[5]),
            execution_timeframe=str(row[6]),
            atr_trailing_multiplier=float(row[7]),
            max_holding_bars=int(row[8]),
            opened_at_ms=int(row[9]),
            signal_reason_codes=json.loads(row[10]),
            model_base=str(row[11]),
            adapter_version=row[12],
            ai_snapshot=json.loads(row[13]) if row[13] else None,
            bars_held=int(row[14]),
            highest_high=float(row[15]),
            lowest_low=float(row[16]),
            last_processed_candle_close_time_ms=_int_or_none(row[17]),
            atr_trail_history=json.loads(row[18]),
            exit_policy=str(row[19]),
            fixed_take_profit_r=float(row[20]),
        )

    def update_runtime_status(
        self,
        *,
        symbol: str,
        state: str | object = _MISSING,
        active_listen_key: str | None | object = _MISSING,
        last_user_stream_event_ms: int | None | object = _MISSING,
        last_mark_price_event_ms: int | None | object = _MISSING,
        last_execution_kline_close_ms: int | None | object = _MISSING,
        last_account_snapshot_ms: int | None | object = _MISSING,
        last_reconcile_ms: int | None | object = _MISSING,
        last_error: str | None | object = _MISSING,
    ) -> RuntimeStatus:
        current = self.load_runtime_status(symbol)
        status = RuntimeStatus(
            symbol=symbol,
            state=current.state if current is not None else "IDLE",
            active_listen_key=current.active_listen_key if current is not None else None,
            last_user_stream_event_ms=current.last_user_stream_event_ms if current is not None else None,
            last_mark_price_event_ms=current.last_mark_price_event_ms if current is not None else None,
            last_execution_kline_close_ms=current.last_execution_kline_close_ms if current is not None else None,
            last_account_snapshot_ms=current.last_account_snapshot_ms if current is not None else None,
            last_reconcile_ms=current.last_reconcile_ms if current is not None else None,
            last_error=current.last_error if current is not None else None,
            updated_at_ms=_utc_now_ms(),
        )
        status = RuntimeStatus(
            symbol=status.symbol,
            state=status.state if state is _MISSING else str(state),
            active_listen_key=status.active_listen_key if active_listen_key is _MISSING else active_listen_key,
            last_user_stream_event_ms=(
                status.last_user_stream_event_ms
                if last_user_stream_event_ms is _MISSING
                else last_user_stream_event_ms
            ),
            last_mark_price_event_ms=(
                status.last_mark_price_event_ms
                if last_mark_price_event_ms is _MISSING
                else last_mark_price_event_ms
            ),
            last_execution_kline_close_ms=(
                status.last_execution_kline_close_ms
                if last_execution_kline_close_ms is _MISSING
                else last_execution_kline_close_ms
            ),
            last_account_snapshot_ms=(
                status.last_account_snapshot_ms
                if last_account_snapshot_ms is _MISSING
                else last_account_snapshot_ms
            ),
            last_reconcile_ms=(
                status.last_reconcile_ms
                if last_reconcile_ms is _MISSING
                else last_reconcile_ms
            ),
            last_error=status.last_error if last_error is _MISSING else last_error,
            updated_at_ms=status.updated_at_ms,
        )
        with _runtime_db_connection(self.database_path) as conn:
            conn.execute(
                """
                INSERT INTO runtime_status (
                    symbol,
                    state,
                    active_listen_key,
                    last_user_stream_event_ms,
                    last_mark_price_event_ms,
                    last_execution_kline_close_ms,
                    last_account_snapshot_ms,
                    last_reconcile_ms,
                    last_error,
                    updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    state=excluded.state,
                    active_listen_key=excluded.active_listen_key,
                    last_user_stream_event_ms=excluded.last_user_stream_event_ms,
                    last_mark_price_event_ms=excluded.last_mark_price_event_ms,
                    last_execution_kline_close_ms=excluded.last_execution_kline_close_ms,
                    last_account_snapshot_ms=excluded.last_account_snapshot_ms,
                    last_reconcile_ms=excluded.last_reconcile_ms,
                    last_error=excluded.last_error,
                    updated_at_ms=excluded.updated_at_ms
                """,
                (
                    status.symbol,
                    status.state,
                    status.active_listen_key,
                    status.last_user_stream_event_ms,
                    status.last_mark_price_event_ms,
                    status.last_execution_kline_close_ms,
                    status.last_account_snapshot_ms,
                    status.last_reconcile_ms,
                    status.last_error,
                    status.updated_at_ms,
                ),
            )
            conn.commit()
        return status

    def acquire_lockout(
        self,
        *,
        symbol: str,
        code: str,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> RiskLockout:
        self.initialize_storage()
        timestamp = _utc_now_ms()
        details_json = json.dumps(details, sort_keys=True) if details is not None else None
        existing = None
        with _runtime_db_connection(self.database_path) as conn:
            existing = conn.execute(
                """
                SELECT reason, details_json, activated_at_ms
                FROM risk_lockouts
                WHERE symbol = ? AND code = ?
                """,
                (symbol, code),
            ).fetchone()
        activated_at_ms = int(existing[2]) if existing is not None else timestamp
        with _runtime_db_connection(self.database_path) as conn:
            conn.execute(
                """
                INSERT INTO risk_lockouts (
                    symbol, code, reason, details_json, activated_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, code) DO UPDATE SET
                    reason=excluded.reason,
                    details_json=excluded.details_json,
                    updated_at_ms=excluded.updated_at_ms
                """,
                (symbol, code, reason, details_json, activated_at_ms, timestamp),
            )
            conn.commit()
        if existing is None:
            self.record_incident(
                level="WARNING",
                event_type="risk_lockout_acquired",
                message=f"risk lockout acquired: {code}",
                details={"symbol": symbol, "code": code, "reason": reason, "details": details},
            )
        return RiskLockout(
            symbol=symbol,
            code=code,
            reason=reason,
            details_json=details_json,
            activated_at_ms=activated_at_ms,
            updated_at_ms=timestamp,
        )

    def release_lockout(
        self,
        *,
        symbol: str,
        code: str,
        reason: str,
    ) -> None:
        self.initialize_storage()
        with _runtime_db_connection(self.database_path) as conn:
            deleted = conn.execute(
                "DELETE FROM risk_lockouts WHERE symbol = ? AND code = ?",
                (symbol, code),
            ).rowcount
            conn.commit()
        if deleted:
            self.record_incident(
                level="INFO",
                event_type="risk_lockout_released",
                message=f"risk lockout released: {code}",
                details={"symbol": symbol, "code": code, "reason": reason},
            )

    def active_lockouts(self, symbol: str) -> list[RiskLockout]:
        self.initialize_storage()
        with _runtime_db_connection(self.database_path) as conn:
            rows = conn.execute(
                """
                SELECT symbol, code, reason, details_json, activated_at_ms, updated_at_ms
                FROM risk_lockouts
                WHERE symbol = ?
                ORDER BY activated_at_ms ASC
                """,
                (symbol,),
            ).fetchall()
        return [
            RiskLockout(
                symbol=str(row[0]),
                code=str(row[1]),
                reason=str(row[2]),
                details_json=row[3],
                activated_at_ms=int(row[4]),
                updated_at_ms=int(row[5]),
            )
            for row in rows
        ]

    def recent_incidents(
        self,
        *,
        limit: int = 20,
        level: str | None = None,
        since_ms: int | None = None,
    ) -> list[ExecutionIncident]:
        self.initialize_storage()
        query = """
            SELECT level, event_type, message, details_json, recorded_at_ms
            FROM execution_incidents
            WHERE 1 = 1
        """
        params: list[Any] = []
        if level is not None:
            query += " AND level = ?"
            params.append(level)
        if since_ms is not None:
            query += " AND recorded_at_ms >= ?"
            params.append(since_ms)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with _runtime_db_connection(self.database_path) as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            ExecutionIncident(
                level=str(row[0]),
                event_type=str(row[1]),
                message=str(row[2]),
                details=json.loads(row[3]) if row[3] else None,
                recorded_at_ms=int(row[4]),
            )
            for row in rows
        ]

    def latest_account_snapshot(self, symbol: str) -> dict[str, Any] | None:
        self.initialize_storage()
        with _runtime_db_connection(self.database_path) as conn:
            row = conn.execute(
                """
                SELECT snapshot_type, payload_json, recorded_at_ms
                FROM account_snapshots
                WHERE symbol = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (symbol,),
            ).fetchone()
        if row is None:
            return None
        return {
            "snapshot_type": str(row[0]),
            "payload": json.loads(row[1]),
            "recorded_at_ms": int(row[2]),
        }

    def recent_order_events(
        self,
        symbol: str,
        *,
        limit: int = 10,
        since_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        self.initialize_storage()
        query = """
            SELECT
                event_type, order_id, client_order_id, side, order_type, status,
                reduce_only, quantity, price, avg_price, stop_price, event_time_ms
            FROM order_events
            WHERE symbol = ?
        """
        params: list[Any] = [symbol]
        if since_ms is not None:
            query += " AND event_time_ms >= ?"
            params.append(since_ms)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with _runtime_db_connection(self.database_path) as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {
                "event_type": str(row[0]),
                "order_id": row[1],
                "client_order_id": row[2],
                "side": row[3],
                "order_type": row[4],
                "status": row[5],
                "reduce_only": bool(row[6]),
                "quantity": row[7],
                "price": row[8],
                "avg_price": row[9],
                "stop_price": row[10],
                "event_time_ms": int(row[11]),
            }
            for row in rows
        ]

    def recent_position_events(
        self,
        symbol: str,
        *,
        limit: int = 10,
        since_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        self.initialize_storage()
        query = """
            SELECT
                position_side, position_amount, entry_price, unrealized_pnl,
                margin_type, event_time_ms
            FROM position_events
            WHERE symbol = ?
        """
        params: list[Any] = [symbol]
        if since_ms is not None:
            query += " AND event_time_ms >= ?"
            params.append(since_ms)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with _runtime_db_connection(self.database_path) as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {
                "position_side": row[0],
                "position_amount": row[1],
                "entry_price": row[2],
                "unrealized_pnl": row[3],
                "margin_type": row[4],
                "event_time_ms": int(row[5]),
            }
            for row in rows
        ]

    def persist_closed_trade_record(
        self,
        *,
        trade_record: TradeRecord,
        closed_at_ms: int,
    ) -> TradeRecord:
        self.initialize_storage()
        review = build_trade_review(trade_record=trade_record, closed_at_ms=closed_at_ms)
        with _runtime_db_connection(self.database_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO closed_trade_records (
                    trade_id, symbol, exit_reason, closed_at_ms, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    trade_record.trade_id,
                    trade_record.symbol,
                    trade_record.exit_reason,
                    closed_at_ms,
                    json.dumps(_json_ready(trade_record.to_dict()), sort_keys=True),
                ),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO trade_reviews (
                    trade_id, symbol, closed_at_ms, payload_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    review.trade_id,
                    review.symbol,
                    review.closed_at_ms,
                    json.dumps(_json_ready(review.to_dict()), sort_keys=True),
                ),
            )
            conn.commit()
        self._handle_closed_trade_review(review)
        return trade_record

    def persist_exchange_income_records(
        self,
        rows: list[dict[str, Any]],
        *,
        synced_at_ms: int,
        related_trade_id: str | None = None,
    ) -> list[dict[str, Any]]:
        self.initialize_storage()
        records = [
            record
            for record in (
                _exchange_income_record_from_payload(
                    row,
                    synced_at_ms=synced_at_ms,
                    related_trade_id=related_trade_id,
                )
                for row in rows
                if isinstance(row, dict)
            )
            if record is not None
        ]
        if not records:
            return []
        with _runtime_db_connection(self.database_path) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO exchange_income_records (
                    income_key, related_trade_id, symbol, income_type, income_usdt,
                    asset, info, time_ms, tran_id, trade_id, payload_json, synced_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record["income_key"],
                        record["related_trade_id"],
                        record["symbol"],
                        record["income_type"],
                        record["income_usdt"],
                        record["asset"],
                        record["info"],
                        record["time_ms"],
                        record["tran_id"],
                        record["trade_id"],
                        json.dumps(_json_ready(record["payload"]), sort_keys=True),
                        record["synced_at_ms"],
                    )
                    for record in records
                ],
            )
            conn.commit()
        return records

    def exchange_income_records_since(
        self,
        *,
        since_ms: int | None = None,
        until_ms: int | None = None,
        symbol: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self.initialize_storage()
        query = """
            SELECT
                income_key, related_trade_id, symbol, income_type, income_usdt,
                asset, info, time_ms, tran_id, trade_id, payload_json, synced_at_ms
            FROM exchange_income_records
            WHERE 1 = 1
        """
        params: list[Any] = []
        if since_ms is not None:
            query += " AND time_ms >= ?"
            params.append(since_ms)
        if until_ms is not None:
            query += " AND time_ms <= ?"
            params.append(until_ms)
        if symbol is not None:
            query += " AND symbol = ?"
            params.append(symbol)
        query += " ORDER BY time_ms ASC, income_type ASC, income_key ASC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with _runtime_db_connection(self.database_path) as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {
                "income_key": row[0],
                "related_trade_id": row[1],
                "symbol": row[2],
                "income_type": row[3],
                "income_usdt": float(row[4]),
                "asset": row[5],
                "info": row[6],
                "time_ms": int(row[7]),
                "tran_id": row[8],
                "trade_id": row[9],
                "payload": json.loads(row[10]),
                "synced_at_ms": int(row[11]),
            }
            for row in rows
        ]

    def _handle_closed_trade_review(self, review: TradeReview) -> None:
        if review.outcome != "LOSS":
            return

        self.record_incident(
            level="WARNING",
            event_type="loss_trade_review_generated",
            message="closed losing trade was automatically reviewed",
            details=review.to_dict(),
        )

        if review.primary_cause == "operational_safety_exit":
            self.acquire_lockout(
                symbol=review.symbol,
                code="manual_review_required",
                reason="loss review detected an operational safety exit",
                details=review.to_dict(),
            )
            self.update_runtime_status(
                symbol=review.symbol,
                state="PAUSED",
                last_error="loss review requires operational inspection",
            )
            return

        recent_reviews = self.recent_trade_reviews(review.symbol, limit=10)
        matching_losses = [
            item
            for item in recent_reviews
            if item.get("outcome") == "LOSS"
            and item.get("primary_cause") == review.primary_cause
        ]
        repeated_loss_threshold = 3
        if len(matching_losses) < repeated_loss_threshold:
            return

        matching_trade_ids = [
            str(item.get("trade_id"))
            for item in matching_losses[:repeated_loss_threshold]
        ]
        self.acquire_lockout(
            symbol=review.symbol,
            code="repeated_loss_pattern_review",
            reason=(
                f"loss cause {review.primary_cause} repeated "
                f"{repeated_loss_threshold} times in recent reviews"
            ),
            details={
                "primary_cause": review.primary_cause,
                "market_pattern": review.market_pattern,
                "matching_trade_ids": matching_trade_ids,
                "rule_change_candidates": review.rule_change_candidates,
                "handling_decision": review.handling_decision,
            },
        )
        self.update_runtime_status(
            symbol=review.symbol,
            state="PAUSED",
            last_error=f"repeated loss pattern requires review: {review.primary_cause}",
        )

    def recent_trade_records(
        self,
        symbol: str | None = None,
        *,
        limit: int = 10,
        since_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        self.initialize_storage()
        query = """
            SELECT closed_at_ms, payload_json
            FROM closed_trade_records
            WHERE 1 = 1
        """
        params: list[Any] = []
        if symbol is not None:
            query += " AND symbol = ?"
            params.append(symbol)
        if since_ms is not None:
            query += " AND closed_at_ms >= ?"
            params.append(since_ms)
        query += " ORDER BY closed_at_ms DESC LIMIT ?"
        params.append(limit)
        with _runtime_db_connection(self.database_path) as conn:
            rows = conn.execute(query, params).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            record = json.loads(row[1])
            record["closed_at_ms"] = int(row[0])
            records.append(record)
        return records

    def trade_records_since(
        self,
        symbol: str | None,
        *,
        since_ms: int,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self.initialize_storage()
        query = """
            SELECT closed_at_ms, payload_json
            FROM closed_trade_records
            WHERE closed_at_ms >= ?
        """
        params: list[Any] = [since_ms]
        if symbol is not None:
            query += " AND symbol = ?"
            params.append(symbol)
        query += " ORDER BY closed_at_ms DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with _runtime_db_connection(self.database_path) as conn:
            rows = conn.execute(query, params).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            record = json.loads(row[1])
            record["closed_at_ms"] = int(row[0])
            records.append(record)
        return records

    def recent_trade_reviews(
        self,
        symbol: str,
        *,
        limit: int = 10,
        since_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        self.initialize_storage()
        query = """
            SELECT payload_json
            FROM trade_reviews
            WHERE symbol = ?
        """
        params: list[Any] = [symbol]
        if since_ms is not None:
            query += " AND closed_at_ms >= ?"
            params.append(since_ms)
        query += " ORDER BY closed_at_ms DESC LIMIT ?"
        params.append(limit)
        with _runtime_db_connection(self.database_path) as conn:
            rows = conn.execute(query, params).fetchall()
        return [json.loads(row[0]) for row in rows]

    def account_risk_overview(self, symbol: str, *, now_ms: int | None = None) -> dict[str, Any]:
        self.initialize_storage()
        current_ms = now_ms or _utc_now_ms()
        window_ms = 86_400_000
        recent_window = self.trade_records_since(None, since_ms=current_ms - window_ms)
        daily_pnl = sum(_trade_realized_pnl(record) for record in recent_window)
        daily_realized_r = sum(_trade_realized_r(record) for record in recent_window)
        recent_trades = self.recent_trade_records(
            None,
            limit=max(10, self.settings.max_consecutive_losses * 2 or 10),
        )
        consecutive_losses = _consecutive_loss_count(recent_trades)
        last_trade = recent_trades[0] if recent_trades else None
        cooldown_ms = self.settings.cooldown_after_loss_minutes * 60_000
        cooldown_remaining_ms = 0
        last_trade_was_loss = False
        if last_trade is not None:
            last_trade_pnl = _trade_realized_pnl(last_trade)
            last_trade_was_loss = last_trade_pnl < 0.0
            if last_trade_was_loss and cooldown_ms > 0:
                cooldown_remaining_ms = max(
                    0,
                    cooldown_ms - (current_ms - _trade_closed_at_ms(last_trade)),
                )
        return {
            "scope": "account",
            "requested_symbol": symbol,
            "window_start_ms": current_ms - window_ms,
            "window_end_ms": current_ms,
            "recent_24h_pnl_after_fees_usdt": daily_pnl,
            "recent_24h_realized_r": daily_realized_r,
            "recent_24h_trade_count": len(recent_window),
            "consecutive_losses": consecutive_losses,
            "last_trade_was_loss": last_trade_was_loss,
            "cooldown_remaining_ms": cooldown_remaining_ms,
            "max_daily_loss_r": self.settings.max_daily_loss_r,
            "max_consecutive_losses": self.settings.max_consecutive_losses,
            "cooldown_after_loss_minutes": self.settings.cooldown_after_loss_minutes,
        }

    def trade_performance_summary(self, symbol: str, *, since_ms: int = 0) -> dict[str, Any]:
        self.initialize_storage()
        records = self.trade_records_since(symbol, since_ms=since_ms)
        if not records:
            return {
                "total_trade_count": 0,
                "decisive_trade_count": 0,
                "win_count": 0,
                "loss_count": 0,
                "flat_count": 0,
                "win_rate_pct": None,
                "total_realized_pnl_after_fees_usdt": 0.0,
                "total_realized_r": 0.0,
                "gross_profit_usdt": 0.0,
                "gross_loss_usdt": 0.0,
                "profit_factor": None,
                "expectancy_r_per_trade": 0.0,
                "average_pnl_after_fees_usdt": 0.0,
                "average_realized_r": 0.0,
                "average_win_usdt": None,
                "average_loss_usdt": None,
                "average_win_r": None,
                "average_loss_r": None,
                "best_trade_id": None,
                "best_trade_pnl_after_fees_usdt": None,
                "best_trade_r": None,
                "worst_trade_id": None,
                "worst_trade_pnl_after_fees_usdt": None,
                "worst_trade_r": None,
                "first_closed_at_ms": None,
                "last_closed_at_ms": None,
            }

        scored_records = [
            {
                "record": record,
                "pnl_usdt": _trade_realized_pnl(record),
                "realized_r": _trade_realized_r(record),
                "closed_at_ms": _trade_closed_at_ms(record),
            }
            for record in records
        ]
        wins = [item for item in scored_records if item["pnl_usdt"] > 0.0]
        losses = [item for item in scored_records if item["pnl_usdt"] < 0.0]
        flats = [item for item in scored_records if item["pnl_usdt"] == 0.0]
        decisive_trade_count = len(wins) + len(losses)
        total_pnl = sum(item["pnl_usdt"] for item in scored_records)
        total_r = sum(item["realized_r"] for item in scored_records)
        gross_profit = sum(item["pnl_usdt"] for item in wins)
        gross_loss = abs(sum(item["pnl_usdt"] for item in losses))
        best_trade = max(scored_records, key=lambda item: item["pnl_usdt"])
        worst_trade = min(scored_records, key=lambda item: item["pnl_usdt"])
        average_win_usdt = sum(item["pnl_usdt"] for item in wins) / len(wins) if wins else None
        average_loss_usdt = sum(item["pnl_usdt"] for item in losses) / len(losses) if losses else None
        average_win_r = sum(item["realized_r"] for item in wins) / len(wins) if wins else None
        average_loss_r = sum(item["realized_r"] for item in losses) / len(losses) if losses else None
        return {
            "total_trade_count": len(scored_records),
            "decisive_trade_count": decisive_trade_count,
            "win_count": len(wins),
            "loss_count": len(losses),
            "flat_count": len(flats),
            "win_rate_pct": (
                (len(wins) / decisive_trade_count) * 100.0
                if decisive_trade_count > 0
                else None
            ),
            "total_realized_pnl_after_fees_usdt": total_pnl,
            "total_realized_r": total_r,
            "gross_profit_usdt": gross_profit,
            "gross_loss_usdt": gross_loss,
            "profit_factor": (gross_profit / gross_loss) if gross_loss > 0.0 else None,
            "expectancy_r_per_trade": total_r / len(scored_records),
            "average_pnl_after_fees_usdt": total_pnl / len(scored_records),
            "average_realized_r": total_r / len(scored_records),
            "average_win_usdt": average_win_usdt,
            "average_loss_usdt": average_loss_usdt,
            "average_win_r": average_win_r,
            "average_loss_r": average_loss_r,
            "best_trade_id": best_trade["record"].get("trade_id"),
            "best_trade_pnl_after_fees_usdt": best_trade["pnl_usdt"],
            "best_trade_r": best_trade["realized_r"],
            "worst_trade_id": worst_trade["record"].get("trade_id"),
            "worst_trade_pnl_after_fees_usdt": worst_trade["pnl_usdt"],
            "worst_trade_r": worst_trade["realized_r"],
            "first_closed_at_ms": min(item["closed_at_ms"] for item in scored_records),
            "last_closed_at_ms": max(item["closed_at_ms"] for item in scored_records),
        }

    def latest_account_equity_usdt(self, symbol: str) -> float | None:
        self.initialize_storage()
        with _runtime_db_connection(self.database_path) as conn:
            rows = conn.execute(
                """
                SELECT payload_json
                FROM account_snapshots
                WHERE symbol = ? AND snapshot_type = 'user_stream_account_update'
                ORDER BY id DESC
                LIMIT 20
                """,
                (symbol,),
            ).fetchall()
        for row in rows:
            payload = json.loads(row[0])
            account_update = payload.get("a", {})
            balances = account_update.get("B", [])
            if not isinstance(balances, list):
                continue
            for balance in balances:
                if str(balance.get("a")) != "USDT":
                    continue
                for key in ("cw", "wb"):
                    value = balance.get(key)
                    if value in (None, ""):
                        continue
                    return float(value)
        return None

    def forward_trade_progress(
        self,
        symbol: str,
        *,
        target_min_trades: int = 30,
        target_max_trades: int = 50,
        now_ms: int | None = None,
    ) -> dict[str, Any]:
        self.initialize_storage()
        current_ms = now_ms or _utc_now_ms()
        records = sorted(
            self.trade_records_since(symbol, since_ms=0),
            key=_trade_closed_at_ms,
        )
        trade_count = len(records)
        first_closed_at_ms = _trade_closed_at_ms(records[0]) if records else None
        last_closed_at_ms = _trade_closed_at_ms(records[-1]) if records else None
        observed_days = 0.0
        if first_closed_at_ms is not None and last_closed_at_ms is not None:
            observed_days = max(
                (last_closed_at_ms - first_closed_at_ms) / 86_400_000.0,
                0.0,
            )
        lookback_7d = current_ms - (7 * 86_400_000)
        lookback_30d = current_ms - (30 * 86_400_000)
        trades_last_7d = sum(
            1 for record in records if _trade_closed_at_ms(record) >= lookback_7d
        )
        trades_last_30d = sum(
            1 for record in records if _trade_closed_at_ms(record) >= lookback_30d
        )
        trade_rate_per_day = (
            trade_count / max(observed_days, 1.0)
            if trade_count > 0
            else 0.0
        )
        remaining_to_min = max(0, target_min_trades - trade_count)
        remaining_to_max = max(0, target_max_trades - trade_count)
        return {
            "closed_trade_count": trade_count,
            "target_min_trades": target_min_trades,
            "target_max_trades": target_max_trades,
            "remaining_to_min": remaining_to_min,
            "remaining_to_max": remaining_to_max,
            "ready_for_sizing_review": trade_count >= target_min_trades,
            "first_closed_at_ms": first_closed_at_ms,
            "last_closed_at_ms": last_closed_at_ms,
            "observed_days": observed_days,
            "trades_last_7d": trades_last_7d,
            "trades_last_30d": trades_last_30d,
            "average_trades_per_day": trade_rate_per_day,
            "estimated_days_to_min": (
                remaining_to_min / trade_rate_per_day
                if trade_rate_per_day > 0 and remaining_to_min > 0
                else 0.0 if remaining_to_min == 0 else None
            ),
            "estimated_days_to_max": (
                remaining_to_max / trade_rate_per_day
                if trade_rate_per_day > 0 and remaining_to_max > 0
                else 0.0 if remaining_to_max == 0 else None
            ),
        }

    def position_sizing_review(
        self,
        symbol: str,
        *,
        min_trade_count: int = 30,
        target_trade_count: int = 50,
        entry_notional_usdt: float = 1000.0,
        leverage: float | None = None,
        account_equity_usdt: float | None = None,
    ) -> dict[str, Any]:
        self.initialize_storage()
        records = sorted(
            self.trade_records_since(symbol, since_ms=0),
            key=_trade_closed_at_ms,
        )
        resolved_leverage = leverage or float(self.settings.live_start_leverage)
        current_one_r_usdt = (entry_notional_usdt / resolved_leverage) * 0.05
        progress = self.forward_trade_progress(
            symbol,
            target_min_trades=min_trade_count,
            target_max_trades=target_trade_count,
        )
        if len(records) < min_trade_count:
            return {
                "eligible": False,
                "reason": (
                    f"need at least {min_trade_count} closed trades before sizing review"
                ),
                "trade_count": len(records),
                "current_entry_notional_usdt": entry_notional_usdt,
                "leverage": resolved_leverage,
                "current_one_r_usdt": current_one_r_usdt,
                "progress": progress,
            }

        scored_records = [
            {
                "trade_id": record.get("trade_id"),
                "closed_at_ms": _trade_closed_at_ms(record),
                "pnl_usdt": _trade_realized_pnl(record),
                "realized_r": _trade_realized_r(record),
            }
            for record in records
        ]
        decisive_records = [
            item for item in scored_records if item["pnl_usdt"] != 0.0
        ]
        wins = [item for item in decisive_records if item["pnl_usdt"] > 0.0]
        losses = [item for item in decisive_records if item["pnl_usdt"] < 0.0]
        total_r = sum(item["realized_r"] for item in scored_records)
        max_drawdown_r = _max_drawdown_r(scored_records)
        worst_rolling_24h_r = _worst_rolling_window_r(
            scored_records,
            window_ms=86_400_000,
        )
        max_loss_streak = _max_consecutive_loss_count(scored_records)
        average_win_r = (
            sum(item["realized_r"] for item in wins) / len(wins)
            if wins
            else None
        )
        average_loss_r = (
            sum(item["realized_r"] for item in losses) / len(losses)
            if losses
            else None
        )
        win_rate = (
            len(wins) / len(decisive_records)
            if decisive_records
            else 0.0
        )
        kelly_fraction = None
        if (
            average_win_r is not None
            and average_loss_r is not None
            and average_win_r > 0.0
            and average_loss_r < 0.0
        ):
            reward_to_risk = average_win_r / abs(average_loss_r)
            if reward_to_risk > 0.0:
                kelly_fraction = max(
                    0.0,
                    win_rate - ((1.0 - win_rate) / reward_to_risk),
                )

        recommended_fractions = {
            "conservative": (
                min(kelly_fraction * 0.25, 0.005)
                if kelly_fraction is not None
                else None
            ),
            "balanced": (
                min(kelly_fraction * 0.5, 0.01)
                if kelly_fraction is not None
                else None
            ),
            "aggressive": (
                min(kelly_fraction, 0.02)
                if kelly_fraction is not None
                else None
            ),
        }

        resolved_equity = account_equity_usdt
        equity_source = "argument" if account_equity_usdt is not None else None
        if resolved_equity is None:
            resolved_equity = self.latest_account_equity_usdt(symbol)
            if resolved_equity is not None:
                equity_source = "user_stream_account_update"

        candidate_sizing = {}
        if resolved_equity is not None and resolved_equity > 0.0:
            for label, fraction in recommended_fractions.items():
                if fraction is None or fraction <= 0.0:
                    continue
                one_r_usdt = resolved_equity * fraction
                implied_notional = one_r_usdt * resolved_leverage / 0.05
                candidate_sizing[label] = {
                    "risk_fraction_of_equity": fraction,
                    "one_r_usdt": one_r_usdt,
                    "implied_entry_notional_usdt": implied_notional,
                    "notional_multiplier_vs_current": (
                        implied_notional / entry_notional_usdt
                        if entry_notional_usdt > 0.0
                        else None
                    ),
                    "sample_total_pnl_usdt": total_r * one_r_usdt,
                    "sample_max_drawdown_usdt": max_drawdown_r * one_r_usdt,
                    "sample_worst_24h_loss_usdt": abs(min(0.0, worst_rolling_24h_r)) * one_r_usdt,
                }

        return {
            "eligible": True,
            "trade_count": len(records),
            "current_entry_notional_usdt": entry_notional_usdt,
            "leverage": resolved_leverage,
            "current_one_r_usdt": current_one_r_usdt,
            "account_equity_usdt": resolved_equity,
            "account_equity_source": equity_source,
            "progress": progress,
            "edge_stats": {
                "total_realized_r": total_r,
                "average_realized_r": total_r / len(scored_records),
                "win_rate_pct": win_rate * 100.0 if decisive_records else None,
                "average_win_r": average_win_r,
                "average_loss_r": average_loss_r,
                "profit_factor": (
                    sum(item["pnl_usdt"] for item in wins) / abs(sum(item["pnl_usdt"] for item in losses))
                    if wins and losses and sum(item["pnl_usdt"] for item in losses) != 0.0
                    else None
                ),
                "kelly_fraction_of_equity": kelly_fraction,
            },
            "risk_stats": {
                "max_drawdown_r": max_drawdown_r,
                "worst_rolling_24h_realized_r": worst_rolling_24h_r,
                "max_consecutive_losses": max_loss_streak,
            },
            "recommended_risk_fraction_of_equity": recommended_fractions,
            "candidate_position_sizes": candidate_sizing,
        }

    def record_account_snapshot(
        self,
        *,
        symbol: str,
        snapshot_type: str,
        payload: dict[str, Any],
        recorded_at_ms: int | None = None,
    ) -> None:
        self.initialize_storage()
        timestamp = recorded_at_ms or _utc_now_ms()
        with _runtime_db_connection(self.database_path) as conn:
            conn.execute(
                """
                INSERT INTO account_snapshots (
                    symbol, snapshot_type, payload_json, recorded_at_ms
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    symbol,
                    snapshot_type,
                    json.dumps(_json_ready(payload), sort_keys=True),
                    timestamp,
                ),
            )
            conn.commit()
        self.update_runtime_status(
            symbol=symbol,
            last_account_snapshot_ms=timestamp,
        )

    def record_order_event(
        self,
        *,
        symbol: str,
        payload: dict[str, Any],
        event_time_ms: int,
    ) -> None:
        order = payload.get("o", payload)
        with _runtime_db_connection(self.database_path) as conn:
            conn.execute(
                """
                INSERT INTO order_events (
                    symbol, event_type, order_id, client_order_id, side, order_type,
                    status, reduce_only, quantity, price, avg_price, stop_price, payload_json, event_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    str(payload.get("e", "ORDER_EVENT")),
                    str(order.get("i", order.get("orderId", ""))),
                    str(order.get("c", order.get("clientOrderId", ""))),
                    str(order.get("S", order.get("side", ""))),
                    str(order.get("o", order.get("type", order.get("orderType", "")))),
                    str(order.get("X", order.get("status", ""))),
                    1 if bool(order.get("R", order.get("reduceOnly", False))) else 0,
                    _float_or_none(order.get("q", order.get("origQty", order.get("quantity")))),
                    _float_or_none(order.get("p", order.get("price"))),
                    _float_or_none(order.get("ap", order.get("avgPrice"))),
                    _float_or_none(order.get("sp", order.get("stopPrice", order.get("triggerPrice")))),
                    json.dumps(_json_ready(payload), sort_keys=True),
                    event_time_ms,
                ),
            )
            conn.commit()

    def record_position_event(
        self,
        *,
        symbol: str,
        payload: dict[str, Any],
        event_time_ms: int,
    ) -> None:
        with _runtime_db_connection(self.database_path) as conn:
            conn.execute(
                """
                INSERT INTO position_events (
                    symbol, position_side, position_amount, entry_price, unrealized_pnl,
                    margin_type, payload_json, event_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    str(payload.get("ps", payload.get("positionSide", ""))),
                    _float_or_none(payload.get("pa", payload.get("positionAmt"))),
                    _float_or_none(payload.get("ep", payload.get("entryPrice"))),
                    _float_or_none(payload.get("up", payload.get("unRealizedProfit"))),
                    str(payload.get("mt", payload.get("marginType", ""))),
                    json.dumps(_json_ready(payload), sort_keys=True),
                    event_time_ms,
                ),
            )
            conn.commit()

    def clear_expected_state(self, symbol: str) -> None:
        self.initialize_storage()
        with _runtime_db_connection(self.database_path) as conn:
            conn.execute(
                "DELETE FROM expected_runtime_state WHERE symbol = ?",
                (symbol,),
            )
            conn.commit()

    def clear_managed_trade_state(self, symbol: str) -> None:
        self.initialize_storage()
        with _runtime_db_connection(self.database_path) as conn:
            conn.execute(
                "DELETE FROM managed_trade_state WHERE symbol = ?",
                (symbol,),
            )
            conn.commit()

    def place_entry_and_protection(
        self,
        *,
        entry_intent: OrderIntent,
        hard_stop_intent: OrderIntent,
        timestamp: int,
        expected_leverage: int,
        expected_margin_mode: str = "ISOLATED",
        expected_position_mode: str = "ONE_WAY",
        skip_preflight: bool = False,
        signal_reason_codes: list[str] | None = None,
        model_base: str = "rule_only",
        adapter_version: str | None = None,
        ai_snapshot: dict[str, Any] | None = None,
        execution_timeframe: str | None = None,
    ) -> BundlePlacementResult:
        self.initialize_storage()
        self.update_runtime_status(symbol=entry_intent.symbol, state="ENTRY_PENDING", last_error=None)
        engine = self._require_engine()
        try:
            prepared = engine.prepare_entry_and_protection(
                entry_intent=entry_intent,
                hard_stop_intent=hard_stop_intent,
                timestamp=timestamp,
                preflight=not skip_preflight,
            )
            result = engine.place_entry_and_protection(
                entry_intent=prepared.entry_intent,
                hard_stop_intent=prepared.hard_stop_intent,
                timestamp=timestamp,
                expected_leverage=expected_leverage,
                expected_margin_mode=expected_margin_mode,
                expected_position_mode=expected_position_mode,
                skip_preflight=True,
            )
        except Exception as exc:
            self.record_incident(
                level="ERROR",
                event_type="bundle_place_failed",
                message="entry/protection placement failed",
                details={"symbol": entry_intent.symbol, "error": str(exc)},
            )
            self.update_runtime_status(
                symbol=entry_intent.symbol,
                state="PAUSED",
                last_error=str(exc),
            )
            raise

        expected_state = ExpectedRuntimeState(
            symbol=entry_intent.symbol,
            expected_position_qty=_filled_quantity(
                result.entry_order,
                fallback=prepared.entry_intent.quantity,
            ),
            expected_stop_price_mark=float(prepared.hard_stop_intent.stop_price or 0.0),
            expected_leverage=expected_leverage,
            expected_margin_mode=expected_margin_mode,
            expected_position_mode=expected_position_mode,
            updated_at_ms=_utc_now_ms(),
        )
        self.save_expected_state(expected_state)
        managed_trade_state = ManagedTradeState(
            symbol=entry_intent.symbol,
            side="LONG" if entry_intent.side == "BUY" else "SHORT",
            quantity=expected_state.expected_position_qty,
            leverage_at_entry=float(expected_leverage),
            entry_contract_price_avg=_float_or_none(
                result.entry_order.get("avgPrice", result.entry_order.get("price"))
            ) or prepared.reference_mark_price,
            entry_mark_price=float(prepared.reference_mark_price),
            execution_timeframe=execution_timeframe or self.settings.execution_timeframe,
            atr_trailing_multiplier=self.settings.atr_trailing_multiplier,
            max_holding_bars=self.settings.max_holding_bars,
            opened_at_ms=int(result.entry_order.get("updateTime", timestamp)),
            signal_reason_codes=list(signal_reason_codes or []),
            model_base=model_base,
            adapter_version=adapter_version,
            ai_snapshot=_json_ready(ai_snapshot) if ai_snapshot is not None else None,
            bars_held=0,
            highest_high=_float_or_none(result.entry_order.get("avgPrice")) or prepared.reference_mark_price,
            lowest_low=_float_or_none(result.entry_order.get("avgPrice")) or prepared.reference_mark_price,
            last_processed_candle_close_time_ms=None,
            atr_trail_history=[],
            exit_policy=self.settings.exit_policy,
            fixed_take_profit_r=self.settings.fixed_take_profit_r,
        )
        self.save_managed_trade_state(managed_trade_state)
        self.record_incident(
            level="INFO",
            event_type="bundle_placed",
            message="entry and protective stop placed",
            details={
                "symbol": expected_state.symbol,
                "expected_position_qty": expected_state.expected_position_qty,
                "expected_stop_price_mark": expected_state.expected_stop_price_mark,
                "expected_leverage": expected_state.expected_leverage,
                "expected_margin_mode": expected_state.expected_margin_mode,
                "expected_position_mode": expected_state.expected_position_mode,
                "managed_trade_state": managed_trade_state,
            },
        )
        self.update_runtime_status(
            symbol=entry_intent.symbol,
            state="PROTECTED",
            last_error=None,
        )
        return BundlePlacementResult(
            entry_order=result.entry_order,
            hard_stop_order=result.hard_stop_order,
            expected_state=expected_state,
        )

    def recover_and_reconcile(self, *, symbol: str, timestamp: int) -> RuntimeRecoveryResult:
        self.initialize_storage()
        self.update_runtime_status(symbol=symbol, state="RECONCILING", last_reconcile_ms=timestamp)
        engine = self._require_engine()
        stored_state = self.load_expected_state(symbol)
        if stored_state is None:
            self.record_incident(
                level="WARNING",
                event_type="recovery_missing_state",
                message="no stored runtime state for symbol",
                details={"symbol": symbol},
            )
            return RuntimeRecoveryResult(
                ok=False,
                stored_state=None,
                reconciliation=None,
                reason="missing_expected_runtime_state",
            )

        reconciliation = engine.reconcile_state(
            symbol=symbol,
            expected_position_qty=stored_state.expected_position_qty,
            expected_stop_price_mark=stored_state.expected_stop_price_mark,
            expected_leverage=stored_state.expected_leverage,
            expected_margin_mode=stored_state.expected_margin_mode,
            expected_position_mode=stored_state.expected_position_mode,
            timestamp=timestamp,
        )
        self.record_incident(
            level="INFO" if reconciliation.ok else "WARNING",
            event_type="recovery_reconciliation",
            message="runtime recovery reconciliation completed",
            details={
                "symbol": symbol,
                "ok": reconciliation.ok,
                "mismatches": reconciliation.mismatches,
                "stored_state": stored_state.to_dict(),
            },
        )
        self.record_account_snapshot(
            symbol=symbol,
            snapshot_type="rest_reconciliation",
            payload={
                "ok": reconciliation.ok,
                "mismatches": reconciliation.mismatches,
                "remote_state": reconciliation.remote_state,
            },
            recorded_at_ms=timestamp,
        )
        self.update_runtime_status(
            symbol=symbol,
            state="PROTECTED" if reconciliation.ok and stored_state.expected_position_qty > 0 else "READY",
            last_reconcile_ms=timestamp,
            last_error=None if reconciliation.ok else ",".join(reconciliation.mismatches),
        )
        return RuntimeRecoveryResult(
            ok=reconciliation.ok,
            stored_state=stored_state,
            reconciliation=reconciliation,
        )

    def runtime_summary(
        self,
        symbol: str,
        *,
        incident_limit: int = 10,
        display_since_ms: int | None = None,
    ) -> dict[str, Any]:
        trade_since_ms = display_since_ms if display_since_ms is not None else 0
        return {
            "strategy_context": runtime_strategy_context(self.settings),
            "runtime_status": (
                self.load_runtime_status(symbol).to_dict()
                if self.load_runtime_status(symbol) is not None
                else None
            ),
            "expected_state": (
                self.load_expected_state(symbol).to_dict()
                if self.load_expected_state(symbol) is not None
                else None
            ),
            "managed_trade_state": (
                self.load_managed_trade_state(symbol).to_dict()
                if self.load_managed_trade_state(symbol) is not None
                else None
            ),
            "active_lockouts": [lockout.to_dict() for lockout in self.active_lockouts(symbol)],
            "recent_incidents": [
                incident.__dict__
                for incident in self.recent_incidents(
                    limit=incident_limit,
                    since_ms=display_since_ms,
                )
            ],
            "latest_account_snapshot": self.latest_account_snapshot(symbol),
            "recent_order_events": self.recent_order_events(
                symbol,
                limit=incident_limit,
                since_ms=display_since_ms,
            ),
            "recent_position_events": self.recent_position_events(
                symbol,
                limit=incident_limit,
                since_ms=display_since_ms,
            ),
            "recent_trade_records": self.recent_trade_records(
                symbol,
                limit=incident_limit,
                since_ms=display_since_ms,
            ),
            "recent_trade_reviews": self.recent_trade_reviews(
                symbol,
                limit=incident_limit,
                since_ms=display_since_ms,
            ),
            "account_risk_overview": self.account_risk_overview(symbol),
            "trade_performance_summary": self.trade_performance_summary(
                symbol,
                since_ms=trade_since_ms,
            ),
            "display_since_ms": display_since_ms,
            "database_path": str(self.database_path),
        }


def _filled_quantity(order: dict[str, Any], *, fallback: float) -> float:
    for key in ("executedQty", "cumQty"):
        value = order.get(key)
        if value is None:
            continue
        quantity = float(value)
        if quantity > 0:
            return quantity
    return fallback


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return _json_ready(asdict(value))
    return value


def _exchange_income_record_from_payload(
    row: dict[str, Any],
    *,
    synced_at_ms: int,
    related_trade_id: str | None,
) -> dict[str, Any] | None:
    income = _float_or_none(row.get("income"))
    time_ms = _int_or_none(row.get("time"))
    if income is None or time_ms is None:
        return None
    symbol = str(row.get("symbol") or "").upper()
    income_type = str(row.get("incomeType") or row.get("income_type") or "").upper()
    asset = str(row.get("asset") or "").upper()
    tran_id = str(row.get("tranId") or row.get("trandId") or row.get("tran_id") or "")
    trade_id = str(row.get("tradeId") or row.get("trade_id") or "")
    info = str(row.get("info") or "")
    income_key = "|".join(
        [
            income_type,
            tran_id,
            trade_id,
            symbol,
            str(row.get("income") or income),
            str(time_ms),
        ]
    )
    return {
        "income_key": income_key,
        "related_trade_id": related_trade_id,
        "symbol": symbol,
        "income_type": income_type,
        "income_usdt": income,
        "asset": asset,
        "info": info,
        "time_ms": time_ms,
        "tran_id": tran_id,
        "trade_id": trade_id,
        "payload": row,
        "synced_at_ms": synced_at_ms,
    }


def build_runtime_trade_record(
    *,
    trade_state: ManagedTradeState,
    exit_reason: str,
    exit_contract_price: float,
    exit_mark_price: float,
    closed_at_ms: int,
    notes: str,
) -> TradeRecord:
    position = PositionState(
        side=trade_state.side,
        quantity=trade_state.quantity,
        leverage_at_entry=trade_state.leverage_at_entry,
        entry_contract_price_avg=trade_state.entry_contract_price_avg,
        entry_mark_price=trade_state.entry_mark_price,
        symbol=trade_state.symbol,
    )
    policy = PolicyVersionInfo(
        policy_version="policy_v1",
        strategy_version="strategy_v1",
        feature_schema_version="features_v1",
        model_base=trade_state.model_base,
        adapter_version=trade_state.adapter_version,
        runtime_version="testnet_runtime_v1",
    )
    favorable, adverse = _excursions_for_trade_state(trade_state)
    return TradeRecord.from_closed_position(
        trade_id=f"rt-{trade_state.symbol.lower()}-{trade_state.opened_at_ms}-{closed_at_ms}",
        opened_at=_dt_ms(trade_state.opened_at_ms),
        closed_at=_dt_ms(closed_at_ms),
        position=position,
        policy=policy,
        exit_reason=exit_reason,
        exit_contract_price_avg=exit_contract_price,
        exit_mark_price=exit_mark_price,
        max_favorable_excursion_usdt=favorable,
        max_adverse_excursion_usdt=adverse,
        signal_reason_codes=trade_state.signal_reason_codes,
        ai_snapshot=trade_state.ai_snapshot,
        atr_trail_history=trade_state.atr_trail_history,
        notes=notes,
    )


def build_trade_review(*, trade_record: TradeRecord, closed_at_ms: int) -> TradeReview:
    pnl_after_fees = (
        trade_record.realized_pnl_after_fees_usdt
        if trade_record.realized_pnl_after_fees_usdt is not None
        else trade_record.realized_pnl_usdt
    )
    risk_unit = max(trade_record.hard_stop_trigger_loss_usdt, 1e-9)
    mfe = trade_record.max_favorable_excursion_usdt
    mae = trade_record.max_adverse_excursion_usdt
    outcome = "WIN" if pnl_after_fees > 0 else "LOSS" if pnl_after_fees < 0 else "FLAT"

    if trade_record.exit_reason == "SYSTEM_FAILSAFE_EXIT":
        primary_cause = "operational_safety_exit"
        market_pattern = "runtime_protection"
        explanation = (
            "The runtime flattened the position because exchange state, protection orders, "
            "or market data heartbeats were no longer trustworthy. This exit is operational, "
            "not a pure strategy signal."
        )
        action_items = [
            "Review exchange orders, local incidents, and websocket health before resuming entries.",
            "Treat the loss as infrastructure risk first, strategy risk second.",
        ]
    elif trade_record.exit_reason == "TIME_STOP":
        primary_cause = "no_follow_through"
        market_pattern = "stalled_setup"
        explanation = (
            "The setup never delivered enough continuation within the allowed holding window. "
            "From a chart perspective this is usually drift, range behavior, or weak impulse quality."
        )
        action_items = [
            "Demand stronger higher-timeframe alignment before taking the same setup again.",
            "Reduce participation when the first few bars fail to extend the move.",
        ]
    elif trade_record.exit_reason == "FIXED_TAKE_PROFIT":
        primary_cause = "target_harvest"
        market_pattern = "planned_extension_complete"
        explanation = (
            "Price reached the predefined take-profit objective before the holding window expired. "
            "This indicates the setup delivered enough directional follow-through to capture the planned move."
        )
        action_items = [
            "Keep comparing the fixed target distance against the average excursion of future winning trades.",
            "If later data shows frequent overshoot after target hit, reconsider a runner component instead of full exit.",
        ]
    elif trade_record.exit_reason == "EARLY_FAIL_EXIT":
        primary_cause = "fast_failure"
        market_pattern = "weak_initial_impulse"
        explanation = (
            "The trade failed the early follow-through check and was closed before the full holding window. "
            "This usually means the entry did not attract enough immediate directional participation."
        )
        action_items = [
            "Review the first two to three candles after entry and demand stronger immediate expansion.",
            "Avoid repeating this setup when funding or crowding is already stretched against the trade.",
        ]
    elif trade_record.exit_reason == "ATR_TRAIL_EXIT" and pnl_after_fees >= 0:
        primary_cause = "trend_exhaustion"
        market_pattern = "pullback_after_expansion"
        explanation = (
            "Price did move in favor first, then mean-reverted enough to trip the ATR trail. "
            "This is typical when a trend leg exhausts or compresses after expansion."
        )
        action_items = [
            "Keep the trailing logic; this is normal trend-harvesting behavior.",
            "If too much profit was given back, review whether the entry came late into an extended move.",
        ]
    elif trade_record.exit_reason == "BREAK_EVEN_STOP_EXIT":
        primary_cause = "protected_retrace"
        market_pattern = "initial_extension_then_reversion"
        explanation = (
            "The trade moved far enough to arm the break-even stop, then reverted back through the protected level. "
            "This usually means the setup had initial impulse but failed to develop into a full continuation."
        )
        action_items = [
            "Check whether the move regularly stalls near the same extension distance before choosing larger targets.",
            "Use this pattern to decide whether scratch exits or partial profits are more suitable than full runners.",
        ]
    elif trade_record.exit_reason == "PARTIAL_TAKE_PROFIT":
        primary_cause = "scaled_profit_capture"
        market_pattern = "initial_extension_harvested"
        explanation = (
            "Part of the position was realized at the preset target while the remainder stayed open. "
            "Treat this record as a scaling event rather than a full thesis completion."
        )
        action_items = [
            "Review the paired follow-up exit to judge whether the runner added value after the partial take profit.",
            "Do not evaluate this event in isolation; use the full position sequence when comparing exit policies.",
        ]
    elif trade_record.exit_reason == "ATR_TRAIL_EXIT":
        primary_cause = "volatility_whipsaw"
        market_pattern = "reversal_after_entry"
        explanation = (
            "The trade failed to build enough favorable excursion before volatility reversed into the ATR trail. "
            "That usually means thin continuation or a choppy breakout attempt."
        )
        action_items = [
            "Be stricter on breakout quality when ATR is already expanded.",
            "Favor entries closer to structure instead of chasing late candles.",
        ]
    elif trade_record.exit_reason == "HARD_STOP_MARK_PRICE" and mfe <= 0.0:
        primary_cause = "immediate_rejection"
        market_pattern = "no_follow_through_after_entry"
        explanation = (
            "Price moved directly against the entry and never produced meaningful favorable excursion "
            "before the hard stop. This is a classic immediate rejection or mistimed entry."
        )
        action_items = [
            "Tighten setup selection and avoid entries without immediate continuation.",
            "Check whether the entry was taken into nearby swing resistance or support.",
        ]
    elif trade_record.exit_reason == "HARD_STOP_MARK_PRICE":
        primary_cause = "failed_breakout"
        market_pattern = "initial_follow_through_then_reversal"
        explanation = (
            "The trade showed some favorable movement, then reversed through the hard stop. "
            "That pattern is consistent with breakout failure, liquidity sweep, or a late entry into extension."
        )
        action_items = [
            "Review whether the entry was too extended from VWAP or EMA structure.",
            "Favor earlier entries inside the impulse rather than after the move is already stretched.",
        ]
    else:
        primary_cause = "generic_exit"
        market_pattern = "mixed_conditions"
        explanation = (
            "The position was closed by a generic exit path. Review the trade context together with "
            "excursion metrics and incident logs before reusing the setup."
        )
        action_items = [
            "Check entry timing, signal context, and runtime incidents together.",
        ]

    ai_snapshot = trade_record.ai_snapshot or {}
    if ai_snapshot.get("entry_action") == "reduce_size":
        action_items.append(
            "The AI gate already flagged reduced size; keep classifying this setup family as lower quality."
        )
    elif ai_snapshot.get("entry_action") == "veto":
        action_items.append(
            "The AI gate was already cautious on this pattern; compare future similar trades against the veto reasons."
        )

    rule_change_candidates = _rule_change_candidates_for_review(
        outcome=outcome,
        primary_cause=primary_cause,
        market_pattern=market_pattern,
        trade_record=trade_record,
        pnl_after_fees=pnl_after_fees,
        risk_unit=risk_unit,
        mfe=mfe,
        mae=mae,
    )
    handling_decision = _loss_handling_decision(
        outcome=outcome,
        primary_cause=primary_cause,
        market_pattern=market_pattern,
        rule_change_candidates=rule_change_candidates,
    )

    return TradeReview(
        trade_id=trade_record.trade_id,
        symbol=trade_record.symbol,
        review_version="trade_review_v2",
        closed_at_ms=closed_at_ms,
        outcome=outcome,
        primary_cause=primary_cause,
        market_pattern=market_pattern,
        explanation=explanation,
        action_items=action_items,
        rule_change_candidates=rule_change_candidates,
        handling_decision=handling_decision,
        evidence={
            "exit_reason": trade_record.exit_reason,
            "realized_pnl_after_fees_usdt": pnl_after_fees,
            "risk_multiple": pnl_after_fees / risk_unit,
            "max_favorable_excursion_r": mfe / risk_unit,
            "max_adverse_excursion_r": mae / risk_unit,
            "max_favorable_excursion_usdt": mfe,
            "max_adverse_excursion_usdt": mae,
            "atr_trail_steps": len(trade_record.atr_trail_history),
            "signal_reason_codes": list(trade_record.signal_reason_codes),
            "ai_snapshot": trade_record.ai_snapshot,
        },
    )


def _loss_handling_decision(
    *,
    outcome: str,
    primary_cause: str,
    market_pattern: str,
    rule_change_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    if outcome != "LOSS":
        return {
            "action": "record_only",
            "pause_new_entries": False,
            "auto_apply_rule_changes": False,
            "reason": "no loss handling needed for non-losing trade",
        }
    if primary_cause == "operational_safety_exit":
        return {
            "action": "pause_for_manual_review",
            "pause_new_entries": True,
            "auto_apply_rule_changes": False,
            "reason": "operational exits must be inspected before entries resume",
        }
    return {
        "action": "cooldown_and_monitor_pattern",
        "pause_new_entries": False,
        "auto_apply_rule_changes": False,
        "reason": (
            "single losing trade should use the configured cooldown; repeated matching "
            "loss causes trigger repeated_loss_pattern_review"
        ),
        "market_pattern": market_pattern,
        "candidate_count": len(rule_change_candidates),
    }


def _rule_change_candidates_for_review(
    *,
    outcome: str,
    primary_cause: str,
    market_pattern: str,
    trade_record: TradeRecord,
    pnl_after_fees: float,
    risk_unit: float,
    mfe: float,
    mae: float,
) -> list[dict[str, Any]]:
    if outcome != "LOSS":
        return []

    base = {
        "auto_apply": False,
        "validation_rule": (
            "Do not apply from one trade. Recheck after a repeated pattern across at least "
            "10 similar losses or the next forward-review sample."
        ),
        "evidence": {
            "exit_reason": trade_record.exit_reason,
            "risk_multiple": pnl_after_fees / risk_unit,
            "max_favorable_excursion_r": mfe / risk_unit,
            "max_adverse_excursion_r": mae / risk_unit,
            "signal_reason_codes": list(trade_record.signal_reason_codes),
            "market_pattern": market_pattern,
        },
    }

    candidates: list[dict[str, Any]] = []
    if primary_cause == "no_follow_through":
        candidates.append(
            {
                **base,
                "scope": "entry_filter",
                "proposal": "Require stronger continuation confirmation before re-entering this setup family.",
                "rationale": (
                    "The position used time without reaching the planned objective, which points to weak impulse "
                    "quality rather than stop placement alone."
                ),
            }
        )
    elif primary_cause == "fast_failure":
        candidates.append(
            {
                **base,
                "scope": "early_failure_filter",
                "proposal": "Tighten the first-bars follow-through requirement or reduce size for the same setup family.",
                "rationale": (
                    "The trade failed quickly after entry, so the useful rule change is to reject weak immediate "
                    "participation before full risk is exposed."
                ),
            }
        )
    elif primary_cause in {"immediate_rejection", "failed_breakout"}:
        candidates.append(
            {
                **base,
                "scope": "entry_timing",
                "proposal": "Add a late-entry guard around VWAP/EMA extension or require a cleaner pullback before entry.",
                "rationale": (
                    "The loss pattern is consistent with entering after the move was already stretched or rejected."
                ),
            }
        )
    elif primary_cause == "volatility_whipsaw":
        candidates.append(
            {
                **base,
                "scope": "volatility_filter",
                "proposal": "Reject entries when ATR expansion is high without matching volume participation.",
                "rationale": (
                    "The setup reversed into the trail before enough favorable excursion developed, which suggests "
                    "choppy volatility rather than clean trend continuation."
                ),
            }
        )
    elif primary_cause == "operational_safety_exit":
        candidates.append(
            {
                **base,
                "scope": "operations",
                "proposal": "Do not tune strategy rules from this trade; inspect runtime health and exchange state first.",
                "rationale": "This exit is infrastructure-driven, so changing entry logic would hide the real problem.",
            }
        )
    return candidates


def _excursions_for_trade_state(trade_state: ManagedTradeState) -> tuple[float, float]:
    entry = trade_state.entry_contract_price_avg
    qty = trade_state.quantity
    if trade_state.side == "LONG":
        favorable = max(0.0, (trade_state.highest_high - entry) * qty)
        adverse = max(0.0, (entry - trade_state.lowest_low) * qty)
    else:
        favorable = max(0.0, (entry - trade_state.lowest_low) * qty)
        adverse = max(0.0, (trade_state.highest_high - entry) * qty)
    return favorable, adverse


def _trade_closed_at_ms(record: dict[str, Any]) -> int:
    closed_at_ms = record.get("closed_at_ms")
    if closed_at_ms is not None:
        return int(closed_at_ms)
    closed_at = datetime.fromisoformat(str(record["closed_at"]))
    return int(closed_at.timestamp() * 1000)


def _trade_realized_pnl(record: dict[str, Any]) -> float:
    value = record.get("realized_pnl_after_fees_usdt")
    if value in (None, ""):
        value = record.get("realized_pnl_usdt", 0.0)
    return float(value)


def _trade_realized_r(record: dict[str, Any]) -> float:
    risk_unit = float(record.get("hard_stop_trigger_loss_usdt") or 0.0)
    if risk_unit <= 0.0:
        return 0.0
    return _trade_realized_pnl(record) / risk_unit


def _consecutive_loss_count(records: list[dict[str, Any]]) -> int:
    count = 0
    for record in records:
        if _trade_realized_pnl(record) < 0.0:
            count += 1
            continue
        break
    return count


def _max_consecutive_loss_count(records: list[dict[str, Any]]) -> int:
    streak = 0
    max_streak = 0
    for record in records:
        if _trade_realized_pnl(record) < 0.0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def _max_drawdown_r(records: list[dict[str, Any]]) -> float:
    cumulative_r = 0.0
    peak_r = 0.0
    max_drawdown = 0.0
    for record in records:
        cumulative_r += float(record["realized_r"])
        peak_r = max(peak_r, cumulative_r)
        max_drawdown = max(max_drawdown, peak_r - cumulative_r)
    return max_drawdown


def _worst_rolling_window_r(records: list[dict[str, Any]], *, window_ms: int) -> float:
    if not records:
        return 0.0
    worst = 0.0
    window_sum = 0.0
    start_index = 0
    for end_index, record in enumerate(records):
        window_sum += float(record["realized_r"])
        current_close_ms = int(record["closed_at_ms"])
        while (
            start_index <= end_index
            and current_close_ms - int(records[start_index]["closed_at_ms"]) > window_ms
        ):
            window_sum -= float(records[start_index]["realized_r"])
            start_index += 1
        worst = min(worst, window_sum)
    return worst


def _dt_ms(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
    existing = {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column in existing:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
