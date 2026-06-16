from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from ai_auto_trading.ai.gate import AIGateConfig, AITradeAssistant
from ai_auto_trading.backtest.replay import (
    BacktestConfig,
    BacktestResult,
    _build_decision_report,
    _compute_metrics,
    run_hybrid_backtest,
)
from ai_auto_trading.models import TradeRecord
from ai_auto_trading.strategy.runtime_profiles import RuntimeEntryProfile


@dataclass(frozen=True)
class ProfileBacktestRun:
    profile_name: str
    priority: int
    result: BacktestResult


@dataclass(frozen=True)
class CombinedProfileBacktestResult:
    combined_result: BacktestResult
    standalone_runs: list[ProfileBacktestRun]
    accepted_by_profile: dict[str, int]
    rejected_by_profile: dict[str, int]
    exact_time_conflicts: int


def run_profiled_backtest(
    *,
    profiles: list[RuntimeEntryProfile],
    contract_candles_by_timeframe: dict[str, list[dict[str, Any]]],
    lower_mark_price_candles: list[dict[str, Any]],
    funding_rate_rows: list[dict[str, Any]] | None,
    base_config: BacktestConfig,
    ai_trade_assistant: AITradeAssistant | None = None,
    ai_gate_config: AIGateConfig | None = None,
) -> CombinedProfileBacktestResult:
    standalone_runs: list[ProfileBacktestRun] = []
    for profile in sorted(profiles, key=lambda item: (-item.priority, item.name)):
        profile_config = replace(
            base_config,
            execution_timeframe=profile.params.execution_timeframe,
            confirmation_timeframe=profile.params.confirmation_timeframe
            or profile.params.execution_timeframe,
            macro_timeframe=profile.params.macro_timeframe
            or profile.params.confirmation_timeframe
            or profile.params.execution_timeframe,
        )
        result = run_hybrid_backtest(
            contract_candles_by_timeframe=contract_candles_by_timeframe,
            lower_mark_price_candles=lower_mark_price_candles,
            funding_rate_rows=funding_rate_rows,
            config=profile_config,
            strategy_params=profile.params,
            ai_trade_assistant=ai_trade_assistant,
            ai_gate_config=ai_gate_config,
        )
        tagged_records = [
            replace(
                record,
                signal_reason_codes=[f"profile:{profile.name}", *record.signal_reason_codes],
            )
            for record in result.trade_records
        ]
        standalone_runs.append(
            ProfileBacktestRun(
                profile_name=profile.name,
                priority=profile.priority,
                result=BacktestResult(
                    trade_records=tagged_records,
                    metrics=result.metrics,
                    decision_report=result.decision_report,
                ),
            )
        )

    combined_records, accepted_by_profile, rejected_by_profile, exact_time_conflicts = _combine_trade_records(
        standalone_runs
    )
    combined_metrics = _compute_metrics(combined_records)
    combined_result = BacktestResult(
        trade_records=combined_records,
        metrics=combined_metrics,
        decision_report=_build_decision_report(combined_metrics, base_config),
    )
    return CombinedProfileBacktestResult(
        combined_result=combined_result,
        standalone_runs=standalone_runs,
        accepted_by_profile=accepted_by_profile,
        rejected_by_profile=rejected_by_profile,
        exact_time_conflicts=exact_time_conflicts,
    )


def _combine_trade_records(
    runs: list[ProfileBacktestRun],
) -> tuple[list[TradeRecord], dict[str, int], dict[str, int], int]:
    candidates: list[tuple[int, int, int, str, list[TradeRecord]]] = []
    for run in runs:
        grouped_records: dict[str, list[TradeRecord]] = {}
        for record in sorted(
            run.result.trade_records,
            key=lambda item: (_iso_to_ms(item.opened_at), _iso_to_ms(item.closed_at), item.trade_id),
        ):
            grouped_records.setdefault(record.opened_at, []).append(record)
        for opened_at, records in grouped_records.items():
            candidates.append(
                (
                    _iso_to_ms(opened_at),
                    -run.priority,
                    max(_iso_to_ms(record.closed_at) for record in records),
                    run.profile_name,
                    records,
                )
            )
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

    accepted_records: list[TradeRecord] = []
    accepted_by_profile: dict[str, int] = {}
    rejected_by_profile: dict[str, int] = {}
    exact_time_conflicts = 0
    open_until_ms = -1

    for index, candidate in enumerate(candidates):
        opened_at_ms, _negative_priority, closed_at_ms, profile_name, records = candidate
        if opened_at_ms < open_until_ms:
            rejected_by_profile[profile_name] = rejected_by_profile.get(profile_name, 0) + 1
            continue
        if index + 1 < len(candidates) and candidates[index + 1][0] == opened_at_ms:
            exact_time_conflicts += 1
        accepted_records.extend(records)
        accepted_by_profile[profile_name] = accepted_by_profile.get(profile_name, 0) + 1
        open_until_ms = closed_at_ms
    return accepted_records, accepted_by_profile, rejected_by_profile, exact_time_conflicts


def _iso_to_ms(value: str) -> int:
    return int(datetime.fromisoformat(value).timestamp() * 1000)
