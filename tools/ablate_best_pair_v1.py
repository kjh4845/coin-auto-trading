from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pyarrow.parquet as pq

from ai_auto_trading.backtest.profile_runner import run_profiled_backtest
from ai_auto_trading.backtest.replay import (
    BacktestConfig,
    _compute_metrics,
    load_funding_rate_parquet,
    load_kline_parquet,
    resample_klines,
)
from ai_auto_trading.settings import load_settings
from ai_auto_trading.strategy.runtime_profiles import (
    RuntimeEntryProfile,
    build_runtime_entry_profiles,
    required_runtime_timeframes,
    runtime_stream_interval,
)

_DAY_MS = 86_400_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ablation search for the current best_pair_v1 runtime strategy."
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument(
        "--contract-parquet",
        default="data/raw/binance/contract_klines/btcusdt/1m_5y_v1.parquet",
    )
    parser.add_argument(
        "--mark-parquet",
        default="data/raw/binance/mark_price_klines/btcusdt/1m_5y_v1.parquet",
    )
    parser.add_argument(
        "--funding-parquet",
        default="data/raw/binance/funding_rate/btcusdt/5y_v1.parquet",
    )
    parser.add_argument(
        "--output",
        default="data/backtests/best_pair_v1_ablation_v1.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    symbol = args.symbol.upper()
    contract_path = Path(args.contract_parquet)
    mark_path = Path(args.mark_parquet)
    funding_path = Path(args.funding_parquet)
    output_path = Path(args.output)

    settings = replace(load_settings(), strategy_mode="best_pair_v1")
    base_profiles = build_runtime_entry_profiles(settings)

    contract_1m = load_kline_parquet(contract_path, symbol=symbol, interval="1m")
    mark_1m = load_kline_parquet(mark_path, symbol=symbol, interval="1m")
    funding_rows = load_funding_rate_parquet(funding_path, symbol=symbol)
    if not contract_1m or not mark_1m or not funding_rows:
        raise ValueError("missing contract, mark, or funding rows for ablation run")

    latest_close_time = _latest_close_time(contract_path)
    required_timeframes = required_runtime_timeframes(settings)
    contract_by_timeframe = {
        timeframe: resample_klines(
            contract_1m,
            target_interval=timeframe,
            source_interval="1m",
        )
        for timeframe in required_timeframes
    }

    base_config = BacktestConfig(
        symbol=symbol,
        execution_timeframe=runtime_stream_interval(settings),
        confirmation_timeframe=settings.confirmation_timeframe,
        macro_timeframe=settings.macro_timeframe,
        lower_mark_timeframe="1m",
        leverage_at_entry=float(settings.live_start_leverage),
        entry_notional_usdt=1000.0,
        fee_bps=4.0,
        slippage_bps=2.0,
        atr_trailing_multiplier=settings.atr_trailing_multiplier,
        atr_trail_activation_profit_r=settings.atr_trail_activation_profit_r,
        atr_trail_min_bars=settings.atr_trail_min_bars,
        max_holding_bars=settings.max_holding_bars,
        min_trades_for_decision=1,
        min_profit_factor=1.0,
        max_allowed_drawdown_usdt=200.0,
    )

    results: list[dict[str, Any]] = []
    for variant_name, variant_profiles in _variants(base_profiles):
        combined = run_profiled_backtest(
            profiles=variant_profiles,
            contract_candles_by_timeframe=contract_by_timeframe,
            lower_mark_price_candles=mark_1m,
            funding_rate_rows=funding_rows,
            base_config=base_config,
        )
        records = combined.combined_result.trade_records
        results.append(
            {
                "variant": variant_name,
                "profiles": {
                    profile.name: _profile_payload(profile)
                    for profile in variant_profiles
                },
                "metrics": {
                    "5y": _metrics_payload(combined.combined_result.metrics),
                    "3y": _metrics_payload(_window_metrics(records, latest_close_time - (365 * 3 * _DAY_MS))),
                    "365d": _metrics_payload(_window_metrics(records, latest_close_time - (365 * _DAY_MS))),
                    "90d": _metrics_payload(_window_metrics(records, latest_close_time - (90 * _DAY_MS))),
                },
                "combination": {
                    "accepted_by_profile": combined.accepted_by_profile,
                    "rejected_by_profile": combined.rejected_by_profile,
                    "exact_time_conflicts": combined.exact_time_conflicts,
                },
                "active_condition_count": sum(
                    _active_condition_count(profile) for profile in variant_profiles
                ),
            }
        )

    results.sort(
        key=lambda item: (
            _metric_value(item["metrics"]["5y"], "total_pnl_after_fees_usdt"),
            _metric_value(item["metrics"]["5y"], "profit_factor"),
            _metric_value(item["metrics"]["365d"], "total_pnl_after_fees_usdt"),
            _metric_value(item["metrics"]["365d"], "profit_factor"),
        ),
        reverse=True,
    )

    profitable = [
        row
        for row in results
        if row["metrics"]["5y"]["total_pnl_after_fees_usdt"] > 0
        and row["metrics"]["365d"]["total_pnl_after_fees_usdt"] > 0
    ]
    least_conditions_profitable = sorted(
        profitable,
        key=lambda item: (
            item["active_condition_count"],
            -_metric_value(item["metrics"]["5y"], "total_pnl_after_fees_usdt"),
            -_metric_value(item["metrics"]["365d"], "total_pnl_after_fees_usdt"),
        ),
    )

    payload = {
        "symbol": symbol,
        "base_variant": "base",
        "latest_close_time": latest_close_time,
        "best_by_5y_pnl": results[0] if results else None,
        "best_by_5y_pf": max(
            (row for row in results if row["metrics"]["5y"]["trades"] >= 20),
            key=lambda item: (
                _metric_value(item["metrics"]["5y"], "profit_factor"),
                _metric_value(item["metrics"]["5y"], "total_pnl_after_fees_usdt"),
            ),
            default=None,
        ),
        "best_by_365d_pnl": max(
            results,
            key=lambda item: (
                _metric_value(item["metrics"]["365d"], "total_pnl_after_fees_usdt"),
                _metric_value(item["metrics"]["365d"], "profit_factor"),
            ),
            default=None,
        ),
        "least_conditions_profitable": least_conditions_profitable[:5],
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _latest_close_time(contract_path: Path) -> int:
    parquet = pq.ParquetFile(contract_path)
    table = parquet.read_row_group(parquet.num_row_groups - 1, columns=["close_time"])
    column = table.column("close_time")
    return int(column[-1].as_py())


def _variants(
    base_profiles: list[RuntimeEntryProfile],
) -> list[tuple[str, list[RuntimeEntryProfile]]]:
    by_name = {profile.name: profile for profile in base_profiles}
    variants: list[tuple[str, list[RuntimeEntryProfile]]] = [("base", base_profiles)]
    short_name = "short_best_5m"
    long_name = "long_best_30m"

    short_mutations: list[tuple[str, Callable[[RuntimeEntryProfile], RuntimeEntryProfile]]] = [
        ("short_no_vwap", lambda profile: _mutate_params(profile, require_vwap=False)),
        ("short_no_rsi", lambda profile: _mutate_params(profile, require_rsi=False)),
        ("short_no_price_reclaim", lambda profile: _mutate_params(profile, require_price_reclaim=False)),
        ("short_no_roc", lambda profile: _mutate_params(profile, require_roc=False)),
        ("short_no_micro", lambda profile: _mutate_params(profile, use_micro=False)),
        ("short_no_regime", lambda profile: _mutate_params(profile, use_regime=False)),
        ("short_no_volume", lambda profile: _mutate_params(profile, min_volume_ratio_20=0.0)),
        ("short_no_funding", lambda profile: _mutate_params(profile, min_short_funding_rate=None)),
        ("short_no_taker", lambda profile: _mutate_params(profile, max_short_taker_buy_ratio=None)),
    ]
    long_mutations: list[tuple[str, Callable[[RuntimeEntryProfile], RuntimeEntryProfile]]] = [
        ("long_no_vwap", lambda profile: _mutate_params(profile, require_vwap=False)),
        ("long_no_price_reclaim", lambda profile: _mutate_params(profile, require_price_reclaim=False)),
        ("long_no_micro", lambda profile: _mutate_params(profile, use_micro=False)),
        ("long_no_confirmation", lambda profile: _mutate_params(profile, use_confirmation=False)),
        ("long_no_macro", lambda profile: _mutate_params(profile, use_macro=False)),
    ]

    for name, mutator in short_mutations:
        variants.append(
            (
                name,
                [
                    mutator(by_name[short_name]),
                    by_name[long_name],
                ],
            )
        )
    for name, mutator in long_mutations:
        variants.append(
            (
                name,
                [
                    by_name[short_name],
                    mutator(by_name[long_name]),
                ],
            )
        )

    combo_mutations = [
        (
            "short_no_rsi__long_no_vwap",
            _mutate_params(by_name[short_name], require_rsi=False),
            _mutate_params(by_name[long_name], require_vwap=False),
        ),
        (
            "short_no_roc__long_no_vwap",
            _mutate_params(by_name[short_name], require_roc=False),
            _mutate_params(by_name[long_name], require_vwap=False),
        ),
        (
            "short_no_funding__long_no_vwap",
            _mutate_params(by_name[short_name], min_short_funding_rate=None),
            _mutate_params(by_name[long_name], require_vwap=False),
        ),
        (
            "short_no_rsi__long_no_confirmation",
            _mutate_params(by_name[short_name], require_rsi=False),
            _mutate_params(by_name[long_name], use_confirmation=False),
        ),
    ]
    for name, short_profile, long_profile in combo_mutations:
        variants.append((name, [short_profile, long_profile]))

    return variants


def _mutate_params(
    profile: RuntimeEntryProfile,
    **changes: Any,
) -> RuntimeEntryProfile:
    return replace(profile, params=replace(profile.params, **changes))


def _profile_payload(profile: RuntimeEntryProfile) -> dict[str, Any]:
    params = profile.params
    return {
        "priority": profile.priority,
        "execution_timeframe": params.execution_timeframe,
        "micro_timeframe": params.micro_timeframe if params.use_micro else None,
        "confirmation_timeframe": params.confirmation_timeframe if params.use_confirmation else None,
        "macro_timeframe": params.macro_timeframe if params.use_macro else None,
        "regime_timeframe": params.regime_timeframe if params.use_regime else None,
        "anchor_timeframe": params.anchor_timeframe if params.use_anchor else None,
        "active_conditions": _active_conditions(params),
        "active_condition_count": _active_condition_count(profile),
    }


def _active_conditions(params) -> list[str]:
    conditions: list[str] = []
    if params.use_confirmation:
        conditions.append("confirmation_trend")
    if params.use_macro:
        conditions.append("macro_trend")
    if params.use_regime and params.regime_timeframe:
        conditions.append("regime_trend")
    if params.use_anchor and params.anchor_timeframe:
        conditions.append("anchor_trend")
    if params.use_micro and params.micro_timeframe:
        conditions.append("micro_trend")
    if params.require_vwap:
        conditions.append("vwap_pullback")
    if params.require_rsi:
        conditions.append("rsi_reclaim")
    if params.require_price_reclaim:
        conditions.append("price_reclaim")
    if params.require_roc:
        conditions.append("roc_direction")
    if params.min_volume_ratio_20 > 0:
        conditions.append("volume_participation")
    if params.max_long_funding_rate is not None or params.min_short_funding_rate is not None:
        conditions.append("funding_filter")
    if params.min_long_taker_buy_ratio is not None or params.max_short_taker_buy_ratio is not None:
        conditions.append("taker_flow")
    return conditions


def _active_condition_count(profile: RuntimeEntryProfile) -> int:
    return len(_active_conditions(profile.params))


def _window_metrics(records: list[Any], start_ms: int) -> dict[str, Any]:
    filtered = [
        record
        for record in records
        if _iso_to_ms(record.closed_at) >= start_ms
    ]
    return _compute_metrics(filtered)


def _metrics_payload(metrics: Any) -> dict[str, Any]:
    if is_dataclass(metrics):
        return asdict(metrics)
    return dict(metrics)


def _iso_to_ms(value: str) -> int:
    return int(datetime.fromisoformat(value).timestamp() * 1000)


def _metric_value(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    return 0.0 if value is None else float(value)


if __name__ == "__main__":
    raise SystemExit(main())
