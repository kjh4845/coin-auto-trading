from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_auto_trading.backtest.replay import (
    BacktestConfig,
    load_funding_rate_parquet,
    load_kline_parquet,
    resample_klines,
    run_hybrid_backtest,
)
from ai_auto_trading.strategy.rule_based import RuleStrategyParameters

_DAY_MS = 86_400_000
_REQUIRED_TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]


@dataclass(frozen=True)
class ExecutionPreset:
    name: str
    execution_timeframe: str
    micro_timeframe: str | None
    confirmation_timeframe: str | None
    macro_timeframe: str | None
    regime_timeframe: str | None
    anchor_timeframe: str | None


@dataclass(frozen=True)
class EntryTemplate:
    name: str
    side_mode: str
    use_confirmation: bool
    use_macro: bool
    use_regime: bool
    use_anchor: bool
    use_micro: bool
    require_vwap: bool
    require_rsi: bool
    require_price_reclaim: bool
    require_roc: bool
    min_higher_tf_ema_spread_pct: float
    min_volume_ratio_20: float
    long_micro_spread_pct: float
    short_micro_spread_pct: float
    max_long_funding_rate: float | None
    min_short_funding_rate: float | None
    min_long_taker_buy_ratio: float | None
    max_short_taker_buy_ratio: float | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explore rule-strategy combinations for one symbol.")
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument(
        "--contract-parquet",
        default="data/raw/binance/contract_klines/ethusdt/1m_5y_v1.parquet",
    )
    parser.add_argument(
        "--mark-parquet",
        default="data/raw/binance/mark_price_klines/ethusdt/1m_5y_v1.parquet",
    )
    parser.add_argument(
        "--funding-parquet",
        default="data/raw/binance/funding_rate/ethusdt/5y_v1.parquet",
    )
    parser.add_argument("--output", default="data/backtests/ethusdt_combo_explorer_v1.json")
    parser.add_argument("--markdown-output", default="data/backtests/ethusdt_combo_explorer_v1.md")
    parser.add_argument("--top-entries", type=int, default=8)
    parser.add_argument("--top-exit-candidates", type=int, default=3)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    symbol = args.symbol.upper()
    started = time.perf_counter()

    contract_1m = load_kline_parquet(Path(args.contract_parquet), symbol=symbol, interval="1m")
    mark_1m = load_kline_parquet(Path(args.mark_parquet), symbol=symbol, interval="1m")
    funding_rows = load_funding_rate_parquet(Path(args.funding_parquet), symbol=symbol)
    if not contract_1m or not mark_1m or not funding_rows:
        raise ValueError("missing ETH contract, mark, or funding rows")

    latest_close = max(int(row["close_time"]) for row in contract_1m)
    contract_by_timeframe = {
        timeframe: resample_klines(contract_1m, target_interval=timeframe, source_interval="1m")
        for timeframe in _REQUIRED_TIMEFRAMES
    }
    full_window = {"contract": contract_by_timeframe, "mark": mark_1m, "funding": funding_rows}
    window_365 = _slice_window(full_window, latest_close - (365 * _DAY_MS))
    window_90 = _slice_window(full_window, latest_close - (90 * _DAY_MS))

    candidates = [
        (preset, template)
        for preset in _execution_presets()
        for template in _entry_templates()
    ]
    print(json.dumps({"event": "coarse_scan_start", "symbol": symbol, "candidates": len(candidates)}), flush=True)

    coarse_rows: list[dict[str, Any]] = []
    for index, (preset, template) in enumerate(candidates, start=1):
        params = _build_params(preset, template)
        config = _base_config(symbol=symbol, preset=preset)
        metrics_365 = _run_metrics(window_365, config=config, params=params)
        if metrics_365["trades"] > 0:
            coarse_rows.append(
                {
                    "preset": asdict(preset),
                    "template": asdict(template),
                    "params": asdict(params),
                    "365d": metrics_365,
                    "score_365d": round(_entry_score(metrics_365), 6),
                }
            )
        if index % 10 == 0 or index == len(candidates):
            print(
                json.dumps(
                    {
                        "event": "coarse_scan_progress",
                        "done": index,
                        "total": len(candidates),
                        "kept": len(coarse_rows),
                    }
                ),
                flush=True,
            )

    coarse_rows.sort(key=_coarse_sort_key, reverse=True)
    shortlisted = coarse_rows[: max(1, args.top_entries)]
    print(json.dumps({"event": "full_validation_start", "shortlisted": len(shortlisted)}), flush=True)

    validated: list[dict[str, Any]] = []
    for index, row in enumerate(shortlisted, start=1):
        preset = ExecutionPreset(**row["preset"])
        params = RuleStrategyParameters(**row["params"])
        config = _base_config(symbol=symbol, preset=preset)
        metrics_90 = _run_metrics(window_90, config=config, params=params)
        metrics_5y = _run_metrics(full_window, config=config, params=params)
        enriched = {
            **row,
            "90d": metrics_90,
            "5y": metrics_5y,
            "robust_score": round(_robust_score(metrics_5y, row["365d"], metrics_90), 6),
        }
        validated.append(enriched)
        print(
            json.dumps(
                {
                    "event": "full_validation_progress",
                    "done": index,
                    "total": len(shortlisted),
                    "preset": row["preset"]["name"],
                    "template": row["template"]["name"],
                    "trades_5y": metrics_5y["trades"],
                    "pf_5y": metrics_5y["profit_factor"],
                }
            ),
            flush=True,
        )

    validated.sort(key=_validated_sort_key, reverse=True)
    exit_candidates = validated[: max(1, args.top_exit_candidates)]
    exit_variants = _exit_variants()
    exit_results: list[dict[str, Any]] = []
    print(
        json.dumps(
            {
                "event": "exit_validation_start",
                "candidates": len(exit_candidates),
                "exit_variants": len(exit_variants),
            }
        ),
        flush=True,
    )

    total_exit_runs = len(exit_candidates) * len(exit_variants)
    exit_run_index = 0
    for candidate in exit_candidates:
        preset = ExecutionPreset(**candidate["preset"])
        params = RuleStrategyParameters(**candidate["params"])
        base_config = _base_config(symbol=symbol, preset=preset)
        for variant_name, changes in exit_variants:
            exit_run_index += 1
            config = replace(base_config, **changes)
            metrics_5y = _run_metrics(full_window, config=config, params=params)
            metrics_365 = _run_metrics(window_365, config=config, params=params)
            metrics_90 = _run_metrics(window_90, config=config, params=params)
            exit_results.append(
                {
                    "candidate_preset": candidate["preset"],
                    "candidate_template": candidate["template"],
                    "exit_variant": variant_name,
                    "exit_config": changes,
                    "5y": metrics_5y,
                    "365d": metrics_365,
                    "90d": metrics_90,
                    "robust_score": round(_robust_score(metrics_5y, metrics_365, metrics_90), 6),
                }
            )
            print(
                json.dumps(
                    {
                        "event": "exit_validation_progress",
                        "done": exit_run_index,
                        "total": total_exit_runs,
                        "variant": variant_name,
                        "trades_5y": metrics_5y["trades"],
                        "pf_5y": metrics_5y["profit_factor"],
                    }
                ),
                flush=True,
            )

    exit_results.sort(key=_validated_sort_key, reverse=True)
    output = {
        "symbol": symbol,
        "window": {
            "start_utc": _iso_from_ms(int(contract_1m[0]["open_time"])),
            "end_utc": _iso_from_ms(int(contract_1m[-1]["close_time"])),
        },
        "coarse_candidate_count": len(candidates),
        "coarse_with_trades": len(coarse_rows),
        "top_entries_requested": args.top_entries,
        "top_exit_candidates_requested": args.top_exit_candidates,
        "best_entry_default_exit": validated[0] if validated else None,
        "best_exit_adjusted": exit_results[0] if exit_results else None,
        "validated_top": validated,
        "exit_adjusted_top": exit_results[:20],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "notes": [
            "Coarse scan uses 365d with fixed_tp_time_stop at 1.0R.",
            "Validated entries are retested on 90d and 5y.",
            "Exit-adjusted candidates retest the strongest entries across multiple exit policies.",
        ],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_round_payload(output), indent=2, sort_keys=True), encoding="utf-8")
    markdown_path = Path(args.markdown_output)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_markdown_summary(_round_payload(output)), encoding="utf-8")
    print(json.dumps({"event": "done", "output": str(output_path), "markdown_output": str(markdown_path)}), flush=True)
    return 0


def _execution_presets() -> list[ExecutionPreset]:
    return [
        ExecutionPreset("fast_3m", "3m", "5m", "15m", "1h", "4h", "1d"),
        ExecutionPreset("balanced_5m", "5m", "15m", "1h", "4h", "1d", None),
        ExecutionPreset("medium_15m", "15m", "30m", "1h", "4h", "1d", None),
        ExecutionPreset("slow_30m", "30m", "1h", "4h", "1d", None, None),
    ]


def _entry_templates() -> list[EntryTemplate]:
    templates: list[EntryTemplate] = []
    styles = [
        {
            "name": "trend_price",
            "use_regime": False,
            "use_anchor": False,
            "require_vwap": False,
            "require_rsi": False,
            "require_roc": False,
            "min_higher_tf_ema_spread_pct": 0.0,
            "min_volume_ratio_20": 0.0,
            "use_flow": False,
        },
        {
            "name": "pullback",
            "use_regime": False,
            "use_anchor": False,
            "require_vwap": True,
            "require_rsi": False,
            "require_roc": False,
            "min_higher_tf_ema_spread_pct": 0.0,
            "min_volume_ratio_20": 0.0,
            "use_flow": False,
        },
        {
            "name": "momentum",
            "use_regime": False,
            "use_anchor": False,
            "require_vwap": True,
            "require_rsi": True,
            "require_roc": True,
            "min_higher_tf_ema_spread_pct": 0.0,
            "min_volume_ratio_20": 0.0,
            "use_flow": False,
        },
        {
            "name": "regime_flow",
            "use_regime": True,
            "use_anchor": False,
            "require_vwap": True,
            "require_rsi": False,
            "require_roc": False,
            "min_higher_tf_ema_spread_pct": 0.0005,
            "min_volume_ratio_20": 0.0,
            "use_flow": True,
        },
        {
            "name": "strict_volume_flow",
            "use_regime": True,
            "use_anchor": True,
            "require_vwap": True,
            "require_rsi": True,
            "require_roc": True,
            "min_higher_tf_ema_spread_pct": 0.001,
            "min_volume_ratio_20": 1.0,
            "use_flow": True,
        },
    ]
    for side_mode in ("long_only", "short_only", "both"):
        for style in styles:
            use_flow = bool(style["use_flow"])
            templates.append(
                EntryTemplate(
                    name=f"{side_mode}_{style['name']}",
                    side_mode=side_mode,
                    use_confirmation=True,
                    use_macro=True,
                    use_regime=bool(style["use_regime"]),
                    use_anchor=bool(style["use_anchor"]),
                    use_micro=True,
                    require_vwap=bool(style["require_vwap"]),
                    require_rsi=bool(style["require_rsi"]),
                    require_price_reclaim=True,
                    require_roc=bool(style["require_roc"]),
                    min_higher_tf_ema_spread_pct=float(style["min_higher_tf_ema_spread_pct"]),
                    min_volume_ratio_20=float(style["min_volume_ratio_20"]),
                    long_micro_spread_pct=0.0005 if side_mode != "short_only" else 0.0,
                    short_micro_spread_pct=0.0011 if side_mode != "long_only" else 0.0,
                    max_long_funding_rate=0.0 if use_flow and side_mode != "short_only" else None,
                    min_short_funding_rate=0.00005 if use_flow and side_mode != "long_only" else None,
                    min_long_taker_buy_ratio=0.55 if use_flow and side_mode != "short_only" else None,
                    max_short_taker_buy_ratio=0.45 if use_flow and side_mode != "long_only" else None,
                )
            )
    return templates


def _build_params(preset: ExecutionPreset, template: EntryTemplate) -> RuleStrategyParameters:
    return RuleStrategyParameters(
        execution_timeframe=preset.execution_timeframe,
        micro_timeframe=preset.micro_timeframe,
        confirmation_timeframe=preset.confirmation_timeframe or preset.execution_timeframe,
        macro_timeframe=preset.macro_timeframe or (preset.confirmation_timeframe or preset.execution_timeframe),
        regime_timeframe=preset.regime_timeframe,
        anchor_timeframe=preset.anchor_timeframe,
        use_confirmation=template.use_confirmation and preset.confirmation_timeframe is not None,
        use_macro=template.use_macro and preset.macro_timeframe is not None,
        use_regime=template.use_regime and preset.regime_timeframe is not None,
        use_anchor=template.use_anchor and preset.anchor_timeframe is not None,
        use_micro=template.use_micro and preset.micro_timeframe is not None,
        require_vwap=template.require_vwap,
        require_rsi=template.require_rsi,
        require_price_reclaim=template.require_price_reclaim,
        require_roc=template.require_roc,
        min_higher_tf_ema_spread_pct=template.min_higher_tf_ema_spread_pct,
        min_volume_ratio_20=template.min_volume_ratio_20,
        min_micro_long_ema_spread_pct=template.long_micro_spread_pct,
        min_micro_short_ema_spread_pct=template.short_micro_spread_pct,
        max_long_funding_rate=template.max_long_funding_rate,
        min_short_funding_rate=template.min_short_funding_rate,
        min_long_taker_buy_ratio=template.min_long_taker_buy_ratio,
        max_short_taker_buy_ratio=template.max_short_taker_buy_ratio,
        allow_long_entries=template.side_mode != "short_only",
        allow_short_entries=template.side_mode != "long_only",
    )


def _base_config(*, symbol: str, preset: ExecutionPreset) -> BacktestConfig:
    return BacktestConfig(
        symbol=symbol,
        execution_timeframe=preset.execution_timeframe,
        confirmation_timeframe=preset.confirmation_timeframe or preset.execution_timeframe,
        macro_timeframe=preset.macro_timeframe or (preset.confirmation_timeframe or preset.execution_timeframe),
        lower_mark_timeframe="1m",
        leverage_at_entry=2.0,
        entry_notional_usdt=1000.0,
        fee_bps=4.0,
        slippage_bps=2.0,
        atr_trailing_multiplier=2.5,
        atr_trail_activation_profit_r=0.5,
        atr_trail_min_bars=2,
        exit_policy="fixed_tp_time_stop",
        fixed_take_profit_r=1.0,
        max_holding_bars=8,
        min_trades_for_decision=1,
        min_profit_factor=1.0,
        max_allowed_drawdown_usdt=200.0,
    )


def _exit_variants() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("fixed_tp_0_75r", {"exit_policy": "fixed_tp_time_stop", "fixed_take_profit_r": 0.75}),
        ("fixed_tp_1_0r", {"exit_policy": "fixed_tp_time_stop", "fixed_take_profit_r": 1.0}),
        ("fixed_tp_1_25r", {"exit_policy": "fixed_tp_time_stop", "fixed_take_profit_r": 1.25}),
        ("fixed_tp_1_5r", {"exit_policy": "fixed_tp_time_stop", "fixed_take_profit_r": 1.5}),
        ("break_even_time_stop", {"exit_policy": "break_even_time_stop", "break_even_activation_profit_r": 0.5}),
        ("time_stop_only", {"exit_policy": "time_stop_only"}),
        ("atr_trail", {"exit_policy": "atr_trail"}),
    ]


def _slice_window(window: dict[str, Any], start_time: int) -> dict[str, Any]:
    return {
        "contract": {
            timeframe: [row for row in rows if int(row["open_time"]) >= start_time]
            for timeframe, rows in window["contract"].items()
        },
        "mark": [row for row in window["mark"] if int(row["open_time"]) >= start_time],
        "funding": [row for row in window["funding"] if int(row["funding_time"]) >= start_time],
    }


def _iso_from_ms(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()


def _run_metrics(
    window: dict[str, Any],
    *,
    config: BacktestConfig,
    params: RuleStrategyParameters,
) -> dict[str, Any]:
    result = run_hybrid_backtest(
        contract_candles_by_timeframe=window["contract"],
        lower_mark_price_candles=window["mark"],
        funding_rate_rows=window["funding"],
        config=config,
        strategy_params=params,
    )
    metrics = result.metrics.to_dict()
    pnls = [
        float(
            record.realized_pnl_after_fees_usdt
            if record.realized_pnl_after_fees_usdt is not None
            else record.realized_pnl_usdt - record.fees_usdt
        )
        for record in result.trade_records
    ]
    wins = [value for value in pnls if value > 0.0]
    losses = [value for value in pnls if value <= 0.0]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
    metrics["avg_win_usdt"] = avg_win
    metrics["avg_loss_usdt"] = avg_loss
    metrics["avg_reward_risk"] = (avg_win / avg_loss) if avg_loss else 0.0
    return metrics


def _entry_score(metrics: dict[str, Any]) -> float:
    if metrics["trades"] < 5:
        return -10_000.0 + metrics["trades"]
    score = metrics["total_pnl_after_fees_usdt"]
    score += metrics["profit_factor"] * 60.0
    score += metrics["win_rate"] * 80.0
    score += min(metrics["trades"], 80) * 0.4
    score -= max(metrics["max_drawdown_usdt"] - 80.0, 0.0) * 0.5
    return score


def _robust_score(metrics_5y: dict[str, Any], metrics_365: dict[str, Any], metrics_90: dict[str, Any]) -> float:
    score = metrics_5y["total_pnl_after_fees_usdt"]
    score += metrics_5y["profit_factor"] * 80.0
    score += metrics_5y["avg_reward_risk"] * 40.0
    score += metrics_5y["win_rate"] * 80.0
    score += min(metrics_5y["trades"], 160) * 0.25
    score += max(metrics_365["total_pnl_after_fees_usdt"], -100.0) * 0.7
    score += max(metrics_90["total_pnl_after_fees_usdt"], -50.0) * 0.5
    score -= max(metrics_5y["max_drawdown_usdt"] - 120.0, 0.0) * 0.35
    if metrics_5y["trades"] < 20:
        score -= 500.0
    if metrics_5y["profit_factor"] < 1.0:
        score -= 500.0
    if metrics_365["total_pnl_after_fees_usdt"] < 0:
        score -= 120.0
    return score


def _coarse_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = row["365d"]
    return (
        row["score_365d"],
        metrics["profit_factor"],
        metrics["total_pnl_after_fees_usdt"],
        metrics["win_rate"],
    )


def _validated_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    metrics = row["5y"]
    recent = row["365d"]
    return (
        row["robust_score"],
        metrics["total_pnl_after_fees_usdt"],
        metrics["profit_factor"],
        recent["total_pnl_after_fees_usdt"],
        metrics["win_rate"],
    )


def _round_payload(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _round_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_payload(item) for item in value]
    return value


def _markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['symbol']} Combination Explorer V1",
        "",
        f"Window: `{payload['window']['start_utc']}` to `{payload['window']['end_utc']}`",
        "",
        f"Coarse candidates: `{payload['coarse_candidate_count']}`",
        "",
        "## Best Entry Candidates",
        "",
        "| Rank | Preset | Template | 5Y Trades | 5Y Win | 5Y PF | 5Y R/R | 5Y PnL | 365D PnL | 90D PnL |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(payload.get("validated_top", [])[:10], start=1):
        lines.append(
            "| "
            f"{index} | {row['preset']['name']} | {row['template']['name']} | "
            f"{row['5y']['trades']} | {row['5y']['win_rate'] * 100:.2f}% | "
            f"{row['5y']['profit_factor']:.2f} | {row['5y']['avg_reward_risk']:.2f} | "
            f"{row['5y']['total_pnl_after_fees_usdt']:.2f} | "
            f"{row['365d']['total_pnl_after_fees_usdt']:.2f} | "
            f"{row['90d']['total_pnl_after_fees_usdt']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Best Exit-Adjusted Candidates",
            "",
            "| Rank | Preset | Template | Exit | 5Y Trades | 5Y Win | 5Y PF | 5Y R/R | 5Y PnL | 365D PnL | 90D PnL |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for index, row in enumerate(payload.get("exit_adjusted_top", [])[:10], start=1):
        lines.append(
            "| "
            f"{index} | {row['candidate_preset']['name']} | {row['candidate_template']['name']} | "
            f"{row['exit_variant']} | {row['5y']['trades']} | {row['5y']['win_rate'] * 100:.2f}% | "
            f"{row['5y']['profit_factor']:.2f} | {row['5y']['avg_reward_risk']:.2f} | "
            f"{row['5y']['total_pnl_after_fees_usdt']:.2f} | "
            f"{row['365d']['total_pnl_after_fees_usdt']:.2f} | "
            f"{row['90d']['total_pnl_after_fees_usdt']:.2f} |"
        )
    lines.extend(["", "Notes:", *[f"- {note}" for note in payload["notes"]]])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
