from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from ai_auto_trading.backtest.replay import (
    BacktestConfig,
    load_funding_rate_parquet,
    load_kline_parquet,
    resample_klines,
    run_hybrid_backtest,
)
from ai_auto_trading.models import TradeRecord
from ai_auto_trading.strategy.rule_based import RuleStrategyParameters

_DAY_MS = 86_400_000


@dataclass(frozen=True)
class ProfileDefinition:
    name: str
    priority: int
    execution_timeframe: str
    micro_timeframe: str | None
    confirmation_timeframe: str | None
    macro_timeframe: str | None
    regime_timeframe: str | None
    anchor_timeframe: str | None
    params: RuleStrategyParameters


def main() -> int:
    symbol = "BTCUSDT"
    contract_path = Path("data/raw/binance/contract_klines/btcusdt/1m_5y_v1.parquet")
    mark_path = Path("data/raw/binance/mark_price_klines/btcusdt/1m_5y_v1.parquet")
    funding_path = Path("data/raw/binance/funding_rate/btcusdt/5y_v1.parquet")

    contract_1m = load_kline_parquet(contract_path, symbol=symbol, interval="1m")
    mark_1m = load_kline_parquet(mark_path, symbol=symbol, interval="1m")
    funding_rows = load_funding_rate_parquet(funding_path, symbol=symbol)
    latest_close = max(int(row["close_time"]) for row in contract_1m)

    required_timeframes = sorted(
        {
            timeframe
            for profile in _profiles()
            for timeframe in profile.params.required_timeframes()
        },
        key=_timeframe_sort_key,
    )
    contract_by_timeframe = {
        timeframe: resample_klines(contract_1m, target_interval=timeframe, source_interval="1m")
        for timeframe in required_timeframes
    }

    windows = {
        "30d": _slice_window(
            contract_by_timeframe,
            mark_1m,
            funding_rows,
            latest_close - (30 * _DAY_MS),
        ),
        "365d": _slice_window(
            contract_by_timeframe,
            mark_1m,
            funding_rows,
            latest_close - (365 * _DAY_MS),
        ),
        "3y": _slice_window(
            contract_by_timeframe,
            mark_1m,
            funding_rows,
            latest_close - (365 * 3 * _DAY_MS),
        ),
        "5y": {
            "contract": contract_by_timeframe,
            "mark": mark_1m,
            "funding": funding_rows,
        },
    }

    payload: dict[str, Any] = {"profiles": [profile.name for profile in _profiles()]}
    for label, window in windows.items():
        profile_results = {
            profile.name: _run_profile(profile=profile, window=window)
            for profile in _profiles()
        }
        combined = _combine_profile_results(profile_results)
        payload[label] = {
            "standalone": {
                name: {"metrics": result["metrics"], "trade_count": len(result["trades"])}
                for name, result in profile_results.items()
            },
            "combined": combined,
        }

    output_path = Path("data/backtests/bidirectional_combined_v1.json")
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


def _profiles() -> list[ProfileDefinition]:
    return [
        ProfileDefinition(
            name="short_best_5m",
            priority=2,
            execution_timeframe="5m",
            micro_timeframe="15m",
            confirmation_timeframe="1h",
            macro_timeframe="4h",
            regime_timeframe="1d",
            anchor_timeframe=None,
            params=RuleStrategyParameters(
                execution_timeframe="5m",
                micro_timeframe="15m",
                confirmation_timeframe="1h",
                macro_timeframe="4h",
                regime_timeframe="1d",
                anchor_timeframe=None,
                use_confirmation=True,
                use_macro=True,
                use_regime=True,
                use_anchor=False,
                use_micro=True,
                require_vwap=True,
                require_rsi=True,
                require_price_reclaim=True,
                require_roc=True,
                allow_long_entries=False,
                allow_short_entries=True,
                min_higher_tf_ema_spread_pct=0.0005,
                min_volume_ratio_20=1.0,
                min_micro_long_ema_spread_pct=0.0,
                min_micro_short_ema_spread_pct=0.0011,
                max_long_funding_rate=None,
                min_short_funding_rate=0.00005,
                min_long_taker_buy_ratio=None,
                max_short_taker_buy_ratio=0.45,
            ),
        ),
        ProfileDefinition(
            name="long_best_30m",
            priority=1,
            execution_timeframe="30m",
            micro_timeframe="1h",
            confirmation_timeframe="4h",
            macro_timeframe="1d",
            regime_timeframe=None,
            anchor_timeframe=None,
            params=RuleStrategyParameters(
                execution_timeframe="30m",
                micro_timeframe="1h",
                confirmation_timeframe="4h",
                macro_timeframe="1d",
                regime_timeframe=None,
                anchor_timeframe=None,
                use_confirmation=True,
                use_macro=True,
                use_regime=False,
                use_anchor=False,
                use_micro=True,
                require_vwap=True,
                require_rsi=False,
                require_price_reclaim=True,
                require_roc=False,
                allow_long_entries=True,
                allow_short_entries=False,
                min_higher_tf_ema_spread_pct=0.0,
                min_volume_ratio_20=0.0,
                min_micro_long_ema_spread_pct=0.0005,
                min_micro_short_ema_spread_pct=0.0,
                max_long_funding_rate=None,
                min_short_funding_rate=None,
                min_long_taker_buy_ratio=None,
                max_short_taker_buy_ratio=None,
            ),
        ),
    ]


def _run_profile(*, profile: ProfileDefinition, window: dict[str, Any]) -> dict[str, Any]:
    config = BacktestConfig(
        symbol="BTCUSDT",
        execution_timeframe=profile.execution_timeframe,
        confirmation_timeframe=profile.confirmation_timeframe or profile.execution_timeframe,
        macro_timeframe=profile.macro_timeframe
        or (profile.confirmation_timeframe or profile.execution_timeframe),
        lower_mark_timeframe="1m",
        leverage_at_entry=5.0,
        entry_notional_usdt=1000.0,
        fee_bps=4.0,
        slippage_bps=2.0,
        atr_trailing_multiplier=2.5,
        atr_trail_activation_profit_r=0.5,
        atr_trail_min_bars=2,
        max_holding_bars=8,
        min_trades_for_decision=1,
        min_profit_factor=1.0,
        max_allowed_drawdown_usdt=200.0,
    )
    result = run_hybrid_backtest(
        contract_candles_by_timeframe=window["contract"],
        lower_mark_price_candles=window["mark"],
        funding_rate_rows=window["funding"],
        config=config,
        strategy_params=profile.params,
    )
    trades = []
    for record in result.trade_records:
        trades.append(
            {
                "profile": profile.name,
                "priority": profile.priority,
                "trade_id": record.trade_id,
                "opened_at_ms": _dt_to_ms(record.opened_at),
                "closed_at_ms": _dt_to_ms(record.closed_at),
                "side": record.side,
                "record": record,
            }
        )
    return {"metrics": _trade_metrics(result.trade_records), "trades": trades}


def _combine_profile_results(profile_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        trade
        for result in profile_results.values()
        for trade in result["trades"]
    ]
    candidates.sort(
        key=lambda item: (item["opened_at_ms"], -item["priority"], item["closed_at_ms"])
    )

    accepted: list[TradeRecord] = []
    rejected: list[dict[str, Any]] = []
    open_until_ms = -1
    current_trade_id: str | None = None
    current_profile: str | None = None
    accepted_counts: dict[str, int] = defaultdict(int)
    rejected_counts: dict[str, int] = defaultdict(int)
    exact_time_conflicts = 0

    for index, candidate in enumerate(candidates):
        record = candidate["record"]
        if candidate["opened_at_ms"] < open_until_ms:
            rejected.append(
                {
                    "trade_id": record.trade_id,
                    "profile": candidate["profile"],
                    "reason": "overlap_with_open_position",
                    "blocked_by_trade_id": current_trade_id,
                    "blocked_by_profile": current_profile,
                    "opened_at_ms": candidate["opened_at_ms"],
                }
            )
            rejected_counts[candidate["profile"]] += 1
            continue
        if (
            index + 1 < len(candidates)
            and candidates[index + 1]["opened_at_ms"] == candidate["opened_at_ms"]
        ):
            exact_time_conflicts += 1
        accepted.append(record)
        accepted_counts[candidate["profile"]] += 1
        open_until_ms = candidate["closed_at_ms"]
        current_trade_id = record.trade_id
        current_profile = candidate["profile"]

    return {
        "metrics": _trade_metrics(accepted),
        "accepted_trade_count": len(accepted),
        "rejected_trade_count": len(rejected),
        "accepted_by_profile": dict(accepted_counts),
        "rejected_by_profile": dict(rejected_counts),
        "exact_time_conflicts": exact_time_conflicts,
        "calendar": {
            "monthly": _calendar_breakdown(accepted, "%Y-%m"),
            "daily": _calendar_breakdown(accepted, "%Y-%m-%d"),
        },
    }


def _trade_metrics(records: list[TradeRecord]) -> dict[str, Any]:
    pnls = [float(record.realized_pnl_after_fees_usdt or 0.0) for record in records]
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    total_fees = sum(float(record.fees_usdt) for record in records)
    total_slippage = sum(float(record.slippage_usdt) for record in records)
    total_realized = sum(float(record.realized_pnl_usdt) for record in records)
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for pnl in pnls:
        cumulative += pnl
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    avg_win = mean(wins) if wins else 0.0
    avg_loss = mean(losses) if losses else 0.0
    reward_risk = (
        avg_win / abs(avg_loss) if wins and losses and avg_loss != 0 else None
    )
    return {
        "trades": len(records),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (len(wins) / len(records)) if records else 0.0,
        "total_realized_pnl_usdt": total_realized,
        "total_fees_usdt": total_fees,
        "total_slippage_usdt": total_slippage,
        "total_pnl_after_fees_usdt": sum(pnls),
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else 0.0,
        "max_drawdown_usdt": max_drawdown,
        "avg_win_after_fees_usdt": avg_win,
        "avg_loss_after_fees_usdt": avg_loss,
        "reward_risk_ratio": reward_risk,
        "expectancy_after_fees_usdt_per_trade": (mean(pnls) if pnls else 0.0),
    }


def _calendar_breakdown(records: list[TradeRecord], fmt: str) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"trades": 0, "wins": 0, "losses": 0, "pnl_after_fees_usdt": 0.0}
    )
    for record in records:
        closed_at = datetime.fromisoformat(record.closed_at)
        key = closed_at.strftime(fmt)
        pnl = float(record.realized_pnl_after_fees_usdt or 0.0)
        bucket = buckets[key]
        bucket["trades"] += 1
        bucket["pnl_after_fees_usdt"] += pnl
        if pnl >= 0:
            bucket["wins"] += 1
        else:
            bucket["losses"] += 1
    output = []
    for key in sorted(buckets):
        bucket = buckets[key]
        trades = bucket["trades"]
        output.append(
            {
                "bucket": key,
                "trades": trades,
                "wins": bucket["wins"],
                "losses": bucket["losses"],
                "win_rate": (bucket["wins"] / trades) if trades else 0.0,
                "pnl_after_fees_usdt": round(bucket["pnl_after_fees_usdt"], 6),
            }
        )
    return output


def _slice_window(
    contract_by_timeframe: dict[str, list[dict[str, Any]]],
    mark_rows: list[dict[str, Any]],
    funding_rows: list[dict[str, Any]],
    start_time: int,
) -> dict[str, Any]:
    return {
        "contract": {
            timeframe: [row for row in rows if int(row["open_time"]) >= start_time]
            for timeframe, rows in contract_by_timeframe.items()
        },
        "mark": [row for row in mark_rows if int(row["open_time"]) >= start_time],
        "funding": [row for row in funding_rows if int(row["funding_time"]) >= start_time],
    }


def _dt_to_ms(value: str) -> int:
    return int(datetime.fromisoformat(value).timestamp() * 1000)


def _timeframe_sort_key(timeframe: str) -> tuple[int, str]:
    unit = timeframe[-1]
    magnitude = int(timeframe[:-1])
    unit_order = {"m": 0, "h": 1, "d": 2}
    return (unit_order.get(unit, 99), magnitude, timeframe)


if __name__ == "__main__":
    raise SystemExit(main())
