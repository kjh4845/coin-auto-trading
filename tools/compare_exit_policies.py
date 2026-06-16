from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path

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
    build_runtime_entry_profiles,
    required_runtime_timeframes,
)


def _closed_at_ms(value: str) -> int:
    return int(datetime.fromisoformat(value).timestamp() * 1000)


def _window_payload(trade_records: list) -> dict[str, object]:
    metrics = _compute_metrics(trade_records)
    exit_counts = Counter(record.exit_reason for record in trade_records)
    exit_pnl = defaultdict(float)
    for record in trade_records:
        pnl = record.realized_pnl_after_fees_usdt
        if pnl is None:
            pnl = record.realized_pnl_usdt - record.fees_usdt
        exit_pnl[record.exit_reason] += pnl
    wins = [record.realized_pnl_after_fees_usdt for record in trade_records if (record.realized_pnl_after_fees_usdt or 0.0) > 0]
    losses = [record.realized_pnl_after_fees_usdt for record in trade_records if (record.realized_pnl_after_fees_usdt or 0.0) < 0]
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss_abs = abs(sum(losses) / len(losses)) if losses else 0.0
    reward_risk = (avg_win / avg_loss_abs) if avg_loss_abs else None
    return {
        "trades": metrics.trades,
        "wins": metrics.wins,
        "losses": metrics.losses,
        "win_rate_pct": round(metrics.win_rate * 100.0, 2),
        "profit_factor": round(metrics.profit_factor, 4),
        "pnl_after_fees_usdt": round(metrics.total_pnl_after_fees_usdt, 2),
        "max_drawdown_usdt": round(metrics.max_drawdown_usdt, 2),
        "avg_win_usdt": round(avg_win, 2),
        "avg_loss_usdt": round(-avg_loss_abs, 2),
        "reward_risk": round(reward_risk, 3) if reward_risk is not None else None,
        "exit_counts": dict(exit_counts),
        "exit_pnl_after_fees_usdt": {key: round(value, 2) for key, value in exit_pnl.items()},
    }


def main() -> int:
    settings = load_settings()
    profiles = build_runtime_entry_profiles(replace(settings, strategy_mode="best_pair_v1"))
    required_timeframes = required_runtime_timeframes(replace(settings, strategy_mode="best_pair_v1"))

    base = Path("data/raw/binance")
    contract_1m = load_kline_parquet(
        base / "contract_klines" / "btcusdt" / "1m_5y_v1.parquet",
        symbol="BTCUSDT",
        interval="1m",
    )
    mark_1m = load_kline_parquet(
        base / "mark_price_klines" / "btcusdt" / "1m_5y_v1.parquet",
        symbol="BTCUSDT",
        interval="1m",
    )
    funding_path = base / "funding_rate" / "btcusdt" / "5y_v1.parquet"
    funding_rows = (
        load_funding_rate_parquet(funding_path, symbol="BTCUSDT")
        if funding_path.exists()
        else []
    )
    contract_candles_by_timeframe = {
        timeframe: resample_klines(
            contract_1m,
            target_interval=timeframe,
            source_interval="1m",
        )
        for timeframe in required_timeframes
    }

    base_config = BacktestConfig(
        symbol="BTCUSDT",
        execution_timeframe="5m",
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
        max_allowed_drawdown_usdt=100.0,
    )

    last_close_time = max(int(row["close_time"]) for row in contract_candles_by_timeframe["5m"])
    cutoffs = {
        "5y_full": None,
        "3y_recent": last_close_time - (365 * 3 * 24 * 60 * 60 * 1000),
        "1y_recent": last_close_time - (365 * 24 * 60 * 60 * 1000),
    }
    variants = {
        "baseline_atr_trail": replace(base_config, exit_policy="atr_trail"),
        "A_time_stop_only": replace(base_config, exit_policy="time_stop_only"),
        "B_break_even_time_stop": replace(
            base_config,
            exit_policy="break_even_time_stop",
            break_even_activation_profit_r=0.5,
            break_even_min_bars=2,
        ),
        "C_fixed_tp_1r_time_stop": replace(
            base_config,
            exit_policy="fixed_tp_time_stop",
            fixed_take_profit_r=1.0,
        ),
        "D_partial_tp_50pct_1r_runner": replace(
            base_config,
            exit_policy="partial_tp_runner",
            partial_take_profit_r=1.0,
            partial_take_profit_fraction=0.5,
        ),
    }

    payload = {
        "strategy_mode": "best_pair_v1",
        "profiles": [
            {
                "name": profile.name,
                "priority": profile.priority,
                "execution_timeframe": profile.params.execution_timeframe,
            }
            for profile in profiles
        ],
        "base_config": {
            "entry_notional_usdt": base_config.entry_notional_usdt,
            "leverage_at_entry": base_config.leverage_at_entry,
            "fee_bps": base_config.fee_bps,
            "slippage_bps": base_config.slippage_bps,
            "atr_trailing_multiplier": base_config.atr_trailing_multiplier,
            "atr_trail_activation_profit_r": base_config.atr_trail_activation_profit_r,
            "atr_trail_min_bars": base_config.atr_trail_min_bars,
            "max_holding_bars": base_config.max_holding_bars,
        },
        "variants": {},
    }

    for name, config in variants.items():
        combined = run_profiled_backtest(
            profiles=profiles,
            contract_candles_by_timeframe=contract_candles_by_timeframe,
            lower_mark_price_candles=mark_1m,
            funding_rate_rows=funding_rows,
            base_config=config,
        )
        window_payloads: dict[str, object] = {}
        for window_name, cutoff in cutoffs.items():
            records = (
                combined.combined_result.trade_records
                if cutoff is None
                else [
                    record
                    for record in combined.combined_result.trade_records
                    if _closed_at_ms(record.closed_at) >= cutoff
                ]
            )
            window_payloads[window_name] = _window_payload(records)
        payload["variants"][name] = {
            "config": {
                "exit_policy": config.exit_policy,
                "break_even_activation_profit_r": config.break_even_activation_profit_r,
                "break_even_min_bars": config.break_even_min_bars,
                "fixed_take_profit_r": config.fixed_take_profit_r,
                "partial_take_profit_r": config.partial_take_profit_r,
                "partial_take_profit_fraction": config.partial_take_profit_fraction,
            },
            "profile_acceptance": combined.accepted_by_profile,
            "profile_rejections": combined.rejected_by_profile,
            "exact_time_conflicts": combined.exact_time_conflicts,
            "windows": window_payloads,
        }

    output_path = Path("data/backtests/best_pair_v1_exit_policy_compare_v1.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
