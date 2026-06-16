from __future__ import annotations

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


def _window_metrics(trade_records: list) -> dict[str, float | int]:
    metrics = _compute_metrics(trade_records)
    return {
        "trades": metrics.trades,
        "win_rate_pct": round(metrics.win_rate * 100.0, 2),
        "profit_factor": round(metrics.profit_factor, 4),
        "pnl_after_fees_usdt": round(metrics.total_pnl_after_fees_usdt, 2),
        "max_drawdown_usdt": round(metrics.max_drawdown_usdt, 2),
    }


def main() -> int:
    settings = load_settings()
    profiled_settings = replace(settings, strategy_mode="best_pair_v1")
    profiles = build_runtime_entry_profiles(profiled_settings)
    required_timeframes = required_runtime_timeframes(profiled_settings)

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
    funding_rows = load_funding_rate_parquet(
        base / "funding_rate" / "btcusdt" / "5y_v1.parquet",
        symbol="BTCUSDT",
    )
    contract_candles_by_timeframe = {
        timeframe: resample_klines(contract_1m, target_interval=timeframe, source_interval="1m")
        for timeframe in required_timeframes
    }
    last_close_time = max(int(row["close_time"]) for row in contract_candles_by_timeframe["5m"])
    cutoffs = {
        "5y_full": None,
        "3y_recent": last_close_time - (365 * 3 * 24 * 60 * 60 * 1000),
        "1y_recent": last_close_time - (365 * 24 * 60 * 60 * 1000),
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
        exit_policy="fixed_tp_time_stop",
        max_holding_bars=settings.max_holding_bars,
        min_trades_for_decision=1,
        min_profit_factor=1.0,
        max_allowed_drawdown_usdt=100.0,
    )

    rounds = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    payload = {
        "strategy_mode": "best_pair_v1",
        "exit_policy": "fixed_tp_time_stop",
        "rounds": {},
    }
    for value in rounds:
        config = replace(base_config, fixed_take_profit_r=value)
        combined = run_profiled_backtest(
            profiles=profiles,
            contract_candles_by_timeframe=contract_candles_by_timeframe,
            lower_mark_price_candles=mark_1m,
            funding_rate_rows=funding_rows,
            base_config=config,
        )
        windows = {}
        for name, cutoff in cutoffs.items():
            records = (
                combined.combined_result.trade_records
                if cutoff is None
                else [
                    record
                    for record in combined.combined_result.trade_records
                    if _closed_at_ms(record.closed_at) >= cutoff
                ]
            )
            windows[name] = _window_metrics(records)
        payload["rounds"][str(value)] = windows

    output_path = Path("data/backtests/fixed_tp_round_sweep_v1.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
