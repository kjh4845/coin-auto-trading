from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

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
class StrategyFamily:
    name: str
    use_confirmation: bool
    use_macro: bool
    use_regime: bool
    use_anchor: bool
    use_micro: bool
    require_vwap: bool
    require_rsi: bool
    require_price_reclaim: bool
    require_roc: bool
    use_volume: bool
    use_funding: bool
    use_taker: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search rule-based strategy combinations")
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
    parser.add_argument("--output", default="data/backtests/strategy_combo_search_v3.json")
    parser.add_argument("--shortlist", type=int, default=30)
    parser.add_argument("--top", type=int, default=20)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    symbol = args.symbol.upper()
    contract_path = Path(args.contract_parquet)
    mark_path = Path(args.mark_parquet)
    funding_path = Path(args.funding_parquet)
    output_path = Path(args.output)

    latest_close = _latest_close_time(contract_path)
    cutoff_365 = latest_close - (365 * _DAY_MS)
    cutoff_30 = latest_close - (30 * _DAY_MS)

    recent_window = _load_window(
        contract_path=contract_path,
        mark_path=mark_path,
        funding_path=funding_path,
        symbol=symbol,
        start_time=cutoff_365,
    )
    recent_30_window = _slice_window(recent_window, cutoff_30)
    config = BacktestConfig(
        symbol=symbol,
        execution_timeframe="3m",
        confirmation_timeframe="15m",
        macro_timeframe="1h",
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

    coarse_rows = []
    for family in _strategy_families():
        for params, side_mode in _family_parameter_grid(family):
            metrics_365 = _run_window(recent_window, config=config, params=params)
            if metrics_365["trades"] == 0:
                continue
            score_365 = _score_recent(metrics_365)
            coarse_rows.append(
                {
                    "family": family.name,
                    "side_mode": side_mode,
                    "params": _params_to_payload(params),
                    "365d": metrics_365,
                    "score_365d": round(score_365, 6),
                }
            )

    coarse_rows.sort(
        key=lambda row: (
            row["score_365d"],
            row["365d"]["profit_factor"],
            row["365d"]["total_pnl_after_fees_usdt"],
            row["365d"]["win_rate"],
        ),
        reverse=True,
    )
    shortlist = coarse_rows[: args.shortlist]

    long_window = _load_window(
        contract_path=contract_path,
        mark_path=mark_path,
        funding_path=funding_path,
        symbol=symbol,
        start_time=None,
    )
    validated = []
    for row in shortlist:
        params = _payload_to_params(row["params"])
        metrics_5y = _run_window(long_window, config=config, params=params)
        metrics_30 = _run_window(recent_30_window, config=config, params=params)
        final_score = _score_balanced(metrics_5y, row["365d"], metrics_30)
        validated.append(
            {
                **row,
                "5y": metrics_5y,
                "30d": metrics_30,
                "final_score": round(final_score, 6),
            }
        )

    validated.sort(
        key=lambda row: (
            row["final_score"],
            row["5y"]["total_pnl_after_fees_usdt"],
            row["365d"]["total_pnl_after_fees_usdt"],
            row["5y"]["win_rate"],
        ),
        reverse=True,
    )

    best_balanced = next(
        (
            row
            for row in validated
            if row["365d"]["trades"] >= 5
            and row["5y"]["trades"] >= 20
            and row["5y"]["profit_factor"] >= 1.0
        ),
        validated[0] if validated else None,
    )
    winrate_candidates = [row for row in validated if row["5y"]["trades"] >= 20]
    winrate_candidates.sort(
        key=lambda row: (
            row["5y"]["win_rate"],
            row["5y"]["profit_factor"],
            row["5y"]["total_pnl_after_fees_usdt"],
        ),
        reverse=True,
    )
    best_winrate = winrate_candidates[0] if winrate_candidates else None

    payload = {
        "symbol": symbol,
        "search_space_with_recent_trades": len(coarse_rows),
        "shortlist_count": len(shortlist),
        "families": [asdict(family) for family in _strategy_families()],
        "best_balanced": best_balanced,
        "best_winrate": best_winrate,
        "validated_top": validated[: args.top],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


def _latest_close_time(contract_path: Path) -> int:
    parquet = pq.ParquetFile(contract_path)
    table = parquet.read_row_group(parquet.num_row_groups - 1, columns=["close_time"])
    column = table.column("close_time")
    return int(column[-1].as_py())


def _load_window(
    *,
    contract_path: Path,
    mark_path: Path,
    funding_path: Path,
    symbol: str,
    start_time: int | None,
) -> dict[str, Any]:
    contract_1m = load_kline_parquet(
        contract_path,
        symbol=symbol,
        interval="1m",
        start_time=start_time,
    )
    mark_1m = load_kline_parquet(
        mark_path,
        symbol=symbol,
        interval="1m",
        start_time=start_time,
    )
    funding_rows = load_funding_rate_parquet(
        funding_path,
        symbol=symbol,
        start_time=start_time,
    )
    return {
        "contract": {
            timeframe: resample_klines(contract_1m, target_interval=timeframe, source_interval="1m")
            for timeframe in _REQUIRED_TIMEFRAMES
        },
        "mark": mark_1m,
        "funding": funding_rows,
    }


def _slice_window(window: dict[str, Any], start_time: int) -> dict[str, Any]:
    return {
        "contract": {
            timeframe: [row for row in rows if int(row["open_time"]) >= start_time]
            for timeframe, rows in window["contract"].items()
        },
        "mark": [row for row in window["mark"] if int(row["open_time"]) >= start_time],
        "funding": [
            row for row in window["funding"] if int(row["funding_time"]) >= start_time
        ],
    }


def _run_window(
    window: dict[str, Any], *, config: BacktestConfig, params: RuleStrategyParameters
) -> dict[str, Any]:
    result = run_hybrid_backtest(
        contract_candles_by_timeframe=window["contract"],
        lower_mark_price_candles=window["mark"],
        funding_rate_rows=window["funding"],
        config=config,
        strategy_params=params,
    )
    return result.metrics.to_dict()


def _strategy_families() -> list[StrategyFamily]:
    return [
        StrategyFamily(
            name="trend_price",
            use_confirmation=False,
            use_macro=True,
            use_regime=False,
            use_anchor=False,
            use_micro=True,
            require_vwap=False,
            require_rsi=False,
            require_price_reclaim=True,
            require_roc=False,
            use_volume=False,
            use_funding=False,
            use_taker=False,
        ),
        StrategyFamily(
            name="trend_pullback",
            use_confirmation=False,
            use_macro=True,
            use_regime=False,
            use_anchor=False,
            use_micro=True,
            require_vwap=True,
            require_rsi=False,
            require_price_reclaim=True,
            require_roc=False,
            use_volume=False,
            use_funding=False,
            use_taker=False,
        ),
        StrategyFamily(
            name="trend_pullback_momentum",
            use_confirmation=True,
            use_macro=True,
            use_regime=False,
            use_anchor=False,
            use_micro=True,
            require_vwap=True,
            require_rsi=True,
            require_price_reclaim=True,
            require_roc=True,
            use_volume=False,
            use_funding=False,
            use_taker=False,
        ),
        StrategyFamily(
            name="regime_pullback",
            use_confirmation=False,
            use_macro=True,
            use_regime=True,
            use_anchor=False,
            use_micro=True,
            require_vwap=True,
            require_rsi=False,
            require_price_reclaim=True,
            require_roc=False,
            use_volume=False,
            use_funding=False,
            use_taker=False,
        ),
        StrategyFamily(
            name="regime_anchor_pullback",
            use_confirmation=False,
            use_macro=True,
            use_regime=True,
            use_anchor=True,
            use_micro=True,
            require_vwap=True,
            require_rsi=False,
            require_price_reclaim=True,
            require_roc=False,
            use_volume=False,
            use_funding=False,
            use_taker=False,
        ),
        StrategyFamily(
            name="crowded_flow_pullback",
            use_confirmation=True,
            use_macro=True,
            use_regime=True,
            use_anchor=True,
            use_micro=True,
            require_vwap=True,
            require_rsi=False,
            require_price_reclaim=True,
            require_roc=False,
            use_volume=False,
            use_funding=True,
            use_taker=True,
        ),
        StrategyFamily(
            name="strict_confluence",
            use_confirmation=True,
            use_macro=True,
            use_regime=True,
            use_anchor=True,
            use_micro=True,
            require_vwap=True,
            require_rsi=True,
            require_price_reclaim=True,
            require_roc=True,
            use_volume=True,
            use_funding=True,
            use_taker=True,
        ),
    ]


def _family_parameter_grid(
    family: StrategyFamily,
) -> list[tuple[RuleStrategyParameters, str]]:
    side_modes = {
        "short_only": (False, True),
        "long_only": (True, False),
        "both": (True, True),
    }
    rows: list[tuple[RuleStrategyParameters, str]] = []
    for side_mode, (allow_long, allow_short) in side_modes.items():
        if not family.use_micro:
            long_micro_values = [0.0]
            short_micro_values = [0.0]
        elif side_mode == "short_only":
            long_micro_values = [0.0]
            short_micro_values = [0.0010, 0.0011, 0.00115]
        elif side_mode == "long_only":
            long_micro_values = [0.0008, 0.0010, 0.0012]
            short_micro_values = [0.0]
        else:
            long_micro_values = [0.0008, 0.0010]
            short_micro_values = [0.0010, 0.0011]
        if not family.use_funding:
            long_funding_values = [None]
            short_funding_values = [None]
        elif side_mode == "short_only":
            long_funding_values = [None]
            short_funding_values = [0.0001, 0.00005, 0.0]
        elif side_mode == "long_only":
            long_funding_values = [-0.0001, -0.00005, 0.0]
            short_funding_values = [None]
        else:
            long_funding_values = [-0.0001, 0.0]
            short_funding_values = [0.0001, 0.0]
        if not family.use_taker:
            long_taker_values = [None]
            short_taker_values = [None]
        elif side_mode == "short_only":
            long_taker_values = [None]
            short_taker_values = [0.45, 0.43]
        elif side_mode == "long_only":
            long_taker_values = [0.55, 0.57]
            short_taker_values = [None]
        else:
            long_taker_values = [0.55]
            short_taker_values = [0.45]
        volume_values = [0.0] if not family.use_volume else [1.0, 1.05]
        higher_spread_values = [0.0] if not family.use_regime else [0.0, 0.0005, 0.001]
        for (
            higher_spread,
            volume_ratio,
            long_micro,
            short_micro,
            long_funding,
            short_funding,
            long_taker,
            short_taker,
        ) in product(
            higher_spread_values,
            volume_values,
            long_micro_values,
            short_micro_values,
            long_funding_values,
            short_funding_values,
            long_taker_values,
            short_taker_values,
        ):
            rows.append(
                (
                    RuleStrategyParameters(
                        execution_timeframe="3m",
                        micro_timeframe="5m",
                        confirmation_timeframe="15m",
                        macro_timeframe="1h",
                        regime_timeframe="4h",
                        anchor_timeframe="1d",
                        use_confirmation=family.use_confirmation,
                        use_macro=family.use_macro,
                        use_regime=family.use_regime,
                        use_anchor=family.use_anchor,
                        use_micro=family.use_micro,
                        require_vwap=family.require_vwap,
                        require_rsi=family.require_rsi,
                        require_price_reclaim=family.require_price_reclaim,
                        require_roc=family.require_roc,
                        min_higher_tf_ema_spread_pct=higher_spread,
                        min_volume_ratio_20=volume_ratio,
                        min_micro_long_ema_spread_pct=long_micro,
                        min_micro_short_ema_spread_pct=short_micro,
                        max_long_funding_rate=long_funding,
                        min_short_funding_rate=short_funding,
                        min_long_taker_buy_ratio=long_taker,
                        max_short_taker_buy_ratio=short_taker,
                        allow_long_entries=allow_long,
                        allow_short_entries=allow_short,
                    ),
                    side_mode,
                )
            )
    return rows


def _score_recent(metrics_365: dict[str, Any]) -> float:
    score = 0.0
    score += metrics_365["total_pnl_after_fees_usdt"]
    score += metrics_365["profit_factor"] * 40.0
    score += metrics_365["win_rate"] * 40.0
    score += min(metrics_365["trades"], 60) * 0.5
    score -= max(metrics_365["max_drawdown_usdt"] - 30.0, 0.0) * 0.6
    return score


def _score_balanced(
    metrics_5y: dict[str, Any], metrics_365: dict[str, Any], metrics_30: dict[str, Any]
) -> float:
    score = 0.0
    score += 200.0 if metrics_5y["total_pnl_after_fees_usdt"] > 0 else 0.0
    score += 100.0 if metrics_5y["profit_factor"] >= 1.0 else 0.0
    score += metrics_5y["profit_factor"] * 50.0
    score += metrics_5y["win_rate"] * 60.0
    score += min(metrics_5y["trades"], 120) * 0.4
    score += max(metrics_365["total_pnl_after_fees_usdt"], -50.0)
    score += 80.0 if metrics_365["trades"] >= 10 else (40.0 if metrics_365["trades"] >= 3 else 0.0)
    score += metrics_30["trades"] * 1.5
    score -= max(metrics_5y["max_drawdown_usdt"] - 50.0, 0.0) * 0.4
    return score


def _params_to_payload(params: RuleStrategyParameters) -> dict[str, Any]:
    return {
        key: value
        for key, value in asdict(params).items()
        if key
        not in {
            "execution_timeframe",
            "micro_timeframe",
            "confirmation_timeframe",
            "macro_timeframe",
            "regime_timeframe",
            "anchor_timeframe",
            "atr_pct_min",
            "atr_pct_max",
            "max_spread_bps",
            "pullback_vwap_distance_pct",
            "long_rsi_min",
            "short_rsi_max",
            "long_roc_min",
            "short_roc_max",
        }
    }


def _payload_to_params(payload: dict[str, Any]) -> RuleStrategyParameters:
    return RuleStrategyParameters(
        execution_timeframe="3m",
        micro_timeframe="5m",
        confirmation_timeframe="15m",
        macro_timeframe="1h",
        regime_timeframe="4h",
        anchor_timeframe="1d",
        **payload,
    )


if __name__ == "__main__":
    raise SystemExit(main())
