from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_auto_trading.backtest.replay import BacktestConfig, run_hybrid_backtest
from ai_auto_trading.models import TradeRecord
from ai_auto_trading.strategy.rule_based import RuleStrategyParameters
from tools.search_rule_combinations import _latest_close_time, _load_window, _score_balanced, _slice_window

_DAY_MS = 86_400_000


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
class CandidateProfile:
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
    allow_long_entries: bool
    allow_short_entries: bool
    min_higher_tf_ema_spread_pct: float
    min_volume_ratio_20: float
    min_micro_long_ema_spread_pct: float
    min_micro_short_ema_spread_pct: float
    max_long_funding_rate: float | None
    min_short_funding_rate: float | None
    min_long_taker_buy_ratio: float | None
    max_short_taker_buy_ratio: float | None


def main() -> int:
    symbol = "BTCUSDT"
    contract_path = Path("data/raw/binance/contract_klines/btcusdt/1m_5y_v1.parquet")
    mark_path = Path("data/raw/binance/mark_price_klines/btcusdt/1m_5y_v1.parquet")
    funding_path = Path("data/raw/binance/funding_rate/btcusdt/5y_v1.parquet")
    output_path = Path("data/backtests/timeframe_profile_analysis_v1.json")

    latest_close = _latest_close_time(contract_path)
    full_window = _load_window(
        contract_path=contract_path,
        mark_path=mark_path,
        funding_path=funding_path,
        symbol=symbol,
        start_time=None,
    )
    window_365 = _slice_window(full_window, latest_close - (365 * _DAY_MS))
    window_30 = _slice_window(full_window, latest_close - (30 * _DAY_MS))

    results: list[dict[str, Any]] = []
    best_by_preset: dict[str, dict[str, Any]] = {}
    for preset in _execution_presets():
        for profile in _candidate_profiles():
            params = _build_params(preset, profile)
            config = BacktestConfig(
                symbol=symbol,
                execution_timeframe=preset.execution_timeframe,
                confirmation_timeframe=preset.confirmation_timeframe or preset.execution_timeframe,
                macro_timeframe=preset.macro_timeframe or (preset.confirmation_timeframe or preset.execution_timeframe),
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

            result_30 = run_hybrid_backtest(
                contract_candles_by_timeframe=window_30["contract"],
                lower_mark_price_candles=window_30["mark"],
                funding_rate_rows=window_30["funding"],
                config=config,
                strategy_params=params,
            )
            result_365 = run_hybrid_backtest(
                contract_candles_by_timeframe=window_365["contract"],
                lower_mark_price_candles=window_365["mark"],
                funding_rate_rows=window_365["funding"],
                config=config,
                strategy_params=params,
            )
            result_5y = run_hybrid_backtest(
                contract_candles_by_timeframe=full_window["contract"],
                lower_mark_price_candles=full_window["mark"],
                funding_rate_rows=full_window["funding"],
                config=config,
                strategy_params=params,
            )
            score = _score_balanced(
                result_5y.metrics.to_dict(),
                result_365.metrics.to_dict(),
                result_30.metrics.to_dict(),
            )
            payload = {
                "preset": preset.name,
                "candidate": profile.name,
                "config": {
                    "execution_timeframe": preset.execution_timeframe,
                    "micro_timeframe": preset.micro_timeframe,
                    "confirmation_timeframe": preset.confirmation_timeframe,
                    "macro_timeframe": preset.macro_timeframe,
                    "regime_timeframe": preset.regime_timeframe,
                    "anchor_timeframe": preset.anchor_timeframe,
                },
                "params": _params_payload(profile),
                "30d": result_30.metrics.to_dict(),
                "365d": result_365.metrics.to_dict(),
                "5y": result_5y.metrics.to_dict(),
                "score": round(score, 6),
            }
            results.append(payload)
            current_best = best_by_preset.get(preset.name)
            if current_best is None or _sort_key(payload) > _sort_key(current_best):
                best_by_preset[preset.name] = payload

    results.sort(key=_sort_key, reverse=True)
    best_overall = results[0]
    best_profile = next(
        (
            row
            for row in results
            if row["365d"]["trades"] >= 5
            and row["5y"]["profit_factor"] >= 1.0
            and row["5y"]["trades"] >= 20
        ),
        best_overall,
    )

    best_preset_breakdowns: dict[str, Any] = {}
    for preset_name, payload in best_by_preset.items():
        preset = next(item for item in _execution_presets() if item.name == preset_name)
        profile = next(item for item in _candidate_profiles() if item.name == payload["candidate"])
        params = _build_params(preset, profile)
        config = BacktestConfig(
            symbol=symbol,
            execution_timeframe=preset.execution_timeframe,
            confirmation_timeframe=preset.confirmation_timeframe or preset.execution_timeframe,
            macro_timeframe=preset.macro_timeframe or (preset.confirmation_timeframe or preset.execution_timeframe),
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
        result_365 = run_hybrid_backtest(
            contract_candles_by_timeframe=window_365["contract"],
            lower_mark_price_candles=window_365["mark"],
            funding_rate_rows=window_365["funding"],
            config=config,
            strategy_params=params,
        )
        result_5y = run_hybrid_backtest(
            contract_candles_by_timeframe=full_window["contract"],
            lower_mark_price_candles=full_window["mark"],
            funding_rate_rows=full_window["funding"],
            config=config,
            strategy_params=params,
        )
        best_preset_breakdowns[preset_name] = {
            "candidate": payload["candidate"],
            "monthly_5y": _calendar_breakdown(result_5y.trade_records, "%Y-%m"),
            "daily_365d": _calendar_breakdown(result_365.trade_records, "%Y-%m-%d"),
        }

    output = {
        "best_overall": best_overall,
        "best_recent_active": best_profile,
        "best_by_execution_timeframe": best_by_preset,
        "results": results,
        "calendar_breakdowns": best_preset_breakdowns,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    return 0


def _execution_presets() -> list[ExecutionPreset]:
    return [
        ExecutionPreset(
            name="1m",
            execution_timeframe="1m",
            micro_timeframe="5m",
            confirmation_timeframe="15m",
            macro_timeframe="1h",
            regime_timeframe="4h",
            anchor_timeframe="1d",
        ),
        ExecutionPreset(
            name="3m",
            execution_timeframe="3m",
            micro_timeframe="5m",
            confirmation_timeframe="15m",
            macro_timeframe="1h",
            regime_timeframe="4h",
            anchor_timeframe="1d",
        ),
        ExecutionPreset(
            name="5m",
            execution_timeframe="5m",
            micro_timeframe="15m",
            confirmation_timeframe="1h",
            macro_timeframe="4h",
            regime_timeframe="1d",
            anchor_timeframe=None,
        ),
        ExecutionPreset(
            name="30m",
            execution_timeframe="30m",
            micro_timeframe="1h",
            confirmation_timeframe="4h",
            macro_timeframe="1d",
            regime_timeframe=None,
            anchor_timeframe=None,
        ),
        ExecutionPreset(
            name="1h",
            execution_timeframe="1h",
            micro_timeframe="4h",
            confirmation_timeframe="1d",
            macro_timeframe=None,
            regime_timeframe=None,
            anchor_timeframe=None,
        ),
    ]


def _candidate_profiles() -> list[CandidateProfile]:
    return [
        CandidateProfile(
            name="strict_short_medium",
            use_confirmation=True,
            use_macro=True,
            use_regime=True,
            use_anchor=True,
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
        CandidateProfile(
            name="strict_short_strict",
            use_confirmation=True,
            use_macro=True,
            use_regime=True,
            use_anchor=True,
            use_micro=True,
            require_vwap=True,
            require_rsi=True,
            require_price_reclaim=True,
            require_roc=True,
            allow_long_entries=False,
            allow_short_entries=True,
            min_higher_tf_ema_spread_pct=0.001,
            min_volume_ratio_20=1.05,
            min_micro_long_ema_spread_pct=0.0,
            min_micro_short_ema_spread_pct=0.00115,
            max_long_funding_rate=None,
            min_short_funding_rate=0.0001,
            min_long_taker_buy_ratio=None,
            max_short_taker_buy_ratio=0.43,
        ),
        CandidateProfile(
            name="crowded_short_medium",
            use_confirmation=True,
            use_macro=True,
            use_regime=True,
            use_anchor=True,
            use_micro=True,
            require_vwap=True,
            require_rsi=False,
            require_price_reclaim=True,
            require_roc=False,
            allow_long_entries=False,
            allow_short_entries=True,
            min_higher_tf_ema_spread_pct=0.0005,
            min_volume_ratio_20=0.0,
            min_micro_long_ema_spread_pct=0.0,
            min_micro_short_ema_spread_pct=0.0011,
            max_long_funding_rate=None,
            min_short_funding_rate=0.00005,
            min_long_taker_buy_ratio=None,
            max_short_taker_buy_ratio=0.45,
        ),
        CandidateProfile(
            name="crowded_long_relaxed",
            use_confirmation=True,
            use_macro=True,
            use_regime=True,
            use_anchor=True,
            use_micro=True,
            require_vwap=True,
            require_rsi=False,
            require_price_reclaim=True,
            require_roc=False,
            allow_long_entries=True,
            allow_short_entries=False,
            min_higher_tf_ema_spread_pct=0.0,
            min_volume_ratio_20=0.0,
            min_micro_long_ema_spread_pct=0.0008,
            min_micro_short_ema_spread_pct=0.0,
            max_long_funding_rate=0.0,
            min_short_funding_rate=None,
            min_long_taker_buy_ratio=0.55,
            max_short_taker_buy_ratio=None,
        ),
        CandidateProfile(
            name="crowded_both_medium",
            use_confirmation=True,
            use_macro=True,
            use_regime=True,
            use_anchor=True,
            use_micro=True,
            require_vwap=True,
            require_rsi=False,
            require_price_reclaim=True,
            require_roc=False,
            allow_long_entries=True,
            allow_short_entries=True,
            min_higher_tf_ema_spread_pct=0.0005,
            min_volume_ratio_20=0.0,
            min_micro_long_ema_spread_pct=0.0010,
            min_micro_short_ema_spread_pct=0.0011,
            max_long_funding_rate=0.0,
            min_short_funding_rate=0.00005,
            min_long_taker_buy_ratio=0.55,
            max_short_taker_buy_ratio=0.45,
        ),
    ]


def _build_params(preset: ExecutionPreset, profile: CandidateProfile) -> RuleStrategyParameters:
    return RuleStrategyParameters(
        execution_timeframe=preset.execution_timeframe,
        micro_timeframe=preset.micro_timeframe,
        confirmation_timeframe=preset.confirmation_timeframe or preset.execution_timeframe,
        macro_timeframe=preset.macro_timeframe or (preset.confirmation_timeframe or preset.execution_timeframe),
        regime_timeframe=preset.regime_timeframe,
        anchor_timeframe=preset.anchor_timeframe,
        use_confirmation=profile.use_confirmation and preset.confirmation_timeframe is not None,
        use_macro=profile.use_macro and preset.macro_timeframe is not None,
        use_regime=profile.use_regime and preset.regime_timeframe is not None,
        use_anchor=profile.use_anchor and preset.anchor_timeframe is not None,
        use_micro=profile.use_micro and preset.micro_timeframe is not None,
        require_vwap=profile.require_vwap,
        require_rsi=profile.require_rsi,
        require_price_reclaim=profile.require_price_reclaim,
        require_roc=profile.require_roc,
        min_higher_tf_ema_spread_pct=profile.min_higher_tf_ema_spread_pct,
        min_volume_ratio_20=profile.min_volume_ratio_20,
        min_micro_long_ema_spread_pct=profile.min_micro_long_ema_spread_pct,
        min_micro_short_ema_spread_pct=profile.min_micro_short_ema_spread_pct,
        max_long_funding_rate=profile.max_long_funding_rate,
        min_short_funding_rate=profile.min_short_funding_rate,
        min_long_taker_buy_ratio=profile.min_long_taker_buy_ratio,
        max_short_taker_buy_ratio=profile.max_short_taker_buy_ratio,
        allow_long_entries=profile.allow_long_entries,
        allow_short_entries=profile.allow_short_entries,
    )


def _params_payload(profile: CandidateProfile) -> dict[str, Any]:
    return {
        "use_confirmation": profile.use_confirmation,
        "use_macro": profile.use_macro,
        "use_regime": profile.use_regime,
        "use_anchor": profile.use_anchor,
        "use_micro": profile.use_micro,
        "require_vwap": profile.require_vwap,
        "require_rsi": profile.require_rsi,
        "require_price_reclaim": profile.require_price_reclaim,
        "require_roc": profile.require_roc,
        "allow_long_entries": profile.allow_long_entries,
        "allow_short_entries": profile.allow_short_entries,
        "min_higher_tf_ema_spread_pct": profile.min_higher_tf_ema_spread_pct,
        "min_volume_ratio_20": profile.min_volume_ratio_20,
        "min_micro_long_ema_spread_pct": profile.min_micro_long_ema_spread_pct,
        "min_micro_short_ema_spread_pct": profile.min_micro_short_ema_spread_pct,
        "max_long_funding_rate": profile.max_long_funding_rate,
        "min_short_funding_rate": profile.min_short_funding_rate,
        "min_long_taker_buy_ratio": profile.min_long_taker_buy_ratio,
        "max_short_taker_buy_ratio": profile.max_short_taker_buy_ratio,
    }


def _calendar_breakdown(records: list[TradeRecord], fmt: str) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"trades": 0, "wins": 0, "losses": 0, "pnl_after_fees_usdt": 0.0}
    )
    for record in records:
        closed_at = datetime.fromisoformat(record.closed_at)
        key = closed_at.strftime(fmt)
        bucket = buckets[key]
        bucket["trades"] += 1
        pnl = float(record.realized_pnl_after_fees_usdt or 0.0)
        bucket["pnl_after_fees_usdt"] += pnl
        if pnl >= 0.0:
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


def _sort_key(payload: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        payload["score"],
        payload["5y"]["total_pnl_after_fees_usdt"],
        payload["365d"]["total_pnl_after_fees_usdt"],
        payload["5y"]["win_rate"],
    )


if __name__ == "__main__":
    raise SystemExit(main())
