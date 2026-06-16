from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_auto_trading.backtest.replay import BacktestResult


@dataclass(frozen=True)
class AIComparisonReport:
    rule_only_metrics: dict[str, Any]
    ai_gated_metrics: dict[str, Any]
    deltas: dict[str, Any]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_ai_comparison_report(
    *,
    rule_only_result: "BacktestResult",
    ai_gated_result: "BacktestResult",
) -> AIComparisonReport:
    rule_metrics = rule_only_result.metrics.to_dict()
    ai_metrics = ai_gated_result.metrics.to_dict()
    deltas = {
        "trades": ai_metrics["trades"] - rule_metrics["trades"],
        "total_pnl_after_fees_usdt": (
            ai_metrics["total_pnl_after_fees_usdt"] - rule_metrics["total_pnl_after_fees_usdt"]
        ),
        "profit_factor": ai_metrics["profit_factor"] - rule_metrics["profit_factor"],
        "max_drawdown_usdt": ai_metrics["max_drawdown_usdt"] - rule_metrics["max_drawdown_usdt"],
        "win_rate": ai_metrics["win_rate"] - rule_metrics["win_rate"],
    }
    summary = (
        "AI gate improved net pnl and profit factor."
        if deltas["total_pnl_after_fees_usdt"] > 0 and deltas["profit_factor"] >= 0
        else "AI gate did not improve the baseline across the main risk-adjusted metrics."
    )
    return AIComparisonReport(
        rule_only_metrics=rule_metrics,
        ai_gated_metrics=ai_metrics,
        deltas=deltas,
        summary=summary,
    )
