from __future__ import annotations

import unittest
from dataclasses import replace

from ai_auto_trading.settings import load_settings
from ai_auto_trading.strategy.runtime_profiles import (
    build_runtime_entry_profiles,
    build_runtime_entry_profiles_for_symbol,
    required_runtime_timeframes,
    runtime_stream_interval,
)


class RuntimeProfilesTest(unittest.TestCase):
    def test_single_profile_uses_settings(self) -> None:
        settings = replace(
            load_settings(),
            strategy_mode="single_profile",
            execution_timeframe="3m",
            micro_timeframe="5m",
            confirmation_timeframe="15m",
            macro_timeframe="1h",
            regime_timeframe="4h",
            anchor_timeframe="1d",
            allow_long_entries=True,
            allow_short_entries=False,
        )
        profiles = build_runtime_entry_profiles(settings)
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0].name, "single_profile")
        self.assertEqual(profiles[0].params.execution_timeframe, "3m")
        self.assertTrue(profiles[0].params.allow_long_entries)
        self.assertFalse(profiles[0].params.allow_short_entries)

    def test_best_pair_v1_builds_expected_profiles(self) -> None:
        settings = replace(load_settings(), strategy_mode="best_pair_v1")
        profiles = build_runtime_entry_profiles(settings)
        self.assertEqual([profile.name for profile in profiles], ["short_best_5m", "long_best_30m"])
        self.assertEqual(runtime_stream_interval(settings), "5m")
        self.assertEqual(
            required_runtime_timeframes(settings),
            ["5m", "15m", "1h", "4h", "1d", "30m"],
        )
        self.assertEqual(profiles[0].params.execution_timeframe, "5m")
        self.assertEqual(profiles[1].params.execution_timeframe, "30m")

    def test_multi_symbol_priority_profiles_are_symbol_specific(self) -> None:
        settings = replace(load_settings(), strategy_mode="multi_symbol_priority_v1")
        self.assertEqual(
            [profile.name for profile in build_runtime_entry_profiles_for_symbol(settings, "BTCUSDT")],
            ["short_best_5m", "long_best_30m"],
        )
        eth_profiles = build_runtime_entry_profiles_for_symbol(settings, "ETHUSDT")
        sol_profiles = build_runtime_entry_profiles_for_symbol(settings, "SOLUSDT")
        self.assertEqual([profile.name for profile in eth_profiles], ["eth_medium_15m_both_pullback_time_stop_only"])
        self.assertEqual([profile.name for profile in sol_profiles], ["sol_medium_15m_both_momentum_time_stop_only"])
        self.assertFalse(eth_profiles[0].params.require_rsi)
        self.assertTrue(sol_profiles[0].params.require_rsi)
        self.assertEqual(runtime_stream_interval(settings, symbol="BTCUSDT"), "5m")
        self.assertEqual(runtime_stream_interval(settings, symbol="ETHUSDT"), "15m")
        self.assertEqual(runtime_stream_interval(settings, symbol="SOLUSDT"), "15m")
        self.assertEqual(required_runtime_timeframes(settings, symbol="ETHUSDT"), ["15m", "30m", "1h", "4h"])


if __name__ == "__main__":
    unittest.main()
