from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ai_auto_trading.settings import load_settings, required_layout, validate_leverage


class SettingsTest(unittest.TestCase):
    def test_defaults_load(self) -> None:
        settings = load_settings()
        self.assertEqual(settings.trading_symbol, "BTCUSDT")
        self.assertEqual(settings.strategy_mode, "best_pair_v1")
        self.assertEqual(settings.execution_timeframe, "3m")
        self.assertEqual(settings.micro_timeframe, "5m")
        self.assertEqual(settings.regime_timeframe, "4h")
        self.assertEqual(settings.anchor_timeframe, "1d")
        self.assertEqual(settings.system_leverage_cap, 10)
        self.assertEqual(settings.local_model_base, "google/gemma-4-E2B-it")
        self.assertEqual(settings.ai_max_latency_ms, 15000)
        self.assertFalse(settings.allow_long_entries)
        self.assertTrue(settings.allow_short_entries)
        self.assertEqual(settings.max_long_funding_rate, -0.0001)
        self.assertEqual(settings.min_short_funding_rate, 0.0001)
        self.assertEqual(settings.min_long_taker_buy_ratio, 0.57)
        self.assertEqual(settings.max_short_taker_buy_ratio, 0.43)
        self.assertEqual(settings.min_micro_long_ema_spread_pct, 0.0015)
        self.assertEqual(settings.min_micro_short_ema_spread_pct, 0.00115)
        self.assertEqual(settings.min_higher_tf_ema_spread_pct, 0.001)
        self.assertEqual(settings.min_volume_ratio_20, 1.05)
        self.assertEqual(settings.exit_policy, "fixed_tp_time_stop")
        self.assertEqual(settings.fixed_take_profit_r, 1.0)
        self.assertEqual(settings.atr_trail_activation_profit_r, 0.5)
        self.assertEqual(settings.atr_trail_min_bars, 2)
        self.assertEqual(settings.max_daily_loss_r, 3.0)
        self.assertEqual(settings.max_consecutive_losses, 3)
        self.assertEqual(settings.cooldown_after_loss_minutes, 30)

    def test_required_layout_contains_policy_directories(self) -> None:
        settings = load_settings()
        layout = {str(path) for path in required_layout(settings)}
        self.assertTrue(
            any(path.endswith("/policy/approved_live") for path in layout)
        )
        self.assertTrue(any(path.endswith("/data/raw") for path in layout))

    def test_testnet_credentials_prefer_testnet_specific_variables(self) -> None:
        with patch.dict(
            os.environ,
            {
                "BINANCE_TESTNET_API_KEY": "testnet-key",
                "BINANCE_TESTNET_API_SECRET": "testnet-secret",
                "BINANCE_API_KEY": "legacy-key",
                "BINANCE_API_SECRET": "legacy-secret",
            },
            clear=False,
        ):
            settings = load_settings()
        self.assertEqual(settings.binance_testnet_api_key, "testnet-key")
        self.assertEqual(settings.binance_testnet_api_secret, "testnet-secret")
        self.assertEqual(settings.binance_api_key, "legacy-key")
        self.assertEqual(settings.binance_api_secret, "legacy-secret")

    def test_mainnet_credentials_use_mainnet_specific_variables(self) -> None:
        with patch.dict(
            os.environ,
            {
                "BINANCE_API_KEY": "mainnet-key",
                "BINANCE_API_SECRET": "mainnet-secret",
            },
            clear=False,
        ):
            settings = load_settings()
        self.assertEqual(settings.binance_api_key, "mainnet-key")
        self.assertEqual(settings.binance_api_secret, "mainnet-secret")

    def test_repo_local_env_file_is_loaded_when_process_var_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            (repo_root / ".env").write_text(
                "\n".join(
                    [
                        "BINANCE_TESTNET_API_KEY=dotenv-key",
                        "BINANCE_TESTNET_API_SECRET=dotenv-secret",
                        "LOCAL_MODEL_PATH=/tmp/gemma4-e2b/model",
                    ]
                ),
                encoding="utf-8",
            )
            with patch("ai_auto_trading.settings._repo_root", return_value=repo_root):
                with patch.dict(
                    os.environ,
                    {
                        "BINANCE_TESTNET_API_KEY": "",
                        "BINANCE_TESTNET_API_SECRET": "",
                        "LOCAL_MODEL_PATH": "",
                    },
                    clear=False,
                ):
                    settings = load_settings()
        self.assertEqual(settings.binance_testnet_api_key, "dotenv-key")
        self.assertEqual(settings.binance_testnet_api_secret, "dotenv-secret")
        self.assertEqual(settings.local_model_path, "/tmp/gemma4-e2b/model")

    def test_validate_leverage_rejects_system_cap_breach(self) -> None:
        with self.assertRaisesRegex(ValueError, "system cap"):
            validate_leverage(11)

    def test_invalid_strategy_mode_is_rejected(self) -> None:
        with patch.dict(os.environ, {"STRATEGY_MODE": "invalid-mode"}, clear=False):
            with self.assertRaisesRegex(ValueError, "STRATEGY_MODE"):
                load_settings()


if __name__ == "__main__":
    unittest.main()
