from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from ai_auto_trading.data.historical import BinanceHistoricalDownloader
from ai_auto_trading.settings import load_settings


class HistoricalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.downloader = BinanceHistoricalDownloader(load_settings())

    def test_normalize_contract_kline_row(self) -> None:
        row = [
            1776258780000,
            "74250.10",
            "74263.70",
            "74233.80",
            "74263.70",
            "27.407",
            1776258839999,
            "2034923.91750",
            1218,
            "15.863",
            "1177792.63830",
            "0",
        ]
        normalized = self.downloader._normalize_kline_row(
            "BTCUSDT", "1m", "contract_klines", row
        )
        self.assertEqual(normalized["symbol"], "BTCUSDT")
        self.assertEqual(normalized["interval"], "1m")
        self.assertEqual(normalized["number_of_trades"], 1218)
        self.assertAlmostEqual(normalized["close"], 74263.70)

    def test_write_parquet_roundtrip(self) -> None:
        rows = [
            {
                "dataset": "contract_klines",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "open_time": 1,
                "open_time_iso": "1970-01-01T00:00:00+00:00",
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "volume": 10.0,
                "close_time": 2,
                "close_time_iso": "1970-01-01T00:00:00+00:00",
                "quote_asset_volume": 15.0,
                "number_of_trades": 1,
                "taker_buy_base_asset_volume": 5.0,
                "taker_buy_quote_asset_volume": 7.5,
                "ignore": "0",
                "collected_at_ms": 3,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "sample.parquet"
            self.downloader.write_parquet(rows, output_path)
            table = pq.read_table(output_path)
            self.assertEqual(table.num_rows, 1)
            self.assertIn("symbol", table.column_names)

    def test_row_advance_timestamp_supports_known_datasets(self) -> None:
        self.assertEqual(
            self.downloader._row_advance_timestamp(
                "contract_klines", {"close_time": 10}
            ),
            11,
        )
        self.assertEqual(
            self.downloader._row_advance_timestamp(
                "funding_rate", {"funding_time": 20}
            ),
            21,
        )

    def test_backfill_and_store_rejects_invalid_range(self) -> None:
        with self.assertRaises(ValueError):
            self.downloader.backfill_and_store(
                dataset="contract_klines",
                symbol="BTCUSDT",
                interval_or_period="1m",
                start_time=10,
                end_time=10,
            )

    def test_backfill_and_store_accumulates_batches_without_parquet_roundtrip(self) -> None:
        calls: list[int] = []

        def fake_fetch_rows(**kwargs):
            cursor = kwargs["start_time"]
            calls.append(cursor)
            if len(calls) == 1:
                return [
                    {"close_time": 19, "open_time": 0},
                    {"close_time": 29, "open_time": 20},
                ]
            if len(calls) == 2:
                return [
                    {"close_time": 39, "open_time": 30},
                ]
            return []

        self.downloader._fetch_rows = fake_fetch_rows  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "backfill.parquet"
            result = self.downloader.backfill_and_store(
                dataset="contract_klines",
                symbol="BTCUSDT",
                interval_or_period="1m",
                start_time=10,
                end_time=100,
                output_path=output_path,
            )

        self.assertEqual(result.total_rows, 3)
        self.assertEqual(result.batches, 2)
        self.assertEqual(calls, [10, 30, 40])

    def test_fetch_funding_rate_tolerates_empty_mark_price(self) -> None:
        self.downloader._get_json = lambda *_args, **_kwargs: [  # type: ignore[method-assign]
            {
                "symbol": "BTCUSDT",
                "fundingTime": "123",
                "fundingRate": "0.0001",
                "markPrice": "",
            }
        ]

        rows = self.downloader.fetch_funding_rate(symbol="BTCUSDT")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["funding_rate"], 0.0001)
        self.assertIsNone(rows[0]["mark_price"])


if __name__ == "__main__":
    unittest.main()
