from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from ai_auto_trading.data.mark_price_recorder import MarkPriceRecorder


class MarkPriceRecorderTest(unittest.TestCase):
    def _sample_payload(self) -> dict[str, str | int]:
        return {
            "e": "markPriceUpdate",
            "E": 1562305380000,
            "s": "BTCUSDT",
            "p": "11794.15000000",
            "ap": "11794.15000000",
            "i": "11784.62659091",
            "P": "11784.25641265",
            "r": "0.00038167",
            "T": 1562306400000,
        }

    def test_normalize_message(self) -> None:
        recorder = MarkPriceRecorder()
        event = recorder.normalize_message(self._sample_payload())
        self.assertEqual(event.symbol, "BTCUSDT")
        self.assertAlmostEqual(event.mark_price, 11794.15)
        self.assertEqual(event.event_type, "markPriceUpdate")

    def test_mainnet_stream_url_uses_market_route(self) -> None:
        recorder = MarkPriceRecorder()
        self.assertIn(
            "fstream.binance.com/market/ws/btcusdt@markPrice@1s",
            recorder.stream_url,
        )

    def test_record_payloads_writes_sqlite_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "mark_price.sqlite3"
            recorder = MarkPriceRecorder(database_path=db_path)
            result = recorder.record_payloads([self._sample_payload(), self._sample_payload()])
            self.assertEqual(result.messages_recorded, 2)

            with sqlite3.connect(db_path) as conn:
                message_count = conn.execute(
                    "SELECT COUNT(*) FROM mark_price_events"
                ).fetchone()[0]
                health_count = conn.execute(
                    "SELECT COUNT(*) FROM recorder_health_events"
                ).fetchone()[0]
            self.assertEqual(message_count, 2)
            self.assertEqual(health_count, 2)

    def test_record_payloads_logs_gap_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "mark_price.sqlite3"
            first_payload = self._sample_payload()
            second_payload = dict(first_payload)
            second_payload["E"] = int(first_payload["E"]) + 3000
            recorder = MarkPriceRecorder(database_path=db_path, stream_speed="1s")
            result = recorder.record_payloads([first_payload, second_payload])
            self.assertEqual(result.messages_recorded, 2)
            self.assertEqual(result.health_events_recorded, 3)

            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    """
                    SELECT event_type, details_json
                    FROM recorder_health_events
                    WHERE event_type = 'mark_price_gap_detected'
                    """
                ).fetchall()
            self.assertEqual(len(rows), 1)
            self.assertIn('"gap_ms": 3000', rows[0][1])


if __name__ == "__main__":
    unittest.main()
