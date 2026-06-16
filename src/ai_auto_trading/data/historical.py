from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pyarrow as pa
import pyarrow.parquet as pq

from ai_auto_trading.settings import Settings, load_settings


def _utc_now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def _iso_from_ms(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


@dataclass(frozen=True)
class HistoricalBatchResult:
    dataset: str
    interval: str | None
    rows: int
    path: Path


@dataclass(frozen=True)
class HistoricalBackfillResult:
    dataset: str
    interval: str | None
    total_rows: int
    path: Path
    batches: int


class BinanceHistoricalDownloader:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.base_url = self.settings.binance_futures_base_url.rstrip("/")

    def _get_json(self, path: str, params: dict[str, Any]) -> Any:
        query = urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{query}"
        last_error: Exception | None = None
        for attempt in range(6):
            try:
                with urlopen(url, timeout=30) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                last_error = exc
                if exc.code not in {429, 500, 502, 503, 504} or attempt == 5:
                    raise
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(2 ** attempt, 16)
                time.sleep(delay)
            except URLError as exc:
                last_error = exc
                if attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 16))
        if last_error is not None:
            raise last_error
        raise RuntimeError("unexpected downloader state")

    def _fetch_rows(
        self,
        *,
        dataset: str,
        symbol: str,
        interval_or_period: str | None = None,
        limit: int = 500,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[dict[str, Any]]:
        if dataset == "contract_klines":
            if not interval_or_period:
                raise ValueError("interval_or_period is required for contract_klines")
            return self.fetch_contract_klines(
                symbol=symbol,
                interval=interval_or_period,
                limit=limit,
                start_time=start_time,
                end_time=end_time,
            )
        if dataset == "mark_price_klines":
            if not interval_or_period:
                raise ValueError("interval_or_period is required for mark_price_klines")
            return self.fetch_mark_price_klines(
                symbol=symbol,
                interval=interval_or_period,
                limit=limit,
                start_time=start_time,
                end_time=end_time,
            )
        if dataset == "funding_rate":
            return self.fetch_funding_rate(
                symbol=symbol,
                limit=limit,
                start_time=start_time,
                end_time=end_time,
            )
        if dataset == "open_interest_hist":
            if not interval_or_period:
                raise ValueError("interval_or_period is required for open_interest_hist")
            return self.fetch_open_interest_hist(
                symbol=symbol,
                period=interval_or_period,
                limit=limit,
                start_time=start_time,
                end_time=end_time,
            )
        raise ValueError(f"unsupported dataset: {dataset}")

    def fetch_contract_klines(
        self,
        *,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[dict[str, Any]]:
        payload = self._get_json(
            "/fapi/v1/klines",
            {
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
                "startTime": start_time,
                "endTime": end_time,
            },
        )
        return [self._normalize_kline_row(symbol, interval, "contract_klines", row) for row in payload]

    def fetch_mark_price_klines(
        self,
        *,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[dict[str, Any]]:
        payload = self._get_json(
            "/fapi/v1/markPriceKlines",
            {
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
                "startTime": start_time,
                "endTime": end_time,
            },
        )
        return [self._normalize_kline_row(symbol, interval, "mark_price_klines", row) for row in payload]

    def fetch_funding_rate(
        self,
        *,
        symbol: str,
        limit: int = 500,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[dict[str, Any]]:
        payload = self._get_json(
            "/fapi/v1/fundingRate",
            {
                "symbol": symbol,
                "limit": limit,
                "startTime": start_time,
                "endTime": end_time,
            },
        )
        rows: list[dict[str, Any]] = []
        for item in payload:
            funding_time = int(item["fundingTime"])
            rows.append(
                {
                    "dataset": "funding_rate",
                    "symbol": item["symbol"],
                    "funding_time": funding_time,
                    "funding_time_iso": _iso_from_ms(funding_time),
                    "funding_rate": float(item["fundingRate"]),
                    "mark_price": _optional_float(item.get("markPrice")),
                    "collected_at_ms": _utc_now_ms(),
                }
            )
        return rows

    def fetch_open_interest_hist(
        self,
        *,
        symbol: str,
        period: str,
        limit: int = 30,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[dict[str, Any]]:
        payload = self._get_json(
            "/futures/data/openInterestHist",
            {
                "symbol": symbol,
                "period": period,
                "limit": limit,
                "startTime": start_time,
                "endTime": end_time,
            },
        )
        rows: list[dict[str, Any]] = []
        for item in payload:
            timestamp = int(item["timestamp"])
            rows.append(
                {
                    "dataset": "open_interest_hist",
                    "symbol": item["symbol"],
                    "period": period,
                    "timestamp": timestamp,
                    "timestamp_iso": _iso_from_ms(timestamp),
                    "sum_open_interest": float(item["sumOpenInterest"]),
                    "sum_open_interest_value": float(item["sumOpenInterestValue"]),
                    "cmc_circulating_supply": float(item["CMCCirculatingSupply"]),
                    "collected_at_ms": _utc_now_ms(),
                }
            )
        return rows

    def write_parquet(self, rows: list[dict[str, Any]], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(rows)
        tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
        pq.write_table(table, tmp_path)
        tmp_path.replace(output_path)
        return output_path

    def default_output_path(self, dataset: str, symbol: str, interval_or_period: str | None) -> Path:
        base = self.settings.data_dir / "raw" / "binance" / dataset / symbol.lower()
        suffix = interval_or_period if interval_or_period else "latest"
        return base / f"{suffix}.parquet"

    def fetch_and_store(
        self,
        *,
        dataset: str,
        symbol: str,
        interval_or_period: str | None = None,
        limit: int = 500,
        start_time: int | None = None,
        end_time: int | None = None,
        output_path: Path | None = None,
    ) -> HistoricalBatchResult:
        rows = self._fetch_rows(
            dataset=dataset,
            symbol=symbol,
            interval_or_period=interval_or_period,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
        )
        target_path = output_path or self.default_output_path(dataset, symbol, interval_or_period)
        self.write_parquet(rows, target_path)
        return HistoricalBatchResult(
            dataset=dataset,
            interval=interval_or_period,
            rows=len(rows),
            path=target_path,
        )

    def backfill_and_store(
        self,
        *,
        dataset: str,
        symbol: str,
        interval_or_period: str | None = None,
        start_time: int,
        end_time: int,
        limit: int = 500,
        output_path: Path | None = None,
    ) -> HistoricalBackfillResult:
        if start_time >= end_time:
            raise ValueError("start_time must be smaller than end_time")

        all_rows: list[dict[str, Any]] = []
        cursor = start_time
        batches = 0

        while cursor < end_time:
            rows = self._fetch_rows(
                dataset=dataset,
                symbol=symbol,
                interval_or_period=interval_or_period,
                limit=limit,
                start_time=cursor,
                end_time=end_time,
            )
            if not rows:
                break
            all_rows.extend(rows)
            batches += 1

            last_timestamp = self._row_advance_timestamp(dataset, rows[-1])
            if last_timestamp <= cursor:
                break
            cursor = last_timestamp

        target_path = output_path or self.default_output_path(dataset, symbol, interval_or_period)
        self.write_parquet(all_rows, target_path)
        return HistoricalBackfillResult(
            dataset=dataset,
            interval=interval_or_period,
            total_rows=len(all_rows),
            path=target_path,
            batches=batches,
        )

    @staticmethod
    def _normalize_kline_row(
        symbol: str,
        interval: str,
        dataset: str,
        row: list[Any],
    ) -> dict[str, Any]:
        open_time = int(row[0])
        close_time = int(row[6])
        return {
            "dataset": dataset,
            "symbol": symbol,
            "interval": interval,
            "open_time": open_time,
            "open_time_iso": _iso_from_ms(open_time),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
            "close_time": close_time,
            "close_time_iso": _iso_from_ms(close_time),
            "quote_asset_volume": float(row[7]),
            "number_of_trades": int(row[8]),
            "taker_buy_base_asset_volume": float(row[9]),
            "taker_buy_quote_asset_volume": float(row[10]),
            "ignore": str(row[11]),
            "collected_at_ms": _utc_now_ms(),
        }

    @staticmethod
    def _row_advance_timestamp(dataset: str, row: dict[str, Any]) -> int:
        if dataset in {"contract_klines", "mark_price_klines"}:
            return int(row["close_time"]) + 1
        if dataset == "funding_rate":
            return int(row["funding_time"]) + 1
        if dataset == "open_interest_hist":
            return int(row["timestamp"]) + 1
        raise ValueError(f"unsupported dataset: {dataset}")
