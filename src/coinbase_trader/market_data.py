from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Callable

from coinbase.rest import RESTClient

from coinbase_trader.models import Candle, ProductSnapshot

logger = logging.getLogger(__name__)


class MarketDataClient:
    def __init__(
        self,
        product_id: str,
        granularity: str,
        lookback_candles: int,
        timeout_seconds: int = 20,
        retries: int = 2,
    ) -> None:
        self._product_id = product_id
        self._granularity = granularity
        self._lookback_candles = lookback_candles
        self._retries = retries
        self._public_client = RESTClient(timeout=timeout_seconds)

    def fetch_snapshot(self) -> ProductSnapshot:
        product = self._call_with_retry(lambda: self._public_client.get_public_product(self._product_id))
        book = self._call_with_retry(lambda: self._public_client.get_public_product_book(self._product_id, limit=10))
        end = datetime.now(UTC)
        start = end - self._granularity_delta() * self._lookback_candles
        candles_response = self._call_with_retry(
            lambda: self._public_client.get_public_candles(
                product_id=self._product_id,
                start=int(start.timestamp()),
                end=int(end.timestamp()),
                granularity=self._granularity,
            )
        )

        price = Decimal(str(product["price"]))
        best_bid = Decimal(str(book["pricebook"]["bids"][0]["price"]))
        best_ask = Decimal(str(book["pricebook"]["asks"][0]["price"]))
        spread_bps = ((best_ask - best_bid) / price) * Decimal("10000")

        candles = [
            Candle(
                start=datetime.fromtimestamp(int(candle["start"]), UTC),
                low=Decimal(str(candle["low"])),
                high=Decimal(str(candle["high"])),
                open=Decimal(str(candle["open"])),
                close=Decimal(str(candle["close"])),
                volume=Decimal(str(candle["volume"])),
            )
            for candle in sorted(candles_response["candles"], key=lambda item: int(item["start"]))
        ]
        logger.info("Fetched %s candles for %s", len(candles), self._product_id)

        return ProductSnapshot(
            product_id=self._product_id,
            price=price,
            best_bid=best_bid,
            best_ask=best_ask,
            spread_bps=spread_bps,
            candles=candles,
        )

    def _granularity_delta(self) -> timedelta:
        mapping: dict[str, timedelta] = {
            "ONE_MINUTE": timedelta(minutes=1),
            "FIVE_MINUTE": timedelta(minutes=5),
            "FIFTEEN_MINUTE": timedelta(minutes=15),
            "THIRTY_MINUTE": timedelta(minutes=30),
            "ONE_HOUR": timedelta(hours=1),
            "TWO_HOUR": timedelta(hours=2),
            "SIX_HOUR": timedelta(hours=6),
            "ONE_DAY": timedelta(days=1),
        }
        try:
            return mapping[self._granularity]
        except KeyError as exc:
            raise ValueError(f"Unsupported granularity: {self._granularity}") from exc

    def _call_with_retry(self, fn: Callable[[], Any]) -> Any:
        attempts = self._retries + 1
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == attempts:
                    break
                delay = min(3.0, 0.5 * attempt)
                logger.warning(
                    "Market data call failed for %s (attempt %s/%s): %s; retrying in %.1fs",
                    self._product_id,
                    attempt,
                    attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Market data call failed unexpectedly without error.")

