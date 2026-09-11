#!/usr/bin/env python3
"""Collect a reconstructable Binance Spot limit-order-book feed.

The public Binance feed contains aggregated price levels, not individual order
identifiers.  Each daily gzip JSONL partition starts with a book snapshot and is
followed by diff-depth events, so it can be replayed independently.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import logging
import os
import random
import re
import signal
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable


LOG = logging.getLogger("binance_lob")
DEFAULT_REST_BASE = "https://data-api.binance.vision"
DEFAULT_WS_BASE = "wss://data-stream.binance.vision:443"
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9_]{2,30}$")


class SequenceGap(RuntimeError):
    """Raised when one or more order-book updates have been missed."""


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def utc_date_from_ns(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, UTC).date().isoformat()


def compact_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def is_zero_quantity(quantity: str) -> bool:
    return Decimal(quantity) == 0


@dataclass
class LocalOrderBook:
    bids: dict[str, str]
    asks: dict[str, str]
    last_update_id: int

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> "LocalOrderBook":
        return cls(
            bids={price: quantity for price, quantity in snapshot["bids"]},
            asks={price: quantity for price, quantity in snapshot["asks"]},
            last_update_id=int(snapshot["lastUpdateId"]),
        )

    def apply(self, event: dict[str, Any]) -> bool:
        """Apply a Binance diff-depth event; return False for an old event."""
        first_update_id = int(event["U"])
        final_update_id = int(event["u"])

        if final_update_id <= self.last_update_id:
            return False
        if first_update_id > self.last_update_id + 1:
            raise SequenceGap(
                f"expected update <= {self.last_update_id + 1}, got {first_update_id}"
            )

        self._apply_side(self.bids, event["b"])
        self._apply_side(self.asks, event["a"])
        self.last_update_id = final_update_id
        return True

    @staticmethod
    def _apply_side(side: dict[str, str], updates: Iterable[list[str]]) -> None:
        for price, quantity in updates:
            if is_zero_quantity(quantity):
                side.pop(price, None)
            else:
                side[price] = quantity

    def snapshot_record(self, received_at_ns: int, reason: str) -> dict[str, Any]:
        return {
            "type": "snapshot",
            "received_at_ns": received_at_ns,
            "reason": reason,
            "last_update_id": self.last_update_id,
            "bids": sorted(self.bids.items(), key=lambda level: Decimal(level[0]), reverse=True),
            "asks": sorted(self.asks.items(), key=lambda level: Decimal(level[0])),
        }


class DailyGzipWriter:
    """Write UTC-day partitions and atomically finalize them on clean shutdown."""

    def __init__(
        self,
        output_dir: Path,
        symbol: str,
        snapshot_limit: int,
        stream_name: str,
        flush_every: int,
    ) -> None:
        self.output_dir = output_dir
        self.symbol = symbol
        self.snapshot_limit = snapshot_limit
        self.stream_name = stream_name
        self.flush_every = flush_every
        self.session_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        self.current_day: str | None = None
        self.handle: gzip.GzipFile | None = None
        self.text_handle: Any = None
        self.temp_path: Path | None = None
        self.records_since_flush = 0

    def ensure_day(self, received_at_ns: int, book: LocalOrderBook, reason: str) -> None:
        day = utc_date_from_ns(received_at_ns)
        if day == self.current_day:
            return
        self._finalize_current_file()
        self._open(day)
        self.write(book.snapshot_record(received_at_ns, reason=reason))

    def write_synchronized_snapshot(
        self, received_at_ns: int, book: LocalOrderBook, reason: str
    ) -> None:
        """Start a partition or mark an in-day reconnect with a fresh snapshot."""
        if utc_date_from_ns(received_at_ns) == self.current_day:
            self.write(book.snapshot_record(received_at_ns, reason=reason))
        else:
            self.ensure_day(received_at_ns, book, reason=reason)

    def _open(self, day: str) -> None:
        partition = self.output_dir / f"symbol={self.symbol}" / f"date={day}"
        partition.mkdir(parents=True, exist_ok=True)
        basename = f"part-{self.session_id}-{os.getpid()}.jsonl.gz"
        self.temp_path = partition / f"{basename}.inprogress"
        self.handle = gzip.open(self.temp_path, mode="wb", compresslevel=6)
        self.text_handle = self.handle
        self.current_day = day
        self.records_since_flush = 0
        self.write(
            {
                "type": "metadata",
                "schema_version": 1,
                "exchange": "binance",
                "market": "spot",
                "feed": "diff_depth",
                "symbol": self.symbol,
                "stream": self.stream_name,
                "event_time_unit": "microsecond",
                "receive_time_unit": "nanosecond",
                "snapshot_limit_per_side": self.snapshot_limit,
                "created_at": utc_now_iso(),
            }
        )
        LOG.info("writing %s", self.temp_path)

    def write(self, record: dict[str, Any]) -> None:
        if self.text_handle is None:
            raise RuntimeError("writer has not been opened")
        payload = (compact_json(record) + "\n").encode("utf-8")
        self.text_handle.write(payload)
        self.records_since_flush += 1
        if self.records_since_flush >= self.flush_every:
            self.text_handle.flush()
            self.records_since_flush = 0

    def write_event(self, received_at_ns: int, event: dict[str, Any]) -> None:
        self.write(
            {
                "type": "depth_update",
                "received_at_ns": received_at_ns,
                "data": event,
            }
        )

    def _finalize_current_file(self) -> None:
        if self.handle is None or self.temp_path is None:
            return
        self.handle.close()
        final_path = self.temp_path.with_suffix("")
        self.temp_path.replace(final_path)
        LOG.info("finalized %s", final_path)
        self.handle = None
        self.text_handle = None
        self.temp_path = None
        self.current_day = None

    def close(self) -> None:
        self._finalize_current_file()


def fetch_snapshot(rest_base: str, symbol: str, limit: int, timeout: float) -> dict[str, Any]:
    query = urllib.parse.urlencode({"symbol": symbol, "limit": limit})
    url = f"{rest_base.rstrip('/')}/api/v3/depth?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "microstructure-lob/1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"snapshot request failed: {error}") from error

    if not isinstance(payload, dict) or not {"lastUpdateId", "bids", "asks"} <= payload.keys():
        raise RuntimeError(f"unexpected snapshot response: {payload!r}")
    return payload


async def receive_depth_events(websocket: Any, queue: asyncio.Queue[tuple[int, dict[str, Any]]]) -> None:
    async for raw_message in websocket:
        received_at_ns = time.time_ns()
        event = json.loads(raw_message)
        if event.get("e") == "serverShutdown":
            raise RuntimeError("Binance announced a server shutdown")
        if event.get("e") != "depthUpdate":
            LOG.warning("ignoring unexpected websocket message: %s", event)
            continue
        await queue.put((received_at_ns, event))


async def synchronize_book(
    queue: asyncio.Queue[tuple[int, dict[str, Any]]],
    args: argparse.Namespace,
) -> tuple[LocalOrderBook, list[tuple[int, dict[str, Any]]], int]:
    """Synchronize a REST snapshot with already-buffered WebSocket events."""
    first = await asyncio.wait_for(queue.get(), timeout=args.message_timeout)
    buffered = [first]

    for attempt in range(1, args.snapshot_retries + 1):
        snapshot = await asyncio.to_thread(
            fetch_snapshot,
            args.rest_base,
            args.symbol,
            args.snapshot_limit,
            args.http_timeout,
        )
        snapshot_received_at_ns = time.time_ns()
        while not queue.empty():
            buffered.append(queue.get_nowait())

        first_stream_id = int(buffered[0][1]["U"])
        snapshot_id = int(snapshot["lastUpdateId"])
        if snapshot_id >= first_stream_id:
            break
        LOG.warning(
            "snapshot %s predates first buffered update %s; retrying (%s/%s)",
            snapshot_id,
            first_stream_id,
            attempt,
            args.snapshot_retries,
        )
    else:
        raise RuntimeError("could not synchronize REST snapshot with WebSocket stream")

    book = LocalOrderBook.from_snapshot(snapshot)
    relevant = [item for item in buffered if int(item[1]["u"]) > book.last_update_id]
    if relevant and int(relevant[0][1]["U"]) > book.last_update_id + 1:
        raise SequenceGap("gap between REST snapshot and first buffered event")
    return book, relevant, snapshot_received_at_ns


async def websocket_session(
    args: argparse.Namespace,
    writer: DailyGzipWriter,
    stop_event: asyncio.Event,
) -> None:
    try:
        from websockets.asyncio.client import connect
    except ImportError as error:
        raise RuntimeError(
            "missing dependency: install it with `python -m pip install -r requirements.txt`"
        ) from error

    stream_name = f"{args.symbol.lower()}@depth@{args.speed}"
    url = f"{args.ws_base.rstrip('/')}/ws/{stream_name}?timeUnit=MICROSECOND"
    queue: asyncio.Queue[tuple[int, dict[str, Any]]] = asyncio.Queue(maxsize=args.queue_size)

    async with connect(
        url,
        open_timeout=args.http_timeout,
        close_timeout=10,
        ping_interval=None,
        max_queue=4096,
    ) as websocket:
        LOG.info("connected to %s", url)
        receiver = asyncio.create_task(receive_depth_events(websocket, queue))
        try:
            book, buffered, snapshot_time_ns = await synchronize_book(queue, args)
            writer.write_synchronized_snapshot(snapshot_time_ns, book, reason="session_start")
            LOG.info("synchronized at update ID %s", book.last_update_id)

            for received_at_ns, event in buffered:
                writer.ensure_day(received_at_ns, book, reason="utc_day_start")
                if book.apply(event):
                    writer.write_event(received_at_ns, event)

            while not stop_event.is_set():
                get_event = asyncio.create_task(queue.get())
                stop_wait = asyncio.create_task(stop_event.wait())
                done, pending = await asyncio.wait(
                    {get_event, stop_wait},
                    timeout=args.message_timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                if not done:
                    raise TimeoutError(f"no depth update for {args.message_timeout:g} seconds")
                if stop_wait in done:
                    return

                received_at_ns, event = get_event.result()
                writer.ensure_day(received_at_ns, book, reason="utc_day_start")
                if book.apply(event):
                    writer.write_event(received_at_ns, event)
        finally:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)


async def collect(args: argparse.Namespace) -> None:
    try:
        import websockets.asyncio.client  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "missing dependency: install it with `python -m pip install -r requirements.txt`"
        ) from error

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signal_name in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signal_name, stop_event.set)
        except NotImplementedError:
            pass

    if args.duration is not None:
        loop.call_later(args.duration, stop_event.set)

    stream_name = f"{args.symbol.lower()}@depth@{args.speed}"
    writer = DailyGzipWriter(
        output_dir=args.output_dir,
        symbol=args.symbol,
        snapshot_limit=args.snapshot_limit,
        stream_name=stream_name,
        flush_every=args.flush_every,
    )
    failures = 0
    try:
        while not stop_event.is_set():
            try:
                await websocket_session(args, writer, stop_event)
                failures = 0
            except asyncio.CancelledError:
                raise
            except Exception as error:
                failures += 1
                delay = min(args.max_retry_delay, 2 ** min(failures, 6))
                delay *= random.uniform(0.8, 1.2)
                LOG.warning("feed interrupted (%s); resynchronizing in %.1fs", error, delay)
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=delay)
                except TimeoutError:
                    pass
    finally:
        writer.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Binance Spot LOB snapshots and diff-depth updates into daily gzip JSONL files."
    )
    parser.add_argument("--symbol", default="BTCUSDT", type=str.upper)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/raw/binance/spot/lob")
    )
    parser.add_argument("--speed", choices=("100ms", "1000ms"), default="100ms")
    parser.add_argument("--snapshot-limit", type=int, default=5000)
    parser.add_argument("--rest-base", default=DEFAULT_REST_BASE)
    parser.add_argument("--ws-base", default=DEFAULT_WS_BASE)
    parser.add_argument("--http-timeout", type=float, default=20.0)
    parser.add_argument("--message-timeout", type=float, default=60.0)
    parser.add_argument("--snapshot-retries", type=int, default=5)
    parser.add_argument("--max-retry-delay", type=float, default=60.0)
    parser.add_argument("--queue-size", type=int, default=100_000)
    parser.add_argument("--flush-every", type=int, default=1_000)
    parser.add_argument(
        "--duration",
        type=float,
        help="stop after this many seconds; omit to run continuously",
    )
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    args = parser.parse_args(argv)

    if not SYMBOL_PATTERN.fullmatch(args.symbol):
        parser.error("--symbol must contain only A-Z, 0-9, or underscore")
    if not 1 <= args.snapshot_limit <= 5000:
        parser.error("--snapshot-limit must be between 1 and 5000")
    for field in ("http_timeout", "message_timeout", "max_retry_delay", "queue_size", "flush_every"):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.snapshot_retries <= 0:
        parser.error("--snapshot-retries must be positive")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(collect(args))
    except KeyboardInterrupt:
        pass
    except RuntimeError as error:
        LOG.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
