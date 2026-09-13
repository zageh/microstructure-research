#!/usr/bin/env python3
"""Replay Binance LOB captures produced by ``binance_lob.py``.

Input order is authoritative: every snapshot resets the local book, and each
subsequent diff-depth event is checked before it is applied.  Reconstructed
Top-N book states are written as gzip-compressed Parquet to stdout or a file.
"""

from __future__ import annotations

import argparse
import gzip
import heapq
import json
import logging
import sys
import time
from dataclasses import dataclass
from decimal import Decimal #将科学计数法转为具体数字
from pathlib import Path
from typing import Any, Iterator, TextIO

import pyarrow as pa
import pyarrow.parquet as pq

from binance_lob import LocalOrderBook, SequenceGap


LOG = logging.getLogger("replay_lob")


class ReplayError(RuntimeError):
    """Raised when a capture cannot be replayed safely."""


@dataclass
class ReplayStats:
    files: int = 0
    records: int = 0
    snapshots: int = 0
    events: int = 0
    old_events: int = 0
    gaps: int = 0
    emitted_states: int = 0


def discover_inputs(paths: list[Path], include_inprogress: bool) -> list[Path]:
    """Expand files and directories into a stable replay order."""
    discovered: list[Path] = []
    for path in paths:
        if not path.exists():
            raise ReplayError(f"input does not exist: {path}")
        if path.is_file():
            discovered.append(path)
            continue

        discovered.extend(path.rglob("*.jsonl.gz"))
        discovered.extend(path.rglob("*.jsonl"))
        if include_inprogress:
            discovered.extend(path.rglob("*.jsonl.gz.inprogress"))

    unique: dict[Path, Path] = {}
    for path in discovered:
        unique[path.resolve()] = path
    files = sorted(unique.values(), key=lambda path: path.as_posix())
    if not files:
        raise ReplayError("no .jsonl.gz or .jsonl capture files found")
    return files


def is_inprogress(path: Path) -> bool:
    return path.name.endswith(".inprogress")


def open_capture(path: Path) -> TextIO:
    if path.name.endswith(".jsonl.gz") or path.name.endswith(".jsonl.gz.inprogress"):
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(mode="rt", encoding="utf-8")


def iter_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield JSON records, salvaging complete lines from an active gzip file."""
    try:
        with open_capture(path) as handle:
            line_number = 0
            while True:
                try:
                    line = handle.readline()
                except EOFError as error:
                    if is_inprogress(path):
                        LOG.warning("stopped at unfinished gzip tail in %s", path)
                        return
                    raise ReplayError(f"truncated gzip file: {path}") from error
                if not line:
                    return
                line_number += 1
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    if is_inprogress(path):
                        LOG.warning("ignored unfinished JSON line %s:%s", path, line_number)
                        return
                    raise ReplayError(f"invalid JSON at {path}:{line_number}: {error}") from error
                if not isinstance(record, dict):
                    raise ReplayError(f"record must be an object at {path}:{line_number}")
                yield line_number, record
    except (gzip.BadGzipFile, OSError) as error:
        raise ReplayError(f"cannot read {path}: {error}") from error


class ReplayClock:
    def __init__(self, speed: float, max_sleep: float | None) -> None:
        self.speed = speed
        self.max_sleep = max_sleep
        self.previous_ns: int | None = None

    def reset(self) -> None:
        # Snapshot records are synchronization boundaries. Buffered events can
        # have an earlier receive timestamp than the REST snapshot itself.
        self.previous_ns = None

    def wait(self, received_at_ns: int) -> None:
        if self.speed <= 0:
            return
        if self.previous_ns is not None and received_at_ns > self.previous_ns:
            delay = (received_at_ns - self.previous_ns) / 1_000_000_000 / self.speed
            if self.max_sleep is not None:
                delay = min(delay, self.max_sleep)
            if delay > 0:
                time.sleep(delay)
        self.previous_ns = received_at_ns


PARQUET_SCHEMA = pa.schema(
    [
        pa.field("type", pa.string(), nullable=False),
        pa.field("schema_version", pa.int32()),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("levels", pa.int32(), nullable=False),
        pa.field("every_ms", pa.float64()),
        pa.field("price_quantity_encoding", pa.string()),
        pa.field("received_at_ns", pa.int64()),
        pa.field("event_time_us", pa.int64()),
        pa.field("trigger", pa.string()),
        pa.field("last_update_id", pa.int64()),
        pa.field("best_bid", pa.string()),
        pa.field("best_ask", pa.string()),
        pa.field("mid_price", pa.string()),
        pa.field("spread", pa.string()),
        pa.field("depth_imbalance", pa.string()),
        pa.field(
            "bids",
            pa.list_(
                pa.struct(
                    [
                        pa.field("price", pa.string(), nullable=False),
                        pa.field("quantity", pa.string(), nullable=False),
                    ]
                )
            ),
        ),
        pa.field(
            "asks",
            pa.list_(
                pa.struct(
                    [
                        pa.field("price", pa.string(), nullable=False),
                        pa.field("quantity", pa.string(), nullable=False),
                    ]
                )
            ),
        ),
    ],
    metadata={b"compression": b"gzip"},
)


class ParquetOutput:
    """Write gzip Parquet, finalizing files atomically after a successful run."""

    ROW_GROUP_SIZE = 10_000

    def __init__(self, path: Path | None, overwrite: bool, enabled: bool = True) -> None:
        self.path = path
        self.overwrite = overwrite
        self.enabled = enabled
        self.temp_path: Path | None = None
        self.writer: pq.ParquetWriter | None = None
        self.records: list[dict[str, Any]] = []

    def __enter__(self) -> "ParquetOutput":
        if not self.enabled:
            return self
        if self.path is None or str(self.path) == "-":
            self.writer = pq.ParquetWriter(
                sys.stdout.buffer, PARQUET_SCHEMA, compression="gzip"
            )
            return self

        if self.path.exists() and not self.overwrite:
            raise ReplayError(f"output already exists (use --overwrite): {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_path = Path(f"{self.path}.inprogress")
        if self.temp_path.exists() and not self.overwrite:
            raise ReplayError(
                f"temporary output already exists (use --overwrite): {self.temp_path}"
            )
        self.writer = pq.ParquetWriter(
            self.temp_path, PARQUET_SCHEMA, compression="gzip"
        )
        return self

    def write(self, record: dict[str, Any]) -> None:
        if self.writer is None:
            raise RuntimeError("output is not open")
        normalized = dict(record)
        for side in ("bids", "asks"):
            if side in normalized:
                normalized[side] = [
                    {"price": price, "quantity": quantity}
                    for price, quantity in normalized[side]
                ]
        self.records.append(normalized)
        if len(self.records) >= self.ROW_GROUP_SIZE:
            self._flush()

    def _flush(self) -> None:
        if not self.records:
            return
        if self.writer is None:
            raise RuntimeError("output is not open")
        table = pa.Table.from_pylist(self.records, schema=PARQUET_SCHEMA)
        self.writer.write_table(table, row_group_size=self.ROW_GROUP_SIZE)
        self.records.clear()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.writer is None:
            return
        if exc_type is not None:
            self.writer.close()
            return
        self._flush()
        self.writer.close()
        if self.temp_path is not None:
            self.temp_path.replace(str(self.path))


def decimal_text(value: Decimal) -> str:
    return format(value, "f")


def top_levels(
    book: LocalOrderBook, levels: int
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    bids = heapq.nlargest(levels, book.bids.items(), key=lambda item: Decimal(item[0]))
    asks = heapq.nsmallest(levels, book.asks.items(), key=lambda item: Decimal(item[0]))
    return bids, asks


def make_state(
    book: LocalOrderBook,
    symbol: str,
    levels: int,
    received_at_ns: int,
    event_time_us: int | None,
    trigger: str,
) -> dict[str, Any]:
    bids, asks = top_levels(book, levels)
    best_bid = Decimal(bids[0][0]) if bids else None
    best_ask = Decimal(asks[0][0]) if asks else None
    spread = best_ask - best_bid if best_bid is not None and best_ask is not None else None
    mid_price = (
        (best_bid + best_ask) / Decimal(2)
        if best_bid is not None and best_ask is not None
        else None
    )
    bid_quantity = sum((Decimal(quantity) for _, quantity in bids), Decimal(0))
    ask_quantity = sum((Decimal(quantity) for _, quantity in asks), Decimal(0))
    total_quantity = bid_quantity + ask_quantity
    imbalance = (
        (bid_quantity - ask_quantity) / total_quantity if total_quantity != 0 else None
    )

    return {
        "type": "lob_state",
        "symbol": symbol,
        "received_at_ns": received_at_ns,
        "event_time_us": event_time_us,
        "trigger": trigger,
        "last_update_id": book.last_update_id,
        "levels": levels,
        "best_bid": decimal_text(best_bid) if best_bid is not None else None,
        "best_ask": decimal_text(best_ask) if best_ask is not None else None,
        "mid_price": decimal_text(mid_price) if mid_price is not None else None,
        "spread": decimal_text(spread) if spread is not None else None,
        "depth_imbalance": decimal_text(imbalance) if imbalance is not None else None,
        "bids": bids,
        "asks": asks,
    }


class ReplayEngine:
    def __init__(self, args: argparse.Namespace, output: ParquetOutput) -> None:
        self.args = args
        self.output = output
        self.stats = ReplayStats()
        self.book: LocalOrderBook | None = None
        self.symbol: str | None = args.symbol
        self.clock = ReplayClock(args.speed, args.max_sleep)
        self.last_emitted_ns: int | None = None
        self.output_metadata_written = False

    def process_files(self, files: list[Path]) -> ReplayStats:
        for path in files:
            if self.args.max_events is not None and self.stats.events >= self.args.max_events:
                break
            self.stats.files += 1
            LOG.info("replaying %s", path)
            for line_number, record in iter_records(path):
                self.stats.records += 1
                try:
                    should_stop = self.process_record(record)
                except (KeyError, TypeError, ValueError, ArithmeticError) as error:
                    raise ReplayError(f"invalid record at {path}:{line_number}: {error}") from error
                except SequenceGap as error:
                    self.stats.gaps += 1
                    if self.args.on_gap == "skip-until-snapshot":
                        LOG.warning("%s:%s: %s; waiting for next snapshot", path, line_number, error)
                        self.book = None
                        self.clock.reset()
                        self.last_emitted_ns = None
                        continue
                    raise ReplayError(f"sequence gap at {path}:{line_number}: {error}") from error
                if should_stop:
                    return self.stats
        return self.stats

    def process_record(self, record: dict[str, Any]) -> bool:
        record_type = record.get("type")
        if record_type == "metadata":
            self._process_metadata(record)
            return False
        if record_type == "snapshot":
            self._process_snapshot(record)
            return False
        if record_type == "depth_update":
            return self._process_event(record)
        raise ReplayError(f"unknown record type: {record_type!r}")

    def _process_metadata(self, record: dict[str, Any]) -> None:
        record_symbol = str(record["symbol"]).upper()
        if self.symbol is None:
            self.symbol = record_symbol
        elif record_symbol != self.symbol:
            raise ReplayError(
                f"mixed symbols are not supported: expected {self.symbol}, got {record_symbol}"
            )
        if record.get("feed") != "diff_depth":
            raise ReplayError(f"unsupported feed: {record.get('feed')!r}")

    def _process_snapshot(self, record: dict[str, Any]) -> None:
        snapshot = {
            "lastUpdateId": int(record["last_update_id"]),
            "bids": record["bids"],
            "asks": record["asks"],
        }
        self.book = LocalOrderBook.from_snapshot(snapshot)
        self.stats.snapshots += 1
        self.clock.reset()
        self.last_emitted_ns = None
        received_at_ns = int(record["received_at_ns"])
        self._emit(received_at_ns, None, f"snapshot:{record.get('reason', 'unknown')}")

    def _process_event(self, record: dict[str, Any]) -> bool:
        if self.book is None:
            if self.args.on_gap == "skip-until-snapshot":
                return False
            raise ReplayError("depth update encountered before a usable snapshot")

        received_at_ns = int(record["received_at_ns"])
        event = record["data"]
        if not isinstance(event, dict):
            raise ReplayError("depth_update.data must be an object")
        self.clock.wait(received_at_ns)
        if not self.book.apply(event):
            self.stats.old_events += 1
            return False

        self.stats.events += 1
        interval_ns = int(self.args.every_ms * 1_000_000)
        should_emit = (
            interval_ns == 0
            or self.last_emitted_ns is None
            or received_at_ns < self.last_emitted_ns
            or received_at_ns - self.last_emitted_ns >= interval_ns
        )
        if should_emit:
            event_time = event.get("E")
            self._emit(
                received_at_ns,
                int(event_time) if event_time is not None else None,
                "depth_update",
            )

        return self.args.max_events is not None and self.stats.events >= self.args.max_events

    def _emit(self, received_at_ns: int, event_time_us: int | None, trigger: str) -> None:
        if self.args.validate_only:
            return
        if self.book is None:
            raise RuntimeError("cannot emit without an order book")
        symbol = self.symbol or self.args.symbol or "UNKNOWN"
        if not self.output_metadata_written:
            self.output.write(
                {
                    "type": "replay_metadata",
                    "schema_version": 1,
                    "symbol": symbol,
                    "levels": self.args.levels,
                    "every_ms": self.args.every_ms,
                    "price_quantity_encoding": "decimal_string",
                }
            )
            self.output_metadata_written = True
        self.output.write(
            make_state(
                self.book,
                symbol,
                self.args.levels,
                received_at_ns,
                event_time_us,
                trigger,
            )
        )
        self.last_emitted_ns = received_at_ns
        self.stats.emitted_states += 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild and replay Top-N LOB states from Binance gzip JSONL captures "
            "into gzip-compressed Parquet."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="capture file(s) or directories")
    parser.add_argument("-o", "--output", type=Path, help="Parquet path; default stdout")
    parser.add_argument("--symbol", type=str.upper, help="expected symbol; otherwise read metadata")
    parser.add_argument("--levels", type=int, default=20, help="levels per side to emit")
    parser.add_argument(
        "--every-ms",
        type=float,
        default=1000.0,
        help="minimum receive-time interval between states; 0 emits every update",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help="timing multiplier (1=realtime, 10=10x, 0=no sleeping)",
    )
    parser.add_argument(
        "--max-sleep", type=float, help="cap each replay sleep in seconds"
    )
    parser.add_argument(
        "--on-gap",
        choices=("error", "skip-until-snapshot"),
        default="error",
        help="behavior when an update sequence gap is found",
    )
    parser.add_argument(
        "--include-inprogress",
        action="store_true",
        help="include active .inprogress files when scanning directories",
    )
    parser.add_argument("--validate-only", action="store_true", help="rebuild without output")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output")
    parser.add_argument("--max-events", type=int, help="stop after this many applied updates")
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    args = parser.parse_args(argv)

    if not 1 <= args.levels <= 5000:
        parser.error("--levels must be between 1 and 5000")
    if args.every_ms < 0:
        parser.error("--every-ms cannot be negative")
    if args.speed < 0:
        parser.error("--speed cannot be negative")
    if args.max_sleep is not None and args.max_sleep < 0:
        parser.error("--max-sleep cannot be negative")
    if args.max_events is not None and args.max_events <= 0:
        parser.error("--max-events must be positive")
    if args.validate_only and args.output is not None:
        parser.error("--output cannot be used with --validate-only")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        files = discover_inputs(args.inputs, args.include_inprogress)
        with ParquetOutput(
            args.output, args.overwrite, enabled=not args.validate_only
        ) as output:
            stats = ReplayEngine(args, output).process_files(files)
    except BrokenPipeError:
        return 0
    except (ReplayError, OSError) as error:
        LOG.error("%s", error)
        return 1
    except KeyboardInterrupt:
        LOG.warning("replay interrupted")
        return 130

    LOG.info(
        "complete: files=%s records=%s snapshots=%s events=%s old=%s gaps=%s emitted=%s",
        stats.files,
        stats.records,
        stats.snapshots,
        stats.events,
        stats.old_events,
        stats.gaps,
        stats.emitted_states,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
