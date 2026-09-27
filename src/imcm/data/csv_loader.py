"""Strict, local CSV loading for externally recorded signal data."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any

import numpy as np

from imcm.data.provenance import ExternalSignalProvenance, ImportedSignal, file_sha256
from imcm.data.trace import CsvImportSpec, SignalChannel, SignalTrace


class CsvImportError(ValueError):
    """A structured, user-actionable CSV import error."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        path: str | None = None,
        row: int | None = None,
        column: str | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.path = path
        self.row = row
        self.column = column
        location: list[str] = []
        if path is not None:
            location.append(f"path={path!r}")
        if row is not None:
            location.append(f"row={row}")
        if column is not None:
            location.append(f"column={column!r}")
        suffix = f" ({', '.join(location)})" if location else ""
        super().__init__(f"{code}: {message}{suffix}")


@dataclass(frozen=True)
class _ParsedCsv:
    header: tuple[str, ...]
    time_s: np.ndarray
    channels: tuple[SignalChannel, ...]
    row_count: int
    derived_sample_rate_hz: float
    uniformity_rtol: float


def _fail(
    code: str,
    message: str,
    path: Path,
    *,
    row: int | None = None,
    column: str | None = None,
) -> CsvImportError:
    return CsvImportError(code, message, path=str(path), row=row, column=column)


def _parse_float(value: str, path: Path, row: int, column: str) -> float:
    text = value.strip()
    if not text:
        raise _fail("empty_value", "numeric cells must not be empty", path, row=row, column=column)
    try:
        return float(text)
    except ValueError as exc:
        raise _fail(
            "nonnumeric_value",
            f"expected a floating-point value, got {value!r}",
            path,
            row=row,
            column=column,
        ) from exc


def _read_header(reader: csv.reader, path: Path) -> tuple[str, ...]:
    try:
        raw_header = next(reader)
    except StopIteration as exc:
        raise _fail("empty_file", "CSV file is empty", path) from exc
    header = tuple(cell.strip() for cell in raw_header)
    if not header or any(not cell for cell in header):
        raise _fail("invalid_header", "CSV header contains an empty column name", path, row=1)
    if len(set(header)) != len(header):
        raise _fail("duplicate_header", "CSV header contains duplicate column names", path, row=1)
    return header


def _parse_csv(path: Path, spec: CsvImportSpec) -> _ParsedCsv:
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise _fail("file_read_error", str(exc), path) from exc

    with handle:
        try:
            reader = csv.reader(handle, delimiter=spec.delimiter, strict=True)
            header = _read_header(reader, path)
            requested = (spec.time_column, *(channel.column for channel in spec.channels))
            missing = [name for name in requested if name not in header]
            if missing:
                raise _fail(
                    "missing_column",
                    f"required column(s) not found: {', '.join(missing)}",
                    path,
                    row=1,
                )
            unexpected = [name for name in header if name not in requested]
            if unexpected:
                raise _fail(
                    "unexpected_column",
                    "CSV contains columns not declared in the import specification: "
                    + ", ".join(unexpected),
                    path,
                    row=1,
                )
            time_index = header.index(spec.time_column)
            channel_indices = [header.index(channel.column) for channel in spec.channels]
            time_values: list[float] = []
            channel_values: list[list[float]] = [[] for _ in spec.channels]
            for row_number, row in enumerate(reader, start=2):
                if not row or all(not cell.strip() for cell in row):
                    raise _fail("blank_row", "blank CSV rows are not allowed", path, row=row_number)
                if len(row) != len(header):
                    raise _fail(
                        "ragged_row",
                        f"row has {len(row)} fields; expected {len(header)}",
                        path,
                        row=row_number,
                    )
                time_value = _parse_float(row[time_index], path, row_number, spec.time_column)
                if not math.isfinite(time_value):
                    raise _fail(
                        "nonfinite_time",
                        "time values must be finite",
                        path,
                        row=row_number,
                        column=spec.time_column,
                    )
                time_values.append(time_value)
                for channel, channel_index, values in zip(
                    spec.channels, channel_indices, channel_values
                ):
                    values.append(
                        _parse_float(
                            row[channel_index],
                            path,
                            row_number,
                            channel.column,
                        )
                    )
        except csv.Error as exc:
            raise _fail("csv_syntax_error", str(exc), path) from exc
        except UnicodeError as exc:
            raise _fail("encoding_error", "CSV must be UTF-8 text", path) from exc

    if len(time_values) < 2:
        raise _fail(
            "insufficient_rows",
            "CSV must contain a header and at least two data rows",
            path,
        )
    time_array = np.asarray(time_values, dtype=float)
    intervals = np.diff(time_array)
    if np.any(intervals <= 0.0):
        first_bad = int(np.flatnonzero(intervals <= 0.0)[0]) + 2
        raise _fail(
            "nonmonotonic_time",
            "time values must be strictly increasing",
            path,
            row=first_bad,
            column=spec.time_column,
        )
    derived_rate = 1.0 / float(np.median(intervals))
    if not math.isfinite(derived_rate) or derived_rate <= 0.0:
        raise _fail("invalid_sample_rate", "could not derive a positive sample rate", path)
    median_interval = float(np.median(intervals))
    relative_spread = float(np.max(np.abs(intervals - median_interval)) / median_interval)
    channel_objects = tuple(
        SignalChannel(
            name=channel.name,
            values=np.asarray(values, dtype=float),
            unit=channel.unit,
            source_column=channel.column,
            signal_type=channel.signal_type,
            clipping_bounds=channel.clipping_bounds,
        )
        for channel, values in zip(spec.channels, channel_values)
    )
    return _ParsedCsv(
        header=header,
        time_s=time_array,
        channels=channel_objects,
        row_count=len(time_values),
        derived_sample_rate_hz=derived_rate,
        uniformity_rtol=relative_spread,
    )


def load_signal_csv(
    path: str | Path,
    spec: CsvImportSpec,
    *,
    source_name: str | None = None,
    imported_at_utc: str | None = None,
) -> ImportedSignal:
    """Load one strict wide-format CSV into an unvalidated external trace.

    The file is read twice only at the byte level for its SHA-256 digest; the
    CSV itself is parsed once.  Literal ``NaN`` and ``Inf`` signal values are
    retained for the quality report.  Nonfinite time, malformed rows, missing
    columns, and unsupported units raise :class:`CsvImportError`.
    """
    if not isinstance(spec, CsvImportSpec):
        raise TypeError("spec must be a CsvImportSpec")
    source_path = Path(path)
    if not source_path.is_file():
        raise _fail("file_not_found", "CSV path does not exist or is not a file", source_path)
    parsed = _parse_csv(source_path, spec)
    if spec.sample_rate_hz is None:
        effective_rate = parsed.derived_sample_rate_hz
        rate_source = "derived_from_time"
    else:
        effective_rate = spec.sample_rate_hz
        rate_source = "declared"
    trace = SignalTrace(
        time_s=parsed.time_s,
        channels=parsed.channels,
        sample_rate_hz=effective_rate,
        sample_rate_source=rate_source,
        declared_sample_rate_hz=spec.sample_rate_hz,
        time_unit=spec.time_unit,
        metadata={
            "source_path": str(source_path),
            "declared_columns": parsed.header,
            "derived_sample_rate_hz": parsed.derived_sample_rate_hz,
            "sample_rate_relative_spread": parsed.uniformity_rtol,
            "sample_rate_rtol": spec.sample_rate_rtol,
        },
    )
    timestamp = imported_at_utc or datetime.now(timezone.utc).isoformat()
    provenance = ExternalSignalProvenance(
        source_name=source_name or source_path.name,
        source_sha256=file_sha256(source_path),
        imported_at_utc=timestamp,
        time_column=spec.time_column,
        channel_mapping={channel.name: channel.column for channel in spec.channels},
        units={channel.name: channel.unit for channel in spec.channels},
        sample_rate_hz=effective_rate,
        sample_rate_source=rate_source,
        declared_sample_rate_hz=spec.sample_rate_hz,
        row_count=parsed.row_count,
    )
    return ImportedSignal(trace=trace, provenance=provenance)


__all__ = ["CsvImportError", "load_signal_csv"]