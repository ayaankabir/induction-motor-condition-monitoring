# External CSV import and signal-quality checks

This milestone adds a **local, strict CSV import boundary** and a generic,
read-only signal-quality adapter. It does not add an experimental dataset,
validate a source, diagnose a machine, or change any simulation result.

## Scientific status

A successfully imported file is represented as:

- `source_type = external_csv`
- `provenance_class = external_unvalidated`
- `validation_status = unvalidated_external_input`
- `experimental_validation = false`

These labels are intentionally conservative. “External” does not mean
“experimental,” and successful parsing does not establish calibration,
timestamp correctness, sensor quality, or provenance beyond the recorded source
name and SHA-256 digest. A file must not be described as a project dataset until
its actual provenance and evidence class are known. Repository datasets remain
labeled `simulated`, `literature-example`, or `experimental` as required by the
project rules.

## CSV contract

`imcm.data.load_signal_csv` accepts a strict wide-format UTF-8 CSV and a
`CsvImportSpec`. The importer:

1. requires an exact set of declared columns—no missing or unexpected columns;
2. requires a header and at least two data rows;
3. requires finite, strictly increasing time values;
4. derives the sample rate from the median adjacent time interval when no rate
   is declared;
5. preserves a declared rate separately, even if it disagrees with the time
   vector, so the quality report can expose the mismatch;
6. retains literal `NaN` and positive/negative infinity in signal channels for
   quality reporting; and
7. records source name, UTC import time, SHA-256, parser/schema versions,
   channel mapping, units, sample-count, and sampling metadata.

Blank rows, ragged rows, malformed numeric cells, duplicate/empty headers,
missing/unexpected columns, unsupported units, and non-finite/nonmonotonic time
raise a structured `CsvImportError` with a stable code and available row/column
location.

The loader reads a local file; it does not download, copy, or publish it.

## Canonical units

The importer performs **no unit conversion**. Channel units must match one of
these exact, case-sensitive spellings from `CANONICAL_UNITS`:

| Unit | Meaning |
| --- | --- |
| `A` | ampere |
| `V` | volt |
| `m/s^2` | acceleration |
| `rad/s` | angular velocity |
| `rpm` | revolutions per minute |
| `N*m` | torque |
| `s` | time-valued channel |
| `Hz` | frequency-valued channel |
| `dimensionless` | unitless quantity |

Time itself must be supplied in seconds (`time_unit="s"`). Values such as `mA`,
`ms`, `m/s²`, or `rad/s²` are rejected rather than silently converted.

## Example

```python
from imcm.data import CsvImportSpec, SignalChannelSpec, load_signal_csv
from imcm.validation import analyze_signal_quality

spec = CsvImportSpec(
    time_column="time_s",
    channels=(
        SignalChannelSpec(
            name="phase_a_current",
            column="current_a",
            unit="A",
            signal_type="current",
            clipping_bounds=(-12.0, 12.0),
        ),
        SignalChannelSpec(
            name="radial_acceleration",
            column="acceleration",
            unit="m/s^2",
            signal_type="vibration",
        ),
    ),
    # If supplied, the declared rate is preserved rather than overwritten.
    sample_rate_hz=10_000.0,
    sample_rate_rtol=1.0e-3,
)

imported = load_signal_csv(
    "/absolute/or/local/path/recording.csv",
    spec,
    source_name="operator-supplied recording",
)
quality = analyze_signal_quality(imported.trace)

metadata_only = imported.to_summary_dict()  # JSON-safe; no waveform samples
quality_metadata = quality.to_dict()        # JSON-safe; no waveform samples
```

`ImportedSignal` enforces consistency between its trace and external
provenance, including channel order and source columns, canonical units, sample
count, effective rate, rate source, and declared rate.

## Generic signal-quality adapter

`analyze_signal_quality` consumes any `SignalTrace`; it is not coupled to the CSV
loader or an induction-motor model. It reports:

- non-finite, repeated, decreasing, or nonmonotonic time values;
- insufficient sample count;
- observed interval spread and whether it is uniform within the configured
  relative tolerance;
- agreement between the trace rate and the median interval;
- per-channel finite/non-finite counts and descriptive finite-value statistics;
- constant channels; and
- samples that reach caller-declared clipping bounds.

A hit at a declared bound is an observation, not proof of sensor saturation. A
constant channel or irregular interval is a quality finding, not a machine
fault. The adapter sets `data_modified = false` and `analysis_scope =
"data_quality_only"` in every report.

## Explicit non-operations and limitations

The import/quality path does **not**:

- resample, interpolate, regularize timestamps, or overwrite a declared rate;
- filter, smooth, clip, fill, winsorize, remove, reorder, or otherwise repair data;
- convert units;
- infer missing source information;
- validate calibration, sensor placement, clock synchronization, or hardware;
- perform MCSA feature extraction or fault classification;
- provide diagnosis, severity, confidence, prognosis, or predictive-maintenance
  functionality; or
- provide live monitoring.

The quality report is an objective structural audit of supplied arrays. A clean
report means only that the listed checks found no issue; it does **not** prove
that the data are authentic, calibrated, representative, diagnostically useful,
or suitable for a particular analysis.