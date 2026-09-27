"""External signal import and source-agnostic trace data structures."""

from imcm.data.csv_loader import CsvImportError, load_signal_csv
from imcm.data.provenance import (
    EXTERNAL_PARSER_VERSION,
    EXTERNAL_PROVENANCE_CLASS,
    EXTERNAL_PROVENANCE_SCHEMA_VERSION,
    EXTERNAL_SOURCE_TYPE,
    EXTERNAL_VALIDATION_STATUS,
    ExternalSignalProvenance,
    ImportedSignal,
    file_sha256,
    utc_now_iso,
)
from imcm.data.trace import (
    CANONICAL_SIGNAL_TYPES,
    CANONICAL_UNITS,
    CsvImportSpec,
    SampleRateSource,
    SignalChannel,
    SignalChannelSpec,
    SignalTrace,
)

__all__ = [
    "CANONICAL_SIGNAL_TYPES",
    "CANONICAL_UNITS",
    "CsvImportError",
    "CsvImportSpec",
    "EXTERNAL_PARSER_VERSION",
    "EXTERNAL_PROVENANCE_CLASS",
    "EXTERNAL_PROVENANCE_SCHEMA_VERSION",
    "EXTERNAL_SOURCE_TYPE",
    "EXTERNAL_VALIDATION_STATUS",
    "ExternalSignalProvenance",
    "ImportedSignal",
    "SampleRateSource",
    "SignalChannel",
    "SignalChannelSpec",
    "SignalTrace",
    "file_sha256",
    "load_signal_csv",
    "utc_now_iso",
]