"""Read-only assembly of a deterministic engineering-report data model.

This module consumes the existing result-summary, provenance, run-config,
explanation, and triage structures.  It does not run a simulation, regenerate
an overview, read raw trace archives, or calculate a new scientific metric.
The renderers in :mod:`imcm.reporting.report_html` and
:mod:`imcm.reporting.report_pdf` consume the model returned here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Literal, Mapping, Sequence

from imcm.reporting.explanation import generate_explanation
from imcm.reporting.provenance import (
    ProvenanceError,
    ProvenanceRecord,
    read_manifest,
)
from imcm.reporting.run_config import (
    RunConfig,
    RunConfigError,
    read_run_config,
    run_config_hash,
)
from imcm.reporting.triage import evaluate_triage


REPORT_SCHEMA_VERSION = "engineering_report_v1"
REPORT_MANIFEST_SCHEMA_VERSION = "engineering_report_manifest_v1"
REPORT_GENERATOR_VERSION = "engineering_report_builder_v1"

SourceKind = Literal["simulated_study", "external_unvalidated_signal", "unavailable"]
SourceState = Literal["complete", "partial", "invalid", "unavailable"]
ReportFormat = Literal["json", "html", "pdf", "manifest"]

PROJECT_TITLE = (
    "Induction Motor Condition Monitoring Using a Reduced-Order Model and "
    "Electrical Signal Processing"
)

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})
_PATH_SUFFIXES = _IMAGE_SUFFIXES | frozenset({".json", ".md", ".npz"})

# These are scope boundaries, not scientific or diagnostic assertions.
REPORT_SCOPE_LIMITATIONS: tuple[str, ...] = (
    "This report is a read-only presentation of existing stored/project data; it performs no new simulation, feature extraction, or scientific calculation.",
    "The current repository contains simulated reduced-order studies with literature-example parameters and no experimental validation.",
    "No ML, diagnosis, severity, confidence, health index, prediction, ranking, or predictive-maintenance functionality is added by this report.",
    "A proxy or synthetic condition remains a proxy or synthetic condition; it is not a bar-resolved, calibrated, or real-machine diagnosis.",
)


class EngineeringReportError(ValueError):
    """Base error for report assembly and output failures."""


class ReportInputError(EngineeringReportError):
    """Raised when the required report input cannot be read or interpreted."""


@dataclass(frozen=True)
class ReportFile:
    """A repository-relative source/artifact reference used by the report."""

    path: str
    sha256: str | None
    exists: bool
    content_kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "exists": self.exists,
            "content_kind": self.content_kind,
        }


@dataclass(frozen=True)
class ReportSourceStatus:
    """Completeness state for one source record.

    ``complete`` means the summary, provenance manifest, run config, and all
    referenced local figures were present and valid.  A missing optional
    artifact produces ``partial``; malformed JSON/schema produces
    ``invalid``; a missing required summary produces ``unavailable``.
    """

    source_kind: SourceKind
    state: SourceState
    summary_path: str
    provenance_path: str | None
    run_config_path: str | None
    required_files_present: tuple[str, ...]
    missing_files: tuple[str, ...]
    invalid_files: tuple[str, ...]
    figure_paths: tuple[str, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind,
            "state": self.state,
            "summary_path": self.summary_path,
            "provenance_path": self.provenance_path,
            "run_config_path": self.run_config_path,
            "required_files_present": list(self.required_files_present),
            "missing_files": list(self.missing_files),
            "invalid_files": list(self.invalid_files),
            "figure_paths": list(self.figure_paths),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ReportProvenance:
    """Stored provenance and reproducibility data for one source."""

    manifest: Mapping[str, Any] | None
    run_config: Mapping[str, Any] | None
    parameter_hash: str | None
    parameter_source: str | None
    solver_settings: Mapping[str, Any]
    solver_settings_source: str | None
    initial_condition: Mapping[str, Any]
    git_commit: str | None
    timestamp: str | None
    run_id: str | None
    run_config_hash: str | None
    files: tuple[ReportFile, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": _jsonable(self.manifest),
            "run_config": _jsonable(self.run_config),
            "parameter_hash": self.parameter_hash,
            "parameter_source": self.parameter_source,
            "solver_settings": _jsonable(self.solver_settings),
            "solver_settings_source": self.solver_settings_source,
            "initial_condition": _jsonable(self.initial_condition),
            "git_commit": self.git_commit,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "run_config_hash": self.run_config_hash,
            "files": [item.to_dict() for item in self.files],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ReportCondition:
    """One existing condition summary and its existing interpretation records."""

    condition_id: str
    name: str
    classification: str
    mechanism: str
    affected_signal: str
    expected_signature: str
    summary_source: str
    summary: Mapping[str, Any]
    measured: Mapping[str, Any]
    explanation: Mapping[str, Any] | None
    triage: Mapping[str, Any] | None
    limitations: tuple[str, ...]
    provenance: ReportProvenance
    source_status: ReportSourceStatus
    figures: tuple[ReportFile, ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "name": self.name,
            "classification": self.classification,
            "mechanism": self.mechanism,
            "affected_signal": self.affected_signal,
            "expected_signature": self.expected_signature,
            "summary_source": self.summary_source,
            "summary": _jsonable(self.summary),
            "measured": _jsonable(self.measured),
            "explanation": _jsonable(self.explanation),
            "triage": _jsonable(self.triage),
            "limitations": list(self.limitations),
            "provenance": self.provenance.to_dict(),
            "source_status": self.source_status.to_dict(),
            "figures": [item.to_dict() for item in self.figures],
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class ExternalReportRecord:
    """Metadata-only report view of an explicitly supplied external record."""

    source_id: str
    source_kind: Literal["external_unvalidated_signal"]
    state: SourceState
    summary: Mapping[str, Any]
    quality_report: Mapping[str, Any] | None
    source_digest: str
    source_name: str | None
    source_sha256: str | None
    imported_at_utc: str | None
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_kind": self.source_kind,
            "state": self.state,
            "summary": _jsonable(self.summary),
            "quality_report": _jsonable(self.quality_report),
            "source_digest": self.source_digest,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "imported_at_utc": self.imported_at_utc,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class EngineeringReport:
    """Canonical, renderer-independent engineering report data."""

    schema_version: str
    report_id: str
    project: Mapping[str, Any]
    conditions: tuple[ReportCondition, ...]
    external_sources: tuple[ExternalReportRecord, ...]
    input_files: tuple[ReportFile, ...]
    overview_artifacts: tuple[ReportFile, ...]
    stored_consistency_checks: tuple[Mapping[str, Any], ...]
    stored_all_checks_pass: bool | None
    limitations: tuple[str, ...]
    generated_at_utc: str | None
    report_git_commit: str | None
    root: Path = field(repr=False, compare=False)

    def source_data_status(self) -> dict[str, str]:
        simulated_states = [item.source_status.state for item in self.conditions]
        if not simulated_states:
            simulated = "unavailable"
        elif all(state == "complete" for state in simulated_states):
            simulated = "complete"
        else:
            simulated = "partial"
        if not self.external_sources:
            external = "none_supplied"
        elif all(item.state == "complete" for item in self.external_sources):
            external = "complete"
        else:
            external = "partial"
        return {"simulated": simulated, "external": external}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "generated_at_utc": self.generated_at_utc,
            "report_git_commit": self.report_git_commit,
            "project": _jsonable(self.project),
            "source_data_status": self.source_data_status(),
            "conditions": [item.to_dict() for item in self.conditions],
            "external_sources": [item.to_dict() for item in self.external_sources],
            "input_files": [item.to_dict() for item in self.input_files],
            "overview_artifacts": [item.to_dict() for item in self.overview_artifacts],
            "stored_consistency_checks": [_jsonable(item) for item in self.stored_consistency_checks],
            "stored_all_checks_pass": self.stored_all_checks_pass,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class ReportArtifacts:
    """Paths and hashes returned after writing report artifacts."""

    paths: Mapping[str, Path]
    hashes: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "paths": {key: path.name for key, path in self.paths.items()},
            "sha256": dict(self.hashes),
        }


# ---------------------------------------------------------------------------
# Small JSON/path helpers
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    """Convert report values to deterministic JSON-compatible primitives."""

    if isinstance(value, ReportFile):
        return value.to_dict()
    if isinstance(value, ReportSourceStatus):
        return value.to_dict()
    if isinstance(value, ReportProvenance):
        return value.to_dict()
    if isinstance(value, ReportCondition):
        return value.to_dict()
    if isinstance(value, ExternalReportRecord):
        return value.to_dict()
    if isinstance(value, EngineeringReport):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise EngineeringReportError(
            f"report data are not canonical JSON: {type(exc).__name__}"
        ) from exc


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_payload(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _resolve_reference(
    reference: str, root: Path, results_dir: Path
) -> Path | None:
    """Resolve a repository-local reference without opening it."""

    if not isinstance(reference, str) or not reference.strip():
        return None
    text = reference.strip()
    if text.startswith(("http://", "https://")):
        return None
    raw = Path(text)
    candidates: list[Path] = []
    if raw.is_absolute():
        try:
            candidates.append(raw)
        except (OSError, ValueError):
            return None
        # Older overview artifacts may contain a path from another checkout.
        # Relocate only paths that contain a results directory; never open an
        # arbitrary external path.
        if "results" in raw.parts:
            index = raw.parts.index("results")
            candidates.append(root.joinpath(*raw.parts[index:]))
    else:
        if raw.parts and raw.parts[0] == "results":
            candidates.append(root / raw)
        else:
            candidates.append(results_dir / raw)
            candidates.append(root / raw)

    root_resolved = root.resolve()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError):
            continue
        try:
            if not resolved.is_relative_to(root_resolved):
                continue
        except AttributeError:  # pragma: no cover - Python 3.11+ guarantee
            try:
                resolved.relative_to(root_resolved)
            except ValueError:
                continue
        return resolved
    return None


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _looks_like_path(value: str) -> bool:
    if not isinstance(value, str) or not value:
        return False
    lowered = value.lower()
    if lowered.startswith(("http://", "https://")):
        return False
    suffix = Path(value).suffix.lower()
    if suffix not in _PATH_SUFFIXES:
        return False
    return value.startswith(("/", "./", "../")) or "results" in Path(value).parts


def _sanitize_paths(
    value: Any, root: Path, results_dir: Path
) -> Any:
    """Normalize local path strings for portable report payloads."""

    if isinstance(value, str):
        if _looks_like_path(value):
            resolved = _resolve_reference(value, root, results_dir)
            if resolved is not None:
                return _relative_path(resolved, root)
        return value
    if isinstance(value, Mapping):
        return {str(key): _sanitize_paths(item, root, results_dir) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_paths(item, root, results_dir) for item in value]
    return value


def _normalise_digest_payload(
    value: Any, root: Path, results_dir: Path
) -> Any:
    return _sanitize_paths(value, root, results_dir)


def _read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, "file_not_found"
    except OSError:
        return None, "file_read_error"
    except json.JSONDecodeError:
        return None, "invalid_json"


def _file_digest(
    path: Path, root: Path, results_dir: Path, *, json_content: bool
) -> str | None:
    if not path.is_file():
        return None
    try:
        if json_content:
            data, error = _read_json(path)
            if error is None and data is not None:
                return _sha256_payload(_normalise_digest_payload(data, root, results_dir))
        return _sha256_bytes(path.read_bytes())
    except OSError:
        return None


def _file_record(
    path: Path | None,
    root: Path,
    results_dir: Path,
    *,
    kind: str,
    json_content: bool = False,
) -> ReportFile | None:
    if path is None:
        return None
    relative = _relative_path(path, root)
    return ReportFile(
        path=relative,
        sha256=_file_digest(path, root, results_dir, json_content=json_content),
        exists=path.is_file(),
        content_kind=kind,
    )


def _iter_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item)


def _collect_image_files(
    value: Any,
    root: Path,
    results_dir: Path,
) -> tuple[tuple[ReportFile, ...], tuple[str, ...], tuple[str, ...]]:
    files: dict[str, ReportFile] = {}
    missing: list[str] = []
    invalid: list[str] = []
    for reference in _iter_strings(value):
        suffix = Path(reference).suffix.lower()
        if suffix not in _IMAGE_SUFFIXES:
            continue
        resolved = _resolve_reference(reference, root, results_dir)
        if resolved is None:
            invalid.append(reference)
            continue
        relative = _relative_path(resolved, root)
        if not resolved.is_file():
            missing.append(relative)
            continue
        record = _file_record(resolved, root, results_dir, kind="image")
        if record is not None:
            files[relative] = record
    return (
        tuple(files[key] for key in sorted(files)),
        tuple(dict.fromkeys(missing)),
        tuple(dict.fromkeys(invalid)),
    )


def _dedupe(values: Sequence[Any]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        text = value.strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return tuple(result)


def _summary_solver_settings(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Project already-stored summary settings; do not derive new values."""

    names = (
        "solver",
        "max_step_s",
        "rtol",
        "atol",
        "t_end_s",
        "output_dt_s",
    )
    return {name: summary[name] for name in names if name in summary}


def _summary_initial_condition(summary: Mapping[str, Any]) -> dict[str, Any]:
    value = summary.get("initial_condition")
    if isinstance(value, Mapping):
        return dict(value)
    if value is not None:
        return {"description": str(value)}
    return {}


def _source_kind_for_summary(summary: Mapping[str, Any]) -> str | None:
    value = summary.get("provenance")
    return value if isinstance(value, str) else None


def _cross_link_notes(
    manifest: Mapping[str, Any] | None,
    run_config: Mapping[str, Any] | None,
    manifest_path: str | None,
    run_config_path: str | None,
) -> tuple[str, ...]:
    notes: list[str] = []
    if manifest is not None and run_config is not None:
        link = manifest.get("run_config")
        if isinstance(link, Mapping):
            linked_path = link.get("path")
            linked_id = link.get("run_id")
            linked_hash = link.get("hash_sha256")
            if linked_path is not None and linked_path != run_config_path:
                notes.append("manifest run_config path differs from loaded run config")
            if linked_id is not None and linked_id != run_config.get("run_id"):
                notes.append("manifest run_id differs from loaded run config")
            if linked_hash is not None:
                try:
                    actual_hash = run_config_hash(run_config)
                except (RunConfigError, TypeError, ValueError):
                    actual_hash = None
                if actual_hash is not None and linked_hash != actual_hash:
                    notes.append("manifest run-config hash differs from loaded run config")
    config_link = run_config.get("provenance_manifest") if run_config else None
    if isinstance(config_link, Mapping):
        linked_path = config_link.get("path")
        if linked_path is not None and linked_path != manifest_path:
            notes.append("run-config manifest path differs from loaded manifest")
    return _dedupe(notes)


def _load_manifest_and_config(
    manifest_path: Path | None,
    run_config_path: Path | None,
    root: Path,
    results_dir: Path,
) -> tuple[
    Mapping[str, Any] | None,
    Mapping[str, Any] | None,
    ReportProvenance | None,
    tuple[ReportFile, ...],
    tuple[str, ...],
]:
    manifest: Mapping[str, Any] | None = None
    run_config: Mapping[str, Any] | None = None
    files: list[ReportFile] = []
    notes: list[str] = []

    manifest_record = _file_record(
        manifest_path, root, results_dir, kind="json", json_content=True
    )
    if manifest_record is not None:
        files.append(manifest_record)
    config_record = _file_record(
        run_config_path, root, results_dir, kind="json", json_content=True
    )
    if config_record is not None:
        files.append(config_record)

    if manifest_path is not None and manifest_path.is_file():
        try:
            manifest = read_manifest(manifest_path).to_dict()
        except (OSError, ValueError, TypeError, ProvenanceError) as exc:
            notes.append(f"provenance manifest invalid: {type(exc).__name__}")
    if run_config_path is not None and run_config_path.is_file():
        try:
            run_config = read_run_config(run_config_path).to_dict()
        except (OSError, ValueError, TypeError, RunConfigError) as exc:
            notes.append(f"run config invalid: {type(exc).__name__}")

    manifest_rel = _relative_path(manifest_path, root) if manifest_path else None
    config_rel = _relative_path(run_config_path, root) if run_config_path else None
    notes.extend(_cross_link_notes(manifest, run_config, manifest_rel, config_rel))

    if manifest is not None:
        parameter_hash = manifest.get("parameter_hash")
        parameter_source = manifest.get("parameter_source")
        solver_settings = manifest.get("solver_settings")
        initial_condition = manifest.get("initial_condition")
        git_commit = manifest.get("git_commit")
        timestamp = manifest.get("timestamp")
        solver_source = "provenance_manifest"
    else:
        parameter_hash = None
        parameter_source = None
        solver_settings = {}
        initial_condition = {}
        git_commit = None
        timestamp = None
        solver_source = None

    if run_config is not None:
        run_id = run_config.get("run_id")
        config_hash = run_config_hash(run_config)
        if not parameter_hash:
            parameter_hash = None
        if not solver_settings:
            config_solver = run_config.get("solver_settings")
            if isinstance(config_solver, Mapping):
                solver_settings = dict(config_solver)
                solver_source = "run_config"
        if not initial_condition:
            config_initial = run_config.get("initial_condition")
            if isinstance(config_initial, Mapping):
                initial_condition = dict(config_initial)
    else:
        run_id = None
        config_hash = None

    if not isinstance(parameter_hash, str):
        parameter_hash = None
    if not isinstance(parameter_source, str):
        parameter_source = None
    if not isinstance(git_commit, str):
        git_commit = None
    if not isinstance(timestamp, str):
        timestamp = None
    if not isinstance(solver_settings, Mapping):
        solver_settings = {}
    if not isinstance(initial_condition, Mapping):
        initial_condition = {}

    provenance = ReportProvenance(
        manifest=_sanitize_paths(manifest, root, results_dir) if manifest is not None else None,
        run_config=_sanitize_paths(run_config, root, results_dir) if run_config is not None else None,
        parameter_hash=parameter_hash,
        parameter_source=parameter_source,
        solver_settings=dict(solver_settings),
        solver_settings_source=solver_source,
        initial_condition=dict(initial_condition),
        git_commit=git_commit,
        timestamp=timestamp,
        run_id=run_id,
        run_config_hash=config_hash,
        files=tuple(files),
        notes=_dedupe(notes),
    )
    return manifest, run_config, provenance, tuple(files), provenance.notes


def _source_paths(
    summary_path: Path, root: Path, results_dir: Path
) -> tuple[Path | None, Path | None]:
    name = summary_path.name
    if name.endswith("_summary.json"):
        stem = name[: -len("_summary.json")]
        manifest = results_dir / f"{stem}_provenance.json"
        config = results_dir / f"{stem}_run_config.json"
    else:
        manifest = None
        config = None
    # The paths are already inside results_dir; retain this helper separate so
    # the naming convention is explicit and testable.
    return manifest, config


def _condition_record(
    raw: Mapping[str, Any],
    overview: Mapping[str, Any],
    root: Path,
    results_dir: Path,
) -> ReportCondition:
    condition_id = str(
        raw.get("key")
        or raw.get("condition_id")
        or raw.get("id")
        or "unknown"
    )
    reference = raw.get("summary_source")
    if not isinstance(reference, str) or not reference.strip():
        reference = "healthy_startup_summary.json" if condition_id == "healthy" else None
    summary_path = (
        _resolve_reference(reference, root, results_dir)
        if isinstance(reference, str)
        else None
    )
    expected_summary_path = (
        summary_path
        if summary_path is not None
        else results_dir / f"{condition_id}_summary.json"
    )
    summary, summary_error = (
        _read_json(expected_summary_path) if expected_summary_path.is_file() else (None, "file_not_found")
    )
    if not isinstance(summary, Mapping):
        summary = {}
    summary_record = _file_record(
        expected_summary_path, root, results_dir, kind="json", json_content=True
    )
    if summary_record is not None:
        # The source is included separately in input_files, not in figures.
        pass

    manifest_path, run_config_path = _source_paths(
        expected_summary_path, root, results_dir
    )
    manifest, run_config, provenance, provenance_files, provenance_notes = _load_manifest_and_config(
        manifest_path, run_config_path, root, results_dir
    )

    # Fill only presentation fields from the stored summary/overview; do not
    # calculate any new metric.
    if manifest is None:
        parameter_source = summary.get("parameter_provenance") or summary.get("parameter_source")
        if isinstance(parameter_source, str):
            provenance = replace(
                provenance,
                parameter_source=parameter_source,
                solver_settings=_summary_solver_settings(summary),
                solver_settings_source="summary" if _summary_solver_settings(summary) else None,
                initial_condition=_summary_initial_condition(summary),
            )
    if manifest is None and run_config is not None:
        # A valid run config is still useful metadata, but the manifest remains
        # absent and the source state is partial.
        parameter_source = run_config.get("parameter_source")
        if isinstance(parameter_source, str) and provenance.parameter_source is None:
            provenance = replace(provenance, parameter_source=parameter_source)

    figures, figure_missing, figure_invalid = _collect_image_files(
        (raw, summary, manifest, run_config), root, results_dir
    )
    required: list[str] = []
    missing: list[str] = []
    invalid: list[str] = []
    for record in (summary_record, *provenance_files, *figures):
        if record is None:
            continue
        required.append(record.path)
        if record.exists:
            pass
        else:
            missing.append(record.path)
    if summary_error is not None:
        if summary_error == "file_not_found":
            missing.append(_relative_path(expected_summary_path, root))
        else:
            invalid.append(_relative_path(expected_summary_path, root))
    if manifest_path is not None and not manifest_path.is_file():
        missing.append(_relative_path(manifest_path, root))
    if run_config_path is not None and not run_config_path.is_file():
        missing.append(_relative_path(run_config_path, root))
    missing.extend(figure_missing)
    invalid.extend(figure_invalid)
    missing = list(dict.fromkeys(missing))
    invalid = list(dict.fromkeys(invalid))
    present = [path for path in required if path not in missing and path not in invalid]
    notes = list(provenance_notes)
    if summary_error is not None:
        notes.append(f"summary unavailable or invalid: {summary_error}")
    if figure_missing:
        notes.append("one or more referenced figures are missing")
    if figure_invalid:
        notes.append("one or more referenced figures are outside the repository")
    if provenance.parameter_source is None:
        notes.append("parameter source is not recorded in the available summary/manifest")
    if not provenance.git_commit:
        notes.append("source git commit is not recorded")
    if not provenance.timestamp:
        notes.append("source timestamp is not recorded")
    if not provenance.run_config_hash:
        notes.append("run-config hash is not recorded")

    if invalid:
        state: SourceState = "invalid"
    elif summary_error == "file_not_found" or not summary_record or not summary_record.exists:
        state = "unavailable"
    elif missing:
        state = "partial"
    else:
        state = "complete"

    source_status = ReportSourceStatus(
        source_kind="simulated_study",
        state=state,
        summary_path=_relative_path(expected_summary_path, root),
        provenance_path=(
            _relative_path(manifest_path, root) if manifest_path is not None else None
        ),
        run_config_path=(
            _relative_path(run_config_path, root) if run_config_path is not None else None
        ),
        required_files_present=_dedupe(present),
        missing_files=_dedupe(missing),
        invalid_files=_dedupe(invalid),
        figure_paths=tuple(item.path for item in figures),
        notes=_dedupe(notes),
    )

    explanation: Mapping[str, Any] | None = None
    triage: Mapping[str, Any] | None = None
    errors: list[str] = []
    if summary_record is not None and summary_record.exists and summary:
        try:
            explanation = generate_explanation(condition_id, dict(summary)).to_dict()
        except Exception as exc:  # preserve report assembly even for a bad optional explanation
            errors.append(f"explanation unavailable: {type(exc).__name__}")
        try:
            triage = evaluate_triage(dict(summary), healthy_baseline=dict(overview)).to_dict()
        except Exception as exc:  # triage semantics remain owned by triage.py
            errors.append(f"triage unavailable: {type(exc).__name__}")
    elif summary_record is None or not summary_record.exists:
        errors.append("summary is not available for explanation/triage")

    return ReportCondition(
        condition_id=condition_id,
        name=str(raw.get("name") or condition_id),
        classification=str(raw.get("classification") or "Not reported"),
        mechanism=str(raw.get("mechanism") or "Not reported"),
        affected_signal=str(raw.get("affected_signal") or "Not reported"),
        expected_signature=str(raw.get("expected_signature") or "Not reported"),
        summary_source=_relative_path(expected_summary_path, root),
        summary=_sanitize_paths(summary, root, results_dir),
        measured=_sanitize_paths(raw.get("measured") if isinstance(raw.get("measured"), Mapping) else {}, root, results_dir),
        explanation=explanation,
        triage=triage,
        limitations=_dedupe(raw.get("limitations", []) if isinstance(raw.get("limitations"), list) else []),
        provenance=provenance,
        source_status=source_status,
        figures=figures,
        errors=_dedupe(errors),
    )


def _external_metadata_only(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
    """Extract only the existing metadata projection, never waveform samples."""

    if "summary" in record and isinstance(record.get("summary"), Mapping):
        summary = dict(record["summary"])
    elif "imported_summary" in record and isinstance(record.get("imported_summary"), Mapping):
        summary = dict(record["imported_summary"])
    else:
        summary = {
            key: value
            for key, value in record.items()
            if key not in {"quality_report", "quality", "source_digest"}
        }
    quality_value = record.get("quality_report")
    if not isinstance(quality_value, Mapping):
        quality_value = record.get("quality")
    quality = dict(quality_value) if isinstance(quality_value, Mapping) else None

    provenance = summary.get("provenance")
    provenance = dict(provenance) if isinstance(provenance, Mapping) else {}
    trace = summary.get("trace")
    trace_metadata: dict[str, Any] = {}
    if isinstance(trace, Mapping):
        for key in (
            "sample_count",
            "duration_s",
            "sample_rate_hz",
            "sample_rate_source",
            "declared_sample_rate_hz",
            "time_unit",
        ):
            if key in trace:
                trace_metadata[key] = trace[key]
        channels = trace.get("channels")
        if isinstance(channels, list):
            trace_metadata["channels"] = [
                {
                    key: channel[key]
                    for key in (
                        "name",
                        "unit",
                        "signal_type",
                        "source_column",
                        "clipping_bounds",
                    )
                    if isinstance(channel, Mapping) and key in channel
                }
                for channel in channels
                if isinstance(channel, Mapping)
            ]
    metadata = {"provenance": provenance, "trace": trace_metadata}
    if quality is not None:
        metadata["quality_report"] = quality
    return metadata, quality


def _external_record(record: Mapping[str, Any], index: int) -> ExternalReportRecord:
    summary, quality = _external_metadata_only(record)
    provenance = summary.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    source_name = provenance.get("source_name")
    source_sha = provenance.get("source_sha256")
    imported_at = provenance.get("imported_at_utc")
    notes: list[str] = []
    valid = (
        provenance.get("source_type") == "external_csv"
        and provenance.get("provenance_class") == "external_unvalidated"
        and provenance.get("validation_status") == "unvalidated_external_input"
        and provenance.get("experimental_validation") is False
    )
    if not valid:
        state: SourceState = "invalid"
        notes.append("external record does not carry the required unvalidated external provenance tags")
    else:
        state = "complete"
    if quality is not None and quality.get("analysis_scope") not in (None, "data_quality_only"):
        state = "invalid"
        notes.append("external quality record is not data_quality_only")
    source_id = f"external_{index + 1:02d}"
    if isinstance(source_name, str) and source_name.strip():
        source_id = source_name.strip()
    return ExternalReportRecord(
        source_id=source_id,
        source_kind="external_unvalidated_signal",
        state=state,
        summary=summary,
        quality_report=quality,
        source_digest=_sha256_payload(summary),
        source_name=source_name if isinstance(source_name, str) else None,
        source_sha256=source_sha if isinstance(source_sha, str) else None,
        imported_at_utc=imported_at if isinstance(imported_at, str) else None,
        notes=_dedupe(notes),
    )


def _project_metadata(
    overview: Mapping[str, Any],
    conditions: Sequence[ReportCondition],
    overview_path: Path,
    root: Path,
) -> dict[str, Any]:
    healthy = next((item for item in conditions if item.condition_id == "healthy"), None)
    healthy_summary = healthy.summary if healthy is not None else {}
    project: dict[str, Any] = {
        "title": PROJECT_TITLE,
        "source_bundle": _relative_path(overview_path, root),
        "scope_note": overview.get("scope_note", ""),
        "stored_overview_label": overview.get("label", ""),
        "condition_count": len(conditions),
        "data_origin": "simulated",
        "parameter_provenance": healthy_summary.get("parameter_provenance"),
        "model_name": healthy_summary.get("parameter_name"),
        "scenario_name": healthy_summary.get("scenario_name"),
        "park_convention": healthy_summary.get("park_convention"),
        "supply_frequency_hz": healthy_summary.get("supply_frequency_hz"),
        "load_type": healthy_summary.get("load_type"),
        "load_torque_nm": healthy_summary.get("load_torque_nm"),
        "solver": healthy_summary.get("solver"),
        "max_step_s": healthy_summary.get("max_step_s"),
        "rtol": healthy_summary.get("rtol"),
        "atol": healthy_summary.get("atol"),
        "t_end_s": healthy_summary.get("t_end_s"),
        "output_dt_s": healthy_summary.get("output_dt_s"),
        "initial_condition": healthy_summary.get("initial_condition"),
    }
    return _sanitize_paths(project, root, overview_path.parent)


def _collect_overview_artifacts(
    overview: Mapping[str, Any], root: Path, results_dir: Path
) -> tuple[tuple[ReportFile, ...], tuple[str, ...], tuple[str, ...]]:
    return _collect_image_files(overview.get("artifacts", {}), root, results_dir)


def _all_input_files(
    overview_file: ReportFile | None,
    conditions: Sequence[ReportCondition],
    overview_artifacts: Sequence[ReportFile],
) -> tuple[ReportFile, ...]:
    records: dict[str, ReportFile] = {}
    records_to_add: list[ReportFile | None] = [overview_file]
    for condition in conditions:
        records_to_add.extend(condition.provenance.files)
        records_to_add.extend(condition.figures)
    records_to_add.extend(overview_artifacts)
    for record in records_to_add:
        if record is not None:
            records[record.path] = record
    # Summary records are not present in provenance.files; add them from the
    # source status path and make the list deterministic.
    for condition in conditions:
        records.setdefault(
            condition.summary_source,
            ReportFile(
                path=condition.summary_source,
                sha256=None,
                exists=condition.source_status.state != "unavailable",
                content_kind="json",
            ),
        )
    for condition in conditions:
        for figure in condition.figures:
            records[figure.path] = figure
    return tuple(records[key] for key in sorted(records))


def _validate_report_timestamp(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise EngineeringReportError("generated_at_utc must be a non-empty ISO-8601 string or None")
    from datetime import datetime

    text = value.strip()
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise EngineeringReportError("generated_at_utc must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None:
        raise EngineeringReportError("generated_at_utc must include a timezone")
    return text


def build_engineering_report(
    results_dir: Path,
    *,
    root: Path | None = None,
    external_sources: Sequence[Mapping[str, Any]] = (),
    generated_at_utc: str | None = None,
    report_git_commit: str | None = None,
) -> EngineeringReport:
    """Build a report model from existing JSON and raster artifacts.

    No NPZ is opened.  A missing optional manifest/run config/figure is
    represented in the source status; a missing or malformed overview is a
    hard input error because the report cannot identify the study bundle.
    """

    results_path = Path(results_dir)
    root_path = Path(root) if root is not None else results_path.parent
    if not results_path.is_absolute():
        results_path = (Path.cwd() / results_path).resolve()
    if not root_path.is_absolute():
        root_path = (Path.cwd() / root_path).resolve()
    results_path = results_path.resolve()
    root_path = root_path.resolve()
    generated_at_utc = _validate_report_timestamp(generated_at_utc)
    if report_git_commit is not None and (
        not isinstance(report_git_commit, str) or not report_git_commit.strip()
    ):
        raise EngineeringReportError("report_git_commit must be a non-empty string or None")

    overview_path = results_path / "fault_overview_summary.json"
    overview, error = _read_json(overview_path)
    if error is not None or not isinstance(overview, Mapping):
        raise ReportInputError(
            f"required overview is unavailable or invalid: {overview_path.name}"
        )
    overview_record = _file_record(
        overview_path, root_path, results_path, kind="json", json_content=True
    )
    raw_conditions = overview.get("conditions")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ReportInputError("required overview does not contain a non-empty conditions list")

    conditions = tuple(
        _condition_record(item, overview, root_path, results_path)
        for item in raw_conditions
        if isinstance(item, Mapping)
    )
    if not conditions:
        raise ReportInputError("required overview contains no condition mappings")

    overview_artifacts, overview_missing, overview_invalid = _collect_overview_artifacts(
        overview, root_path, results_path
    )
    external = tuple(
        _external_record(item, index)
        for index, item in enumerate(external_sources)
        if isinstance(item, Mapping)
    )
    limitations: list[str] = []
    if isinstance(overview.get("scope_note"), str):
        limitations.append(overview["scope_note"])
    for condition in conditions:
        limitations.extend(condition.limitations)
        if isinstance(condition.explanation, Mapping):
            limitations.extend(
                str(item)
                for item in condition.explanation.get("standing_limitations", [])
                if isinstance(item, str)
            )
        if isinstance(condition.triage, Mapping):
            limitations.extend(
                str(item)
                for item in condition.triage.get("limitations", [])
                if isinstance(item, str)
            )
    limitations.extend(REPORT_SCOPE_LIMITATIONS)
    limitations = list(_dedupe(limitations))

    project = _project_metadata(overview, conditions, overview_path, root_path)
    stored_checks = tuple(
        item
        for item in overview.get("consistency_checks", [])
        if isinstance(item, Mapping)
    ) if isinstance(overview.get("consistency_checks"), list) else ()
    stored_all_checks_pass = (
        overview.get("all_checks_pass")
        if isinstance(overview.get("all_checks_pass"), bool)
        else None
    )

    provisional = EngineeringReport(
        schema_version=REPORT_SCHEMA_VERSION,
        report_id="",
        project=project,
        conditions=conditions,
        external_sources=external,
        input_files=_all_input_files(
            overview_record, conditions, overview_artifacts
        ),
        overview_artifacts=overview_artifacts,
        stored_consistency_checks=stored_checks,
        stored_all_checks_pass=stored_all_checks_pass,
        limitations=tuple(limitations),
        generated_at_utc=generated_at_utc,
        report_git_commit=report_git_commit.strip() if report_git_commit else None,
        root=root_path,
    )
    identity_payload = provisional.to_dict()
    identity_payload["report_id"] = ""
    report_id = f"report-{_sha256_payload(identity_payload)[:16]}"
    return replace(provisional, report_id=report_id)


# ---------------------------------------------------------------------------
# Output orchestration
# ---------------------------------------------------------------------------


def _preflight_output_paths(
    output_dir: Path, formats: Sequence[str], overwrite: bool
) -> dict[str, Path]:
    names = {
        "json": "imcm_engineering_report.json",
        "html": "imcm_engineering_report.html",
        "pdf": "imcm_engineering_report.pdf",
        "manifest": "imcm_engineering_report_manifest.json",
    }
    selected = tuple(dict.fromkeys(formats))
    unknown = [fmt for fmt in selected if fmt not in names]
    if unknown:
        raise EngineeringReportError(f"unsupported report format(s): {', '.join(unknown)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {fmt: output_dir / names[fmt] for fmt in selected}
    if not overwrite:
        existing = [str(path) for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing report artifact(s): "
                + ", ".join(existing)
            )
    return paths


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_engineering_report(
    report: EngineeringReport,
    output_dir: Path,
    *,
    formats: Sequence[ReportFormat] = ("json", "html", "pdf", "manifest"),
    overwrite: bool = False,
) -> ReportArtifacts:
    """Write the selected report formats from one canonical model.

    The source bundle is never written.  The default output names are unique
    to this reporting layer and existing files are protected unless the caller
    explicitly passes ``overwrite=True``.
    """

    from imcm.reporting.report_html import render_html
    from imcm.reporting.report_pdf import render_pdf

    selected = tuple(dict.fromkeys(formats))
    paths = _preflight_output_paths(Path(output_dir), selected, overwrite)

    # Render all content before replacing any final artifact.
    json_bytes = (
        (_canonical_json(report.to_dict()) + "\n").encode("utf-8")
        if "json" in selected
        else None
    )
    html_bytes = render_html(report).encode("utf-8") if "html" in selected else None

    pdf_bytes: bytes | None = None
    pdf_temp: Path | None = None
    if "pdf" in selected:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=output_path, prefix=".imcm-engineering-report-", suffix=".pdf", delete=False
        ) as handle:
            pdf_temp = Path(handle.name)
        try:
            render_pdf(report, pdf_temp, overwrite=True)
            pdf_bytes = pdf_temp.read_bytes()
        finally:
            if pdf_temp.exists():
                pdf_temp.unlink()

    payloads: dict[str, bytes] = {}
    if json_bytes is not None:
        payloads["json"] = json_bytes
    if html_bytes is not None:
        payloads["html"] = html_bytes
    if pdf_bytes is not None:
        payloads["pdf"] = pdf_bytes

    output_hashes = {fmt: _sha256_bytes(data) for fmt, data in payloads.items()}
    if "manifest" in selected:
        manifest_payload = {
            "schema_version": REPORT_MANIFEST_SCHEMA_VERSION,
            "report_schema_version": report.schema_version,
            "report_generator_version": REPORT_GENERATOR_VERSION,
            "report_id": report.report_id,
            "generated_at_utc": report.generated_at_utc,
            "report_git_commit": report.report_git_commit,
            "input_files": [item.to_dict() for item in report.input_files],
            "outputs": [
                {
                    "format": fmt,
                    "path": paths[fmt].name,
                    "sha256": output_hashes[fmt],
                }
                for fmt in ("json", "html", "pdf")
                if fmt in payloads
            ],
        }
        payloads["manifest"] = (
            json.dumps(manifest_payload, indent=2, sort_keys=True, ensure_ascii=False)
            + "\n"
        ).encode("utf-8")

    for fmt, data in payloads.items():
        _atomic_write(paths[fmt], data)

    hashes = {fmt: _sha256_bytes(data) for fmt, data in payloads.items()}
    return ReportArtifacts(paths=paths, hashes=hashes)


__all__ = [
    "EngineeringReport",
    "EngineeringReportError",
    "ExternalReportRecord",
    "PROJECT_TITLE",
    "REPORT_GENERATOR_VERSION",
    "REPORT_MANIFEST_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "REPORT_SCOPE_LIMITATIONS",
    "ReportArtifacts",
    "ReportCondition",
    "ReportFile",
    "ReportInputError",
    "ReportProvenance",
    "ReportSourceStatus",
    "build_engineering_report",
    "write_engineering_report",
]