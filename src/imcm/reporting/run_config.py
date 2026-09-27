"""Machine-readable run-configuration records for IMCM experiments.

Every experiment runner in this repository produces a *run config*: a compact,
canonical description of exactly what was simulated and how. A run config is
paired with a provenance manifest (see :mod:`imcm.reporting.provenance`) so
that any output file can be traced back to the condition, scenario,
parameters, solver settings, and code revision that produced it.

This module is deliberately narrow. It records values that already exist in
the project; it never computes physics, never changes solver settings, and
never invents parameters. All six first-milestone studies share the same
schema so downstream tooling can consume them uniformly.

Canonical schema (``schema_version`` = :data:`RUN_CONFIG_SCHEMA_VERSION`)::

    schema_version
    run_id
    condition_id
    condition_kind
    scenario_name
    park_convention
    motor_parameters
    solver_settings
    initial_condition
    analysis_windows
    condition_parameters        # carries the tagged ``type``
    outputs
    provenance_manifest

The one required tag inside ``condition_parameters`` is ``type``, which must
be one of :data:`CONDITION_TYPES`.

The module imports only the standard library so it can be used without the
scientific stack.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Union

__all__ = [
    "RUN_CONFIG_SCHEMA_VERSION",
    "CONDITION_TYPES",
    "RunConfigError",
    "RunConfig",
    "run_config_hash",
    "make_run_id",
    "write_run_config",
    "read_run_config",
    "emit_run_artifacts",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Version of the run-config schema emitted by this module.
RUN_CONFIG_SCHEMA_VERSION = "1.0"

#: The only permitted ``condition_parameters["type"]`` tags.
CONDITION_TYPES: frozenset[str] = frozenset(
    {
        "healthy",
        "stator_resistance_imbalance",
        "increased_mechanical_load",
        "supply_voltage_unbalance",
        "bearing_outer_race",
        "rotor_asymmetry",
    }
)

#: Field names that every run config must define (canonical order).
_REQUIRED_FIELDS: tuple[str, ...] = (
    "schema_version",
    "run_id",
    "condition_id",
    "condition_kind",
    "scenario_name",
    "park_convention",
    "motor_parameters",
    "solver_settings",
    "initial_condition",
    "analysis_windows",
    "condition_parameters",
    "outputs",
    "provenance_manifest",
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RunConfigError(ValueError):
    """Raised when a run config is malformed or mislabelled."""


# ---------------------------------------------------------------------------
# Small helpers (mirrors imcm.reporting.provenance)
# ---------------------------------------------------------------------------


def _require_text(name: str, value: object) -> str:
    """Validate that ``value`` is a non-empty string and return it stripped."""
    if not isinstance(value, str):
        raise RunConfigError(f"{name} must be a string, got {type(value).__name__}")
    stripped = value.strip()
    if not stripped:
        raise RunConfigError(f"{name} must be a non-empty string")
    return stripped


def _require_mapping(name: str, value: object) -> dict[str, Any]:
    """Validate that ``value`` is a mapping and return a plain dict copy."""
    if not isinstance(value, Mapping):
        raise RunConfigError(
            f"{name} must be a mapping (e.g. dict), got {type(value).__name__}"
        )
    return dict(value)


def _require_sequence_of_mappings(
    name: str, value: object
) -> list[dict[str, Any]]:
    """Validate a sequence of mappings (e.g. ``analysis_windows``)."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RunConfigError(f"{name} must be a sequence of mappings")
    normalised: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise RunConfigError(
                f"every entry in {name} must be a mapping, "
                f"got {type(item).__name__}"
            )
        normalised.append(dict(item))
    return normalised


def _normalise_text_sequence(name: str, value: object) -> tuple[str, ...]:
    """Validate a sequence of non-empty strings (e.g. ``outputs``)."""
    if isinstance(value, (str, bytes)):
        raise RunConfigError(f"{name} must be a sequence of strings, not a string")
    try:
        items = list(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise RunConfigError(f"{name} must be an iterable of strings") from exc
    normalised: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise RunConfigError(
                f"every entry in {name} must be a string, got {type(item).__name__}"
            )
        stripped = item.strip()
        if not stripped:
            raise RunConfigError(f"every entry in {name} must be non-empty")
        normalised.append(stripped)
    return tuple(normalised)


def _jsonable(value: Any) -> Any:
    """Recursively convert mappings/sequences into plain JSON-safe values."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (str, bytes)):
        return value.decode("utf-8") if isinstance(value, bytes) else value
    if isinstance(value, Sequence):
        return [_jsonable(item) for item in value]
    return value


def _canonical_bytes(payload: Any) -> bytes:
    """Deterministic canonical encoding used for run-config hashing."""
    return json.dumps(
        _jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# Deterministic identifiers
# ---------------------------------------------------------------------------


def run_config_hash(config: Union["RunConfig", Mapping[str, Any]]) -> str:
    """Return a deterministic SHA-256 hash of a run config.

    Used to link a provenance manifest back to the exact run config that
    produced it. The hash is stable across machines and runs because the
    payload is canonicalised (sorted keys, no insignificant whitespace).
    """
    data = config.to_dict() if isinstance(config, RunConfig) else dict(config)
    return hashlib.sha256(_canonical_bytes(data)).hexdigest()


def make_run_id(
    *,
    condition_id: str,
    scenario_name: str,
    motor_parameters: Mapping[str, Any],
    solver_settings: Mapping[str, Any],
    condition_parameters: Mapping[str, Any],
) -> str:
    """Build a deterministic run id from the identifying config fields.

    The id is derived only from values already present in the project, so the
    same condition and settings always yield the same id.
    """
    identity = {
        "condition_id": condition_id,
        "scenario_name": scenario_name,
        "motor_parameters": _jsonable(motor_parameters),
        "solver_settings": _jsonable(solver_settings),
        "condition_parameters": _jsonable(condition_parameters),
    }
    digest = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    return f"run-{digest[:16]}"


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """A canonical description of one simulated experiment run.

    Parameters
    ----------
    schema_version:
        Version tag; must equal :data:`RUN_CONFIG_SCHEMA_VERSION`.
    run_id:
        Deterministic identifier for the run (see :func:`make_run_id`).
    condition_id:
        Stable condition key (e.g. ``healthy``, ``fault_01``).
    condition_kind:
        Human-readable nature of the condition. For controlled fault /
        operating-condition studies this is the existing honesty descriptor
        (e.g. ``operating_condition_change_not_internal_fault``).
    scenario_name:
        ``OperatingScenario.name`` used for the run.
    park_convention:
        Locked Park convention string.
    motor_parameters:
        Full motor parameter set used.
    solver_settings:
        Integrator configuration (method, tolerances, max_step, span).
    initial_condition:
        Initial state description (start from rest).
    analysis_windows:
        Sequence of named analysis windows (start/end times).
    condition_parameters:
        Condition-specific values. Must carry ``{"type": <CONDITION_TYPES>}``.
    outputs:
        Files written for the run (relative to the repository root).
    provenance_manifest:
        Reference to the paired provenance manifest (at least a path).
    """

    schema_version: str
    run_id: str
    condition_id: str
    condition_kind: str
    scenario_name: str
    park_convention: str
    motor_parameters: Mapping[str, Any]
    solver_settings: Mapping[str, Any]
    initial_condition: Mapping[str, Any]
    analysis_windows: Sequence[Mapping[str, Any]]
    condition_parameters: Mapping[str, Any]
    outputs: Sequence[str]
    provenance_manifest: Mapping[str, Any]

    # -- validation ---------------------------------------------------------

    def __post_init__(self) -> None:
        self.schema_version = _require_text("schema_version", self.schema_version)
        if self.schema_version != RUN_CONFIG_SCHEMA_VERSION:
            raise RunConfigError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {RUN_CONFIG_SCHEMA_VERSION!r}"
            )
        self.run_id = _require_text("run_id", self.run_id)
        self.condition_id = _require_text("condition_id", self.condition_id)
        self.condition_kind = _require_text("condition_kind", self.condition_kind)
        self.scenario_name = _require_text("scenario_name", self.scenario_name)
        self.park_convention = _require_text(
            "park_convention", self.park_convention
        )
        self.motor_parameters = _require_mapping(
            "motor_parameters", self.motor_parameters
        )
        self.solver_settings = _require_mapping(
            "solver_settings", self.solver_settings
        )
        self.initial_condition = _require_mapping(
            "initial_condition", self.initial_condition
        )
        self.analysis_windows = _require_sequence_of_mappings(
            "analysis_windows", self.analysis_windows
        )
        self.condition_parameters = _require_mapping(
            "condition_parameters", self.condition_parameters
        )
        self._validate_condition_type(self.condition_parameters)
        self.outputs = _normalise_text_sequence("outputs", self.outputs)
        self.provenance_manifest = _require_mapping(
            "provenance_manifest", self.provenance_manifest
        )

    @staticmethod
    def _validate_condition_type(condition_parameters: Mapping[str, Any]) -> str:
        """Validate the required tagged ``type`` inside condition parameters."""
        if "type" not in condition_parameters:
            raise RunConfigError(
                "condition_parameters must include a 'type' tag"
            )
        raw = condition_parameters["type"]
        if not isinstance(raw, str):
            raise RunConfigError(
                "condition_parameters['type'] must be a string, "
                f"got {type(raw).__name__}"
            )
        tag = raw.strip().lower()
        if tag not in CONDITION_TYPES:
            allowed = ", ".join(sorted(CONDITION_TYPES))
            raise RunConfigError(
                f"invalid condition_parameters['type'] {raw!r}; "
                f"allowed tags are: {allowed}"
            )
        return tag

    @property
    def condition_type(self) -> str:
        """The validated canonical condition tag."""
        return self._validate_condition_type(self.condition_parameters)

    # -- convenience constructors ------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        condition_id: str,
        condition_kind: str,
        scenario_name: str,
        park_convention: str,
        motor_parameters: Mapping[str, Any],
        solver_settings: Mapping[str, Any],
        initial_condition: Mapping[str, Any],
        analysis_windows: Sequence[Mapping[str, Any]],
        condition_parameters: Mapping[str, Any],
        outputs: Iterable[str] = (),
        provenance_manifest: Mapping[str, Any] | None = None,
        run_id: str | None = None,
        schema_version: str = RUN_CONFIG_SCHEMA_VERSION,
    ) -> "RunConfig":
        """Build a run config, deriving ``run_id`` when it is omitted."""
        resolved_id = run_id or make_run_id(
            condition_id=condition_id,
            scenario_name=scenario_name,
            motor_parameters=motor_parameters,
            solver_settings=solver_settings,
            condition_parameters=condition_parameters,
        )
        return cls(
            schema_version=schema_version,
            run_id=resolved_id,
            condition_id=condition_id,
            condition_kind=condition_kind,
            scenario_name=scenario_name,
            park_convention=park_convention,
            motor_parameters=motor_parameters,
            solver_settings=solver_settings,
            initial_condition=initial_condition,
            analysis_windows=analysis_windows,
            condition_parameters=condition_parameters,
            outputs=outputs,
            provenance_manifest=provenance_manifest or {},
        )

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe plain-dict representation."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "condition_id": self.condition_id,
            "condition_kind": self.condition_kind,
            "scenario_name": self.scenario_name,
            "park_convention": self.park_convention,
            "motor_parameters": _jsonable(self.motor_parameters),
            "solver_settings": _jsonable(self.solver_settings),
            "initial_condition": _jsonable(self.initial_condition),
            "analysis_windows": _jsonable(self.analysis_windows),
            "condition_parameters": _jsonable(self.condition_parameters),
            "outputs": list(self.outputs),
            "provenance_manifest": _jsonable(self.provenance_manifest),
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise the run config to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunConfig":
        """Reconstruct a run config from a mapping, enforcing required fields."""
        if not isinstance(data, Mapping):
            raise RunConfigError(
                f"run config data must be a mapping, got {type(data).__name__}"
            )
        missing = [name for name in _REQUIRED_FIELDS if name not in data]
        if missing:
            raise RunConfigError(
                f"run config data is missing required fields: {missing}"
            )
        return cls(**{name: data[name] for name in _REQUIRED_FIELDS})

    @classmethod
    def from_json(cls, text: str) -> "RunConfig":
        """Parse a run config from a JSON string."""
        if not isinstance(text, str):
            raise RunConfigError(
                f"from_json expects a string, got {type(text).__name__}"
            )
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RunConfigError(f"invalid run-config JSON: {exc}") from exc
        if not isinstance(data, Mapping):
            raise RunConfigError("run-config JSON must decode to an object")
        return cls.from_dict(data)

    # -- reporting ----------------------------------------------------------

    def summary(self) -> str:
        """A short human-readable one-liner for logs and report headers."""
        return (
            f"[{self.condition_type}] run_id={self.run_id} "
            f"condition={self.condition_id} scenario={self.scenario_name!r} "
            f"park={self.park_convention!r} outputs={len(self.outputs)}"
        )


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def _coerce_config(config: object) -> RunConfig:
    """Accept either a RunConfig or a raw mapping."""
    if isinstance(config, RunConfig):
        return config
    if isinstance(config, Mapping):
        return RunConfig.from_dict(config)
    raise TypeError(
        "config must be a RunConfig or a mapping, "
        f"got {type(config).__name__}"
    )


def write_run_config(
    path: Union[str, Path], config: Union[RunConfig, Mapping[str, Any]]
) -> Path:
    """Write a run config to ``path`` as pretty-printed JSON.

    Parent directories are created if needed. Returns the written path.
    """
    resolved = _coerce_config(config)
    target = Path(path)
    parent = target.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    target.write_text(resolved.to_json() + "\n", encoding="utf-8")
    return target


def read_run_config(path: Union[str, Path]) -> RunConfig:
    """Read a run config previously written by :func:`write_run_config`."""
    text = Path(path).read_text(encoding="utf-8")
    return RunConfig.from_json(text)


# ---------------------------------------------------------------------------
# Emission helper (run config + provenance manifest, linked)
# ---------------------------------------------------------------------------


def emit_run_artifacts(
    run_config: RunConfig,
    provenance: object,
    *,
    run_config_path: Union[str, Path],
    manifest_path: Union[str, Path],
    root: Union[str, Path],
) -> tuple[Path, Path]:
    """Write a run config and its provenance manifest, cross-linked.

    The run config records the manifest path (known up front). The manifest
    then records the run config path, ``run_id``, ``schema_version`` and the
    deterministic :func:`run_config_hash`, so the pair is fully traceable in
    both directions.

    ``provenance`` is a :class:`~imcm.reporting.provenance.ProvenanceRecord`
    (imported lazily to keep this module standard-library only). Its
    ``run_config`` field is populated here.

    Returns ``(run_config_path, manifest_path)``.
    """
    # Lazy imports keep module import cheap and avoid a hard dependency cycle.
    from imcm.reporting.provenance import ProvenanceRecord, write_manifest

    if not isinstance(provenance, ProvenanceRecord):
        raise TypeError(
            "provenance must be a ProvenanceRecord, "
            f"got {type(provenance).__name__}"
        )

    root_path = Path(root)
    config_path = write_run_config(run_config_path, run_config)
    digest = run_config_hash(run_config)

    provenance.run_config = {
        "path": str(Path(run_config_path).relative_to(root_path)),
        "run_id": run_config.run_id,
        "schema_version": run_config.schema_version,
        "hash_sha256": digest,
    }
    manifest_written = write_manifest(manifest_path, provenance)
    return config_path, manifest_written
