"""Data schema and JSONL serialization for quantum error mitigation.

Location: src/data_schema.py
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any


DATASET_DOMAINS = {
    "synthetic_noise_sim",
    "ibm_hardware_test",
}

GATE_NAMES = {"H", "RX", "RY", "RZ", "CX"}


class DuplicateRecordError(ValueError):
    """Raised when a record with an existing record_id is appended."""


@dataclass(frozen=True)
class ExperimentRecord:
    # Identification
    record_id: str
    circuit_id: str
    experiment_id: str

    # Data provenance
    dataset_domain: str

    # Circuit description
    circuit_type: str
    n_qubits: int
    circuit_depth: int
    cx_count: int
    single_qubit_gate_count: int
    mean_rotation_angle: float
    observable: str
    circuit_params: dict[str, Any]

    # Execution conditions
    noise_category: str
    backend_name: str
    shots: int
    seed: int
    noise_params: dict[str, Any] = field(default_factory=dict)
    calibration_snapshot_id: str | None = None

    # Results: must be explicitly supplied
    ideal_expval: float = 0.0
    noisy_expval: float = 0.0
    delta: float = 0.0

    # Reproducibility
    timestamp: str = ""
    software_versions: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate record structure, metadata and numerical consistency."""

        # Required string fields
        for name in (
            "record_id",
            "circuit_id",
            "experiment_id",
            "dataset_domain",
            "circuit_type",
            "observable",
            "noise_category",
            "backend_name",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

        if self.dataset_domain not in DATASET_DOMAINS:
            raise ValueError(
                f"Unsupported dataset_domain: {self.dataset_domain}"
            )

        # Integer fields
        for name in (
            "n_qubits",
            "circuit_depth",
            "cx_count",
            "single_qubit_gate_count",
            "shots",
            "seed",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer, got {type(value).__name__}")

        if self.n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")

        if self.circuit_depth < 0:
            raise ValueError("circuit_depth cannot be negative")

        if self.cx_count < 0 or self.single_qubit_gate_count < 0:
            raise ValueError("Gate counts cannot be negative")

        if self.shots <= 0:
            raise ValueError("shots must be positive")

        if self.seed < 0:
            raise ValueError("seed must be non-negative")

        # Numeric fields
        for name in (
            "ideal_expval",
            "noisy_expval",
            "delta",
            "mean_rotation_angle",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")

        if self.mean_rotation_angle < 0:
            raise ValueError("mean_rotation_angle cannot be negative")

        # Expectation values of a Pauli observable must be in [-1, 1].
        tolerance = 1e-7
        for name in ("ideal_expval", "noisy_expval"):
            value = getattr(self, name)
            if not -1.0 - tolerance <= value <= 1.0 + tolerance:
                raise ValueError(f"{name} must be in [-1, 1]")

        # The target must be calculated from the same record's results.
        expected_delta = self.ideal_expval - self.noisy_expval
        if not math.isclose(
            self.delta,
            expected_delta,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"delta ({self.delta}) must equal ideal_expval - noisy_expval ({expected_delta})"
            )

        # Validate circuit specification.
        if not isinstance(self.circuit_params, dict):
            raise ValueError("circuit_params must be a dictionary")

        gates = self.circuit_params.get("gates")
        if not isinstance(gates, list):
            raise ValueError("circuit_params must contain a gates list")

        if self.circuit_params.get("n_qubits") != self.n_qubits:
            raise ValueError(
                "circuit_params.n_qubits must match n_qubits"
            )

        if self.circuit_params.get("observable") != self.observable:
            raise ValueError(
                "circuit_params.observable must match observable"
            )

        rotation_angles = []
        calc_cx_count = 0
        calc_single_qubit_count = 0

        for gate in gates:
            if not isinstance(gate, dict):
                raise ValueError("Each item in circuit_params.gates must be a dictionary")

            gate_name = gate.get("name")
            if gate_name not in GATE_NAMES:
                raise ValueError(f"Unknown or missing gate name: {gate_name}")

            if gate_name == "H":
                qubit = gate.get("qubit")
                if type(qubit) is not int or not 0 <= qubit < self.n_qubits:
                    raise ValueError(f"Invalid H qubit index: {qubit}")
                calc_single_qubit_count += 1

            elif gate_name in {"RX", "RY", "RZ"}:
                qubit = gate.get("qubit")
                if type(qubit) is not int or not 0 <= qubit < self.n_qubits:
                    raise ValueError(f"Invalid {gate_name} qubit index: {qubit}")

                theta = gate.get("theta")
                if (
                    isinstance(theta, bool)
                    or not isinstance(theta, (int, float))
                    or not math.isfinite(theta)
                ):
                    raise ValueError(
                        f"{gate_name} gate must have a finite numeric theta"
                    )

                # Mean rotation angle represents mean of absolute values
                rotation_angles.append(abs(float(theta)))
                calc_single_qubit_count += 1

            elif gate_name == "CX":
                control = gate.get("control")
                target = gate.get("target")

                if (
                    type(control) is not int
                    or type(target) is not int
                    or control == target
                    or not 0 <= control < self.n_qubits
                    or not 0 <= target < self.n_qubits
                ):
                    raise ValueError(f"Invalid CX control/target indices: {control}, {target}")

                calc_cx_count += 1

        if calc_cx_count != self.cx_count:
            raise ValueError(
                f"cx_count ({self.cx_count}) does not match circuit_params ({calc_cx_count})"
            )

        if calc_single_qubit_count != self.single_qubit_gate_count:
            raise ValueError(
                f"single_qubit_gate_count ({self.single_qubit_gate_count}) does not match circuit_params ({calc_single_qubit_count})"
            )

        if rotation_angles:
            expected_mean = sum(rotation_angles) / len(rotation_angles)
            if not math.isclose(
                self.mean_rotation_angle,
                expected_mean,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    f"mean_rotation_angle ({self.mean_rotation_angle}) does not match circuit_params ({expected_mean})"
                )
        elif self.mean_rotation_angle != 0:
            raise ValueError(
                "mean_rotation_angle must be zero when there are no rotations"
            )

        # Timestamp must be ISO-8601 with timezone.
        if not isinstance(self.timestamp, str) or not self.timestamp.strip():
            raise ValueError("timestamp must be a non-empty ISO-8601 string")

        try:
            dt = datetime.fromisoformat(
                self.timestamp.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError(
                "timestamp must be a valid ISO-8601 datetime"
            ) from exc

        if dt.tzinfo is None or dt.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")

        # Hardware results must be tied to calibration information.
        if (
            self.dataset_domain == "ibm_hardware_test"
            and not self.calibration_snapshot_id
        ):
            raise ValueError(
                "Hardware records require calibration_snapshot_id"
            )

        if not isinstance(self.noise_params, dict):
            raise ValueError("noise_params must be a dictionary")

        if not isinstance(self.software_versions, dict):
            raise ValueError("software_versions must be a dictionary")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


def make_id(prefix: str, data: dict[str, Any]) -> str:
    """Create a deterministic ID from canonical JSON."""
    if not prefix or not isinstance(prefix, str):
        raise ValueError("prefix must be a non-empty string")

    payload = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def make_circuit_id(circuit_params: dict[str, Any]) -> str:
    """Identify a circuit by its complete canonical specification."""
    return make_id("circuit", circuit_params)


def make_experiment_id(
    circuit_id: str,
    dataset_domain: str,
    noise_category: str,
    noise_params: dict[str, Any],
    backend_name: str,
    shots: int,
    calibration_snapshot_id: str | None = None,
) -> str:
    """Identify the experimental configuration, excluding run seed."""
    return make_id(
        "experiment",
        {
            "circuit_id": circuit_id,
            "dataset_domain": dataset_domain,
            "noise_category": noise_category,
            "noise_params": noise_params,
            "backend_name": backend_name,
            "shots": shots,
            "calibration_snapshot_id": calibration_snapshot_id,
        },
    )


def make_record_id(
    experiment_id: str,
    seed: int,
    run_index: int = 0,
) -> str:
    """Identify a particular run of an experimental configuration."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    if type(run_index) is not int or run_index < 0:
        raise ValueError("run_index must be a non-negative integer")

    return make_id(
        "record",
        {
            "experiment_id": experiment_id,
            "seed": seed,
            "run_index": run_index,
        },
    )


def get_existing_record_ids(path: str | Path) -> set[str]:
    """Read record IDs from an existing JSONL dataset."""
    path = Path(path)

    if not path.exists():
        return set()

    existing: set[str] = set()

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
                record_id = item["record_id"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"Invalid JSONL record at line {line_number} in {path}"
                ) from exc

            if not isinstance(record_id, str):
                raise ValueError(
                    f"Invalid record_id at line {line_number} in {path}"
                )

            if record_id in existing:
                raise ValueError(
                    f"Duplicate record_id {record_id} at line {line_number}"
                )

            existing.add(record_id)

    return existing


def append_jsonl(
    path: str | Path,
    record: ExperimentRecord,
) -> None:
    """Validate and append a record with a unique record_id."""
    record.validate()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = get_existing_record_ids(path)

    if record.record_id in existing_ids:
        raise DuplicateRecordError(
            f"Record {record.record_id} already exists in {path}"
        )

    with path.open("a", encoding="utf-8") as file:
        file.write(record.to_json() + "\n")