
"""Generate a multi-qubit quantum error prediction dataset."""

import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

try:
    from .multiqubit_circuit_generator import (
        CIRCUIT_FAMILIES,
        generate_multiqubit_circuit,
    )
except ImportError:
    from multiqubit_circuit_generator import (
        CIRCUIT_FAMILIES,
        generate_multiqubit_circuit,
    )


OUTPUT_PATH = Path("data/raw/multiqubit_dataset.csv")

SHOTS_OPTIONS = (256, 1024, 4096)
NOISE_LEVELS = (0.001, 0.003, 0.005, 0.01, 0.02)
QUBIT_COUNTS = tuple(range(2, 9))

CIRCUITS_PER_QUBIT_COUNT = {
    2: 60,
    3: 50,
    4: 40,
    5: 30,
    6: 20,
    7: 15,
    8: 10,
}

BASIS_GATES = ["h", "rx", "ry", "rz", "cx"]


def observable_operator(n_qubits, qubits):
    """Create a Pauli Z or ZZ observable."""
    pauli = ["I"] * n_qubits
    for qubit in qubits:
        pauli[n_qubits - 1 - qubit] = "Z"

    return SparsePauliOp.from_list([
        ("".join(pauli), 1.0)
    ])


def make_observables(n_qubits):
    """Return every Z_i and adjacent ZZ observable."""
    observables = [
        (f"Z_{q}", (q,))
        for q in range(n_qubits)
    ]
    observables.extend(
        (f"ZZ_{q}_{q + 1}", (q, q + 1))
        for q in range(n_qubits - 1)
    )
    return observables


def ideal_expectations(circuit, observables):
    """Compute exact ideal expectation values."""
    state = Statevector.from_instruction(circuit)
    values = {}

    for name, qubits in observables:
        operator = observable_operator(circuit.num_qubits, qubits)
        values[name] = float(
            state.expectation_value(operator).real
        )

    return values


def build_noise_model(probability):
    """Apply depolarizing errors after supported gates."""
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    if not 0.0 <= probability <= 1.0:
        raise ValueError("Noise probability must be in [0, 1]")

    noise_model = NoiseModel()

    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(probability, 1),
        ["h", "rx", "ry", "rz"],
    )
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(min(1.0, 1.5 * probability), 2),
        ["cx"],
    )

    return noise_model


def noisy_expectations(circuit, shots, simulator, seed):
    """Measure all qubits once and estimate Z and adjacent ZZ values."""
    from qiskit import QuantumCircuit

    n_qubits = circuit.num_qubits
    measured = QuantumCircuit(n_qubits, n_qubits)
    measured.compose(circuit, inplace=True)
    measured.measure(range(n_qubits), range(n_qubits))

    compiled = transpile(
        measured,
        basis_gates=BASIS_GATES,
        optimization_level=0,
        seed_transpiler=seed,
    )

    result = simulator.run(
        compiled,
        shots=shots,
        seed_simulator=seed,
    ).result()

    counts = result.get_counts()
    total = sum(counts.values())

    if total == 0:
        raise RuntimeError("Simulator returned no measurement counts")

    sums = {
        **{f"Z_{q}": 0 for q in range(n_qubits)},
        **{
            f"ZZ_{q}_{q + 1}": 0
            for q in range(n_qubits - 1)
        },
    }

    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        z = [
            1 if bits[-1 - q] == "0" else -1
            for q in range(n_qubits)
        ]

        for q in range(n_qubits):
            sums[f"Z_{q}"] += z[q] * count

        for q in range(n_qubits - 1):
            sums[f"ZZ_{q}_{q + 1}"] += z[q] * z[q + 1] * count

    return {
        name: value / total
        for name, value in sums.items()
    }


def circuit_id_from_spec(spec):
    """Create a stable ID from canonical circuit metadata."""
    canonical = json.dumps(
        spec,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()[:16]


def generate_dataset(
    seed=42,
    shots_options=SHOTS_OPTIONS,
    noise_levels=NOISE_LEVELS,
    qubit_counts=QUBIT_COUNTS,
    circuits_per_qubit_count=None,
    output_path=OUTPUT_PATH,
):
    """Generate circuits and all configured simulation experiments."""
    if not shots_options or any(s <= 0 for s in shots_options):
        raise ValueError("All shot counts must be positive")
    if len(set(shots_options)) != len(shots_options):
        raise ValueError("shots_options must not contain duplicates")
    if not noise_levels or any(
        not 0.0 <= p <= 1.0 for p in noise_levels
    ):
        raise ValueError("Noise probabilities must be in [0, 1]")
    if len(set(noise_levels)) != len(noise_levels):
        raise ValueError("noise_levels must not contain duplicates")
    if not qubit_counts or any(n < 2 for n in qubit_counts):
        raise ValueError("All qubit counts must be >= 2")

    counts_by_n = dict(CIRCUITS_PER_QUBIT_COUNT)
    if circuits_per_qubit_count is not None:
        counts_by_n.update(circuits_per_qubit_count)

    for n in qubit_counts:
        if counts_by_n.get(n, 0) < 1:
            raise ValueError(
                f"Missing positive circuit count for {n} qubits"
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    records = []
    seen_circuits = set()
    circuit_counter = 0

    # Reuse one simulator per noise level.
    simulators = {
        p: AerSimulator(noise_model=build_noise_model(p))
        for p in noise_levels
    }

    total_circuits_expected = sum(
        counts_by_n[n] * len(CIRCUIT_FAMILIES)
        for n in qubit_counts
    )
    print(f"Target unique circuits: {total_circuits_expected}")

    for n_qubits in qubit_counts:
        target_count = counts_by_n[n_qubits]

        for family in CIRCUIT_FAMILIES:
            family_count = 0
            attempts = 0
            max_attempts = target_count * 100

            while family_count < target_count:
                attempts += 1
                if attempts > max_attempts:
                    raise RuntimeError(
                        f"Could not generate enough unique circuits: "
                        f"{family}, {n_qubits} qubits"
                    )

                circuit_seed = rng.randrange(0, 2**31)
                n_layers = rng.randint(1, 3)

                circuit, metadata = generate_multiqubit_circuit(
                    seed=circuit_seed,
                    n_qubits=n_qubits,
                    family=family,
                    n_layers=n_layers,
                )

                spec = {
                    "n_qubits": n_qubits,
                    "family": family,
                    "n_layers": n_layers,
                    "gates": metadata["gates"],
                }
                circuit_id = circuit_id_from_spec(spec)

                if circuit_id in seen_circuits:
                    continue

                seen_circuits.add(circuit_id)
                family_count += 1
                circuit_counter += 1

                observables = make_observables(n_qubits)
                ideal_values = ideal_expectations(circuit, observables)

                for noise_probability in noise_levels:
                    simulator = simulators[noise_probability]

                    for shots in shots_options:
                        experiment_seed = rng.randrange(0, 2**31)

                        noisy_values = noisy_expectations(
                            circuit=circuit,
                            shots=shots,
                            simulator=simulator,
                            seed=experiment_seed,
                        )

                        for observable_name, _ in observables:
                            ideal = ideal_values[observable_name]
                            noisy = noisy_values[observable_name]

                            records.append({
                                "record_id": (
                                    f"{circuit_id}_{observable_name}_"
                                    f"{noise_probability}_{shots}_"
                                    f"{experiment_seed}"
                                ),
                                "circuit_id": circuit_id,
                                "circuit_type": family,
                                "n_qubits": n_qubits,
                                "circuit_depth": metadata["circuit_depth"],
                                "single_qubit_gate_count": (
                                    metadata["single_qubit_gate_count"]
                                ),
                                "cx_count": metadata["cx_count"],
                                "mean_rotation_angle": (
                                    metadata["mean_rotation_angle"]
                                ),
                                "observable": observable_name,
                                "circuit_params": json.dumps({
                                    **spec,
                                    "observable": observable_name,
                                }, sort_keys=True),
                                "noise_category": "depolarizing",
                                "noise_params": json.dumps({
                                    "probability_1q": noise_probability,
                                    "probability_2q": min(
                                        1.0, 1.5 * noise_probability
                                    ),
                                }, sort_keys=True),
                                "backend_name": "AerSimulator",
                                "shots": shots,
                                "seed": experiment_seed,
                                "ideal_expval": ideal,
                                "noisy_expval": noisy,
                                "delta": ideal - noisy,
                                "timestamp": datetime.now(
                                    timezone.utc
                                ).isoformat(),
                            })

                if circuit_counter % 10 == 0:
                    print(
                        f"Circuits: {circuit_counter}/"
                        f"{total_circuits_expected} | "
                        f"Records: {len(records)}"
                    )

    expected_rows = sum(
        counts_by_n[n] * len(CIRCUIT_FAMILIES)
        * (2 * n - 1)
        * len(shots_options)
        * len(noise_levels)
        for n in qubit_counts
    )

    if len(records) != expected_rows:
        raise RuntimeError(
            f"Row count mismatch: expected {expected_rows}, "
            f"got {len(records)}"
        )

    fieldnames = list(records[0].keys())
    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print("\nDataset generation complete.")
    print(f"Unique circuits: {len(seen_circuits)}")
    print(f"Experiment rows: {len(records)}")
    print(f"CSV saved to: {output_path.resolve()}")


if __name__ == "__main__":
    generate_dataset(seed=42)
