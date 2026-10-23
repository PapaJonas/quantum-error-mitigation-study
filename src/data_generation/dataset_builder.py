
"""Generate a single-qubit quantum error mitigation dataset."""

import csv
import json
import math
import random
from pathlib import Path
from datetime import datetime, timezone

from qiskit.quantum_info import Statevector, Pauli
from qiskit_aer import AerSimulator


OUTPUT_PATH = Path("data/raw/single_qubit_dataset.csv")

SHOTS_OPTIONS = (256, 1024, 4096)

NOISE_LEVELS = (
    0.001, 0.002, 0.003, 0.004, 0.005,
    0.0075, 0.010, 0.0125, 0.015, 0.0175,
    0.020, 0.0225, 0.025, 0.0275, 0.030,
    0.0325, 0.035, 0.0375, 0.040, 0.045,
    0.050,
)

BASE_ANGLES = (
    math.pi / 6,
    math.pi / 4,
    math.pi / 3,
    math.pi / 2,
)


def ideal_expectation(circuit):
    """Calculate exact ideal expectation value of Z."""
    state = Statevector.from_instruction(circuit)
    return float(state.expectation_value(Pauli("Z")).real)


def build_noise_model(probability):
    """Apply depolarizing noise after each supported gate."""
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    if not 0.0 <= probability <= 1.0:
        raise ValueError("Noise probability must be in [0, 1]")

    noise_model = NoiseModel()
    error = depolarizing_error(probability, 1)

    # Same single-qubit noise after each gate.
    noise_model.add_all_qubit_quantum_error(
        error, ["h", "rx", "ry", "rz"]
    )
    return noise_model


def noisy_expectation(circuit, shots, noise_probability, seed):
    """Estimate Z from finite-shot noisy measurements."""
    measured = circuit.copy()
    measured.measure_all()

    simulator = AerSimulator(
        noise_model=build_noise_model(noise_probability)
    )

    result = simulator.run(
        measured,
        shots=shots,
        seed_simulator=seed,
    ).result()

    counts = result.get_counts()
    total = sum(counts.values())

    # Qiskit measurement convention: bit 0 is the rightmost bit.
    # For a single qubit, 0 contributes +1 and 1 contributes -1.
    expectation = sum(
        (1 if bitstring[-1] == "0" else -1) * count
        for bitstring, count in counts.items()
    ) / total

    return float(expectation)


def generate_circuit(rng, n_gates):
    """Generate one circuit with a mixture of fixed and random angles."""
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(1)
    gates = []

    for _ in range(n_gates):
        gate = rng.choice(("RX", "RY", "RZ", "H"))

        if gate == "H":
            circuit.h(0)
            gates.append({"gate": "H"})
        else:
            theta = (
                rng.choice(BASE_ANGLES)
                if rng.random() < 0.65
                else rng.uniform(0, 2 * math.pi)
            )

            getattr(circuit, gate.lower())(theta, 0)
            gates.append({"gate": gate, "theta": theta})

    circuit_spec = {
        "n_qubits": 1,
        "initial_state": "0",
        "observable": "Z",
        "gates": gates,
    }

    return circuit, circuit_spec


def circuit_id_from_spec(circuit_spec):
    """Stable ID based on the canonical circuit specification."""
    canonical = json.dumps(
        circuit_spec, sort_keys=True, separators=(",", ":")
    )
    import hashlib
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def generate_dataset(
    n_circuits=5000,
    seed=42,
    shots_options=SHOTS_OPTIONS,
    noise_levels=NOISE_LEVELS,
):
    """Generate unique circuits and all configured experiments."""
    if n_circuits < 1:
        raise ValueError("n_circuits must be positive")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    seen = set()
    records = []

    while len(seen) < n_circuits:
        n_gates = rng.randint(1, 10)
        circuit, spec = generate_circuit(rng, n_gates)
        circuit_id = circuit_id_from_spec(spec)

        if circuit_id in seen:
            continue
        seen.add(circuit_id)

        ideal = ideal_expectation(circuit)
        gates = spec["gates"]
        rotation_angles = [
            gate["theta"] for gate in gates if "theta" in gate
        ]

        circuit_type = (
            "fixed_angle"
            if any(
                math.isclose(
                    angle / base,
                    round(angle / base),
                    abs_tol=1e-8,
                )
                for angle in rotation_angles
                for base in BASE_ANGLES
            )
            else "mixed_rotation"
        )

        for noise_probability in noise_levels:
            for shots in shots_options:
                experiment_seed = rng.randrange(0, 2**31)

                noisy = noisy_expectation(
                    circuit,
                    shots=shots,
                    noise_probability=noise_probability,
                    seed=experiment_seed,
                )

                records.append({
                    "record_id": (
                        f"{circuit_id}_{noise_probability}_{shots}"
                    ),
                    "circuit_id": circuit_id,
                    "circuit_type": circuit_type,
                    "n_qubits": 1,
                    "circuit_depth": circuit.depth(),
                    "single_qubit_gate_count": len(gates),
                    "cx_count": 0,
                    "mean_rotation_angle": (
                        sum(rotation_angles) / len(rotation_angles)
                        if rotation_angles else 0.0
                    ),
                    "observable": "Z",
                    "circuit_params": json.dumps(spec, sort_keys=True),
                    "noise_category": "depolarizing",
                    "noise_params": json.dumps({
                        "probability": noise_probability
                    }),
                    "backend_name": "AerSimulator",
                    "shots": shots,
                    "seed": experiment_seed,
                    "ideal_expval": ideal,
                    "noisy_expval": noisy,
                    "delta": ideal - noisy,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        if len(seen) % 100 == 0:
            print(f"Generated {len(seen)}/{n_circuits} circuits")

    fieldnames = list(records[0].keys())

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nUnique circuits: {len(seen)}")
    print(f"Experiment rows: {len(records)}")
    print(f"CSV saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    generate_dataset(n_circuits=5000, seed=42)
