
import math
import random

from qiskit import QuantumCircuit


BASE_ANGLES = (
    math.pi / 6,
    math.pi / 4,
    math.pi / 3,
    math.pi / 2,
)


def generate_single_qubit_circuit(seed: int, n_gates: int = 5):
    """Generate a reproducible one-qubit circuit."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if n_gates < 1:
        raise ValueError("n_gates must be >= 1")

    rng = random.Random(seed)
    circuit = QuantumCircuit(1)

    gate_names = ("RX", "RY", "RZ", "H")
    gates = []

    for _ in range(n_gates):
        gate = rng.choice(gate_names)

        if gate == "H":
            circuit.h(0)
            gates.append({"gate": "H"})
            continue

        # Частина кутів — точно базові, частина — випадкові.
        if rng.random() < 0.65:
            theta = rng.choice(BASE_ANGLES)
        else:
            theta = rng.uniform(0.0, 2.0 * math.pi)

        if gate == "RX":
            circuit.rx(theta, 0)
        elif gate == "RY":
            circuit.ry(theta, 0)
        else:
            circuit.rz(theta, 0)

        gates.append({
            "gate": gate,
            "theta": theta,
        })

    metadata = {
        "seed": seed,
        "n_qubits": 1,
        "circuit_depth": circuit.depth(),
        "single_qubit_gate_count": len(gates),
        "cx_count": 0,
        "observable": "Z",
        "gates": gates,
    }

    return circuit, metadata
