
"""Generate parameterized multi-qubit quantum circuits."""

import math
import random

from qiskit import QuantumCircuit


CIRCUIT_FAMILIES = (
    "ghz",
    "bell_pairs",
    "rotation_cx_ladder",
    "rotation_cx_ring",
    "random_rotation_cx",
    "qft_like",
    "bv_like",
    "cx_dense_random",
)


def generate_multiqubit_circuit(
    seed: int,
    n_qubits: int,
    family: str,
    n_layers: int = 1,
):
    """Return a generated circuit and metadata describing its gates."""

    if n_qubits < 2:
        raise ValueError("n_qubits must be >= 2")
    if family not in CIRCUIT_FAMILIES:
        raise ValueError(f"Unknown circuit family: {family}")
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")

    rng = random.Random(seed)
    circuit = QuantumCircuit(n_qubits)
    gates = []

    def add_gate(name, qubits, angle=None):
        if name == "h":
            circuit.h(qubits[0])
        elif name == "rx":
            circuit.rx(angle, qubits[0])
        elif name == "ry":
            circuit.ry(angle, qubits[0])
        elif name == "rz":
            circuit.rz(angle, qubits[0])
        elif name == "cx":
            circuit.cx(qubits[0], qubits[1])
        else:
            raise ValueError(f"Unsupported gate: {name}")

        item = {"gate": name, "qubits": list(qubits)}
        if angle is not None:
            item["angle"] = float(angle)
        gates.append(item)

    def add_random_rotations():
        for q in range(n_qubits):
            add_gate("rx", [q], rng.uniform(-math.pi, math.pi))
            add_gate("ry", [q], rng.uniform(-math.pi, math.pi))
            add_gate("rz", [q], rng.uniform(-math.pi, math.pi))

    if family == "ghz":
        add_random_rotations()
        add_gate("h", [0])
        for q in range(n_qubits - 1):
            add_gate("cx", [q, q + 1])

    elif family == "bell_pairs":
        add_random_rotations()
        for q in range(0, n_qubits - 1, 2):
            add_gate("h", [q])
            add_gate("cx", [q, q + 1])

    elif family == "rotation_cx_ladder":
        for _ in range(n_layers):
            add_random_rotations()
            for q in range(n_qubits - 1):
                add_gate("cx", [q, q + 1])

    elif family == "rotation_cx_ring":
        for _ in range(n_layers):
            add_random_rotations()
            for q in range(n_qubits - 1):
                add_gate("cx", [q, q + 1])
            add_gate("cx", [n_qubits - 1, 0])

    elif family in ("random_rotation_cx", "cx_dense_random"):
        for _ in range(n_layers):
            add_random_rotations()

            pairs = [
                (control, target)
                for control in range(n_qubits)
                for target in range(n_qubits)
                if control != target
            ]
            rng.shuffle(pairs)

            if family == "random_rotation_cx":
                min_cx = n_qubits - 1
                max_cx = min(len(pairs), 2 * n_qubits)
            else:
                min_cx = n_qubits
                max_cx = len(pairs)

            n_cx = rng.randint(min_cx, max_cx)
            for control, target in pairs[:n_cx]:
                add_gate("cx", [control, target])

    elif family == "qft_like":
        # QFT-inspired pattern, not a full exact QFT implementation.
        add_random_rotations()
        for target in range(n_qubits):
            add_gate("h", [target])
            for control in range(target + 1, n_qubits):
                add_gate("cx", [control, target])

    elif family == "bv_like":
        # Bernstein–Vazirani-inspired pattern.
        add_random_rotations()
        secret = [rng.randint(0, 1) for _ in range(n_qubits - 1)]
        add_gate("h", [n_qubits - 1])

        for q, bit in enumerate(secret):
            if bit:
                add_gate("cx", [q, n_qubits - 1])

        for q in range(n_qubits - 1):
            add_gate("h", [q])

    single_qubit_gate_count = sum(
        gate["gate"] != "cx" for gate in gates
    )
    cx_count = sum(gate["gate"] == "cx" for gate in gates)
    rotation_angles = [
        gate["angle"] for gate in gates if "angle" in gate
    ]

    metadata = {
        "seed": seed,
        "n_qubits": n_qubits,
        "family": family,
        "n_layers": n_layers,
        "gates": gates,
        "circuit_depth": circuit.depth(),
        "single_qubit_gate_count": single_qubit_gate_count,
        "cx_count": cx_count,
        "mean_rotation_angle": (
            sum(abs(angle) for angle in rotation_angles)
            / len(rotation_angles)
            if rotation_angles else 0.0
        ),
    }

    return circuit, metadata
