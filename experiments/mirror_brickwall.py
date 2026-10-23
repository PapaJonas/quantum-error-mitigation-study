"""Preliminary brickwall mirror-circuit simulation; no model is trained here."""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, amplitude_damping_error, depolarizing_error
from qiskit_ibm_runtime.fake_provider import FakeLagosV2, FakeNairobiV2


SIZES = (2, 4, 6, 8)
LAYERS = (2, 4, 8)
SEEDS = range(5)
SHOTS = 1024
OUTPUT = Path(__file__).resolve().parents[1] / "results"


def mirror_circuit(qubits, layers, seed):
    rng = random.Random(seed)
    forward = QuantumCircuit(qubits)
    for layer in range(layers):
        for qubit in range(qubits):
            if rng.getrandbits(1):
                forward.h(qubit)
            else:
                forward.x(qubit)
        for qubit in range(layer % 2, qubits - 1, 2):
            forward.cx(qubit, qubit + 1)

    circuit = forward.copy()
    circuit.barrier()  # Keep the forward and inverse gates for the noisy simulation.
    circuit.compose(forward.inverse(), inplace=True)
    circuit.measure_all()
    return circuit


def noise_models():
    depolarizing = NoiseModel()
    for gate in ("h", "x"):
        depolarizing.add_all_qubit_quantum_error(depolarizing_error(0.005, 1), gate)
    depolarizing.add_all_qubit_quantum_error(depolarizing_error(0.03, 2), "cx")

    damping = NoiseModel()
    for gate in ("h", "x"):
        damping.add_all_qubit_quantum_error(amplitude_damping_error(0.01), gate)
    two_qubit_damping = amplitude_damping_error(0.03).tensor(amplitude_damping_error(0.03))
    damping.add_all_qubit_quantum_error(two_qubit_damping, "cx")

    readout = NoiseModel()
    readout.add_all_qubit_readout_error(ReadoutError([[0.98, 0.02], [0.02, 0.98]]))
    return {"ideal": None, "depolarizing": depolarizing, "amplitude_damping": damping, "readout_flip": readout}


def main():
    circuits = [(qubits, layers, seed, mirror_circuit(qubits, layers, seed))
                for qubits in SIZES for layers in LAYERS for seed in SEEDS]
    models = [(name, AerSimulator(noise_model=noise), max(SIZES))
              for name, noise in noise_models().items()]
    for backend_class in (FakeLagosV2, FakeNairobiV2):
        backend = backend_class()
        models.append((backend.name, AerSimulator.from_backend(backend), backend.num_qubits))
    rows = []
    for name, simulator, max_qubits in models:
        selected = [entry for entry in circuits if entry[0] <= max_qubits]
        compiled = [transpile(circuit, simulator, optimization_level=0, seed_transpiler=42)
                    for _, _, _, circuit in selected]
        assert all(circuit.count_ops().get("cx", 0) > 0 for circuit in compiled)
        result = simulator.run(compiled, shots=SHOTS, seed_simulator=42).result()
        for index, (qubits, layers, seed, _) in enumerate(selected):
            counts = result.get_counts(index)
            assert sum(counts.values()) == SHOTS
            all_zero = counts.get("0" * qubits, 0)
            if name == "ideal":
                assert all_zero == SHOTS, (qubits, layers, seed, counts)
            rows.append({"qubits": qubits, "layers": layers, "seed": seed, "noise": name,
                         "shots": SHOTS, "all_zero": all_zero,
                         "zero_bits": sum(bits.count("0") * count for bits, count in counts.items()),
                         "counts": json.dumps(counts, sort_keys=True)})

    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT / "mirror_runs.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    groups = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        group = groups[row["qubits"], row["layers"], row["noise"]]
        group[0] += row["shots"]
        group[1] += row["all_zero"]
        group[2] += row["zero_bits"]
    with (OUTPUT / "mirror_summary.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(("qubits", "layers", "noise", "shots", "p_all_zero", "mean_zeros", "zero_fraction"))
        for (qubits, layers, name), (shots, all_zero, zero_bits) in sorted(groups.items()):
            writer.writerow((qubits, layers, name, shots, f"{all_zero / shots:.6f}",
                             f"{zero_bits / shots:.6f}", f"{zero_bits / (shots * qubits):.6f}"))
    print(f"Saved {len(rows)} runs and {len(groups)} summaries to {OUTPUT}")


if __name__ == "__main__":
    main()
