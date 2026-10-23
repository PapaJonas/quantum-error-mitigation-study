"""Collect a reproducible synthetic dataset for Z-on-qubit-0 error mitigation."""

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, amplitude_damping_error, depolarizing_error
from qiskit_ibm_runtime.fake_provider import FakeLagosV2, FakeNairobiV2


OUTPUT = Path(__file__).resolve().parents[1] / "data"
QUBITS = (2, 4, 6, 8)
LAYERS = (2, 4, 8)
SEEDS = range(20)
SHOTS = 1024
SPLIT_SEED = 20261019
SIMULATOR_SEED = 42
DEPOLARIZING = (("low", 0.002, 0.012), ("high", 0.008, 0.048))
DAMPING = (("low", 0.003, 0.015), ("high", 0.015, 0.060))
READOUT = (("low", 0.01), ("high", 0.04))


def make_circuit(qubits, layers, seed):
    rng = random.Random(seed + 100 * qubits + 10_000 * layers)
    circuit = QuantumCircuit(qubits)
    ry_angles, rz_angles = [], []
    for layer in range(layers):
        for qubit in range(qubits):
            ry, rz = rng.uniform(-math.pi, math.pi), rng.uniform(-math.pi, math.pi)
            circuit.ry(ry, qubit)
            circuit.rz(rz, qubit)
            ry_angles.append(abs(ry))
            rz_angles.append(abs(rz))
        for qubit in range(layer % 2, qubits - 1, 2):
            circuit.cx(qubit, qubit + 1)
    features = {
        "ry_abs_mean": sum(ry_angles) / len(ry_angles),
        "rz_abs_mean": sum(rz_angles) / len(rz_angles),
        "cx_count": circuit.count_ops().get("cx", 0),
        "circuit_depth": circuit.depth(),
    }
    return circuit, features


def profiles():
    for level, one, two in DEPOLARIZING:
        noise = NoiseModel()
        noise.add_all_qubit_quantum_error(depolarizing_error(one, 1), ("ry", "rz"))
        noise.add_all_qubit_quantum_error(depolarizing_error(two, 2), "cx")
        yield f"depolarizing_{level}", AerSimulator(noise_model=noise), max(QUBITS)
    for level, one, two in DAMPING:
        noise = NoiseModel()
        noise.add_all_qubit_quantum_error(amplitude_damping_error(one), ("ry", "rz"))
        error = amplitude_damping_error(two).tensor(amplitude_damping_error(two))
        noise.add_all_qubit_quantum_error(error, "cx")
        yield f"amplitude_damping_{level}", AerSimulator(noise_model=noise), max(QUBITS)
    for level, probability in READOUT:
        noise = NoiseModel()
        noise.add_all_qubit_readout_error(ReadoutError([[1 - probability, probability],
                                                        [probability, 1 - probability]]))
        yield f"readout_flip_{level}", AerSimulator(noise_model=noise), max(QUBITS)
    for backend_class in (FakeLagosV2, FakeNairobiV2):
        backend = backend_class()
        yield backend.name, AerSimulator.from_backend(backend), backend.num_qubits


def save_snapshot_calibrations():
    rows, snapshots = [], {}
    for backend_class in (FakeLagosV2, FakeNairobiV2):
        backend = backend_class()
        properties = backend.properties()
        snapshots[backend.name] = properties.last_update_date.isoformat()
        for qubit in range(backend.num_qubits):
            for name in ("T1", "T2", "readout_error"):
                rows.append((backend.name, snapshots[backend.name], name, str(qubit),
                             properties.qubit_property(qubit)[name][0]))
            for gate in ("x", "sx"):
                rows.append((backend.name, snapshots[backend.name], f"{gate}_error", str(qubit),
                             properties.gate_error(gate, (qubit,))))
        for gate in properties.gates:
            if gate.gate == "cx":
                rows.append((backend.name, snapshots[backend.name], "cx_error",
                             "-".join(map(str, gate.qubits)), properties.gate_error("cx", gate.qubits)))
    path = OUTPUT / "snapshot_calibrations.csv"
    with path.open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(("backend", "snapshot_time", "metric", "qubits", "value"))
        writer.writerows(sorted(rows))
    return snapshots, path


def main():
    shuffled = list(SEEDS)
    random.Random(SPLIT_SEED).shuffle(shuffled)
    splits = {seed: split for split, group in (
        ("train", shuffled[:16]), ("validation", shuffled[16:18]), ("test", shuffled[18:]))
        for seed in group}
    circuits = []
    for qubits in QUBITS:
        for layers in LAYERS:
            for seed in SEEDS:
                circuit, features = make_circuit(qubits, layers, seed)
                observable = SparsePauliOp("I" * (qubits - 1) + "Z")
                ideal = float(Statevector.from_instruction(circuit).expectation_value(observable).real)
                measured = circuit.copy()
                measured.measure_all()
                circuits.append((qubits, layers, seed, ideal, features, measured))

    rows = []
    for profile, simulator, capacity in profiles():
        selected = [item for item in circuits if item[0] <= capacity]
        compiled = transpile([item[-1] for item in selected], simulator,
                             optimization_level=0, seed_transpiler=SIMULATOR_SEED)
        result = simulator.run(compiled, shots=SHOTS, seed_simulator=SIMULATOR_SEED).result()
        for index, (qubits, layers, seed, ideal, features, _) in enumerate(selected):
            counts = result.get_counts(index)
            assert sum(counts.values()) == SHOTS
            # Qiskit displays logical qubit 0 at the right end of each count string.
            z0_ones = sum(count for bits, count in counts.items() if bits[-1] == "1")
            noisy = 1 - 2 * z0_ones / SHOTS
            rows.append({
                "circuit_id": f"q{qubits}_l{layers}_s{seed:02d}",
                "split": splits[seed], "qubits": qubits, "layers": layers, "seed": seed,
                **features, "noise_profile": profile, "shots": SHOTS, "z0_ones": z0_ones,
                "ideal_z0": ideal, "noisy_z0": noisy, "delta": ideal - noisy,
            })

    assert len(rows) == 1800 and len({(row["circuit_id"], row["noise_profile"]) for row in rows}) == len(rows)
    assert {row["split"] for row in rows} == {"train", "validation", "test"}
    assert all(-1.000001 <= row[key] <= 1.000001 for row in rows for key in ("ideal_z0", "noisy_z0"))
    rows.sort(key=lambda row: (row["qubits"], row["layers"], row["seed"], row["noise_profile"]))

    OUTPUT.mkdir(exist_ok=True)
    dataset = OUTPUT / "synthetic_expectations.csv"
    with dataset.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{value:.10f}" if isinstance(value, float) else value
                             for key, value in row.items()})

    groups = defaultdict(list)
    for row in rows:
        groups[row["noise_profile"], row["split"]].append(abs(row["delta"]))
    with (OUTPUT / "summary.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(("noise_profile", "split", "rows", "raw_mae"))
        for (profile, split), errors in sorted(groups.items()):
            writer.writerow((profile, split, len(errors), f"{sum(errors) / len(errors):.6f}"))

    snapshots, calibrations = save_snapshot_calibrations()
    manifest = {
        "source": "local Qiskit Aer simulation and packaged fake-backend snapshots; no hardware runs",
        "observable": "Z on logical qubit 0",
        "qubits": QUBITS, "layers": LAYERS, "seeds": list(SEEDS), "shots": SHOTS,
        "split_seed": SPLIT_SEED, "split_seeds": {
            split: sorted(seed for seed, assigned in splits.items() if assigned == split)
            for split in ("train", "validation", "test")},
        "simulator_seed": SIMULATOR_SEED,
        "noise_parameters": {
            "depolarizing": {level: {"single_gate": one, "cx": two}
                             for level, one, two in DEPOLARIZING},
            "amplitude_damping": {level: {"single_gate": one, "cx_per_qubit": two}
                                  for level, one, two in DAMPING},
            "readout_flip": {level: probability for level, probability in READOUT},
        },
        "versions": {package: version(package) for package in
                     ("qiskit", "qiskit-aer", "qiskit-ibm-runtime")},
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "snapshot_times": snapshots,
        "snapshot_calibrations_sha256": hashlib.sha256(calibrations.read_bytes()).hexdigest(),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(rows)} synthetic rows and summary to {OUTPUT}")


if __name__ == "__main__":
    main()
