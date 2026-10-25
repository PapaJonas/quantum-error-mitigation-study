"""Run a small, deliberately limited set of the test-seed circuits on real IBM
Quantum hardware and compare against the exact Statevector ground truth.

Scope is intentionally small (2-qubit circuits only, from the fixed test seeds
in data/manifest.json) to stay well inside a trial-account's queue/usage
budget. This does not feed the training data -- it is a separate, explicitly
labeled real-hardware check, per data/README.md's "Пізніше: дані реального
пристрою" protocol (only test seeds, full run provenance recorded).

Usage:
    uv run experiments/real_device_validation.py
"""

import csv
import json
import sys
from pathlib import Path

from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_data import make_circuit  # noqa: E402


OUTPUT = Path(__file__).resolve().parents[1] / "results" / "real_device"
QUBITS = 2
LAYERS = (2, 4, 8)
TEST_SEEDS = (10, 18)  # from data/manifest.json split_seeds.test
SHOTS = 1024
BACKEND_NAME = "ibm_fez"


def main():
    service = QiskitRuntimeService()
    backend = service.backend(BACKEND_NAME)
    properties = backend.properties()

    specs = [(QUBITS, layers, seed) for layers in LAYERS for seed in TEST_SEEDS]

    circuits, idealvals, features_list = [], [], []
    for qubits, layers, seed in specs:
        circuit, features = make_circuit(qubits, layers, seed)
        observable = SparsePauliOp("I" * (qubits - 1) + "Z")
        ideal = float(Statevector.from_instruction(circuit).expectation_value(observable).real)
        measured = circuit.copy()
        measured.measure_all()
        circuits.append(measured)
        idealvals.append(ideal)
        features_list.append(features)

    print(f"Prepared {len(circuits)} circuits (qubits={QUBITS}) for backend {BACKEND_NAME}.")
    print(f"Backend status: {backend.status()}")

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuits = pm.run(circuits)

    sampler = Sampler(mode=backend)
    job = sampler.run(isa_circuits, shots=SHOTS)
    print(f"Submitted job_id={job.job_id()}. Waiting for result...")
    result = job.result()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (qubits, layers, seed) in enumerate(specs):
        pub_result = result[index]
        counts = pub_result.data.meas.get_counts()
        total = sum(counts.values())
        assert total == SHOTS
        z0_ones = sum(count for bits, count in counts.items() if bits[-1] == "1")
        noisy = 1 - 2 * z0_ones / SHOTS
        ideal = idealvals[index]
        rows.append({
            "circuit_id": f"q{qubits}_l{layers}_s{seed:02d}",
            "qubits": qubits, "layers": layers, "seed": seed,
            **features_list[index],
            "backend": BACKEND_NAME,
            "backend_version": properties.backend_version if properties else None,
            "calibration_time": properties.last_update_date.isoformat() if properties else None,
            "job_id": job.job_id(),
            "shots": SHOTS,
            "z0_ones": z0_ones,
            "ideal_z0": ideal,
            "noisy_z0": noisy,
            "delta": ideal - noisy,
        })

    csv_path = OUTPUT / "real_device_test_results.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "backend": BACKEND_NAME,
        "job_id": job.job_id(),
        "qubits": QUBITS,
        "layers": list(LAYERS),
        "test_seeds": list(TEST_SEEDS),
        "shots": SHOTS,
        "n_circuits": len(rows),
    }
    with (OUTPUT / "manifest.json").open("w") as file:
        json.dump(manifest, file, indent=2)

    print(f"\nSaved {len(rows)} real-device rows to {csv_path}")
    for row in rows:
        print(f"  {row['circuit_id']}: ideal={row['ideal_z0']:+.4f} noisy={row['noisy_z0']:+.4f} "
              f"delta={row['delta']:+.4f}")


if __name__ == "__main__":
    main()
