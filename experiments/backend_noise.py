"""Step 2: 2-qubit entangled circuit evaluated under simulated IBM backend noise.

Compares exact statevector ground truth, shot-noise sampling, and 
simulated fake-backend noise on observable <Z0> and two-qubit basis states.

Usage:
    uv run experiments/step2_backend_noise.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Pauli, Statevector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeLagosV2


def build_bell_like_circuit(theta: float) -> tuple[QuantumCircuit, float]:
    """Construct state preparation Ry(theta) -> CX with exact ground truth."""
    qc = QuantumCircuit(2, 2)
    qc.ry(theta, 0)
    qc.cx(0, 1)

    # Exact expectation value of Z on qubit 0 (Pauli 'IZ' in Qiskit little-endian ordering)
    state = Statevector.from_instruction(qc)
    ideal_z0 = float(state.expectation_value(Pauli("IZ")).real)

    qc.measure([0, 1], [0, 1])
    return qc, ideal_z0


def compute_z0_from_counts(counts: dict[str, int], shots: int) -> float:
    """Calculate <Z0> expectation value: P(q0=0) - P(q0=1)."""
    # In Qiskit bitstrings, qubit 0 is the rightmost character: bitstring[-1]
    ones_on_q0 = sum(count for bitstring, count in counts.items() if bitstring[-1] == "1")
    return 1.0 - 2.0 * (ones_on_q0 / shots)


def run_experiment(theta: float = np.pi / 3, shots: int = 1024, seed: int = 42):
    backend = FakeLagosV2()
    backend_name = backend.name

    qc, exact_ideal_z0 = build_bell_like_circuit(theta)

    # 1. Ideal sampled run (AerSimulator without noise model)
    ideal_sim = AerSimulator()
    transpiled_ideal_qc = transpile(qc, ideal_sim, optimization_level=0)
    ideal_result = ideal_sim.run(transpiled_ideal_qc, shots=shots, seed_simulator=seed).result()
    ideal_counts = ideal_result.get_counts()
    sampled_ideal_z0 = compute_z0_from_counts(ideal_counts, shots)

    # 2. Simulated backend noise run
    # AerSimulator.from_backend configures basis gates, coupling map, and noise model
    noisy_sim = AerSimulator.from_backend(backend)
    transpiled_noisy_qc = transpile(qc, backend=backend, optimization_level=0)
    noisy_result = noisy_sim.run(transpiled_noisy_qc, shots=shots, seed_simulator=seed).result()
    noisy_counts = noisy_result.get_counts()
    noisy_z0 = compute_z0_from_counts(noisy_counts, shots)

    # Compute errors relative to analytical ground truth
    shot_noise_error = exact_ideal_z0 - sampled_ideal_z0
    simulated_backend_error = exact_ideal_z0 - noisy_z0

    # Console summary
    print("=" * 60)
    print(f"  2-QUBIT EXPERIMENT: EXACT VS SAMPLED VS {backend_name.upper()}")
    print("=" * 60)
    print(f"Target state parameter theta:     {theta:.4f} rad (pi/3)")
    print(f"Shots per circuit:                {shots}")
    print("-" * 60)
    print(f"1. Exact ideal <Z0>:              {exact_ideal_z0:+.4f} (ground truth)")
    print(f"2. Ideal sampled <Z0>:            {sampled_ideal_z0:+.4f} (shot noise: {shot_noise_error:+.4f})")
    print(f"3. Noisy sampled <Z0>:            {noisy_z0:+.4f} (simulated error: {simulated_backend_error:+.4f})")
    print("-" * 60)
    print(f"Ideal basis state counts:         {ideal_counts}")
    print(f"Noisy basis state counts:         {noisy_counts}")
    print("=" * 60)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # Subplot 1: Full 2-qubit state distributions
    basis_states = ["00", "01", "10", "11"]
    ideal_probs = [ideal_counts.get(b, 0) / shots for b in basis_states]
    noisy_probs = [noisy_counts.get(b, 0) / shots for b in basis_states]

    x = np.arange(len(basis_states))
    width = 0.35

    axes[0].bar(x - width / 2, ideal_probs, width, label="Ideal sampled", color="steelblue")
    axes[0].bar(x + width / 2, noisy_probs, width, label=f"Noisy ({backend_name})", color="indianred")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(basis_states)
    axes[0].set_xlabel("Basis state |q1 q0>")
    axes[0].set_ylabel("Observed probability")
    axes[0].set_title("Two-qubit outcome distributions")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Subplot 2: Target observable <Z0> comparison
    labels = ["Exact ideal", "Ideal sampled", "Noisy sampled"]
    values = [exact_ideal_z0, sampled_ideal_z0, noisy_z0]
    colors = ["gray", "steelblue", "indianred"]

    axes[1].bar(labels, values, color=colors, width=0.55)
    axes[1].axhline(
        exact_ideal_z0,
        color="black",
        linestyle="--",
        label=f"Exact ideal = {exact_ideal_z0:.3f}",
    )
    axes[1].set_ylabel("<Z0>")
    axes[1].set_title("Target observable")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = Path("results/backend_noise_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    run_experiment()