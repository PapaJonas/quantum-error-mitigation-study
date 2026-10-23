
"""
1-qubit noise experiment: ideal vs finite-shot vs noisy.

Usage:
    uv run experiments/hello_noise.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


THETA = np.pi / 3
P_GATE = 0.05
SHOTS = 100_000
SEED = 67


def build_circuit(measure: bool = True) -> QuantumCircuit:
    """Prepare a single qubit with Rx(theta), optionally measure it."""
    qc = QuantumCircuit(1, 1)
    qc.rx(THETA, 0)

    if measure:
        qc.measure(0, 0)

    return qc


def estimate_z(counts: dict[str, int], shots: int) -> float:
    """Estimate <Z> from computational-basis measurement counts."""
    ones = counts.get("1", 0)
    return 1.0 - 2.0 * ones / shots


def run_experiment():
    # Exact ideal expectation value (no sampling, no noise).
    state = Statevector.from_instruction(build_circuit(measure=False))
    ideal_z = float(state.expectation_value(Pauli("Z")).real)

    # Ideal circuit sampled a finite number of times.
    ideal_simulator = AerSimulator()
    ideal_result = ideal_simulator.run(
        build_circuit(),
        shots=SHOTS,
        seed_simulator=SEED,
    ).result()
    ideal_counts = ideal_result.get_counts()
    sampled_ideal_z = estimate_z(ideal_counts, SHOTS)

    # Same circuit, now with depolarizing noise after Rx.
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(P_GATE, 1),
        ["rx"],
    )

    noisy_simulator = AerSimulator(noise_model=noise_model)
    noisy_result = noisy_simulator.run(
        build_circuit(),
        shots=SHOTS,
        seed_simulator=SEED,
    ).result()
    noisy_counts = noisy_result.get_counts()
    noisy_z = estimate_z(noisy_counts, SHOTS)

    # Theoretical expectation and bias under this noise model.
    theoretical_noisy_z = (1.0 - P_GATE) * ideal_z
    theoretical_bias = ideal_z - theoretical_noisy_z

    # Errors relative to the exact ideal result.
    ideal_sampling_error = ideal_z - sampled_ideal_z
    noisy_error = ideal_z - noisy_z

    print("=" * 58)
    print("1-QUBIT NOISE EXPERIMENT")
    print("=" * 58)
    print(f"Rotation angle:                 {THETA:.6f} rad")
    print(f"Depolarizing parameter:         {P_GATE:.4f}")
    print(f"Shots per circuit:              {SHOTS:,}")
    print("-" * 58)
    print(f"Exact ideal <Z>:                {ideal_z:+.6f}")
    print(f"Ideal finite-shot <Z>:          {sampled_ideal_z:+.6f}")
    print(f"Theoretical noisy <Z>:          {theoretical_noisy_z:+.6f}")
    print(f"Measured noisy <Z>:             {noisy_z:+.6f}")
    print("-" * 58)
    print(f"Ideal sampling error:           {ideal_sampling_error:+.6f}")
    print(f"Noisy error (ideal - measured): {noisy_error:+.6f}")
    print(f"Theoretical noise bias:         {theoretical_bias:+.6f}")
    print("=" * 58)

    # Plot the comparison of estimates and their errors.
    labels = [
        "Exact ideal",
        "Ideal sampled",
        "Noisy theoretical",
        "Noisy sampled",
    ]
    values = [
        ideal_z,
        sampled_ideal_z,
        theoretical_noisy_z,
        noisy_z,
    ]

    errors = [
        0.0,
        ideal_z - sampled_ideal_z,
        ideal_z - theoretical_noisy_z,
        ideal_z - noisy_z,
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].bar(labels, values)
    axes[0].set_title("Expectation value <Z>")
    axes[0].set_ylabel("<Z>")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(labels, errors)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].axhline(
        theoretical_bias,
        color="red",
        linestyle="--",
        label=f"Theoretical bias = {theoretical_bias:.4f}",
    )
    axes[1].set_title("Error relative to exact ideal value")
    axes[1].set_ylabel("Ideal <Z> - estimated <Z>")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()

    output_path = Path("results/error_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    run_experiment()
