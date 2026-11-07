"""
Surface code quantum memory with superconducting-style noise (Stim + PyMatching).

Stores one logical bit in a rotated surface code, runs noisy rounds of error
checking, decodes, and reports how often the stored bit is lost for 3x3, 5x5
and 7x7 grids of data qubits. Saves a plot to qec_surface_code.png.

Noise uses the same device numbers as the repetition-code file (T1/T2
relaxation over the real round time, CNOT, readout and reset errors).
"""

import numpy as np
import matplotlib.pyplot as plt
import stim
import pymatching

# Fixed device model (nominal superconducting qubit values)
T1, T2 = 150e-6, 100e-6
T_CX, T_MEAS, T_RESET = 300e-9, 800e-9, 400e-9
P_CX, P_READOUT, P_RESET = 5e-3, 1.5e-2, 5e-3
T_ROUND = 4 * T_CX + T_MEAS + T_RESET   # 4 CNOT layers + readout + reset

CODE_SIZES = [3, 5, 7]                               # grid size of data qubits
NOISE_LEVELS = [0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.5]   # 1.0 = nominal device
SHOTS = 20000


def relaxation_prob(t, noise):
    """Total Pauli-twirled T1/T2 error probability over time t."""
    t1, t2 = T1 / noise, T2 / noise
    px = (1 - np.exp(-t / t1)) / 4
    pz = (1 - np.exp(-t / t2)) / 2 - px
    return 2 * px + pz


def build_circuit(size, noise):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=size,
        rounds=size,
        before_round_data_depolarization=min(relaxation_prob(T_ROUND, noise), 0.75),
        after_clifford_depolarization=min(P_CX * noise + relaxation_prob(T_CX, noise), 0.75),
        before_measure_flip_probability=min(P_READOUT * noise, 0.5),
        after_reset_flip_probability=min(P_RESET * noise, 0.5),
    )


def failure_rate(size, noise):
    circuit = build_circuit(size, noise)
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    detections, observables = circuit.compile_detector_sampler().sample(
        SHOTS, separate_observables=True
    )
    predicted = matcher.decode_batch(detections)
    return float(np.mean(np.any(predicted != observables, axis=1)))


def main():
    results = {n: [] for n in CODE_SIZES}
    for n in CODE_SIZES:
        for level in NOISE_LEVELS:
            results[n].append(failure_rate(n, level))
        print(f"{n}x{n} surface code: done")

    i_nom = NOISE_LEVELS.index(1.0)
    print("\nStored-bit failure rate at nominal device noise:")
    for n in CODE_SIZES:
        print(f"  {n}x{n} grid ({n * n} data qubits): {100 * results[n][i_nom]:.2f}%")

    plt.figure(figsize=(6.5, 4.5))
    for n in CODE_SIZES:
        plt.loglog(NOISE_LEVELS, np.maximum(results[n], 0.5 / SHOTS), "o-", label=f"{n}x{n} grid")
    plt.axvline(1.0, color="gray", ls="--", lw=1, label="nominal device")
    plt.xlabel("noise level (1 = nominal)")
    plt.ylabel("stored-bit failure rate")
    plt.title("Quantum memory (surface code)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("qec_surface_code.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
