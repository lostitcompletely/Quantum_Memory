"""
Figures for the surface code memory project.

Produces:
  qiskit_surface_code_round.png  Qiskit circuit for one syndrome-extraction round
                                 of the 3x3 (9 data qubit) surface code
  transmon_noise_summary.png     Simulated flux pulse, |11> <-> |02> population
                                 swap during the CZ, DRAG pulse, and the CZ
                                 Pauli error probabilities used by the memory
                                 experiment
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

import superconduct as sc

X_CHECKS = [(1, 2, 4, 5), (3, 4, 6, 7), (0, 1), (7, 8)]
Z_CHECKS = [(0, 1, 3, 4), (4, 5, 7, 8), (2, 5), (3, 6)]


def surface_code_round():
    data = QuantumRegister(9, "data")
    xa = QuantumRegister(4, "X-check")
    za = QuantumRegister(4, "Z-check")
    syn = ClassicalRegister(8, "syndrome")
    qc = QuantumCircuit(data, xa, za, syn)

    for i, chk in enumerate(X_CHECKS):
        qc.h(xa[i])
        for q in chk:
            qc.cx(xa[i], data[q])
        qc.h(xa[i])
        qc.measure(xa[i], syn[i])
    qc.barrier()
    for i, chk in enumerate(Z_CHECKS):
        for q in chk:
            qc.cx(data[q], za[i])
        qc.measure(za[i], syn[4 + i])

    fig = qc.draw("mpl", fold=40, style="iqp")
    fig.savefig("qiskit_surface_code_round.png", dpi=200, bbox_inches="tight")
    print(f"Qiskit circuit: {qc.num_qubits} qubits, depth {qc.depth()}, gates {dict(qc.count_ops())}")
    return qc


def transmon_figure():
    pulse = sc.tune_cz()
    deltas, dt, total = sc.flux_pulse(*pulse)
    t = (np.arange(len(deltas)) + 1) * dt

    psi = np.zeros(9, complex)
    psi[4] = 1.0                                   # |11>
    p11, p02 = [], []
    for d in deltas:
        psi = expm(-1j * sc.h_two(d) * dt) @ psi
        pop = np.abs(psi) ** 2
        p11.append(pop[4])
        p02.append(pop[2])                         # |02>

    T, step = 30.0, 0.1
    tt = (np.arange(int(round(T / step))) + 0.5) * step
    amp, alpha = np.pi / T, sc.TWO_PI * sc.ALPHA
    ox = amp * np.sin(np.pi * tt / T) ** 2
    oy = -(amp * (np.pi / T) * np.sin(2 * np.pi * tt / T)) / alpha

    p2 = sc.cz_noise(pulse, sc.T1, sc.T2)
    labels = ["IXYZ"[a] + "IXYZ"[b] for a in range(4) for b in range(4)][1:]

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))

    ax[0, 0].plot(t, sc.F_A + deltas, lw=2)
    ax[0, 0].axhline(sc.F_B + sc.ALPHA, color="gray", ls="--", lw=1, label="|11> <-> |02> resonance")
    ax[0, 0].set(title="Flux pulse on tunable transmon", xlabel="time (ns)", ylabel="qubit A frequency (GHz)")
    ax[0, 0].legend()

    ax[0, 1].plot(t, p11, lw=2, label="|11>")
    ax[0, 1].plot(t, p02, lw=2, label="|02> (leakage state)")
    ax[0, 1].set(title="CZ gate: population swap and return", xlabel="time (ns)", ylabel="population")
    ax[0, 1].legend()

    ax[1, 0].plot(tt, ox / sc.TWO_PI * 1e3, lw=2, label="in-phase")
    ax[1, 0].plot(tt, oy / sc.TWO_PI * 1e3, lw=2, label="DRAG quadrature")
    ax[1, 0].set(title="30 ns pi/2 pulse (DRAG)", xlabel="time (ns)", ylabel="drive (MHz)")
    ax[1, 0].legend()

    ax[1, 1].bar(range(15), p2[1:])
    ax[1, 1].set_xticks(range(15))
    ax[1, 1].set_xticklabels(labels, rotation=90)
    ax[1, 1].set_yscale("log")
    ax[1, 1].set(title=f"Simulated CZ Pauli errors (total {100 * (1 - p2[0]):.2f}%)",
                 xlabel="Pauli error", ylabel="probability")

    fig.suptitle("Transmon simulation to surface code noise", fontsize=14)
    fig.tight_layout()
    fig.savefig("transmon_noise_summary.png", dpi=200)


if __name__ == "__main__":
    surface_code_round()
    transmon_figure()
    plt.show()