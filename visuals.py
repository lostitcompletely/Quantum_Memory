"""
Figures for qec_memory_realistic.py (repetition-code quantum memory, device-style noise).

Put this file in the same folder as qec_memory_realistic.py and run:
    python visuals.py          (save it as visuals.py on that branch)
Needs: pip install qiskit qiskit-aer pylatexenc pymatching matplotlib numpy

Produces:
  qiskit_repetition_code.png   the actual Qiskit memory circuit (3 data qubits, 1 round)
  quantum_memory_results.png   failure rate vs noise level for each code size,
                               plus the nominal error budget behind it
  decoder_graph.png            the space-time graph PyMatching decodes on
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import qec_memory as qm

SIZES = qm.DISTANCES       # numbers of data qubits (odd)


def circuit_diagram():
    qc = qm.build_circuit(3, 1)
    fig = qc.draw("mpl", fold=45, style="iqp")
    fig.savefig("qiskit_repetition_code.png", dpi=200, bbox_inches="tight")
    print(f"Qiskit circuit: {qc.num_qubits} qubits, depth {qc.depth()}, gates {dict(qc.count_ops())}")


def error_budget():
    """Nominal error probabilities behind the noise model in qec_memory_realistic.py."""
    def relax(t):
        px = (1 - np.exp(-t / qm.T1)) / 4
        pz = (1 - np.exp(-t / qm.T2)) / 2 - px
        return 2 * px + pz
    return {
        "data idle\n(per round)": relax(qm.T_MEAS + qm.T_RESET),
        "CNOT": qm.P_CX + relax(qm.T_CX),
        "readout": qm.P_READOUT,
        "reset": qm.P_RESET,
    }


def results_figure():
    scales = list(qm.SCALES)
    results = {n: [qm.logical_error_rate(n, s, qm.SHOTS)[0] for s in scales] for n in SIZES}
    budget = error_budget()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})
    for n in SIZES:
        ax[0].loglog(scales, np.maximum(results[n], 0.5 / qm.SHOTS), "o-", lw=2,
                     label=f"{n} data qubits")
    ax[0].axvline(1.0, color="gray", ls="--", lw=1, label="nominal device")
    ax[0].set(title="Repetition-code quantum memory", xlabel="noise scale (1 = nominal)",
              ylabel="stored-bit failure rate")
    ax[0].legend()
    ax[0].grid(True, which="both", alpha=0.3)

    ax[1].bar(list(budget), [100 * v for v in budget.values()], color="tab:blue")
    ax[1].set(title="Nominal error budget", ylabel="error probability (%)")
    for i, v in enumerate(budget.values()):
        ax[1].text(i, 100 * v, f"{100 * v:.2f}%", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig("quantum_memory_results.png", dpi=200)


def decoder_graph():
    size, rounds = 5, 3
    n = size - 1
    fig, ax = plt.subplots(figsize=(7, 6))
    blue, orange, green = "tab:blue", "tab:orange", "tab:green"

    for t in range(rounds + 1):
        for i in range(n - 1):
            ax.plot([i, i + 1], [t, t], color=blue, lw=2)                      # data-qubit error
        ax.plot([0, -1], [t, t], color="gray", ls="--", lw=1.5)               # left boundary
        ax.plot([n - 1, n], [t, t], color="gray", ls="--", lw=1.5)            # right boundary
        if t < rounds:
            for i in range(n):
                ax.plot([i, i], [t, t + 1], color=orange, lw=2)                # measurement error
            for i in range(n - 1):
                ax.plot([i, i + 1], [t, t + 1], color=green, lw=2)             # between CNOT layers

    xs = [i for t in range(rounds + 1) for i in range(n)]
    ys = [t for t in range(rounds + 1) for i in range(n)]
    ax.scatter(xs, ys, s=120, color="white", edgecolor="black", zorder=3)
    ax.scatter([-1] * (rounds + 1) + [n] * (rounds + 1), list(range(rounds + 1)) * 2,
               s=90, marker="s", color="lightgray", edgecolor="black", zorder=3)

    ax.legend(handles=[
        Line2D([0], [0], color=blue, lw=2, label="data-qubit error"),
        Line2D([0], [0], color=orange, lw=2, label="measurement error"),
        Line2D([0], [0], color=green, lw=2, label="error between CNOT layers"),
        Line2D([0], [0], color="gray", ls="--", lw=1.5, label="boundary"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    ax.set(title="Decoder graph (5 data qubits, 3 rounds)", xlabel="check position", ylabel="round")
    ax.set_xticks(range(n))
    ax.set_yticks(range(rounds + 1))
    fig.tight_layout()
    fig.savefig("decoder_graph.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    circuit_diagram()
    decoder_graph()
    results_figure()
    plt.show()