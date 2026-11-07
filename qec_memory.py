"""
Quantum memory experiment: bit-flip repetition code under circuit-level noise.

- Qiskit circuit: d data qubits, d-1 ancillas, `rounds` of syndrome extraction
  with mid-circuit measurement + reset, then final data readout.
- Noise (Qiskit Aer): depolarizing on CX and idle, reset flip, readout flip.
- Decoder: minimum-weight perfect matching (PyMatching) on the space-time
  detector graph.
- Output: logical error rate vs physical error rate for several distances
  (threshold plot) -> qec_memory_threshold.png

Install:  pip install qiskit qiskit-aer pymatching matplotlib numpy
Run:      python qec_memory.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pymatching
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error, ReadoutError

# ----------------------------- settings -----------------------------------
DISTANCES = [3, 5, 7]                      # odd distances only
P_VALUES = np.logspace(-3, -1.5, 8)        # physical error rates
SHOTS = 5000
# --------------------------------------------------------------------------


def build_circuit(d, rounds):
    data = QuantumRegister(d, "data")
    anc = QuantumRegister(d - 1, "anc")
    syn = [ClassicalRegister(d - 1, f"s{r}") for r in range(rounds)]
    fin = ClassicalRegister(d, "final")
    qc = QuantumCircuit(data, anc, *syn, fin)

    # Logical |0> = |00...0>. Each round measures Z_i Z_{i+1} parities.
    for r in range(rounds):
        for i in range(d - 1):
            qc.cx(data[i], anc[i])
            qc.cx(data[i + 1], anc[i])
        qc.measure(anc, syn[r])
        qc.reset(anc)
        qc.id(data)  # idle noise on data qubits while ancillas are read out
        qc.barrier()
    qc.measure(data, fin)
    return qc


def make_noise_model(p):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p, 2), ["cx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p, 1), ["id"])
    nm.add_all_qubit_quantum_error(pauli_error([("X", p), ("I", 1 - p)]), ["reset"])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
    return nm


def build_matcher(d, rounds):
    """Space-time matching graph. Node (t, i) = detector for check i at time t."""
    n = d - 1
    node = lambda t, i: t * n + i
    m = pymatching.Matching()
    for t in range(rounds + 1):
        for i in range(n - 1):  # data-qubit errors between neighbouring checks
            m.add_edge(node(t, i), node(t, i + 1))
        m.add_boundary_edge(node(t, 0), fault_ids={0})  # data qubit 0 = logical readout
        m.add_boundary_edge(node(t, n - 1))             # data qubit d-1
        if t < rounds:
            for i in range(n):  # measurement errors
                m.add_edge(node(t, i), node(t + 1, i))
    return m


def parse_key(key, rounds):
    """Aer key: 'final s{R-1} ... s0' (space separated, bit 0 rightmost)."""
    parts = key.split()[::-1]  # -> s0, s1, ..., final
    bits = [np.array([int(c) for c in p[::-1]], dtype=np.uint8) for p in parts]
    return bits[:rounds], bits[rounds]


def detection_events(syn, fin):
    prev = np.zeros_like(syn[0])
    det = []
    for s in syn:
        det.append(s ^ prev)
        prev = s
    det.append((fin[:-1] ^ fin[1:]) ^ prev)  # final syndrome from data readout
    return np.concatenate(det)


def logical_error_rate(d, p, shots):
    rounds = d
    sim = AerSimulator(method="stabilizer", noise_model=make_noise_model(p))
    qc = transpile(build_circuit(d, rounds), sim, optimization_level=0)
    counts = sim.run(qc, shots=shots).result().get_counts()

    keys = list(counts)
    dets, observed = [], []
    for k in keys:
        syn, fin = parse_key(k, rounds)
        dets.append(detection_events(syn, fin))
        observed.append(fin[0])  # logical Z readout = data qubit 0

    predicted = build_matcher(d, rounds).decode_batch(np.array(dets, dtype=np.uint8))[:, 0]
    wrong = predicted != np.array(observed)
    return sum(counts[k] for k, w in zip(keys, wrong) if w) / shots


def main():
    results = {d: [] for d in DISTANCES}
    print(f"{'d':>3} {'p':>9} {'logical error':>14}")
    for d in DISTANCES:
        for p in P_VALUES:
            pl = logical_error_rate(d, p, SHOTS)
            results[d].append(pl)
            print(f"{d:>3} {p:>9.4f} {pl:>14.4f}")

    plt.figure(figsize=(6, 4.5))
    for d in DISTANCES:
        plt.loglog(P_VALUES, np.maximum(results[d], 0.5 / SHOTS), "o-", label=f"d = {d}")
    plt.xlabel("physical error rate p")
    plt.ylabel("logical error rate")
    plt.title("Repetition-code memory, circuit-level noise")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("qec_memory_threshold.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
