"""
Repetition code (d data qubits, d-1 ancillas) simulated in Qiskit Aer, decoded
with PyMatching on a space-time graph that includes measurement errors and the
diagonal edges from the two-layer CNOT schedule.

Noise model (device-like, not a single error rate):
  - T1/T2 relaxation set by real gate/measurement/reset durations
  - data qubits idle for the full measure + reset time every round
  - CNOT: depolarizing error + relaxation during the 300 ns gate
  - readout assignment error, reset error
  - T1/T2 relaxation is Pauli-twirled by default (fast, uses the stabilizer
    simulator). Set EXACT_T1 = True for exact amplitude/phase damping
    (slow; use small distances only).

Sweep variable: SCALE. SCALE = 1 is the nominal device below; SCALE = 2 means
all gate/readout/reset errors are doubled and T1, T2 are halved.
"""

import numpy as np
import matplotlib.pyplot as plt
import pymatching
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error,
    ReadoutError,
)

# ------------------------- nominal device parameters ----------------------
T1 = 150e-6            # s
T2 = 100e-6            # s (must be <= 2*T1)
T_UNIT = 100e-9        # s, duration of one 'id' idle step
T_CX = 300e-9          # s
T_MEAS = 800e-9        # s
T_RESET = 400e-9       # s
P_CX = 5e-3            # CNOT depolarizing error
P_READOUT = 1.5e-2     # readout assignment error
P_RESET = 5e-3         # reset flip error

# ----------------------------- experiment settings ------------------------
DISTANCES = [3, 5, 7]                   # odd only
SCALES = np.logspace(-1, 0.4, 8)        # 0.1x ... 2.5x nominal noise
SHOTS = 5000
EXACT_T1 = False                        # True: exact thermal relaxation (slow)
# --------------------------------------------------------------------------

N_IDLE_MEAS = round((T_MEAS + T_RESET) / T_UNIT)
N_IDLE_CX = round(T_CX / T_UNIT)


def relaxation(t, scale):
    t1, t2 = T1 / scale, T2 / scale
    if EXACT_T1:
        return thermal_relaxation_error(t1, t2, t)
    # Pauli-twirled amplitude + phase damping
    px = (1 - np.exp(-t / t1)) / 4
    pz = (1 - np.exp(-t / t2)) / 2 - px
    return pauli_error([("X", px), ("Y", px), ("Z", pz), ("I", 1 - 2 * px - pz)])


def make_noise_model(scale):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(relaxation(T_UNIT, scale), ["id"])

    cx_relax = relaxation(T_CX, scale)
    cx_err = depolarizing_error(min(P_CX * scale, 0.75), 2).compose(cx_relax.tensor(cx_relax))
    nm.add_all_qubit_quantum_error(cx_err, ["cx"])

    p_reset = min(P_RESET * scale, 0.5)
    nm.add_all_qubit_quantum_error(pauli_error([("X", p_reset), ("I", 1 - p_reset)]), ["reset"])

    p_ro = min(P_READOUT * scale, 0.5)
    nm.add_all_qubit_readout_error(ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]]))
    return nm


def build_circuit(d, rounds):
    data = QuantumRegister(d, "data")
    anc = QuantumRegister(d - 1, "anc")
    syn = [ClassicalRegister(d - 1, f"s{r}") for r in range(rounds)]
    fin = ClassicalRegister(d, "final")
    qc = QuantumCircuit(data, anc, *syn, fin)

    for r in range(rounds):
        # CNOT layer 1: data i -> anc i
        for i in range(d - 1):
            qc.cx(data[i], anc[i])
        for _ in range(N_IDLE_CX):
            qc.id(data[d - 1])          # unused in this layer, idles
        # CNOT layer 2: data i+1 -> anc i
        for i in range(d - 1):
            qc.cx(data[i + 1], anc[i])
        for _ in range(N_IDLE_CX):
            qc.id(data[0])              # unused in this layer, idles
        qc.barrier()
        # Readout + reset: data qubits sit idle the whole time
        qc.measure(anc, syn[r])
        qc.reset(anc)
        for _ in range(N_IDLE_MEAS):
            qc.id(data)
        qc.barrier()
    qc.measure(data, fin)
    return qc


def build_matcher(d, rounds):
    """Detector (t, i) = check i at time t. Includes measurement-error (vertical)
    and CNOT-schedule (diagonal) edges."""
    n = d - 1
    node = lambda t, i: t * n + i
    m = pymatching.Matching()
    for t in range(rounds + 1):
        for i in range(n - 1):
            m.add_edge(node(t, i), node(t, i + 1))
        m.add_boundary_edge(node(t, 0), fault_ids={0})   # data qubit 0 = logical readout
        m.add_boundary_edge(node(t, n - 1))
        if t < rounds:
            for i in range(n):
                m.add_edge(node(t, i), node(t + 1, i))       # measurement error
            for i in range(n - 1):
                m.add_edge(node(t, i), node(t + 1, i + 1))   # data error between CNOT layers
    return m


def parse_key(key, rounds):
    parts = key.split()[::-1]  # s0, s1, ..., final
    bits = [np.array([int(c) for c in p[::-1]], dtype=np.uint8) for p in parts]
    return bits[:rounds], bits[rounds]


def detection_events(syn, fin):
    prev = np.zeros_like(syn[0])
    det = []
    for s in syn:
        det.append(s ^ prev)
        prev = s
    det.append((fin[:-1] ^ fin[1:]) ^ prev)
    return np.concatenate(det)


def logical_error_rate(d, scale, shots):
    rounds = d
    method = "automatic" if EXACT_T1 else "stabilizer"
    sim = AerSimulator(method=method, noise_model=make_noise_model(scale))
    qc = transpile(build_circuit(d, rounds), sim, optimization_level=0)
    counts = sim.run(qc, shots=shots).result().get_counts()

    keys = list(counts)
    dets, observed = [], []
    for k in keys:
        syn, fin = parse_key(k, rounds)
        dets.append(detection_events(syn, fin))
        observed.append(fin[0])

    predicted = build_matcher(d, rounds).decode_batch(np.array(dets, dtype=np.uint8))[:, 0]
    wrong = predicted != np.array(observed)
    p_l = sum(counts[k] for k, w in zip(keys, wrong) if w) / shots
    per_round = 0.5 * (1 - (1 - 2 * min(p_l, 0.4999)) ** (1 / rounds))
    return p_l, per_round


def main():
    results = {d: [] for d in DISTANCES}
    print(f"{'d':>3} {'scale':>7} {'P_L (d rounds)':>15} {'per round':>11}")
    for d in DISTANCES:
        for s in SCALES:
            p_l, per_round = logical_error_rate(d, s, SHOTS)
            results[d].append(p_l)
            print(f"{d:>3} {s:>7.3f} {p_l:>15.4f} {per_round:>11.5f}")

    plt.figure(figsize=(6.5, 4.5))
    for d in DISTANCES:
        plt.loglog(SCALES, np.maximum(results[d], 0.5 / SHOTS), "o-", label=f"d = {d}")
    plt.axvline(1.0, color="gray", ls="--", lw=1, label="nominal device")
    plt.xlabel("noise scale (1 = nominal T1/T2, gate, readout errors)")
    plt.ylabel("logical error rate")
    plt.title("Repetition-code memory, superconducting-style noise")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("qec_memory_realistic.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
