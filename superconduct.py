"""
Surface code quantum memory whose noise comes from a superconducting-qubit simulation.

Step 1: simulate two coupled transmons (3 levels each) with a
  Lindblad master equation (T1 decay + dephasing):
    - a flux-pulse CZ gate, tuned automatically (leakage, qubit swap and
      conditional phase all come out of the simulation)
    - a DRAG-driven 30 ns pi/2 pulse (the physical gate behind the Hadamard)
    - idle during measurement + reset
  Each simulated gate is converted to Pauli error probabilities.
Step 2 (error correction): those probabilities are inserted into a Stim surface
  code circuit, which is decoded with PyMatching.

Output: stored-bit failure rate for 3x3, 5x5 and 7x7 grids of data qubits at
the nominal device and a plot (qec_surface_code_physical.png).
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
import stim
import pymatching
from scipy.linalg import expm
from scipy.optimize import minimize

# ------------------------- device parameters ------------------------------
F_A, F_B = 5.0, 6.0        # GHz: tunable qubit A (idle), fixed qubit B
ALPHA = -0.2               # GHz: anharmonicity of both transmons
G = 0.012                  # GHz: A-B coupling
T1, T2 = 150e3, 100e3      # ns
T_MEAS, T_RESET = 800.0, 400.0   # ns
P_OVERLAP = 5e-2           # readout overlap error (not simulated)
P_RESET = 5e-3             # reset error (not simulated)

CODE_SIZES = [3, 5, 7]                              # grid size of data qubits
NOISE_LEVELS = [0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.5]  # 1.0 = nominal device
SHOTS = 20000
# --------------------------------------------------------------------------

TWO_PI = 2 * np.pi
A1 = np.diag(np.sqrt([1.0, 2.0]), 1)     # 3-level annihilation operator
I3 = np.eye(3)
AA, BB = np.kron(A1, I3), np.kron(I3, A1)
PAULIS = [np.eye(2), np.array([[0, 1], [1, 0]], complex),
          np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0]).astype(complex)]


# ------------------------------ physics -----------------------------------
def collapse_ops(ops, t1, t2):
    tphi = 1.0 / (1.0 / t2 - 1.0 / (2.0 * t1))
    cs = []
    for a in ops:
        cs.append(np.sqrt(1.0 / t1) * a)
        cs.append(np.sqrt(2.0 / tphi) * (a.conj().T @ a))
    return cs


def super_h(H):
    I = np.eye(H.shape[0])
    return -1j * (np.kron(I, H) - np.kron(H.T, I))


def super_d(cs, d):
    I = np.eye(d)
    L = np.zeros((d * d, d * d), complex)
    for c in cs:
        cdc = c.conj().T @ c
        L += np.kron(c.conj(), c) - 0.5 * np.kron(I, cdc) - 0.5 * np.kron(cdc.T, I)
    return L


def super_prop(Hs, dt, cs, d):
    Ld = super_d(cs, d)
    S = np.eye(d * d, dtype=complex)
    for H in Hs:
        S = expm((super_h(H) + Ld) * dt) @ S
    return S


def comp_idx(n):
    levels = itertools.product(range(3), repeat=n)
    return [i for i, lv in enumerate(levels) if max(lv) < 2]


def embed_comp(Wc, n):
    W = np.eye(3 ** n, dtype=complex)
    idx = comp_idx(n)
    W[np.ix_(idx, idx)] = Wc
    return W


def pauli_probs(S, n, W):
    """Pauli-twirl a simulated channel. W is applied after the channel
    (it undoes the ideal gate and virtual-Z frame). Returns 4**n probabilities
    (index = base-4 digits I,X,Y,Z per qubit) and the leakage."""
    d, dc, idx = 3 ** n, 2 ** n, comp_idx(n)
    labels = list(itertools.product(range(4), repeat=n))
    R = {}
    for lab in labels:
        Pc = np.array([[1.0]])
        for k in lab:
            Pc = np.kron(Pc, PAULIS[k])
        P9 = np.zeros((d, d), complex)
        P9[np.ix_(idx, idx)] = Pc
        out = (S @ P9.reshape(-1, order="F")).reshape(d, d, order="F")
        out = W @ out @ W.conj().T
        R[lab] = np.real(np.trace(P9 @ out)) / dc
    r0 = R[labels[0]]
    leak = 1 - r0
    anti = lambda a, b: 1 if (a != 0 and b != 0 and a != b) else 0
    p = np.zeros(len(labels))
    for qi, Q in enumerate(labels):
        tot = 0.0
        for P in labels:
            par = sum(anti(a, b) for a, b in zip(P, Q)) % 2
            tot += R[P] / r0 * (-1 if par else 1)
        p[qi] = tot / 4 ** n
    p = np.clip(p, 0, None)
    p /= p.sum()
    return (1 - leak) * p + leak / 4 ** n


def h_two(delta):
    nA, nB = AA.conj().T @ AA, BB.conj().T @ BB
    return TWO_PI * ((F_A + delta) * nA + F_B * nB
                     + ALPHA / 2 * nA @ (nA - np.eye(9)) + ALPHA / 2 * nB @ (nB - np.eye(9))
                     + G * (AA.conj().T @ BB + AA @ BB.conj().T))


def h_two_idle():
    nA, nB = AA.conj().T @ AA, BB.conj().T @ BB
    return TWO_PI * (F_A * nA + F_B * nB
                     + ALPHA / 2 * nA @ (nA - np.eye(9)) + ALPHA / 2 * nB @ (nB - np.eye(9)))


def flux_pulse(delta_pk, hold, ramp=5.0, dt=0.25):
    total = hold + 2 * ramp
    n = int(np.ceil(total / dt))
    dt = total / n
    t = (np.arange(n) + 0.5) * dt
    s = np.where(t < ramp, 0.5 * (1 - np.cos(np.pi * t / ramp)),
                 np.where(t > ramp + hold, 0.5 * (1 - np.cos(np.pi * (total - t) / ramp)), 1.0))
    return delta_pk * s, dt, total


def cz_unitary(delta_pk, hold):
    deltas, dt, total = flux_pulse(delta_pk, hold)
    U = np.eye(9, dtype=complex)
    for d in deltas:
        U = expm(-1j * h_two(d) * dt) @ U
    return expm(1j * h_two_idle() * total) @ U, total   # frame of the idle qubits


def cz_cost(x):
    U, _ = cz_unitary(x[0], x[1])
    idx = comp_idx(2)
    U4 = U[np.ix_(idx, idx)]
    leak = 1 - np.mean(np.sum(np.abs(U4) ** 2, axis=0))
    off = (np.sum(np.abs(U4) ** 2) - np.sum(np.abs(np.diag(U4)) ** 2)) / 4
    cond = np.angle(U4[3, 3]) - np.angle(U4[1, 1]) - np.angle(U4[2, 2]) + np.angle(U4[0, 0])
    err = (cond % TWO_PI) - np.pi          # conditional phase minus pi, wrapped
    return leak + off + err ** 2 / 20.0


def tune_cz():
    res = minimize(cz_cost, [0.8, 25.0], method="Nelder-Mead",
                   options=dict(xatol=1e-5, fatol=1e-10, maxiter=300))
    return res.x


def cz_noise(pulse, t1, t2):
    delta_pk, hold = pulse
    deltas, dt, total = flux_pulse(delta_pk, hold)
    S = super_prop([h_two(d) for d in deltas], dt, collapse_ops([AA, BB], t1, t2), 9)
    U, _ = cz_unitary(delta_pk, hold)
    idx = comp_idx(2)
    U4 = U[np.ix_(idx, idx)]
    th1 = -(np.angle(U4[2, 2]) - np.angle(U4[0, 0]))   # virtual-Z corrections
    th2 = -(np.angle(U4[1, 1]) - np.angle(U4[0, 0]))
    K = np.diag([1, np.exp(1j * th2), np.exp(1j * th1), np.exp(1j * (th1 + th2))])
    Wc = np.diag([1, 1, 1, -1]).conj().T @ K
    W = embed_comp(Wc, 2) @ expm(1j * h_two_idle() * total)
    return pauli_probs(S, 2, W)


def rx(th):
    return np.array([[np.cos(th / 2), -1j * np.sin(th / 2)],
                     [-1j * np.sin(th / 2), np.cos(th / 2)]])


def rz(c):
    return np.diag([np.exp(-1j * c / 2), np.exp(1j * c / 2)])


def x90_noise(t1, t2, T=30.0, dt=0.1):
    n = int(round(T / dt))
    t = (np.arange(n) + 0.5) * dt
    amp, alpha = np.pi / T, TWO_PI * ALPHA
    ox = amp * np.sin(np.pi * t / T) ** 2
    oy = -(amp * (np.pi / T) * np.sin(2 * np.pi * t / T)) / alpha      # DRAG
    nop = A1.conj().T @ A1
    Hs = [alpha / 2 * nop @ (nop - np.eye(3)) + 0.5 * x * (A1 + A1.conj().T)
          + 0.5 * y * (1j * A1.conj().T - 1j * A1) for x, y in zip(ox, oy)]
    S = super_prop(Hs, dt, collapse_ops([A1], t1, t2), 3)
    U = np.eye(3, dtype=complex)
    for H in Hs:
        U = expm(-1j * H * dt) @ U
    idx = comp_idx(1)
    U2 = U[np.ix_(idx, idx)]
    target = rx(np.pi / 2)
    best = max(np.linspace(-np.pi, np.pi, 1441),
               key=lambda c: abs(np.trace(target.conj().T @ rz(c) @ U2)))  # virtual Z
    return pauli_probs(S, 1, embed_comp(target.conj().T @ rz(best), 1))


def idle_noise(t, t1, t2):
    S = expm(super_d(collapse_ops([A1], t1, t2), 3) * t)
    return pauli_probs(S, 1, np.eye(3, dtype=complex))


def readout_flip(t1, t2, level):
    """T1 decay during half the readout window (simulated) + overlap error."""
    S = expm(super_d(collapse_ops([A1], t1, t2), 3) * (T_MEAS / 2))
    rho1 = np.zeros((3, 3), complex)
    rho1[1, 1] = 1
    out = (S @ rho1.reshape(-1, order="F")).reshape(3, 3, order="F")
    return min(P_OVERLAP * level + 0.5 * np.real(out[0, 0]), 0.5)


def device_noise(level, pulse):
    t1, t2 = T1 / level, T2 / level
    p2 = cz_noise(pulse, t1, t2)
    swap_zx = {0: 0, 1: 3, 2: 2, 3: 1}      # CX = H(target) CZ H(target): X<->Z on target
    p_cx = np.zeros(16)
    for a in range(4):
        for b in range(4):
            p_cx[a * 4 + swap_zx[b]] = p2[a * 4 + b]
    return dict(
        cx=list(p_cx[1:]),
        h=list(x90_noise(t1, t2)[1:]),
        idle=list(idle_noise(T_MEAS + T_RESET, t1, t2)[1:]),
        readout=readout_flip(t1, t2, level),
        reset=min(P_RESET * level, 0.5),
        cz_error=1 - p2[0],
    )


# --------------------------- error correction -----------------------------
def find_data_qubits(circuit):
    found = set()
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            found |= find_data_qubits(inst.body_copy())
        elif inst.name == "M":
            found.update(t.value for t in inst.targets_copy())
    return found


def add_noise(circuit, data, noise):
    out = stim.Circuit()
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            out.append(stim.CircuitRepeatBlock(inst.repeat_count,
                                               add_noise(inst.body_copy(), data, noise)))
            continue
        out.append(inst)
        if inst.name == "CX":
            q = [t.value for t in inst.targets_copy()]
            for k in range(0, len(q), 2):
                out.append("PAULI_CHANNEL_2", q[k:k + 2], noise["cx"])
        elif inst.name == "H":
            out.append("PAULI_CHANNEL_1", [t.value for t in inst.targets_copy()], noise["h"])
        elif inst.name == "MR":
            out.append("PAULI_CHANNEL_1", sorted(data), noise["idle"])
    return out


def failure_rate(size, noise):
    clean = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=size, rounds=size,
        before_measure_flip_probability=noise["readout"],
        after_reset_flip_probability=noise["reset"])
    circuit = add_noise(clean, find_data_qubits(clean), noise)
    dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    detections, observables = circuit.compile_detector_sampler().sample(
        SHOTS, separate_observables=True)
    predicted = matcher.decode_batch(detections)
    return float(np.mean(np.any(predicted != observables, axis=1)))


def main():
    print("Simulating transmons and tuning the CZ gate...")
    pulse = tune_cz()
    results = {n: [] for n in CODE_SIZES}
    nominal = None
    for level in NOISE_LEVELS:
        noise = device_noise(level, pulse)
        if level == 1.0:
            nominal = noise
            print(f"Simulated nominal device: CZ error {100 * noise['cz_error']:.2f}%, "
                  f"pi/2 gate error {100 * sum(noise['h']):.3f}%, "
                  f"idle error per round {100 * sum(noise['idle']):.2f}%")
        for n in CODE_SIZES:
            results[n].append(failure_rate(n, noise))
        print(f"noise level {level}: done")

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
    plt.title("Surface code memory, noise from transmon simulation")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("qec_surface_code_physical.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()