# Quantum Memory Simulations

Simulations of a **quantum memory**: store one logical bit of information across
many noisy physical qubits, run rounds of error correction, decode, and measure
how often the stored bit survives. Larger codes should lose the bit less often
once the hardware is good enough.

The repository has two branches, one for each approach.

| Branch | Code | Noise | Files |
|---|---|---|---|
| `main` | Repetition code (Qiskit Aer) | Device-style: T1/T2 relaxation tied to real gate, measurement and reset times, plus CNOT, readout and reset errors | `qec_memory.py`, `visuals.py`, `README.md` |
| `stim-version` | Rotated surface code (Stim) | Derived from a simulation of superconducting transmon qubits | `surface_code.py`, `superconduct.py`, `visuals.py` |

Each branch has its own `visuals.py` that produces figures for the code on that
branch.

---

## `main`: repetition-code quantum memory (Qiskit)

`qec_memory.py` builds the memory as a Qiskit circuit and simulates it in Qiskit
Aer:

- n data qubits and n-1 ancillas, initialised in logical |0>
- n rounds of syndrome extraction: CNOTs in two parallel layers measure each
  neighbouring Z_i Z_{i+1} parity, followed by mid-circuit measurement and reset
- data qubits idle through readout and reset each round
- decoded with minimum-weight perfect matching (PyMatching) on a space-time
  graph that includes measurement errors and errors between CNOT layers

Noise (nominal device): T1 150 us, T2 100 us, CNOT 300 ns with 0.5% error,
measurement 800 ns, reset 400 ns, readout error 1.5%, reset error 0.5%. A
noise scale sweeps all errors up or down (1 = nominal).

```
pip install qiskit qiskit-aer pymatching matplotlib numpy pylatexenc
python qec_memory.py
python visuals.py
```

`visuals.py` produces:
- `qiskit_repetition_code.png`: the actual Qiskit memory circuit
- `quantum_memory_results.png`: stored-bit failure rate vs noise scale for 3, 5
  and 7 data qubits, next to the nominal error budget
- `decoder_graph.png`: the space-time graph the decoder matches on

---

## `stim-version`: surface-code quantum memory (Stim)

A repetition code only protects against bit flips. The surface code protects
against bit and phase flips, so this branch is the full quantum memory. It runs
3x3, 5x5 and 7x7 grids of data qubits in Stim, decoded with PyMatching.

- `surface_code.py`: surface code memory using Stim's built-in noise
  parameters, with T1/T2 approximated as depolarizing error.
- `superconduct.py`: the same memory, but the noise comes from a physical
  simulation. Two coupled 3-level transmons are evolved with a Lindblad master
  equation (T1 decay and dephasing). That covers a flux-pulse CZ gate (auto-tuned,
  with leakage and unwanted qubit swapping), a 30 ns DRAG pi/2 pulse and idle
  decoherence. Each simulated gate is Pauli-twirled and inserted into the Stim
  circuit.
- `visuals.py`: the Qiskit circuit for one syndrome-extraction round of the 3x3
  surface code, plus the transmon simulation figures (flux pulse, |11>/|02>
  population swap, DRAG pulse, CZ Pauli errors).

```
git checkout stim-version
pip install numpy scipy stim pymatching matplotlib qiskit pylatexenc
python superconduct.py
python visuals.py
```

Simulated nominal device: CZ error about 0.14%, pi/2 gate error about 0.05%,
idle error about 0.8% per round.

| Parameter | Value |
|---|---|
| Qubit A (tunable) idle frequency | 5.0 GHz |
| Qubit B (fixed) frequency | 6.0 GHz |
| Anharmonicity | -200 MHz |
| A-B coupling | 12 MHz |
| T1 / T2 | 150 us / 100 us |
| Measurement / reset time | 800 ns / 400 ns |
| Readout overlap error (not simulated) | 1% |
| Reset error (not simulated) | 0.5% |

---

## Why the code sizes are odd

Odd sizes (3, 5, 7) leave the decoder no ties when counting errors, so each
size step really does tolerate more errors. Even sizes add qubits without
adding protection.

## Known approximations

| Approximation | Fix |
|---|---|
| `main`: the repetition code only protects against bit flips | Use the surface code on the `stim-version` branch |
| `main`: T1/T2 relaxation is Pauli-twirled so the fast stabilizer simulator can run it | Set `EXACT_T1 = True` and use a single small code size to check against exact amplitude damping |
| `stim-version`: readout overlap and reset error are fixed numbers, not simulated | Replace `P_OVERLAP` and `P_RESET` in `superconduct.py` with measured values |
| `stim-version`: coherent errors are Pauli-twirled, which can misjudge their effect at large sizes | Cross-check the smallest code against an exact density-matrix simulation |
| `stim-version`: data-qubit idle noise is applied during measurement and reset only, not during CNOT layers | Add a `PAULI_CHANNEL_1` after each `CX` layer using `idle_noise()` with the layer duration |
| `stim-version`: both transmons share the same T1/T2 | Give `collapse_ops()` separate rates per qubit |

## Troubleshooting

- Slow run: reduce `SHOTS` or drop the largest code size at the top of the file.