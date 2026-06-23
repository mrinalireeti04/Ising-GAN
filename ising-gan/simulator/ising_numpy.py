"""
ising_numpy.py — Stage A: Pure NumPy Ising Model Simulator
============================================================
Reference implementation that faithfully mirrors the Fortran source
(fortran/ising_model.f90) line-by-line.

Physics:
  - 2D Ising model on an L×L square lattice with periodic boundary conditions
  - Metropolis Monte Carlo update rule
  - Ferromagnetic coupling J > 0, units where k_B = 1

All functions are written to be easily verifiable against the Fortran
by hand.  Speed is secondary — correctness is the only goal here.
Stage B (ising_numba.py) uses Numba JIT for production speed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ─────────────────────────────────────────────────────────────────────────────
#  INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def initialize_lattice(L: int, seed: int | None = None) -> NDArray[np.int8]:
    """
    Return a random ±1 lattice of shape (L, L) as int8.

    Matches Fortran InitializeSpins:
        IF (rand < 0.5) → +1 ELSE → -1
    """
    rng = np.random.default_rng(seed)
    # choice([-1, 1]) with equal probability
    flat = rng.integers(0, 2, size=L * L, dtype=np.int8)
    flat = np.where(flat == 0, np.int8(-1), np.int8(1))
    return flat.reshape(L, L)


# ─────────────────────────────────────────────────────────────────────────────
#  SITE-LEVEL HELPERS  (used by unit tests and NumPy reference sweep)
# ─────────────────────────────────────────────────────────────────────────────

def neighbor_sum(spins: NDArray[np.int8], x: int, y: int, L: int) -> int:
    """
    Sum of 4 nearest-neighbour spins with periodic boundaries (0-indexed).

    Mirrors Fortran NeighborSum:
        xm = MOD(x-2+L, L) + 1   →  (x-1) % L  in Python
        xp = MOD(x,   L) + 1   →  (x+1) % L  in Python
        ym = MOD(y-2+L, L) + 1   →  (y-1) % L
        yp = MOD(y,   L) + 1   →  (y+1) % L
    """
    xm = (x - 1) % L
    xp = (x + 1) % L
    ym = (y - 1) % L
    yp = (y + 1) % L
    return int(spins[xm, y]) + int(spins[xp, y]) + int(spins[x, ym]) + int(spins[x, yp])


# ─────────────────────────────────────────────────────────────────────────────
#  METROPOLIS SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def metropolis_sweep(
    spins: NDArray[np.int8],
    L: int,
    beta: float,
    J: float = 1.0,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int8]:
    """
    One Monte Carlo sweep: L² random single-spin flip attempts (in-place).

    Exactly mirrors Fortran PerformCycle:
        N = L*L
        DO step = 1, N
            pick random (x, y)
            deltaE = 2 * J * spins(x,y) * NeighborSum(x, y)
            IF deltaE <= 0 → flip
            ELSE flip with prob exp(-beta * deltaE)
        END DO

    Parameters
    ----------
    spins : (L, L) int8 array — modified in-place
    L     : lattice side length
    beta  : inverse temperature 1/T
    J     : coupling constant (default 1.0)
    rng   : optional numpy Generator for reproducibility

    Returns
    -------
    spins : same array (in-place), returned for chaining convenience
    """
    if rng is None:
        rng = np.random.default_rng()

    N = L * L
    # Pre-draw all random numbers for this sweep (faster than calling rng N times)
    xs = rng.integers(0, L, size=N)
    ys = rng.integers(0, L, size=N)
    rand_acc = rng.random(size=N)   # acceptance probabilities

    for k in range(N):
        x, y = int(xs[k]), int(ys[k])
        ns = (int(spins[(x - 1) % L, y]) +
              int(spins[(x + 1) % L, y]) +
              int(spins[x, (y - 1) % L]) +
              int(spins[x, (y + 1) % L]))
        dE = 2.0 * J * int(spins[x, y]) * ns

        if dE <= 0.0:
            spins[x, y] = -spins[x, y]
        elif rand_acc[k] < np.exp(-beta * dE):
            spins[x, y] = -spins[x, y]

    return spins


# ─────────────────────────────────────────────────────────────────────────────
#  THERMODYNAMIC OBSERVABLES
# ─────────────────────────────────────────────────────────────────────────────

def total_energy(spins: NDArray[np.int8], L: int, J: float = 1.0) -> float:
    """
    Total Hamiltonian energy of the lattice.

    Mirrors Fortran TotalEnergy:
        E = Σ_{i,j} −J × s(i,j) × NeighborSum(i,j)
        E = 0.5 × E          ← corrects double-counting

    Vectorised equivalent using np.roll (counts each bond once, no 0.5 needed):
        E = −J × Σ_{i,j} s(i,j) × [s(i+1,j) + s(i,j+1)]
    """
    s = spins.astype(np.float64)
    right = np.roll(s, -1, axis=1)   # s(i, j+1)
    down  = np.roll(s, -1, axis=0)   # s(i+1, j)
    return float(-J * np.sum(s * (right + down)))


def total_magnetization(spins: NDArray[np.int8], L: int) -> float:
    """
    Absolute total magnetization.

    Mirrors Fortran: magnetization = ABS(TotalMagnetization(spins, L))
    """
    return float(abs(np.sum(spins)))


def calculate_correlation(spins: NDArray[np.int8], L: int) -> NDArray[np.float64]:
    """
    Spin-spin correlation function along the row (vertical) direction.

    Mirrors Fortran CalculateCorrelation:
        DO shift = 1, L
            corr(shift) = Σ_{row,col} s(row,col) × s((row+shift-1) mod L + 1, col)
            corr(shift) /= L²
        END DO

    Python (0-indexed): paired row = (row + shift) % L
    Returns array of length L.  corr[0] = shift-1-row correlation, etc.
    """
    s = spins.astype(np.float64)
    corr = np.empty(L, dtype=np.float64)
    N = L * L
    for shift in range(1, L + 1):
        # np.roll(s, -shift, axis=0): element [i,j] gets value from [(i+shift)%L, j]
        shifted = np.roll(s, -shift, axis=0)
        corr[shift - 1] = np.sum(s * shifted) / N
    return corr


# ─────────────────────────────────────────────────────────────────────────────
#  FULL SIMULATION RUN AT ONE TEMPERATURE
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    L: int,
    T: float,
    J: float = 1.0,
    eq_cycles: int = 500_000,
    meas_cycles: int = 1_000,
    meas_gap: int = 100,
    seed: int | None = None,
    store_snapshots: bool = False,
) -> dict:
    """
    Full simulation at a single temperature, matching the Fortran main loop.

    Parameters
    ----------
    L              : lattice side length
    T              : temperature
    J              : coupling constant (default 1.0)
    eq_cycles      : equilibration sweeps (Fortran: 500,000)
    meas_cycles    : measurement snapshots to save (Fortran: 1,000)
    meas_gap       : sweeps between snapshots (Fortran: 100)
    seed           : optional RNG seed for reproducibility
    store_snapshots: if True, return full L×L arrays (memory-intensive)

    Returns
    -------
    dict with keys:
        T, L, J, beta
        mean_energy, mean_energy_per_spin
        mean_magnetization, mean_magnetization_per_spin
        specific_heat, susceptibility
        corr_fn, corr_array
        energies, magnetizations        (per-snapshot arrays)
        snapshots                        (list, only if store_snapshots=True)
    """
    beta = 1.0 / T
    N = L * L

    rng = np.random.default_rng(seed)
    spins = initialize_lattice(L, seed=rng.integers(0, 2**31))

    # ── Equilibration ──────────────────────────────────────────────────────
    for _ in range(eq_cycles):
        metropolis_sweep(spins, L, beta, J, rng)

    # ── Measurement phase ──────────────────────────────────────────────────
    energies       = np.empty(meas_cycles, dtype=np.float64)
    magnetizations = np.empty(meas_cycles, dtype=np.float64)
    corr_accum     = np.zeros(L, dtype=np.float64)
    snapshots      = []

    for step in range(meas_cycles):
        # Decorrelation gap between snapshots
        for _ in range(meas_gap):
            metropolis_sweep(spins, L, beta, J, rng)

        E = total_energy(spins, L, J)
        M = total_magnetization(spins, L)
        corr = calculate_correlation(spins, L)

        energies[step]       = E
        magnetizations[step] = M
        corr_accum           += corr

        if store_snapshots:
            snapshots.append(spins.copy())

    # ── Thermodynamic averages (matching Fortran formulas exactly) ──────────
    mean_E  = np.mean(energies)
    mean_E2 = np.mean(energies ** 2)
    mean_M  = np.mean(magnetizations)
    mean_M2 = np.mean(magnetizations ** 2)

    specific_heat  = (mean_E2 - mean_E ** 2) / (T ** 2 * N)
    susceptibility = (mean_M2 - mean_M ** 2) / (T * N)

    # Scalar correlation function (Fortran: SUM(correlation)/(L*measurement_cycle))
    corr_fn = float(np.sum(corr_accum) / (L * meas_cycles))

    result = {
        "T":                        T,
        "L":                        L,
        "J":                        J,
        "beta":                     beta,
        "mean_energy":              mean_E,
        "mean_energy_per_spin":     mean_E / N,
        "mean_magnetization":       mean_M,
        "mean_magnetization_per_spin": mean_M / N,
        "specific_heat":            specific_heat,
        "susceptibility":           susceptibility,
        "corr_fn":                  corr_fn,
        "corr_array":               corr_accum / meas_cycles,
        "energies":                 energies,
        "magnetizations":           magnetizations,
    }
    if store_snapshots:
        result["snapshots"] = snapshots

    return result
