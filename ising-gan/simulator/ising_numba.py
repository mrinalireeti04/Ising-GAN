"""
ising_numba.py — Stage B: Numba-Accelerated Ising Model Simulator
==================================================================
JIT-compiled reimplementation of Stage A (ising_numpy.py) using Numba @njit.

The physics and algorithmic logic are IDENTICAL to Stage A.
Only the Python-level loops are replaced by @njit-compiled functions,
giving near-C performance (~10–50× faster than Stage A's Python loops).

Usage:
    from simulator.ising_numba import (
        initialize_lattice, metropolis_sweep,
        total_energy, total_magnetization,
        calculate_correlation, run_simulation, warmup
    )

    warmup()  # pre-compile JIT functions once before timing

Note on random numbers:
    Numba maintains its own internal RNG state (seeded via np.random.seed()
    before calling @njit functions).  For reproducible runs, call
    np.random.seed(seed) in the CALLER before invoking these functions.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray


# ─────────────────────────────────────────────────────────────────────────────
#  JIT-COMPILED CORE KERNELS
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _neighbor_sum_nb(spins: np.ndarray, x: int, y: int, L: int) -> int:
    """4-nearest-neighbour sum with periodic boundaries (0-indexed)."""
    xm = (x - 1) % L
    xp = (x + 1) % L
    ym = (y - 1) % L
    yp = (y + 1) % L
    return int(spins[xm, y]) + int(spins[xp, y]) + int(spins[x, ym]) + int(spins[x, yp])


@njit(cache=True)
def _metropolis_sweep_nb(spins: np.ndarray, L: int, beta: float, J: float) -> None:
    """
    In-place Metropolis sweep: L² random flip attempts.
    Exactly mirrors Fortran PerformCycle loop.
    """
    N = L * L
    for _ in range(N):
        # Random site selection: mirrors Fortran RANDOM_NUMBER → INT(rand*L)+1
        x = int(np.random.random() * L)
        y = int(np.random.random() * L)

        ns = _neighbor_sum_nb(spins, x, y, L)
        dE = 2.0 * J * float(spins[x, y]) * float(ns)

        if dE <= 0.0:
            spins[x, y] = -spins[x, y]
        elif np.random.random() < np.exp(-beta * dE):
            spins[x, y] = -spins[x, y]


@njit(cache=True)
def _total_energy_nb(spins: np.ndarray, L: int, J: float) -> float:
    """
    Total lattice energy.  Counts each bond once (right + down neighbors)
    which is equivalent to Fortran's all-4-neighbors × 0.5.
    """
    E = 0.0
    for i in range(L):
        for j in range(L):
            ip = (i + 1) % L
            jp = (j + 1) % L
            E -= J * float(spins[i, j]) * (float(spins[ip, j]) + float(spins[i, jp]))
    return E


@njit(cache=True)
def _total_magnetization_nb(spins: np.ndarray, L: int) -> float:
    """Absolute total magnetization. Mirrors Fortran ABS(TotalMagnetization)."""
    M = 0.0
    for i in range(L):
        for j in range(L):
            M += float(spins[i, j])
    if M < 0.0:
        M = -M
    return M


@njit(cache=True)
def _calculate_correlation_nb(spins: np.ndarray, L: int) -> np.ndarray:
    """
    Row-direction correlation function for shifts 1 … L.
    Mirrors Fortran CalculateCorrelation exactly.
    Returns array of length L.
    """
    corr = np.zeros(L)
    N = L * L
    for shift in range(1, L + 1):
        c = 0.0
        for i in range(L):
            ip = (i + shift) % L
            for j in range(L):
                c += float(spins[i, j]) * float(spins[ip, j])
        corr[shift - 1] = c / N
    return corr


@njit(cache=True)
def _equilibrate_nb(spins: np.ndarray, L: int, beta: float, J: float,
                    eq_cycles: int) -> None:
    """Run eq_cycles sweeps without recording (equilibration phase)."""
    for _ in range(eq_cycles):
        _metropolis_sweep_nb(spins, L, beta, J)


@njit(cache=True)
def _measure_nb(
    spins: np.ndarray,
    L: int,
    beta: float,
    J: float,
    meas_cycles: int,
    meas_gap: int,
) -> tuple:
    """
    Measurement phase: meas_cycles snapshots, each preceded by meas_gap sweeps.
    Returns (energies, magnetizations, corr_accum) as flat arrays.
    """
    energies       = np.empty(meas_cycles)
    magnetizations = np.empty(meas_cycles)
    corr_accum     = np.zeros(L)

    for step in range(meas_cycles):
        for _ in range(meas_gap):
            _metropolis_sweep_nb(spins, L, beta, J)

        energies[step]       = _total_energy_nb(spins, L, J)
        magnetizations[step] = _total_magnetization_nb(spins, L)
        corr_accum           += _calculate_correlation_nb(spins, L)

    return energies, magnetizations, corr_accum


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API  (mirrors ising_numpy.py interface)
# ─────────────────────────────────────────────────────────────────────────────

def initialize_lattice(L: int, seed: int | None = None) -> NDArray[np.int8]:
    """Random ±1 lattice — same as Stage A."""
    rng = np.random.default_rng(seed)
    flat = rng.integers(0, 2, size=L * L, dtype=np.int8)
    flat = np.where(flat == 0, np.int8(-1), np.int8(1))
    return flat.reshape(L, L)


def metropolis_sweep(
    spins: NDArray[np.int8],
    L: int,
    beta: float,
    J: float = 1.0,
) -> NDArray[np.int8]:
    """Numba-accelerated single MC sweep (in-place)."""
    _metropolis_sweep_nb(spins, L, beta, J)
    return spins


def total_energy(spins: NDArray[np.int8], L: int, J: float = 1.0) -> float:
    """Numba-accelerated energy calculation."""
    return float(_total_energy_nb(spins, L, J))


def total_magnetization(spins: NDArray[np.int8], L: int) -> float:
    """Numba-accelerated absolute magnetization."""
    return float(_total_magnetization_nb(spins, L))


def calculate_correlation(spins: NDArray[np.int8], L: int) -> NDArray[np.float64]:
    """Numba-accelerated row-direction correlation function."""
    return _calculate_correlation_nb(spins, L)


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
    Full simulation at a single temperature using Numba kernels.
    Interface is identical to ising_numpy.run_simulation.

    Note: snapshots cannot be returned from @njit scope, so when
    store_snapshots=True, extra sweeps are run in Python and spins
    are copied after each measurement batch.
    """
    beta = 1.0 / T
    N    = L * L

    if seed is not None:
        np.random.seed(seed)   # Seeds Numba's internal RNG

    spins = initialize_lattice(L, seed=seed)

    # Equilibration (fully compiled)
    _equilibrate_nb(spins, L, beta, J, eq_cycles)

    if store_snapshots:
        # Must run measurement in Python to capture snapshots
        energies       = np.empty(meas_cycles)
        magnetizations = np.empty(meas_cycles)
        corr_accum     = np.zeros(L)
        snapshots      = []

        for step in range(meas_cycles):
            _equilibrate_nb(spins, L, beta, J, meas_gap)
            energies[step]       = _total_energy_nb(spins, L, J)
            magnetizations[step] = _total_magnetization_nb(spins, L)
            corr_accum          += _calculate_correlation_nb(spins, L)
            snapshots.append(spins.copy())
    else:
        energies, magnetizations, corr_accum = _measure_nb(
            spins, L, beta, J, meas_cycles, meas_gap
        )
        snapshots = []

    mean_E  = float(np.mean(energies))
    mean_E2 = float(np.mean(energies ** 2))
    mean_M  = float(np.mean(magnetizations))
    mean_M2 = float(np.mean(magnetizations ** 2))

    specific_heat  = (mean_E2 - mean_E ** 2) / (T ** 2 * N)
    susceptibility = (mean_M2 - mean_M ** 2) / (T * N)
    corr_fn        = float(np.sum(corr_accum) / (L * meas_cycles))

    result = {
        "T":                            T,
        "L":                            L,
        "J":                            J,
        "beta":                         beta,
        "mean_energy":                  mean_E,
        "mean_energy_per_spin":         mean_E / N,
        "mean_magnetization":           mean_M,
        "mean_magnetization_per_spin":  mean_M / N,
        "specific_heat":                specific_heat,
        "susceptibility":               susceptibility,
        "corr_fn":                      corr_fn,
        "corr_array":                   corr_accum / meas_cycles,
        "energies":                     energies,
        "magnetizations":               magnetizations,
    }
    if store_snapshots:
        result["snapshots"] = snapshots

    return result


def warmup(L: int = 8) -> None:
    """
    Pre-compile all Numba JIT functions by running a tiny simulation.
    Call this ONCE before any timed benchmarks or production runs.
    First-call compilation typically takes 10–30 s.
    """
    print("Warming up Numba JIT compilation (first call only, ~15 s)…")
    spins = initialize_lattice(L, seed=0)
    _metropolis_sweep_nb(spins, L, 0.44, 1.0)
    _total_energy_nb(spins, L, 1.0)
    _total_magnetization_nb(spins, L)
    _calculate_correlation_nb(spins, L)
    _equilibrate_nb(spins, L, 0.44, 1.0, 2)
    _, _, _ = _measure_nb(spins, L, 0.44, 1.0, 2, 1)
    print("Numba JIT compilation complete.")
