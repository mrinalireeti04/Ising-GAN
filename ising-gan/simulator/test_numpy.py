"""
test_numpy.py — Unit Tests for Stage A (ising_numpy.py)
========================================================
Run with:  python -m pytest simulator/test_numpy.py -v
       or:  python simulator/test_numpy.py

Four physics-based unit tests required by the checklist:
  1. All-up lattice   → E = −2J×L², |M|/N = 1.0
  2. All-down lattice → same energy by symmetry, |M|/N = 1.0
  3. Flip ΔE in uniform lattice → ΔE = +8J > 0, flip prob < 1
  4. High-T equilibration → ⟨|M|⟩/N ≈ 0 (within statistical noise)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
import pytest

from simulator.ising_numpy import (
    initialize_lattice,
    neighbor_sum,
    metropolis_sweep,
    total_energy,
    total_magnetization,
    calculate_correlation,
    run_simulation,
)
from simulator.constants import J as J_DEFAULT, T_CRITICAL


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 1 — All-up lattice: energy and magnetization
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("L", [4, 8, 16, 32])
def test_all_up_energy(L):
    """
    Ground state: all spins +1.
    Expected: E = −2 × J × L²  (each of L² sites contributes −2J to avoid double-count)

    Derivation:
        Each spin has 4 neighbors = +1, so NeighborSum = 4.
        Fortran: E = Σ -J × s × ns  ×  0.5 = L² × (−4J) × 0.5 = −2J×L²
        NumPy equivalent:  Σ -J × s × (right+down) = L² × (−2J) [counts each bond once]
    """
    spins = np.ones((L, L), dtype=np.int8)
    E = total_energy(spins, L, J=J_DEFAULT)
    expected = -2.0 * J_DEFAULT * L * L
    assert abs(E - expected) < 1e-9, (
        f"L={L}: E={E:.6f}, expected {expected:.6f}"
    )


@pytest.mark.parametrize("L", [4, 8, 16, 32])
def test_all_up_magnetization(L):
    """Ground state: all spins +1 → |M|/N = 1.0 exactly."""
    spins = np.ones((L, L), dtype=np.int8)
    M = total_magnetization(spins, L)
    N = L * L
    assert abs(M / N - 1.0) < 1e-12, f"L={L}: |M|/N = {M/N}"


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 2 — All-down lattice: energy by symmetry
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("L", [4, 8, 16, 32])
def test_all_down_energy(L):
    """
    All-down lattice (all −1) has the same energy as all-up.
    The Hamiltonian is symmetric under global spin flip.
    """
    spins = -np.ones((L, L), dtype=np.int8)
    E = total_energy(spins, L, J=J_DEFAULT)
    expected = -2.0 * J_DEFAULT * L * L
    assert abs(E - expected) < 1e-9, (
        f"L={L}: E={E:.6f}, expected {expected:.6f}"
    )


@pytest.mark.parametrize("L", [4, 8, 16, 32])
def test_all_down_magnetization(L):
    """All-down: |M|/N = 1.0 (absolute value removes sign)."""
    spins = -np.ones((L, L), dtype=np.int8)
    M = total_magnetization(spins, L)
    N = L * L
    assert abs(M / N - 1.0) < 1e-12, f"L={L}: |M|/N = {M/N}"


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 3 — Flip ΔE in uniform background: ΔE = +8J, flip probability < 1
# ─────────────────────────────────────────────────────────────────────────────

def test_delta_E_unfavourable_flip():
    """
    In a uniform all-up lattice, trying to flip any spin from +1 to −1:
        ΔE = 2 × J × spins[x,y] × NeighborSum(x,y)
           = 2 × 1.0 × (+1) × (+4) = +8J

    This is energetically unfavourable, so flip probability = exp(−β×8J) < 1.
    At T=1 (β=1): prob ≈ exp(−8) ≈ 3.35 × 10⁻⁴
    """
    L = 10
    J = J_DEFAULT
    spins = np.ones((L, L), dtype=np.int8)

    x, y = 5, 5
    ns = neighbor_sum(spins, x, y, L)
    assert ns == 4, f"Expected NeighborSum=4 for uniform lattice, got {ns}"

    dE = 2.0 * J * int(spins[x, y]) * ns
    assert dE == pytest.approx(8.0 * J), f"Expected ΔE = 8J = {8*J}, got {dE}"
    assert dE > 0.0, "ΔE must be positive (unfavourable flip)"

    # Flip probability at T=1
    T = 1.0
    beta = 1.0 / T
    flip_prob = math.exp(-beta * dE)
    assert 0.0 < flip_prob < 1.0, f"Expected 0 < flip_prob < 1, got {flip_prob}"
    assert flip_prob == pytest.approx(math.exp(-8.0), rel=1e-10)


def test_delta_E_favourable_flip():
    """
    In an all-down lattice, a lone +1 spin at center has ns = −4:
        ΔE = 2 × J × (+1) × (−4) = −8J < 0
    → flip is accepted unconditionally.
    """
    L = 10
    J = J_DEFAULT
    spins = -np.ones((L, L), dtype=np.int8)
    spins[5, 5] = 1   # lone +1 spin in all-down background

    ns = neighbor_sum(spins, 5, 5, L)
    assert ns == -4, f"Expected NeighborSum=-4, got {ns}"

    dE = 2.0 * J * int(spins[5, 5]) * ns
    assert dE == pytest.approx(-8.0 * J)
    assert dE < 0.0, "ΔE must be negative (favourable flip)"


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 4 — High-temperature equilibration: ⟨|M|⟩/N ≈ 0
# ─────────────────────────────────────────────────────────────────────────────

def test_high_temperature_disordered():
    """
    At very high T (T=10), thermal noise dominates.
    After equilibration, the mean |M|/N should be close to 0.
    For a random lattice, ⟨|M|/N⟩ ≈ sqrt(1/N) ~ 0.05 for L=16.
    We accept anything < 0.15 as sufficient evidence of disorder.
    """
    L = 16
    T = 10.0
    beta = 1.0 / T

    rng = np.random.default_rng(42)
    spins = initialize_lattice(L, seed=42)

    # Equilibrate
    for _ in range(2000):
        metropolis_sweep(spins, L, beta, J=J_DEFAULT, rng=rng)

    # Measure over 200 snapshots
    mag_values = []
    for _ in range(200):
        for _ in range(10):
            metropolis_sweep(spins, L, beta, J=J_DEFAULT, rng=rng)
        M = total_magnetization(spins, L) / (L * L)
        mag_values.append(M)

    mean_mag = np.mean(mag_values)
    assert mean_mag < 0.15, (
        f"High-T test failed: ⟨|M|/N⟩ = {mean_mag:.4f} (expected < 0.15 for T=10)"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 5 — Correlation function properties
# ─────────────────────────────────────────────────────────────────────────────

def test_correlation_zero_shift_is_one():
    """
    With shift = L (wraps to 0), correlation should equal 1.0 (self-correlation).
    corr[L-1] = Σ s(i,j) * s((i+L)%L, j) / N = Σ s(i,j)^2 / N = N/N = 1
    """
    L = 8
    rng = np.random.default_rng(7)
    spins = initialize_lattice(L, seed=7)
    corr = calculate_correlation(spins, L)
    # corr[L-1] corresponds to shift = L (full wrap-around = self)
    assert abs(corr[L - 1] - 1.0) < 1e-10, (
        f"Self-correlation (shift=L) should be 1.0, got {corr[L-1]}"
    )


def test_correlation_symmetry():
    """Correlation function values must be in [−1, 1] for any lattice."""
    L = 16
    spins = initialize_lattice(L, seed=99)
    corr = calculate_correlation(spins, L)
    assert np.all(corr >= -1.0 - 1e-10), "Correlation values below -1"
    assert np.all(corr <=  1.0 + 1e-10), "Correlation values above +1"


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 6 — Neighbor sum boundary conditions
# ─────────────────────────────────────────────────────────────────────────────

def test_periodic_boundary_corner():
    """
    For a site at corner (0, 0) in an L×L lattice,
    the periodic neighbours should be (L-1,0), (1,0), (0,L-1), (0,1).
    Verify with a known all-up lattice.
    """
    L = 6
    spins = np.ones((L, L), dtype=np.int8)
    ns = neighbor_sum(spins, 0, 0, L)
    assert ns == 4, f"Corner site in all-up lattice: expected ns=4, got {ns}"


def test_periodic_boundary_known():
    """
    Craft a small lattice where we know the neighbor sum by inspection.
    """
    L = 4
    spins = np.zeros((L, L), dtype=np.int8)
    spins[0, 0] = 1
    spins[1, 0] = 1
    spins[3, 0] = 1  # = (0-1)%4, so left neighbor of row 0
    spins[0, 1] = 1  # right neighbor
    # Site (0,0): left=(3,0)=+1, right=(1,0)=+1, up=(3,0)... wait
    # Let me do x=row, y=col. neighbor_sum at (0,0):
    # xm=(0-1)%4=3 → spins[3,0] = +1
    # xp=(0+1)%4=1 → spins[1,0] = +1
    # ym=(0-1)%4=3 → spins[0,3] = 0
    # yp=(0+1)%4=1 → spins[0,1] = +1
    # Total = 1+1+0+1 = 3
    ns = neighbor_sum(spins, 0, 0, L)
    assert ns == 3, f"Expected neighbor_sum=3 at (0,0), got {ns}"


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN (run without pytest)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    tests = [
        ("All-up energy (L=8)",            lambda: test_all_up_energy(8)),
        ("All-up magnetization (L=8)",     lambda: test_all_up_magnetization(8)),
        ("All-down energy (L=8)",          lambda: test_all_down_energy(8)),
        ("All-down magnetization (L=8)",   lambda: test_all_down_magnetization(8)),
        ("ΔE unfavourable flip (+8J)",     test_delta_E_unfavourable_flip),
        ("ΔE favourable flip (−8J)",       test_delta_E_favourable_flip),
        ("High-T disorder (T=10, L=16)",   test_high_temperature_disordered),
        ("Correlation self-shift = 1.0",   test_correlation_zero_shift_is_one),
        ("Correlation in [−1, 1]",         test_correlation_symmetry),
        ("Periodic boundary corner",        test_periodic_boundary_corner),
        ("Periodic boundary known case",   test_periodic_boundary_known),
    ]

    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS]  {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL]  {name}")
            print(f"          {e}")
            failed += 1

    print(f"\n{passed}/{passed+failed} tests passed.")
    if failed:
        sys.exit(1)
