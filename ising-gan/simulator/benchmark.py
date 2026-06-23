"""
benchmark.py — Stage A vs Stage B Speed Comparison
====================================================
Run with:  python simulator/benchmark.py

Confirms that the Numba-accelerated Stage B (ising_numba.py) is ≥ 10× faster
than the pure-Python Stage A (ising_numpy.py) at the same lattice size.

Also verifies STATISTICAL AGREEMENT: both stages must produce compatible
energy and magnetization distributions when run with the same initial state
and same number of sweeps.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import numpy as np
from scipy import stats

import simulator.ising_numpy as stage_a
import simulator.ising_numba as stage_b


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

L_BENCH       = 32       # Lattice size for benchmark
T_BENCH       = 2.5      # Temperature
J             = 1.0
EQ_CYCLES     = 500      # Short equilibration for benchmark
MEAS_CYCLES   = 200      # Measurement snapshots
MEAS_GAP      = 5        # Sweeps between snapshots
SEED          = 1234


# ─────────────────────────────────────────────────────────────────────────────
#  SPEED BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

def time_stage_a() -> float:
    """Time Stage A for MEAS_CYCLES sweeps (after equilibration)."""
    beta = 1.0 / T_BENCH
    rng  = np.random.default_rng(SEED)
    spins = stage_a.initialize_lattice(L_BENCH, seed=SEED)

    # Equilibrate (short)
    for _ in range(EQ_CYCLES):
        stage_a.metropolis_sweep(spins, L_BENCH, beta, J, rng)

    t0 = time.perf_counter()
    for _ in range(MEAS_CYCLES):
        for _ in range(MEAS_GAP):
            stage_a.metropolis_sweep(spins, L_BENCH, beta, J, rng)
    elapsed = time.perf_counter() - t0
    return elapsed


def time_stage_b() -> float:
    """Time Stage B for the same workload (after JIT warm-up)."""
    beta = 1.0 / T_BENCH
    np.random.seed(SEED)
    spins = stage_b.initialize_lattice(L_BENCH, seed=SEED)

    # Equilibrate
    stage_b._equilibrate_nb(spins, L_BENCH, beta, J, EQ_CYCLES)

    t0 = time.perf_counter()
    for _ in range(MEAS_CYCLES):
        stage_b._equilibrate_nb(spins, L_BENCH, beta, J, MEAS_GAP)
    elapsed = time.perf_counter() - t0
    return elapsed


# ─────────────────────────────────────────────────────────────────────────────
#  STATISTICAL AGREEMENT TEST
# ─────────────────────────────────────────────────────────────────────────────

def statistical_agreement_test(n_meas: int = 300, alpha: float = 0.05):
    """
    Run both stages independently at the same T for n_meas measurements.
    Use a two-sample KS test to confirm the energy and magnetization
    distributions are statistically indistinguishable.

    Note: Because both stages use independent random streams, we can only
    test distributional agreement, not sample-by-sample equality.
    """
    beta = 1.0 / T_BENCH

    # --- Stage A ---
    rng_a  = np.random.default_rng(42)
    spins_a = stage_a.initialize_lattice(L_BENCH, seed=42)
    for _ in range(2000):
        stage_a.metropolis_sweep(spins_a, L_BENCH, beta, J, rng_a)

    E_a, M_a = [], []
    for _ in range(n_meas):
        for _ in range(MEAS_GAP):
            stage_a.metropolis_sweep(spins_a, L_BENCH, beta, J, rng_a)
        E_a.append(stage_a.total_energy(spins_a, L_BENCH, J))
        M_a.append(stage_a.total_magnetization(spins_a, L_BENCH))

    # --- Stage B ---
    np.random.seed(99)
    spins_b = stage_b.initialize_lattice(L_BENCH, seed=99)
    stage_b._equilibrate_nb(spins_b, L_BENCH, beta, J, 2000)

    E_b, M_b = [], []
    for _ in range(n_meas):
        stage_b._equilibrate_nb(spins_b, L_BENCH, beta, J, MEAS_GAP)
        E_b.append(stage_b.total_energy(spins_b, L_BENCH, J))
        M_b.append(stage_b.total_magnetization(spins_b, L_BENCH))

    E_a, M_a = np.array(E_a), np.array(M_a)
    E_b, M_b = np.array(E_b), np.array(M_b)

    # KS tests
    ks_e = stats.ks_2samp(E_a, E_b)
    ks_m = stats.ks_2samp(M_a, M_b)

    return {
        "mean_E_A":       float(np.mean(E_a)) / (L_BENCH**2),
        "mean_E_B":       float(np.mean(E_b)) / (L_BENCH**2),
        "mean_M_A":       float(np.mean(M_a)) / (L_BENCH**2),
        "mean_M_B":       float(np.mean(M_b)) / (L_BENCH**2),
        "ks_energy_pval": ks_e.pvalue,
        "ks_mag_pval":    ks_m.pvalue,
        "agree_energy":   ks_e.pvalue > alpha,
        "agree_mag":      ks_m.pvalue > alpha,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Ising-GAN: Stage A vs Stage B Benchmark")
    print(f"  L={L_BENCH}, T={T_BENCH}, eq={EQ_CYCLES}, "
          f"meas={MEAS_CYCLES}, gap={MEAS_GAP}")
    print("=" * 60)

    # ── JIT warm-up ───────────────────────────────────────────────────────
    print("\n[1/3] Warming up Numba JIT …")
    stage_b.warmup(L=8)

    # ── Speed test ────────────────────────────────────────────────────────
    print(f"\n[2/3] Timing {MEAS_CYCLES}×{MEAS_GAP} sweeps at L={L_BENCH} …")

    t_a = time_stage_a()
    print(f"  Stage A (NumPy):  {t_a:.3f} s")

    t_b = time_stage_b()
    print(f"  Stage B (Numba):  {t_b:.4f} s")

    speedup = t_a / max(t_b, 1e-9)
    print(f"  Speedup:          {speedup:.1f}×")

    if speedup >= 10:
        print("  ✓ Stage B is ≥ 10× faster — PASS")
    else:
        print(f"  ✗ Speedup {speedup:.1f}× < 10× — check Numba installation")

    # ── Statistical agreement ─────────────────────────────────────────────
    print(f"\n[3/3] Statistical agreement test (n=300 samples each) …")
    res = statistical_agreement_test(n_meas=300)

    print(f"  ⟨E⟩/N  — Stage A: {res['mean_E_A']:+.4f},  Stage B: {res['mean_E_B']:+.4f}")
    print(f"  ⟨M⟩/N  — Stage A: {res['mean_M_A']:.4f},   Stage B: {res['mean_M_B']:.4f}")
    print(f"  KS energy p-value: {res['ks_energy_pval']:.4f}  "
          f"→ {'✓ agree' if res['agree_energy'] else '✗ DIFFER'}")
    print(f"  KS magnet p-value: {res['ks_mag_pval']:.4f}  "
          f"→ {'✓ agree' if res['agree_mag'] else '✗ DIFFER'}")

    print("\n" + "=" * 60)
    if res["agree_energy"] and res["agree_mag"] and speedup >= 10:
        print("  ALL CHECKS PASSED — Stage B approved for production.")
    else:
        print("  SOME CHECKS FAILED — review output above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
