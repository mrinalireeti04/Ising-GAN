"""
constants.py — Shared parameters extracted from the Fortran source.

All values are taken directly from fortran/ising_model.f90.
Do NOT modify these — they define the canonical physics.
"""

# ──────────────────────────────────────────────
#  FORTRAN CANONICAL PARAMETERS  (read-only)
# ──────────────────────────────────────────────

L_FORTRAN: int = 128          # Lattice side length (grid is L × L)
J: float = 1.0                # Ferromagnetic coupling constant (J > 0)

# Monte Carlo cycle counts (Fortran production values)
EQ_CYCLES: int = 500_000      # Equilibration sweeps before measurement
MEAS_CYCLES: int = 1_000      # Snapshots saved per temperature
MEAS_GAP: int = 100           # Sweeps between snapshots (decorrelation)

# Temperature sweep  (Fortran: T = 4.0, dT = (4-1)/49, 50 steps descending)
T_START: float = 4.0
T_END: float = 1.0
N_TEMPS: int = 50
DT: float = (T_START - T_END) / (N_TEMPS - 1)   # ≈ 0.06122

# Critical temperature (Onsager exact solution)
T_CRITICAL: float = 2.0 / ((__import__('math').log(1 + __import__('math').sqrt(2))))
# T_CRITICAL ≈ 2.26919

# Phase label threshold (matches Fortran: IF T < 2.269 → 'F')
T_PHASE_BOUNDARY: float = 2.269

# ──────────────────────────────────────────────
#  VALIDATION / QUICK-RUN PARAMETERS
#  Used for Step 4 plots — reduced for CPU speed.
#  Physics is still correct; just fewer samples.
# ──────────────────────────────────────────────

# Lattice sizes for validation plots (matches checklist Step 4)
L_VALIDATION = [30, 50, 70, 100]

# Reduced cycle counts for validation runs on CPU
EQ_CYCLES_VAL: int = 8_000    # Much less than 500,000 but sufficient for L ≤ 100
MEAS_CYCLES_VAL: int = 500    # Snapshots per temperature
MEAS_GAP_VAL: int = 20        # Sweeps between snapshots

# Full-size production parameters (use on GPU / Kaggle)
EQ_CYCLES_PROD: int = 500_000
MEAS_CYCLES_PROD: int = 1_000
MEAS_GAP_PROD: int = 100

# ──────────────────────────────────────────────
#  TEMPERATURE SCHEDULES
# ──────────────────────────────────────────────

def fortran_temperatures():
    """
    Reproduce the exact Fortran temperature schedule:
    T = 4.0, 4.0 - dT, 4.0 - 2*dT, ..., ~1.0   (50 points, descending).
    Returns a list of 50 float temperatures.
    """
    return [T_START - i * DT for i in range(N_TEMPS)]


def weighted_temperatures():
    """
    Proposal-defined weighted sweep (Section 4.2):
    dense near T_c ≈ 2.269, coarse elsewhere.
    ~37 points total.
    """
    import numpy as np
    low   = list(np.arange(1.0, 2.0, 0.2))          # 5 points
    dense = list(np.arange(2.0, 2.61, 0.05))         # 13 points (includes 2.25, 2.27, 2.30)
    high  = list(np.arange(2.8, 4.01, 0.2))          # 7 points
    return sorted(set(low + dense + high))
