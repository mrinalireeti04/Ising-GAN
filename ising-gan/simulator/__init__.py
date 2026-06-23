"""
simulator/__init__.py
Exposes the Stage B (Numba) API as the default simulator interface.
Import Stage A explicitly from simulator.ising_numpy when needed.
"""
from simulator.ising_numba import (
    initialize_lattice,
    metropolis_sweep,
    total_energy,
    total_magnetization,
    calculate_correlation,
    run_simulation,
    warmup,
)

__all__ = [
    "initialize_lattice",
    "metropolis_sweep",
    "total_energy",
    "total_magnetization",
    "calculate_correlation",
    "run_simulation",
    "warmup",
]
