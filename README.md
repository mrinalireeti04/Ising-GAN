# Ising-GAN

**Learning Physically Consistent Ising Spin Configurations with Deep Generative Adversarial Networks**

A Python reimplementation of a Monte Carlo Ising simulator coupled with a GAN-based generative pipeline and physics-based validation.

## Project Structure

```
ising-gan/
  fortran/          ← original Fortran source (read-only reference)
  simulator/        ← Python Ising simulator (Stage A: NumPy, Stage B: Numba)
  data/             ← generated spin configurations (CSV + PNG)
  gan/              ← GAN model code (DCGAN → WGAN-GP → Conditional)
  evaluation/       ← physics metrics and comparison scripts
  notebooks/        ← one Jupyter notebook per supervisor update
  plots/            ← saved validation and comparison figures
  reports/          ← supervisor update notes and analysis
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run unit tests (Stage A NumPy)
python simulator/test_numpy.py

# 3. Run Stage A vs Stage B benchmark
python simulator/benchmark.py

# 4. Run physics validation (generates all 4 plots)
python simulator/validate.py

# Full Fortran-equivalent parameters (slow — use on GPU):
# python simulator/validate.py --full
```

## Key Parameters (from Fortran source)

| Parameter | Value | Description |
|---|---|---|
| L | 128 | Lattice side length |
| J | 1.0 | Coupling constant |
| equilibrium_cycle | 500,000 | MC sweeps before measurement |
| measurement_cycle | 1,000 | Snapshots per temperature |
| measurement_cycle_gap | 100 | Sweeps between snapshots |
| T_c | ≈ 2.2692 | Critical temperature (Onsager exact) |
| Temperature sweep | 50 pts, 4.0 → 1.0 | dT = (4-1)/49 ≈ 0.0612 |

## Physics

The 2D Ising model Hamiltonian:

```
H = -J × Σ_{<i,j>} s_i × s_j
```

Metropolis Monte Carlo update rule: flip spin with probability `min(1, exp(-ΔE/kT))`.

## Validation Targets

- `⟨E⟩/N ≈ −2.0` at T=1.0, `≈ −0.6` at T=3.5
- `⟨|M|⟩/N ≈ 1.0` at T=1.0, → 0 above T_c  
- `C_v` and `χ_T` peak at T ≈ T_c = 2.269
- Larger L → sharper transition (finite-size scaling)
