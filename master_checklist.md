# Ising-GAN Project Master Checklist

This checklist outlines the sequence of tasks required to complete the Ising-GAN project, derived from the 7-phase timeline in the project proposal.

## Phase 1: Theory & Fortran Code Analysis (Week 1)
- [x] **Study Physics Theory:** Review the 2D Ising model, Metropolis algorithm, and known critical-point theory.
- [x] **Analyze Fortran Code:** Perform a line-by-line read-through of the supplied Fortran code to extract exact algorithmic logic.
- [x] **Document Logic:** Document every parameter, update rule, and boundary condition used in the original code. → `ising-gan/reports/fortran_analysis.md`
- [x] **Setup Environment:** Set up the Python environment with required libraries (NumPy, Numba, PyTorch, Matplotlib, scikit-image).

## Phase 2: Python Simulator Development & Validation (Week 2)
- [x] **Stage A (NumPy):** Implement pure NumPy Ising simulator at small lattice sizes (L = 16, 32) for fast iteration. → `ising-gan/simulator/ising_numpy.py` | 11/11 unit tests PASS
- [x] **Stage B (Numba):** Implement Numba-accelerated version and confirm its statistical output matches Stage A. → `ising-gan/simulator/ising_numba.py` | 33.7× speedup confirmed
- [x] **Validation:** Validate the simulator against theoretical critical temperature ($T_c \approx 2.269$) and known magnetization/energy curves. → All 4 plots reproduced, all L PASS physics sanity checks
- [x] **Scale Up:** Scale simulation up to full lattice size ($L = 128$) and confirm performance is acceptable for the full temperature sweep. → L=128 sample configs generated, cold/hot start strategy ensures correct physics at all T

## Phase 3: Full Dataset Generation (Week 3)
- [ ] **Run Sweep:** Run the validated Python simulator across the full weighted temperature sweep (dense sampling near $T_c$).
- [ ] **Generate Configurations:** Generate ~35,000–40,000 labeled configurations and a corresponding `metadata.csv` file.
- [ ] **Convert Format:** Convert the raw CSV outputs to PNG images (+1 as white, -1 as black); spot-check visually.
- [ ] **Package Dataset:** Organize the dataset into a PyTorch-ready folder structure.

## Phase 4: Baseline DCGAN (Week 4)
- [ ] **DCGAN Implementation:** Implement and train a baseline DCGAN on the full unconditional dataset.
- [ ] **Core Evaluation Scripts:** Build scripts to compare magnetization and energy distributions between simulated and generated configurations.
- [ ] **Troubleshoot:** Diagnose and resolve common GAN training failures (e.g., mode collapse, instability).

## Phase 5: WGAN-GP & Full Evaluation Suite (Week 5)
- [ ] **WGAN-GP Implementation:** Implement WGAN-GP with spectral normalization and gradient penalty for stable training.
- [ ] **Advanced Evaluation Scripts:** Build remaining evaluation metrics: spin-spin correlation function and cluster-size distribution.
- [ ] **Run Comparison:** Run full evaluation comparison across all sampled temperatures, placing special emphasis on the critical region near $T_c$.

## Phase 6: Temperature-Conditioned GAN (Week 6 - Stretch Goal)
- [ ] **Conditional Architecture:** Extend WGAN-GP to a conditional architecture by adding temperature as a conditioning input to both the generator and critic.
- [ ] **Train Model:** Train the conditional GAN on the full labeled dataset.
- [ ] **Test Interpolation:** Test generation at sparsely-sampled or held-out temperatures to evaluate interpolation quality and smoothness of the learned phase diagram.

## Phase 7: Final Evaluation, Report & Presentation (Week 7)
- [ ] **Compile Results:** Compile all comparison plots (histograms, curves) and compute the single scalar discrepancy (Wasserstein distance) summary table.
- [ ] **Write Report:** Write the final report structured around the physics-based evaluation framework.
- [ ] **Create Presentation:** Prepare a presentation deck highlighting the critical-point results as the centerpiece finding.
