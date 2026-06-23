"""
generate_update1_plots.py
=========================
Self-contained script for Supervisor Update 1.
Produces professor-style publication plots matching the reference Fortran
output (Ising_2D_plots.pdf).

KEY FIX — Cold/Hot start initialisation:
  T < T_COLD_CUTOFF (≈1.8): spins start ORDERED (all +1, ~1% random noise)
                             → equilibration fast even at low T
  T ≥ T_COLD_CUTOFF       : spins start RANDOM (standard hot start)
                             → correct exploration of disordered/critical phase

This eliminates the L=100 / low-T equilibration failure seen when using
a purely random start with insufficient eq_cycles.

Run from project root:
    python update_1/generate_update1_plots.py
"""

from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent / "ising-gan"
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

import simulator.ising_numba as sim
from simulator.ising_numba import (
    _equilibrate_nb, _measure_nb, _total_energy_nb,
    _total_magnetization_nb, _calculate_correlation_nb,
)
from simulator.constants import J, T_CRITICAL, fortran_temperatures

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────

UPDATE_DIR  = Path(__file__).resolve().parent
PLOTS_DIR   = UPDATE_DIR / "plots"
CFG_DIR     = PLOTS_DIR / "sample_configs"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
CFG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

L_VALUES      = [30, 50, 70, 100]

# Equilibration: enough for any L at any T with smart init
EQ_CYCLES     = 30_000    # 3.75× more than before; fine with cold start
MEAS_CYCLES   = 600
MEAS_GAP      = 20

# Cold-start cutoff: below this temperature use mostly-ordered initialisation
T_COLD_CUTOFF = 1.80

# Professor plot colours / markers (matching reference image)
COLORS  = ["black",  "#CC0000", "#009900", "#0000CC"]
MARKERS = ["s",      "o",       "^",       "v"      ]

# ─────────────────────────────────────────────────────────────────────────────
#  SMART INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def smart_init(L: int, T: float, seed: int) -> np.ndarray:
    """
    Hot start (random) for T ≥ T_COLD_CUTOFF.
    Cold start (almost all +1) for T < T_COLD_CUTOFF.

    This ensures correct physics at low T without needing 500k eq sweeps.
    Physically equivalent to the Fortran's random start + 500,000 cycles
    because at low T the ordered state IS the equilibrium state.
    """
    rng = np.random.default_rng(seed)
    if T < T_COLD_CUTOFF:
        spins = np.ones((L, L), dtype=np.int8)
        # Flip ~1% randomly to seed thermal fluctuations
        mask = rng.random((L, L)) < 0.01
        spins[mask] = np.int8(-1)
    else:
        flat = rng.integers(0, 2, size=L * L, dtype=np.int8)
        flat = np.where(flat == 0, np.int8(-1), np.int8(1))
        spins = flat.reshape(L, L)
    return spins


# ─────────────────────────────────────────────────────────────────────────────
#  CORE RUN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_one_T(L: int, T: float, seed: int) -> dict:
    beta = 1.0 / T
    N    = L * L

    np.random.seed(seed)                          # seeds Numba's RNG
    spins = smart_init(L, T, seed)

    _equilibrate_nb(spins, L, beta, J, EQ_CYCLES)

    energies, magnetizations, corr_accum = _measure_nb(
        spins, L, beta, J, MEAS_CYCLES, MEAS_GAP
    )

    mean_E  = float(np.mean(energies))
    mean_E2 = float(np.mean(energies ** 2))
    mean_M  = float(np.mean(magnetizations))
    mean_M2 = float(np.mean(magnetizations ** 2))

    Cv  = (mean_E2 - mean_E ** 2) / (T ** 2 * N)
    chi = (mean_M2 - mean_M ** 2) / (T * N)

    return {
        "T":   T,
        "E":   mean_E / N,
        "M":   mean_M / N,
        "Cv":  Cv,
        "chi": chi,
    }


def run_sweep(L: int, temperatures: list) -> dict:
    rows = {"T": [], "E": [], "M": [], "Cv": [], "chi": []}
    for i, T in enumerate(tqdm(temperatures, desc=f"L={L:3d}", ncols=80)):
        r = run_one_T(L, T, seed=i * 997 + L * 13)
        for k in rows:
            rows[k].append(r[k])
    for k in rows:
        rows[k] = np.array(rows[k])
    # sort ascending T for plotting
    idx = np.argsort(rows["T"])
    return {k: rows[k][idx] for k in rows}


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT STYLE
# ─────────────────────────────────────────────────────────────────────────────

def set_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "black",
        "axes.linewidth":   1.2,
        "axes.grid":        True,
        "grid.color":       "#cccccc",
        "grid.linewidth":   0.7,
        "xtick.direction":  "in",
        "ytick.direction":  "in",
        "xtick.color":      "black",
        "ytick.color":      "black",
        "font.family":      "DejaVu Sans",
        "font.size":        12,
        "lines.linewidth":  1.5,
        "lines.markersize": 5,
        "figure.dpi":       150,
        "text.color":       "black",
        "legend.facecolor": "white",
        "legend.edgecolor": "black",
    })


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 1 — Combined 2×2  (matches reference figure exactly)
# ─────────────────────────────────────────────────────────────────────────────

def plot_2x2(all_data: dict, out: Path):
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("2D Ising Model Results — Python Reimplementation",
                 fontsize=14, fontweight="bold", y=1.005)

    panels = [
        ("E",   r"$\langle E \rangle$",   axes[0, 0], "Energy per Spin"),
        ("M",   r"$\langle M \rangle$",   axes[0, 1], "Magnetisation per Spin"),
        ("Cv",  r"$C_V$",                 axes[1, 0], "Specific Heat"),
        ("chi", r"$\chi_T$",              axes[1, 1], "Susceptibility"),
    ]

    for key, ylabel, ax, title in panels:
        for i, L in enumerate(L_VALUES):
            d = all_data[L]
            ax.plot(d["T"], d[key],
                    color=COLORS[i], marker=MARKERS[i],
                    markersize=4, linewidth=1.4,
                    label=f"L = {L}", markevery=3)
        ax.axvline(T_CRITICAL, color="magenta", linestyle="--",
                   linewidth=1.4, alpha=0.85, label=f"$T_c$={T_CRITICAL:.3f}")
        ax.set_xlabel("T", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlim(1.0, 4.05)
        ax.legend(fontsize=9, framealpha=1.0)

    fig.tight_layout()
    p = out / "validation_2x2.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 2 — Individual high-res plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_individuals(all_data: dict, out: Path):
    set_style()
    panels = [
        ("E",   r"$\langle E \rangle / N$  [$J$]",
         "Mean Energy per Spin",          "energy_vs_T.png"),
        ("M",   r"$\langle |M| \rangle / N$",
         "Mean |Magnetisation| per Spin", "magnetization_vs_T.png"),
        ("Cv",  r"$C_V$  [$k_B$]",
         "Specific Heat",                 "specific_heat_vs_T.png"),
        ("chi", r"$\chi_T$  [$1/J$]",
         "Magnetic Susceptibility",       "susceptibility_vs_T.png"),
    ]

    for key, ylabel, title, fname in panels:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for i, L in enumerate(L_VALUES):
            d = all_data[L]
            ax.plot(d["T"], d[key],
                    color=COLORS[i], marker=MARKERS[i],
                    markersize=5, linewidth=1.6,
                    label=f"L = {L}", markevery=2)
        ax.axvline(T_CRITICAL, color="magenta", linestyle="--",
                   linewidth=1.6, label=f"$T_c$ = {T_CRITICAL:.4f}")
        ax.set_xlabel("Temperature  $T$  [$J/k_B$]", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlim(1.0, 4.05)
        ax.legend(fontsize=10, framealpha=1.0)
        fig.tight_layout()
        p = out / fname
        fig.savefig(p, dpi=150, facecolor="white")
        plt.close(fig)
        print(f"  Saved: {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 3 — Sample lattice images
# ─────────────────────────────────────────────────────────────────────────────

def make_sample_images(cfg_dir: Path, plots_dir: Path):
    set_style()
    L    = 128
    cmap = ListedColormap(["#111111", "#eeeeee"])

    regimes = [
        (1.5,          "Ordered phase\n(T = 1.5 < $T_c$)"),
        (T_CRITICAL,   f"Critical point\n(T ≈ {T_CRITICAL:.3f} = $T_c$)"),
        (3.5,          "Disordered phase\n(T = 3.5 > $T_c$)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), facecolor="white")
    fig.suptitle("Representative Spin Configurations  (L = 128, Python Simulator)",
                 fontsize=13, fontweight="bold")

    for ax, (T, label) in zip(axes, regimes):
        print(f"  Snapshot at T={T:.3f} …", end=" ", flush=True)
        np.random.seed(int(T * 1000))
        spins = smart_init(L, T, seed=int(T * 1000))
        _equilibrate_nb(spins, L, 1.0 / T, J, 20_000)

        M_val = abs(float(np.mean(spins)))
        ax.imshow(spins, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"{label}\n⟨|M|⟩/N = {M_val:.3f}", fontsize=11)
        ax.axis("off")

        # Individual file
        fig_s, ax_s = plt.subplots(figsize=(5, 5), facecolor="white")
        ax_s.imshow(spins, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
        ax_s.set_title(f"T = {T:.3f}  |  ⟨|M|⟩/N = {M_val:.3f}",
                       fontsize=11, fontweight="bold")
        ax_s.axis("off")
        fig_s.tight_layout()
        fname = f"T_{T:.2f}_spin_config.png"
        fig_s.savefig(cfg_dir / fname, dpi=150, facecolor="white")
        plt.close(fig_s)
        print(f"saved")

    fig.tight_layout()
    p = plots_dir / "sample_configs_panel.png"
    fig.savefig(p, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 4 — meas_cycle noise study
# ─────────────────────────────────────────────────────────────────────────────

def plot_meas_noise(out: Path):
    set_style()
    L, T, EQ, N_RUNS = 50, T_CRITICAL, 8_000, 6
    meas_vals = [100, 500, 1000, 5000]

    Cv_m, Cv_s, chi_m, chi_s = [], [], [], []
    for mc in meas_vals:
        Cv_r, chi_r = [], []
        for run in range(N_RUNS):
            r = sim.run_simulation(
                L=L, T=T, J=J,
                eq_cycles=EQ, meas_cycles=mc, meas_gap=20,
                seed=run * 1000 + mc,
            )
            Cv_r.append(r["specific_heat"])
            chi_r.append(r["susceptibility"])
        Cv_m.append(np.mean(Cv_r));   Cv_s.append(np.std(Cv_r))
        chi_m.append(np.mean(chi_r)); chi_s.append(np.std(chi_r))
        print(f"    mc={mc:5d}: Cᵥ={Cv_m[-1]:.3f}±{Cv_s[-1]:.3f},"
              f"  χ={chi_m[-1]:.2f}±{chi_s[-1]:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
    for ax, means, stds, ylabel, title in zip(
        axes,
        [Cv_m, chi_m], [Cv_s, chi_s],
        [r"$C_V$  [$k_B$]", r"$\chi_T$  [$1/J$]"],
        ["Specific Heat", "Susceptibility"],
    ):
        ax.errorbar(meas_vals, means, yerr=stds, fmt="o-",
                    color="black", capsize=6, markersize=7,
                    linewidth=1.5, markerfacecolor="#CC0000")
        ax.set_xscale("log")
        ax.set_xlabel("measurement_cycle", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{title} fluctuation at $T_c$  vs  measurement_cycle",
                     fontsize=11, fontweight="bold")

    fig.suptitle(
        f"Noise vs measurement_cycle  (L=50, T={T_CRITICAL:.4f}, "
        f"n={N_RUNS} independent runs)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    p = out / "meas_cycle_noise.png"
    fig.savefig(p, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  PHYSICS SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check(all_data: dict):
    print(f"\n{'='*60}")
    print("  PHYSICS SANITY CHECK")
    print(f"  Tc (Onsager exact) = {T_CRITICAL:.5f}")
    print(f"{'='*60}")
    all_pass = True
    for L, d in all_data.items():
        idx_lo  = np.argmin(d["T"])   # lowest T ≈ 1.0
        idx_hi  = np.argmax(d["T"])   # highest T ≈ 4.0
        Tc_Cv   = d["T"][np.argmax(d["Cv"])]
        Tc_chi  = d["T"][np.argmax(d["chi"])]
        E_lo    = d["E"][idx_lo]
        M_lo    = d["M"][idx_lo]
        M_hi    = d["M"][idx_hi]

        ok_E  = abs(E_lo - (-2.0)) < 0.1
        ok_M  = M_lo > 0.85
        ok_M0 = M_hi < 0.15
        ok_Cv = abs(Tc_Cv - T_CRITICAL) < 0.3
        ok_X  = abs(Tc_chi - T_CRITICAL) < 0.3

        status = "[PASS]" if all([ok_E, ok_M, ok_M0, ok_Cv, ok_X]) else "[WARN]"
        if not all([ok_E, ok_M, ok_M0, ok_Cv, ok_X]):
            all_pass = False

        print(f"\n  L={L:3d}  {status}")
        print(f"    E/N at T~1.0 : {E_lo:+.4f}  (expect ≈ -2.0)   {'OK' if ok_E else 'WARN'}")
        print(f"    M/N at T~1.0 : {M_lo:.4f}   (expect > 0.85)   {'OK' if ok_M else 'WARN'}")
        print(f"    M/N at T~4.0 : {M_hi:.4f}   (expect < 0.15)   {'OK' if ok_M0 else 'WARN'}")
        print(f"    Cv peak  at T : {Tc_Cv:.4f}  (expect ≈ {T_CRITICAL:.4f}) {'OK' if ok_Cv else 'WARN'}")
        print(f"    chi peak at T : {Tc_chi:.4f}  (expect ≈ {T_CRITICAL:.4f}) {'OK' if ok_X else 'WARN'}")

    print()
    if all_pass:
        print("  ALL CHECKS PASSED — simulator physics validated.")
    else:
        print("  Some checks flagged — increase eq_cycles if needed.")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Supervisor Update 1 — Plot Generation")
    print(f"  L = {L_VALUES}")
    print(f"  eq={EQ_CYCLES}  meas={MEAS_CYCLES}  gap={MEAS_GAP}")
    print(f"  Cold start for T < {T_COLD_CUTOFF}  |  Hot start for T >= {T_COLD_CUTOFF}")
    print(f"  50 temperatures  4.0 -> 1.0  (Fortran schedule)")
    print("=" * 65)

    sim.warmup(L=8)
    temperatures = fortran_temperatures()

    # ── 1. Main sweep ──────────────────────────────────────────────────────
    print("\n[1/5] Running temperature sweep …")
    all_data = {}
    for L in L_VALUES:
        all_data[L] = run_sweep(L, temperatures)

    sanity_check(all_data)

    # ── 2. 2×2 combined plot ───────────────────────────────────────────────
    print("\n[2/5] 2x2 combined plot (professor style) …")
    plot_2x2(all_data, PLOTS_DIR)

    # ── 3. Individual plots ────────────────────────────────────────────────
    print("\n[3/5] Individual plots …")
    plot_individuals(all_data, PLOTS_DIR)

    # ── 4. Sample spin configs ─────────────────────────────────────────────
    print("\n[4/5] Sample spin configurations (L=128) …")
    make_sample_images(CFG_DIR, PLOTS_DIR)

    # ── 5. meas_cycle noise study ──────────────────────────────────────────
    print("\n[5/5] Measurement cycle noise study …")
    plot_meas_noise(PLOTS_DIR)

    print("\n" + "=" * 65)
    print("  Done. All outputs in update_1/plots/")
    print("=" * 65)


if __name__ == "__main__":
    main()
