"""
validate.py — Step 4: Physics Validation
=========================================
Reproduces the 4 reference plots from Ising_2D_plots.pdf and performs
all validation checks required by the checklist.

Run with:
    python simulator/validate.py                    # all checks, reduced params
    python simulator/validate.py --full             # full Fortran params (slow!)
    python simulator/validate.py --step 4e          # only measurement_cycle study
    python simulator/validate.py --step 4f          # only sample images

Output:
    plots/validation/energy_vs_T.png
    plots/validation/magnetization_vs_T.png
    plots/validation/specific_heat_vs_T.png
    plots/validation/susceptibility_vs_T.png
    plots/validation/meas_cycle_comparison.png
    plots/validation/sample_configs/T_1.50_config.png
    plots/validation/sample_configs/T_2.27_config.png
    plots/validation/sample_configs/T_3.50_config.png
"""

from __future__ import annotations

import sys
import os
import argparse
from pathlib import Path

# Allow running from any directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for script use
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from simulator.constants import (
    J, T_CRITICAL, T_PHASE_BOUNDARY,
    L_VALIDATION,
    EQ_CYCLES_VAL, MEAS_CYCLES_VAL, MEAS_GAP_VAL,
    EQ_CYCLES_PROD, MEAS_CYCLES_PROD, MEAS_GAP_PROD,
    fortran_temperatures,
)
import simulator.ising_numba as sim

# ─────────────────────────────────────────────────────────────────────────────
#  PLOT STYLE
# ─────────────────────────────────────────────────────────────────────────────

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
MARKERS = ["o", "s", "^", "D", "v"]

def apply_style():
    plt.rcParams.update({
        "figure.facecolor":  "#0d1117",
        "axes.facecolor":    "#161b22",
        "axes.edgecolor":    "#30363d",
        "axes.labelcolor":   "#e6edf3",
        "axes.titlecolor":   "#e6edf3",
        "axes.grid":         True,
        "grid.color":        "#21262d",
        "grid.linewidth":    0.8,
        "xtick.color":       "#8b949e",
        "ytick.color":       "#8b949e",
        "text.color":        "#e6edf3",
        "legend.facecolor":  "#161b22",
        "legend.edgecolor":  "#30363d",
        "legend.labelcolor": "#e6edf3",
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "lines.linewidth":   1.8,
        "lines.markersize":  5,
    })

# ─────────────────────────────────────────────────────────────────────────────
#  CORE SWEEP FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_temperature_sweep(
    L: int,
    temperatures: list[float],
    eq_cycles: int,
    meas_cycles: int,
    meas_gap: int,
    desc: str = "",
    seed_base: int = 0,
) -> dict:
    """
    Run the full temperature sweep for one lattice size.
    Returns dict with arrays indexed by temperature.
    """
    results = {
        "T":              [],
        "mean_E_per_N":   [],
        "mean_M_per_N":   [],
        "specific_heat":  [],
        "susceptibility": [],
        "corr_fn":        [],
    }

    label = desc or f"L={L}"
    for i, T in enumerate(tqdm(temperatures, desc=label, ncols=80)):
        seed = seed_base + i * 997   # different seed per temperature
        r = sim.run_simulation(
            L=L, T=T, J=J,
            eq_cycles=eq_cycles,
            meas_cycles=meas_cycles,
            meas_gap=meas_gap,
            seed=seed,
            store_snapshots=False,
        )
        results["T"].append(T)
        results["mean_E_per_N"].append(r["mean_energy_per_spin"])
        results["mean_M_per_N"].append(r["mean_magnetization_per_spin"])
        results["specific_heat"].append(r["specific_heat"])
        results["susceptibility"].append(r["susceptibility"])
        results["corr_fn"].append(r["corr_fn"])

    for k in results:
        results[k] = np.array(results[k])
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4a–4d:  Four Validation Plots
# ─────────────────────────────────────────────────────────────────────────────

def run_four_plots(
    L_values: list[int],
    eq_cycles: int,
    meas_cycles: int,
    meas_gap: int,
    out_dir: Path,
):
    """Reproduce the 4 reference plots from Ising_2D_plots.pdf."""
    temperatures = fortran_temperatures()   # 50 points, 4.0 → 1.0

    print(f"\n{'='*60}")
    print(f"  Steps 4a–4d: Validation Plots")
    print(f"  L_values = {L_values}")
    print(f"  T sweep: {temperatures[-1]:.4f} → {temperatures[0]:.4f}  ({len(temperatures)} points)")
    print(f"  eq_cycles={eq_cycles},  meas_cycles={meas_cycles},  meas_gap={meas_gap}")
    print(f"{'='*60}")

    all_results = {}
    for L in L_values:
        all_results[L] = run_temperature_sweep(
            L=L,
            temperatures=temperatures,
            eq_cycles=eq_cycles,
            meas_cycles=meas_cycles,
            meas_gap=meas_gap,
            desc=f"L={L:3d}",
        )

    apply_style()

    # ── 4a: Energy vs T ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, L in enumerate(L_values):
        r = all_results[L]
        ax.plot(r["T"], r["mean_E_per_N"],
                color=COLORS[i], marker=MARKERS[i], markersize=4,
                label=f"L = {L}", markevery=3)
    ax.axvline(T_CRITICAL, color="#f0883e", linestyle="--", linewidth=1.5,
               label=f"$T_c$ = {T_CRITICAL:.4f}", alpha=0.85)
    ax.set_xlabel("Temperature  $T$  [$J/k_B$]")
    ax.set_ylabel(r"$\langle E \rangle / N$  [$J$]")
    ax.set_title("Step 4a — Mean Energy per Spin vs Temperature")
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    path = out_dir / "energy_vs_T.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved {path.relative_to(PROJECT_ROOT)}")

    # ── 4b: Magnetization vs T ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, L in enumerate(L_values):
        r = all_results[L]
        ax.plot(r["T"], r["mean_M_per_N"],
                color=COLORS[i], marker=MARKERS[i], markersize=4,
                label=f"L = {L}", markevery=3)
    ax.axvline(T_CRITICAL, color="#f0883e", linestyle="--", linewidth=1.5,
               label=f"$T_c$ = {T_CRITICAL:.4f}", alpha=0.85)
    ax.set_xlabel("Temperature  $T$  [$J/k_B$]")
    ax.set_ylabel(r"$\langle |M| \rangle / N$")
    ax.set_title("Step 4b — Mean |Magnetization| per Spin vs Temperature")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    path = out_dir / "magnetization_vs_T.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved {path.relative_to(PROJECT_ROOT)}")

    # ── 4c: Specific Heat vs T ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, L in enumerate(L_values):
        r = all_results[L]
        ax.plot(r["T"], r["specific_heat"],
                color=COLORS[i], marker=MARKERS[i], markersize=4,
                label=f"L = {L}", markevery=3)
    ax.axvline(T_CRITICAL, color="#f0883e", linestyle="--", linewidth=1.5,
               label=f"$T_c$ = {T_CRITICAL:.4f}", alpha=0.85)
    ax.set_xlabel("Temperature  $T$  [$J/k_B$]")
    ax.set_ylabel(r"$C_v$  [$k_B$]")
    ax.set_title("Step 4c — Specific Heat vs Temperature")
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    path = out_dir / "specific_heat_vs_T.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved {path.relative_to(PROJECT_ROOT)}")

    # ── 4d: Susceptibility vs T ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, L in enumerate(L_values):
        r = all_results[L]
        ax.plot(r["T"], r["susceptibility"],
                color=COLORS[i], marker=MARKERS[i], markersize=4,
                label=f"L = {L}", markevery=3)
    ax.axvline(T_CRITICAL, color="#f0883e", linestyle="--", linewidth=1.5,
               label=f"$T_c$ = {T_CRITICAL:.4f}", alpha=0.85)
    ax.set_xlabel("Temperature  $T$  [$J/k_B$]")
    ax.set_ylabel(r"$\chi_T$  [$1/J$]")
    ax.set_title("Step 4d — Magnetic Susceptibility vs Temperature")
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    path = out_dir / "susceptibility_vs_T.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved {path.relative_to(PROJECT_ROOT)}")

    # ── Combined 2×2 panel (supervisor-ready) ─────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    panels = [
        ("mean_E_per_N",   r"$\langle E \rangle / N$  [$J$]",   "4a — Energy per Spin"),
        ("mean_M_per_N",   r"$\langle |M| \rangle / N$",        "4b — |Magnetization| per Spin"),
        ("specific_heat",  r"$C_v$  [$k_B$]",                   "4c — Specific Heat"),
        ("susceptibility", r"$\chi_T$  [$1/J$]",                "4d — Susceptibility"),
    ]

    for ax, (key, ylabel, title) in zip(axes, panels):
        for i, L in enumerate(L_values):
            r = all_results[L]
            ax.plot(r["T"], r[key], color=COLORS[i], marker=MARKERS[i],
                    markersize=3, label=f"L={L}", markevery=3)
        ax.axvline(T_CRITICAL, color="#f0883e", linestyle="--", linewidth=1.2,
                   alpha=0.85)
        ax.set_xlabel("$T$", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, framealpha=0.85)

    fig.suptitle("Python Ising Simulator — Physics Validation", fontsize=14, y=1.01)
    path = out_dir / "validation_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {path.relative_to(PROJECT_ROOT)}")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4e: Vary measurement_cycle
# ─────────────────────────────────────────────────────────────────────────────

def run_meas_cycle_study(out_dir: Path):
    """
    Step 4e: Fix L=50, T=T_c ≈ 2.269, vary measurement_cycle.
    Compare noise in Cᵥ and χT across different measurement counts.
    """
    print(f"\n{'='*60}")
    print("  Step 4e — measurement_cycle sensitivity study")
    print(f"  L=50, T={T_CRITICAL:.4f}")
    print(f"{'='*60}")

    L = 50
    T = T_CRITICAL
    beta = 1.0 / T
    meas_cycle_values = [100, 500, 1000, 5000]
    EQ = 5_000   # fixed equilibration

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for i, mc in enumerate(meas_cycle_values):
        print(f"  Running meas_cycles = {mc} …", end=" ", flush=True)

        # Run multiple independent realisations to show fluctuation
        n_runs  = 6
        Cv_vals = []
        chi_vals = []

        for run in range(n_runs):
            r = sim.run_simulation(
                L=L, T=T, J=J,
                eq_cycles=EQ,
                meas_cycles=mc,
                meas_gap=MEAS_GAP_VAL,
                seed=run * 1000 + i * 77,
            )
            Cv_vals.append(r["specific_heat"])
            chi_vals.append(r["susceptibility"])

        print(f"Cᵥ = {np.mean(Cv_vals):.4f} ± {np.std(Cv_vals):.4f},  "
              f"χ = {np.mean(chi_vals):.4f} ± {np.std(chi_vals):.4f}")

        x = i + 1
        # Plot as error bars
        axes[0].errorbar(mc, np.mean(Cv_vals), yerr=np.std(Cv_vals),
                         fmt="o", color=COLORS[i % len(COLORS)],
                         capsize=5, markersize=7, label=f"mc={mc}")
        axes[1].errorbar(mc, np.mean(chi_vals), yerr=np.std(chi_vals),
                         fmt="s", color=COLORS[i % len(COLORS)],
                         capsize=5, markersize=7, label=f"mc={mc}")

    for ax, title, ylabel in zip(
        axes,
        [r"Specific Heat $C_v$ at $T_c$", r"Susceptibility $\chi_T$ at $T_c$"],
        [r"$C_v$ [$k_B$]",                r"$\chi_T$ [$1/J$]"],
    ):
        ax.set_xlabel("measurement_cycle  (log scale)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_xscale("log")
        ax.legend(fontsize=9, framealpha=0.85)

    fig.suptitle(
        f"Step 4e — Fluctuation vs measurement_cycle\n"
        f"L={L}, T={T:.4f} ≈ Tc, eq={EQ}  (n={n_runs} independent runs each)",
        fontsize=12
    )
    fig.tight_layout()
    path = out_dir / "meas_cycle_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved {path.relative_to(PROJECT_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4f: Sample lattice images
# ─────────────────────────────────────────────────────────────────────────────

def generate_sample_images(out_dir: Path):
    """
    Step 4f: Generate representative lattice images at 3 temperatures.
    T=1.5  → ordered phase (large uniform domains)
    T≈2.27 → critical point (fractal multi-scale clusters)
    T=3.5  → disordered phase (salt-and-pepper noise)
    """
    print(f"\n{'='*60}")
    print("  Step 4f — Sample lattice images")
    print(f"{'='*60}")

    sample_dir = out_dir / "sample_configs"
    sample_dir.mkdir(parents=True, exist_ok=True)

    target_temps = {
        "T_1.50 (ordered)":    1.50,
        "T_2.27 (critical)":   2.269,
        "T_3.50 (disordered)": 3.50,
    }

    L = 128   # use full lattice for visually representative images

    # Spin colormap: -1 → black (#0d1117), +1 → white (#e6edf3)
    cmap = ListedColormap(["#0d1117", "#e6edf3"])

    apply_style()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.patch.set_facecolor("#0d1117")

    for ax, (label, T) in zip(axes, target_temps.items()):
        print(f"  Generating L={L} lattice at {label} (T={T}) …", end=" ", flush=True)

        r = sim.run_simulation(
            L=L, T=T, J=J,
            eq_cycles=10_000,    # enough to equilibrate L=128 for visual inspection
            meas_cycles=1,
            meas_gap=1,
            seed=int(T * 1000),
            store_snapshots=True,
        )
        snap = r["snapshots"][0]

        ax.imshow(snap, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"{label}\n"
                     f"T = {T:.3f},  "
                     f"⟨M⟩/N = {abs(np.sum(snap))/(L*L):.3f}",
                     fontsize=11)
        ax.axis("off")

        # Save individual image
        fname = f"T_{T:.2f}_config.png"
        fig_single, ax_s = plt.subplots(figsize=(5, 5))
        fig_single.patch.set_facecolor("#0d1117")
        ax_s.imshow(snap, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
        ax_s.set_title(f"{label}  |  T = {T:.3f}", color="#e6edf3", fontsize=12)
        ax_s.axis("off")
        fig_single.tight_layout()
        fig_single.savefig(sample_dir / fname, dpi=150, facecolor="#0d1117")
        plt.close(fig_single)
        print(f"saved {fname}")

    fig.suptitle("Step 4f — Representative Spin Configurations (L=128)",
                 fontsize=13, color="#e6edf3", y=1.02)
    fig.tight_layout()
    path = out_dir / "sample_configs_panel.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  ✓ Saved {path.relative_to(PROJECT_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
#  VALIDATION SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_validation_summary(all_results: dict):
    """
    Print a quick physics sanity-check table to stdout.
    """
    print(f"\n{'='*60}")
    print("  Physics Sanity Checks")
    print(f"{'='*60}")
    print(f"  Theoretical Tc = {T_CRITICAL:.5f}")
    print()

    for L, r in all_results.items():
        T_arr  = r["T"]
        E_arr  = r["mean_E_per_N"]
        M_arr  = r["mean_M_per_N"]
        Cv_arr = r["specific_heat"]
        chi_arr= r["susceptibility"]

        # Check 1: energy at T=1.0 (lowest T) should be near −2.0
        idx_low  = np.argmin(T_arr)
        idx_high = np.argmax(T_arr)
        E_low  = E_arr[idx_low]
        E_high = E_arr[idx_high]

        # Check 2: Cv peak temperature
        T_Cv_peak  = T_arr[np.argmax(Cv_arr)]
        T_chi_peak = T_arr[np.argmax(chi_arr)]

        # Check 3: magnetization at lowest T
        M_low  = M_arr[idx_low]
        M_high = M_arr[idx_high]

        print(f"  L = {L:3d}:")
        print(f"    ⟨E⟩/N at T={T_arr[idx_low]:.2f}: {E_low:+.4f}  (expect ≈ −2.0)")
        print(f"    ⟨E⟩/N at T={T_arr[idx_high]:.2f}: {E_high:+.4f}  (expect ≈ −0.5 to −0.7)")
        print(f"    ⟨M⟩/N at T={T_arr[idx_low]:.2f}: {M_low:.4f}  (expect ≈ 1.0)")
        print(f"    ⟨M⟩/N at T={T_arr[idx_high]:.2f}: {M_high:.4f}  (expect ≈ 0.0)")
        print(f"    Cᵥ  peak at T = {T_Cv_peak:.4f}   (expect ≈ {T_CRITICAL:.4f})")
        print(f"    χT  peak at T = {T_chi_peak:.4f}  (expect ≈ {T_CRITICAL:.4f})")
        print()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ising-GAN Physics Validation — Steps 4a through 4f"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Use full Fortran parameters (eq=500k, meas=1000). Very slow on CPU."
    )
    parser.add_argument(
        "--step", choices=["4a4d", "4e", "4f", "all"], default="all",
        help="Which sub-step(s) to run (default: all)"
    )
    parser.add_argument(
        "--L", nargs="+", type=int, default=None,
        help="Override lattice sizes (e.g., --L 30 50)"
    )
    args = parser.parse_args()

    # ── Output directories ─────────────────────────────────────────────────
    out_dir = PROJECT_ROOT / "plots" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sample_configs").mkdir(exist_ok=True)

    # ── Parameters ────────────────────────────────────────────────────────
    if args.full:
        eq_c, meas_c, meas_g = EQ_CYCLES_PROD, MEAS_CYCLES_PROD, MEAS_GAP_PROD
        print("!! Using FULL Fortran parameters — this will take many hours on CPU !!")
    else:
        eq_c, meas_c, meas_g = EQ_CYCLES_VAL, MEAS_CYCLES_VAL, MEAS_GAP_VAL
        print(f"Using VALIDATION parameters: eq={eq_c}, meas={meas_c}, gap={meas_g}")

    L_values = args.L or L_VALIDATION

    # ── Warm up JIT ───────────────────────────────────────────────────────
    print()
    sim.warmup(L=8)

    # ── Run selected steps ────────────────────────────────────────────────
    all_results = {}

    if args.step in ("4a4d", "all"):
        all_results = run_four_plots(
            L_values=L_values,
            eq_cycles=eq_c,
            meas_cycles=meas_c,
            meas_gap=meas_g,
            out_dir=out_dir,
        )
        print_validation_summary(all_results)

    if args.step in ("4e", "all"):
        run_meas_cycle_study(out_dir=out_dir)

    if args.step in ("4f", "all"):
        generate_sample_images(out_dir=out_dir)

    print(f"\n{'='*60}")
    print("  Validation complete.")
    print(f"  All plots saved to:  {out_dir.relative_to(PROJECT_ROOT)}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
