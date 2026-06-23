# Fortran Code Analysis — Step 1 Documentation
# Ising-GAN Project | Phase 1: Theory & Fortran Code Analysis

## Source File
`fortran/ising_model.f90` (read-only reference copy)

---

## 1. Hardcoded Parameters

| Parameter | Fortran Value | Notes |
|---|---|---|
| `L` | 128 | Lattice side length; grid is L × L = 16,384 spins |
| `equilibrium_cycle` | 500,000 | MC sweeps before measurement; allows thermal equilibration |
| `measurement_cycle` | 1,000 | Snapshots saved per temperature after equilibration |
| `measurement_cycle_gap` | 100 | MC sweeps between each saved snapshot (decorrelation) |
| `J` | 1.0 | Ferromagnetic coupling constant |
| `T_start` | 4.0 | First temperature; sweep descends from here |
| `dT` | `(4.0 − 1.0) / 49.0 ≈ 0.06122` | Fixed step (overrides the initial `dT = 0.05`) |
| Total temperatures | 50 | `iT = 1 … 50`; T runs from ~4.0 down to ~1.0 |
| `beta` | `1.0 / T` | Inverse temperature (k_B = 1 throughout) |

**Temperature schedule:**
```
T[i] = 4.0 - i * (3.0/49.0),  i = 0, 1, ..., 49
T[0]  = 4.000000  (highest — disordered)
T[24] ≈ 2.469388
T[~37]≈ 2.269 (critical point falls within the sweep)
T[49] ≈ 1.000000  (lowest — ordered)
```

---

## 2. Metropolis Update Rule (`PerformCycle`)

One Monte Carlo **sweep** = **L × L = 16,384** individual flip attempts.

```
For each of the L² attempts in a sweep:
  1. Pick a random site (x, y):
       x = 1 + INT(rand * L)          [Fortran 1-indexed]
       y = 1 + INT(rand * L)
  2. Compute energy cost of flipping spin at (x, y):
       ΔE = 2 × J × spins(x,y) × NeighborSum(x, y)
  3. Accept the flip:
       if ΔE ≤ 0 → flip unconditionally
       else       → flip with probability exp(−β × ΔE)
```

**Python translation (0-indexed):**
```python
x = int(np.random.random() * L)   # ∈ [0, L-1]
y = int(np.random.random() * L)
ns = spins[(x-1)%L, y] + spins[(x+1)%L, y] + spins[x, (y-1)%L] + spins[x, (y+1)%L]
dE = 2.0 * J * spins[x, y] * ns
if dE <= 0:
    spins[x, y] = -spins[x, y]
elif np.random.random() < np.exp(-beta * dE):
    spins[x, y] = -spins[x, y]
```

---

## 3. Boundary Conditions

**Periodic (toroidal)** via modular arithmetic.

Fortran (1-indexed):
```fortran
xm = MOD(x - 2 + L, L) + 1    ! left  neighbor
xp = MOD(x, L) + 1             ! right neighbor
ym = MOD(y - 2 + L, L) + 1    ! up    neighbor
yp = MOD(y, L) + 1             ! down  neighbor
```

Python (0-indexed) equivalent:
```python
xm = (x - 1) % L
xp = (x + 1) % L
ym = (y - 1) % L
yp = (y + 1) % L
```

The modular arithmetic ensures spins on the left edge interact with spins on the right edge (and top with bottom), eliminating boundary effects.

---

## 4. Energy Calculation (`TotalEnergy`)

```fortran
TotalEnergy = 0.0
DO row = 1, L
   DO col = 1, L
      TotalEnergy = TotalEnergy - J * spins(row,col) * NeighborSum(row,col)
   END DO
END DO
TotalEnergy = 0.5 * TotalEnergy
```

The factor of **0.5** corrects for double-counting each pair (each bond is visited twice in the double loop). 

**Equivalent NumPy:**
```python
right = np.roll(spins, -1, axis=1)
down  = np.roll(spins, -1, axis=0)
E = -J * np.sum(spins * (right + down))   # counts each bond once, no 0.5 needed
```

**Verification:** For an all-up lattice (all +1), each spin has 4 neighbors each = +1:
- NeighborSum = 4 for every site
- Contribution per site = −J × (+1) × 4 = −4J
- Sum over L² sites × 0.5 = −2J × L² ✓ (ground state energy)

---

## 5. Magnetization Calculation (`TotalMagnetization`)

```fortran
TotalMagnetization = SUM of all spins(row,col)
! Called as: magnetization = ABS(TotalMagnetization(...))
```

The **absolute value** is taken at the call site to remove sign degeneracy (ferromagnetic domains can align either way). The Python version returns `abs(np.sum(spins))`.

**Observable:** `⟨M⟩/N` where N = L² (per-spin magnetization, range [0, 1]).

---

## 6. Correlation Function (`CalculateCorrelation`)

```fortran
DO shift = 1, L
   correlation(shift) = 0.0
   DO row = 1, L
      DO col = 1, L
         correlation(shift) += spins(row,col) * spins(MOD(row+shift-1, L)+1, col)
      END DO
   END DO
   correlation(shift) = correlation(shift) / (L*L)
END DO
```

- **Direction:** shift is applied **along the row (first) index** — i.e., the vertical direction.
- **shift = 1** → autocorrelation with the spin one row below.
- **shift = L** → wraps around (periodic), so `spins(row,col) * spins(row,col)` = 1 always.
- Averaged over all L × L sites per snapshot.
- Accumulated over all measurement steps; final scalar:
  ```
  correlation_function = SUM(correlation_array) / (L * measurement_cycle)
  ```

**Python equivalent:**
```python
for shift in range(1, L+1):
    corr[shift-1] = np.sum(spins * np.roll(spins, -shift, axis=0)) / (L*L)
```

---

## 7. Thermodynamic Observables (from accumulators)

After `measurement_cycle` steps, per temperature:

| Observable | Formula (Fortran) | Physical Meaning |
|---|---|---|
| `⟨E⟩/N` | `total_energy / (measurement_cycle × N)` | Mean energy per spin |
| `⟨M⟩/N` | `total_magnetization / (measurement_cycle × N)` | Mean |magnetization| per spin |
| `Cᵥ` | `(⟨E²⟩ − ⟨E⟩²) / (T² × N)` | Specific heat |
| `χT` | `(⟨M²⟩ − ⟨M⟩²) / (T × N)` | Magnetic susceptibility |
| `corr_fn` | `SUM(correlation) / (L × measurement_cycle)` | Scalar correlation |
| `corr_length` | `−1 / ln(corr_fn)` | Correlation length estimate |

---

## 8. Phase Label

```fortran
IF (T < 2.269) THEN
    phase = 'F'    ! Ferromagnetic (ordered)
ELSE
    phase = 'P'    ! Paramagnetic (disordered)
END IF
```

The critical temperature Tc = 2/ln(1+√2) ≈ **2.26919** is hard-coded as the boundary.

---

## 9. Output Files Generated

| Filename | Format | Content |
|---|---|---|
| `Ising_2D_L_128_cycles_1000.dat` | Text | One row per temperature: T, ⟨E⟩/N, ⟨M⟩/N, Cᵥ, χT, corr_fn, corr_length |
| `Ising_2D_L_128_cycles_1000.csv` | CSV | ALL snapshots concatenated: Temperature, Phase, spin_0, …, spin_16383 |
| `Ising_2D_L_128_cycles_1000_T_<iT>.csv` | CSV | Same format, snapshots for temperature index iT only |
| `Ising_2D_L_128_cycles_1000_T_<iT>.bin` | Binary stream | Raw int8 spin array (128×128) per snapshot, unformatted |

**CSV column order:** `Temperature, Phase, spin_0, spin_1, …, spin_16383`  
where `spin_k = spins[k // L, k % L]` (row-major, 0-indexed).

---

## 10. Initialization (`InitializeSpins`)

Random ±1 assignment: `rand < 0.5 → +1`, else `−1`.  
Equivalent to `np.random.choice([-1, 1], size=(L, L))`.

No fixed seed is set — each run is stochastic. The Python implementation will support optional seeding for reproducibility.
