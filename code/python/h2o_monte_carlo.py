#!/usr/bin/env python3
"""
Monte Carlo H₂O Formation Simulation

Monte Carlo is inherently stable (no time integration issues).
Samples configurations from Boltzmann distribution: P ∝ exp(-E/kT)

Key physics:
- At high T: all configurations roughly equally likely → no persistent bonds
- At low T: low-energy configurations dominate → stable molecules form
- Cooling: gradual transition from chaos to order

This directly demonstrates:
1. DISSIPATION: Cooling increases hierarchy (stable geometry favored)
2. SURVIVAL BIAS: High-energy configs rejected at low T
3. RESONANCE: Morse potential well traps atoms at equilibrium distance
"""

import numpy as np
import json
from pathlib import Path

# Constants
KB = 8.617e-5  # eV/K

# Potential parameters
# Morse potential for O-H: V(r) = D(1 - e^(-a(r-r0)))^2 - D
D_OH = 4.8     # eV (bond energy)
ALPHA_OH = 2.0  # Å^-1
R0_OH = 0.96    # Å (equilibrium)

# LJ for non-bonded: V(r) = 4ε[(σ/r)^12 - (σ/r)^6]
EPSILON_HH = 0.005  # eV (very weak)
SIGMA_HH = 2.4      # Å
EPSILON_OO = 0.010  # eV
SIGMA_OO = 3.0      # Å

# H₂O angle: V(θ) = k(θ - θ0)² (only if 2 H bonded to O)
K_ANGLE = 1.0           # eV/rad²
THETA0 = 104.5 * np.pi / 180  # radians

# Bond threshold
R_BOND = 1.4  # Å


def morse_energy(r, D=D_OH, alpha=ALPHA_OH, r0=R0_OH):
    """Morse potential energy."""
    return D * (1 - np.exp(-alpha * (r - r0)))**2 - D


def lj_energy(r, epsilon, sigma):
    """Lennard-Jones energy."""
    if r < 0.5:
        return 1e10  # Hard core
    sr = sigma / r
    return 4 * epsilon * (sr**12 - sr**6)


def angle_energy(pos_O, pos_H1, pos_H2, box_size):
    """Angular potential energy for H-O-H angle."""
    v1 = pos_H1 - pos_O
    v2 = pos_H2 - pos_O
    # Minimum image
    v1 = v1 - box_size * np.round(v1 / box_size)
    v2 = v2 - box_size * np.round(v2 / box_size)

    r1 = np.linalg.norm(v1)
    r2 = np.linalg.norm(v2)

    if r1 < 0.1 or r2 < 0.1:
        return 0

    cos_theta = np.dot(v1, v2) / (r1 * r2)
    cos_theta = np.clip(cos_theta, -1, 1)
    theta = np.arccos(cos_theta)

    return K_ANGLE * (theta - THETA0)**2


def total_energy(pos_H, pos_O, box_size):
    """Compute total potential energy."""
    n_H = len(pos_H)
    n_O = len(pos_O)
    E = 0.0

    # O-H interactions (Morse)
    bonds_per_O = [[] for _ in range(n_O)]
    for i in range(n_O):
        for j in range(n_H):
            dr = pos_H[j] - pos_O[i]
            dr = dr - box_size * np.round(dr / box_size)
            r = np.linalg.norm(dr)

            E += morse_energy(r)

            if r < R_BOND:
                bonds_per_O[i].append(j)

    # Angle potential for H₂O (if O has 2 H bonded)
    for i, bonds in enumerate(bonds_per_O):
        if len(bonds) >= 2:
            # Take first two bonded H
            h1, h2 = bonds[0], bonds[1]
            E += angle_energy(pos_O[i], pos_H[h1], pos_H[h2], box_size)

    # H-H (weak LJ)
    for i in range(n_H):
        for j in range(i+1, n_H):
            dr = pos_H[j] - pos_H[i]
            dr = dr - box_size * np.round(dr / box_size)
            r = np.linalg.norm(dr)
            if r < 5.0:
                E += lj_energy(r, EPSILON_HH, SIGMA_HH)

    # O-O (weak LJ)
    for i in range(n_O):
        for j in range(i+1, n_O):
            dr = pos_O[j] - pos_O[i]
            dr = dr - box_size * np.round(dr / box_size)
            r = np.linalg.norm(dr)
            if r < 6.0:
                E += lj_energy(r, EPSILON_OO, SIGMA_OO)

    return E


def count_molecules(pos_H, pos_O, box_size, r_cutoff=R_BOND):
    """Count H₂O molecules and OH radicals."""
    n_O = len(pos_O)
    n_H = len(pos_H)

    bonds_per_O = [[] for _ in range(n_O)]

    for i in range(n_O):
        for j in range(n_H):
            dr = pos_H[j] - pos_O[i]
            dr = dr - box_size * np.round(dr / box_size)
            r = np.linalg.norm(dr)
            if r < r_cutoff:
                bonds_per_O[i].append((j, r))

    n_h2o = 0
    n_oh = 0
    molecules = []

    for i, bonds in enumerate(bonds_per_O):
        if len(bonds) >= 2:
            n_h2o += 1
            # Compute angle
            h1, r1 = bonds[0]
            h2, r2 = bonds[1]
            v1 = pos_H[h1] - pos_O[i]
            v2 = pos_H[h2] - pos_O[i]
            v1 = v1 - box_size * np.round(v1 / box_size)
            v2 = v2 - box_size * np.round(v2 / box_size)
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            angle = np.arccos(np.clip(cos_theta, -1, 1)) * 180 / np.pi
            molecules.append({"type": "H2O", "O": i, "H": [h1, h2],
                            "r_OH": [r1, r2], "angle": angle})
        elif len(bonds) == 1:
            n_oh += 1
            h, r = bonds[0]
            molecules.append({"type": "OH", "O": i, "H": [h], "r_OH": [r]})

    return n_h2o, n_oh, molecules


def metropolis_step(pos_H, pos_O, box_size, T, step_size=0.1):
    """Single Metropolis Monte Carlo step."""
    n_H = len(pos_H)
    n_O = len(pos_O)

    # Current energy
    E_old = total_energy(pos_H, pos_O, box_size)

    # Try moving a random atom
    if np.random.random() < n_H / (n_H + n_O):
        # Move H
        idx = np.random.randint(n_H)
        pos_H_new = pos_H.copy()
        pos_H_new[idx] += np.random.uniform(-step_size, step_size, 3)
        pos_H_new[idx] = pos_H_new[idx] % box_size
        E_new = total_energy(pos_H_new, pos_O, box_size)

        if E_new < E_old or np.random.random() < np.exp(-(E_new - E_old) / (KB * T)):
            return pos_H_new, pos_O, E_new, True
        return pos_H, pos_O, E_old, False
    else:
        # Move O
        idx = np.random.randint(n_O)
        pos_O_new = pos_O.copy()
        pos_O_new[idx] += np.random.uniform(-step_size, step_size, 3)
        pos_O_new[idx] = pos_O_new[idx] % box_size
        E_new = total_energy(pos_H, pos_O_new, box_size)

        if E_new < E_old or np.random.random() < np.exp(-(E_new - E_old) / (KB * T)):
            return pos_H, pos_O_new, E_new, True
        return pos_H, pos_O, E_old, False


def run_simulation(n_H=16, n_O=8, box_size=8.0, T_high=5000, T_low=300,
                   n_cool_steps=50000, n_equil_steps=20000):
    """
    Run Monte Carlo simulation with cooling.

    High T → Low T transition demonstrates geometry selection.
    """
    print("╔" + "═" * 60 + "╗")
    print("║" + " MONTE CARLO H₂O FORMATION ".center(60) + "║")
    print("╚" + "═" * 60 + "╝\n")

    print(f"Configuration:")
    print(f"  {n_H} H + {n_O} O atoms in {box_size}³ Å box")
    print(f"  Cooling: {T_high} K → {T_low} K over {n_cool_steps} steps")
    print(f"  Equilibration at {T_low} K: {n_equil_steps} steps\n")

    # Initialize positions (grid to avoid overlaps)
    np.random.seed(42)

    # Place atoms with some initial separation
    all_pos = []
    n_atoms = n_H + n_O
    for _ in range(n_atoms):
        while True:
            pos = np.random.uniform(1, box_size-1, 3)
            if all(np.linalg.norm(pos - p) > 1.5 for p in all_pos):
                all_pos.append(pos)
                break

    pos_H = np.array(all_pos[:n_H])
    pos_O = np.array(all_pos[n_H:])

    # Initial energy
    E = total_energy(pos_H, pos_O, box_size)
    print(f"Initial energy: {E:.2f} eV")
    n_h2o, n_oh, _ = count_molecules(pos_H, pos_O, box_size)
    print(f"Initial molecules: {n_h2o} H₂O, {n_oh} OH\n")

    # History
    history = {"T": [], "E": [], "h2o": [], "oh": [], "accept": []}

    print(f"{'Phase':<12} {'Step':>8} {'T (K)':>8} {'E (eV)':>12} {'H₂O':>5} {'OH':>5} {'Accept':>8}")
    print("-" * 62)

    # Phase 1: Cooling
    accepted = 0
    total = 0
    for step in range(n_cool_steps):
        # Linear cooling schedule
        T = T_high + (T_low - T_high) * step / n_cool_steps

        # Adaptive step size
        step_size = 0.3 * (T / T_high) + 0.05

        pos_H, pos_O, E, accept = metropolis_step(pos_H, pos_O, box_size, T, step_size)
        accepted += accept
        total += 1

        if step % (n_cool_steps // 10) == 0:
            n_h2o, n_oh, _ = count_molecules(pos_H, pos_O, box_size)
            accept_rate = accepted / total if total > 0 else 0
            print(f"{'Cooling':<12} {step:>8} {T:>8.0f} {E:>12.2f} {n_h2o:>5} {n_oh:>5} {accept_rate:>8.1%}")
            history["T"].append(T)
            history["E"].append(E)
            history["h2o"].append(n_h2o)
            history["oh"].append(n_oh)
            history["accept"].append(accept_rate)
            accepted = 0
            total = 0

    # Phase 2: Equilibration at low T
    print("-" * 62)
    for step in range(n_equil_steps):
        pos_H, pos_O, E, accept = metropolis_step(pos_H, pos_O, box_size, T_low, 0.05)
        accepted += accept
        total += 1

        if step % (n_equil_steps // 5) == 0:
            n_h2o, n_oh, _ = count_molecules(pos_H, pos_O, box_size)
            accept_rate = accepted / total if total > 0 else 0
            print(f"{'Equilib':<12} {step:>8} {T_low:>8.0f} {E:>12.2f} {n_h2o:>5} {n_oh:>5} {accept_rate:>8.1%}")

    # Final analysis
    print("\n" + "=" * 62)
    print("FINAL STATE ANALYSIS")
    print("=" * 62)

    n_h2o, n_oh, molecules = count_molecules(pos_H, pos_O, box_size)
    efficiency = n_h2o / n_O * 100

    print(f"\nMolecule formation:")
    print(f"  Complete H₂O: {n_h2o} / {n_O} possible ({efficiency:.0f}%)")
    print(f"  OH radicals: {n_oh}")
    print(f"  Unbonded O: {n_O - n_h2o - n_oh}")
    print(f"  Final energy: {E:.2f} eV")

    # Show molecule geometries
    print("\nFormed molecules:")
    for mol in molecules:
        if mol["type"] == "H2O":
            r1, r2 = mol["r_OH"]
            angle = mol["angle"]
            # Check if geometry is good
            r_ok = 0.85 < r1 < 1.1 and 0.85 < r2 < 1.1
            a_ok = 95 < angle < 115
            status = "✓" if (r_ok and a_ok) else "~"
            print(f"  {status} H₂O: O-H = {r1:.2f}, {r2:.2f} Å; ∠HOH = {angle:.1f}° "
                  f"(ideal: 0.96 Å, 104.5°)")
        else:
            print(f"    OH: O-H = {mol['r_OH'][0]:.2f} Å")

    # Theory connection
    print("\n" + "=" * 62)
    print("PHYSICS INTERPRETATION")
    print("=" * 62)
    print(f"""
Monte Carlo directly demonstrates BOLTZMANN SELECTION:

At T = {T_high} K (high):
  • kT = {KB * T_high:.2f} eV >> D_OH = {D_OH} eV × (1/10)
  • All configurations accessible → no stable bonds
  • P(bonded) ≈ P(unbonded) → random motion

At T = {T_low} K (low):
  • kT = {KB * T_low:.3f} eV << D_OH = {D_OH} eV
  • Only low-energy configs survive Metropolis test
  • P(bonded) >> P(unbonded) → stable molecules

The cooling process IS geometry selection:
  • High-energy (wrong geometry) → rejected
  • Low-energy (H₂O geometry) → accepted
  • Hierarchy H = D_OH/(kT) increases: {D_OH/(KB*T_high):.0f} → {D_OH/(KB*T_low):.0f}

This explains supernova → molecules → life:
  • 10⁹ K: plasma (no molecules)
  • 10⁴ K: atoms form
  • 10³ K: molecules form (H₂O, CO, organics)
  • 300 K: complex chemistry → life
""")

    # Save results
    results = {
        "parameters": {
            "n_H": n_H, "n_O": n_O, "box_size": box_size,
            "T_high": T_high, "T_low": T_low,
            "n_cool_steps": n_cool_steps, "n_equil_steps": n_equil_steps
        },
        "final": {
            "n_h2o": n_h2o, "n_oh": n_oh,
            "efficiency_percent": efficiency,
            "energy_eV": float(E),
            "molecules": molecules
        },
        "history": history,
        "theory": {
            "hierarchy_high_T": D_OH / (KB * T_high),
            "hierarchy_low_T": D_OH / (KB * T_low),
            "kT_high_eV": KB * T_high,
            "kT_low_eV": KB * T_low
        }
    }

    output_dir = Path(__file__).parent.parent.parent / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "h2o_monte_carlo.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    results = run_simulation(
        n_H=20,           # 20 H atoms
        n_O=10,           # 10 O atoms (2:1 for H₂O)
        box_size=9.0,     # Reasonable density
        T_high=6000,      # Start very hot
        T_low=300,        # Cool to room temp
        n_cool_steps=100000,
        n_equil_steps=30000
    )

    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)
    print(f"H₂O molecules formed: {results['final']['n_h2o']} / 10 possible")
    print(f"Formation efficiency: {results['final']['efficiency_percent']:.0f}%")
    print(f"Hierarchy increase: {results['theory']['hierarchy_high_T']:.0f}× → "
          f"{results['theory']['hierarchy_low_T']:.0f}×")
