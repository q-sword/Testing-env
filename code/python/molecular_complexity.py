#!/usr/bin/env python3
"""
Multi-Molecular Formation Simulation

Extends H₂O simulation to more complex molecules:
1. CO₂ - Linear geometry (O=C=O, 180°)
2. Glycine - Simplest amino acid (NH₂-CH₂-COOH)
3. Adenine - DNA base (C₅H₅N₅)

Key insight: As molecular complexity increases:
- More atoms → more degrees of freedom
- Correct geometry becomes rarer in random configs
- Hierarchy H needed for stability increases
- Formation requires SEQUENTIAL assembly (hierarchical)

This explains the "ladder of complexity":
Atoms → Simple molecules → Complex molecules → Polymers → Life
Each step requires increasing hierarchy and specific conditions.
"""

import numpy as np
import json
from pathlib import Path

# Constants
KB = 8.617e-5  # eV/K

# ============================================================
# MOLECULAR PARAMETERS
# ============================================================

# Bond energies (eV)
BOND_ENERGIES = {
    'C=O': 7.5,    # Double bond in CO₂
    'C-O': 3.6,    # Single bond
    'C-C': 3.6,    # C-C single
    'C=C': 6.3,    # C=C double
    'C-H': 4.3,    # C-H
    'C-N': 3.0,    # C-N single
    'C=N': 6.2,    # C=N double
    'N-H': 4.0,    # N-H
    'O-H': 4.8,    # O-H
    'N=N': 9.8,    # N=N triple (N₂)
}

# Bond lengths (Å)
BOND_LENGTHS = {
    'C=O': 1.16,   # CO₂
    'C-O': 1.43,
    'C-C': 1.54,
    'C=C': 1.34,
    'C-H': 1.09,
    'C-N': 1.47,
    'C=N': 1.27,
    'N-H': 1.01,
    'O-H': 0.96,
}

# Ideal angles (degrees)
ANGLES = {
    'O=C=O': 180.0,    # CO₂ linear
    'H-O-H': 104.5,    # H₂O
    'H-N-H': 107.0,    # NH₂ group
    'H-C-H': 109.5,    # tetrahedral
    'C-C-C': 109.5,    # tetrahedral backbone
    'O-C-O': 120.0,    # carboxyl
    'N-C-N': 120.0,    # purine ring
}


class Molecule:
    """Base class for molecular simulations."""

    def __init__(self, name, atoms, bonds, angles):
        """
        atoms: dict of {element: count}
        bonds: list of (atom1_idx, atom2_idx, bond_type)
        angles: list of (atom1, atom2, atom3, ideal_angle)
        """
        self.name = name
        self.atoms = atoms
        self.bonds = bonds
        self.angles = angles
        self.n_atoms = sum(atoms.values())

    def bond_energy(self, bond_type):
        return BOND_ENERGIES.get(bond_type, 3.0)

    def bond_length(self, bond_type):
        return BOND_LENGTHS.get(bond_type, 1.5)


# ============================================================
# CO₂ SIMULATION
# ============================================================

def simulate_co2_formation(n_C=8, n_O=16, box_size=10.0, T_high=8000, T_low=300,
                           n_steps=80000):
    """
    Simulate CO₂ formation: C + O + O → O=C=O

    CO₂ has linear geometry (180°) - very specific constraint.
    Tests whether geometry selection produces correct structure.
    """
    print("=" * 70)
    print("CO₂ FORMATION SIMULATION")
    print("=" * 70)

    # Morse parameters for C=O double bond
    D_CO = 7.5   # eV (strong double bond)
    ALPHA = 2.2  # Å^-1
    R0 = 1.16    # Å

    # Angle potential for O=C=O (should be 180°)
    K_ANGLE = 2.0  # eV/rad² (strong preference for linear)
    THETA0 = np.pi  # 180 degrees

    # Masses
    M_C = 12.0
    M_O = 16.0

    def morse_energy(r, D=D_CO, alpha=ALPHA, r0=R0):
        if r < 0.5:
            return 1e6
        return D * (1 - np.exp(-alpha * (r - r0)))**2 - D

    def lj_repulsion(r, epsilon=0.01, sigma=2.5):
        if r < 0.5:
            return 1e6
        if r > 5.0:
            return 0
        sr = sigma / r
        return 4 * epsilon * (sr**12 - sr**6)

    def compute_energy(pos_C, pos_O, box_size):
        n_C = len(pos_C)
        n_O = len(pos_O)
        E = 0.0

        # C-O interactions (Morse for potential bonds)
        bonds_per_C = [[] for _ in range(n_C)]
        for i in range(n_C):
            for j in range(n_O):
                dr = pos_O[j] - pos_C[i]
                dr = dr - box_size * np.round(dr / box_size)
                r = np.linalg.norm(dr)
                E += morse_energy(r)
                if r < 1.5:  # Bond cutoff
                    bonds_per_C[i].append((j, dr))

        # Angle penalty for C with 2 O bonded (should be linear)
        for i, bonds in enumerate(bonds_per_C):
            if len(bonds) >= 2:
                # Get vectors to first two O atoms
                _, v1 = bonds[0]
                _, v2 = bonds[1]
                r1, r2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if r1 > 0.1 and r2 > 0.1:
                    cos_theta = np.dot(v1, v2) / (r1 * r2)
                    theta = np.arccos(np.clip(cos_theta, -1, 1))
                    # Penalty for deviation from 180°
                    E += K_ANGLE * (theta - THETA0)**2

        # C-C repulsion
        for i in range(n_C):
            for j in range(i+1, n_C):
                dr = pos_C[j] - pos_C[i]
                dr = dr - box_size * np.round(dr / box_size)
                r = np.linalg.norm(dr)
                E += lj_repulsion(r)

        # O-O repulsion
        for i in range(n_O):
            for j in range(i+1, n_O):
                dr = pos_O[j] - pos_O[i]
                dr = dr - box_size * np.round(dr / box_size)
                r = np.linalg.norm(dr)
                E += lj_repulsion(r, epsilon=0.005)

        return E, bonds_per_C

    def count_co2(pos_C, pos_O, box_size):
        """Count complete CO₂ molecules."""
        n_co2 = 0
        molecules = []
        _, bonds_per_C = compute_energy(pos_C, pos_O, box_size)

        for i, bonds in enumerate(bonds_per_C):
            if len(bonds) >= 2:
                # Check if linear (angle close to 180°)
                _, v1 = bonds[0]
                _, v2 = bonds[1]
                r1, r2 = np.linalg.norm(v1), np.linalg.norm(v2)
                cos_theta = np.dot(v1, v2) / (r1 * r2 + 1e-10)
                angle = np.arccos(np.clip(cos_theta, -1, 1)) * 180 / np.pi

                if angle > 160:  # Close to linear
                    n_co2 += 1
                    molecules.append({
                        'C': i, 'r_CO': [r1, r2], 'angle': angle
                    })

        return n_co2, molecules

    # Initialize
    np.random.seed(123)
    pos_C = np.random.uniform(1, box_size-1, (n_C, 3))
    pos_O = np.random.uniform(1, box_size-1, (n_O, 3))

    print(f"\nConfiguration: {n_C} C + {n_O} O atoms")
    print(f"Cooling: {T_high} K → {T_low} K over {n_steps} steps")

    # Run MC
    history = []
    for step in range(n_steps):
        T = T_high + (T_low - T_high) * step / n_steps

        # Try random move
        if np.random.random() < 0.5:
            idx = np.random.randint(n_C)
            pos_new = pos_C.copy()
            step_size = 0.3 * (T / T_high) + 0.05
            pos_new[idx] = (pos_new[idx] + np.random.uniform(-step_size, step_size, 3)) % box_size
            E_old, _ = compute_energy(pos_C, pos_O, box_size)
            E_new, _ = compute_energy(pos_new, pos_O, box_size)
            if E_new < E_old or np.random.random() < np.exp(-(E_new - E_old)/(KB * T)):
                pos_C = pos_new
        else:
            idx = np.random.randint(n_O)
            pos_new = pos_O.copy()
            step_size = 0.3 * (T / T_high) + 0.05
            pos_new[idx] = (pos_new[idx] + np.random.uniform(-step_size, step_size, 3)) % box_size
            E_old, _ = compute_energy(pos_C, pos_O, box_size)
            E_new, _ = compute_energy(pos_C, pos_new, box_size)
            if E_new < E_old or np.random.random() < np.exp(-(E_new - E_old)/(KB * T)):
                pos_O = pos_new

        if step % (n_steps // 10) == 0:
            E, _ = compute_energy(pos_C, pos_O, box_size)
            n_co2, _ = count_co2(pos_C, pos_O, box_size)
            print(f"  Step {step:>6}: T={T:.0f} K, E={E:.1f} eV, CO₂={n_co2}/{n_C}")
            history.append({'step': step, 'T': T, 'E': E, 'n_co2': n_co2})

    # Final count
    E_final, _ = compute_energy(pos_C, pos_O, box_size)
    n_co2, molecules = count_co2(pos_C, pos_O, box_size)

    print(f"\nFinal: {n_co2} / {n_C} CO₂ molecules formed ({100*n_co2/n_C:.0f}%)")
    for mol in molecules[:5]:  # Show first 5
        print(f"  CO₂: C-O = {mol['r_CO'][0]:.2f}, {mol['r_CO'][1]:.2f} Å, "
              f"∠OCO = {mol['angle']:.1f}° (ideal: 1.16 Å, 180°)")

    return {
        'n_co2': n_co2,
        'max_possible': n_C,
        'efficiency': n_co2 / n_C,
        'molecules': molecules,
        'hierarchy_ratio': D_CO / (KB * T_low)
    }


# ============================================================
# GLYCINE (SIMPLEST AMINO ACID) SIMULATION
# ============================================================

def simulate_glycine_formation(n_units=5, box_size=12.0, T_high=6000, T_low=300,
                                n_steps=100000):
    """
    Simulate glycine formation: NH₂-CH₂-COOH

    Glycine structure:
    - Amino group: NH₂ (N bonded to 2 H)
    - Central C bonded to H, H, N, C
    - Carboxyl group: COOH (C bonded to =O and -OH)

    This requires HIERARCHICAL ASSEMBLY:
    1. First: NH₂ forms (N + 2H)
    2. Then: CH₂ forms (C + 2H)
    3. Then: COOH forms (C + O + O + H)
    4. Finally: Connect NH₂-CH₂-COOH

    Atoms per glycine: N(1) + C(2) + O(2) + H(5) = 10 atoms
    """
    print("\n" + "=" * 70)
    print("GLYCINE (AMINO ACID) FORMATION SIMULATION")
    print("=" * 70)

    # Atoms needed per glycine
    n_N = n_units
    n_C = 2 * n_units
    n_O = 2 * n_units
    n_H = 5 * n_units

    print(f"\nTarget: {n_units} glycine molecules")
    print(f"Atoms: {n_N} N + {n_C} C + {n_O} O + {n_H} H = {n_N+n_C+n_O+n_H} total")

    # Bond parameters
    bonds = {
        'N-H': (4.0, 1.01),
        'C-H': (4.3, 1.09),
        'C-N': (3.0, 1.47),
        'C-C': (3.6, 1.54),
        'C-O': (3.6, 1.43),
        'C=O': (7.5, 1.23),
        'O-H': (4.8, 0.96),
    }

    def get_bond_params(type1, type2):
        """Get bond energy and length for atom pair."""
        key = f"{type1}-{type2}"
        if key in bonds:
            return bonds[key]
        key = f"{type2}-{type1}"
        if key in bonds:
            return bonds[key]
        return (0.5, 3.0)  # Weak default

    # Initialize positions
    np.random.seed(456)

    atoms = []
    for _ in range(n_N):
        atoms.append(('N', np.random.uniform(1, box_size-1, 3)))
    for _ in range(n_C):
        atoms.append(('C', np.random.uniform(1, box_size-1, 3)))
    for _ in range(n_O):
        atoms.append(('O', np.random.uniform(1, box_size-1, 3)))
    for _ in range(n_H):
        atoms.append(('H', np.random.uniform(1, box_size-1, 3)))

    def compute_energy(atoms, box_size):
        """Compute total energy."""
        E = 0.0
        n = len(atoms)
        for i in range(n):
            for j in range(i+1, n):
                t1, p1 = atoms[i]
                t2, p2 = atoms[j]
                dr = p2 - p1
                dr = dr - box_size * np.round(dr / box_size)
                r = np.linalg.norm(dr)

                D, r0 = get_bond_params(t1, t2)

                if r < 0.5:
                    E += 1e6
                elif r < 2.0:
                    # Morse potential
                    alpha = 2.0
                    E += D * (1 - np.exp(-alpha * (r - r0)))**2 - D
                else:
                    # Weak LJ
                    if r < 5.0:
                        sr = 2.5 / r
                        E += 0.01 * (sr**12 - sr**6)
        return E

    def count_bonds(atoms, box_size, r_cut=1.8):
        """Count bonds by type."""
        bond_counts = {}
        n = len(atoms)
        for i in range(n):
            for j in range(i+1, n):
                t1, p1 = atoms[i]
                t2, p2 = atoms[j]
                dr = p2 - p1
                dr = dr - box_size * np.round(dr / box_size)
                r = np.linalg.norm(dr)

                if r < r_cut:
                    key = tuple(sorted([t1, t2]))
                    bond_counts[key] = bond_counts.get(key, 0) + 1

        return bond_counts

    def estimate_glycine(bond_counts):
        """Estimate complete glycine molecules from bonds."""
        # Glycine needs: 2 N-H, 2 C-H, 1 C-N, 1 C-C, 1 C=O (or C-O), 1 O-H
        n_nh = bond_counts.get(('H', 'N'), 0)
        n_ch = bond_counts.get(('C', 'H'), 0)
        n_cn = bond_counts.get(('C', 'N'), 0)
        n_cc = bond_counts.get(('C', 'C'), 0)
        n_co = bond_counts.get(('C', 'O'), 0)
        n_oh = bond_counts.get(('H', 'O'), 0)

        # Limiting factor
        glycine_est = min(n_nh // 2, n_ch // 2, n_cn, n_cc, n_co // 2, n_oh)
        return glycine_est

    print(f"Cooling: {T_high} K → {T_low} K over {n_steps} steps\n")

    # Run MC simulation
    for step in range(n_steps):
        T = T_high + (T_low - T_high) * step / n_steps

        # Random move
        idx = np.random.randint(len(atoms))
        t, pos = atoms[idx]
        step_size = 0.3 * (T / T_high) + 0.03

        atoms_new = atoms.copy()
        new_pos = (pos + np.random.uniform(-step_size, step_size, 3)) % box_size
        atoms_new[idx] = (t, new_pos)

        E_old = compute_energy(atoms, box_size)
        E_new = compute_energy(atoms_new, box_size)

        if E_new < E_old or np.random.random() < np.exp(-(E_new - E_old)/(KB * T)):
            atoms = atoms_new

        if step % (n_steps // 10) == 0:
            E = compute_energy(atoms, box_size)
            bonds = count_bonds(atoms, box_size)
            n_gly = estimate_glycine(bonds)
            print(f"  Step {step:>6}: T={T:.0f} K, E={E:.1f} eV, ~glycine={n_gly}")

    # Final analysis
    E_final = compute_energy(atoms, box_size)
    bonds_final = count_bonds(atoms, box_size)
    n_glycine = estimate_glycine(bonds_final)

    print(f"\nFinal bond counts:")
    for bond_type, count in sorted(bonds_final.items()):
        expected = ""
        if bond_type == ('H', 'N'):
            expected = f" (need {2*n_units} for {n_units} glycine)"
        elif bond_type == ('C', 'H'):
            expected = f" (need {2*n_units} for {n_units} glycine)"
        elif bond_type == ('C', 'N'):
            expected = f" (need {n_units} for {n_units} glycine)"
        print(f"  {bond_type[0]}-{bond_type[1]}: {count}{expected}")

    print(f"\nEstimated glycine molecules: ~{n_glycine} / {n_units}")

    # Hierarchy calculation
    avg_bond_E = np.mean([D for D, r in bonds.values()])
    H_high = avg_bond_E / (KB * T_high)
    H_low = avg_bond_E / (KB * T_low)

    print(f"\nHierarchy: {H_high:.0f} → {H_low:.0f} (×{H_low/H_high:.0f} increase)")

    return {
        'estimated_glycine': n_glycine,
        'target': n_units,
        'bonds': {f"{k[0]}-{k[1]}": v for k, v in bonds_final.items()},
        'hierarchy_low': H_low
    }


# ============================================================
# ADENINE (DNA BASE) SIMULATION
# ============================================================

def simulate_adenine_formation(n_units=3, box_size=15.0, T_high=5000, T_low=300,
                                n_steps=120000):
    """
    Simulate adenine formation: C₅H₅N₅

    Adenine structure (purine base):
    - Two fused rings: 6-membered + 5-membered
    - Contains: 5 C, 5 N, 5 H
    - Multiple C-N, C=N, N-H bonds
    - Highly specific geometry

    This is MUCH harder than H₂O or CO₂:
    - 15 atoms per molecule
    - Ring closure requires precise geometry
    - Multiple resonance structures

    Demonstrates: Complex molecules need MORE hierarchy (lower T, longer time)
    """
    print("\n" + "=" * 70)
    print("ADENINE (DNA BASE) FORMATION SIMULATION")
    print("=" * 70)

    # Atoms per adenine: C₅H₅N₅
    n_C = 5 * n_units
    n_H = 5 * n_units
    n_N = 5 * n_units

    print(f"\nTarget: {n_units} adenine molecules (C₅H₅N₅)")
    print(f"Atoms: {n_C} C + {n_H} H + {n_N} N = {n_C+n_H+n_N} total")

    # Bond parameters (adenine has aromatic character)
    bond_params = {
        'C-N': (4.5, 1.35),  # Aromatic C-N
        'C=N': (6.0, 1.30),
        'C-C': (4.8, 1.40),  # Aromatic
        'C-H': (4.3, 1.08),
        'N-H': (4.0, 1.01),
    }

    # Initialize
    np.random.seed(789)
    atoms = []
    for _ in range(n_C):
        atoms.append(('C', np.random.uniform(2, box_size-2, 3)))
    for _ in range(n_H):
        atoms.append(('H', np.random.uniform(2, box_size-2, 3)))
    for _ in range(n_N):
        atoms.append(('N', np.random.uniform(2, box_size-2, 3)))

    def compute_energy(atoms, box_size):
        E = 0.0
        n = len(atoms)
        for i in range(n):
            for j in range(i+1, n):
                t1, p1 = atoms[i]
                t2, p2 = atoms[j]
                dr = p2 - p1
                dr = dr - box_size * np.round(dr / box_size)
                r = np.linalg.norm(dr)

                # Get bond params
                key = f"{t1}-{t2}"
                if key not in bond_params:
                    key = f"{t2}-{t1}"
                D, r0 = bond_params.get(key, (0.5, 3.0))

                if r < 0.4:
                    E += 1e6
                elif r < 2.0:
                    alpha = 2.0
                    E += D * (1 - np.exp(-alpha * (r - r0)))**2 - D
                elif r < 5.0:
                    sr = 2.5 / r
                    E += 0.01 * (sr**12 - sr**6)
        return E

    def count_ring_fragments(atoms, box_size):
        """Count potential ring fragments (chains of C and N)."""
        n = len(atoms)
        bonds = []

        for i in range(n):
            for j in range(i+1, n):
                t1, p1 = atoms[i]
                t2, p2 = atoms[j]
                if t1 == 'H' or t2 == 'H':
                    continue  # Skip H for ring counting

                dr = p2 - p1
                dr = dr - box_size * np.round(dr / box_size)
                r = np.linalg.norm(dr)

                if r < 1.6:  # Bond cutoff
                    bonds.append((i, j))

        # Count C-N bonds (key for purine structure)
        cn_bonds = 0
        cc_bonds = 0
        nn_bonds = 0

        for i, j in bonds:
            t1, _ = atoms[i]
            t2, _ = atoms[j]
            pair = tuple(sorted([t1, t2]))
            if pair == ('C', 'N'):
                cn_bonds += 1
            elif pair == ('C', 'C'):
                cc_bonds += 1
            elif pair == ('N', 'N'):
                nn_bonds += 1

        # Adenine has 8 C-N/C-C bonds in rings
        # Rough estimate of adenine-like fragments
        ring_score = cn_bonds + cc_bonds

        return {'C-N': cn_bonds, 'C-C': cc_bonds, 'N-N': nn_bonds,
                'ring_score': ring_score}

    print(f"Cooling: {T_high} K → {T_low} K over {n_steps} steps")
    print("(Adenine requires slow cooling due to complex ring structure)\n")

    # Run MC
    for step in range(n_steps):
        T = T_high + (T_low - T_high) * step / n_steps

        idx = np.random.randint(len(atoms))
        t, pos = atoms[idx]
        step_size = 0.25 * (T / T_high) + 0.02

        atoms_new = atoms.copy()
        new_pos = (pos + np.random.uniform(-step_size, step_size, 3)) % box_size
        atoms_new[idx] = (t, new_pos)

        E_old = compute_energy(atoms, box_size)
        E_new = compute_energy(atoms_new, box_size)

        if E_new < E_old or np.random.random() < np.exp(-(E_new - E_old)/(KB * T)):
            atoms = atoms_new

        if step % (n_steps // 10) == 0:
            E = compute_energy(atoms, box_size)
            rings = count_ring_fragments(atoms, box_size)
            print(f"  Step {step:>6}: T={T:.0f} K, E={E:.1f} eV, "
                  f"C-N={rings['C-N']}, C-C={rings['C-C']}")

    # Final analysis
    E_final = compute_energy(atoms, box_size)
    rings_final = count_ring_fragments(atoms, box_size)

    # Each adenine has 7 C-N bonds and 1 C-C bond in rings
    # Plus NH₂ group adds 1 C-N
    estimated_adenine = min(rings_final['C-N'] // 8, n_units)

    print(f"\nFinal ring bond counts:")
    print(f"  C-N bonds: {rings_final['C-N']} (need ~8 per adenine)")
    print(f"  C-C bonds: {rings_final['C-C']} (need ~1 per adenine)")
    print(f"  Ring score: {rings_final['ring_score']}")

    print(f"\nEstimated adenine-like structures: ~{estimated_adenine} / {n_units}")

    # Note complexity
    print(f"""
Note: True adenine formation requires:
- Precise ring closure (5+6 member rings)
- Correct H placement on N atoms
- Resonance stabilization

In nature, this happens via:
1. HCN polymerization (prebiotic)
2. Enzymatic synthesis (biological)

Both require CATALYSIS (lowering activation barriers)
This is equivalent to increasing effective hierarchy H.
""")

    H_low = 5.0 / (KB * T_low)  # Average bond energy
    return {
        'ring_score': rings_final['ring_score'],
        'cn_bonds': rings_final['C-N'],
        'estimated_adenine': estimated_adenine,
        'target': n_units,
        'hierarchy_low': H_low
    }


# ============================================================
# COMPLEXITY HIERARCHY ANALYSIS
# ============================================================

def analyze_complexity_hierarchy():
    """
    Analyze how hierarchy requirements scale with molecular complexity.

    Key insight: More complex molecules need higher H to form.
    """
    print("\n" + "=" * 70)
    print("MOLECULAR COMPLEXITY HIERARCHY ANALYSIS")
    print("=" * 70)

    molecules = [
        {"name": "H₂", "atoms": 2, "bonds": 1, "D_avg": 4.5, "geometry": "linear"},
        {"name": "H₂O", "atoms": 3, "bonds": 2, "D_avg": 4.8, "geometry": "bent 104.5°"},
        {"name": "CO₂", "atoms": 3, "bonds": 2, "D_avg": 7.5, "geometry": "linear 180°"},
        {"name": "NH₃", "atoms": 4, "bonds": 3, "D_avg": 4.0, "geometry": "pyramidal"},
        {"name": "CH₄", "atoms": 5, "bonds": 4, "D_avg": 4.3, "geometry": "tetrahedral"},
        {"name": "Glycine", "atoms": 10, "bonds": 9, "D_avg": 4.0, "geometry": "complex"},
        {"name": "Adenine", "atoms": 15, "bonds": 15, "D_avg": 5.0, "geometry": "rings"},
        {"name": "ATP", "atoms": 47, "bonds": 50, "D_avg": 4.5, "geometry": "very complex"},
    ]

    print(f"\n{'Molecule':<12} {'Atoms':<8} {'Bonds':<8} {'D_avg (eV)':<12} "
          f"{'H at 300K':<12} {'T_form (K)':<12}")
    print("-" * 70)

    T_room = 300
    for mol in molecules:
        H_300K = mol['D_avg'] / (KB * T_room)

        # Estimate formation temperature
        # Need H > N_atoms for stability (rough rule)
        H_needed = mol['atoms']
        T_form = mol['D_avg'] / (KB * H_needed)

        print(f"{mol['name']:<12} {mol['atoms']:<8} {mol['bonds']:<8} "
              f"{mol['D_avg']:<12.1f} {H_300K:<12.0f} {T_form:<12.0f}")

    print(f"""
INTERPRETATION:

1. HIERARCHY REQUIREMENT: H > N_atoms (roughly)
   - More atoms → need higher H → need lower T or stronger bonds

2. FORMATION TEMPERATURE:
   - Simple molecules (H₂, H₂O): form at ~1000-2000 K
   - Complex molecules (amino acids): form at ~500-1000 K
   - Polymers (proteins, DNA): form at ~300 K with CATALYSIS

3. THE CATALYST EFFECT:
   - Enzymes lower activation barriers
   - Equivalent to increasing effective D_avg
   - This raises H without lowering T

4. ORIGIN OF LIFE LADDER:
   Supernova (10⁹ K) → Cooling → Atoms (10⁴ K) → Simple molecules (10³ K)
   → Complex molecules (10² K + catalysis) → Polymers → Life

Each step requires HIGHER HIERARCHY:
   H_atoms < H_molecules < H_polymers < H_cells < H_organisms
""")

    return molecules


# ============================================================
# MAIN
# ============================================================

def main():
    print("╔" + "═" * 68 + "╗")
    print("║" + " MOLECULAR COMPLEXITY SIMULATION ".center(68) + "║")
    print("║" + " From Simple Molecules to DNA Bases ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    results = {}

    # 1. CO₂
    print("\n" + "▶" * 35)
    results['co2'] = simulate_co2_formation(
        n_C=6, n_O=12, box_size=10.0,
        T_high=6000, T_low=300, n_steps=60000
    )

    # 2. Glycine
    print("\n" + "▶" * 35)
    results['glycine'] = simulate_glycine_formation(
        n_units=4, box_size=12.0,
        T_high=5000, T_low=300, n_steps=80000
    )

    # 3. Adenine
    print("\n" + "▶" * 35)
    results['adenine'] = simulate_adenine_formation(
        n_units=2, box_size=12.0,
        T_high=4000, T_low=300, n_steps=100000
    )

    # 4. Complexity analysis
    results['hierarchy_analysis'] = analyze_complexity_hierarchy()

    # Save results
    output_dir = Path(__file__).parent.parent.parent / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert for JSON
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    save_results = {
        'co2': {k: convert(v) for k, v in results['co2'].items() if k != 'molecules'},
        'glycine': {k: convert(v) for k, v in results['glycine'].items()},
        'adenine': {k: convert(v) for k, v in results['adenine'].items()},
    }

    output_file = output_dir / "molecular_complexity.json"
    with open(output_file, 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: GEOMETRY SELECTION ACROSS MOLECULAR COMPLEXITY")
    print("=" * 70)
    print(f"""
                Formation     Hierarchy
Molecule        Efficiency    (H at 300K)    Geometry Constraint
--------        ----------    -----------    -------------------
H₂O             100%          186            Bent (104.5°)
CO₂             {results['co2']['efficiency']*100:.0f}%           290            Linear (180°)
Glycine         ~{results['glycine']['estimated_glycine']}/{4}          155            Complex backbone
Adenine         ~{results['adenine']['estimated_adenine']}/{2}           193            Fused rings

KEY INSIGHT:
More complex geometry → harder to form randomly → needs higher H

This is why life requires:
1. Cooling (increases H)
2. Catalysis (increases effective D)
3. Time (hierarchical assembly)

The unified framework explains the LADDER OF COMPLEXITY:
Energy → Particles → Atoms → Molecules → Polymers → Cells → Life

Each step is GEOMETRY SELECTION at a different scale!
""")

    return results


if __name__ == "__main__":
    results = main()
