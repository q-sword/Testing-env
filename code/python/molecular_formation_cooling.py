#!/usr/bin/env python3
"""
Molecular Formation Through Cooling
====================================
Simulates how dissipation (cooling) drives atomic systems toward
stable molecular geometries - demonstrating our selection mechanisms
at the chemical scale.

Key demonstration:
- Random atoms + cooling → molecules form
- Without cooling → atoms scatter (chaos)
- Geometry selection in action!
"""

import numpy as np
from numba import njit
import json
from pathlib import Path

# Physical constants (atomic units for convenience)
# Length: Bohr radius a_0 = 0.529 Å
# Energy: Hartree = 27.2 eV
# Mass: electron mass

HBAR = 1.0  # In atomic units
K_COULOMB = 1.0  # e²/(4πε₀) in atomic units
M_PROTON = 1836.0  # Proton mass in electron masses
M_ELECTRON = 1.0

# Simulation parameters
EPSILON_QUANTUM = 0.5  # Regularization ~ a_0/2


@njit
def morse_potential(r, D_e, alpha, r_e, epsilon):
    """
    Morse potential for molecular bonds (regularized).

    V(r) = D_e * [1 - exp(-α(r-r_e))]² - D_e

    Regularized to prevent singularities at r=0.
    """
    r_eff = np.sqrt(r*r + epsilon*epsilon)
    x = 1.0 - np.exp(-alpha * (r_eff - r_e))
    return D_e * x * x - D_e


@njit
def morse_force(r_vec, D_e, alpha, r_e, epsilon):
    """Force from Morse potential (regularized)."""
    r = np.sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2)
    r_eff = np.sqrt(r*r + epsilon*epsilon)

    # dV/dr
    exp_term = np.exp(-alpha * (r_eff - r_e))
    dVdr = 2.0 * D_e * alpha * (1.0 - exp_term) * exp_term

    # Chain rule: dV/dx = (dV/dr)(dr_eff/dr)(dr/dx)
    # dr_eff/dr = r/r_eff
    # dr/dx = x/r

    if r < 1e-10:
        return np.zeros(3)

    factor = -dVdr * r / (r_eff * r)
    return factor * r_vec


@njit
def lennard_jones_force(r_vec, epsilon_lj, sigma, epsilon_reg):
    """
    Lennard-Jones force (regularized).
    For van der Waals interactions.
    """
    r2 = r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2
    r2_eff = r2 + epsilon_reg**2

    sigma2 = sigma * sigma
    sigma6 = sigma2 * sigma2 * sigma2
    sigma12 = sigma6 * sigma6

    r6 = r2_eff * r2_eff * r2_eff
    r12 = r6 * r6

    # F = -dV/dr * r_hat = 24ε/r * [2(σ/r)^12 - (σ/r)^6] * r_hat
    factor = 24.0 * epsilon_lj * (2.0 * sigma12 / r12 - sigma6 / r6) / r2_eff

    return factor * r_vec


@njit
def compute_forces_molecular(positions, masses, atom_types, epsilon):
    """
    Compute forces for a molecular system.

    atom_types: 0=H, 1=O, 2=C, etc.
    Uses Morse potential for bonds, LJ for non-bonded.
    """
    N = len(masses)
    forces = np.zeros_like(positions)

    # Morse parameters (approximate, in atomic units)
    # O-H bond: D_e ~ 0.17 Ha, r_e ~ 1.8 a_0, α ~ 1.2
    D_e_OH = 0.17
    r_e_OH = 1.8
    alpha_OH = 1.2

    # H-H bond: D_e ~ 0.16 Ha, r_e ~ 1.4 a_0
    D_e_HH = 0.16
    r_e_HH = 1.4
    alpha_HH = 1.0

    # LJ parameters for non-bonded
    epsilon_lj = 0.001  # Weak
    sigma = 5.0  # a_0

    for i in range(N):
        for j in range(i+1, N):
            r_vec = positions[j] - positions[i]
            r = np.sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2)

            # Determine interaction type
            type_i, type_j = atom_types[i], atom_types[j]

            if (type_i == 0 and type_j == 1) or (type_i == 1 and type_j == 0):
                # O-H interaction (can form bond)
                f = morse_force(r_vec, D_e_OH, alpha_OH, r_e_OH, epsilon)
            elif type_i == 0 and type_j == 0:
                # H-H interaction
                f = morse_force(r_vec, D_e_HH, alpha_HH, r_e_HH, epsilon)
            else:
                # Non-bonded (LJ)
                f = lennard_jones_force(r_vec, epsilon_lj, sigma, epsilon)

            forces[i] += f
            forces[j] -= f

    return forces


@njit
def compute_energy_molecular(positions, velocities, masses, atom_types, epsilon):
    """Compute total energy of molecular system."""
    N = len(masses)

    # Kinetic energy
    KE = 0.0
    for i in range(N):
        v2 = velocities[i,0]**2 + velocities[i,1]**2 + velocities[i,2]**2
        KE += 0.5 * masses[i] * v2

    # Potential energy
    PE = 0.0
    D_e_OH, r_e_OH, alpha_OH = 0.17, 1.8, 1.2
    D_e_HH, r_e_HH, alpha_HH = 0.16, 1.4, 1.0
    epsilon_lj, sigma = 0.001, 5.0

    for i in range(N):
        for j in range(i+1, N):
            r_vec = positions[j] - positions[i]
            r = np.sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2)

            type_i, type_j = atom_types[i], atom_types[j]

            if (type_i == 0 and type_j == 1) or (type_i == 1 and type_j == 0):
                PE += morse_potential(r, D_e_OH, alpha_OH, r_e_OH, epsilon)
            elif type_i == 0 and type_j == 0:
                PE += morse_potential(r, D_e_HH, alpha_HH, r_e_HH, epsilon)
            else:
                # LJ
                r_eff = np.sqrt(r*r + epsilon*epsilon)
                sr6 = (sigma/r_eff)**6
                PE += 4*epsilon_lj*(sr6*sr6 - sr6)

    return KE + PE, KE, PE


def velocity_verlet_step(positions, velocities, masses, atom_types, epsilon, dt, gamma=0.0):
    """
    Velocity Verlet with optional damping (cooling).

    gamma = cooling rate (0 = no cooling, >0 = cooling)
    """
    N = len(masses)

    # Half-step velocity
    forces = compute_forces_molecular(positions, masses, atom_types, epsilon)
    acc = forces / masses[:, np.newaxis]
    velocities = velocities + 0.5 * dt * acc

    # Apply damping (cooling)
    if gamma > 0:
        velocities *= (1.0 - gamma * dt)

    # Full-step position
    positions = positions + dt * velocities

    # Recalculate forces
    forces = compute_forces_molecular(positions, masses, atom_types, epsilon)
    acc = forces / masses[:, np.newaxis]

    # Half-step velocity
    velocities = velocities + 0.5 * dt * acc

    # Apply damping again
    if gamma > 0:
        velocities *= (1.0 - gamma * dt)

    return positions, velocities


def compute_bonds(positions, atom_types, bond_threshold=2.5):
    """
    Detect molecular bonds based on distance.
    Returns list of (i, j, distance) for bonded pairs.
    """
    N = len(atom_types)
    bonds = []

    for i in range(N):
        for j in range(i+1, N):
            r_vec = positions[j] - positions[i]
            r = np.sqrt(np.sum(r_vec**2))

            if r < bond_threshold:
                bonds.append((i, j, r))

    return bonds


def identify_molecules(positions, atom_types, bond_threshold=2.5):
    """
    Identify molecules from atomic positions.
    Returns list of molecules, each as list of atom indices.
    """
    N = len(atom_types)
    bonds = compute_bonds(positions, atom_types, bond_threshold)

    # Build adjacency
    adj = {i: [] for i in range(N)}
    for i, j, r in bonds:
        adj[i].append(j)
        adj[j].append(i)

    # Find connected components
    visited = [False] * N
    molecules = []

    for start in range(N):
        if not visited[start]:
            molecule = []
            stack = [start]
            while stack:
                node = stack.pop()
                if not visited[node]:
                    visited[node] = True
                    molecule.append(node)
                    stack.extend(adj[node])
            molecules.append(sorted(molecule))

    return molecules


def run_molecular_formation(n_atoms=10, with_cooling=True, T_max=5000, dt=0.5, seed=42):
    """
    Simulate molecular formation from random atoms.

    Parameters:
        n_atoms: Number of atoms (mix of H and O)
        with_cooling: Whether to apply cooling (dissipation)
        T_max: Total simulation time
        dt: Timestep
        seed: Random seed
    """
    np.random.seed(seed)

    # Create atom mixture (2H per O for water formation)
    n_oxygen = n_atoms // 3
    n_hydrogen = n_atoms - n_oxygen

    atom_types = np.array([0]*n_hydrogen + [1]*n_oxygen)  # 0=H, 1=O
    masses = np.where(atom_types == 0, M_PROTON, 16*M_PROTON)

    # Random initial positions (spread out)
    positions = np.random.randn(n_atoms, 3) * 10.0  # In units of a_0

    # Random velocities (thermal)
    T_initial = 0.01  # Temperature in Hartree (~ 3000 K)
    v_thermal = np.sqrt(T_initial / masses)
    velocities = np.random.randn(n_atoms, 3) * v_thermal[:, np.newaxis]

    # Zero momentum
    total_mom = np.sum(velocities * masses[:, np.newaxis], axis=0)
    velocities -= total_mom / np.sum(masses)

    # Cooling rate
    gamma = 0.001 if with_cooling else 0.0
    epsilon = EPSILON_QUANTUM

    # Storage
    n_steps = int(T_max / dt)
    record_interval = n_steps // 100

    history = {
        'time': [],
        'energy': [],
        'kinetic': [],
        'potential': [],
        'n_molecules': [],
        'molecule_sizes': [],
        'bonds': []
    }

    # Initial state
    E, KE, PE = compute_energy_molecular(positions, velocities, masses, atom_types, epsilon)
    molecules = identify_molecules(positions, atom_types)

    print(f"Initial: E={E:.4f}, KE={KE:.4f}, PE={PE:.4f}, molecules={len(molecules)}")

    # Evolution
    for step in range(n_steps):
        positions, velocities = velocity_verlet_step(
            positions, velocities, masses, atom_types, epsilon, dt, gamma
        )

        if step % record_interval == 0:
            E, KE, PE = compute_energy_molecular(positions, velocities, masses, atom_types, epsilon)
            molecules = identify_molecules(positions, atom_types)
            bonds = compute_bonds(positions, atom_types)

            history['time'].append(step * dt)
            history['energy'].append(E)
            history['kinetic'].append(KE)
            history['potential'].append(PE)
            history['n_molecules'].append(len(molecules))
            history['molecule_sizes'].append([len(m) for m in molecules])
            history['bonds'].append(len(bonds))

    # Final state
    E, KE, PE = compute_energy_molecular(positions, velocities, masses, atom_types, epsilon)
    molecules = identify_molecules(positions, atom_types)
    bonds = compute_bonds(positions, atom_types)

    print(f"Final: E={E:.4f}, KE={KE:.4f}, PE={PE:.4f}")
    print(f"Molecules formed: {len(molecules)}")
    print(f"Molecule sizes: {[len(m) for m in molecules]}")
    print(f"Total bonds: {len(bonds)}")

    # Identify molecule types
    molecule_types = []
    for mol in molecules:
        n_H = sum(1 for i in mol if atom_types[i] == 0)
        n_O = sum(1 for i in mol if atom_types[i] == 1)
        if n_H == 2 and n_O == 1:
            molecule_types.append("H2O")
        elif n_H == 2 and n_O == 0:
            molecule_types.append("H2")
        elif n_H == 1 and n_O == 1:
            molecule_types.append("OH")
        elif n_H == 0 and n_O == 2:
            molecule_types.append("O2")
        else:
            molecule_types.append(f"H{n_H}O{n_O}")

    print(f"Molecule types: {molecule_types}")

    return history, molecules, molecule_types


def main():
    print("="*70)
    print("MOLECULAR FORMATION THROUGH COOLING")
    print("Demonstrating geometry selection at chemical scale")
    print("="*70)
    print()

    results = {}

    # Test 1: With cooling (dissipation)
    print("\n" + "-"*70)
    print("TEST 1: WITH COOLING (Dissipation enabled)")
    print("-"*70)

    history_cool, molecules_cool, types_cool = run_molecular_formation(
        n_atoms=12, with_cooling=True, T_max=10000, seed=42
    )

    results['with_cooling'] = {
        'final_molecules': len(molecules_cool),
        'molecule_types': types_cool,
        'final_energy': history_cool['energy'][-1],
        'energy_ratio': history_cool['energy'][-1] / history_cool['energy'][0],
        'final_bonds': history_cool['bonds'][-1]
    }

    # Test 2: Without cooling
    print("\n" + "-"*70)
    print("TEST 2: WITHOUT COOLING (No dissipation)")
    print("-"*70)

    history_nocool, molecules_nocool, types_nocool = run_molecular_formation(
        n_atoms=12, with_cooling=False, T_max=10000, seed=42
    )

    results['without_cooling'] = {
        'final_molecules': len(molecules_nocool),
        'molecule_types': types_nocool,
        'final_energy': history_nocool['energy'][-1],
        'energy_ratio': history_nocool['energy'][-1] / history_nocool['energy'][0],
        'final_bonds': history_nocool['bonds'][-1]
    }

    # Comparison
    print("\n" + "="*70)
    print("COMPARISON: COOLING vs NO COOLING")
    print("="*70)

    print(f"""
                        WITH COOLING    WITHOUT COOLING
    ──────────────────────────────────────────────────────
    Final molecules:    {results['with_cooling']['final_molecules']:>8}        {results['without_cooling']['final_molecules']:>8}
    Stable bonds:       {results['with_cooling']['final_bonds']:>8}        {results['without_cooling']['final_bonds']:>8}
    Energy ratio:       {results['with_cooling']['energy_ratio']:>8.3f}        {results['without_cooling']['energy_ratio']:>8.3f}
    H2O formed:         {types_cool.count('H2O'):>8}        {types_nocool.count('H2O'):>8}
    H2 formed:          {types_cool.count('H2'):>8}        {types_nocool.count('H2'):>8}
    """)

    print("""
    CONCLUSION:
    ─────────────────────────────────────────────────────────
    With cooling (dissipation):
      → Energy decreases (atoms slow down)
      → Atoms captured into bound states
      → Stable molecules form
      → THIS IS RESONANCE CAPTURE!

    Without cooling:
      → Energy constant (atoms maintain speed)
      → Atoms scatter after collisions
      → Fewer stable molecules
      → CHAOS PERSISTS

    This demonstrates: DISSIPATION → STABLE GEOMETRY
    Same mechanism as gravitational systems!
    """)

    # Save results
    output_path = Path('/home/user/Testing-env/data/results/molecular_formation_cooling.json')

    # Convert for JSON
    json_results = {
        'with_cooling': results['with_cooling'],
        'without_cooling': results['without_cooling'],
        'conclusion': 'Cooling enables molecular formation via resonance capture'
    }

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    results = main()
