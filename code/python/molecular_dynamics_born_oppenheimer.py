#!/usr/bin/env python3
"""
================================================================================
BORN-OPPENHEIMER MOLECULAR DYNAMICS WITH QUANTUM REGULARIZATION
================================================================================

SIMPLIFIED MODEL: Nuclei are FIXED, only electrons move.

This is the proper way to test electronic structure stability:
  • Nuclei held at experimental bond length
  • Electrons evolve in fixed nuclear potential
  • Test if electronic wavefunction is stable (λ < 0)

Uses validated principles from three-body code:
  1. ε = ℏ/(m_e·v) ~ 0.1-1.0 a₀ (quantum regularization)
  2. Yoshida 6th order symplectic integrator
  3. Benettin method for Lyapunov exponents

Atomic units: ℏ = m_e = e = k_e = 1

Date: December 30, 2025
================================================================================
"""

import numpy as np
from numba import njit
import time as pytime
import json

# ============================================================================
# ATOMIC UNITS
# ============================================================================

HBAR = 1.0
M_E = 1.0
E_CHARGE = 1.0
K_E = 1.0

# Physical conversions
A0_SI = 5.29e-11        # m
EH_SI = 4.36e-18        # J
EH_EV = 27.211          # eV
TIME_AU = 2.42e-17      # s

# ============================================================================
# YOSHIDA 6TH ORDER COEFFICIENTS
# ============================================================================

w1 = 0.78451361047755726382
w2 = 0.23557321335935813368
w3 = -1.17767998417887100695
w0 = 1.0 - 2.0*(w1 + w2 + w3)

YOSHIDA6_C = np.array([w3, w2, w1, w0, w1, w2, w3, 0.0])
YOSHIDA6_D = np.array([
    w3/2, (w3+w2)/2, (w2+w1)/2, (w1+w0)/2,
    (w0+w1)/2, (w1+w2)/2, (w2+w3)/2, w3/2
])

# ============================================================================
# FORCES (BORN-OPPENHEIMER: NUCLEI FIXED)
# ============================================================================

@njit
def compute_forces_electrons(r_electrons, r_nuclei, Z_nuclei, epsilon):
    """
    Compute forces on electrons in fixed nuclear potential.

    Electron-electron repulsion: +1/(|r_i - r_j|² + ε²)^(1/2)
    Electron-nucleus attraction: -Z/(|r_i - R_A|² + ε²)^(1/2)

    Args:
        r_electrons: (N_e, 3) electron positions
        r_nuclei: (N_nuc, 3) nuclear positions (FIXED)
        Z_nuclei: (N_nuc,) nuclear charges
        epsilon: Regularization scale

    Returns:
        forces: (N_e, 3) forces on electrons
    """
    N_e = len(r_electrons)
    N_nuc = len(r_nuclei)
    forces = np.zeros((N_e, 3))

    # Electron-electron repulsion
    for i in range(N_e):
        for j in range(i+1, N_e):
            r_vec = r_electrons[j] - r_electrons[i]
            r = np.sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2)
            r_reg = np.sqrt(r**2 + epsilon**2)

            F_mag = K_E / (r_reg**3)  # Repulsion (same charge)
            F_vec = F_mag * r_vec

            forces[i] += F_vec
            forces[j] -= F_vec

    # Electron-nucleus attraction
    for i in range(N_e):
        for A in range(N_nuc):
            r_vec = r_electrons[i] - r_nuclei[A]
            r = np.sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2)
            r_reg = np.sqrt(r**2 + epsilon**2)

            F_mag = -K_E * Z_nuclei[A] / (r_reg**3)  # Attraction (opposite charge)
            F_vec = F_mag * r_vec

            forces[i] += F_vec

    return forces


@njit
def compute_energy_electrons(r_electrons, v_electrons, r_nuclei, Z_nuclei, epsilon):
    """
    Total electronic energy (nuclei fixed).

    E = KE + V_ee + V_eN
    """
    N_e = len(r_electrons)
    N_nuc = len(r_nuclei)

    # Kinetic energy
    KE = 0.0
    for i in range(N_e):
        v_sq = v_electrons[i, 0]**2 + v_electrons[i, 1]**2 + v_electrons[i, 2]**2
        KE += 0.5 * M_E * v_sq

    # Electron-electron repulsion
    V_ee = 0.0
    for i in range(N_e):
        for j in range(i+1, N_e):
            r_vec = r_electrons[j] - r_electrons[i]
            r = np.sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2)
            r_reg = np.sqrt(r**2 + epsilon**2)
            V_ee += K_E / r_reg

    # Electron-nucleus attraction
    V_eN = 0.0
    for i in range(N_e):
        for A in range(N_nuc):
            r_vec = r_electrons[i] - r_nuclei[A]
            r = np.sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2)
            r_reg = np.sqrt(r**2 + epsilon**2)
            V_eN += -K_E * Z_nuclei[A] / r_reg

    return KE + V_ee + V_eN


# ============================================================================
# YOSHIDA 6TH ORDER INTEGRATOR
# ============================================================================

@njit
def yoshida6_step_BO(r_electrons, v_electrons, r_nuclei, Z_nuclei, epsilon, dt):
    """
    Yoshida 6th order step for Born-Oppenheimer dynamics.

    Only electrons move, nuclei are fixed.
    """
    for i in range(len(YOSHIDA6_D)):
        # Velocity kick
        forces = compute_forces_electrons(r_electrons, r_nuclei, Z_nuclei, epsilon)
        for j in range(len(r_electrons)):
            v_electrons[j] += YOSHIDA6_D[i] * dt * (forces[j] / M_E)

        # Position drift
        if i < len(YOSHIDA6_C) - 1 or YOSHIDA6_C[i] != 0.0:
            for j in range(len(r_electrons)):
                r_electrons[j] += YOSHIDA6_C[i] * dt * v_electrons[j]

    return r_electrons, v_electrons


# ============================================================================
# LYAPUNOV EXPONENT
# ============================================================================

@njit
def compute_lyapunov_BO(r_ref, v_ref, r_pert, v_pert, r_nuclei, Z_nuclei,
                        epsilon, dt, num_steps, tau_renorm):
    """
    Compute Lyapunov exponent for electronic motion.

    λ < 0: Electronic structure is STABLE
    λ > 0: Electronic wavefunction is CHAOTIC (unphysical!)
    """
    perturbation_size = 1e-8

    # Initial separation
    sep_initial = 0.0
    for i in range(len(r_ref)):
        diff = r_pert[i] - r_ref[i]
        sep_initial += diff[0]**2 + diff[1]**2 + diff[2]**2
    sep_initial = np.sqrt(sep_initial)

    lyapunov_sum = 0.0
    renorm_count = 0

    for step in range(num_steps):
        # Integrate both trajectories
        r_ref, v_ref = yoshida6_step_BO(r_ref, v_ref, r_nuclei, Z_nuclei, epsilon, dt)
        r_pert, v_pert = yoshida6_step_BO(r_pert, v_pert, r_nuclei, Z_nuclei, epsilon, dt)

        # Renormalization
        if (step + 1) % tau_renorm == 0:
            sep_current = 0.0
            for i in range(len(r_ref)):
                diff = r_pert[i] - r_ref[i]
                sep_current += diff[0]**2 + diff[1]**2 + diff[2]**2
            sep_current = np.sqrt(sep_current)

            if sep_current > 0:
                growth = sep_current / sep_initial
                lyapunov_sum += np.log(growth)
                renorm_count += 1

                # Rescale
                scale_factor = perturbation_size / sep_current
                for i in range(len(r_ref)):
                    delta = r_pert[i] - r_ref[i]
                    r_pert[i] = r_ref[i] + scale_factor * delta

    if renorm_count > 0:
        lambda_val = lyapunov_sum / (renorm_count * tau_renorm * dt)
    else:
        lambda_val = 0.0

    return lambda_val


# ============================================================================
# INITIAL CONDITIONS
# ============================================================================

def create_molecule_BO(molecule_name, epsilon=0.1):
    """
    Create Born-Oppenheimer initial conditions.

    Nuclei at experimental bond length (fixed).
    Electrons initialized in orbital configuration.
    """
    molecules = {
        'H2': {
            'nuclei_pos': [np.array([-0.7, 0, 0]), np.array([0.7, 0, 0])],  # R = 1.4 a₀
            'nuclei_Z': [1, 1],
            'N_electrons': 2,
            'description': 'H₂ molecule (2 electrons, R = 1.4 a₀)'
        },
        'H2+': {
            'nuclei_pos': [np.array([-1.0, 0, 0]), np.array([1.0, 0, 0])],  # R = 2.0 a₀
            'nuclei_Z': [1, 1],
            'N_electrons': 1,
            'description': 'H₂⁺ ion (1 electron, R = 2.0 a₀)'
        },
        'HeH+': {
            'nuclei_pos': [np.array([-0.4, 0, 0]), np.array([0.4, 0, 0])],  # R = 0.8 a₀
            'nuclei_Z': [2, 1],  # He + H
            'N_electrons': 2,
            'description': 'HeH⁺ molecular ion'
        },
    }

    if molecule_name not in molecules:
        raise ValueError(f"Unknown molecule: {molecule_name}")

    mol = molecules[molecule_name]
    r_nuclei = np.array(mol['nuclei_pos'])
    Z_nuclei = np.array(mol['nuclei_Z'])
    N_e = mol['N_electrons']

    # Initialize electrons in orbital configuration
    r_electrons = []
    v_electrons = []

    # Place electrons between nuclei with orbital velocities
    for i in range(N_e):
        # Position: circling the bond center
        angle = 2 * np.pi * i / N_e
        radius = 0.5  # a₀ from bond center
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 0.0
        r_electrons.append([x, y, z])

        # Velocity: orbital motion
        v_mag = 1.0  # atomic units
        vx = -v_mag * np.sin(angle)
        vy = +v_mag * np.cos(angle)
        vz = 0.0
        v_electrons.append([vx, vy, vz])

    r_electrons = np.array(r_electrons)
    v_electrons = np.array(v_electrons)

    return r_electrons, v_electrons, r_nuclei, Z_nuclei, epsilon, mol['description']


# ============================================================================
# RUN SIMULATION
# ============================================================================

def run_born_oppenheimer(molecule_name, total_time=100.0, dt=0.001, epsilon=0.1, verbose=True):
    """
    Run Born-Oppenheimer molecular dynamics.

    Args:
        molecule_name: 'H2', 'H2+', or 'HeH+'
        total_time: Integration time (atomic units)
        dt: Timestep
        epsilon: Regularization scale
        verbose: Print output
    """
    r_electrons, v_electrons, r_nuclei, Z_nuclei, epsilon, description = \
        create_molecule_BO(molecule_name, epsilon)

    if verbose:
        print("="*80)
        print(f"BORN-OPPENHEIMER DYNAMICS: {molecule_name}")
        print("="*80)
        print(f"Description: {description}")
        print(f"Nuclei (FIXED):")
        for i, (pos, Z) in enumerate(zip(r_nuclei, Z_nuclei)):
            print(f"  Nucleus {i+1}: Z={Z}, position={pos}")
        print(f"Electrons: {len(r_electrons)}")
        print(f"Epsilon: {epsilon:.3f} a₀")
        print()

    # Initial energy
    E0 = compute_energy_electrons(r_electrons, v_electrons, r_nuclei, Z_nuclei, epsilon)

    if verbose:
        print(f"Initial electronic energy: E = {E0:.6f} E_H = {E0*EH_EV:.3f} eV")
        print()

    # Lyapunov calculation
    num_steps = int(total_time / dt)
    tau_renorm = max(100, num_steps // 100)

    if verbose:
        print(f"Integration:")
        print(f"  Time: {total_time} a.u. = {total_time*TIME_AU*1e15:.1f} fs")
        print(f"  Timestep: {dt} a.u. = {dt*TIME_AU*1e18:.2f} as")
        print(f"  Steps: {num_steps:,}")
        print()
        print("Computing Lyapunov exponent...", flush=True)

    # Prepare trajectories
    r_ref = r_electrons.copy()
    v_ref = v_electrons.copy()
    r_pert = r_electrons.copy() + 1e-8 * np.random.randn(*r_electrons.shape)
    v_pert = v_electrons.copy()

    # Compute
    start = pytime.time()
    lambda_val = compute_lyapunov_BO(
        r_ref, v_ref, r_pert, v_pert, r_nuclei, Z_nuclei,
        epsilon, dt, num_steps, tau_renorm
    )
    runtime = pytime.time() - start

    # Energy conservation
    E_final = compute_energy_electrons(r_ref, v_ref, r_nuclei, Z_nuclei, epsilon)
    dE_rel = abs(E_final - E0) / abs(E0) if E0 != 0 else 0.0

    stable = lambda_val < 0

    if verbose:
        print()
        print("="*80)
        print("RESULTS")
        print("="*80)
        print(f"Lyapunov exponent: λ = {lambda_val:+.6f}")
        print(f"Status: {'✓ STABLE' if stable else '✗ CHAOTIC'}")
        print()
        print(f"Energy conservation:")
        print(f"  |ΔE/E₀| = {dE_rel:.3e}")
        print()
        print(f"Runtime: {runtime:.1f} s")
        print("="*80)

    return {
        'molecule': molecule_name,
        'lambda': lambda_val,
        'stable': stable,
        'epsilon': epsilon,
        'dE_rel': dE_rel,
        'runtime': runtime
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys

    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*10 + "BORN-OPPENHEIMER MOLECULAR DYNAMICS VALIDATION" + " "*21 + "║")
    print("║" + " "*10 + "Quantum Regularization + Yoshida 6th Order" + " "*24 + "║")
    print("╚" + "="*78 + "╝")
    print()

    molecules = ['H2+', 'H2', 'HeH+']
    results = []

    for mol in molecules:
        result = run_born_oppenheimer(mol, total_time=100.0, epsilon=0.1)
        results.append(result)
        print()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print(f"{'Molecule':<10} {'λ':<12} {'Status':<10} {'|ΔE/E₀|':<12}")
    print("-"*50)

    stable_count = 0
    for res in results:
        status = "✓ STABLE" if res['stable'] else "✗ CHAOTIC"
        if res['stable']:
            stable_count += 1
        print(f"{res['molecule']:<10} {res['lambda']:<+12.6f} {status:<10} {res['dE_rel']:<12.3e}")

    print()
    print(f"Stability: {stable_count}/{len(results)} = {100*stable_count/len(results):.0f}%")

    if stable_count == len(results):
        print()
        print("🎉 ALL MOLECULES STABLE (λ < 0)")
        print("   Quantum regularization stabilizes electronic structure! ✓")

    print("="*80)

    # Save
    with open('data/results/born_oppenheimer_validation.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to: data/results/born_oppenheimer_validation.json")
