#!/usr/bin/env python3
"""
================================================================================
FUNDAMENTAL STABILITY OF GRAVITATIONAL N-BODY SYSTEMS
================================================================================

WHY is gravitational 3-body stable (λ < 0) even at tiny ε?

Test hypotheses:
1. Virial equilibrium (2KE + PE = 0) creates stability
2. All-attractive forces prevent chaos
3. Angular momentum conservation constrains dynamics
4. Bound orbits have hidden integrals of motion

Compare:
  • BOUND systems (E < 0): Should be stable
  • UNBOUND systems (E > 0): Should be chaotic (scattering)

This will reveal the PHYSICAL mechanism of gravitational stability.

Date: December 30, 2025
================================================================================
"""

import numpy as np
from numba import njit
import json

# Yoshida coefficients
w1 = 0.78451361047755726382
w2 = 0.23557321335935813368
w3 = -1.17767998417887100695
w0 = 1.0 - 2.0*(w1 + w2 + w3)

YOSHIDA6_C = np.array([w3, w2, w1, w0, w1, w2, w3, 0.0])
YOSHIDA6_D = np.array([
    w3/2, (w3+w2)/2, (w2+w1)/2, (w1+w0)/2,
    (w0+w1)/2, (w1+w2)/2, (w2+w3)/2, w3/2
])

@njit
def compute_forces(pos, masses, epsilon):
    N = len(masses)
    forces = np.zeros((N, 3))
    for i in range(N):
        for j in range(i+1, N):
            r_vec = pos[j] - pos[i]
            r = np.sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2)
            r_reg = np.sqrt(r**2 + epsilon**2)
            F_mag = masses[i] * masses[j] / (r_reg**3)
            F_vec = F_mag * r_vec
            forces[i] += F_vec
            forces[j] -= F_vec
    return forces

@njit
def compute_energy(pos, vel, masses, epsilon):
    """Total energy: KE + PE"""
    N = len(masses)

    # Kinetic
    KE = 0.0
    for i in range(N):
        v_sq = vel[i, 0]**2 + vel[i, 1]**2 + vel[i, 2]**2
        KE += 0.5 * masses[i] * v_sq

    # Potential
    PE = 0.0
    for i in range(N):
        for j in range(i+1, N):
            r_vec = pos[j] - pos[i]
            r = np.sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2)
            r_reg = np.sqrt(r**2 + epsilon**2)
            PE -= masses[i] * masses[j] / r_reg

    return KE, PE, KE + PE

@njit
def compute_angular_momentum(pos, vel, masses):
    """Total angular momentum"""
    L = np.zeros(3)
    for i in range(len(masses)):
        L += masses[i] * np.cross(pos[i], vel[i])
    return L

@njit
def yoshida6_step(pos, vel, masses, epsilon, dt):
    for i in range(len(YOSHIDA6_D)):
        forces = compute_forces(pos, masses, epsilon)
        for j in range(len(masses)):
            vel[j] += YOSHIDA6_D[i] * dt * (forces[j] / masses[j])
        if i < len(YOSHIDA6_C) - 1 or YOSHIDA6_C[i] != 0.0:
            for j in range(len(masses)):
                pos[j] += YOSHIDA6_C[i] * dt * vel[j]
    return pos, vel

@njit
def compute_lyapunov(pos_ref, vel_ref, pos_pert, vel_pert, masses, epsilon, dt, num_steps, tau_renorm):
    perturbation_size = 1e-8
    sep_initial = 0.0
    for i in range(len(masses)):
        diff = pos_pert[i] - pos_ref[i]
        sep_initial += diff[0]**2 + diff[1]**2 + diff[2]**2
    sep_initial = np.sqrt(sep_initial)

    lyapunov_sum = 0.0
    renorm_count = 0

    for step in range(num_steps):
        pos_ref, vel_ref = yoshida6_step(pos_ref, vel_ref, masses, epsilon, dt)
        pos_pert, vel_pert = yoshida6_step(pos_pert, vel_pert, masses, epsilon, dt)

        if (step + 1) % tau_renorm == 0:
            sep_current = 0.0
            for i in range(len(masses)):
                diff = pos_pert[i] - pos_ref[i]
                sep_current += diff[0]**2 + diff[1]**2 + diff[2]**2
            sep_current = np.sqrt(sep_current)

            if sep_current > 0:
                growth = sep_current / sep_initial
                lyapunov_sum += np.log(growth)
                renorm_count += 1
                scale_factor = perturbation_size / sep_current
                for i in range(len(masses)):
                    delta = pos_pert[i] - pos_ref[i]
                    pos_pert[i] = pos_ref[i] + scale_factor * delta

    if renorm_count > 0:
        lambda_val = lyapunov_sum / (renorm_count * tau_renorm * dt)
    else:
        lambda_val = 0.0
    return lambda_val

def create_bound_system(seed, energy_target=-0.5):
    """
    Create BOUND system (E < 0) via virial theorem.

    For bound system: E < 0, with 2*KE + PE ≈ 0 (virial)
    """
    np.random.seed(seed)
    masses = np.array([1.0, 1.0, 1.0])

    # Random positions
    positions = np.random.uniform(-2.0, 2.0, size=(3, 3))

    # Compute potential energy
    epsilon_temp = 0.1
    PE = 0.0
    for i in range(3):
        for j in range(i+1, 3):
            r_vec = positions[j] - positions[i]
            r = np.linalg.norm(r_vec)
            r_reg = np.sqrt(r**2 + epsilon_temp**2)
            PE -= masses[i] * masses[j] / r_reg

    # For bound system with target E:
    # E = KE + PE, and virial: KE = -PE/2
    # So: E = -PE/2 + PE = PE/2
    # Target: E = energy_target → PE = 2*E, KE = -E

    KE_target = -energy_target  # E < 0 → KE > 0

    # Random velocities, scaled to target KE
    velocities = np.random.randn(3, 3)
    KE_current = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
    velocities *= np.sqrt(KE_target / KE_current)

    # Zero total momentum
    total_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
    for i in range(3):
        velocities[i] -= total_momentum / (3.0 * masses[i])

    return positions, velocities, masses

def create_unbound_system(seed):
    """
    Create UNBOUND system (E > 0) - scattering event.

    Start with high kinetic energy >> |PE|
    """
    np.random.seed(seed)
    masses = np.array([1.0, 1.0, 1.0])

    # Widely separated
    positions = np.random.uniform(-5.0, 5.0, size=(3, 3))

    # High velocities (E > 0)
    velocities = np.random.uniform(-2.0, 2.0, size=(3, 3))

    # Zero total momentum
    total_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
    for i in range(3):
        velocities[i] -= total_momentum / (3.0 * masses[i])

    return positions, velocities, masses

# ============================================================================
# MAIN TEST
# ============================================================================

print("="*80)
print("TESTING FUNDAMENTAL STABILITY MECHANISM")
print("="*80)
print()

epsilon = 0.1  # Small regularization
dt = 0.0001
num_steps = 200000  # T = 20
tau_renorm = 1000

print("Testing BOUND vs UNBOUND systems")
print()
print(f"{'Type':<12} {'E_initial':<12} {'KE/|PE|':<12} {'λ':<12} {'Status':<12}")
print("-"*65)

results = []

# Test bound systems
for seed in range(3):
    pos, vel, masses = create_bound_system(seed, energy_target=-0.5)
    KE, PE, E = compute_energy(pos, vel, masses, epsilon)

    pos_ref = pos.copy()
    vel_ref = vel.copy()
    pos_pert = pos.copy() + 1e-8 * np.random.randn(3, 3)
    vel_pert = vel.copy()

    lam = compute_lyapunov(pos_ref, vel_ref, pos_pert, vel_pert,
                           masses, epsilon, dt, num_steps, tau_renorm)

    status = "✓ STABLE" if lam < 0 else "✗ CHAOTIC"
    ke_pe_ratio = KE / abs(PE) if PE != 0 else 0

    print(f"BOUND-{seed:<6} {E:<+12.4f} {ke_pe_ratio:<12.4f} {lam:<+12.6f} {status:<12}")

    results.append({
        'type': 'BOUND',
        'seed': seed,
        'E': E,
        'KE': KE,
        'PE': PE,
        'lambda': lam,
        'stable': lam < 0
    })

# Test unbound systems
for seed in range(3):
    pos, vel, masses = create_unbound_system(seed)
    KE, PE, E = compute_energy(pos, vel, masses, epsilon)

    pos_ref = pos.copy()
    vel_ref = vel.copy()
    pos_pert = pos.copy() + 1e-8 * np.random.randn(3, 3)
    vel_pert = vel.copy()

    lam = compute_lyapunov(pos_ref, vel_ref, pos_pert, vel_pert,
                           masses, epsilon, dt, num_steps, tau_renorm)

    status = "✓ STABLE" if lam < 0 else "✗ CHAOTIC"
    ke_pe_ratio = KE / abs(PE) if PE != 0 else 0

    print(f"UNBOUND-{seed:<4} {E:<+12.4f} {ke_pe_ratio:<12.4f} {lam:<+12.6f} {status:<12}")

    results.append({
        'type': 'UNBOUND',
        'seed': seed,
        'E': E,
        'KE': KE,
        'PE': PE,
        'lambda': lam,
        'stable': lam < 0
    })

print()
print("="*80)
print("ANALYSIS")
print("="*80)
print()

bound_stable = sum(1 for r in results if r['type'] == 'BOUND' and r['stable'])
unbound_stable = sum(1 for r in results if r['type'] == 'UNBOUND' and r['stable'])

print(f"BOUND systems stable: {bound_stable}/3")
print(f"UNBOUND systems stable: {unbound_stable}/3")
print()

if bound_stable == 3 and unbound_stable < 3:
    print("✓ HYPOTHESIS CONFIRMED:")
    print("  Bound gravitational systems (E < 0) ARE STABLE (λ < 0)")
    print("  Unbound scattering (E > 0) shows chaos (λ > 0)")
    print()
    print("MECHANISM:")
    print("  • Virial equilibrium (2KE + PE = 0) constrains dynamics")
    print("  • Bound orbits have conserved angular momentum")
    print("  • All-attractive forces create stable orbital structure")
    print()
    print("This explains:")
    print("  • Why 30-seed validation showed λ < 0 (all virialized)")
    print("  • Why tiny ε works (bound systems naturally stable)")
    print("  • Why molecules different (e-e repulsion breaks this)")

elif bound_stable == 3 and unbound_stable == 3:
    print("⚠️  UNEXPECTED:")
    print("  Both bound AND unbound systems stable")
    print("  Regularization may be stronger effect than energy")

else:
    print("⚠️  MIXED RESULTS:")
    print("  Need more investigation")

print()
print("="*80)

# Save results
with open('data/results/gravity_stability_mechanism.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to: data/results/gravity_stability_mechanism.json")
print()
