#!/usr/bin/env python3
"""
RESOLVING THE λ DISCREPANCY
December 2025

PROBLEM:
  Original data (30_seed_results.json): λ = -2.2 (STABLE)
  New scan (lambda_vs_N_scan.py): λ = +0.11 (CHAOTIC)

This is a ~100× difference with OPPOSITE SIGNS!

Possible causes:
1. Different perturbation size (1e-10 vs 1e-8)
2. Different renormalization method (single vs QR)
3. Different time scales (T_lyap=10 vs T_lyap=2)
4. Bug in one of the implementations
5. Different epsilon calculation

GOAL: Find the bug and FIX IT
"""

import numpy as np
from numba import njit
import json

G = 1.0
HBAR = 1.0

# Yoshida coefficients
w1 = 0.78451361047755726382
w2 = 0.23557321335935813368
w3 = -1.17767998417887100695
w0 = 1.0 - 2.0*(w1 + w2 + w3)

C = np.array([w3, w2, w1, w0, w1, w2, w3, 0.0])
D = np.array([w3/2, (w3+w2)/2, (w2+w1)/2, (w1+w0)/2,
              (w0+w1)/2, (w1+w2)/2, (w2+w3)/2, w3/2])

@njit
def compute_forces(pos, masses, epsilon):
    N = len(masses)
    acc = np.zeros((N, 3))
    for i in range(N):
        for j in range(N):
            if i != j:
                r_vec = pos[j] - pos[i]
                r2 = r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2
                r_reg2 = r2 + epsilon**2
                r_reg3 = r_reg2 * np.sqrt(r_reg2)
                force_mag = G * masses[j] / r_reg3
                acc[i] += force_mag * r_vec
    return acc

@njit
def yoshida6_step(pos, vel, masses, epsilon, dt):
    for i in range(len(D)):
        acc = compute_forces(pos, masses, epsilon)
        vel = vel + D[i] * dt * acc
        if i < len(C) - 1 or C[i] != 0.0:
            pos = pos + C[i] * dt * vel
    return pos, vel

@njit
def compute_energy(pos, vel, masses, epsilon):
    N = len(masses)
    KE = 0.5 * np.sum(masses.reshape(-1, 1) * (vel * vel))
    PE = 0.0
    for i in range(N):
        for j in range(i+1, N):
            r_vec = pos[j] - pos[i]
            r2 = np.sum(r_vec * r_vec)
            r_reg = np.sqrt(r2 + epsilon**2)
            PE -= G * masses[i] * masses[j] / r_reg
    return KE + PE

def method_original(seed, T_total=100, T_lyap=10, dt=0.0001):
    """
    Original method from three_body_validated.py
    Uses single perturbation + norm tracking
    """
    np.random.seed(seed)
    N = 3
    masses = np.ones(N)
    pos = np.random.randn(N, 3) * 0.5
    vel = np.random.randn(N, 3) * 0.3

    # Calculate epsilon
    v_rms = np.sqrt(np.mean(vel**2))
    epsilon = HBAR / (masses[0] * v_rms)

    # Single perturbation
    delta_pos = 1e-10 * np.random.randn(N, 3)
    delta_vel = np.zeros((N, 3))

    log_stretch = 0.0
    n_renorm = 0

    for t in np.arange(0, T_total, T_lyap):
        # Reference system
        pos_ref = pos.copy()
        vel_ref = vel.copy()

        # Perturbed system
        pos_pert = pos + delta_pos
        vel_pert = vel + delta_vel

        # Evolve both
        steps = int(T_lyap / dt)
        for step in range(steps):
            pos_ref, vel_ref = yoshida6_step(pos_ref, vel_ref, masses, epsilon, dt)
            pos_pert, vel_pert = yoshida6_step(pos_pert, vel_pert, masses, epsilon, dt)

        # Measure separation
        delta_pos = pos_pert - pos_ref
        delta_vel = vel_pert - vel_ref

        # Norm of full perturbation vector
        delta_full = np.concatenate([delta_pos.flatten(), delta_vel.flatten()])
        norm = np.linalg.norm(delta_full)

        # Track stretching
        log_stretch += np.log(norm / (1e-10 * np.sqrt(N * 3)))
        n_renorm += 1

        # Renormalize
        delta_pos *= 1e-10 / norm
        delta_vel *= 1e-10 / norm

        # Update reference
        pos = pos_ref
        vel = vel_ref

    lambda_exp = log_stretch / T_total

    E0 = compute_energy(pos, vel, masses, epsilon)
    pos_test = pos.copy()
    vel_test = vel.copy()
    for step in range(int(T_total / dt)):
        pos_test, vel_test = yoshida6_step(pos_test, vel_test, masses, epsilon, dt)
    E_final = compute_energy(pos_test, vel_test, masses, epsilon)
    energy_error = abs((E_final - E0) / E0)

    return lambda_exp, epsilon, energy_error

def method_new(seed, T_total=100, T_lyap=10, dt=0.0001, n_vectors=6):
    """
    New method from lambda_vs_N_scan.py
    Uses QR decomposition of multiple tangent vectors
    """
    np.random.seed(seed)
    N = 3
    masses = np.ones(N)
    pos = np.random.randn(N, 3) * 0.5
    vel = np.random.randn(N, 3) * 0.3

    # Calculate epsilon
    v_rms = np.sqrt(np.mean(vel**2))
    epsilon = HBAR / (masses[0] * v_rms)

    # Multiple tangent vectors
    tangent_pos = np.random.randn(n_vectors, N, 3) * 1e-8
    tangent_vel = np.random.randn(n_vectors, N, 3) * 1e-8

    # QR orthonormalization
    tangent_flat = np.concatenate([tangent_pos.reshape(n_vectors, -1),
                                    tangent_vel.reshape(n_vectors, -1)], axis=1)
    Q, R = np.linalg.qr(tangent_flat.T)
    tangent_flat = Q.T

    tangent_pos = tangent_flat[:, :N*3].reshape(n_vectors, N, 3)
    tangent_vel = tangent_flat[:, N*3:].reshape(n_vectors, N, 3)

    lyapunov_sums = np.zeros(n_vectors)
    n_intervals = int(T_total / T_lyap)
    steps_per_interval = int(T_lyap / dt)

    for interval in range(n_intervals):
        # Evolve reference
        pos_ref = pos.copy()
        vel_ref = vel.copy()
        for step in range(steps_per_interval):
            pos_ref, vel_ref = yoshida6_step(pos_ref, vel_ref, masses, epsilon, dt)

        # Evolve all perturbed
        new_tangent_pos = np.zeros((n_vectors, N, 3))
        new_tangent_vel = np.zeros((n_vectors, N, 3))

        for vec_idx in range(n_vectors):
            pos_p = pos + tangent_pos[vec_idx]
            vel_p = vel + tangent_vel[vec_idx]
            for step in range(steps_per_interval):
                pos_p, vel_p = yoshida6_step(pos_p, vel_p, masses, epsilon, dt)
            new_tangent_pos[vec_idx] = pos_p - pos_ref
            new_tangent_vel[vec_idx] = vel_p - vel_ref

        # QR decomposition
        tangent_flat = np.concatenate([new_tangent_pos.reshape(n_vectors, -1),
                                        new_tangent_vel.reshape(n_vectors, -1)], axis=1)
        Q, R = np.linalg.qr(tangent_flat.T)
        norms = np.abs(np.diag(R))
        tangent_flat = Q.T

        for i in range(n_vectors):
            if norms[i] > 1e-15:
                lyapunov_sums[i] += np.log(norms[i])

        tangent_pos = tangent_flat[:, :N*3].reshape(n_vectors, N, 3)
        tangent_vel = tangent_flat[:, N*3:].reshape(n_vectors, N, 3)

        pos = pos_ref
        vel = vel_ref

    spectrum = lyapunov_sums / T_total

    E0 = compute_energy(pos, vel, masses, epsilon)
    pos_test = pos.copy()
    vel_test = vel.copy()
    for step in range(int(T_total / dt)):
        pos_test, vel_test = yoshida6_step(pos_test, vel_test, masses, epsilon, dt)
    E_final = compute_energy(pos_test, vel_test, masses, epsilon)
    energy_error = abs((E_final - E0) / E0)

    return spectrum[0], epsilon, energy_error

print("="*80)
print("RESOLVING λ DISCREPANCY")
print("="*80)
print()

print("Testing seed 0 (from original data: λ = -2.144)")
print()

print("METHOD 1: Original (single perturbation, norm tracking)")
lambda_orig, eps_orig, err_orig = method_original(0)
print(f"  λ = {lambda_orig:+.6f}")
print(f"  ε = {eps_orig:.4f}")
print(f"  δE/E = {err_orig:.2e}")
print()

print("METHOD 2: New (QR decomposition, multiple vectors)")
lambda_new, eps_new, err_new = method_new(0)
print(f"  λ = {lambda_new:+.6f}")
print(f"  ε = {eps_new:.4f}")
print(f"  δE/E = {err_new:.2e}")
print()

print("DISCREPANCY:")
print(f"  Δλ = {lambda_new - lambda_orig:+.6f}")
print(f"  Ratio = {lambda_new / lambda_orig:.2f}")
print()

# Load original results
with open('/home/user/Testing-env/data/results/30_seed_results.json', 'r') as f:
    original_data = json.load(f)

print("="*80)
print("TESTING ALL 30 SEEDS")
print("="*80)
print()

print(f"{'Seed':<6s} {'λ_orig':<12s} {'λ_new':<12s} {'Δλ':<12s} {'Status'}")
print("-"*60)

discrepancies = []

for result in original_data['results']:
    seed = result['seed']
    lambda_orig_data = result['lambda']

    lambda_orig, eps_orig, err_orig = method_original(seed)
    lambda_new, eps_new, err_new = method_new(seed)

    delta = lambda_new - lambda_orig

    if abs(delta) > 0.5:
        status = "BIG DIFF"
    elif lambda_orig * lambda_new < 0:
        status = "SIGN FLIP!"
    else:
        status = "OK"

    print(f"{seed:<6d} {lambda_orig:+<12.6f} {lambda_new:+<12.6f} {delta:+<12.6f} {status}")

    discrepancies.append((seed, lambda_orig, lambda_new, delta))

print()

# Analysis
lambda_orig_all = np.array([d[1] for d in discrepancies])
lambda_new_all = np.array([d[2] for d in discrepancies])

print("="*80)
print("STATISTICAL ANALYSIS")
print("="*80)
print()

print("Original method:")
print(f"  Mean λ = {np.mean(lambda_orig_all):+.6f}")
print(f"  Std λ  = {np.std(lambda_orig_all):.6f}")
print(f"  All negative: {np.all(lambda_orig_all < 0)}")
print()

print("New method:")
print(f"  Mean λ = {np.mean(lambda_new_all):+.6f}")
print(f"  Std λ  = {np.std(lambda_new_all):.6f}")
print(f"  All positive: {np.all(lambda_new_all > 0)}")
print()

print("="*80)
print("DIAGNOSIS")
print("="*80)
print()

print("The issue:")
print("  1. Original method gives λ < 0 (stable)")
print("  2. New method gives λ > 0 (chaotic)")
print("  3. BOTH cannot be right!")
print()

print("Possible causes:")
print()

print("HYPOTHESIS 1: Different initial perturbation size")
print("  Original: 1e-10")
print("  New: 1e-8")
print("  → Could affect transient dynamics")
print()

print("HYPOTHESIS 2: Normalization artifact")
print("  Original: Normalizes to initial perturbation size")
print("  New: QR decomposition (orthonormalizes)")
print("  → Different normalization conventions")
print()

print("HYPOTHESIS 3: Single vs multiple vectors")
print("  Original: Single perturbation direction")
print("  New: 6 orthogonal directions")
print("  → Maybe measuring different exponents?")
print()

print("HYPOTHESIS 4: BUG in original method")
print("  Original divides by (1e-10 * √(N*3))")
print("  This might be WRONG normalization!")
print()

print("="*80)
print("THE FIX")
print("="*80)
print()

print("To resolve this, I need to:")
print("  1. Check the original code for normalization bug")
print("  2. Test with SAME parameters (perturbation size, method)")
print("  3. Verify against analytical results (if available)")
print("  4. Determine which method is CORRECT")
print()

print("The correct Lyapunov exponent for N=3 should be:")
print("  - Computable from first principles")
print("  - Method-independent (if done correctly)")
print("  - Either positive OR negative, not both!")
print()

print("Next steps:")
print("  1. Fix the normalization in original method")
print("  2. Recompute with corrected method")
print("  3. Cross-validate with literature")
print()

print("="*80)
