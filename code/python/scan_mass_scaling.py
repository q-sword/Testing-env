#!/usr/bin/env python3
"""
QUANTUM → CLASSICAL TRANSITION SCAN
December 2025

Run N=30 Lyapunov calculations at multiple mass scales to measure
λ_max(ε/r) empirically and find where quantum effects become negligible.

Strategy: Scale particle mass from M=1 to M=300
"""

import numpy as np
from numba import njit, prange
import time
import sys

G = 1.0
HBAR = 1.0

w1 = 0.78451361047755726382
w2 = 0.23557321335935813368
w3 = -1.17767998417887100695
w0 = 1.0 - 2.0*(w1 + w2 + w3)

C = np.array([w3, w2, w1, w0, w1, w2, w3, 0.0])
D = np.array([w3/2, (w3+w2)/2, (w2+w1)/2, (w1+w0)/2,
              (w0+w1)/2, (w1+w2)/2, (w2+w3)/2, w3/2])

@njit
def compute_forces_exact(pos, masses, epsilon):
    """Exact pairwise forces (serial for trajectory parallelism)"""
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
        acc = compute_forces_exact(pos, masses, epsilon)
        vel = vel + D[i] * dt * acc
        if i < len(C) - 1 or C[i] != 0.0:
            pos = pos + C[i] * dt * vel
    return pos, vel

@njit(parallel=True)
def evolve_all_tangents_exact(pos_ref, vel_ref, tangent_pos, tangent_vel,
                               masses, epsilon, dt, num_steps):
    n_vectors = tangent_pos.shape[0]
    N = len(masses)

    # Evolve reference
    pos_r = pos_ref.copy()
    vel_r = vel_ref.copy()
    for step in range(num_steps):
        pos_r, vel_r = yoshida6_step(pos_r, vel_r, masses, epsilon, dt)

    # Evolve ALL perturbed systems in PARALLEL
    new_tangent_pos = np.zeros((n_vectors, N, 3))
    new_tangent_vel = np.zeros((n_vectors, N, 3))

    for vec_idx in prange(n_vectors):
        pos_p = pos_ref + tangent_pos[vec_idx]
        vel_p = vel_ref + tangent_vel[vec_idx]
        for step in range(num_steps):
            pos_p, vel_p = yoshida6_step(pos_p, vel_p, masses, epsilon, dt)
        new_tangent_pos[vec_idx] = pos_p - pos_r
        new_tangent_vel[vec_idx] = vel_p - vel_r

    return pos_r, vel_r, new_tangent_pos, new_tangent_vel

def full_qr_decomposition(vectors):
    Q, R = np.linalg.qr(vectors.T)
    norms = np.abs(np.diag(R))
    return Q.T, norms

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

def run_single_mass_scale(mass_scale, seed=42, N=30, T_total=20, T_lyap=2,
                          dt=0.001, n_vectors=12):
    """
    Run Lyapunov calculation for a single mass scaling

    Shorter than full T=50 for speed: T=20, intervals every 2 time units
    """

    np.random.seed(seed)

    # Base masses (scaled)
    masses = mass_scale * np.ones(N)

    # Initial conditions (same spatial/velocity distribution)
    pos = np.random.randn(N, 3) * 0.5
    vel = np.random.randn(N, 3) * 0.3

    # Quantum regularization: ε = ℏ/(m·v_rms)
    v_rms = np.sqrt(np.mean(vel**2))
    epsilon = HBAR / (masses[0] * v_rms)

    # System properties
    r_rms = np.sqrt(np.mean(np.sum(pos**2, axis=1)))
    eps_over_r = epsilon / r_rms

    E0 = compute_energy(pos, vel, masses, epsilon)

    # Initialize tangent vectors
    tangent_pos = np.random.randn(n_vectors, N, 3) * 1e-8
    tangent_vel = np.random.randn(n_vectors, N, 3) * 1e-8

    tangent_flat = np.concatenate([tangent_pos.reshape(n_vectors, -1),
                                    tangent_vel.reshape(n_vectors, -1)], axis=1)
    tangent_flat, _ = full_qr_decomposition(tangent_flat)

    tangent_pos = tangent_flat[:, :N*3].reshape(n_vectors, N, 3)
    tangent_vel = tangent_flat[:, N*3:].reshape(n_vectors, N, 3)

    lyapunov_sums = np.zeros(n_vectors)
    n_intervals = int(T_total / T_lyap)
    steps_per_interval = int(T_lyap / dt)

    start_time = time.time()

    for interval in range(n_intervals):
        pos, vel, new_tangent_pos, new_tangent_vel = evolve_all_tangents_exact(
            pos, vel, tangent_pos, tangent_vel, masses, epsilon, dt, steps_per_interval
        )

        tangent_flat = np.concatenate([new_tangent_pos.reshape(n_vectors, -1),
                                        new_tangent_vel.reshape(n_vectors, -1)], axis=1)

        tangent_flat, norms = full_qr_decomposition(tangent_flat)

        for i in range(n_vectors):
            if norms[i] > 1e-15:
                lyapunov_sums[i] += np.log(norms[i])

        tangent_pos = tangent_flat[:, :N*3].reshape(n_vectors, N, 3)
        tangent_vel = tangent_flat[:, N*3:].reshape(n_vectors, N, 3)

    spectrum = lyapunov_sums / T_total
    E_final = compute_energy(pos, vel, masses, epsilon)
    energy_drift = abs((E_final - E0) / E0)
    total_time = time.time() - start_time

    return {
        'mass_scale': mass_scale,
        'epsilon': epsilon,
        'r_rms': r_rms,
        'eps_over_r': eps_over_r,
        'v_rms': v_rms,
        'lambda_max': spectrum[0],
        'lambda_sum': np.sum(spectrum),
        'spectrum': spectrum,
        'energy_drift': energy_drift,
        'runtime': total_time,
        'E0': E0,
        'E_final': E_final,
    }

def main():
    print()
    print("="*80)
    print("QUANTUM → CLASSICAL TRANSITION SCAN")
    print("="*80)
    print()

    print("JIT warmup...")
    np.random.seed(0)
    _ = evolve_all_tangents_exact(
        np.random.randn(5, 3) * 0.5,
        np.random.randn(5, 3) * 0.3,
        np.random.randn(3, 5, 3) * 1e-8,
        np.random.randn(3, 5, 3) * 1e-8,
        np.ones(5),
        1.0,
        0.001,
        10
    )
    print("Ready!")
    print()

    # Mass scaling factors to test
    mass_scales = [1, 3, 10, 30, 100, 300]

    print("Testing mass scales:", mass_scales)
    print(f"Each run: N=30, T=20, dt=0.001, 12 exponents")
    print()

    print("="*80)
    print(f"{'M':<6s} {'ε':<10s} {'ε/r':<10s} {'λ_max':<12s} {'Σλ':<12s} {'δE/E':<12s} {'Time':<8s} {'Regime'}")
    print("-"*80)
    sys.stdout.flush()

    results = []

    for M in mass_scales:
        result = run_single_mass_scale(M, seed=42)
        results.append(result)

        # Determine regime
        eps_r = result['eps_over_r']
        if eps_r > 0.5:
            regime = "Quantum"
        elif eps_r > 0.1:
            regime = "Transition"
        elif eps_r > 0.01:
            regime = "Classical"
        else:
            regime = "Pure class."

        print(f"{M:<6.0f} {result['epsilon']:<10.4f} {result['eps_over_r']:<10.4f} "
              f"{result['lambda_max']:+<12.6f} {result['lambda_sum']:+<12.6f} "
              f"{result['energy_drift']:<12.2e} {result['runtime']:<8.1f} {regime}")
        sys.stdout.flush()

    print()
    print("="*80)
    print("ANALYSIS")
    print("="*80)
    print()

    # Find trends
    eps_r_vals = np.array([r['eps_over_r'] for r in results])
    lambda_vals = np.array([r['lambda_max'] for r in results])

    print("Trend analysis:")
    print()

    # Most quantum vs most classical
    i_quantum = np.argmax(eps_r_vals)
    i_classical = np.argmin(eps_r_vals)

    print(f"Most quantum (M={results[i_quantum]['mass_scale']}):")
    print(f"  ε/r = {results[i_quantum]['eps_over_r']:.3f}")
    print(f"  λ_max = {results[i_quantum]['lambda_max']:+.6f}")
    print()

    print(f"Most classical (M={results[i_classical]['mass_scale']}):")
    print(f"  ε/r = {results[i_classical]['eps_over_r']:.4f}")
    print(f"  λ_max = {results[i_classical]['lambda_max']:+.6f}")
    print()

    # Change in chaos
    delta_lambda = results[i_classical]['lambda_max'] - results[i_quantum]['lambda_max']
    factor = results[i_classical]['lambda_max'] / results[i_quantum]['lambda_max']

    print(f"Change in chaos strength:")
    print(f"  Δλ_max = {delta_lambda:+.6f}")
    print(f"  Ratio = {factor:.2f}×")
    print()

    if delta_lambda > 0:
        print("✓ Hypothesis CONFIRMED: Classical limit is MORE chaotic")
        print("  Quantum regularization STABILIZES the system")
    else:
        print("✗ Hypothesis REJECTED: Classical limit is LESS chaotic")
        print("  Quantum regularization does not stabilize")

    print()

    # Energy conservation check
    max_drift = max(r['energy_drift'] for r in results)
    print(f"Energy conservation: max δE/E = {max_drift:.2e}")
    if max_drift < 1e-10:
        print("✓ Excellent across all scales")
    elif max_drift < 1e-6:
        print("✓ Good")
    else:
        print("⚠ May have issues at some scales")

    print()
    print("="*80)
    print()

    # Save results
    with open('/tmp/mass_scaling_results.txt', 'w') as f:
        f.write("M,epsilon,eps_over_r,lambda_max,lambda_sum,energy_drift,runtime\\n")
        for r in results:
            f.write(f"{r['mass_scale']},{r['epsilon']},{r['eps_over_r']},"
                   f"{r['lambda_max']},{r['lambda_sum']},{r['energy_drift']},{r['runtime']}\\n")

    print("Results saved to: /tmp/mass_scaling_results.txt")
    print()

if __name__ == "__main__":
    main()
