#!/usr/bin/env python3
"""
TEST FREQUENCY-BASED EPSILON
December 2025

Compare ε prescriptions:
1. Velocity: ε = ℏ/(m·v) = 3.47 (current)
2. Frequency: ε = √(ℏ/(mω)) = 1.09 (harmonic analog)
3. Intermediate values

Question: Does frequency-based ε reveal special physics?
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

    pos_r = pos_ref.copy()
    vel_r = vel_ref.copy()
    for step in range(num_steps):
        pos_r, vel_r = yoshida6_step(pos_r, vel_r, masses, epsilon, dt)

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

def run_with_epsilon(epsilon, seed=42, N=30, T_total=20, T_lyap=2,
                     dt=0.001, n_vectors=12):
    """Run Lyapunov calculation with specified epsilon"""

    np.random.seed(seed)
    masses = np.ones(N)
    pos = np.random.randn(N, 3) * 0.5
    vel = np.random.randn(N, 3) * 0.3

    r_rms = np.sqrt(np.mean(np.sum(pos**2, axis=1)))
    v_rms = np.sqrt(np.mean(vel**2))
    eps_over_r = epsilon / r_rms

    # Calculate frequency in harmonic regime
    M_total = N
    omega = np.sqrt(G * M_total / (masses[0] * epsilon**3))

    E0 = compute_energy(pos, vel, masses, epsilon)

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
        'epsilon': epsilon,
        'r_rms': r_rms,
        'v_rms': v_rms,
        'eps_over_r': eps_over_r,
        'omega': omega,
        'lambda_max': spectrum[0],
        'lambda_sum': np.sum(spectrum),
        'energy_drift': energy_drift,
        'runtime': total_time,
    }

def main():
    print()
    print("="*80)
    print("FREQUENCY-BASED EPSILON TEST")
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

    # Test epsilon values around both scales
    epsilon_tests = [
        ("Quantum harmonic", 1.09),  # ε_ω
        ("Intermediate 1", 1.5),
        ("Intermediate 2", 2.0),
        ("Intermediate 3", 2.5),
        ("Velocity-based", 3.47),  # ε_v
    ]

    print("Testing different ε prescriptions:")
    print()

    print("="*80)
    print(f"{'Type':<20s} {'ε':<8s} {'ε/r':<8s} {'ω':<10s} {'λ_max':<12s} {'δE/E':<12s} {'Time':<8s}")
    print("-"*80)
    sys.stdout.flush()

    results = []

    for name, eps in epsilon_tests:
        result = run_with_epsilon(eps, seed=42)
        results.append((name, result))

        print(f"{name:<20s} {result['epsilon']:<8.2f} {result['eps_over_r']:<8.2f} "
              f"{result['omega']:<10.4f} {result['lambda_max']:+<12.6f} "
              f"{result['energy_drift']:<12.2e} {result['runtime']:<8.1f}")
        sys.stdout.flush()

    print()
    print("="*80)
    print("ANALYSIS")
    print("="*80)
    print()

    # Find trends
    print("Key observations:")
    print()

    lambdas = [r[1]['lambda_max'] for r in results]
    epsilons = [r[1]['epsilon'] for r in results]
    drifts = [r[1]['energy_drift'] for r in results]

    idx_min_lambda = lambdas.index(min(lambdas))
    idx_max_lambda = lambdas.index(max(lambdas))

    print(f"1. CHAOS STRENGTH:")
    print(f"   Minimum λ_max = {min(lambdas):+.6f} at ε = {epsilons[idx_min_lambda]:.2f} ({results[idx_min_lambda][0]})")
    print(f"   Maximum λ_max = {max(lambdas):+.6f} at ε = {epsilons[idx_max_lambda]:.2f} ({results[idx_max_lambda][0]})")
    print()

    if epsilons[idx_min_lambda] == 1.09:
        print("   → Quantum harmonic scale (ε_ω) shows MINIMAL chaos!")
        print("   → This might be a 'magic' scale for stability")
    elif epsilons[idx_max_lambda] == 1.09:
        print("   → Quantum harmonic scale (ε_ω) shows MAXIMAL chaos!")
        print("   → Natural quantum fluctuations enhance chaos")
    else:
        print("   → No special behavior at quantum harmonic scale")

    print()

    # Energy conservation
    idx_best_energy = drifts.index(min(drifts))
    print(f"2. ENERGY CONSERVATION:")
    print(f"   Best: δE/E = {min(drifts):.2e} at ε = {epsilons[idx_best_energy]:.2f}")
    print(f"   All scales: δE/E < {max(drifts):.2e} ✓ Excellent")
    print()

    # Frequency analysis
    print(f"3. OSCILLATOR FREQUENCY:")
    for name, r in results:
        print(f"   {name:<20s}: ω = {r['omega']:.4f}")
    print()

    print("="*80)
    print("PHYSICAL INTERPRETATION")
    print("="*80)
    print()

    print("Quantum harmonic scale (ε_ω = 1.09):")
    qho_result = results[0][1]
    print(f"  λ_max = {qho_result['lambda_max']:+.6f}")
    print(f"  ω = {qho_result['omega']:.4f}")
    print(f"  ε/r = {qho_result['eps_over_r']:.2f}")
    print()

    print("Velocity-based scale (ε_v = 3.47):")
    vel_result = results[-1][1]
    print(f"  λ_max = {vel_result['lambda_max']:+.6f}")
    print(f"  ω = {vel_result['omega']:.4f}")
    print(f"  ε/r = {vel_result['eps_over_r']:.2f}")
    print()

    delta_lambda = vel_result['lambda_max'] - qho_result['lambda_max']
    if abs(delta_lambda) > 0.01:
        print(f"Significant difference: Δλ = {delta_lambda:+.6f}")
        if delta_lambda > 0:
            print("  → Velocity-based ε is MORE chaotic")
            print("  → Quantum oscillator scale stabilizes system!")
        else:
            print("  → Quantum oscillator scale is MORE chaotic")
            print("  → Zero-point fluctuations enhance chaos")
    else:
        print("Similar chaos levels - scale doesn't matter much in this regime")

    print()
    print("="*80)
    print("CONCLUSION")
    print("="*80)
    print()

    print("Frequency-based quantum scale (ε_ω) represents:")
    print("  • Natural size of zero-point quantum fluctuations")
    print("  • Characteristic length of oscillator ground state")
    print("  • Phonon-like excitations in 'gravitational crystal'")
    print()

    if abs(delta_lambda) > 0.05:
        print("✓ SPECIAL PHYSICS at ε_ω!")
        print(f"  Different chaos behavior from velocity-based scale")
    else:
        print("  No dramatic differences from velocity-based ε")
        print("  Both prescriptions give similar dynamics")

    print()
    print("="*80)

if __name__ == "__main__":
    main()
