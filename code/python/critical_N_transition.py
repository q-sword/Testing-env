#!/usr/bin/env python3
"""
Critical N Transition Analysis
==============================
Systematically scan N=3,4,5,...,30 to find where chaos emerges in N-body systems.

Key Questions:
1. At what N_c does λ transition from negative (stable) to positive (chaotic)?
2. How does λ scale with N?
3. Is there a sharp transition or gradual crossover?

Uses the validated Yoshida 6th order symplectic integrator with Benettin Lyapunov method.
"""

import numpy as np
from numba import njit, prange
import json
import time
from pathlib import Path

# Physical constants (normalized units)
G = 1.0
HBAR = 1.0

# Yoshida 6th order coefficients (8-stage)
W1 = -1.17767998417887
W2 = 0.235573213359357
W3 = 0.784513610477560
W0 = 1.0 - 2*(W1 + W2 + W3)

C = np.array([W3/2, (W3+W2)/2, (W2+W1)/2, (W1+W0)/2,
              (W0+W1)/2, (W1+W2)/2, (W2+W3)/2, W3/2])
D = np.array([W3, W2, W1, W0, W1, W2, W3, 0.0])


@njit
def compute_forces(positions, masses, epsilon):
    """Compute regularized gravitational forces."""
    N = len(masses)
    forces = np.zeros_like(positions)
    eps2 = epsilon * epsilon

    for i in range(N):
        for j in range(i+1, N):
            r_vec = positions[j] - positions[i]
            r2 = r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2
            r_soft = np.sqrt(r2 + eps2)
            f_mag = G * masses[i] * masses[j] / (r_soft * r_soft * r_soft)
            f_vec = f_mag * r_vec
            forces[i] += f_vec
            forces[j] -= f_vec

    return forces


@njit
def compute_accelerations(positions, masses, epsilon):
    """Compute accelerations from forces."""
    forces = compute_forces(positions, masses, epsilon)
    N = len(masses)
    acc = np.zeros_like(positions)
    for i in range(N):
        acc[i] = forces[i] / masses[i]
    return acc


@njit
def yoshida_step(positions, velocities, masses, epsilon, dt):
    """Single Yoshida 6th order step."""
    pos = positions.copy()
    vel = velocities.copy()

    for k in range(8):
        # Position update
        pos += C[k] * dt * vel
        # Velocity update (if D[k] != 0)
        if D[k] != 0.0:
            acc = compute_accelerations(pos, masses, epsilon)
            vel += D[k] * dt * acc

    return pos, vel


@njit
def compute_energy(positions, velocities, masses, epsilon):
    """Compute total energy."""
    N = len(masses)
    eps2 = epsilon * epsilon

    # Kinetic energy
    KE = 0.0
    for i in range(N):
        v2 = velocities[i, 0]**2 + velocities[i, 1]**2 + velocities[i, 2]**2
        KE += 0.5 * masses[i] * v2

    # Potential energy (regularized)
    PE = 0.0
    for i in range(N):
        for j in range(i+1, N):
            r_vec = positions[j] - positions[i]
            r2 = r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2
            r_soft = np.sqrt(r2 + eps2)
            PE -= G * masses[i] * masses[j] / r_soft

    return KE + PE


@njit
def tangent_evolution(positions, velocities, masses, epsilon, dt, delta):
    """Evolve tangent vector for Lyapunov calculation."""
    N = len(masses)

    # Normalize tangent vector
    norm = 0.0
    for i in range(N):
        for j in range(3):
            norm += delta[i, j]**2 + delta[N+i, j]**2
    norm = np.sqrt(norm)
    delta = delta / norm

    # Perturbed state
    pos_pert = positions + delta[:N] * 1e-8
    vel_pert = velocities + delta[N:] * 1e-8

    # Evolve both states
    pos_new, vel_new = yoshida_step(positions, velocities, masses, epsilon, dt)
    pos_pert_new, vel_pert_new = yoshida_step(pos_pert, vel_pert, masses, epsilon, dt)

    # New tangent vector
    delta_new = np.zeros((2*N, 3))
    delta_new[:N] = (pos_pert_new - pos_new) / 1e-8
    delta_new[N:] = (vel_pert_new - vel_new) / 1e-8

    # Compute stretching factor
    new_norm = 0.0
    for i in range(N):
        for j in range(3):
            new_norm += delta_new[i, j]**2 + delta_new[N+i, j]**2
    new_norm = np.sqrt(new_norm)

    return pos_new, vel_new, delta_new, np.log(new_norm)


def initialize_N_body(N, seed=None):
    """Initialize N-body system in virial equilibrium."""
    if seed is not None:
        np.random.seed(seed)

    # Equal masses normalized to total mass = 1
    masses = np.ones(N) / N

    # Random positions in unit sphere
    positions = np.random.randn(N, 3)
    for i in range(N):
        r = np.sqrt(positions[i, 0]**2 + positions[i, 1]**2 + positions[i, 2]**2)
        if r > 1:
            positions[i] /= r

    # Center of mass at origin
    com = np.sum(positions * masses[:, np.newaxis], axis=0)
    positions -= com

    # Compute potential energy
    eps2 = 0.01  # Small regularization for initialization
    PE = 0.0
    for i in range(N):
        for j in range(i+1, N):
            r_vec = positions[j] - positions[i]
            r = np.sqrt(np.sum(r_vec**2) + eps2)
            PE -= G * masses[i] * masses[j] / r

    # Virial equilibrium: 2*KE = -PE
    target_KE = -PE / 2

    # Random velocities
    velocities = np.random.randn(N, 3)

    # Zero total momentum
    total_mom = np.sum(velocities * masses[:, np.newaxis], axis=0)
    velocities -= total_mom / np.sum(masses)

    # Scale to target KE
    current_KE = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
    velocities *= np.sqrt(target_KE / current_KE)

    return positions, velocities, masses


def compute_lyapunov_for_N(N, T_max=50, dt=0.001, num_seeds=5, verbose=True):
    """
    Compute Lyapunov exponent for N-body system.

    Returns: mean λ, std λ, list of individual λ values
    """
    # Compute quantum regularization scale
    # For N bodies of total mass 1, each has mass 1/N
    # Typical velocity from virial: v_rms ~ sqrt(G*M_total/r) ~ 1
    v_rms = 1.0
    m_typical = 1.0 / N
    epsilon = HBAR / (m_typical * v_rms)

    if verbose:
        print(f"  N={N}: ε={epsilon:.4f}, testing {num_seeds} seeds...")

    lyapunov_values = []
    energy_drifts = []

    for seed in range(num_seeds):
        # Initialize
        pos, vel, masses = initialize_N_body(N, seed=seed*1000 + N)

        # Initial tangent vector (random)
        delta = np.random.randn(2*N, 3)
        norm = np.sqrt(np.sum(delta**2))
        delta /= norm

        # Initial energy
        E0 = compute_energy(pos, vel, masses, epsilon)

        # Integrate and accumulate Lyapunov sum
        lyap_sum = 0.0
        t = 0.0
        renorm_interval = 100  # Renormalize every 100 steps
        step = 0

        while t < T_max:
            for _ in range(renorm_interval):
                pos, vel, delta, log_stretch = tangent_evolution(
                    pos, vel, masses, epsilon, dt, delta
                )
                lyap_sum += log_stretch
                step += 1
            t = step * dt

        # Final Lyapunov exponent
        total_time = step * dt
        lyapunov = lyap_sum / total_time
        lyapunov_values.append(lyapunov)

        # Energy drift
        E_final = compute_energy(pos, vel, masses, epsilon)
        drift = abs(E_final - E0) / abs(E0)
        energy_drifts.append(drift)

    mean_lyap = np.mean(lyapunov_values)
    std_lyap = np.std(lyapunov_values)
    mean_drift = np.mean(energy_drifts)

    if verbose:
        status = "STABLE (λ<0)" if mean_lyap < 0 else "CHAOTIC (λ>0)"
        print(f"    λ = {mean_lyap:.4f} ± {std_lyap:.4f}, δE/E = {mean_drift:.2e} [{status}]")

    return mean_lyap, std_lyap, lyapunov_values, mean_drift


def run_N_transition_scan():
    """
    Systematic scan of N=3 to N=30 to find chaos transition.
    """
    print("="*70)
    print("CRITICAL N TRANSITION ANALYSIS")
    print("Finding where chaos emerges in N-body gravitational systems")
    print("="*70)
    print()

    results = {
        'description': 'N-body chaos transition scan',
        'method': 'Benettin Lyapunov with Yoshida 6th order integrator',
        'regularization': 'ε = ℏ/(m·v_rms), quantum de Broglie scale',
        'T_max': 50,
        'dt': 0.001,
        'num_seeds': 5,
        'data': []
    }

    N_values = list(range(3, 31))  # N=3 to N=30

    print("Parameters:")
    print(f"  Integration time: T = {results['T_max']}")
    print(f"  Timestep: dt = {results['dt']}")
    print(f"  Seeds per N: {results['num_seeds']}")
    print(f"  N range: {N_values[0]} to {N_values[-1]}")
    print()

    start_time = time.time()

    for N in N_values:
        mean_lyap, std_lyap, all_lyaps, energy_drift = compute_lyapunov_for_N(
            N, T_max=results['T_max'], dt=results['dt'],
            num_seeds=results['num_seeds'], verbose=True
        )

        # Compute epsilon for this N
        m_typical = 1.0 / N
        epsilon = HBAR / (m_typical * 1.0)  # v_rms ~ 1

        results['data'].append({
            'N': N,
            'epsilon': epsilon,
            'lambda_mean': mean_lyap,
            'lambda_std': std_lyap,
            'lambda_values': all_lyaps,
            'energy_drift': energy_drift,
            'stable': mean_lyap < 0
        })

    elapsed = time.time() - start_time
    results['runtime_seconds'] = elapsed

    # Analysis
    print()
    print("="*70)
    print("ANALYSIS")
    print("="*70)

    # Find transition point
    stable_N = [d['N'] for d in results['data'] if d['stable']]
    chaotic_N = [d['N'] for d in results['data'] if not d['stable']]

    print()
    print(f"Stable (λ < 0): N = {stable_N if stable_N else 'None'}")
    print(f"Chaotic (λ > 0): N = {chaotic_N if chaotic_N else 'None'}")

    if stable_N and chaotic_N:
        N_c = min(chaotic_N)
        print(f"\n*** Critical N_c ≈ {N_c} ***")
        print(f"    Chaos emerges at N ≥ {N_c}")
    elif not chaotic_N:
        print("\n*** ALL SYSTEMS STABLE ***")
        print("    No chaos found up to N=30!")
    else:
        print("\n*** ALL SYSTEMS CHAOTIC ***")
        print("    Even N=3 shows chaos!")

    # Scaling analysis
    N_arr = np.array([d['N'] for d in results['data']])
    lambda_arr = np.array([d['lambda_mean'] for d in results['data']])

    # Fit λ vs N (for chaotic regime)
    if len(chaotic_N) >= 3:
        chaotic_mask = lambda_arr > 0
        if np.sum(chaotic_mask) >= 3:
            log_N = np.log(N_arr[chaotic_mask])
            log_lambda = np.log(lambda_arr[chaotic_mask])
            coeffs = np.polyfit(log_N, log_lambda, 1)
            scaling_exp = coeffs[0]
            print(f"\nScaling in chaotic regime: λ ∝ N^{scaling_exp:.3f}")
            results['scaling_exponent'] = scaling_exp

    # Save results
    output_path = Path('/home/user/Testing-env/data/results/N_transition_scan.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Total runtime: {elapsed:.1f} seconds")

    # Summary table
    print()
    print("="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'N':>4} | {'ε':>8} | {'λ_mean':>10} | {'λ_std':>8} | {'δE/E':>10} | Status")
    print("-"*70)
    for d in results['data']:
        status = "STABLE" if d['stable'] else "CHAOTIC"
        print(f"{d['N']:>4} | {d['epsilon']:>8.4f} | {d['lambda_mean']:>10.4f} | "
              f"{d['lambda_std']:>8.4f} | {d['energy_drift']:>10.2e} | {status}")

    return results


if __name__ == "__main__":
    results = run_N_transition_scan()
