#!/usr/bin/env python3
"""
BENETTIN METHOD - PROPER UNITS

Fix the units issue. Use DIMENSIONLESS units where:
  G = 1 (gravitational constant)
  M_total = 1 (total mass)
  R_typical = 1 (typical separation)

Then ε is dimensionless and should be O(1).
"""

import numpy as np
from numba import njit
import json

print("="*80)
print("BENETTIN METHOD - 30 SEED VALIDATION (PROPER UNITS)")
print("="*80)
print()

# ===========================================================================
# YOSHIDA 6TH ORDER SYMPLECTIC INTEGRATOR
# ===========================================================================

@njit
def yoshida_step(pos, vel, masses, epsilon, dt):
    """Yoshida 6th order symplectic integrator"""
    w0 = -1.17767998417887
    w1 = 0.235573213359357
    w2 = 0.784513610477560

    c1 = w2 / 2
    c2 = (w1 + w2) / 2
    c3 = (w0 + w1) / 2
    c4 = c3
    c5 = c2
    c6 = c1

    d1 = w2
    d2 = w1
    d3 = w0
    d4 = d2
    d5 = d1

    pos += c1 * dt * vel
    acc = compute_acceleration(pos, masses, epsilon)
    vel += d1 * dt * acc

    pos += c2 * dt * vel
    acc = compute_acceleration(pos, masses, epsilon)
    vel += d2 * dt * acc

    pos += c3 * dt * vel
    acc = compute_acceleration(pos, masses, epsilon)
    vel += d3 * dt * acc

    pos += c4 * dt * vel
    acc = compute_acceleration(pos, masses, epsilon)
    vel += d4 * dt * acc

    pos += c5 * dt * vel
    acc = compute_acceleration(pos, masses, epsilon)
    vel += d5 * dt * acc

    pos += c6 * dt * vel

    return pos, vel

@njit
def compute_acceleration(pos, masses, epsilon):
    """Compute accelerations with quantum regularization"""
    N = len(pos)
    acc = np.zeros_like(pos)

    for i in range(N):
        for j in range(N):
            if i != j:
                r_vec = pos[j] - pos[i]
                r = np.sqrt(np.sum(r_vec**2))
                r_eff_3 = (r**2 + epsilon**2)**(3/2)
                force_mag = masses[j] / r_eff_3
                acc[i] += force_mag * r_vec

    return acc

@njit
def compute_energy(pos, vel, masses, epsilon):
    """Compute total energy"""
    N = len(pos)

    KE = 0.0
    for i in range(N):
        KE += 0.5 * masses[i] * np.sum(vel[i]**2)

    PE = 0.0
    for i in range(N):
        for j in range(i+1, N):
            r_vec = pos[j] - pos[i]
            r = np.sqrt(np.sum(r_vec**2))
            r_eff = np.sqrt(r**2 + epsilon**2)
            PE -= masses[i] * masses[j] / r_eff

    return KE + PE

# ===========================================================================
# BENETTIN METHOD
# ===========================================================================

@njit
def benettin_lyapunov(pos_init, vel_init, masses, epsilon, dt, T_total, T_renorm):
    """Compute Lyapunov exponent using Benettin method"""
    N = len(pos_init)
    delta_0 = 1e-10

    pos_ref = pos_init.copy()
    vel_ref = vel_init.copy()

    pos_pert = pos_init.copy()
    vel_pert = vel_init.copy()

    # Add perturbation
    delta_pos = np.random.randn(N, 3)
    delta_pos = delta_pos / np.linalg.norm(delta_pos) * delta_0
    pos_pert += delta_pos

    log_sum = 0.0
    n_renorm = 0

    n_steps_total = int(T_total / dt)
    n_steps_renorm = int(T_renorm / dt)

    for step in range(n_steps_total):
        pos_ref, vel_ref = yoshida_step(pos_ref, vel_ref, masses, epsilon, dt)
        pos_pert, vel_pert = yoshida_step(pos_pert, vel_pert, masses, epsilon, dt)

        if (step + 1) % n_steps_renorm == 0:
            delta_pos = pos_pert - pos_ref
            delta_vel = vel_pert - vel_ref
            delta_full = np.concatenate((delta_pos.flatten(), delta_vel.flatten()))
            norm = np.linalg.norm(delta_full)

            log_sum += np.log(norm / delta_0)
            n_renorm += 1

            if norm > 0:
                pos_pert = pos_ref + delta_pos * (delta_0 / norm)
                vel_pert = vel_ref + delta_vel * (delta_0 / norm)

    lambda_max = log_sum / T_total

    E_final = compute_energy(pos_ref, vel_ref, masses, epsilon)
    E_init = compute_energy(pos_init, vel_init, masses, epsilon)
    energy_error = abs(E_final - E_init) / abs(E_init) if abs(E_init) > 0 else 0.0

    return lambda_max, epsilon, energy_error

# ===========================================================================
# INITIALIZATION (DIMENSIONLESS UNITS)
# ===========================================================================

@njit
def initialize_nbody(N, seed):
    """
    Initialize N-body system in DIMENSIONLESS units:
      G = 1
      M_total = 1 → each mass = 1/N
      R_typical ~ 1
    """
    np.random.seed(seed)

    masses = np.ones(N) / N  # Equal masses summing to 1

    # Random positions in unit sphere
    pos = np.random.randn(N, 3)
    for i in range(N):
        r = np.linalg.norm(pos[i])
        if r > 0:
            pos[i] = pos[i] / r * np.random.rand()**(1/3)

    # Remove center of mass
    cm_pos = np.sum(pos, axis=0) / N
    pos -= cm_pos

    # Initialize velocities from virial
    # For bound system: 2*KE + PE ≈ 0 (virial theorem)
    epsilon_temp = 0.1

    PE = 0.0
    for i in range(N):
        for j in range(i+1, N):
            r_vec = pos[j] - pos[i]
            r = np.sqrt(np.sum(r_vec**2))
            r_eff = np.sqrt(r**2 + epsilon_temp**2)
            PE -= masses[i] * masses[j] / r_eff

    KE_target = -PE / 2.0

    # Random velocities
    vel = np.random.randn(N, 3)

    # Scale to target KE
    KE_current = 0.5 * np.sum(vel**2) / N
    if KE_current > 0:
        scale = np.sqrt(KE_target / KE_current)
        vel *= scale

    # Remove center of mass velocity
    cm_vel = np.sum(vel, axis=0) / N
    vel -= cm_vel

    return pos, vel, masses

@njit
def compute_epsilon_from_system(pos, vel, masses):
    """
    Compute ε from system parameters in dimensionless units.

    For ε_v = ℏ/(mv), in dimensionless units with ℏ = 1, m = 1/N:
      ε_v = 1 / ((1/N) * v_rms) = N / v_rms
    """
    N = len(pos)

    # RMS velocity
    v_squared = 0.0
    for i in range(N):
        v_squared += np.sum(vel[i]**2)
    v_rms = np.sqrt(v_squared / N)

    # CORRECTED: Use original formula ε = ℏ/(m * v_rms)
    # In dimensionless units with ℏ = 1, m = 1/N:
    #   ε = 1 / ((1/N) * v_rms) = N / v_rms
    epsilon = N / v_rms if v_rms > 0 else float(N)

    return epsilon

# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("DIMENSIONLESS UNITS:")
    print("-"*80)
    print("  G = 1 (gravitational constant)")
    print("  M_total = 1 (total mass)")
    print("  R_typical ~ 1 (typical separation)")
    print("  ε ~ O(1) (quantum scale)")
    print()

    N = 30
    dt = 0.01  # Balance of speed and accuracy
    T_total = 200.0  # Long enough to see convergence
    T_renorm = 10.0
    n_seeds = 30

    print(f"N particles: {N}")
    print(f"Timestep: {dt}")
    print(f"Total time: {T_total}")
    print(f"Renormalization interval: {T_renorm}")
    print(f"Number of seeds: {n_seeds}")
    print()

    print("="*80)
    print("RUNNING BENETTIN METHOD ON 30 SEEDS")
    print("="*80)
    print()

    print(f"{'Seed':<6s} {'λ':<15s} {'ε':<12s} {'δE/E':<12s} {'Status':<15s}")
    print("-"*80)

    results = []

    for seed in range(n_seeds):
        pos, vel, masses = initialize_nbody(N, seed)
        epsilon = compute_epsilon_from_system(pos, vel, masses)

        lambda_max, eps_used, energy_error = benettin_lyapunov(
            pos, vel, masses, epsilon, dt, T_total, T_renorm
        )

        if lambda_max < -0.001:
            status = "STABLE"
        elif lambda_max < 0:
            status = "WEAKLY STABLE"
        elif abs(lambda_max) < 0.001:
            status = "NEUTRAL"
        else:
            status = "CHAOTIC"

        print(f"{seed:<6d} {lambda_max:<15.6f} {epsilon:<12.6f} {energy_error:<12.2e} {status:<15s}")

        results.append({
            'seed': int(seed),
            'lambda': float(lambda_max),
            'epsilon': float(epsilon),
            'energy_error': float(energy_error),
            'status': status
        })

    print()
    print("="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)
    print()

    lambdas = np.array([r['lambda'] for r in results])

    print(f"Mean λ: {np.mean(lambdas):.8f}")
    print(f"Std λ:  {np.std(lambdas):.8f}")
    print(f"Min λ:  {np.min(lambdas):.8f}")
    print(f"Max λ:  {np.max(lambdas):.8f}")
    print()

    n_negative = np.sum(lambdas < -0.001)
    n_positive = np.sum(lambdas > 0.001)
    n_neutral = np.sum(np.abs(lambdas) < 0.001)

    print(f"Seeds with λ < -0.001:  {n_negative}/{n_seeds} ({100*n_negative/n_seeds:.1f}%)")
    print(f"Seeds with λ > +0.001:  {n_positive}/{n_seeds} ({100*n_positive/n_seeds:.1f}%)")
    print(f"Seeds with |λ| < 0.001: {n_neutral}/{n_seeds} ({100*n_neutral/n_seeds:.1f}%)")
    print()

    max_energy_error = max([r['energy_error'] for r in results])
    mean_energy_error = np.mean([r['energy_error'] for r in results])

    print(f"Energy conservation:")
    print(f"  Mean δE/E: {mean_energy_error:.2e}")
    print(f"  Max δE/E:  {max_energy_error:.2e}")
    print()

    print("="*80)
    print("PHYSICAL INTERPRETATION")
    print("="*80)
    print()

    if np.mean(lambdas) < -0.001:
        print("✓ VALIDATION SUCCESSFUL")
        print()
        print(f"Mean λ = {np.mean(lambdas):.6f} < 0:")
        print("  - Perturbations SHRINK exponentially")
        print("  - System is STABLE")
        print("  - This is why molecules exist!")
    elif abs(np.mean(lambdas)) < 0.001:
        print("⚠ MARGINAL RESULT")
        print()
        print(f"Mean λ ≈ {np.mean(lambdas):.8f} ≈ 0:")
        print("  - System is NEUTRALLY stable")
        print("  - May need longer integration time")
        print("  - Or stronger quantum regularization")
    else:
        print("✗ VALIDATION FAILED")
        print()
        print(f"Mean λ = {np.mean(lambdas):.6f} > 0:")
        print("  - System appears chaotic")
        print("  - Contradicts physical reality")

    print()
    print("="*80)

    output = {
        'method': 'Benettin (1980) - Dimensionless Units',
        'n_seeds': n_seeds,
        'N_particles': N,
        'dt': dt,
        'T_total': T_total,
        'T_renorm': T_renorm,
        'statistics': {
            'mean_lambda': float(np.mean(lambdas)),
            'std_lambda': float(np.std(lambdas)),
            'min_lambda': float(np.min(lambdas)),
            'max_lambda': float(np.max(lambdas)),
            'n_negative': int(n_negative),
            'n_positive': int(n_positive),
            'n_neutral': int(n_neutral),
            'mean_energy_error': float(mean_energy_error),
            'max_energy_error': float(max_energy_error),
        },
        'seed_results': results
    }

    output_path = '/home/user/Testing-env/data/results/benettin_proper_units.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {output_path}")
    print()

if __name__ == '__main__':
    main()
