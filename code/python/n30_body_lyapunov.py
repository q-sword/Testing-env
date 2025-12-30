#!/usr/bin/env python3
"""
N=30 BODY QUANTUM REGULARIZATION TEST
December 2025

Test if quantum regularization stabilizes a 30-body system.
Computes Lyapunov exponent for N=30 gravitational bodies.

WARNING: This is computationally intensive!
- O(N²) force calculations
- Long integration times
- Expect ~30-60 minutes runtime
"""

import numpy as np
from numba import njit, prange
import time

# =============================================================================
# CONSTANTS
# =============================================================================

G = 1.0
HBAR = 1.0

# Yoshida 6th order coefficients
w1 = 0.78451361047755726382
w2 = 0.23557321335935813368
w3 = -1.17767998417887100695
w0 = 1.0 - 2.0*(w1 + w2 + w3)

C = np.array([w3, w2, w1, w0, w1, w2, w3, 0.0])
D = np.array([w3/2, (w3+w2)/2, (w2+w1)/2, (w1+w0)/2,
              (w0+w1)/2, (w1+w2)/2, (w2+w3)/2, w3/2])

# =============================================================================
# FORCE COMPUTATION (NUMBA ACCELERATED)
# =============================================================================

@njit(parallel=True)
def compute_forces_parallel(pos, masses, epsilon):
    """Compute gravitational forces with quantum regularization"""
    N = len(masses)
    acc = np.zeros((N, 3))

    for i in prange(N):
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
    """Single Yoshida 6th order step"""
    for i in range(len(D)):
        acc = compute_forces_parallel(pos, masses, epsilon)
        vel = vel + D[i] * dt * acc
        if i < len(C) - 1 or C[i] != 0.0:
            pos = pos + C[i] * dt * vel
    return pos, vel

@njit
def compute_energy(pos, vel, masses, epsilon):
    """Compute total energy"""
    N = len(masses)
    KE = 0.5 * np.sum(masses.reshape(-1, 1) * (vel * vel))

    PE = 0.0
    for i in range(N):
        for j in range(i+1, N):
            r_vec = pos[j] - pos[i]
            r2 = r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2
            r_reg = np.sqrt(r2 + epsilon**2)
            PE -= G * masses[i] * masses[j] / r_reg

    return KE + PE

# =============================================================================
# LYAPUNOV EXPONENT CALCULATION
# =============================================================================

@njit
def evolve_system(pos, vel, masses, epsilon, dt, num_steps):
    """Evolve system for num_steps"""
    for step in range(num_steps):
        pos, vel = yoshida6_step(pos, vel, masses, epsilon, dt)
    return pos, vel

def compute_lyapunov_n30(seed=42, N=30, T_total=100, T_lyap=10, dt=0.0005):
    """
    Compute largest Lyapunov exponent for N=30 body system

    Parameters:
        seed: Random seed for initial conditions
        N: Number of bodies (30)
        T_total: Total integration time
        T_lyap: Renormalization interval
        dt: Timestep (larger than 3-body due to computational cost)

    Returns:
        lambda_exp: Largest Lyapunov exponent
        epsilon: Quantum regularization parameter
        energy_drift: Relative energy error
    """

    print(f"Setting up N={N} body system (seed={seed})...")

    # Generate initial conditions
    np.random.seed(seed)
    masses = np.ones(N)
    pos = np.random.randn(N, 3) * 0.5
    vel = np.random.randn(N, 3) * 0.3

    # Compute epsilon
    v_rms = np.sqrt(np.mean(vel**2))
    epsilon = HBAR / v_rms

    print(f"  v_rms = {v_rms:.4f}")
    print(f"  ε = {epsilon:.4f}")
    print()

    # Initial energy
    E0 = compute_energy(pos, vel, masses, epsilon)
    print(f"Initial energy: E₀ = {E0:.6f}")
    print()

    # Initialize perturbation (small displacement in configuration space)
    delta_pos = 1e-10 * np.random.randn(N, 3)
    delta_vel = np.zeros((N, 3))  # Perturbation in positions only

    log_stretch = 0.0
    n_intervals = int(T_total / T_lyap)

    print(f"Computing Lyapunov exponent over T={T_total}...")
    print(f"  Intervals: {n_intervals}")
    print(f"  Renormalization every {T_lyap} time units")
    print()

    pos_ref = pos.copy()
    vel_ref = vel.copy()

    steps_per_interval = int(T_lyap / dt)

    for interval in range(n_intervals):
        print(f"Interval {interval+1}/{n_intervals}...", end=' ', flush=True)

        interval_start = time.time()

        # Create perturbed system
        pos_pert = pos_ref + delta_pos
        vel_pert = vel_ref + delta_vel

        # Evolve both systems
        pos_ref, vel_ref = evolve_system(pos_ref, vel_ref, masses, epsilon, dt, steps_per_interval)
        pos_pert, vel_pert = evolve_system(pos_pert, vel_pert, masses, epsilon, dt, steps_per_interval)

        # Compute separation
        delta_pos_new = pos_pert - pos_ref
        delta_vel_new = vel_pert - vel_ref

        # Use configuration space norm (positions only)
        delta_norm = np.linalg.norm(delta_pos_new)

        # Accumulate log stretch
        log_stretch += np.log(delta_norm / 1e-10)

        # Renormalize perturbation
        delta_pos = (delta_pos_new / delta_norm) * 1e-10
        delta_vel = delta_vel_new / delta_norm * 1e-10  # Keep velocity perturbation small

        interval_time = time.time() - interval_start
        print(f"δ = {delta_norm:.2e}, t = {interval_time:.1f}s")

    # Compute Lyapunov exponent
    lambda_exp = log_stretch / T_total

    # Check energy conservation
    E_final = compute_energy(pos_ref, vel_ref, masses, epsilon)
    energy_drift = abs((E_final - E0) / E0)

    print()
    print(f"Final energy: E = {E_final:.6f}")
    print(f"Energy drift: δE/E₀ = {energy_drift:.3e}")
    print()

    return lambda_exp, epsilon, energy_drift


def main():
    """Run N=30 body Lyapunov test"""

    print("="*80)
    print("N=30 BODY QUANTUM REGULARIZATION TEST")
    print("December 2025")
    print("="*80)
    print()
    print("This will test if quantum regularization stabilizes a 30-body system.")
    print("Expected runtime: 30-60 minutes")
    print()
    print("="*80)
    print()

    # JIT warmup
    print("Warming up JIT compiler...")
    np.random.seed(0)
    masses_warm = np.ones(5)
    pos_warm = np.random.randn(5, 3) * 0.5
    vel_warm = np.random.randn(5, 3) * 0.3
    v_rms_warm = np.sqrt(np.mean(vel_warm**2))
    eps_warm = HBAR / v_rms_warm
    _ = evolve_system(pos_warm, vel_warm, masses_warm, eps_warm, 0.0005, 10)
    print("Warmup complete!")
    print()
    print("="*80)
    print()

    # Run main test
    start_time = time.time()

    lambda_exp, epsilon, energy_drift = compute_lyapunov_n30(
        seed=42,
        N=30,
        T_total=100,   # Integration time
        T_lyap=10,     # Renormalization interval
        dt=0.0005      # Larger timestep for computational efficiency
    )

    elapsed = time.time() - start_time

    # Display results
    print("="*80)
    print("RESULTS")
    print("="*80)
    print()
    print(f"Number of bodies: N = 30")
    print(f"Integration time: T = 100")
    print(f"Timestep: dt = 0.0005")
    print(f"Quantum parameter: ε = {epsilon:.4f}")
    print()
    print(f"Largest Lyapunov exponent: λ = {lambda_exp:+.6f}")
    print(f"Energy conservation: δE/E₀ = {energy_drift:.3e}")
    print()

    if lambda_exp < 0:
        print("✅ STABLE: Negative Lyapunov exponent!")
        print("   Quantum regularization stabilizes N=30 body system")
    elif lambda_exp < 0.01:
        print("⚠️  MARGINAL: Near-zero Lyapunov exponent")
        print("   System may be weakly stable or neutral")
    else:
        print("❌ UNSTABLE: Positive Lyapunov exponent")
        print("   System is chaotic")

    print()
    print(f"Runtime: {elapsed/60:.1f} minutes")
    print()
    print("="*80)
    print()

    # Additional tests with different seeds
    print("Would you like to test multiple seeds? (This would take hours)")
    print("For now, single seed test complete.")
    print()


if __name__ == "__main__":
    main()
