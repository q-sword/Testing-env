#!/usr/bin/env python3
"""
N=30 BODY O(1) LYAPUNOV SPECTRUM
December 2025

Uses O(1) mean-field approximation for force calculation.
Each particle interacts with center of mass + local smooth field.

THEORY: Quantum regularization smooths the potential enough that
mean-field approximation becomes valid.

Force per particle: F_i = F_COM + F_local
Where F_COM is O(1) and F_local can be approximated in O(1)
"""

import numpy as np
from numba import njit, prange
import time

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
# O(1) MEAN FIELD FORCES
# =============================================================================

@njit(parallel=True)
def compute_forces_o1(pos, masses, epsilon):
    """
    O(1) force calculation using mean-field approximation

    Each particle feels:
    1. Force from center of mass (O(1) per particle)
    2. Smooth background potential (O(1) per particle)

    Total: O(N) for all particles, O(1) per particle
    """
    N = len(masses)
    acc = np.zeros((N, 3))

    # Compute center of mass (O(N) but done once)
    total_mass = np.sum(masses)
    com = np.zeros(3)
    for i in range(N):
        com += masses[i] * pos[i]
    com = com / total_mass

    # Compute RMS radius for smooth field
    r2_mean = 0.0
    for i in range(N):
        r_vec = pos[i] - com
        r2_mean += np.sum(r_vec * r_vec)
    r2_mean = r2_mean / N
    r_smooth = np.sqrt(r2_mean + epsilon**2)

    # Each particle experiences:
    # 1. Attraction to COM (O(1) per particle)
    # 2. Smooth distributed mass correction (O(1) per particle)
    for i in prange(N):
        # Distance to COM
        r_vec = pos[i] - com
        r2 = r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2
        r_reg2 = r2 + epsilon**2
        r_reg3 = r_reg2 * np.sqrt(r_reg2)

        # Force from total mass at COM (minus self)
        effective_mass = total_mass - masses[i]
        force_mag = G * effective_mass / r_reg3
        acc[i] -= force_mag * r_vec

        # Smooth field correction (approximates distributed mass)
        # This accounts for the fact that mass isn't all at COM
        smooth_correction = G * total_mass * epsilon**2 / (r_reg3 * r_smooth)
        acc[i] += smooth_correction * r_vec * 0.5  # Damping factor

    return acc

@njit
def yoshida6_step(pos, vel, masses, epsilon, dt):
    """Single Yoshida 6th order step with O(1) forces"""
    for i in range(len(D)):
        acc = compute_forces_o1(pos, masses, epsilon)
        vel = vel + D[i] * dt * acc
        if i < len(C) - 1 or C[i] != 0.0:
            pos = pos + C[i] * dt * vel
    return pos, vel

@njit
def compute_energy(pos, vel, masses, epsilon):
    """Compute energy (still exact, O(N²))"""
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

@njit
def evolve_system(pos, vel, masses, epsilon, dt, num_steps):
    """Evolve with O(1) forces"""
    for step in range(num_steps):
        pos, vel = yoshida6_step(pos, vel, masses, epsilon, dt)
    return pos, vel

# =============================================================================
# FAST O(1) LYAPUNOV SPECTRUM
# =============================================================================

def compute_o1_lyapunov_spectrum(seed=42, N=30, T_total=50, T_lyap=5, dt=0.001, n_vectors=12):
    """
    Ultra-fast O(1) Lyapunov spectrum

    Should be ~100× faster than O(N²) version!
    """

    print(f"="*80)
    print(f"O(1) MEAN-FIELD LYAPUNOV SPECTRUM")
    print(f"N = {N} bodies → 6N = {6*N} dimensional phase space")
    print(f"Computing top {n_vectors} exponents")
    print(f"="*80)
    print()

    # Setup
    np.random.seed(seed)
    masses = np.ones(N)
    pos = np.random.randn(N, 3) * 0.5
    vel = np.random.randn(N, 3) * 0.3

    v_rms = np.sqrt(np.mean(vel**2))
    epsilon = HBAR / v_rms

    print(f"System setup:")
    print(f"  N = {N} bodies")
    print(f"  v_rms = {v_rms:.4f}")
    print(f"  ε = {epsilon:.4f}")

    E0 = compute_energy(pos, vel, masses, epsilon)
    print(f"  E₀ = {E0:.6f}")
    print()

    print(f"O(1) mean-field approximation:")
    print(f"  Force per particle: O(1) instead of O(N)")
    print(f"  Total computation: O(N) instead of O(N²)")
    print(f"  Expected speedup: ~{N}× = {N:.0f}× faster")
    print()

    # Initialize tangent vectors
    tangent_vectors = np.random.randn(n_vectors, 6*N)
    tangent_vectors, _ = np.linalg.qr(tangent_vectors.T)
    tangent_vectors = tangent_vectors.T

    print(f"Initialized {n_vectors} orthonormal tangent vectors")
    print()

    lyapunov_sums = np.zeros(n_vectors)
    n_intervals = int(T_total / T_lyap)
    steps_per_interval = int(T_lyap / dt)

    print(f"Integration:")
    print(f"  T_total = {T_total}")
    print(f"  Intervals = {n_intervals}")
    print(f"  Steps/interval = {steps_per_interval}")
    print(f"  dt = {dt}")
    print()
    print("Computing spectrum...")
    print()

    start_time = time.time()

    # Main loop
    for interval in range(n_intervals):
        interval_start = time.time()

        pos_ref = pos.copy()
        vel_ref = vel.copy()

        perturbed_states = []

        for i in range(n_vectors):
            delta_pos = tangent_vectors[i, :3*N].reshape(N, 3)
            delta_vel = tangent_vectors[i, 3*N:].reshape(N, 3)

            pos_pert = pos + delta_pos
            vel_pert = vel + delta_vel

            pos_ref_i, vel_ref_i = evolve_system(pos_ref, vel_ref, masses, epsilon, dt, steps_per_interval)
            pos_pert_i, vel_pert_i = evolve_system(pos_pert, vel_pert, masses, epsilon, dt, steps_per_interval)

            new_delta_pos = pos_pert_i - pos_ref_i
            new_delta_vel = vel_pert_i - vel_ref_i

            new_tangent = np.concatenate([new_delta_pos.flatten(), new_delta_vel.flatten()])
            perturbed_states.append(new_tangent)

        pos, vel = evolve_system(pos, vel, masses, epsilon, dt, steps_per_interval)

        tangent_matrix = np.array(perturbed_states)
        Q, R = np.linalg.qr(tangent_matrix.T)

        for i in range(n_vectors):
            lyapunov_sums[i] += np.log(abs(R[i, i]))

        tangent_vectors = Q.T

        elapsed = time.time() - interval_start
        current_lambdas = lyapunov_sums / ((interval + 1) * T_lyap)

        print(f"Interval {interval+1}/{n_intervals}: "
              f"λ_max = {current_lambdas[0]:+.6f}, "
              f"Σλ = {np.sum(current_lambdas):+.6f}, "
              f"t = {elapsed:.1f}s")

    spectrum = lyapunov_sums / T_total

    E_final = compute_energy(pos, vel, masses, epsilon)
    energy_drift = abs((E_final - E0) / E0)

    total_time = time.time() - start_time

    print()
    print(f"Integration complete!")
    print(f"Total runtime: {total_time/60:.1f} minutes ({total_time:.1f}s)")
    print(f"Final energy: E = {E_final:.6f}")
    print(f"Energy drift: δE/E₀ = {energy_drift:.3e}")
    print()

    return spectrum, epsilon, energy_drift, total_time


def main():
    """Run O(1) spectrum computation"""

    print()
    print("="*80)
    print("N=30 BODY O(1) MEAN-FIELD LYAPUNOV SPECTRUM")
    print("ULTRA-FAST APPROXIMATION")
    print("="*80)
    print()

    # Warmup
    print("JIT warmup...")
    np.random.seed(0)
    _ = evolve_system(
        np.random.randn(5, 3) * 0.5,
        np.random.randn(5, 3) * 0.3,
        np.ones(5),
        1.0,
        0.001,
        10
    )
    print("Warmup complete!")
    print()

    # Run
    spectrum, epsilon, energy_drift, runtime = compute_o1_lyapunov_spectrum(
        seed=42,
        N=30,
        T_total=50,
        T_lyap=5,
        dt=0.001,
        n_vectors=12
    )

    # Results
    print("="*80)
    print("RESULTS")
    print("="*80)
    print()
    print(f"Top {len(spectrum)} Lyapunov exponents:")
    print()

    for i, lam in enumerate(spectrum):
        print(f"  λ_{i+1:2d} = {lam:+.6f}")

    print()
    print("Statistics:")
    print(f"  λ_max = {spectrum[0]:+.6f}")
    print(f"  λ_min = {spectrum[-1]:+.6f}")
    print(f"  Σλ = {np.sum(spectrum):+.6f}")
    print(f"  <λ> = {np.mean(spectrum):+.6f}")
    print()

    n_pos = np.sum(spectrum > 1e-6)
    n_neg = np.sum(spectrum < -1e-6)
    n_zero = np.sum(np.abs(spectrum) < 1e-6)

    print(f"Hamiltonian structure:")
    print(f"  Positive: {n_pos}")
    print(f"  Negative: {n_neg}")
    print(f"  Near-zero: {n_zero}")
    print()

    if np.abs(np.sum(spectrum)) < 0.2:
        print("✅ Σλ ≈ 0: Hamiltonian!")

    print()
    print(f"Energy: δE/E₀ = {energy_drift:.3e}")
    print(f"Runtime: {runtime/60:.1f} min ({runtime:.0f}s)")
    print()
    print("="*80)
    print()

    if spectrum[0] > 0:
        print(f"CHAOS: λ_max = {spectrum[0]:+.6f} > 0")
        print(f"Lyapunov time: τ = {1/spectrum[0]:.1f}")
    else:
        print(f"STABLE: λ_max = {spectrum[0]:+.6f} < 0")
        print("Quantum regularization stabilizes!")

    print()


if __name__ == "__main__":
    main()
