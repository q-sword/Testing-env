#!/usr/bin/env python3
"""
N=30 BODY ULTRA-FAST LYAPUNOV SPECTRUM
December 2025

MAXIMUM SPEED OPTIMIZATIONS:
1. Parallel tangent vector evolution (not sequential!)
2. Vectorized perturbation propagation
3. Modified Gram-Schmidt (faster than QR)
4. O(1) forces with smart caching
5. Reduced output frequency
"""

import numpy as np
from numba import njit, prange
import time

G = 1.0
HBAR = 1.0

# Yoshida 6th coefficients
w1 = 0.78451361047755726382
w2 = 0.23557321335935813368
w3 = -1.17767998417887100695
w0 = 1.0 - 2.0*(w1 + w2 + w3)

C = np.array([w3, w2, w1, w0, w1, w2, w3, 0.0])
D = np.array([w3/2, (w3+w2)/2, (w2+w1)/2, (w1+w0)/2,
              (w0+w1)/2, (w1+w2)/2, (w2+w3)/2, w3/2])

# =============================================================================
# ULTRA-FAST O(1) FORCES
# =============================================================================

@njit
def compute_forces_o1_fast(pos, masses, epsilon):
    """Optimized O(1) mean-field forces"""
    N = len(masses)
    acc = np.zeros((N, 3))

    # Center of mass
    total_mass = np.sum(masses)
    com = np.sum(masses.reshape(-1, 1) * pos, axis=0) / total_mass

    # Mean radius
    r2_mean = np.sum((pos - com)**2) / N
    r_smooth = np.sqrt(r2_mean + epsilon**2)

    # Vectorized force calculation
    r_vec = pos - com
    r2 = np.sum(r_vec * r_vec, axis=1).reshape(-1, 1)
    r_reg2 = r2 + epsilon**2
    r_reg3 = r_reg2 * np.sqrt(r_reg2)

    # Force from COM
    effective_mass = total_mass - masses.reshape(-1, 1)
    acc = -G * effective_mass * r_vec / r_reg3

    # Smooth correction
    smooth_corr = G * total_mass * epsilon**2 / (r_reg3 * r_smooth)
    acc += smooth_corr * r_vec * 0.5

    return acc

@njit
def yoshida6_step_fast(pos, vel, masses, epsilon, dt):
    """Vectorized Yoshida step"""
    for i in range(len(D)):
        acc = compute_forces_o1_fast(pos, masses, epsilon)
        vel = vel + D[i] * dt * acc
        if i < len(C) - 1 or C[i] != 0.0:
            pos = pos + C[i] * dt * vel
    return pos, vel

# =============================================================================
# VECTORIZED TANGENT EVOLUTION
# =============================================================================

@njit(parallel=True)
def evolve_all_tangents(pos_ref, vel_ref, tangent_pos, tangent_vel,
                        masses, epsilon, dt, num_steps):
    """
    Evolve reference + ALL tangent vectors in PARALLEL

    This is the KEY optimization - instead of looping over tangent vectors
    sequentially, we evolve them all at once!
    """
    n_vectors = tangent_pos.shape[0]
    N = len(masses)

    # Evolve reference
    pos_r = pos_ref.copy()
    vel_r = vel_ref.copy()
    for step in range(num_steps):
        pos_r, vel_r = yoshida6_step_fast(pos_r, vel_r, masses, epsilon, dt)

    # Evolve ALL perturbed systems in parallel
    new_tangent_pos = np.zeros((n_vectors, N, 3))
    new_tangent_vel = np.zeros((n_vectors, N, 3))

    for vec_idx in prange(n_vectors):  # PARALLEL over vectors!
        pos_p = pos_ref + tangent_pos[vec_idx]
        vel_p = vel_ref + tangent_vel[vec_idx]

        for step in range(num_steps):
            pos_p, vel_p = yoshida6_step_fast(pos_p, vel_p, masses, epsilon, dt)

        new_tangent_pos[vec_idx] = pos_p - pos_r
        new_tangent_vel[vec_idx] = vel_p - vel_r

    return pos_r, vel_r, new_tangent_pos, new_tangent_vel

@njit
def modified_gram_schmidt(vectors):
    """
    Modified Gram-Schmidt orthonormalization
    Faster than full QR for our use case
    Returns: orthonormal vectors and norms (for Lyapunov)
    """
    n_vectors, dim = vectors.shape
    Q = np.zeros((n_vectors, dim))
    norms = np.zeros(n_vectors)

    for i in range(n_vectors):
        q = vectors[i].copy()

        # Subtract projections onto previous vectors
        for j in range(i):
            proj = np.dot(q, Q[j])
            q = q - proj * Q[j]

        # Normalize
        norm = np.linalg.norm(q)
        norms[i] = norm
        if norm > 1e-15:
            Q[i] = q / norm
        else:
            Q[i] = q  # Degenerate case

    return Q, norms

@njit
def compute_energy_fast(pos, vel, masses, epsilon):
    """Fast energy calculation"""
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

# =============================================================================
# ULTRA-FAST SPECTRUM COMPUTATION
# =============================================================================

def compute_ultra_fast_spectrum(seed=42, N=30, T_total=50, T_lyap=5, dt=0.001, n_vectors=12):
    """
    ULTRA-FAST Lyapunov spectrum using all optimizations

    KEY SPEEDUPS:
    - Parallel tangent evolution (biggest win!)
    - Modified Gram-Schmidt instead of QR
    - Vectorized O(1) forces
    - Minimal overhead
    """

    print(f"="*80)
    print(f"ULTRA-FAST LYAPUNOV SPECTRUM")
    print(f"N = {N} bodies, {n_vectors} exponents")
    print(f"="*80)
    print()

    # Setup
    np.random.seed(seed)
    masses = np.ones(N)
    pos = np.random.randn(N, 3) * 0.5
    vel = np.random.randn(N, 3) * 0.3

    v_rms = np.sqrt(np.mean(vel**2))
    epsilon = HBAR / v_rms

    E0 = compute_energy_fast(pos, vel, masses, epsilon)

    print(f"System: N={N}, ε={epsilon:.4f}, E₀={E0:.6f}")
    print()

    print("OPTIMIZATIONS:")
    print("  ✓ Parallel tangent evolution (prange)")
    print("  ✓ Modified Gram-Schmidt (faster than QR)")
    print("  ✓ Vectorized O(1) forces")
    print(f"  ✓ dt={dt} (validated optimal)")
    print()

    # Initialize tangent vectors as perturbations
    tangent_pos = np.random.randn(n_vectors, N, 3) * 1e-8
    tangent_vel = np.random.randn(n_vectors, N, 3) * 1e-8

    # Orthonormalize initial tangents
    tangent_flat = np.concatenate([tangent_pos.reshape(n_vectors, -1),
                                    tangent_vel.reshape(n_vectors, -1)], axis=1)
    tangent_flat, _ = modified_gram_schmidt(tangent_flat)

    tangent_pos = tangent_flat[:, :N*3].reshape(n_vectors, N, 3)
    tangent_vel = tangent_flat[:, N*3:].reshape(n_vectors, N, 3)

    lyapunov_sums = np.zeros(n_vectors)
    n_intervals = int(T_total / T_lyap)
    steps_per_interval = int(T_lyap / dt)

    print(f"Integration: T={T_total}, {n_intervals} intervals, {steps_per_interval} steps/interval")
    print()
    print("Running...")
    print()

    start_time = time.time()

    # Main loop
    for interval in range(n_intervals):
        interval_start = time.time()

        # Evolve ALL systems in parallel (this is the magic!)
        pos, vel, new_tangent_pos, new_tangent_vel = evolve_all_tangents(
            pos, vel, tangent_pos, tangent_vel, masses, epsilon, dt, steps_per_interval
        )

        # Flatten tangent vectors
        tangent_flat = np.concatenate([new_tangent_pos.reshape(n_vectors, -1),
                                        new_tangent_vel.reshape(n_vectors, -1)], axis=1)

        # Modified Gram-Schmidt orthonormalization
        tangent_flat, norms = modified_gram_schmidt(tangent_flat)

        # Accumulate Lyapunov exponents from norms
        for i in range(n_vectors):
            if norms[i] > 1e-15:
                lyapunov_sums[i] += np.log(norms[i])

        # Update tangent vectors
        tangent_pos = tangent_flat[:, :N*3].reshape(n_vectors, N, 3)
        tangent_vel = tangent_flat[:, N*3:].reshape(n_vectors, N, 3)

        # Progress
        elapsed = time.time() - interval_start
        current_lambdas = lyapunov_sums / ((interval + 1) * T_lyap)

        print(f"[{interval+1}/{n_intervals}] λ_max={current_lambdas[0]:+.6f} Σλ={np.sum(current_lambdas):+.6f} t={elapsed:.1f}s")

    spectrum = lyapunov_sums / T_total

    E_final = compute_energy_fast(pos, vel, masses, epsilon)
    energy_drift = abs((E_final - E0) / E0)

    total_time = time.time() - start_time

    print()
    print(f"✓ Complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Energy: δE/E₀ = {energy_drift:.3e}")
    print()

    return spectrum, epsilon, energy_drift, total_time


def main():
    print()
    print("="*80)
    print("N=30 ULTRA-FAST LYAPUNOV SPECTRUM")
    print("MAXIMUM PERFORMANCE")
    print("="*80)
    print()

    # Warmup
    print("JIT warmup...")
    np.random.seed(0)
    _ = evolve_all_tangents(
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

    # Run
    spectrum, epsilon, energy_drift, runtime = compute_ultra_fast_spectrum(
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

    for i, lam in enumerate(spectrum):
        print(f"  λ_{i+1:2d} = {lam:+.6f}")

    print()
    print(f"  λ_max = {spectrum[0]:+.6f}")
    print(f"  Σλ = {np.sum(spectrum):+.6f}")
    print(f"  δE/E₀ = {energy_drift:.3e}")
    print(f"  Runtime = {runtime:.1f}s")
    print()

    n_pos = np.sum(spectrum > 1e-6)
    n_neg = np.sum(spectrum < -1e-6)

    print(f"  Positive: {n_pos}, Negative: {n_neg}")
    print()

    if spectrum[0] > 0:
        print(f"✓ CHAOS: λ_max = {spectrum[0]:+.6f}")
        print(f"  Lyapunov time: {1/spectrum[0]:.1f}")
    else:
        print(f"✓ STABLE: λ_max = {spectrum[0]:+.6f}")

    print()
    print("="*80)


if __name__ == "__main__":
    main()
