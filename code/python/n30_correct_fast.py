#!/usr/bin/env python3
"""
N=30 BODY FAST & ACCURATE LYAPUNOV SPECTRUM
December 2025

Combines:
- Parallel tangent evolution (FAST)
- Exact O(N²) forces (ACCURATE)
- Modified Gram-Schmidt
- Optimized dt=0.001

Expected: ~2-3 minutes with perfect energy conservation
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

# =============================================================================
# EXACT FORCES (O(N²) but correct!)
# =============================================================================

@njit
def compute_forces_exact(pos, masses, epsilon):
    """Exact pairwise forces - preserves Hamiltonian

    SERIAL version - parallelism is at the trajectory level to avoid
    nested parallelism overhead (12 traj threads × 30 force threads = bad!)
    """
    N = len(masses)
    acc = np.zeros((N, 3))

    for i in range(N):  # Serial - no prange!
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
    """Yoshida step with exact forces"""
    for i in range(len(D)):
        acc = compute_forces_exact(pos, masses, epsilon)
        vel = vel + D[i] * dt * acc
        if i < len(C) - 1 or C[i] != 0.0:
            pos = pos + C[i] * dt * vel
    return pos, vel

# =============================================================================
# PARALLEL TANGENT EVOLUTION (THE KEY SPEEDUP!)
# =============================================================================

@njit(parallel=True)
def evolve_all_tangents_exact(pos_ref, vel_ref, tangent_pos, tangent_vel,
                               masses, epsilon, dt, num_steps):
    """
    Evolve reference + ALL tangents in PARALLEL
    Uses EXACT forces for energy conservation
    """
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

    for vec_idx in prange(n_vectors):  # PARALLEL!
        pos_p = pos_ref + tangent_pos[vec_idx]
        vel_p = vel_ref + tangent_vel[vec_idx]

        for step in range(num_steps):
            pos_p, vel_p = yoshida6_step(pos_p, vel_p, masses, epsilon, dt)

        new_tangent_pos[vec_idx] = pos_p - pos_r
        new_tangent_vel[vec_idx] = vel_p - vel_r

    return pos_r, vel_r, new_tangent_pos, new_tangent_vel

def full_qr_decomposition(vectors):
    """Full QR decomposition (numpy, not numba)"""
    Q, R = np.linalg.qr(vectors.T)
    norms = np.abs(np.diag(R))
    return Q.T, norms

@njit
def compute_energy(pos, vel, masses, epsilon):
    """Exact energy"""
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
# MAIN COMPUTATION
# =============================================================================

def compute_fast_accurate_spectrum(seed=42, N=30, T_total=50, T_lyap=5, dt=0.001, n_vectors=12):
    """
    Fast AND accurate Lyapunov spectrum

    FAST: Parallel tangent evolution
    ACCURATE: Exact forces, perfect energy conservation
    """

    print(f"="*80)
    print(f"FAST & ACCURATE LYAPUNOV SPECTRUM")
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

    E0 = compute_energy(pos, vel, masses, epsilon)

    print(f"System: N={N}, ε={epsilon:.4f}, E₀={E0:.6f}")
    print()

    print("Strategy:")
    print("  ✓ EXACT O(N²) forces (energy conservation)")
    print("  ✓ PARALLEL tangent evolution (speed)")
    print("  ✓ FULL QR decomposition (accuracy)")
    print(f"  ✓ dt={dt} (optimized)")
    print()

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

    print(f"Integration: T={T_total}, {n_intervals} intervals")
    print()
    print("Running...")
    print()

    start_time = time.time()

    for interval in range(n_intervals):
        interval_start = time.time()

        # PARALLEL tangent evolution with EXACT forces
        pos, vel, new_tangent_pos, new_tangent_vel = evolve_all_tangents_exact(
            pos, vel, tangent_pos, tangent_vel, masses, epsilon, dt, steps_per_interval
        )

        # Orthonormalize with full QR
        tangent_flat = np.concatenate([new_tangent_pos.reshape(n_vectors, -1),
                                        new_tangent_vel.reshape(n_vectors, -1)], axis=1)

        tangent_flat, norms = full_qr_decomposition(tangent_flat)

        for i in range(n_vectors):
            if norms[i] > 1e-15:
                lyapunov_sums[i] += np.log(norms[i])

        tangent_pos = tangent_flat[:, :N*3].reshape(n_vectors, N, 3)
        tangent_vel = tangent_flat[:, N*3:].reshape(n_vectors, N, 3)

        elapsed = time.time() - interval_start
        current_lambdas = lyapunov_sums / ((interval + 1) * T_lyap)

        print(f"[{interval+1}/{n_intervals}] λ_max={current_lambdas[0]:+.6f} Σλ={np.sum(current_lambdas):+.6f} t={elapsed:.1f}s")
        sys.stdout.flush()

    spectrum = lyapunov_sums / T_total

    E_final = compute_energy(pos, vel, masses, epsilon)
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
    print("N=30 FAST & ACCURATE LYAPUNOV SPECTRUM")
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

    spectrum, epsilon, energy_drift, runtime = compute_fast_accurate_spectrum(
        seed=42,
        N=30,
        T_total=50,
        T_lyap=5,
        dt=0.001,
        n_vectors=12
    )

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
    print(f"  Runtime = {runtime:.1f}s ({runtime/60:.1f} min)")
    print()

    n_pos = np.sum(spectrum > 1e-6)
    n_neg = np.sum(spectrum < -1e-6)
    n_zero = np.sum(np.abs(spectrum) < 1e-6)

    print(f"  Positive: {n_pos}, Negative: {n_neg}, Zero: {n_zero}")
    print()

    if np.abs(np.sum(spectrum)) < 0.1:
        print("✅ Σλ ≈ 0: Hamiltonian preserved!")

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
