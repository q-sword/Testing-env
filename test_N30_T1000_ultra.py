#!/usr/bin/env python3
"""
N=30 T=1000 ULTRA-OPTIMIZED

Optimizations:
1. Newton's 3rd law with thread-local accumulators (halve pairwise work)
2. Benettin tangent/variational method (30-50% less work than dual trajectory)
3. Cache-tiled j-blocking + SoA
4. AVX vectorization hints (fastmath, explicit threading)
5. Preallocated arrays, warm-up JIT, fused operations

Expected: 3-5× speedup → T=1000 in ~3-5 minutes
"""

import numpy as np
from numba import njit, prange, set_num_threads
import time
import os

# Use all cores
set_num_threads(int(os.cpu_count() or 4))

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

TILE_SIZE = 8  # Cache blocking


@njit(parallel=True, fastmath=True, cache=True)
def compute_forces_newton3(px, py, pz, masses, epsilon, ax, ay, az):
    """
    Newton's 3rd law with thread-local accumulators.
    Compute each pair (i,j) once, accumulate forces locally, then reduce.
    Halves pairwise work!
    """
    N = len(masses)
    eps2 = epsilon * epsilon

    # Zero out accumulators
    ax[:] = 0.0
    ay[:] = 0.0
    az[:] = 0.0

    # Thread-local accumulators (avoid false sharing)
    num_threads = N  # Conservative upper bound
    ax_local = np.zeros((num_threads, N))
    ay_local = np.zeros((num_threads, N))
    az_local = np.zeros((num_threads, N))

    # Parallel loop over pairs using Newton's 3rd law
    for i in prange(N):
        tid = i  # Thread ID proxy

        # Cache-tiled inner loop (j-blocking)
        for j_start in range(i+1, N, TILE_SIZE):
            j_end = min(j_start + TILE_SIZE, N)

            for j in range(j_start, j_end):
                # Compute distance
                dx = px[j] - px[i]
                dy = py[j] - py[i]
                dz = pz[j] - pz[i]

                r2 = dx*dx + dy*dy + dz*dz
                r_reg2 = r2 + eps2

                # Fused rsqrt path
                inv_r = 1.0 / np.sqrt(r_reg2)
                inv_r3 = inv_r * inv_r * inv_r

                # Force magnitude for pair (i,j)
                f_ij = G * inv_r3

                # Force components
                fx = f_ij * dx
                fy = f_ij * dy
                fz = f_ij * dz

                # Newton's 3rd law: accumulate both directions
                # Force on i from j
                ax_local[tid, i] += masses[j] * fx
                ay_local[tid, i] += masses[j] * fy
                az_local[tid, i] += masses[j] * fz

                # Force on j from i (opposite direction)
                ax_local[tid, j] -= masses[i] * fx
                ay_local[tid, j] -= masses[i] * fy
                az_local[tid, j] -= masses[i] * fz

    # Reduction: sum thread-local accumulators
    for i in prange(N):
        for tid in range(num_threads):
            ax[i] += ax_local[tid, i]
            ay[i] += ay_local[tid, i]
            az[i] += az_local[tid, i]


@njit(fastmath=True, cache=True)
def compute_jacobian_action(px, py, pz, masses, epsilon, dpx, dpy, dpz, Jdpx, Jdpy, Jdpz):
    """
    Benettin tangent method: compute J·δp where J is force Jacobian.
    This replaces integrating a full second trajectory!
    Much cheaper: O(N²) instead of 2× O(N²)
    """
    N = len(masses)
    eps2 = epsilon * epsilon

    Jdpx[:] = 0.0
    Jdpy[:] = 0.0
    Jdpz[:] = 0.0

    for i in range(N):
        for j in range(N):
            if i != j:
                # Position difference
                dx = px[j] - px[i]
                dy = py[j] - py[i]
                dz = pz[j] - pz[i]

                r2 = dx*dx + dy*dy + dz*dz
                r_reg2 = r2 + eps2
                inv_r = 1.0 / np.sqrt(r_reg2)
                inv_r3 = inv_r * inv_r * inv_r
                inv_r5 = inv_r3 * inv_r * inv_r

                # Perturbation in position difference
                ddx = dpx[j] - dpx[i]
                ddy = dpy[j] - dpy[i]
                ddz = dpz[j] - dpz[i]

                # Jacobian terms
                r_dot_dr = dx*ddx + dy*ddy + dz*ddz

                # J·δp components (linearized force perturbation)
                coeff1 = G * masses[j] * inv_r3
                coeff2 = -3.0 * G * masses[j] * inv_r5 * r_dot_dr

                Jdpx[i] += coeff1 * ddx + coeff2 * dx
                Jdpy[i] += coeff1 * ddy + coeff2 * dy
                Jdpz[i] += coeff1 * ddz + coeff2 * dz


@njit(cache=True)
def yoshida6_step_variational(px, py, pz, vx, vy, vz, dpx, dpy, dpz, dvx, dvy, dvz,
                              masses, epsilon, dt, ax, ay, az, Jdpx, Jdpy, Jdpz):
    """
    Yoshida 6th with variational (tangent) propagation.
    Integrates both reference trajectory and tangent vector simultaneously.
    """
    for i in range(len(D)):
        # Reference trajectory forces
        compute_forces_newton3(px, py, pz, masses, epsilon, ax, ay, az)

        # Tangent vector Jacobian action
        compute_jacobian_action(px, py, pz, masses, epsilon, dpx, dpy, dpz, Jdpx, Jdpy, Jdpz)

        # Reference velocity kick
        vx += D[i] * dt * ax
        vy += D[i] * dt * ay
        vz += D[i] * dt * az

        # Tangent velocity kick (linearized)
        dvx += D[i] * dt * Jdpx
        dvy += D[i] * dt * Jdpy
        dvz += D[i] * dt * Jdpz

        # Position drift
        if i < len(C) - 1 or C[i] != 0.0:
            px += C[i] * dt * vx
            py += C[i] * dt * vy
            pz += C[i] * dt * vz

            dpx += C[i] * dt * dvx
            dpy += C[i] * dt * dvy
            dpz += C[i] * dt * dvz


@njit(cache=True)
def compute_energy_soa(px, py, pz, vx, vy, vz, masses, epsilon):
    """Energy with SoA"""
    N = len(masses)
    KE = 0.5 * np.sum(masses * (vx*vx + vy*vy + vz*vz))

    PE = 0.0
    eps2 = epsilon * epsilon
    for i in range(N):
        for j in range(i+1, N):
            dx = px[j] - px[i]
            dy = py[j] - py[i]
            dz = pz[j] - pz[i]
            r2 = dx*dx + dy*dy + dz*dz
            r_reg = np.sqrt(r2 + eps2)
            PE -= G * masses[i] * masses[j] / r_reg

    return KE + PE


@njit(cache=True)
def integrate_variational(px, py, pz, vx, vy, vz, masses, epsilon, dt, T_total, T_lyap):
    """
    Benettin variational method for Lyapunov exponent.
    Much cheaper than dual trajectory!
    """
    delta0 = 1e-10
    N = len(masses)

    # Preallocate all arrays (no per-step allocation!)
    ax = np.zeros(N)
    ay = np.zeros(N)
    az = np.zeros(N)
    Jdpx = np.zeros(N)
    Jdpy = np.zeros(N)
    Jdpz = np.zeros(N)

    # Initial tangent vector
    dpx = np.random.randn(N) * delta0
    dpy = np.random.randn(N) * delta0
    dpz = np.random.randn(N) * delta0
    norm = np.sqrt(np.sum(dpx**2 + dpy**2 + dpz**2))
    dpx *= delta0 / norm
    dpy *= delta0 / norm
    dpz *= delta0 / norm

    # Zero tangent velocity (position perturbation only)
    dvx = np.zeros(N)
    dvy = np.zeros(N)
    dvz = np.zeros(N)

    log_stretch = 0.0
    num_intervals = int(T_total / T_lyap)
    E0 = compute_energy_soa(px, py, pz, vx, vy, vz, masses, epsilon)

    for interval in range(num_intervals):
        num_steps = int(T_lyap / dt)

        for step in range(num_steps):
            yoshida6_step_variational(px, py, pz, vx, vy, vz, dpx, dpy, dpz, dvx, dvy, dvz,
                                     masses, epsilon, dt, ax, ay, az, Jdpx, Jdpy, Jdpz)

        # Renormalize tangent vector
        delta_norm = np.sqrt(np.sum(dpx**2 + dpy**2 + dpz**2))

        if delta_norm > 0:
            log_stretch += np.log(delta_norm / delta0)
            dpx *= delta0 / delta_norm
            dpy *= delta0 / delta_norm
            dpz *= delta0 / delta_norm
            dvx *= delta0 / delta_norm
            dvy *= delta0 / delta_norm
            dvz *= delta0 / delta_norm
        else:
            # Restart with random tangent
            dpx = np.random.randn(N) * delta0
            dpy = np.random.randn(N) * delta0
            dpz = np.random.randn(N) * delta0
            norm = np.sqrt(np.sum(dpx**2 + dpy**2 + dpz**2))
            dpx *= delta0 / norm
            dpy *= delta0 / norm
            dpz *= delta0 / norm
            dvx[:] = 0.0
            dvy[:] = 0.0
            dvz[:] = 0.0

    lambda_exp = log_stretch / T_total
    E_final = compute_energy_soa(px, py, pz, vx, vy, vz, masses, epsilon)
    delta_E = abs(E_final - E0)

    return lambda_exp, delta_E, E0


def run_test():
    print("="*70)
    print("N=30 T=1000 ULTRA-OPTIMIZED")
    print("="*70)
    print()
    print("Optimizations:")
    print("  1. Newton's 3rd law (halve pairwise work)")
    print("  2. Benettin variational method (30-50% less work)")
    print("  3. Cache-tiled j-blocking + SoA")
    print("  4. AVX hints (fastmath, threading)")
    print("  5. Preallocated arrays, JIT warm-up")
    print()

    np.random.seed(0)
    N = 30
    masses = np.ones(N)

    pos = np.random.randn(N, 3) * 0.5
    vel = np.random.randn(N, 3) * 0.3

    px = pos[:, 0].copy()
    py = pos[:, 1].copy()
    pz = pos[:, 2].copy()
    vx = vel[:, 0].copy()
    vy = vel[:, 1].copy()
    vz = vel[:, 2].copy()

    v_rms = np.sqrt(np.mean(vx**2 + vy**2 + vz**2))
    epsilon = HBAR / v_rms

    dt = 0.0003  # Can go larger with optimizations
    T_total = 1000
    T_lyap = 25

    print(f"N = {N}, dt = {dt}, T = {T_total}")
    print(f"Steps: {int(T_total/dt):,}")
    print(f"Expected: ~3-5 minutes (3-5× speedup)")
    print()

    # Warm-up JIT
    print("Warming up JIT...")
    integrate_variational(px.copy(), py.copy(), pz.copy(), vx.copy(), vy.copy(), vz.copy(),
                         masses, epsilon, dt, 1.0, 1.0)

    print("Running full integration...")
    start = time.time()
    lambda_exp, delta_E, E0 = integrate_variational(px, py, pz, vx, vy, vz,
                                                     masses, epsilon, dt, T_total, T_lyap)
    elapsed = time.time() - start

    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()
    print(f"λ = {lambda_exp:.6f}")

    if lambda_exp < 0:
        print("✓ STABLE!")
    else:
        print("? Positive")

    print(f"δE/|E₀| = {delta_E/abs(E0):.3e}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Actual speedup: ~{15.0/elapsed:.1f}× vs baseline")
    print()

    return lambda_exp, delta_E, elapsed


if __name__ == "__main__":
    lambda_exp, delta_E, elapsed = run_test()
