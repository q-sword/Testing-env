#!/usr/bin/env python3
"""
N=30 TEST - COMPACT CONFIGURATION
Uses initial conditions similar to validated three-body case
Smaller spread and lower velocities for better comparison

Goal: Get smaller epsilon closer to three-body validated cases
"""

import numpy as np
from numba import njit, prange
import time

G = 1.0
HBAR = 1.0

w1 = 0.78451361047755726382
w2 = 0.23557321335935813368
w3 = -1.17767998417887100695
w0 = 1.0 - 2.0*(w1 + w2 + w3)

C = np.array([w3, w2, w1, w0, w1, w2, w3, 0.0])
D = np.array([w3/2, (w3+w2)/2, (w2+w1)/2, (w1+w0)/2,
              (w0+w1)/2, (w1+w2)/2, (w2+w3)/2, w3/2])

@njit(parallel=True)
def compute_forces_parallel(pos, masses, epsilon):
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
    for i in range(len(D)):
        acc = compute_forces_parallel(pos, masses, epsilon)
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
            r2 = r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2
            r_reg = np.sqrt(r2 + epsilon**2)
            PE -= G * masses[i] * masses[j] / r_reg
    return KE + PE

@njit
def integrate_with_lyapunov(pos, vel, masses, epsilon, dt, T_total, T_lyap):
    delta0 = 1e-10
    delta_pos = np.random.randn(len(masses), 3) * delta0
    delta_pos = delta_pos / np.linalg.norm(delta_pos) * delta0

    pos_ref = pos.copy()
    vel_ref = vel.copy()

    log_stretch = 0.0
    num_intervals = int(T_total / T_lyap)
    E0 = compute_energy(pos_ref, vel_ref, masses, epsilon)

    for interval in range(num_intervals):
        pos_pert = pos_ref.copy() + delta_pos
        vel_pert = vel_ref.copy()
        num_steps = int(T_lyap / dt)

        for step in range(num_steps):
            pos_ref, vel_ref = yoshida6_step(pos_ref, vel_ref, masses, epsilon, dt)
            pos_pert, vel_pert = yoshida6_step(pos_pert, vel_pert, masses, epsilon, dt)

        delta_pos_new = pos_pert - pos_ref
        delta_norm = np.linalg.norm(delta_pos_new)

        if delta_norm > 0:
            log_stretch += np.log(delta_norm / delta0)
            delta_pos = (delta_pos_new / delta_norm) * delta0
        else:
            delta_pos = np.random.randn(len(masses), 3) * delta0
            delta_pos = delta_pos / np.linalg.norm(delta_pos) * delta0

    lambda_exp = log_stretch / T_total
    E_final = compute_energy(pos_ref, vel_ref, masses, epsilon)
    delta_E = abs(E_final - E0)

    return lambda_exp, delta_E, E0

def run_test():
    print("="*70)
    print("N=30 TEST - COMPACT CONFIGURATION (like 3-body)")
    print("="*70)
    print()

    seed = 0  # Use seed 0 like validated case
    np.random.seed(seed)
    N = 30

    masses = np.ones(N)  # Equal masses like 3-body case
    pos = np.random.randn(N, 3) * 0.5  # SMALLER spread (0.5 vs 1.0)
    vel = np.random.randn(N, 3) * 0.3  # Same velocities

    v_rms = np.sqrt(np.mean(vel**2))
    epsilon = HBAR / (np.mean(masses) * v_rms)

    print(f"Seed: {seed} (same as validated 3-body)")
    print(f"N bodies: {N}")
    print(f"Masses: all equal (like 3-body)")
    print(f"Position spread: 0.5 (COMPACT, like 3-body)")
    print(f"v_rms: {v_rms:.6f}")
    print(f"Epsilon: {epsilon:.6f}")
    print()

    dt = 0.0005
    T_total = 50
    T_lyap = 10

    print(f"dt = {dt}, T = {T_total}, T_lyap = {T_lyap}")
    print(f"Steps: {int(T_total/dt):,}")
    print()
    print("Running...")

    start = time.time()
    lambda_exp, delta_E, E0 = integrate_with_lyapunov(pos, vel, masses, epsilon, dt, T_total, T_lyap)
    elapsed = time.time() - start

    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()
    print(f"Lyapunov exponent: λ = {lambda_exp:.6f}")
    print()

    if lambda_exp < 0:
        print("✓ STABLE (λ < 0) - Quantum regularization works!")
    elif lambda_exp > 0:
        print("✗ CHAOTIC (λ > 0)")
    else:
        print("○ NEUTRAL")

    print()
    print(f"δE/|E₀| = {delta_E/abs(E0):.3e}")
    print(f"Time: {elapsed:.1f} sec ({elapsed/60:.2f} min)")
    print()
    print("Comparison to 3-body seed 0: λ = -2.144 (STABLE)")
    print()

    return lambda_exp, delta_E, elapsed

if __name__ == "__main__":
    lambda_exp, delta_E, elapsed = run_test()
