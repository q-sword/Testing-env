#!/usr/bin/env python3

"""
TIMESTEP TESTING PROTOCOL

Tests different dt values to find the sweet spot:
- Fast enough (fewer steps)
- Accurate enough (energy conservation)

Run this BEFORE full T=1000 to find optimal dt.
Takes ~5 minutes total.
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
def run_simulation(pos, vel, masses, epsilon, dt, num_steps):
    """
    Run simulation for num_steps with given dt
    Returns: (pos_final, vel_final, E0, E_final)
    """
    pos_sim = pos.copy()
    vel_sim = vel.copy()

    E0 = compute_energy(pos_sim, vel_sim, masses, epsilon)

    for step in range(num_steps):
        pos_sim, vel_sim = yoshida6_step(pos_sim, vel_sim, masses, epsilon, dt)

    E_final = compute_energy(pos_sim, vel_sim, masses, epsilon)

    return pos_sim, vel_sim, E0, E_final

def test_timestep(pos, vel, masses, epsilon, dt, T_test):
    """
    Test a specific timestep for accuracy and speed
    Returns: (energy_drift, time_per_step, num_steps)
    """
    num_steps = int(T_test / dt)

    start = time.time()
    pos_final, vel_final, E0, E_final = run_simulation(pos, vel, masses, epsilon, dt, num_steps)
    elapsed = time.time() - start

    energy_drift = abs(E_final - E0) / abs(E0)
    time_per_step = elapsed / num_steps

    return energy_drift, time_per_step, num_steps


def run_tests():
    print("="*70)
    print("TIMESTEP TESTING PROTOCOL")
    print("="*70)
    print()
    print("Testing different dt values to find optimal:")
    print("  - Energy conservation (accuracy)")
    print("  - Runtime (speed)")
    print()
    print("Test duration: T=50 (shorter than full T=1000)")
    print("="*70)
    print()

    # Setup (same as your code)
    np.random.seed(0)
    N = 30
    masses = np.ones(N)
    pos = np.random.randn(N, 3) * 0.5
    vel = np.random.randn(N, 3) * 0.3
    v_rms = np.sqrt(np.mean(vel**2))
    epsilon = HBAR / v_rms

    # Warmup JIT
    print("Warming up JIT compiler...")
    _ = test_timestep(pos, vel, masses, epsilon, 0.0001, 1.0)
    print("Warmup complete!\n")

    # Test range of timesteps
    timesteps = [
        0.0001,   # Very conservative (baseline)
        0.0002,   # Your original
        0.00025,  # My suggestion
        0.0003,   # 50% larger
        0.0004,   # 2× larger
        0.0005,   # 2.5× larger
        0.0008,   # 4× larger (probably too large)
        0.001,    # 5× larger (definitely too large)
    ]

    T_test = 50.0  # Short test

    print(f"{'dt':<12} {'Steps':<10} {'δE/E₀':<15} {'Time(s)':<10} {'T=1000(min)':<12} {'Status'}")
    print("-"*80)

    results = []

    for dt in timesteps:
        energy_drift, time_per_step, num_steps = test_timestep(
            pos, vel, masses, epsilon, dt, T_test
        )

        total_time = time_per_step * num_steps

        # Extrapolate to T=1000
        steps_T1000 = int(1000 / dt)
        time_T1000 = steps_T1000 * time_per_step

        # Determine status
        if energy_drift < 1e-13:
            status = "✅ EXCELLENT"
        elif energy_drift < 1e-11:
            status = "✅ GOOD"
        elif energy_drift < 1e-9:
            status = "⚠️  ACCEPTABLE"
        elif energy_drift < 1e-7:
            status = "⚠️  MARGINAL"
        else:
            status = "❌ TOO LARGE"

        print(f"{dt:<12.5f} {num_steps:<10} {energy_drift:<15.3e} {total_time:<10.2f} {time_T1000/60:<12.1f} {status}")

        results.append({
            'dt': dt,
            'drift': energy_drift,
            'time_T1000': time_T1000,
            'status': status
        })

    print()
    print("="*70)
    print("ANALYSIS")
    print("="*70)
    print()

    # Find optimal (best time with acceptable accuracy)
    acceptable = [r for r in results if r['drift'] < 1e-10]

    if acceptable:
        optimal = min(acceptable, key=lambda r: r['time_T1000'])

        print("RECOMMENDED TIMESTEP:")
        print(f"  dt = {optimal['dt']:.5f}")
        print(f"  Energy drift: {optimal['drift']:.3e}")
        print(f"  T=1000 runtime: {optimal['time_T1000']/60:.1f} minutes")
        print(f"  Status: {optimal['status']}")
        print()

        # Compare to baseline
        baseline = results[0]  # dt=0.0001
        speedup = baseline['time_T1000'] / optimal['time_T1000']

        print(f"SPEEDUP vs dt=0.0001: {speedup:.2f}×")
        print()

        # Safety margin
        if optimal['drift'] < 1e-12:
            print("✅ Excellent safety margin - can probably go slightly larger")
        elif optimal['drift'] < 1e-11:
            print("✅ Good safety margin - this is a safe choice")
        elif optimal['drift'] < 1e-10:
            print("⚠️  Moderate safety margin - don't go larger")

    else:
        print("⚠️  WARNING: No timesteps achieved δE/E₀ < 1e-10!")
        print("   System may be sensitive to timestep.")
        print("   Use dt = 0.0001 (most conservative)")

    print()
    print("="*70)
    print("GUIDELINES")
    print("="*70)
    print()
    print("Energy conservation targets:")
    print("  δE/E₀ < 1e-13: EXCELLENT (machine precision)")
    print("  δE/E₀ < 1e-11: GOOD (publication quality)")
    print("  δE/E₀ < 1e-10: ACCEPTABLE (still very accurate)")
    print("  δE/E₀ < 1e-8:  MARGINAL (use with caution)")
    print("  δE/E₀ > 1e-8:  TOO LARGE (inaccurate)")
    print()
    print("Yoshida 6th order:")
    print("  - Error scales as O(dt⁷)")
    print("  - Very stable for symplectic integration")
    print("  - Can handle larger dt than lower-order methods")
    print()
    print("System-dependent limits:")
    print("  - Depends on ε/<r> ratio")
    print("  - Depends on energy/timescales")
    print("  - Test before using in production!")
    print()


if __name__ == "__main__":
    run_tests()
