#!/usr/bin/env python3
"""Benchmark tangent evolution strategies"""

import numpy as np
from numba import njit, prange
import time

G = 1.0
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
                acc[i] += G * masses[j] * r_vec / r_reg3
    return acc

@njit
def compute_forces_serial(pos, masses, epsilon):
    N = len(masses)
    acc = np.zeros((N, 3))
    for i in range(N):
        for j in range(N):
            if i != j:
                r_vec = pos[j] - pos[i]
                r2 = r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2
                r_reg2 = r2 + epsilon**2
                r_reg3 = r_reg2 * np.sqrt(r_reg2)
                acc[i] += G * masses[j] * r_vec / r_reg3
    return acc

@njit
def yoshida_step_serial(pos, vel, masses, epsilon, dt):
    for i in range(len(D)):
        acc = compute_forces_serial(pos, masses, epsilon)
        vel = vel + D[i] * dt * acc
        if i < len(C) - 1 or C[i] != 0.0:
            pos = pos + C[i] * dt * vel
    return pos, vel

@njit
def yoshida_step_parallel_forces(pos, vel, masses, epsilon, dt):
    for i in range(len(D)):
        acc = compute_forces_parallel(pos, masses, epsilon)
        vel = vel + D[i] * dt * acc
        if i < len(C) - 1 or C[i] != 0.0:
            pos = pos + C[i] * dt * vel
    return pos, vel

@njit(parallel=True)
def evolve_tangents_parallel_outer(pos_ref, vel_ref, tangent_pos, tangent_vel,
                                    masses, epsilon, dt, num_steps):
    n_vectors = tangent_pos.shape[0]
    N = len(masses)

    pos_r = pos_ref.copy()
    vel_r = vel_ref.copy()
    for step in range(num_steps):
        pos_r, vel_r = yoshida_step_serial(pos_r, vel_r, masses, epsilon, dt)

    new_tangent_pos = np.zeros((n_vectors, N, 3))
    new_tangent_vel = np.zeros((n_vectors, N, 3))

    for vec_idx in prange(n_vectors):  # PARALLEL trajectories
        pos_p = pos_ref + tangent_pos[vec_idx]
        vel_p = vel_ref + tangent_vel[vec_idx]
        for step in range(num_steps):
            pos_p, vel_p = yoshida_step_serial(pos_p, vel_p, masses, epsilon, dt)
        new_tangent_pos[vec_idx] = pos_p - pos_r
        new_tangent_vel[vec_idx] = vel_p - vel_r

    return pos_r, vel_r, new_tangent_pos, new_tangent_vel

@njit
def evolve_tangents_serial_outer(pos_ref, vel_ref, tangent_pos, tangent_vel,
                                  masses, epsilon, dt, num_steps):
    n_vectors = tangent_pos.shape[0]
    N = len(masses)

    pos_r = pos_ref.copy()
    vel_r = vel_ref.copy()
    for step in range(num_steps):
        pos_r, vel_r = yoshida_step_parallel_forces(pos_r, vel_r, masses, epsilon, dt)

    new_tangent_pos = np.zeros((n_vectors, N, 3))
    new_tangent_vel = np.zeros((n_vectors, N, 3))

    for vec_idx in range(n_vectors):  # SERIAL trajectories, parallel forces
        pos_p = pos_ref + tangent_pos[vec_idx]
        vel_p = vel_ref + tangent_vel[vec_idx]
        for step in range(num_steps):
            pos_p, vel_p = yoshida_step_parallel_forces(pos_p, vel_p, masses, epsilon, dt)
        new_tangent_pos[vec_idx] = pos_p - pos_r
        new_tangent_vel[vec_idx] = vel_p - vel_r

    return pos_r, vel_r, new_tangent_pos, new_tangent_vel

print("Benchmarking tangent evolution strategies...")
print()

N = 30
n_vectors = 12
num_steps = 100  # Short test

np.random.seed(42)
pos = np.random.randn(N, 3) * 0.5
vel = np.random.randn(N, 3) * 0.3
tangent_pos = np.random.randn(n_vectors, N, 3) * 1e-8
tangent_vel = np.random.randn(n_vectors, N, 3) * 1e-8
masses = np.ones(N)
epsilon = 3.4708
dt = 0.001

# Warmup
_ = evolve_tangents_parallel_outer(pos, vel, tangent_pos, tangent_vel, masses, epsilon, dt, 10)
_ = evolve_tangents_serial_outer(pos, vel, tangent_pos, tangent_vel, masses, epsilon, dt, 10)

print(f"N={N}, {n_vectors} tangent vectors, {num_steps} steps")
print()

# Test 1: Parallel trajectories + serial forces
start = time.time()
r1 = evolve_tangents_parallel_outer(pos, vel, tangent_pos, tangent_vel, masses, epsilon, dt, num_steps)
t1 = time.time() - start

# Test 2: Serial trajectories + parallel forces
start = time.time()
r2 = evolve_tangents_serial_outer(pos, vel, tangent_pos, tangent_vel, masses, epsilon, dt, num_steps)
t2 = time.time() - start

print(f"Parallel trajectories (serial forces):  {t1:.3f}s")
print(f"Serial trajectories (parallel forces): {t2:.3f}s")
print()
print(f"Best: {'Parallel traj' if t1 < t2 else 'Serial traj'} ({min(t1,t2):.3f}s)")
print(f"Speedup ratio: {max(t1,t2)/min(t1,t2):.2f}x")
print()

# Extrapolate to 500 steps
scale = 500 / num_steps
print(f"Estimated for 500 steps:")
print(f"  Parallel traj: {t1*scale:.1f}s ({t1*scale/60:.1f} min)")
print(f"  Serial traj:   {t2*scale:.1f}s ({t2*scale/60:.1f} min)")
