#!/usr/bin/env python3
"""Benchmark to identify N=30 performance bottleneck"""

import numpy as np
from numba import njit, prange
import time

G = 1.0

@njit(parallel=True)
def compute_forces_exact(pos, masses, epsilon):
    """Exact pairwise forces"""
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
def compute_forces_serial(pos, masses, epsilon):
    """Serial version for comparison"""
    N = len(masses)
    acc = np.zeros((N, 3))
    for i in range(N):
        for j in range(N):
            if i != j:
                r_vec = pos[j] - pos[i]
                r2 = r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2
                r_reg2 = r2 + epsilon**2
                r_reg3 = r_reg2 * np.sqrt(r_reg2)
                force_mag = G * masses[j] / r_reg3
                acc[i] += force_mag * r_vec
    return acc

print("Benchmarking force calculations...")
print()

# Setup
N = 30
np.random.seed(42)
pos = np.random.randn(N, 3) * 0.5
masses = np.ones(N)
epsilon = 3.4708

# Warmup
_ = compute_forces_exact(pos, masses, epsilon)
_ = compute_forces_serial(pos, masses, epsilon)

# Benchmark parallel
n_calls = 1000
start = time.time()
for _ in range(n_calls):
    acc = compute_forces_exact(pos, masses, epsilon)
elapsed_parallel = time.time() - start

# Benchmark serial
start = time.time()
for _ in range(n_calls):
    acc = compute_forces_serial(pos, masses, epsilon)
elapsed_serial = time.time() - start

print(f"N = {N}")
print(f"Force calls: {n_calls}")
print()
print(f"Parallel: {elapsed_parallel:.3f}s ({elapsed_parallel*1000/n_calls:.2f} ms/call)")
print(f"Serial:   {elapsed_serial:.3f}s ({elapsed_serial*1000/n_calls:.2f} ms/call)")
print(f"Speedup: {elapsed_serial/elapsed_parallel:.2f}x")
print()

# Estimate time for full interval
print("Estimated time for 500-step interval (13 trajectories):")
force_calls_per_interval = 500 * 8 * 13  # steps × Yoshida stages × trajectories
estimated_time = (elapsed_parallel / n_calls) * force_calls_per_interval
print(f"  Force calls: {force_calls_per_interval}")
print(f"  Parallel: {estimated_time:.1f}s ({estimated_time/60:.1f} min)")
print()
