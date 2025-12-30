#!/usr/bin/env python3
"""
Find the CRITICAL epsilon for gravitational N-body stability.

We know:
  - Molecules: ε/r ~ 0.5-1.0 → λ < 0 (transition)
  - Gravity (previous): ε/r ~ 40 → λ < 0 (too large!)

Question: What is the MINIMUM ε/r that stabilizes gravitational N-body?

This will tell us if there's a UNIVERSAL critical ratio, or if gravity
needs something different.
"""

import numpy as np
from numba import njit
import time

# Yoshida coefficients
w1 = 0.78451361047755726382
w2 = 0.23557321335935813368
w3 = -1.17767998417887100695
w0 = 1.0 - 2.0*(w1 + w2 + w3)

YOSHIDA6_C = np.array([w3, w2, w1, w0, w1, w2, w3, 0.0])
YOSHIDA6_D = np.array([
    w3/2, (w3+w2)/2, (w2+w1)/2, (w1+w0)/2,
    (w0+w1)/2, (w1+w2)/2, (w2+w3)/2, w3/2
])

@njit
def compute_forces(pos, masses, epsilon):
    N = len(masses)
    forces = np.zeros((N, 3))
    for i in range(N):
        for j in range(i+1, N):
            r_vec = pos[j] - pos[i]
            r = np.sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2)
            r_reg = np.sqrt(r**2 + epsilon**2)
            F_mag = masses[i] * masses[j] / (r_reg**3)
            F_vec = F_mag * r_vec
            forces[i] += F_vec
            forces[j] -= F_vec
    return forces

@njit
def yoshida6_step(pos, vel, masses, epsilon, dt):
    for i in range(len(YOSHIDA6_D)):
        forces = compute_forces(pos, masses, epsilon)
        for j in range(len(masses)):
            vel[j] += YOSHIDA6_D[i] * dt * (forces[j] / masses[j])
        if i < len(YOSHIDA6_C) - 1 or YOSHIDA6_C[i] != 0.0:
            for j in range(len(masses)):
                pos[j] += YOSHIDA6_C[i] * dt * vel[j]
    return pos, vel

@njit
def compute_lyapunov(pos_ref, vel_ref, pos_pert, vel_pert, masses, epsilon, dt, num_steps, tau_renorm):
    perturbation_size = 1e-8

    sep_initial = 0.0
    for i in range(len(masses)):
        diff = pos_pert[i] - pos_ref[i]
        sep_initial += diff[0]**2 + diff[1]**2 + diff[2]**2
    sep_initial = np.sqrt(sep_initial)

    lyapunov_sum = 0.0
    renorm_count = 0

    for step in range(num_steps):
        pos_ref, vel_ref = yoshida6_step(pos_ref, vel_ref, masses, epsilon, dt)
        pos_pert, vel_pert = yoshida6_step(pos_pert, vel_pert, masses, epsilon, dt)

        if (step + 1) % tau_renorm == 0:
            sep_current = 0.0
            for i in range(len(masses)):
                diff = pos_pert[i] - pos_ref[i]
                sep_current += diff[0]**2 + diff[1]**2 + diff[2]**2
            sep_current = np.sqrt(sep_current)

            if sep_current > 0:
                growth = sep_current / sep_initial
                lyapunov_sum += np.log(growth)
                renorm_count += 1

                scale_factor = perturbation_size / sep_current
                for i in range(len(masses)):
                    delta = pos_pert[i] - pos_ref[i]
                    pos_pert[i] = pos_ref[i] + scale_factor * delta

    if renorm_count > 0:
        lambda_val = lyapunov_sum / (renorm_count * tau_renorm * dt)
    else:
        lambda_val = 0.0

    return lambda_val

def generate_3body(seed):
    """Generate 3-body system (equal masses)."""
    np.random.seed(seed)
    masses = np.array([1.0, 1.0, 1.0])
    positions = np.random.uniform(-2.0, 2.0, size=(3, 3))
    velocities = np.random.uniform(-0.5, 0.5, size=(3, 3))

    # Zero total momentum
    total_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
    for i in range(3):
        velocities[i] -= total_momentum / (3.0 * masses[i])

    # Compute typical separation
    r_typical = 0.0
    count = 0
    for i in range(3):
        for j in range(i+1, 3):
            r_vec = positions[j] - positions[i]
            r_typical += np.linalg.norm(r_vec)
            count += 1
    r_typical /= count

    return positions, velocities, masses, r_typical

# Test epsilon scan
print("="*70)
print("FINDING CRITICAL EPSILON FOR GRAVITATIONAL 3-BODY")
print("="*70)
print()
print("Testing seed 0 with different epsilon values")
print("Looking for transition from λ > 0 (chaos) to λ < 0 (stable)")
print()

seed = 0
positions, velocities, masses, r_typical = generate_3body(seed)

print(f"System: 3 bodies, m = 1.0 each")
print(f"Typical separation: r ~ {r_typical:.2f}")
print()
print(f"{'ε':<8} {'ε/r':<8} {'λ':<12} {'Status':<12}")
print("-"*45)

# Scan epsilon from small to large
epsilon_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0]

dt = 0.0001
num_steps = 100000  # Shorter for speed
tau_renorm = 1000

for eps in epsilon_values:
    pos_ref = positions.copy()
    vel_ref = velocities.copy()
    pos_pert = positions.copy() + 1e-8 * np.random.randn(3, 3)
    vel_pert = velocities.copy()

    lam = compute_lyapunov(pos_ref, vel_ref, pos_pert, vel_pert,
                           masses, eps, dt, num_steps, tau_renorm)

    status = "✓ STABLE" if lam < 0 else "✗ CHAOTIC"
    ratio = eps / r_typical

    print(f"{eps:<8.2f} {ratio:<8.3f} {lam:<+12.6f} {status:<12}")

print()
print("="*70)
print("INTERPRETATION:")
print("="*70)
print()
print("If transition occurs at ε/r ~ 0.5-1.0:")
print("  → Same as molecules (UNIVERSAL principle)")
print()
print("If transition occurs at much larger ε/r:")
print("  → Gravity may need different mechanism")
print()
print("If NO transition (always chaotic for small ε):")
print("  → Classical 3-body IS inherently chaotic")
print("  → Need quantum OR physical size regularization")
print()
print("="*70)
