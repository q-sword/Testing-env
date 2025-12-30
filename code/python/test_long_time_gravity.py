#!/usr/bin/env python3
"""
Test if gravitational 3-body shows chaos on LONGER timescales.

Current test: T = 10 time units
Classical 3-body chaos: Should appear on timescales ~ few orbits

Maybe we haven't integrated long enough to see chaos?
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/Testing-env')
from code.python.find_critical_epsilon_gravity import *

print("="*70)
print("LONG-TIME GRAVITATIONAL 3-BODY TEST")
print("="*70)
print()

positions, velocities, masses, r_typical = generate_3body(0)

print(f"System: 3 bodies, r ~ {r_typical:.2f}")
print()
print("Testing different integration times with ε = 0.1 (small)")
print()
print(f"{'Time':<10} {'Steps':<12} {'λ':<12} {'Status':<12}")
print("-"*50)

dt = 0.0001
epsilon = 0.1  # Small regularization

for T_total in [10, 50, 100, 500, 1000]:
    num_steps = int(T_total / dt)
    tau_renorm = max(100, num_steps // 100)

    pos_ref = positions.copy()
    vel_ref = velocities.copy()
    pos_pert = positions.copy() + 1e-8 * np.random.randn(3, 3)
    vel_pert = velocities.copy()

    lam = compute_lyapunov(pos_ref, vel_ref, pos_pert, vel_pert,
                           masses, epsilon, dt, num_steps, tau_renorm)

    status = "✓ STABLE" if lam < 0 else "✗ CHAOTIC"

    print(f"{T_total:<10} {num_steps:<12,} {lam:<+12.6f} {status:<12}")

print()
print("="*70)
print("INTERPRETATION:")
print("="*70)
print()
print("If λ remains < 0 even for long times:")
print("  → Bound gravitational 3-body IS stable (not chaotic!)")
print("  → This contradicts the 'famous chaos' of 3-body problem")
print()
print("Possible resolution:")
print("  1. Bound + virialized systems are STABLE")
print("  2. Chaos only appears in close-encounter scattering")
print("  3. Regularization (even tiny) prevents chaos")
print("  4. There IS a physical stabilization mechanism!")
print()
print("="*70)
