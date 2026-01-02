#!/usr/bin/env python3
"""
EXACT FIT SEARCH
================

Find formulas that fit EXACTLY, then look for pattern.
"""

import numpy as np
from itertools import product

print("=" * 80)
print("EXACT FIT SEARCH")
print("=" * 80)

# Experimental values (high precision)
alpha_exp = 1/137.035999084
sin2_W_exp = 0.23121
alpha_s_exp = 0.1179

# G₂ numbers
pi = np.pi
pi2 = np.pi**2

# =============================================================================
# α IS SOLVED - what makes 156 and 14 special?
# =============================================================================
print("\n" + "=" * 80)
print("α: UNDERSTANDING 156 AND 14")
print("=" * 80)

# From 1/α + 156α = 14π²
# We have A = 156 = 12×13, B = 14

# Check: 156/14 = 12×13/14 ≈ 11.14
# Check: 156 + 14 = 170
# Check: 156 - 14 = 142
# Check: 156 × 14 = 2184

print(f"For α: A = 156, B = 14")
print(f"  A/B = {156/14:.4f}")
print(f"  A - B = {156 - 14}")
print(f"  A = 12 × 13 = |Δ| × (|Δ|+1)")
print(f"  B = 14 = dim(G₂)")

# The formula Ax² - Bπ²x + 1 = 0 gives:
# x = (Bπ² - √(B²π⁴ - 4A)) / (2A)
# For small x: x ≈ 1/(Bπ²) as leading order

# =============================================================================
# DIFFERENT FORMULA TYPES FOR sin²θ_W
# =============================================================================
print("\n" + "=" * 80)
print("sin²θ_W: SEARCHING ALL FORMULA TYPES")
print("=" * 80)

target = sin2_W_exp
print(f"Target: {target}")

# Type 1: Simple ratio a/b (already found 3/13 with 0.19% error)
print("\n--- Type 1: a/b ---")
for a in range(1, 10):
    for b in range(5, 60):
        val = a/b
        diff = abs(val - target)/target * 100
        if diff < 0.01:
            print(f"  {a}/{b} = {val:.7f} (diff: {diff:.5f}%)")

# None within 0.01%, so try with π corrections

# Type 2: a/(b + cπ^n)
print("\n--- Type 2: a/(b + c×π^n) ---")
for a in [1, 2, 3, 4, 6, 7, 12, 13, 14]:
    for b in range(1, 20):
        for c in [-2, -1, 1, 2]:
            for n in [0.5, 1, 2]:
                val = a/(b + c*pi**n)
                diff = abs(val - target)/target * 100
                if diff < 0.01:
                    print(f"  {a}/({b}+{c}π^{n}) = {val:.7f} (diff: {diff:.5f}%)")

# Type 3: (a + c×π^n)/b
print("\n--- Type 3: (a + c×π^n)/b ---")
for a in range(1, 10):
    for b in [12, 13, 14, 26, 52]:
        for c in np.arange(-0.5, 0.5, 0.01):
            val = (a + c*pi)/b
            diff = abs(val - target)/target * 100
            if diff < 0.005:
                print(f"  ({a}+{c:.3f}π)/{b} = {val:.7f} (diff: {diff:.5f}%)")

# Type 4: a/(b - 1/(c×π^n))
print("\n--- Type 4: a/(b - 1/(c×π^n)) ---")
for a in [3]:
    for b in [13]:
        for c in range(1, 50):
            for n in [1, 2]:
                val = a/(b - 1/(c*pi**n))
                diff = abs(val - target)/target * 100
                if diff < 0.01:
                    print(f"  {a}/({b}-1/({c}π^{n})) = {val:.7f} (diff: {diff:.5f}%)")

# Type 5: What correction to 3/13 gives exact?
print("\n--- Type 5: Exact correction to 3/13 ---")
exact_correction = target - 3/13
print(f"  3/13 = {3/13:.7f}")
print(f"  Exact = {target:.7f}")
print(f"  Correction needed: {exact_correction:.7f}")
print(f"  Correction/α = {exact_correction/alpha_exp:.4f}")
print(f"  Correction×13/3 = {exact_correction*13/3:.6f}")
print(f"  Correction×137 = {exact_correction*137:.6f}")
print(f"  Correction/π = {exact_correction/pi:.7f}")
print(f"  Correction×π² = {exact_correction*pi2:.6f}")

# Type 6: 3/(13 - δ) where δ is G₂-related
print("\n--- Type 6: 3/(13 - δ) for exact match ---")
# 3/(13 - δ) = target => δ = 13 - 3/target
delta_exact = 13 - 3/target
print(f"  Need: 3/(13 - δ) = {target}")
print(f"  δ = 13 - 3/{target} = {delta_exact:.7f}")
print(f"  δ/α = {delta_exact/alpha_exp:.4f}")
print(f"  δ×137 = {delta_exact*137:.4f}")
print(f"  δ×13 = {delta_exact*13:.5f}")
print(f"  δ×π² = {delta_exact*pi2:.5f}")
print(f"  1/(δ×π) = {1/(delta_exact*pi):.4f}")

# Type 7: (3 + ε)/13 for exact match
print("\n--- Type 7: (3 + ε)/13 for exact match ---")
eps_exact = target*13 - 3
print(f"  Need: (3 + ε)/13 = {target}")
print(f"  ε = {eps_exact:.7f}")
print(f"  ε/α = {eps_exact/alpha_exp:.4f}")
print(f"  ε×137 = {eps_exact*137:.4f}")
print(f"  ε×12 = {eps_exact*12:.5f}")
print(f"  ε/π = {eps_exact/pi:.6f}")

# What if ε = α × k for some nice k?
print(f"\n  If ε = k×α, then k = {eps_exact/alpha_exp:.4f}")
print(f"  If ε = π/k, then k = {pi/eps_exact:.2f}")
print(f"  If ε = 1/k, then k = {1/eps_exact:.2f}")

# =============================================================================
# DIFFERENT FORMULA TYPES FOR α_s
# =============================================================================
print("\n" + "=" * 80)
print("α_s: SEARCHING ALL FORMULA TYPES")
print("=" * 80)

target = alpha_s_exp
print(f"Target: {target}")

# Type 5: What correction to 2/17 gives exact?
print("\n--- Type 5: Exact correction to 2/17 ---")
exact_correction = target - 2/17
print(f"  2/17 = {2/17:.7f}")
print(f"  Exact = {target:.7f}")
print(f"  Correction needed: {exact_correction:.7f}")
print(f"  Correction/α = {exact_correction/alpha_exp:.4f}")
print(f"  Correction×17/2 = {exact_correction*17/2:.6f}")
print(f"  Correction×137 = {exact_correction*137:.6f}")

# Type 6: 2/(17 - δ) for exact
print("\n--- Type 6: 2/(17 - δ) for exact match ---")
delta_exact = 17 - 2/target
print(f"  Need: 2/(17 - δ) = {target}")
print(f"  δ = {delta_exact:.7f}")
print(f"  δ/α = {delta_exact/alpha_exp:.4f}")
print(f"  δ×137 = {delta_exact*137:.4f}")
print(f"  δ×17 = {delta_exact*17:.5f}")
print(f"  δ×π² = {delta_exact*pi2:.5f}")

# Type 7: (2 + ε)/17 for exact
print("\n--- Type 7: (2 + ε)/17 for exact match ---")
eps_exact = target*17 - 2
print(f"  Need: (2 + ε)/17 = {target}")
print(f"  ε = {eps_exact:.7f}")
print(f"  ε/α = {eps_exact/alpha_exp:.4f}")
print(f"  If ε = k×α, then k = {eps_exact/alpha_exp:.4f}")

# =============================================================================
# UNIFIED CORRECTION PATTERN
# =============================================================================
print("\n" + "=" * 80)
print("UNIFIED CORRECTION PATTERN")
print("=" * 80)

# sin²θ_W correction
eps_sin2 = sin2_W_exp*13 - 3
k_sin2 = eps_sin2/alpha_exp

# α_s correction
eps_as = alpha_s_exp*17 - 2
k_as = eps_as/alpha_exp

print(f"""
If we write:
  sin²θ_W = (3 + k₁α)/13
  α_s = (2 + k₂α)/17

Then:
  k₁ = {k_sin2:.4f}
  k₂ = {k_as:.4f}

Are these related?
  k₁/k₂ = {k_sin2/k_as:.4f}
  k₁ + k₂ = {k_sin2 + k_as:.4f}
  k₁ - k₂ = {k_sin2 - k_as:.4f}
  k₁ × k₂ = {k_sin2 * k_as:.4f}
""")

# Check specific values
print(f"k₁ ≈ {k_sin2:.2f} ≈ 3/4 = {3/4:.2f}?  diff = {abs(k_sin2 - 0.75):.4f}")
print(f"k₁ ≈ {k_sin2:.2f} ≈ 10/13 = {10/13:.2f}?  diff = {abs(k_sin2 - 10/13):.4f}")

print(f"k₂ ≈ {k_as:.2f} ≈ 1/3 = {1/3:.2f}?  diff = {abs(k_as - 1/3):.4f}")
print(f"k₂ ≈ {k_as:.2f} ≈ 5/14 = {5/14:.2f}?  diff = {abs(k_as - 5/14):.4f}")

# =============================================================================
# FINAL EXACT FORMULAS
# =============================================================================
print("\n" + "=" * 80)
print("PROPOSED EXACT FORMULAS")
print("=" * 80)

# Test with nice k values
test_k_pairs = [
    (10/13, 5/14),  # Using G₂ numbers
    (3/4, 1/3),
    (0.78, 0.35),
    (10/13, 1/3),
]

print(f"\nExperimental values:")
print(f"  sin²θ_W = {sin2_W_exp}")
print(f"  α_s = {alpha_s_exp}")

print(f"\nTesting (3 + k₁α)/13 and (2 + k₂α)/17:")
for k1, k2 in test_k_pairs:
    sin2_pred = (3 + k1*alpha_exp)/13
    as_pred = (2 + k2*alpha_exp)/17
    diff1 = abs(sin2_pred - sin2_W_exp)/sin2_W_exp*100
    diff2 = abs(as_pred - alpha_s_exp)/alpha_s_exp*100
    print(f"\n  k₁={k1:.4f}, k₂={k2:.4f}:")
    print(f"    sin²θ_W = {sin2_pred:.7f} (diff: {diff1:.4f}%)")
    print(f"    α_s = {as_pred:.7f} (diff: {diff2:.4f}%)")
