#!/usr/bin/env python3
"""
FINDING EXACT FORMULAS FROM G₂ STRUCTURE
=========================================

The current fits are off by ~0.2%.
Let's find formulas that match to 0.01% or better.
"""

import numpy as np
from itertools import product, combinations_with_replacement

print("=" * 80)
print("FINDING EXACT FORMULAS FROM G₂")
print("=" * 80)

# Experimental values
alpha_exp = 1/137.035999084
sin2_W_exp = 0.23121
alpha_s_exp = 0.1179

# G₂ numbers
G2_NUMS = {
    'dim': 14,
    'rank': 2,
    'roots': 12,
    'roots+1': 13,
    'short': 6,
    'long': 6,
    'casimir': 4,
    '156': 156,
    '17': 17,  # dim + 3
    'F4': 52,
    'E6': 78,
    'E7': 133,
    'E8': 248,
}

nums = [1, 2, 3, 4, 6, 7, 8, 12, 13, 14, 17, 52, 78, 133, 156, 248]

# =============================================================================
# DEEP SEARCH FOR sin²θ_W
# =============================================================================
print("\n" + "=" * 80)
print("EXACT FORMULA SEARCH: sin²θ_W = 0.23121")
print("=" * 80)

target = sin2_W_exp
best = []

# Type 1: a/b
for a in nums:
    for b in nums:
        if b > a:
            val = a/b
            diff = abs(val - target)/target * 100
            if diff < 0.5:
                best.append((f"{a}/{b}", val, diff, 'simple'))

# Type 2: a/(b + π)
for a in nums:
    for b in range(1, 200):
        val = a/(b + np.pi)
        diff = abs(val - target)/target * 100
        if diff < 0.1:
            best.append((f"{a}/({b}+π)", val, diff, 'pi'))

# Type 3: a/(b + π²)
for a in nums:
    for b in range(1, 100):
        val = a/(b + np.pi**2)
        diff = abs(val - target)/target * 100
        if diff < 0.1:
            best.append((f"{a}/({b}+π²)", val, diff, 'pi2'))

# Type 4: (a + π/c) / b
for a in nums:
    for b in nums:
        for c in [2, 3, 4, 6, 7, 12, 13, 14]:
            if b > 0:
                val = (a + np.pi/c) / b
                diff = abs(val - target)/target * 100
                if diff < 0.1:
                    best.append((f"({a}+π/{c})/{b}", val, diff, 'mixed'))

# Type 5: a / (b + c/π)
for a in nums:
    for b in range(1, 20):
        for c in nums:
            val = a / (b + c/np.pi)
            diff = abs(val - target)/target * 100
            if diff < 0.1:
                best.append((f"{a}/({b}+{c}/π)", val, diff, 'frac'))

# Type 6: Quadratic roots
# ax² + bx + c = 0  => x = (-b ± √(b²-4ac))/(2a)
for a in nums[:10]:
    for b in nums[:10]:
        for c in [1, 2, 3, 4]:
            disc = b**2 - 4*a*c
            if disc > 0:
                x1 = (-b + np.sqrt(disc))/(2*a)
                x2 = (-b - np.sqrt(disc))/(2*a)
                for x in [x1, x2]:
                    if 0 < x < 1:
                        diff = abs(x - target)/target * 100
                        if diff < 0.1:
                            best.append((f"root of {a}x²+{b}x+{c}=0", x, diff, 'quad'))

# Type 7: a/(b*π - c)
for a in nums[:8]:
    for b in range(1, 10):
        for c in range(1, 50):
            denom = b*np.pi - c
            if denom > 0:
                val = a/denom
                diff = abs(val - target)/target * 100
                if diff < 0.05:
                    best.append((f"{a}/({b}π-{c})", val, diff, 'pilin'))

# Type 8: (a - 1/π²)/b
for a in nums[:8]:
    for b in nums:
        if b > 0:
            val = (a - 1/np.pi**2)/b
            diff = abs(val - target)/target * 100
            if diff < 0.1:
                best.append((f"({a}-1/π²)/{b}", val, diff, 'sub'))

# Type 9: 3/(13 + small correction)
# We know 3/13 is close, what's the exact correction?
for c in range(-100, 0):
    val = 3/(13 + c/1000)
    diff = abs(val - target)/target * 100
    if diff < 0.01:
        best.append((f"3/(13+{c}/1000)", val, diff, 'correct'))

# Sort by accuracy
best.sort(key=lambda x: x[2])

print("\nTop 20 formulas for sin²θ_W:")
print("-" * 70)
seen_vals = set()
count = 0
for formula, val, diff, ftype in best:
    val_rounded = round(val, 8)
    if val_rounded not in seen_vals and count < 20:
        seen_vals.add(val_rounded)
        marker = "***" if diff < 0.01 else "**" if diff < 0.05 else "*" if diff < 0.1 else ""
        print(f"  {formula:30s} = {val:.7f}  ({diff:.5f}%) {marker}")
        count += 1

# =============================================================================
# DEEP SEARCH FOR α_s
# =============================================================================
print("\n" + "=" * 80)
print("EXACT FORMULA SEARCH: α_s = 0.1179")
print("=" * 80)

target = alpha_s_exp
best = []

# Type 1: a/b
for a in nums:
    for b in nums + list(range(100, 260)):
        if b > 0:
            val = a/b
            diff = abs(val - target)/target * 100
            if diff < 0.5:
                best.append((f"{a}/{b}", val, diff, 'simple'))

# Type 2: a/(b + π)
for a in nums:
    for b in range(100, 200):
        val = a/(b + np.pi)
        diff = abs(val - target)/target * 100
        if diff < 0.1:
            best.append((f"{a}/({b}+π)", val, diff, 'pi'))

# Type 3: a/(b + π²)
for a in nums:
    for b in range(50, 150):
        val = a/(b + np.pi**2)
        diff = abs(val - target)/target * 100
        if diff < 0.1:
            best.append((f"{a}/({b}+π²)", val, diff, 'pi2'))

# Type 4: 2/(17 + correction)
for c in range(-100, 100):
    val = 2/(17 + c/1000)
    diff = abs(val - target)/target * 100
    if diff < 0.01:
        best.append((f"2/(17+{c}/1000)", val, diff, 'correct'))

# Type 5: a/(b*π + c)
for a in nums[:8]:
    for b in range(1, 50):
        for c in range(1, 100):
            val = a/(b*np.pi + c)
            diff = abs(val - target)/target * 100
            if diff < 0.05:
                best.append((f"{a}/({b}π+{c})", val, diff, 'pilin'))

# Sort
best.sort(key=lambda x: x[2])

print("\nTop 20 formulas for α_s:")
print("-" * 70)
seen_vals = set()
count = 0
for formula, val, diff, ftype in best:
    val_rounded = round(val, 8)
    if val_rounded not in seen_vals and count < 20:
        seen_vals.add(val_rounded)
        marker = "***" if diff < 0.01 else "**" if diff < 0.05 else "*" if diff < 0.1 else ""
        print(f"  {formula:30s} = {val:.7f}  ({diff:.5f}%) {marker}")
        count += 1

# =============================================================================
# LOOK FOR UNIFIED PATTERN
# =============================================================================
print("\n" + "=" * 80)
print("UNIFIED PATTERN SEARCH")
print("=" * 80)

print("""
All three couplings should follow the same pattern.

For α: 1/α + 156α = 14π²
       where 156 = 12×13, 14 = dim(G₂)

Let's see if sin²θ_W and α_s fit:
  1/x + Ax = Bπ²
""")

# For each coupling, find best A, B
couplings = [
    ('α', alpha_exp),
    ('sin²θ_W', sin2_W_exp),
    ('α_s', alpha_s_exp),
]

for name, x in couplings:
    print(f"\n{name} = {x:.6f}:")
    print(f"  1/x = {1/x:.4f}")

    # Find best integer A, B such that 1/x + Ax = Bπ²
    best_AB = []
    for A in range(1, 200):
        lhs = 1/x + A*x
        B = lhs / np.pi**2
        if abs(B - round(B)) < 0.01:
            best_AB.append((A, round(B), abs(B - round(B))))

    if best_AB:
        best_AB.sort(key=lambda x: x[2])
        A, B, err = best_AB[0]
        print(f"  Best fit: 1/x + {A}x = {B}π² (error: {err:.4f})")
    else:
        # Find best fractional B
        for A in [12, 13, 14, 52, 78, 156]:
            lhs = 1/x + A*x
            B = lhs / np.pi**2
            print(f"  A = {A:3d}: B = {B:.4f}")

# =============================================================================
# THE CORRECTION PATTERN
# =============================================================================
print("\n" + "=" * 80)
print("CORRECTION PATTERN")
print("=" * 80)

print("""
What if the simple formulas (3/13, 2/17) are "bare" values,
and there's a universal correction factor?
""")

# sin²θ_W
bare_sin2 = 3/13
exp_sin2 = sin2_W_exp
ratio_sin2 = exp_sin2 / bare_sin2

# α_s
bare_as = 2/17
exp_as = alpha_s_exp
ratio_as = exp_as / bare_as

print(f"sin²θ_W: exp/bare = {exp_sin2}/{bare_sin2:.6f} = {ratio_sin2:.6f}")
print(f"α_s:     exp/bare = {exp_as}/{bare_as:.6f} = {ratio_as:.6f}")
print(f"\nAverage correction factor: {(ratio_sin2 + ratio_as)/2:.6f}")

# What could this correction be?
avg_corr = (ratio_sin2 + ratio_as)/2 - 1
print(f"\nCorrection δ ≈ {avg_corr:.5f}")
print(f"δ × 137 ≈ {avg_corr * 137:.3f}")
print(f"δ × π ≈ {avg_corr * np.pi:.5f}")
print(f"δ / α ≈ {avg_corr / alpha_exp:.3f}")

# Maybe correction is α × something?
print(f"\nIf correction = k × α:")
print(f"  k = δ/α = {avg_corr/alpha_exp:.3f}")
print(f"  So exp = bare × (1 + {avg_corr/alpha_exp:.1f}α)")

# =============================================================================
# FINAL IMPROVED FORMULAS
# =============================================================================
print("\n" + "=" * 80)
print("IMPROVED FORMULAS")
print("=" * 80)

# Using correction factor
k = 0.26  # From our analysis
correction = 1 + k * alpha_exp

print(f"\nWith correction factor (1 + 0.26α) = {correction:.6f}:")
print(f"  sin²θ_W = (3/13) × {correction:.6f} = {3/13 * correction:.6f}")
print(f"  α_s = (2/17) × {correction:.6f} = {2/17 * correction:.6f}")
print(f"\nCompare to experimental:")
print(f"  sin²θ_W: pred = {3/13 * correction:.6f}, exp = {sin2_W_exp:.6f}")
print(f"  α_s:     pred = {2/17 * correction:.6f}, exp = {alpha_s_exp:.6f}")
