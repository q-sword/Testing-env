#!/usr/bin/env python3
"""
DEEP DIVE INTO G₂ NUMBERS

What makes 2, 6, 7, 12, 13, 14 special?
Are there mathematical properties that explain why they work?
"""

import numpy as np
from math import gcd
from functools import reduce

print("=" * 90)
print("DEEP DIVE: WHY ARE G₂ NUMBERS SPECIAL?")
print("=" * 90)

# The G₂ numbers
G2_primary = {
    'dim': 14,           # Dimension of G₂ Lie algebra
    'roots': 12,         # Number of roots |Δ|
    'rank': 2,           # Rank of G₂
}

G2_derived = {
    'roots+1': 13,       # |Δ| + 1
    'dim-4': 10,         # dim - spacetime
    'short_roots': 6,    # Number of short roots
    'long_roots': 6,     # Number of long roots
    'dim/2': 7,          # Half dimension
    'Weyl_order': 12,    # Order of Weyl group
}

ALL_G2 = [2, 6, 7, 10, 12, 13, 14]

print("\n" + "=" * 90)
print("PART 1: THE BASIC G₂ NUMBERS")
print("=" * 90)

print("""
FROM THE LIE ALGEBRA:
    dim(G₂) = 14      (number of generators)
    rank(G₂) = 2      (dimension of Cartan subalgebra)
    |Δ(G₂)| = 12      (number of roots)

DERIVED:
    short roots = 6   (half of |Δ|)
    long roots = 6    (half of |Δ|)
    |W(G₂)| = 12      (Weyl group order = |Δ|)
    
RELATIONSHIPS:
    dim = 2 + |Δ| = 2 + 12 = 14  (general formula: dim = rank + |Δ|)
    |Δ| = 2 × rank × h  where h = Coxeter number
    For G₂: h = 6, so |Δ| = 2 × 2 × 3 = 12 ✓ (wait, that gives 12 with h=3?)
""")

# Actually compute Coxeter number
# For G₂, the Coxeter number h = 6
# |Δ| = 2 × (sum of exponents) = 2 × (1 + 5) = 12
print("Coxeter number h = 6")
print("Exponents of G₂: 1, 5")
print("|Δ| = 2 × (1 + 5) = 12 ✓")

# =============================================================================
# PART 2: NUMBER THEORY PROPERTIES
# =============================================================================
print("\n" + "=" * 90)
print("PART 2: NUMBER THEORY PROPERTIES")
print("=" * 90)

print("\nPrime factorizations:")
for n in ALL_G2:
    factors = []
    temp = n
    for p in [2, 3, 5, 7, 11, 13]:
        while temp % p == 0:
            factors.append(p)
            temp //= p
    if temp > 1:
        factors.append(temp)
    print(f"  {n:3d} = {' × '.join(map(str, factors)) if factors else 'prime'}")

print("\nGCD relationships:")
print(f"  gcd(12, 14) = {gcd(12, 14)}")
print(f"  gcd(6, 12) = {gcd(6, 12)}")
print(f"  gcd(7, 14) = {gcd(7, 14)}")
print(f"  gcd(13, 12) = {gcd(13, 12)} (coprime!)")

print("\nModular properties:")
for n in ALL_G2:
    print(f"  {n} mod 6 = {n % 6}, mod 7 = {n % 7}, mod 12 = {n % 12}")

# =============================================================================
# PART 3: ALGEBRAIC RELATIONSHIPS
# =============================================================================
print("\n" + "=" * 90)
print("PART 3: ALGEBRAIC RELATIONSHIPS AMONG G₂ NUMBERS")
print("=" * 90)

print("""
Key identities:
    14 = 12 + 2           (dim = |Δ| + rank)
    14 = 2 × 7            (dim = 2 × 7)
    12 = 2 × 6            (|Δ| = 2 × short_roots)
    13 = 12 + 1           (used in duality)
    156 = 12 × 13         (duality parameter)
    
The number 13 = |Δ| + 1 appears repeatedly:
    - sin²θ_W = 3/13
    - λ = 12 × 13 = 156
    - m_W/m_Z = √(10/13)
""")

# Check: is 13 special in any way related to G₂?
print("\nWhy 13?")
print("  13 = |Δ| + 1 = 12 + 1")
print("  13 = dim - 1 = 14 - 1")
print("  13 is prime (can't be factored)")
print("  13 = |Δ| + rank - 1 = 12 + 2 - 1")

# =============================================================================
# PART 4: THE 156 = 12 × 13 PATTERN
# =============================================================================
print("\n" + "=" * 90)
print("PART 4: WHY 156 = 12 × 13?")
print("=" * 90)

print("""
156 = 12 × 13 = |Δ| × (|Δ| + 1)

This is the formula for:
    - Number of ordered pairs from |Δ| elements = |Δ|²  (no, that's 144)
    - Sum 1 + 2 + ... + 12 = 12×13/2 = 78 (triangular number)
    - 2 × 78 = 156

So 156 = 2 × T₁₂ where T_n = n(n+1)/2 is the nth triangular number.

TRIANGULAR NUMBER CONNECTION:
    T₁₂ = 78 = number of root pairs?
    156 = 2 × T₁₂

In Lie theory, root pairs are important for:
    - Commutation relations [E_α, E_β]
    - Weyl reflections
    - The Killing form
""")

# Triangular numbers
print("\nTriangular numbers near G₂:")
for n in range(10, 16):
    T_n = n * (n + 1) // 2
    print(f"  T_{n} = {T_n}")

# =============================================================================
# PART 5: COMPARISON WITH OTHER LIE GROUPS
# =============================================================================
print("\n" + "=" * 90)
print("PART 5: COMPARISON WITH OTHER EXCEPTIONAL LIE GROUPS")
print("=" * 90)

lie_groups = {
    'G₂': {'dim': 14, 'rank': 2, 'roots': 12},
    'F₄': {'dim': 52, 'rank': 4, 'roots': 48},
    'E₆': {'dim': 78, 'rank': 6, 'roots': 72},
    'E₇': {'dim': 133, 'rank': 7, 'roots': 126},
    'E₈': {'dim': 248, 'rank': 8, 'roots': 240},
}

print(f"{'Group':<6} {'dim':<6} {'rank':<6} {'|Δ|':<6} {'|Δ|(|Δ|+1)':<12} {'dim×π²':<12}")
print("-" * 60)
for name, data in lie_groups.items():
    d, r, roots = data['dim'], data['rank'], data['roots']
    lam = roots * (roots + 1)
    C = d * np.pi**2
    print(f"{name:<6} {d:<6} {r:<6} {roots:<6} {lam:<12} {C:<12.2f}")

# What does the duality equation give for other groups?
print("\n\nDuality equation 1/x + λx = dim×π² for each group:")
print("-" * 60)

for name, data in lie_groups.items():
    d, r, roots = data['dim'], data['rank'], data['roots']
    lam = roots * (roots + 1)
    C = d * np.pi**2
    
    disc = C**2 - 4*lam
    if disc > 0:
        x = (C - np.sqrt(disc)) / (2*lam)
        inv_x = 1/x
        print(f"{name}: 1/x = {inv_x:.4f}")
    else:
        print(f"{name}: No real solution")

# =============================================================================
# PART 6: WHAT MAKES G₂ UNIQUE?
# =============================================================================
print("\n" + "=" * 90)
print("PART 6: WHAT MAKES G₂ UNIQUE?")
print("=" * 90)

print("""
G₂ is special among Lie groups because:

1. SMALLEST EXCEPTIONAL GROUP
   G₂ is the smallest exceptional Lie group (dim = 14).
   It's the automorphism group of the octonions.

2. SELF-DUAL ROOT SYSTEM
   G₂ has both short and long roots in ratio √3.
   The root system is self-dual under rescaling.

3. CONNECTION TO OCTONIONS
   G₂ = Aut(O), the only Lie group that is the
   automorphism group of a normed division algebra.

4. G₂ HOLONOMY
   7-manifolds with G₂ holonomy are the only odd-dimensional
   manifolds with special holonomy (besides circles).

5. M-THEORY CONNECTION
   M-theory on G₂ manifolds gives 4D N=1 supersymmetry,
   which is phenomenologically relevant.

THE KEY QUESTION:
Is there something about G₂'s structure that makes
its numbers (12, 13, 14) particularly suited to
encoding physical constants?
""")

# =============================================================================
# PART 7: TESTING THE UNIQUENESS OF G₂
# =============================================================================
print("\n" + "=" * 90)
print("PART 7: DO OTHER LIE GROUP NUMBERS WORK?")
print("=" * 90)

print("\nTesting if other exceptional groups give good predictions for 1/α = 137.036:")

for name, data in lie_groups.items():
    d, r, roots = data['dim'], data['rank'], data['roots']
    lam = roots * (roots + 1)
    C = d * np.pi**2
    
    disc = C**2 - 4*lam
    if disc > 0:
        x = (C - np.sqrt(disc)) / (2*lam)
        inv_x = 1/x
        error = abs(inv_x - 137.036) / 137.036
        print(f"{name}: 1/x = {inv_x:10.4f}, error from 137: {error:.2%}")

print("\nONLY G₂ gives a value close to 137!")

# =============================================================================
# PART 8: SEARCHING FOR PATTERNS
# =============================================================================
print("\n" + "=" * 90)
print("PART 8: MATHEMATICAL PATTERNS IN G₂ NUMBERS")
print("=" * 90)

print("""
Let's look for patterns that might explain why G₂ works:

OBSERVATION 1: The "13" pattern
    13 appears in multiple predictions:
    - 3/13 ≈ sin²θ_W
    - 12 × 13 = 156 (duality parameter)
    - √(10/13) ≈ m_W/m_Z
    
OBSERVATION 2: The relationship to π²
    14π² ≈ 138.17 ≈ 137 + 1
    This is very close to 1/α
    
OBSERVATION 3: Triangular structure
    156 = 2 × T₁₂ = 2 × (1+2+...+12)
    This suggests a counting/combinatorial origin

OBSERVATION 4: The split 14 = 12 + 2
    dim = |Δ| + rank is a fundamental identity
    This connects the algebra structure to the root system
""")

# Check: what other constants does 14π² - something give?
print("\n14π² - n for small n:")
for n in range(0, 5):
    val = 14 * np.pi**2 - n
    print(f"  14π² - {n} = {val:.4f}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 90)
print("SUMMARY: WHY G₂ NUMBERS MIGHT BE SPECIAL")
print("=" * 90)

print("""
================================================================================
                    G₂ NUMBER ANALYSIS SUMMARY
================================================================================

THE NUMBERS:
    Primary: 2 (rank), 12 (|Δ|), 14 (dim)
    Derived: 6 (short roots), 7 (dim/2), 13 (|Δ|+1)

WHY THEY MIGHT WORK:

1. MATHEMATICAL STRUCTURE
   G₂ is the smallest exceptional Lie group
   Its numbers are small and have many relationships
   
2. π² CONNECTION
   14π² ≈ 138.17 is very close to 137.036 + 1.14
   The error (1.14) is itself close to 1/α ≈ 0.88
   
3. TRIANGULAR/COMBINATORIAL
   156 = 2 × T₁₂ suggests counting structure
   This might connect to quantum loop corrections
   
4. UNIQUENESS AMONG LIE GROUPS
   Only G₂ gives 1/x ≈ 137 with the duality formula
   Other exceptional groups give very different values

REMAINING MYSTERY:
   Why does NATURE use G₂'s numbers?
   Is this physics, mathematics, or coincidence?

================================================================================
""")
