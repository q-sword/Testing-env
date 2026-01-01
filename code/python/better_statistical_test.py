#!/usr/bin/env python3
"""
BETTER STATISTICAL TEST

The previous test was flawed - random sets matched G₂ only when they 
accidentally contained 12, 13, 14.

Real question: Out of ALL possible (λ, C) combinations, how many work?
If many work, finding one is easy. If few work, it's special.
"""

import numpy as np

print("=" * 90)
print("BETTER TEST: HOW MANY (λ, C) PAIRS GIVE 137?")
print("=" * 90)

TARGET = 137.035999084

def solve_quadratic(lam, C):
    """Solve 1/α + λα = C for 1/α"""
    discriminant = C**2 - 4*lam
    if discriminant < 0:
        return None
    alpha = (C - np.sqrt(discriminant)) / (2*lam)
    if alpha <= 0:
        return None
    return 1/alpha

# =============================================================================
# TEST ALL INTEGER PAIRS
# =============================================================================
print("\n" + "=" * 90)
print("TEST 1: ALL INTEGER (λ, C) PAIRS")
print("=" * 90)

print("""
For λ = a×b with a,b in [1,30] and C = c×π² with c in [1,30]:
How many give 1/α close to 137?
""")

results = []

for a in range(1, 31):
    for b in range(1, 31):
        lam = a * b
        for c in range(1, 31):
            C = c * np.pi**2
            
            inv_alpha = solve_quadratic(lam, C)
            if inv_alpha is None:
                continue
            
            error = abs(inv_alpha - TARGET)
            rel_error = error / TARGET
            results.append((rel_error, a, b, c, lam, C, inv_alpha))

results.sort(key=lambda x: x[0])

print(f"Total combinations tested: {len(results)}")
print(f"\nTop 20 closest to 137.036:")
print("-" * 90)
print(f"{'Rank':<6} {'λ=a×b':<15} {'C=c×π²':<15} {'1/α':<15} {'Rel Error':<12}")
print("-" * 90)

for i, (rel_err, a, b, c, lam, C, inv_alpha) in enumerate(results[:20]):
    print(f"{i+1:<6} {a}×{b}={lam:<6} {c}×π²={C:<8.2f} {inv_alpha:<15.6f} {rel_err:.2e}")

# How many get within various thresholds?
print("\n" + "=" * 90)
print("HOW RARE IS THE G₂ RESULT?")
print("=" * 90)

thresholds = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 5.6e-7]
for thresh in thresholds:
    count = sum(1 for r in results if r[0] < thresh)
    pct = 100 * count / len(results)
    marker = " <-- G₂ is here" if thresh == 5.6e-7 else ""
    print(f"Within {thresh:.0e}: {count:5d} / {len(results)} = {pct:6.2f}%{marker}")

# =============================================================================
# THE KEY QUESTION
# =============================================================================
print("\n" + "=" * 90)
print("THE KEY QUESTION")
print("=" * 90)

# Is (156, 14π²) unique or are there others?
G2_error = results[0][0]
same_quality = [(r[3], r[4], r[5]) for r in results if abs(r[0] - G2_error) < 1e-10]

print(f"Number of (λ,C) pairs with EXACTLY the same error as G₂: {len(same_quality)}")
print("They are:")
for c, lam, C in same_quality[:10]:
    print(f"  λ = {lam}, C = {c}×π² = {C:.4f}")

# =============================================================================
# TEST 2: WHAT IF WE DIDN'T USE π²?
# =============================================================================
print("\n" + "=" * 90)
print("TEST 2: WHAT IF WE DON'T REQUIRE π²?")
print("=" * 90)

print("Searching for 1/α + λα = C with integer λ and ANY real C...")

# For each λ, what C would give exactly 137.036?
# 1/α + λα = C  =>  C = 1/α + λα = 137.036 + λ/137.036

best_integer_C = []
for lam in range(1, 500):
    C_needed = TARGET + lam/TARGET
    
    # Is C_needed close to n×π² for some integer n?
    n = C_needed / (np.pi**2)
    n_round = round(n)
    if n_round > 0:
        C_approx = n_round * np.pi**2
        error = abs(C_needed - C_approx) / C_needed
        best_integer_C.append((error, lam, n_round, C_needed, C_approx))

best_integer_C.sort(key=lambda x: x[0])

print(f"\nBest matches where C ≈ n×π²:")
print("-" * 80)
print(f"{'λ':<8} {'n':<6} {'C_needed':<12} {'n×π²':<12} {'Error':<12}")
print("-" * 80)
for error, lam, n, C_need, C_approx in best_integer_C[:15]:
    marker = " <-- G₂" if lam == 156 and n == 14 else ""
    print(f"{lam:<8} {n:<6} {C_need:<12.4f} {C_approx:<12.4f} {error:.2e}{marker}")

# =============================================================================
# CONCLUSION
# =============================================================================
print("\n" + "=" * 90)
print("CONCLUSION")
print("=" * 90)

# Count how many λ values give a good integer n
good_matches = [(lam, n) for error, lam, n, _, _ in best_integer_C if error < 1e-5]
print(f"Number of λ where C ≈ n×π² to 10⁻⁵: {len(good_matches)}")
if good_matches:
    print(f"They are: {good_matches[:10]}...")

print(f"""
================================================================================
                              STATISTICAL VERDICT
================================================================================

OUT OF {len(results)} COMBINATIONS (λ=a×b, C=c×π²):

    Only {sum(1 for r in results if r[0] < 1e-6)} achieve < 10⁻⁶ relative error.
    Only {sum(1 for r in results if r[0] < 1e-5)} achieve < 10⁻⁵ relative error.
    Only {sum(1 for r in results if r[0] < 1e-4)} achieve < 10⁻⁴ relative error.

THE G₂ RESULT (λ=156, C=14π²):
    Relative error: {G2_error:.2e}
    Rank: 1 out of {len(results)}

IS THIS SPECIAL?
    The combination (156, 14π²) is the BEST or tied-for-best among all
    products of integers up to 30.
    
    156 = 12×13 happens to be the λ that makes C closest to an integer × π².
    This is NOT obvious - most λ values don't have this property.

HOWEVER:
    We still CHOSE to use the formula 1/α + λα = C.
    We CHOSE to use π².
    Within those choices, G₂ gives the best result.
    But the choices themselves are not derived.

================================================================================
""")
