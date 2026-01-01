#!/usr/bin/env python3
"""
STATISTICAL TEST: Is G₂ special or could any numbers work?

If we can match α = 1/137 with random numbers just as easily,
then our G₂ "derivation" is meaningless.
"""

import numpy as np
from itertools import combinations, permutations
import random

np.random.seed(42)

print("=" * 90)
print("STATISTICAL TEST: CAN RANDOM NUMBERS MATCH 137?")
print("=" * 90)

TARGET = 137.035999084
TOLERANCE = 1e-4  # Match to 4 decimal places (0.001%)

# =============================================================================
# THE G₂ NUMBERS WE USED
# =============================================================================
print("\n" + "=" * 90)
print("PART 1: WHAT G₂ NUMBERS DID WE USE?")
print("=" * 90)

G2_numbers = {
    'dim': 14,
    '|Δ|': 12,
    'rank': 2,
    '|Δ|+1': 13,
    'dim-4': 10,
    '|W|': 12,
}

print("G₂ invariants used:")
for name, val in G2_numbers.items():
    print(f"  {name} = {val}")

print("\nWe also used π² ≈ 9.8696")

# =============================================================================
# THE FORMULA TEMPLATE
# =============================================================================
print("\n" + "=" * 90)
print("PART 2: THE FORMULA TEMPLATE")
print("=" * 90)

print("""
Our formula was:
    1/α + λα = C
    
where:
    λ = a × b       (product of two integers)
    C = c × π²      (integer times π²)

Solving: α = (C - √(C² - 4λ)) / (2λ)
Then:    1/α = ...

We'll test if RANDOM integers can match 137 just as well.
""")

def solve_quadratic(lam, C):
    """Solve 1/α + λα = C for α, return 1/α"""
    discriminant = C**2 - 4*lam
    if discriminant < 0:
        return None
    alpha = (C - np.sqrt(discriminant)) / (2*lam)
    if alpha <= 0:
        return None
    return 1/alpha

# =============================================================================
# TEST WITH G₂ NUMBERS
# =============================================================================
print("\n" + "=" * 90)
print("PART 3: G₂ RESULT")
print("=" * 90)

lam_G2 = 12 * 13  # = 156
C_G2 = 14 * np.pi**2  # = 14π²

inv_alpha_G2 = solve_quadratic(lam_G2, C_G2)
error_G2 = abs(inv_alpha_G2 - TARGET)
rel_error_G2 = error_G2 / TARGET

print(f"λ = 12 × 13 = {lam_G2}")
print(f"C = 14 × π² = {C_G2:.6f}")
print(f"1/α = {inv_alpha_G2:.10f}")
print(f"Error = {error_G2:.6f}")
print(f"Relative error = {rel_error_G2:.2e}")

# =============================================================================
# TEST WITH RANDOM NUMBERS
# =============================================================================
print("\n" + "=" * 90)
print("PART 4: TESTING RANDOM NUMBER SETS")
print("=" * 90)

def test_number_set(numbers, use_pi_squared=True):
    """
    Try all combinations of form:
        λ = a × b
        C = c × π² (or just c if use_pi_squared=False)
    
    Return the best match to 137.
    """
    best_error = float('inf')
    best_params = None
    best_inv_alpha = None
    
    multiplier = np.pi**2 if use_pi_squared else 1
    
    for a in numbers:
        for b in numbers:
            lam = a * b
            if lam <= 0:
                continue
            for c in numbers:
                C = c * multiplier
                if C <= 0:
                    continue
                
                inv_alpha = solve_quadratic(lam, C)
                if inv_alpha is None:
                    continue
                
                error = abs(inv_alpha - TARGET)
                if error < best_error:
                    best_error = error
                    best_params = (a, b, c, lam, C)
                    best_inv_alpha = inv_alpha
    
    return best_error, best_params, best_inv_alpha

# Test with G₂-like random sets
print("\nTesting 1000 random number sets (integers 2-20)...")
print("Each set has 6 numbers like G₂ has ~6 distinct invariants.\n")

n_trials = 1000
n_numbers = 6
number_range = (2, 20)

results = []
matches = 0

for trial in range(n_trials):
    # Generate random integers
    random_nums = [random.randint(*number_range) for _ in range(n_numbers)]
    
    # Test this set
    error, params, inv_alpha = test_number_set(random_nums, use_pi_squared=True)
    rel_error = error / TARGET if inv_alpha else float('inf')
    
    results.append((rel_error, random_nums, params, inv_alpha))
    
    if rel_error < rel_error_G2:
        matches += 1

# Sort by error
results.sort(key=lambda x: x[0])

print(f"G₂ relative error: {rel_error_G2:.2e}")
print(f"Random sets that beat G₂: {matches}/{n_trials} = {100*matches/n_trials:.1f}%")

print("\nTop 10 random results:")
print("-" * 80)
for i, (rel_err, nums, params, inv_alpha) in enumerate(results[:10]):
    if params:
        a, b, c, lam, C = params
        print(f"{i+1}. Numbers {nums}")
        print(f"   λ = {a}×{b} = {lam}, C = {c}×π² = {C:.2f}")
        print(f"   1/α = {inv_alpha:.6f}, rel_error = {rel_err:.2e}")
    print()

# =============================================================================
# STATISTICAL SIGNIFICANCE
# =============================================================================
print("\n" + "=" * 90)
print("PART 5: STATISTICAL SIGNIFICANCE")
print("=" * 90)

# Distribution of errors
errors = [r[0] for r in results if r[0] < 1]  # Only finite results
mean_error = np.mean(errors)
std_error = np.std(errors)
min_error = min(errors)

print(f"Distribution of relative errors from random trials:")
print(f"  Mean:   {mean_error:.2e}")
print(f"  Std:    {std_error:.2e}")
print(f"  Min:    {min_error:.2e}")
print(f"  G₂:     {rel_error_G2:.2e}")

# How many standard deviations is G₂ from the mean?
z_score = (rel_error_G2 - mean_error) / std_error
print(f"\nG₂ result is {z_score:.1f} standard deviations from mean")

# What fraction of random trials beat G₂?
fraction_better = matches / n_trials
print(f"Fraction of random trials with smaller error: {fraction_better:.1%}")

if fraction_better > 0.05:
    verdict = "NOT SIGNIFICANT - random numbers work just as well"
elif fraction_better > 0.01:
    verdict = "MARGINALLY SIGNIFICANT - G₂ is somewhat better than random"
else:
    verdict = "SIGNIFICANT - G₂ is notably better than random"

print(f"\nVERDICT: {verdict}")

# =============================================================================
# PART 6: EXPANDED TEST - MORE OPERATIONS
# =============================================================================
print("\n" + "=" * 90)
print("PART 6: EXPANDED TEST WITH MORE OPERATIONS")
print("=" * 90)

print("""
Let's be more generous and allow more operations:
    λ = a × (b + 1)  [like we used 12 × 13]
    C = c × π²
""")

def test_expanded(numbers):
    """Try λ = a × (b + 1) pattern"""
    best_error = float('inf')
    best_params = None
    best_inv_alpha = None
    
    for a in numbers:
        for b in numbers:
            lam = a * (b + 1)  # The pattern we used
            if lam <= 0:
                continue
            for c in numbers:
                C = c * np.pi**2
                if C <= 0:
                    continue
                
                inv_alpha = solve_quadratic(lam, C)
                if inv_alpha is None:
                    continue
                
                error = abs(inv_alpha - TARGET)
                if error < best_error:
                    best_error = error
                    best_params = (a, b, c, lam, C)
                    best_inv_alpha = inv_alpha
    
    return best_error, best_params, best_inv_alpha

# Test G₂ with this pattern
G2_nums = [14, 12, 2, 7, 6, 10, 13]
error_G2_exp, params_G2_exp, inv_alpha_G2_exp = test_expanded(G2_nums)
rel_error_G2_exp = error_G2_exp / TARGET

print(f"G₂ with expanded pattern: rel_error = {rel_error_G2_exp:.2e}")

# Test random
matches_exp = 0
for trial in range(n_trials):
    random_nums = [random.randint(2, 20) for _ in range(7)]
    error, _, _ = test_expanded(random_nums)
    if error / TARGET < rel_error_G2_exp:
        matches_exp += 1

print(f"Random sets that beat G₂: {matches_exp}/{n_trials} = {100*matches_exp/n_trials:.1f}%")

# =============================================================================
# CONCLUSION
# =============================================================================
print("\n" + "=" * 90)
print("CONCLUSION")
print("=" * 90)

print(f"""
================================================================================
                         STATISTICAL ANALYSIS RESULTS
================================================================================

THE TEST:
    We tried {n_trials} random sets of integers (2-20).
    Each set had {n_numbers} numbers, like G₂ has ~6 invariants.
    We used the same formula template: 1/α + λα = C

RESULTS:
    G₂ relative error:     {rel_error_G2:.2e}
    Random sets better:    {100*matches/n_trials:.1f}%
    
INTERPRETATION:
    If {100*matches/n_trials:.0f}% of random sets beat G₂, then finding 
    a formula that works is {"EASY" if matches/n_trials > 0.05 else "HARD"}.
    
    G₂ is {"NOT" if matches/n_trials > 0.05 else ""} statistically special.

THE HONEST TRUTH:
    With enough numbers and operations, you CAN fit 137.
    The question is whether G₂ does it more naturally than random numbers.
    
    Our test shows: {verdict}

================================================================================
""")
