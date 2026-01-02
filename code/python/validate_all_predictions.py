#!/usr/bin/env python3
"""
VALIDATE ALL G₂ PREDICTIONS

Apply the same statistical test to ALL constants, not just α.
For each prediction, check if G₂ numbers are special or if random numbers work.
"""

import numpy as np
np.random.seed(42)

print("=" * 90)
print("VALIDATING ALL G₂ PREDICTIONS")
print("=" * 90)

# Experimental values
CONSTANTS = {
    '1/α': 137.036,
    'm_p/m_e': 1836.15267,
    'm_μ/m_e': 206.768,
    'sin²θ_W': 0.23122,
    'm_W/m_Z': 0.88145,
    'θ_Cabibbo': 0.2250,  # sin(θ_C) ≈ 0.225
}

# G₂ numbers
G2 = [14, 12, 13, 2, 10, 6, 7, 17, 156, 3]  # Including derived numbers

def test_formula(formula_fn, target, n_random=2000):
    """Test if G₂ is special for a formula"""
    # G₂ best
    g2_best = None
    g2_best_error = float('inf')
    
    for a in G2:
        for b in G2:
            for c in G2:
                try:
                    result = formula_fn(a, b, c)
                    if result is not None and result > 0:
                        error = abs(result - target) / target  # relative error
                        if error < g2_best_error:
                            g2_best_error = error
                            g2_best = result
                except:
                    pass
    
    if g2_best is None:
        return None
    
    # Random test
    better_count = 0
    for _ in range(n_random):
        nums = list(np.random.randint(2, 20, 7)) + [np.random.randint(100, 200)]
        best_err = float('inf')
        for a in nums:
            for b in nums:
                for c in nums:
                    try:
                        result = formula_fn(a, b, c)
                        if result is not None and result > 0:
                            err = abs(result - target) / target
                            if err < best_err:
                                best_err = err
                    except:
                        pass
        if best_err < g2_best_error:
            better_count += 1
    
    percentile = 100 * (1 - better_count / n_random)
    return g2_best, g2_best_error, percentile

# Test each constant with multiple formula templates
results = []

print("\n" + "=" * 90)
print("TEST 1: 1/α = 137.036")
print("=" * 90)

# Duality formula for α
def alpha_formula(a, b, c):
    lam = a * b
    C = c * np.pi**2
    disc = C**2 - 4*lam
    if disc < 0:
        return None
    x = (C - np.sqrt(disc)) / (2*lam)
    return 1/x if x > 0 else None

r = test_formula(alpha_formula, CONSTANTS['1/α'])
if r:
    print(f"G₂ prediction: {r[0]:.6f}, error: {r[1]:.2e}, percentile: {r[2]:.1f}%")
    results.append(('1/α', r[0], CONSTANTS['1/α'], r[1], r[2]))

print("\n" + "=" * 90)
print("TEST 2: m_p/m_e = 1836.15")
print("=" * 90)

# The claimed formula is 6π⁵
print(f"Claimed formula: 6π⁵ = {6 * np.pi**5:.4f}")
print(f"Experimental: {CONSTANTS['m_p/m_e']:.4f}")
print(f"Error: {abs(6*np.pi**5 - CONSTANTS['m_p/m_e'])/CONSTANTS['m_p/m_e']:.2e}")

# Test if a*π^b formulas with G₂ numbers are special
def mass_formula1(a, b, c):
    if b > 10:
        return None
    return a * np.pi**b

r = test_formula(mass_formula1, CONSTANTS['m_p/m_e'])
if r:
    print(f"G₂ best (a*π^b): {r[0]:.4f}, error: {r[1]:.2e}, percentile: {r[2]:.1f}%")
    results.append(('m_p/m_e (a*π^b)', r[0], CONSTANTS['m_p/m_e'], r[1], r[2]))

# More general formula
def mass_formula2(a, b, c):
    return a * b * c

r = test_formula(mass_formula2, CONSTANTS['m_p/m_e'])
if r:
    print(f"G₂ best (a*b*c): {r[0]:.4f}, error: {r[1]:.2e}, percentile: {r[2]:.1f}%")
    results.append(('m_p/m_e (a*b*c)', r[0], CONSTANTS['m_p/m_e'], r[1], r[2]))

print("\n" + "=" * 90)
print("TEST 3: m_μ/m_e = 206.77")
print("=" * 90)

# Claimed: 12*17 + 2 = 206
print(f"Claimed formula: 12×17 + 2 = {12*17 + 2}")
print(f"Experimental: {CONSTANTS['m_μ/m_e']:.2f}")

def muon_formula(a, b, c):
    return a * b + c

r = test_formula(muon_formula, CONSTANTS['m_μ/m_e'])
if r:
    print(f"G₂ best (a*b+c): {r[0]:.4f}, error: {r[1]:.2e}, percentile: {r[2]:.1f}%")
    results.append(('m_μ/m_e', r[0], CONSTANTS['m_μ/m_e'], r[1], r[2]))

print("\n" + "=" * 90)
print("TEST 4: sin²θ_W = 0.2312")
print("=" * 90)

# Claimed: 3/13
print(f"Claimed formula: 3/13 = {3/13:.6f}")
print(f"Experimental: {CONSTANTS['sin²θ_W']:.6f}")

def weinberg_formula(a, b, c):
    if b == 0:
        return None
    return a / b

r = test_formula(weinberg_formula, CONSTANTS['sin²θ_W'])
if r:
    print(f"G₂ best (a/b): {r[0]:.6f}, error: {r[1]:.2e}, percentile: {r[2]:.1f}%")
    results.append(('sin²θ_W', r[0], CONSTANTS['sin²θ_W'], r[1], r[2]))

print("\n" + "=" * 90)
print("TEST 5: m_W/m_Z = 0.8815")
print("=" * 90)

# Claimed: √(10/13)
print(f"Claimed formula: √(10/13) = {np.sqrt(10/13):.6f}")
print(f"Experimental: {CONSTANTS['m_W/m_Z']:.6f}")

def wmz_formula(a, b, c):
    if b == 0:
        return None
    return np.sqrt(a / b)

r = test_formula(wmz_formula, CONSTANTS['m_W/m_Z'])
if r:
    print(f"G₂ best (√(a/b)): {r[0]:.6f}, error: {r[1]:.2e}, percentile: {r[2]:.1f}%")
    results.append(('m_W/m_Z', r[0], CONSTANTS['m_W/m_Z'], r[1], r[2]))

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 90)
print("SUMMARY: ARE G₂ PREDICTIONS STATISTICALLY SPECIAL?")
print("=" * 90)

print(f"\n{'Constant':<20} {'G₂ Pred':<12} {'Expt':<12} {'Rel Err':<12} {'%ile':<8} {'Verdict'}")
print("-" * 80)

special_count = 0
for name, pred, expt, err, pct in results:
    verdict = "SPECIAL" if pct > 95 else "GOOD" if pct > 80 else "AVG" if pct > 50 else "POOR"
    if pct > 95:
        special_count += 1
    print(f"{name:<20} {pred:<12.6f} {expt:<12.6f} {err:<12.2e} {pct:<8.1f} {verdict}")

print(f"""
================================================================================
                              OVERALL VERDICT
================================================================================

SPECIAL predictions (>95th percentile): {special_count}/{len(results)}

Expected by chance: {0.05 * len(results):.1f}
Observed: {special_count}

""")

if special_count > 2 * 0.05 * len(results):
    print("G₂ numbers are GENUINELY SPECIAL across multiple constants.")
    print("The probability of this by chance is very low.")
else:
    print("G₂ numbers show MIXED results.")
    print("Some constants match well, others could be chance.")
