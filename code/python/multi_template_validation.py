#!/usr/bin/env python3
"""
MULTI-TEMPLATE VALIDATION

For each template:
1. What does G₂ predict?
2. How does G₂ rank against random number sets?
3. Is G₂ consistently special or just lucky once?
"""

import numpy as np
np.random.seed(42)

EXPERIMENTAL = 137.036

print("=" * 90)
print("MULTI-TEMPLATE VALIDATION: Is G₂ consistently special?")
print("=" * 90)

# G₂ numbers
G2_nums = [14, 12, 13, 2, 10, 6, 7]  # dim, |Δ|, |Δ|+1, rank, dim-4, |Δ|/2, dim/2

def test_template(template_fn, template_name, n_random=5000):
    """
    Test a template:
    1. Get G₂ prediction
    2. Get predictions from random number sets
    3. See where G₂ ranks
    """
    # G₂ prediction - try all combinations and take best
    g2_best = None
    g2_best_error = float('inf')
    g2_best_params = None
    
    for i, a in enumerate(G2_nums):
        for j, b in enumerate(G2_nums):
            for k, c in enumerate(G2_nums):
                try:
                    result = template_fn(a, b, c)
                    if result is not None and result > 0:
                        error = abs(result - EXPERIMENTAL)
                        if error < g2_best_error:
                            g2_best_error = error
                            g2_best = result
                            g2_best_params = (a, b, c)
                except:
                    pass
    
    # Random predictions
    random_errors = []
    for _ in range(n_random):
        nums = np.random.randint(2, 20, 7)
        best_error = float('inf')
        for i, a in enumerate(nums):
            for j, b in enumerate(nums):
                for k, c in enumerate(nums):
                    try:
                        result = template_fn(a, b, c)
                        if result is not None and result > 0:
                            error = abs(result - EXPERIMENTAL)
                            if error < best_error:
                                best_error = error
                    except:
                        pass
        if best_error < float('inf'):
            random_errors.append(best_error)
    
    # How does G₂ rank?
    if g2_best is None:
        return None
    
    better_than_g2 = sum(1 for e in random_errors if e < g2_best_error)
    percentile = 100 * (1 - better_than_g2 / len(random_errors)) if random_errors else 0
    
    return {
        'name': template_name,
        'g2_prediction': g2_best,
        'g2_error': g2_best_error,
        'g2_params': g2_best_params,
        'percentile': percentile,
        'better_count': better_than_g2,
        'total_random': len(random_errors)
    }

# Define templates (each takes 3 numbers, uses what it needs)
templates = [
    ("1/x + ab·x = c·π²", lambda a,b,c: (c*np.pi**2 - np.sqrt((c*np.pi**2)**2 - 4*a*b))/(2*a*b) if (c*np.pi**2)**2 > 4*a*b else None, lambda r: 1/r if r and r > 0 else None),
    ("1/x = a·π²/b", lambda a,b,c: a*np.pi**2/b if b != 0 else None, lambda r: r),
    ("1/x = a·b/c", lambda a,b,c: a*b/c if c != 0 else None, lambda r: r),
    ("1/x = (a²+b²)/c", lambda a,b,c: (a**2+b**2)/c if c != 0 else None, lambda r: r),
    ("1/x = a·π + b", lambda a,b,c: a*np.pi + b, lambda r: r),
    ("1/x = e^(a/b)", lambda a,b,c: np.exp(a/b) if b != 0 else None, lambda r: r),
    ("1/x = a·ln(b·c)", lambda a,b,c: a*np.log(b*c) if b*c > 0 else None, lambda r: r),
    ("1/x = a·b·c/π²", lambda a,b,c: a*b*c/np.pi**2, lambda r: r),
]

print(f"\nTesting {len(templates)} templates against {EXPERIMENTAL}...")
print("=" * 90)

results = []

for name, fn, transform in templates:
    def template_fn(a, b, c, f=fn, t=transform):
        r = f(a, b, c)
        return t(r) if r is not None else None
    
    result = test_template(template_fn, name, n_random=2000)
    if result:
        results.append(result)
        print(f"\n{name}:")
        print(f"  G₂ best: {result['g2_prediction']:.4f} (params: {result['g2_params']})")
        print(f"  Error: {result['g2_error']:.4f}")
        print(f"  G₂ percentile: {result['percentile']:.1f}% (beats {result['total_random'] - result['better_count']}/{result['total_random']} random)")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 90)
print("SUMMARY: G₂ PERFORMANCE ACROSS TEMPLATES")
print("=" * 90)

print(f"\n{'Template':<25} {'G₂ Prediction':<15} {'Error':<12} {'Percentile':<12} {'Verdict'}")
print("-" * 90)

for r in sorted(results, key=lambda x: x['g2_error']):
    verdict = "SPECIAL" if r['percentile'] > 95 else "GOOD" if r['percentile'] > 80 else "AVERAGE" if r['percentile'] > 50 else "POOR"
    print(f"{r['name']:<25} {r['g2_prediction']:<15.4f} {r['g2_error']:<12.4f} {r['percentile']:<12.1f} {verdict}")

# Count how many templates where G₂ is special
special_count = sum(1 for r in results if r['percentile'] > 95)
good_count = sum(1 for r in results if r['percentile'] > 80)

print(f"""
================================================================================
                              VERDICT
================================================================================

Templates where G₂ is SPECIAL (>95th percentile): {special_count}/{len(results)}
Templates where G₂ is GOOD (>80th percentile): {good_count}/{len(results)}

INTERPRETATION:
""")

if special_count >= len(results) // 2:
    print("G₂ numbers are consistently special across multiple templates.")
    print("This suggests the G₂ structure genuinely encodes something about α.")
elif special_count >= 2:
    print("G₂ numbers are special for some templates but not all.")
    print("This could be coincidence or could indicate a real but partial connection.")
else:
    print("G₂ numbers are NOT consistently special.")
    print("The good result for Template 1 may be coincidental.")

print("================================================================================")
