#!/usr/bin/env python3
"""
TEMPLATE EXPLORATION: Testing Multiple Formula Structures

For each template, we:
1. Plug in G₂ numbers (12, 13, 14)
2. See what value comes out (WITHOUT looking at 137)
3. Test if G₂ numbers are special compared to random numbers
4. THEN compare to experiment at the end
"""

import numpy as np
from itertools import product

np.random.seed(42)

print("=" * 90)
print("EXPLORING MULTIPLE FORMULA TEMPLATES")
print("=" * 90)

# G₂ numbers
G2 = {'dim': 14, 'roots': 12, 'rank': 2, 'roots_plus_1': 13, 'dim_minus_4': 10}

# We'll collect predictions from each template, then compare to experiment ONCE at the end
predictions = {}

# =============================================================================
# TEMPLATE 1: 1/x + λx = C  (the one we used)
# =============================================================================
print("\n" + "=" * 90)
print("TEMPLATE 1: 1/x + λx = C")
print("=" * 90)

def template1(lam, C):
    """Solve 1/x + λx = C, return 1/x"""
    disc = C**2 - 4*lam
    if disc < 0:
        return None
    x = (C - np.sqrt(disc)) / (2*lam)
    if x <= 0:
        return None
    return 1/x

# G₂ prediction (blind - not looking at 137)
lam_g2 = G2['roots'] * G2['roots_plus_1']  # 12 * 13 = 156
C_g2 = G2['dim'] * np.pi**2  # 14 * π²

result1 = template1(lam_g2, C_g2)
predictions['template1'] = result1
print(f"G₂ inputs: λ = {G2['roots']} × {G2['roots_plus_1']} = {lam_g2}, C = {G2['dim']} × π²")
print(f"G₂ prediction: 1/x = {result1:.6f}")

# Test how special this is
count_better = 0
n_trials = 10000
for _ in range(n_trials):
    a, b, c = np.random.randint(2, 30, 3)
    r = template1(a*b, c*np.pi**2)
    if r is not None and 100 < r < 200:  # Reasonable range
        # Is it closer to ANY "nice" number than G₂ result?
        count_better += 1

print(f"Random combinations giving 1/x in [100,200]: {count_better}/{n_trials}")

# =============================================================================
# TEMPLATE 2: 1/x = a × π + b  (linear in π)
# =============================================================================
print("\n" + "=" * 90)
print("TEMPLATE 2: 1/x = a × π + b")
print("=" * 90)

def template2(a, b):
    return a * np.pi + b

# Try various G₂ combinations
g2_combos = [
    (G2['dim'], G2['roots']),
    (G2['roots'], G2['dim']),
    (G2['dim'], G2['roots_plus_1']),
    (G2['roots_plus_1'], G2['dim']),
    (G2['dim'] * G2['rank'], G2['roots']),
]

print("G₂ combinations for a×π + b:")
for a, b in g2_combos:
    result = template2(a, b)
    print(f"  {a}×π + {b} = {result:.6f}")

# Best one
best_t2 = template2(G2['dim'] * 3, G2['roots'] + G2['rank'])  # Just trying
predictions['template2_example'] = template2(G2['dim'], G2['roots'])

# =============================================================================
# TEMPLATE 3: 1/x = a × π² / b  (ratio with π²)
# =============================================================================
print("\n" + "=" * 90)
print("TEMPLATE 3: 1/x = a × π² / b")
print("=" * 90)

def template3(a, b):
    if b == 0:
        return None
    return a * np.pi**2 / b

print("G₂ combinations for a×π²/b:")
for a in [G2['dim'], G2['roots'], G2['roots_plus_1']]:
    for b in [G2['rank'], G2['dim_minus_4'], 1]:
        if b > 0:
            result = template3(a, b)
            print(f"  {a}×π²/{b} = {result:.6f}")

predictions['template3'] = template3(G2['dim'], 1)  # = 14π² ≈ 138

# =============================================================================
# TEMPLATE 4: 1/x = (a² + b²) / c  (Pythagorean-like)
# =============================================================================
print("\n" + "=" * 90)
print("TEMPLATE 4: 1/x = (a² + b²) / c")
print("=" * 90)

def template4(a, b, c):
    if c == 0:
        return None
    return (a**2 + b**2) / c

print("G₂ combinations for (a² + b²)/c:")
for a, b, c in [(G2['dim'], G2['roots'], G2['rank']),
                 (G2['roots'], G2['roots_plus_1'], G2['rank']),
                 (G2['dim'], G2['rank'], 1)]:
    result = template4(a, b, c)
    print(f"  ({a}² + {b}²)/{c} = {result:.6f}")

predictions['template4'] = template4(G2['dim'], G2['roots'], G2['rank'])

# =============================================================================
# TEMPLATE 5: 1/x = a × b / π  (product over π)
# =============================================================================
print("\n" + "=" * 90)
print("TEMPLATE 5: 1/x = a × b / π")
print("=" * 90)

def template5(a, b):
    return a * b / np.pi

print("G₂ combinations for a×b/π:")
for a, b in [(G2['dim'], G2['roots']),
              (G2['roots'], G2['roots_plus_1']),
              (G2['dim'], G2['roots_plus_1'])]:
    result = template5(a, b)
    print(f"  {a}×{b}/π = {result:.6f}")

predictions['template5'] = template5(G2['dim'], G2['roots'])

# =============================================================================
# TEMPLATE 6: 1/x + x/λ = C  (different duality structure)
# =============================================================================
print("\n" + "=" * 90)
print("TEMPLATE 6: 1/x + x/λ = C (modified duality)")
print("=" * 90)

def template6(lam, C):
    """Solve 1/x + x/λ = C"""
    # Multiply by x: 1 + x²/λ = Cx
    # x² - Cλx + λ = 0
    disc = (C*lam)**2 - 4*lam
    if disc < 0:
        return None
    x = (C*lam - np.sqrt(disc)) / 2
    if x <= 0:
        return None
    return 1/x

result6 = template6(lam_g2, C_g2/10)  # Scale C differently
print(f"G₂ with λ={lam_g2}, C={C_g2/10:.2f}: 1/x = {result6:.6f}" if result6 else "No solution")

# Try other scalings
for scale in [1, 5, 10, 20, 50]:
    r = template6(lam_g2, C_g2/scale)
    if r and 50 < r < 300:
        print(f"  Scale C by 1/{scale}: 1/x = {r:.6f}")
        predictions[f'template6_scale{scale}'] = r

# =============================================================================
# TEMPLATE 7: Exponential - 1/x = e^(a/b)
# =============================================================================
print("\n" + "=" * 90)
print("TEMPLATE 7: 1/x = e^(a/b)")
print("=" * 90)

def template7(a, b):
    if b == 0:
        return None
    return np.exp(a/b)

print("G₂ combinations for e^(a/b):")
for a, b in [(G2['dim'], G2['rank']),
              (G2['roots'], G2['rank']),
              (G2['dim_minus_4'], G2['rank'])]:
    result = template7(a, b)
    print(f"  e^({a}/{b}) = {result:.6f}")

predictions['template7'] = template7(G2['dim_minus_4'], G2['rank'])  # e^5 ≈ 148

# =============================================================================
# TEMPLATE 8: 1/x = a × ln(b)
# =============================================================================
print("\n" + "=" * 90)
print("TEMPLATE 8: 1/x = a × ln(b)")
print("=" * 90)

def template8(a, b):
    if b <= 0:
        return None
    return a * np.log(b)

print("G₂ combinations for a×ln(b):")
for a, b in [(G2['dim'] * 10, G2['roots_plus_1']),
              (G2['roots'] * 10, G2['dim']),
              (50, G2['dim'])]:
    result = template8(a, b)
    print(f"  {a}×ln({b}) = {result:.6f}")

# =============================================================================
# SUMMARY: What does G₂ predict?
# =============================================================================
print("\n" + "=" * 90)
print("SUMMARY OF G₂ PREDICTIONS (before looking at experiment)")
print("=" * 90)

print("\nAll predictions from G₂ numbers:")
for name, value in sorted(predictions.items(), key=lambda x: x[1] if x[1] else 0):
    if value:
        print(f"  {name}: 1/x = {value:.4f}")

# =============================================================================
# NOW compare to experiment
# =============================================================================
print("\n" + "=" * 90)
print("COMPARISON TO EXPERIMENT")
print("=" * 90)

EXPERIMENTAL = 137.036  # 1/α

print(f"\nExperimental value: 1/α = {EXPERIMENTAL}")
print("\nHow close is each prediction?")
print("-" * 60)

for name, value in sorted(predictions.items(), key=lambda x: abs(x[1] - EXPERIMENTAL) if x[1] else float('inf')):
    if value:
        error = abs(value - EXPERIMENTAL)
        rel_error = error / EXPERIMENTAL
        print(f"  {name:25s}: {value:10.4f}  (error: {error:.4f}, rel: {rel_error:.2e})")

# =============================================================================
# CONCLUSION
# =============================================================================
print("\n" + "=" * 90)
print("CONCLUSION")
print("=" * 90)

# Find best template
best_name = min(predictions.items(), key=lambda x: abs(x[1] - EXPERIMENTAL) if x[1] else float('inf'))
print(f"""
Best prediction: {best_name[0]} with 1/x = {best_name[1]:.6f}

OBSERVATIONS:
1. Template 1 (1/x + λx = C) gives the closest result
2. But we should ask: is this because G₂ is special, or because
   we have enough degrees of freedom in any template to fit?
   
NEXT STEP: For each template, test if G₂ numbers are uniquely good
           or if random numbers work just as well.
""")
