#!/usr/bin/env python3
"""
IMPROVING THE FITS - Finding more precise formulas
==================================================

Current issues:
- α: 0.0006% match (excellent)
- sin²θ_W: 0.2% off
- α_s: 0.3% off
- Others: various levels of fit

Let's find formulas that actually FIT the data precisely.
"""

import numpy as np
from scipy.optimize import fsolve, minimize
from itertools import product

print("=" * 80)
print("ANALYZING FIT DISCREPANCIES")
print("=" * 80)

# =============================================================================
# EXPERIMENTAL VALUES (high precision)
# =============================================================================
alpha_exp = 1/137.035999084  # Fine structure constant
sin2_W_exp = 0.23121  # Weak mixing angle (MS-bar at M_Z)
alpha_s_exp = 0.1179  # Strong coupling at M_Z (PDG 2022)
mp_me_exp = 1836.15267343
m_mu_me_exp = 206.7682830

# G₂ numbers
DIM = 14
RANK = 2
DELTA = 12  # |Δ| = roots

print(f"""
High-precision experimental values:
  α = 1/{1/alpha_exp:.9f}
  sin²θ_W = {sin2_W_exp:.5f}
  α_s(M_Z) = {alpha_s_exp:.4f}
  m_p/m_e = {mp_me_exp:.5f}
  m_μ/m_e = {m_mu_me_exp:.5f}
""")

# =============================================================================
# CHECK CURRENT FORMULAS
# =============================================================================
print("=" * 80)
print("CURRENT FORMULA ACCURACY")
print("=" * 80)

# α from quadratic
def alpha_quadratic(c1, c2):
    """Solve 1/α + c1*α = c2"""
    # α² * c1 - α * c2 + 1 = 0
    # α = (c2 - sqrt(c2² - 4*c1)) / (2*c1)
    discriminant = c2**2 - 4*c1
    if discriminant < 0:
        return None
    return (c2 - np.sqrt(discriminant)) / (2*c1)

alpha_pred = alpha_quadratic(156, 14*np.pi**2)
print(f"\nα formula: 1/α + 156α = 14π²")
print(f"  Predicted: 1/{1/alpha_pred:.9f}")
print(f"  Experimental: 1/{1/alpha_exp:.9f}")
print(f"  Difference: {abs(1/alpha_pred - 1/alpha_exp):.6f} ({abs(alpha_pred - alpha_exp)/alpha_exp*100:.6f}%)")

# sin²θ_W = 3/13
sin2_pred = 3/13
print(f"\nsin²θ_W formula: 3/13")
print(f"  Predicted: {sin2_pred:.6f}")
print(f"  Experimental: {sin2_W_exp:.6f}")
print(f"  Difference: {abs(sin2_pred - sin2_W_exp):.6f} ({abs(sin2_pred - sin2_W_exp)/sin2_W_exp*100:.3f}%)")

# α_s = 2/17
alpha_s_pred = 2/17
print(f"\nα_s formula: 2/17")
print(f"  Predicted: {alpha_s_pred:.6f}")
print(f"  Experimental: {alpha_s_exp:.6f}")
print(f"  Difference: {abs(alpha_s_pred - alpha_s_exp):.6f} ({abs(alpha_s_pred - alpha_s_exp)/alpha_s_exp*100:.3f}%)")

# =============================================================================
# SEARCH FOR BETTER sin²θ_W FORMULA
# =============================================================================
print("\n" + "=" * 80)
print("SEARCHING FOR BETTER sin²θ_W FORMULA")
print("=" * 80)

print(f"\nTarget: sin²θ_W = {sin2_W_exp:.6f}")
print(f"Current: 3/13 = {3/13:.6f} (off by {abs(3/13 - sin2_W_exp)/sin2_W_exp*100:.3f}%)")

# Try a/b for small integers
print("\nSearching a/b fractions:")
best_sin2 = []
for a in range(1, 50):
    for b in range(1, 200):
        val = a/b
        diff = abs(val - sin2_W_exp)
        if diff/sin2_W_exp < 0.001:  # Within 0.1%
            best_sin2.append((a, b, val, diff/sin2_W_exp*100))

best_sin2.sort(key=lambda x: x[3])
for a, b, val, diff in best_sin2[:10]:
    # Check if b has G₂ meaning
    note = ""
    if b == 13:
        note = "(|Δ|+1)"
    elif b == 56:
        note = "(4×14)"
    elif b % 13 == 0:
        note = f"({b//13}×13)"
    elif b % 14 == 0:
        note = f"({b//14}×14)"
    elif b % 12 == 0:
        note = f"({b//12}×12)"
    print(f"  {a}/{b} = {val:.6f} (diff: {diff:.4f}%) {note}")

# Try formulas with π
print("\nFormulas with π:")
pi_formulas = [
    ("3/(13 + π/100)", 3/(13 + np.pi/100)),
    ("3π/(13π + 1)", 3*np.pi/(13*np.pi + 1)),
    ("(3 + 0.01)/(13)", (3 + 0.01)/13),
    ("3/(13 - 0.015)", 3/(13 - 0.015)),
    ("(π-0.91)/13", (np.pi - 0.91)/13),
]
for name, val in pi_formulas:
    diff = abs(val - sin2_W_exp)/sin2_W_exp*100
    if diff < 0.1:
        print(f"  {name} = {val:.6f} (diff: {diff:.4f}%)")

# What correction is needed?
print(f"\nCorrection analysis:")
print(f"  3/13 = {3/13:.6f}")
print(f"  Experimental = {sin2_W_exp:.6f}")
print(f"  Ratio: {sin2_W_exp / (3/13):.6f}")
print(f"  Difference: {sin2_W_exp - 3/13:.6f}")

# Maybe the formula should be (3 + δ)/13 or 3/(13 - ε)
delta_needed = sin2_W_exp * 13 - 3
epsilon_needed = 13 - 3/sin2_W_exp
print(f"  If (3+δ)/13: δ = {delta_needed:.6f}")
print(f"  If 3/(13-ε): ε = {epsilon_needed:.6f}")

# =============================================================================
# SEARCH FOR BETTER α_s FORMULA
# =============================================================================
print("\n" + "=" * 80)
print("SEARCHING FOR BETTER α_s FORMULA")
print("=" * 80)

print(f"\nTarget: α_s = {alpha_s_exp:.5f}")
print(f"Current: 2/17 = {2/17:.5f} (off by {abs(2/17 - alpha_s_exp)/alpha_s_exp*100:.3f}%)")

# Search fractions
print("\nSearching a/b fractions:")
best_as = []
for a in range(1, 30):
    for b in range(1, 200):
        val = a/b
        diff = abs(val - alpha_s_exp)
        if diff/alpha_s_exp < 0.002:  # Within 0.2%
            best_as.append((a, b, val, diff/alpha_s_exp*100))

best_as.sort(key=lambda x: x[3])
for a, b, val, diff in best_as[:10]:
    note = ""
    if b == 17:
        note = "(dim+3)"
    elif b % 17 == 0:
        note = f"({b//17}×17)"
    elif b % 14 == 0:
        note = f"({b//14}×14)"
    print(f"  {a}/{b} = {val:.5f} (diff: {diff:.4f}%) {note}")

# =============================================================================
# UNIFIED APPROACH: QUADRATIC EQUATIONS
# =============================================================================
print("\n" + "=" * 80)
print("UNIFIED APPROACH: QUADRATIC EQUATIONS FOR ALL")
print("=" * 80)

print("""
For α, we have: 1/α + 156α = 14π²
             or: 156α² - 14π²α + 1 = 0

What if ALL couplings satisfy similar quadratics?
""")

# For α: coefficients are 156 = 12×13, 14π², 1
# Let's see what quadratic sin²θ_W and α_s would need

def find_quadratic_coeffs(x, c1_options, c2_options):
    """Find c1, c2 such that c1*x² - c2*x + 1 ≈ 0"""
    results = []
    for c1 in c1_options:
        for c2 in c2_options:
            residual = abs(c1*x**2 - c2*x + 1)
            if residual < 0.01:
                results.append((c1, c2, residual))
    return sorted(results, key=lambda x: x[2])

# For sin²θ_W
print(f"\nFor sin²θ_W = {sin2_W_exp}:")
print("Looking for c1*x² - c2*x + 1 = 0")

# Check: what c2 do we need for various c1?
for c1 in [12, 13, 14, 26, 39, 52, 156]:
    # c1*x² - c2*x + 1 = 0  =>  c2 = (c1*x² + 1)/x
    c2_needed = (c1 * sin2_W_exp**2 + 1) / sin2_W_exp
    print(f"  c1 = {c1:3d}: c2 = {c2_needed:.4f}")

# For α_s
print(f"\nFor α_s = {alpha_s_exp}:")
for c1 in [12, 13, 14, 52, 68, 119]:
    c2_needed = (c1 * alpha_s_exp**2 + 1) / alpha_s_exp
    print(f"  c1 = {c1:3d}: c2 = {c2_needed:.4f}")

# =============================================================================
# TRY: SAME STRUCTURE AS α
# =============================================================================
print("\n" + "=" * 80)
print("SAME STRUCTURE AS α: 1/x + Ax = Bπ²")
print("=" * 80)

print("""
α satisfies: 1/α + 156α = 14π²

What A, B give sin²θ_W and α_s?
""")

# For sin²θ_W
x = sin2_W_exp
lhs = 1/x + 156*x  # Using same 156
print(f"\nFor sin²θ_W with A=156:")
print(f"  1/x + 156x = {lhs:.4f}")
print(f"  = {lhs/np.pi**2:.4f} × π²")

# What A gives nice B?
for B in [1, 2, 3, 4, 5, 6, 7, 12, 13, 14]:
    # 1/x + Ax = Bπ²
    # A = (Bπ² - 1/x) / x
    A_needed = (B * np.pi**2 - 1/x) / x
    print(f"  B = {B:2d}: A = {A_needed:.2f}")

# For α_s
x = alpha_s_exp
print(f"\nFor α_s:")
for B in [1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14]:
    A_needed = (B * np.pi**2 - 1/x) / x
    print(f"  B = {B:2d}: A = {A_needed:.2f}")

# =============================================================================
# RADIATIVE CORRECTIONS
# =============================================================================
print("\n" + "=" * 80)
print("COULD DIFFERENCES BE RADIATIVE CORRECTIONS?")
print("=" * 80)

print("""
The formulas might be "tree-level" predictions.
Quantum corrections could explain the small discrepancies.
""")

# sin²θ_W
tree_sin2 = 3/13
exp_sin2 = sin2_W_exp
correction_sin2 = (exp_sin2 - tree_sin2) / tree_sin2

print(f"\nsin²θ_W:")
print(f"  Tree-level (3/13): {tree_sin2:.6f}")
print(f"  Experimental: {exp_sin2:.6f}")
print(f"  Correction: {correction_sin2*100:.3f}%")
print(f"  δ = {exp_sin2 - tree_sin2:.6f}")

# If the correction is α/π or similar
alpha_pi = alpha_exp / np.pi
print(f"\n  α/π = {alpha_pi:.6f}")
print(f"  3α/π = {3*alpha_pi:.6f}")
print(f"  Ratio δ/(α/π) = {(exp_sin2 - tree_sin2)/alpha_pi:.3f}")

# α_s
tree_as = 2/17
exp_as = alpha_s_exp
correction_as = (exp_as - tree_as) / tree_as

print(f"\nα_s:")
print(f"  Tree-level (2/17): {tree_as:.6f}")
print(f"  Experimental: {exp_as:.6f}")
print(f"  Correction: {correction_as*100:.3f}%")

# =============================================================================
# COMBINED FORMULA SEARCH
# =============================================================================
print("\n" + "=" * 80)
print("SEARCHING FOR UNIFIED FORMULA")
print("=" * 80)

print("""
Looking for formulas of the form:
  1/x + A*x = B*π^n  or  x = a/(b + c*π^n)

where A, B, a, b, c are simple G₂ numbers.
""")

# Search over G₂ numbers
g2_nums = [1, 2, 3, 6, 7, 8, 12, 13, 14, 17, 52, 78, 133, 156, 248]

print("\nFor sin²θ_W = 0.23121:")
x = sin2_W_exp
best_formulas = []

for a in range(1, 20):
    for b in range(1, 100):
        # Simple: a/b
        val = a/b
        if abs(val - x)/x < 0.0001:  # 0.01%
            best_formulas.append((f"{a}/{b}", val, abs(val-x)/x*100))

        # With π: a/(b + π)
        val = a/(b + np.pi)
        if abs(val - x)/x < 0.0005:
            best_formulas.append((f"{a}/({b}+π)", val, abs(val-x)/x*100))

        # a/(b + π²)
        val = a/(b + np.pi**2)
        if abs(val - x)/x < 0.0005:
            best_formulas.append((f"{a}/({b}+π²)", val, abs(val-x)/x*100))

best_formulas.sort(key=lambda x: x[2])
for formula, val, diff in best_formulas[:15]:
    print(f"  {formula:20s} = {val:.6f} (diff: {diff:.5f}%)")

print("\nFor α_s = 0.1179:")
x = alpha_s_exp
best_formulas = []

for a in range(1, 20):
    for b in range(1, 150):
        val = a/b
        if abs(val - x)/x < 0.0001:
            best_formulas.append((f"{a}/{b}", val, abs(val-x)/x*100))

        val = a/(b + np.pi)
        if abs(val - x)/x < 0.0005:
            best_formulas.append((f"{a}/({b}+π)", val, abs(val-x)/x*100))

best_formulas.sort(key=lambda x: x[2])
for formula, val, diff in best_formulas[:15]:
    print(f"  {formula:20s} = {val:.6f} (diff: {diff:.5f}%)")

# =============================================================================
# THE REAL QUESTION
# =============================================================================
print("\n" + "=" * 80)
print("THE REAL QUESTION")
print("=" * 80)

print("""
Are we doing physics or numerology?

For α: The formula 1/α + 156α = 14π² has PHYSICAL meaning:
  - 156 = ℓ(ℓ+1) angular momentum eigenvalue from loop
  - 14 = dim(G₂) from the manifold
  - π² from heat kernel regularization

For sin²θ_W and α_s: We found 3/13 and 2/17 by searching,
but haven't derived them from loop calculations.

To make these rigorous, we need:
  1. Identify what LOOP INTEGRAL gives sin²θ_W
  2. Show the coefficient comes from G₂ geometry
  3. Same for α_s

The small discrepancies (0.2%, 0.3%) could be:
  - Higher-loop corrections
  - RG running effects
  - Our tree-level formulas need quantum corrections
""")

# =============================================================================
# CHECKING IF CORRECTIONS ARE CONSISTENT
# =============================================================================
print("\n" + "=" * 80)
print("CORRECTION PATTERN ANALYSIS")
print("=" * 80)

# All corrections as ratios
alpha_ratio = (1/137.035999084) / alpha_pred
sin2_ratio = sin2_W_exp / (3/13)
as_ratio = alpha_s_exp / (2/17)

print(f"\nRatio of experimental to predicted:")
print(f"  α:       {alpha_ratio:.8f} (essentially 1)")
print(f"  sin²θ_W: {sin2_ratio:.8f}")
print(f"  α_s:     {as_ratio:.8f}")

print(f"\nCorrection factors (exp/pred - 1):")
print(f"  α:       {(alpha_ratio - 1)*100:.6f}%")
print(f"  sin²θ_W: {(sin2_ratio - 1)*100:.4f}%")
print(f"  α_s:     {(as_ratio - 1)*100:.4f}%")

# Are the corrections related to α?
print(f"\nAre corrections proportional to α ≈ 1/137?")
print(f"  sin²θ_W correction / α = {(sin2_ratio - 1) / alpha_exp:.2f}")
print(f"  α_s correction / α = {(as_ratio - 1) / alpha_exp:.2f}")
