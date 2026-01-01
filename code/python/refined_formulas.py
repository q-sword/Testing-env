#!/usr/bin/env python3
"""
REFINED FORMULAS WITH π CORRECTIONS
====================================

Key discovery: 3/(13 - 1/(13π)) matches sin²θ_W to 0.002%
"""

import numpy as np

pi = np.pi

# Experimental values
alpha_exp = 1/137.035999084
sin2_W_exp = 0.23121
alpha_s_exp = 0.1179

print("=" * 80)
print("REFINED FORMULAS WITH π CORRECTIONS")
print("=" * 80)

# =============================================================================
# sin²θ_W = 3/(13 - 1/(13π))
# =============================================================================
print("\n" + "=" * 80)
print("sin²θ_W FORMULA")
print("=" * 80)

formula1 = 3/(13 - 1/(13*pi))
print(f"\nFormula: 3/(13 - 1/(13π))")
print(f"  = 3/(13 - 1/{13*pi:.4f})")
print(f"  = 3/{13 - 1/(13*pi):.7f}")
print(f"  = {formula1:.7f}")
print(f"\nExperimental: {sin2_W_exp}")
print(f"Difference: {abs(formula1 - sin2_W_exp)/sin2_W_exp*100:.5f}%")

# Interpretation
print(f"""
Interpretation:
  3/(13 - 1/(13π)) = 3/13 × 1/(1 - 1/(13²π))
                   ≈ 3/13 × (1 + 1/(13²π) + ...)
                   = 3/13 × (1 + 1/(169π))

  169 = 13² = (|Δ|+1)²

  The correction 1/(169π) = 1/((|Δ|+1)²π) is a "loop" correction!
""")

# Try variations
print("\nVariations:")
for a, b, c in [(3, 13, 13), (3, 13, 14), (3, 13, 12), (3, 13, 4)]:
    val = a/(b - 1/(c*pi))
    diff = abs(val - sin2_W_exp)/sin2_W_exp*100
    print(f"  {a}/({b} - 1/({c}π)) = {val:.7f} (diff: {diff:.5f}%)")

# =============================================================================
# SEARCH FOR α_s ANALOG
# =============================================================================
print("\n" + "=" * 80)
print("α_s FORMULA SEARCH")
print("=" * 80)

print("\nLooking for: 2/(17 - 1/(c×π)) form")
for c in range(1, 30):
    val = 2/(17 - 1/(c*pi))
    diff = abs(val - alpha_s_exp)/alpha_s_exp*100
    if diff < 0.1:
        print(f"  2/(17 - 1/({c}π)) = {val:.6f} (diff: {diff:.4f}%)")

print("\nLooking for: a/(b - 1/(c×π)) form")
for a in [2, 7, 8, 12, 14]:
    for b in range(10, 200):
        for c in range(1, 50):
            val = a/(b - 1/(c*pi))
            diff = abs(val - alpha_s_exp)/alpha_s_exp*100
            if diff < 0.01:
                print(f"  {a}/({b} - 1/({c}π)) = {val:.6f} (diff: {diff:.5f}%)")

print("\nLooking for: 2/(17 - 1/(c×π²)) form")
for c in range(1, 20):
    val = 2/(17 - 1/(c*pi**2))
    diff = abs(val - alpha_s_exp)/alpha_s_exp*100
    if diff < 0.1:
        print(f"  2/(17 - 1/({c}π²)) = {val:.6f} (diff: {diff:.4f}%)")

# =============================================================================
# UNIFIED PATTERN
# =============================================================================
print("\n" + "=" * 80)
print("UNIFIED PATTERN")
print("=" * 80)

# For sin²θ_W: 3/(13 - 1/(13π))
# The correction is 1/(13π) in the denominator

# What analogous correction works for α_s?
# If 2/(17 - δ) = α_s, then δ = 17 - 2/α_s
delta_as = 17 - 2/alpha_s_exp
print(f"\nFor α_s:")
print(f"  If 2/(17 - δ) = {alpha_s_exp}, then δ = {delta_as:.6f}")
print(f"  δ×π = {delta_as*pi:.5f}")
print(f"  1/δ = {1/delta_as:.4f}")
print(f"  1/(δπ) = {1/(delta_as*pi):.4f}")

# Try: δ = 1/(cπ) for various c
for c in range(1, 50):
    delta_try = 1/(c*pi)
    val = 2/(17 - delta_try)
    diff = abs(val - alpha_s_exp)/alpha_s_exp*100
    if diff < 0.05:
        print(f"  2/(17 - 1/({c}π)) = {val:.6f} (diff: {diff:.4f}%)")

# Try: δ = 1/(c×π²)
for c in range(1, 20):
    delta_try = 1/(c*pi**2)
    val = 2/(17 - delta_try)
    diff = abs(val - alpha_s_exp)/alpha_s_exp*100
    if diff < 0.05:
        print(f"  2/(17 - 1/({c}π²)) = {val:.6f} (diff: {diff:.4f}%)")

# =============================================================================
# THE RELATIONSHIP
# =============================================================================
print("\n" + "=" * 80)
print("THE RELATIONSHIP")
print("=" * 80)

# sin²θ_W correction: 1/(13π) in denominator, diff = 0.002%
# What if α_s has correction 1/(17×something)?

# For sin²θ_W: 13 appears twice (3/13 and 1/13π)
# For α_s: 17 appears once (2/17), what appears in correction?

print("""
Pattern:
  sin²θ_W = 3/(13 - 1/(13π))
          = (3/13) × 13/(13 - 1/(13π))
          = (3/13) × 1/(1 - 1/(13²π))

  So: sin²θ_W ≈ (3/13) × (1 + 1/(13²π))

  The correction involves 13² = 169 = (|Δ|+1)²

For α_s, if we try:
  α_s = (2/17) × (1 + 1/(17²π))
      = (2/17) × (1 + 1/(289π))
""")

# Test this
as_try = (2/17) * (1 + 1/(17**2 * pi))
print(f"\nTest: (2/17) × (1 + 1/(17²π))")
print(f"  = (2/17) × (1 + 1/{17**2 * pi:.2f})")
print(f"  = {as_try:.7f}")
print(f"  Experimental: {alpha_s_exp}")
print(f"  Difference: {abs(as_try - alpha_s_exp)/alpha_s_exp*100:.4f}%")

# Try with different multipliers
print("\nSearching for: (2/17) × (1 + k/(17²π)) form")
for k in np.arange(0.5, 10, 0.1):
    val = (2/17) * (1 + k/(17**2 * pi))
    diff = abs(val - alpha_s_exp)/alpha_s_exp*100
    if diff < 0.01:
        print(f"  k = {k:.2f}: α_s = {val:.7f} (diff: {diff:.5f}%)")

# =============================================================================
# FINAL FORMULAS
# =============================================================================
print("\n" + "=" * 80)
print("BEST FORMULAS FOUND")
print("=" * 80)

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         REFINED FORMULAS                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  α:                                                                          ║
║    1/α + 156α = 14π²                                                        ║
║    Match: 0.00006%                                                          ║
║                                                                              ║
║  sin²θ_W:                                                                    ║
║    sin²θ_W = 3/(13 - 1/(13π))                                               ║
║            = 3/13 × 1/(1 - 1/((|Δ|+1)²π))                                   ║
""")

# Compute and display
sin2_pred = 3/(13 - 1/(13*pi))
diff_sin2 = abs(sin2_pred - sin2_W_exp)/sin2_W_exp*100
print(f"║    = {sin2_pred:.7f}")
print(f"║    Experimental: {sin2_W_exp}")
print(f"║    Match: {diff_sin2:.4f}%")
print("║")

# For α_s, find the best simple formula
best_as_formula = None
best_as_diff = 100
for c in range(1, 100):
    val = 2/(17 - 1/(c*pi))
    diff = abs(val - alpha_s_exp)/alpha_s_exp*100
    if diff < best_as_diff:
        best_as_diff = diff
        best_as_formula = (c, val)

c, val = best_as_formula
print(f"║  α_s:")
print(f"║    α_s = 2/(17 - 1/({c}π))")
print(f"║    = {val:.7f}")
print(f"║    Experimental: {alpha_s_exp}")
print(f"║    Match: {best_as_diff:.4f}%")
print("║")
print("╚══════════════════════════════════════════════════════════════════════════════╝")
