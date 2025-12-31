#!/usr/bin/env python3
"""
HONEST NUMERICAL TEST OF THE G₂ FORMULAS
=========================================

Let's actually TEST these formulas rather than just claim they work.
No hand-waving. Just numbers.
"""

import numpy as np

print("=" * 70)
print("HONEST NUMERICAL TEST OF THE G₂ FORMULAS")
print("=" * 70)

# Experimental values
ALPHA_EXP = 0.0072973525693
SIN2_EXP = 0.23122  # at Z mass
MP_ME_EXP = 1836.15267343
LAMBDA_PLANCK = 1e-122  # approximate

print("\n" + "=" * 70)
print("TEST 1: THE α FORMULA")
print("=" * 70)

print("""
Formula: 1/α + 156α + √2α² + α³/2 = 14π²
""")

# Solve the equation
def solve_alpha_equation():
    """Newton's method to solve 1/α + 156α + √2α² + α³/2 = 14π²"""
    target = 14 * np.pi**2
    alpha = 0.01
    for _ in range(100):
        f = 1/alpha + 156*alpha + np.sqrt(2)*alpha**2 + alpha**3/2 - target
        fp = -1/alpha**2 + 156 + 2*np.sqrt(2)*alpha + 1.5*alpha**2
        alpha_new = alpha - f/fp
        if abs(alpha_new - alpha) < 1e-18:
            break
        alpha = alpha_new
    return alpha

alpha_pred = solve_alpha_equation()

print(f"Predicted α:    {alpha_pred:.15f}")
print(f"Experimental α: {ALPHA_EXP:.15f}")
print(f"Difference:     {alpha_pred - ALPHA_EXP:.2e}")
print(f"Error:          {abs(alpha_pred - ALPHA_EXP)/ALPHA_EXP * 100:.10f}%")

# Verify equation is satisfied
LHS = 1/alpha_pred + 156*alpha_pred + np.sqrt(2)*alpha_pred**2 + alpha_pred**3/2
RHS = 14 * np.pi**2
print(f"\nLHS = {LHS:.15f}")
print(f"RHS = {RHS:.15f}")
print(f"Match: {abs(LHS-RHS) < 1e-12}")

print("\n" + "=" * 70)
print("TEST 2: CONTRIBUTION OF EACH TERM")
print("=" * 70)

print("""
Let's see how much each term contributes:
""")

alpha = ALPHA_EXP
term_1_over_alpha = 1/alpha
term_156_alpha = 156 * alpha
term_sqrt2_alpha2 = np.sqrt(2) * alpha**2
term_alpha3_half = alpha**3 / 2
total = term_1_over_alpha + term_156_alpha + term_sqrt2_alpha2 + term_alpha3_half

print(f"1/α        = {term_1_over_alpha:.10f}  ({term_1_over_alpha/total*100:.4f}%)")
print(f"156α       = {term_156_alpha:.10f}  ({term_156_alpha/total*100:.4f}%)")
print(f"√2α²       = {term_sqrt2_alpha2:.10f}  ({term_sqrt2_alpha2/total*100:.6f}%)")
print(f"α³/2       = {term_alpha3_half:.15f}  ({term_alpha3_half/total*100:.10f}%)")
print(f"───────────────────────────────")
print(f"Total      = {total:.10f}")
print(f"14π²       = {14*np.pi**2:.10f}")

print("""
VERDICT: The α² and α³ terms contribute NOTHING.
         The formula is effectively: 1/α + 156α ≈ 14π²
""")

print("\n" + "=" * 70)
print("TEST 3: THE SIMPLIFIED FORMULA")
print("=" * 70)

print("""
If the α² and α³ terms don't matter, let's test the simplified version:
  1/α + 156α = 14π²
""")

def solve_simplified():
    """Solve 1/α + 156α = 14π²"""
    # 1/α + 156α = C where C = 14π²
    # Multiply by α: 1 + 156α² = Cα
    # 156α² - Cα + 1 = 0
    # α = (C ± √(C² - 4×156)) / (2×156)
    C = 14 * np.pi**2
    discriminant = C**2 - 4*156
    alpha1 = (C - np.sqrt(discriminant)) / (2*156)
    alpha2 = (C + np.sqrt(discriminant)) / (2*156)
    return alpha1, alpha2

alpha_simp1, alpha_simp2 = solve_simplified()
print(f"Simplified solution 1: α = {alpha_simp1:.15f} → 1/α = {1/alpha_simp1:.6f}")
print(f"Simplified solution 2: α = {alpha_simp2:.15f} → 1/α = {1/alpha_simp2:.6f}")
print(f"Experimental:          α = {ALPHA_EXP:.15f} → 1/α = {1/ALPHA_EXP:.6f}")
print(f"\nError (simplified): {abs(alpha_simp1 - ALPHA_EXP)/ALPHA_EXP * 100:.6f}%")
print(f"Error (full):       {abs(alpha_pred - ALPHA_EXP)/ALPHA_EXP * 100:.10f}%")

print("""
The simplified formula (no α², α³ terms) gives 0.000056% error.
The full formula gives 0.00000002% error.
The α² and α³ terms are FINE-TUNING, not essential.
""")

print("\n" + "=" * 70)
print("TEST 4: WHAT MAKES 14π² - 156α SPECIAL?")
print("=" * 70)

print(f"""
The real content of the formula:
  1/α ≈ 14π² - 156α
      ≈ {14*np.pi**2:.4f} - {156*ALPHA_EXP:.4f}
      ≈ {14*np.pi**2 - 156*ALPHA_EXP:.4f}

Breaking it down:
  14 = dim(G₂)
  π² = comes from geometry normalization
  156 = 12 × 13 = roots(G₂) × (roots(G₂) + 1)

So the formula says:
  1/α ≈ dim(G₂)×π² - roots×(roots+1)×α
""")

print("\n" + "=" * 70)
print("TEST 5: UNIQUENESS OF (N=12, r=2)")
print("=" * 70)

print("""
Testing: 1/α + N(N+1)α + √r·α² + α³/r = (N+r)π²

Which (N, r) gives α ≈ 1/137?
""")

def solve_general(N, r):
    """Solve: 1/α + N(N+1)α + √r·α² + α³/r = (N+r)π²"""
    target = (N + r) * np.pi**2
    alpha = 0.01
    for _ in range(100):
        f = 1/alpha + N*(N+1)*alpha + np.sqrt(r)*alpha**2 + alpha**3/r - target
        fp = -1/alpha**2 + N*(N+1) + 2*np.sqrt(r)*alpha + 3*alpha**2/r
        if abs(fp) < 1e-15:
            break
        alpha_new = alpha - f/fp
        if alpha_new <= 0:
            alpha_new = 0.001
        if abs(alpha_new - alpha) < 1e-15:
            break
        alpha = alpha_new
    return alpha

print(f"{'N':>4} {'r':>4} {'1/α':>12} {'Match 137?':>12}")
print("-" * 40)
for N in range(10, 16):
    for r in range(1, 4):
        alpha = solve_general(N, r)
        inv_alpha = 1/alpha
        match = "✓✓✓" if abs(inv_alpha - 137.036) < 0.01 else ""
        print(f"{N:4d} {r:4d} {inv_alpha:12.4f} {match:>12}")

print("""
RESULT: Only (N=12, r=2) gives 1/α ≈ 137.036
        And (N=11, r=3) is close at 137.21 but not exact.
""")

print("\n" + "=" * 70)
print("TEST 6: sin²θ_W FORMULA")
print("=" * 70)

print("""
Formula: sin²θ_W = 3/(13 - πα)
""")

sin2_pred = 3 / (13 - np.pi * ALPHA_EXP)
print(f"Predicted:    {sin2_pred:.8f}")
print(f"Experimental: {SIN2_EXP:.8f}")
print(f"Error:        {abs(sin2_pred - SIN2_EXP)/SIN2_EXP * 100:.4f}%")

print("\n" + "=" * 70)
print("TEST 7: m_p/m_e FORMULA")
print("=" * 70)

print("""
Formula: m_p/m_e = 12 × 153 = 1836
""")

mp_me_pred = 12 * 153
print(f"Predicted:    {mp_me_pred}")
print(f"Experimental: {MP_ME_EXP:.5f}")
print(f"Error:        {abs(mp_me_pred - MP_ME_EXP)/MP_ME_EXP * 100:.4f}%")

print("""
Note: 153 = 156 - 3 = roots(roots+1) - 3
      Where does the "3" come from? Maybe dim(SU(2))?
""")

print("\n" + "=" * 70)
print("TEST 8: COSMOLOGICAL CONSTANT")
print("=" * 70)

print("""
Formula: Λ ∝ α^57 (in Planck units)
""")

alpha_57 = ALPHA_EXP ** 57
print(f"α^57 = {alpha_57:.2e}")
print(f"Λ_obs ≈ 10^-122")
print(f"Ratio: α^57 / 10^-122 = {alpha_57 / 1e-122:.1f}")

print("""
Note: 57 = 4×dim(G₂) + 1 = 4×14 + 1
      This gives the right ORDER OF MAGNITUDE.
""")

print("\n" + "=" * 70)
print("TEST 9: WHAT IF WE CHANGE α² AND α³ COEFFICIENTS?")
print("=" * 70)

print("""
If the α² and α³ terms don't matter, changing their coefficients
should barely affect the answer. Let's test:
""")

def solve_modified(a2_coeff, a3_coeff):
    """Solve: 1/α + 156α + a2_coeff·α² + a3_coeff·α³ = 14π²"""
    target = 14 * np.pi**2
    alpha = 0.01
    for _ in range(100):
        f = 1/alpha + 156*alpha + a2_coeff*alpha**2 + a3_coeff*alpha**3 - target
        fp = -1/alpha**2 + 156 + 2*a2_coeff*alpha + 3*a3_coeff*alpha**2
        alpha_new = alpha - f/fp
        if abs(alpha_new - alpha) < 1e-15:
            break
        alpha = alpha_new
    return alpha

print(f"{'a2':>8} {'a3':>8} {'1/α':>15} {'Error':>15}")
print("-" * 50)
test_cases = [
    (np.sqrt(2), 0.5, "Original"),
    (0, 0, "No α², α³"),
    (1, 1, "a2=1, a3=1"),
    (10, 10, "a2=10, a3=10"),
    (100, 100, "a2=100, a3=100"),
    (-np.sqrt(2), -0.5, "Negative"),
]
for a2, a3, label in test_cases:
    alpha = solve_modified(a2, a3)
    error = abs(1/alpha - 1/ALPHA_EXP) / (1/ALPHA_EXP) * 100
    print(f"{a2:8.2f} {a3:8.2f} {1/alpha:15.8f} {error:14.8f}% ({label})")

print("""
CONCLUSION: The α² and α³ coefficients barely matter!
You could set them to almost anything and still get α ≈ 1/137.
The √2 and 1/2 are NOT constrained by the data.
""")

print("\n" + "=" * 70)
print("FINAL ASSESSMENT")
print("=" * 70)

print("""
╔══════════════════════════════════════════════════════════════════════╗
║                         HONEST VERDICT                                ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  WHAT ACTUALLY WORKS:                                                ║
║  ───────────────────                                                 ║
║  • The equation 1/α + 156α ≈ 14π² gives α to 0.00006% error         ║
║  • (N=12, r=2) really IS unique among reasonable combinations        ║
║  • 14 = dim(G₂), 156 = roots×(roots+1) = 12×13                      ║
║  • These numbers DO come from G₂ = Aut(Octonions)                   ║
║                                                                      ║
║  WHAT'S OVERFITTING:                                                 ║
║  ──────────────────                                                  ║
║  • The α² coefficient (√2) doesn't constrain anything               ║
║  • The α³ coefficient (1/2) doesn't constrain anything              ║
║  • You can set these to almost any value and get 137                ║
║                                                                      ║
║  THE REAL FORMULA:                                                   ║
║  ────────────────                                                    ║
║                                                                      ║
║      1/α ≈ dim(G₂)×π² - roots(roots+1)×α                            ║
║                                                                      ║
║      = 14π² - 156α                                                   ║
║      = 138.17 - 1.14                                                 ║
║      = 137.03                                                        ║
║                                                                      ║
║  THE QUESTION:                                                       ║
║  ────────────                                                        ║
║  Is this a genuine physical relationship, or numerology?             ║
║                                                                      ║
║  Evidence FOR:                                                       ║
║  • G₂ is forced by octonion structure (mathematics)                 ║
║  • 14π² is the natural scale from G₂ volume                         ║
║  • 156 = ℓ(ℓ+1) has angular momentum quantum number form            ║
║  • Uniqueness: only G₂ gives 137                                    ║
║                                                                      ║
║  Evidence AGAINST:                                                   ║
║  • No derivation of why π² appears with coefficient 14              ║
║  • No derivation of why 156α is the correction                      ║
║  • The α², α³ "derivations" are meaningless (terms don't matter)    ║
║                                                                      ║
║  VERDICT: INTRIGUING BUT INCOMPLETE                                  ║
║                                                                      ║
║  The G₂ connection might be real.                                    ║
║  But the full story is: 1/α ≈ 14π² - 156α                           ║
║  And we don't know WHY this holds.                                   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print("\n" + "=" * 70)
print("THE BOTTOM LINE")
print("=" * 70)

print(f"""
The simplified formula:
  1/α + 156α = 14π²

Has quadratic solutions:
  α = (14π² ± √((14π²)² - 624)) / 312

The physical solution:
  α = {alpha_simp1:.15f}
  1/α = {1/alpha_simp1:.10f}

Experimental:
  α = {ALPHA_EXP:.15f}
  1/α = {1/ALPHA_EXP:.10f}

Error: {abs(1/alpha_simp1 - 1/ALPHA_EXP):.6f} in 1/α
       {abs(alpha_simp1 - ALPHA_EXP)/ALPHA_EXP * 100:.6f}%

The formula:
  1/α + [roots(G₂)]×[roots(G₂)+1]×α = [dim(G₂)]×π²

with G₂ being the unique automorphism group of octonions.

WHETHER THIS IS PHYSICS OR COINCIDENCE REMAINS TO BE PROVEN.
But the numbers are what they are.
""")
