#!/usr/bin/env python3
"""
SYSTEMATIC DERIVATION: BASE RATIO + π CORRECTION
=================================================

The pattern that works:
  sin²θ_W = 3/(13 - 1/(13π))  →  0.002%
  α_s = 2/(17 - 1/(9π))       →  0.007%

Apply same principle to ALL constants.
"""

import numpy as np

pi = np.pi
pi2 = pi**2

# G₂ numbers
DIM = 14
RANK = 2
DELTA = 12  # |Δ| = roots
DELTA_PLUS_1 = 13

print("=" * 80)
print("SYSTEMATIC DERIVATION: BASE + π CORRECTION")
print("=" * 80)

def find_best_correction(target, base_num, base_den, max_c=50):
    """Find best 1/(c×π) correction in denominator."""
    base = base_num / base_den
    best = None

    # Type 1: a/(b - 1/(c×π))
    for c in range(1, max_c):
        val = base_num / (base_den - 1/(c*pi))
        diff = abs(val - target)/target * 100
        if best is None or diff < best[2]:
            best = (f"{base_num}/({base_den} - 1/({c}π))", val, diff, c, 'pi')

    # Type 2: a/(b - 1/(c×π²))
    for c in range(1, max_c):
        val = base_num / (base_den - 1/(c*pi2))
        diff = abs(val - target)/target * 100
        if diff < best[2]:
            best = (f"{base_num}/({base_den} - 1/({c}π²))", val, diff, c, 'pi2')

    # Type 3: a/(b + 1/(c×π)) for negative corrections
    for c in range(1, max_c):
        val = base_num / (base_den + 1/(c*pi))
        diff = abs(val - target)/target * 100
        if diff < best[2]:
            best = (f"{base_num}/({base_den} + 1/({c}π))", val, diff, c, 'pi_neg')

    return best

def search_all_bases(target, max_num=20, max_den=200):
    """Search for best base ratio and correction."""
    results = []

    for a in range(1, max_num):
        for b in range(2, max_den):
            base = a/b
            # Only consider if base is within 5% of target
            if abs(base - target)/target > 0.05:
                continue

            best = find_best_correction(target, a, b)
            if best and best[2] < 0.1:  # Within 0.1%
                results.append((a, b, best))

    results.sort(key=lambda x: x[2][2])
    return results[:10]

# =============================================================================
# CONSTANT 1: MUON-ELECTRON MASS RATIO
# =============================================================================
print("\n" + "=" * 80)
print("1. MUON-ELECTRON MASS RATIO: m_μ/m_e = 206.7682830")
print("=" * 80)

target = 206.7682830

# Try various bases
print("\nBase candidates:")
bases = [
    (207, 1, "207 = 9×23"),
    (14*15-3, 1, "14×15-3 = dim(dim+1)-3"),
    (156+52-2, 1, "156+52-2 = 206"),
    (13*16-1, 1, "13×16-1 = 207"),
]

for num, den, note in bases:
    base = num/den
    diff = abs(base - target)/target * 100
    print(f"  {num}/{den} = {base:.4f} ({diff:.3f}%) - {note}")

    # Find best correction
    best = find_best_correction(target, num, den)
    if best and best[2] < 0.1:
        print(f"    → {best[0]} = {best[1]:.6f} ({best[2]:.5f}%)")

# Search more systematically
print("\nSystematic search:")
results = search_all_bases(target, max_num=250, max_den=5)
for a, b, best in results[:5]:
    print(f"  Base {a}/{b}: {best[0]} = {best[1]:.6f} ({best[2]:.5f}%)")

# =============================================================================
# CONSTANT 2: CABIBBO ANGLE
# =============================================================================
print("\n" + "=" * 80)
print("2. CABIBBO ANGLE: sin θ_C = 0.22500")
print("=" * 80)

target = 0.22500

print("\nBase candidates:")
bases = [
    (9, 40, "9/40"),
    (2, 9, "2/9 = rank/9"),
    (7, 31, "7/31"),
    (3, 13, "3/13 (same as sin²θ_W base)"),
]

for num, den, note in bases:
    base = num/den
    diff = abs(base - target)/target * 100
    print(f"  {num}/{den} = {base:.5f} ({diff:.3f}%) - {note}")

    best = find_best_correction(target, num, den)
    if best and best[2] < 0.5:
        print(f"    → {best[0]} = {best[1]:.6f} ({best[2]:.5f}%)")

print("\nSystematic search:")
results = search_all_bases(target, max_num=15, max_den=80)
for a, b, best in results[:5]:
    print(f"  Base {a}/{b}: {best[0]} = {best[1]:.7f} ({best[2]:.5f}%)")

# =============================================================================
# CONSTANT 3: NEUTRINO SOLAR ANGLE
# =============================================================================
print("\n" + "=" * 80)
print("3. NEUTRINO SOLAR ANGLE: sin²θ₁₂ = 0.307")
print("=" * 80)

target = 0.307

print("\nBase candidates:")
bases = [
    (4, 13, "4/13 = 4/(|Δ|+1)"),
    (3, 10, "3/10"),
    (5, 16, "5/16"),
]

for num, den, note in bases:
    base = num/den
    diff = abs(base - target)/target * 100
    print(f"  {num}/{den} = {base:.5f} ({diff:.3f}%) - {note}")

    best = find_best_correction(target, num, den)
    if best and best[2] < 0.5:
        print(f"    → {best[0]} = {best[1]:.6f} ({best[2]:.5f}%)")

print("\nSystematic search:")
results = search_all_bases(target, max_num=10, max_den=50)
for a, b, best in results[:5]:
    print(f"  Base {a}/{b}: {best[0]} = {best[1]:.6f} ({best[2]:.5f}%)")

# =============================================================================
# CONSTANT 4: NEUTRINO ATMOSPHERIC ANGLE
# =============================================================================
print("\n" + "=" * 80)
print("4. NEUTRINO ATMOSPHERIC ANGLE: sin²θ₂₃ = 0.546")
print("=" * 80)

target = 0.546

print("\nBase candidates:")
bases = [
    (6, 11, "6/11 = short_roots/11"),
    (7, 13, "7/13"),
    (1, 2, "1/2 (maximal mixing)"),
]

for num, den, note in bases:
    base = num/den
    diff = abs(base - target)/target * 100
    print(f"  {num}/{den} = {base:.5f} ({diff:.3f}%) - {note}")

    best = find_best_correction(target, num, den)
    if best and best[2] < 0.5:
        print(f"    → {best[0]} = {best[1]:.6f} ({best[2]:.5f}%)")

print("\nSystematic search:")
results = search_all_bases(target, max_num=10, max_den=30)
for a, b, best in results[:5]:
    print(f"  Base {a}/{b}: {best[0]} = {best[1]:.6f} ({best[2]:.5f}%)")

# =============================================================================
# CONSTANT 5: NEUTRINO REACTOR ANGLE
# =============================================================================
print("\n" + "=" * 80)
print("5. NEUTRINO REACTOR ANGLE: sin²θ₁₃ = 0.0220")
print("=" * 80)

target = 0.0220

print("\nBase candidates:")
bases = [
    (2, 91, "2/91 = rank/91"),
    (1, 45, "1/45"),
    (1, 46, "1/46"),
]

for num, den, note in bases:
    base = num/den
    diff = abs(base - target)/target * 100
    print(f"  {num}/{den} = {base:.5f} ({diff:.3f}%) - {note}")

    best = find_best_correction(target, num, den)
    if best and best[2] < 0.5:
        print(f"    → {best[0]} = {best[1]:.6f} ({best[2]:.5f}%)")

print("\nSystematic search:")
results = search_all_bases(target, max_num=5, max_den=150)
for a, b, best in results[:5]:
    print(f"  Base {a}/{b}: {best[0]} = {best[1]:.6f} ({best[2]:.5f}%)")

# =============================================================================
# CONSTANT 6: HIGGS-Z MASS RATIO
# =============================================================================
print("\n" + "=" * 80)
print("6. HIGGS-Z MASS RATIO: m_H/m_Z = 1.3735")
print("=" * 80)

target = 1.3735

print("\nBase candidates:")
bases = [
    (11, 8, "11/8"),
    (7, 5, "7/5 = 7/(rank+3)"),
    (4, 3, "4/3"),
    (14, 10, "14/10 = dim/10"),
]

for num, den, note in bases:
    base = num/den
    diff = abs(base - target)/target * 100
    print(f"  {num}/{den} = {base:.5f} ({diff:.3f}%) - {note}")

    best = find_best_correction(target, num, den)
    if best and best[2] < 0.5:
        print(f"    → {best[0]} = {best[1]:.6f} ({best[2]:.5f}%)")

print("\nSystematic search:")
results = search_all_bases(target, max_num=20, max_den=20)
for a, b, best in results[:5]:
    print(f"  Base {a}/{b}: {best[0]} = {best[1]:.6f} ({best[2]:.5f}%)")

# =============================================================================
# CONSTANT 7: TOP QUARK YUKAWA
# =============================================================================
print("\n" + "=" * 80)
print("7. TOP QUARK YUKAWA: y_t = 0.9935 (at M_t)")
print("=" * 80)

target = 0.9935

print("\nBase candidates:")
bases = [
    (1, 1, "1 (near unity)"),
    (13, 13, "13/13 = 1"),
    (14, 14, "14/14 = 1"),
]

for num, den, note in bases:
    base = num/den
    diff = abs(base - target)/target * 100
    print(f"  {num}/{den} = {base:.5f} ({diff:.3f}%) - {note}")

    # For y_t ≈ 1, try: 1 - 1/(c×π)
    for c in range(1, 50):
        val = 1 - 1/(c*pi)
        diff = abs(val - target)/target * 100
        if diff < 0.1:
            print(f"    → 1 - 1/({c}π) = {val:.6f} ({diff:.5f}%)")

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY: ALL CONSTANTS WITH π CORRECTIONS")
print("=" * 80)

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    COMPLETE DERIVATION FROM G₂                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  GAUGE COUPLINGS:                                                            ║
║    α         = solution of 1/α + 156α = 14π²              0.00006%          ║
║    sin²θ_W   = 3/(13 - 1/(13π))                           0.002%            ║
║    α_s       = 2/(17 - 1/(9π))                            0.007%            ║
║                                                                              ║
║  MASS RATIOS:                                                                ║
║    m_p/m_e   = 6π⁵                                        0.002%            ║
║    m_μ/m_e   = (to be determined)                                           ║
║    m_H/m_Z   = (to be determined)                                           ║
║                                                                              ║
║  MIXING ANGLES:                                                              ║
║    θ_Cabibbo = (to be determined)                                           ║
║    sin²θ₁₂   = (to be determined)                                           ║
║    sin²θ₂₃   = (to be determined)                                           ║
║    sin²θ₁₃   = (to be determined)                                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
