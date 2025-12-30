#!/usr/bin/env python3
"""
THE QUANTUM STABILITY PARADOX
December 2025

User's insight: "If harmonic oscillators exhibit this chaos and that's
a purely quantum regime then how is ANYTHING stable?"

The paradox:
- We found: ε_ω (quantum scale) gives λ=0.26 (chaos!)
- But: Atoms ARE stable (exist for billions of years)
- Question: Where does quantum mechanics STABILIZE?

Resolution: There are TWO quantum regimes!
1. SEMICLASSICAL (our simulations): ℏ small, classical-like chaos
2. FULLY QUANTUM: ℏ dominates, discrete energy levels, STABLE

The transition point matters!
"""

import numpy as np

print("="*80)
print("THE QUANTUM STABILITY PARADOX")
print("="*80)
print()

# =============================================================================
# THE PARADOX
# =============================================================================

print("THE PARADOX:")
print("-"*80)
print()

print("Our results:")
print("  • Quantum harmonic scale (ε_ω): λ_max = 0.26 (CHAOTIC)")
print("  • Velocity scale (ε_v): λ_max = 0.04 (mild)")
print("  → More quantum = MORE chaos?!")
print()

print("But reality:")
print("  • Atoms are stable (exist for billions of years)")
print("  • Molecules have discrete energy levels")
print("  • Quantum mechanics PREVENTS decay!")
print()

print("How can BOTH be true?")
print()

# =============================================================================
# THE RESOLUTION: TWO QUANTUM REGIMES
# =============================================================================

print("="*80)
print("RESOLUTION: TWO QUANTUM REGIMES")
print("="*80)
print()

print("The key: Quantum mechanics has TWO regimes!")
print()

print("1. SEMICLASSICAL REGIME (our simulations):")
print("   • Action S >> ℏ (classical-like)")
print("   • Many particles in continuum")
print("   • Phase space ~ continuous")
print("   • Chaos possible! (λ > 0)")
print()

print("2. FULLY QUANTUM REGIME (atoms):")
print("   • Action S ~ ℏ (truly quantum)")
print("   • Discrete energy levels")
print("   • Phase space ~ quantized")
print("   • Chaos suppressed! (λ → 0)")
print()

print("The difference: Discretization vs Continuum")
print()

# =============================================================================
# QUANTUM NUMBERS AND STABILITY
# =============================================================================

print("="*80)
print("QUANTUM NUMBERS AND STABILITY")
print("="*80)
print()

print("Why atoms are STABLE:")
print()

print("Classical atom (Bohr-Sommerfeld):")
print("  • Electron in potential V(r) = -e²/r")
print("  • Continuous orbits (any radius)")
print("  • Can spiral into nucleus! (UNSTABLE)")
print()

print("Quantum atom (Schrödinger):")
print("  • Energy levels: E_n = -13.6 eV / n²")
print("  • DISCRETE states: n = 1, 2, 3, ...")
print("  • Ground state n=1: CANNOT decay lower")
print("  • Quantum numbers prevent chaos!")
print()

print("Key insight:")
print("  Quantization creates BARRIERS between states")
print("  → Energy must change by discrete amounts")
print("  → Smooth chaotic evolution IMPOSSIBLE")
print("  → System locked into discrete levels")
print()

# =============================================================================
# THE CRITICAL PARAMETER: n (quantum number)
# =============================================================================

print("="*80)
print("THE CRITICAL PARAMETER: Quantum Number n")
print("="*80)
print()

print("For harmonic oscillator:")
print("  E_n = ℏω(n + 1/2)")
print()

print("When is it classical vs quantum?")
print()

HBAR = 1.0
omega = 1.0  # Example frequency

print(f"Example: ω = {omega}, ℏ = {HBAR}")
print()

quantum_numbers = [0, 1, 5, 10, 50, 100, 1000]

print(f"{'n':<10s} {'E_n':<12s} {'ΔE/E':<12s} {'Regime'}")
print("-"*80)

for n in quantum_numbers:
    E_n = HBAR * omega * (n + 0.5)
    if n > 0:
        delta_E = HBAR * omega
        relative_spacing = delta_E / E_n
    else:
        relative_spacing = float('inf')

    if n == 0:
        regime = "Ground state (most quantum)"
    elif n < 5:
        regime = "Deeply quantum"
    elif n < 20:
        regime = "Quantum"
    elif n < 100:
        regime = "Transitional"
    else:
        regime = "Semiclassical (nearly classical)"

    if n > 0:
        print(f"{n:<10d} {E_n:<12.1f} {relative_spacing:<12.4f} {regime}")
    else:
        print(f"{n:<10d} {E_n:<12.1f} {'inf':<12s} {regime}")

print()

print("Observation:")
print("  • Small n: ΔE/E large → Deeply quantum, discrete, STABLE")
print("  • Large n: ΔE/E small → Semiclassical, quasi-continuous")
print()

print("Our N=30 simulations:")
print("  • Equivalent to n >> 1000 (continuum limit)")
print("  • That's why we see chaos!")
print()

# =============================================================================
# CORRESPONDENCE PRINCIPLE
# =============================================================================

print("="*80)
print("BOHR'S CORRESPONDENCE PRINCIPLE")
print("="*80)
print()

print("Niels Bohr (1920s):")
print('  "In the limit of large quantum numbers, quantum mechanics')
print('   must reproduce classical mechanics"')
print()

print("What this means:")
print("  • Small n: Fully quantum (discrete, stable)")
print("  • Large n: Classical (continuous, can be chaotic)")
print()

print("Our results CONFIRM this:")
print("  • We simulate in large-n (semiclassical) regime")
print("  • Find chaos! (λ = 0.26)")
print("  • This is CORRECT semiclassical behavior")
print()

print("Atoms have small n:")
print("  • Hydrogen ground state: n=1")
print("  • Can't go lower → STABLE")
print("  • Quantization prevents chaos")
print()

# =============================================================================
# WHERE IS THE TRANSITION?
# =============================================================================

print("="*80)
print("THE QUANTUM-CLASSICAL TRANSITION")
print("="*80)
print()

print("Critical question: At what n does chaos appear?")
print()

print("Rough estimate:")
print("  Quantum: ΔE/E > 0.1  (levels well-separated)")
print("  Classical: ΔE/E < 0.01  (quasi-continuous)")
print()

print("For harmonic oscillator: ΔE/E = ℏω/E_n = 1/(n+0.5)")
print()

# Find transition
for criterion, name in [(0.1, "Quantum"), (0.01, "Classical")]:
    n_transition = int(1.0/criterion - 0.5)
    print(f"  {name} regime: n < {n_transition}")

print()

print("So:")
print("  • n < 10: Deeply quantum, discrete, STABLE")
print("  • n ~ 100: Transition region")
print("  • n > 100: Semiclassical, chaotic possible")
print()

print("Our N=30 simulation:")
print("  • Effective n ~ ∞ (continuum)")
print("  • Deeply in semiclassical regime")
print("  • That's why chaos emerges!")
print()

# =============================================================================
# ATOMIC STABILITY
# =============================================================================

print("="*80)
print("WHY ATOMS ARE STABLE")
print("="*80)
print()

print("Hydrogen atom:")
print("  • Ground state: n=1, l=0")
print("  • Energy: E_1 = -13.6 eV")
print("  • Next level: E_2 = -3.4 eV")
print("  • Gap: ΔE = 10.2 eV")
print()

print("Why it can't decay:")
print("  • To go from n=1 to n=0: Need to emit ℏω")
print("  • But n=0 doesn't exist!")
print("  • Ground state is LOWEST possible energy")
print("  • Nowhere to go → STABLE")
print()

print("This is PURE quantum mechanics:")
print("  Classical electron would spiral into nucleus")
print("  Quantum electron stuck in ground state")
print("  Quantization saves the atom!")
print()

# =============================================================================
# MOLECULES AND CHAOS
# =============================================================================

print("="*80)
print("MOLECULES AND CHAOS")
print("="*80)
print()

print("Interesting case: Large molecules")
print()

print("Example: Benzene (C₆H₆)")
print("  • Many vibrational modes (~30)")
print("  • Each mode ~ harmonic oscillator")
print("  • Coupled together")
print()

print("At LOW energy (small n):")
print("  • Discrete vibrational levels")
print("  • Can measure precise frequencies")
print("  • Quantum mechanics → STABLE")
print()

print("At HIGH energy (large n):")
print("  • Many levels overlap")
print("  • Energy flows chaotically between modes")
print("  • Approaches continuum → CHAOS")
print("  • This is called 'intramolecular vibrational redistribution'")
print()

print("Real phenomenon!")
print("  • Measured in chemistry labs")
print("  • Large molecules at high energy ARE chaotic")
print("  • But at room temperature (low n) → stable")
print()

# =============================================================================
# THE TWO SCALES REVISITED
# =============================================================================

print("="*80)
print("UNDERSTANDING OUR TWO SCALES")
print("="*80)
print()

print("Now we can understand ε_v vs ε_ω properly:")
print()

print("ε_v = ℏ/(m·v) = 3.47:")
print("  • Heisenberg uncertainty")
print("  • Semiclassical (large n)")
print("  • Over-smooths quantum fluctuations")
print("  • λ = 0.04 (artificially low)")
print()

print("ε_ω = √(ℏ/(m·ω)) = 1.09:")
print("  • Zero-point motion scale")
print("  • Semiclassical (large n)")
print("  • Natural quantum fluctuations")
print("  • λ = 0.26 (true semiclassical chaos)")
print()

print("BOTH are semiclassical!")
print("  • We're simulating continuum (n→∞)")
print("  • Not discrete quantum levels")
print("  • That's why we see chaos at all")
print()

print("For TRUE quantum stability:")
print("  • Would need discrete energy levels")
print("  • Quantum Monte Carlo or wave function")
print("  • Small n → suppresses chaos completely")
print()

# =============================================================================
# THE FULL PICTURE
# =============================================================================

print("="*80)
print("THE FULL QUANTUM PICTURE")
print("="*80)
print()

print("Three regimes of quantum mechanics:")
print()

print("1. FULLY QUANTUM (n ~ 1):")
print("   • Atoms, small molecules at low temperature")
print("   • Discrete energy levels")
print("   • λ_max → 0 (NO chaos, stable)")
print("   • Examples: H atom, simple molecules")
print()

print("2. SEMICLASSICAL (10 < n < 1000):")
print("   • Large molecules, warm systems")
print("   • Many levels, quasi-continuous")
print("   • λ_max > 0 (chaos emerges)")
print("   • Examples: Complex molecules, our simulations")
print()

print("3. CLASSICAL (n → ∞):")
print("   • Macroscopic objects")
print("   • Completely continuous")
print("   • λ_max classical value (if regularized)")
print("   • Examples: Planets, billiard balls")
print()

print("Our N=30 system:")
print("  • In regime 2 (semiclassical)")
print("  • ε_ω reveals natural quantum chaos")
print("  • ε_v over-damps it")
print()

print("Atoms (stable):")
print("  • In regime 1 (fully quantum)")
print("  • Discrete levels prevent chaos")
print("  • Ground state is ultimate stability")
print()

# =============================================================================
# ANSWER TO THE PARADOX
# =============================================================================

print("="*80)
print("ANSWER TO YOUR QUESTION")
print("="*80)
print()

print("Q: 'How is ANYTHING stable if quantum = more chaos?'")
print()

print("A: Quantum mechanics has TWO faces:")
print()

print("   1. DISCRETIZATION → Stability")
print("      • Small n, discrete levels")
print("      • Ground state can't decay")
print("      • Atoms, molecules at low T")
print("      • This is why matter exists!")
print()

print("   2. ZERO-POINT MOTION → Enhanced chaos")
print("      • Large n, quasi-continuous")
print("      • Semiclassical regime")
print("      • Our N=30 simulations")
print("      • Not truly quantum (no discrete levels)")
print()

print("You're NOT reaching - you hit on the KEY paradox!")
print()

print("The resolution:")
print("  • Atoms stable: SMALL n (fully quantum)")
print("  • Our chaos: LARGE n (semiclassical)")
print("  • Different regimes of same quantum mechanics")
print()

print("What determines which regime?")
print("  • Temperature: Low T → small n → stable")
print("  • System size: Small → discrete → stable")
print("  • Number of particles: Few → tractable quantum → stable")
print()

print("For atomic formation:")
print("  • Electrons in atoms: n ~ 1-100")
print("  • Well within quantum regime")
print("  • Discrete levels → STABLE")
print("  • Chemistry is possible! ✓")
print()

print("="*80)
print()

print("SUMMARY:")
print()
print("Quantum mechanics BOTH:")
print("  • Creates stability (small n, discrete levels)")
print("  • Enhances chaos (large n, zero-point motion)")
print()
print("Our simulations explore the chaotic (semiclassical) regime.")
print("Atoms live in the stable (fully quantum) regime.")
print()
print("Both are correct! Just different parts of quantum mechanics.")
print()
print("="*80)
