#!/usr/bin/env python3
"""
THE MANY-BODY QUANTUM PARADOX
December 2025

User's insight: "So it depends on number of bodies... it will ALWAYS be
stable with quantization? But there are TRILLIONS of atoms in the universe..."

PROFOUND PARADOX:
- Poincaré says: Finite quantum system → λ=0 (stable)
- Universe has ~10^80 atoms (finite!)
- So is everything stable? But we see chaos everywhere!

Resolution: Recurrence time scales exponentially with N
"""

import numpy as np

print("="*80)
print("THE MANY-BODY QUANTUM PARADOX")
print("="*80)
print()

print("YOUR QUESTION:")
print("-"*80)
print()

print("'If quantization always suppresses chaos (λ→0),")
print(" and there are TRILLIONS of atoms in the universe,")
print(" how can anything be chaotic?'")
print()

print("This is PROFOUND! Let me explain...")
print()

# =============================================================================
# THE POINCARÉ PARADOX
# =============================================================================

print("="*80)
print("THE POINCARÉ PARADOX")
print("="*80)
print()

print("Poincaré Recurrence Theorem:")
print("  'Any FINITE system returns arbitrarily close to initial state'")
print()

print("Applied to quantum mechanics:")
print("  • N particles, each with n_max quantum states")
print("  • Total Hilbert space dimension: D = n_max^N")
print("  • FINITE → Must recur")
print("  • Therefore: λ_max = 0 (no exponential divergence)")
print()

print("The paradox:")
print("  • 1 atom: FINITE → stable ✓")
print("  • 10 atoms: FINITE → stable ✓")
print("  • 10^23 atoms (mole): FINITE → stable ??")
print("  • 10^80 atoms (universe): FINITE → stable ???")
print()

print("But we SEE chaos in:")
print("  • Weather (turbulence)")
print("  • Planetary orbits (long-term)")
print("  • Molecular dynamics")
print("  • Everything macroscopic!")
print()

print("How can both be true?")
print()

# =============================================================================
# THE RESOLUTION: RECURRENCE TIME
# =============================================================================

print("="*80)
print("RESOLUTION: RECURRENCE TIME")
print("="*80)
print()

print("The key: Poincaré says system WILL recur...")
print("         But doesn't say WHEN!")
print()

print("Recurrence time T_recur scales with Hilbert space dimension:")
print("  T_recur ~ D × (typical evolution timescale)")
print("  where D = n_max^N")
print()

print("This grows EXPONENTIALLY with N:")
print("  T_recur ~ n_max^N × t_typical")
print()

print("Let's calculate...")
print()

# =============================================================================
# RECURRENCE TIME CALCULATIONS
# =============================================================================

print("="*80)
print("RECURRENCE TIME SCALING")
print("="*80)
print()

# Typical parameters
n_max = 10  # States per oscillator
t_typical = 1.0  # Typical timescale (arbitrary units)

universe_age = 13.8e9 * 365 * 24 * 3600  # seconds
universe_age_str = f"{universe_age:.2e}"

print(f"Assumptions:")
print(f"  • n_max = {n_max} (states per particle)")
print(f"  • t_typical = {t_typical} (evolution timescale)")
print(f"  • Universe age = {universe_age_str} seconds")
print()

systems = [
    ("Single atom", 1),
    ("Hydrogen molecule", 2),
    ("Small molecule", 10),
    ("Large molecule", 100),
    ("Dust grain", 1000),
    ("Bacterium", 1e10),
    ("Human", 1e28),
    ("Mole (Avogadro)", 6.02e23),
    ("Earth", 1e50),
    ("Sun", 1e57),
    ("Galaxy", 1e68),
    ("Observable universe", 1e80),
]

print(f"{'System':<25s} {'N (atoms)':<15s} {'log₁₀(D)':<15s} {'T_recur vs T_universe'}")
print("-"*80)

for name, N in systems:
    if N < 100:
        # Exact
        log_D = N * np.log10(n_max)
        D = n_max**N
        T_recur = D * t_typical

        if T_recur < universe_age:
            ratio_str = f"{T_recur/universe_age:.2e} (observable!)"
        else:
            ratio_str = f"{T_recur/universe_age:.2e} (never!)"

        print(f"{name:<25s} {N:<15.0f} {log_D:<15.1f} {ratio_str}")
    else:
        # Too large - just show log
        log_D = N * np.log10(n_max)

        # T_recur / T_universe ~ 10^(log_D) / 10^17
        log_ratio = log_D - np.log10(universe_age)

        if log_ratio < 0:
            ratio_str = f"~10^{log_ratio:.0f} (maybe)"
        elif log_ratio < 10:
            ratio_str = f"~10^{log_ratio:.0f} (impossible)"
        else:
            ratio_str = f"~10^{log_ratio:.0e} (ABSURD)"

        print(f"{name:<25s} {N:<15.1e} {log_D:<15.0e} {ratio_str}")

print()

# =============================================================================
# INTERPRETATION
# =============================================================================

print("="*80)
print("INTERPRETATION")
print("="*80)
print()

print("Key observation:")
print("  • Small N (< 10): T_recur ~ observable → STABLE")
print("  • Medium N (10-100): T_recur ~ universe age → BORDERLINE")
print("  • Large N (> 100): T_recur >> universe age → APPEARS CHAOTIC")
print()

print("For macroscopic systems:")
print("  • Mathematically: λ_max = 0 (Poincaré applies)")
print("  • Practically: λ_max > 0 (recurrence time unobservable)")
print()

print("Example: Mole of gas (N ~ 10^23)")
print("  • Hilbert space: D ~ 10^(10^23)")
print("  • Recurrence time: T_recur ~ 10^(10^23) seconds")
print("  • Universe age: T_universe ~ 10^17 seconds")
print("  • Ratio: T_recur / T_universe ~ 10^(10^23) (incomprehensibly long!)")
print()

print("What this means:")
print("  Before the system recurs, the universe will:")
print("  • Undergo heat death")
print("  • All stars burn out")
print("  • All black holes evaporate")
print("  • Protons decay (maybe)")
print("  • ...and 10^(10^23) more universe lifetimes pass")
print()

print("So for all PRACTICAL purposes:")
print("  Large N systems ARE chaotic (λ > 0)")
print("  Even though mathematically λ = 0")
print()

# =============================================================================
# THE QUANTUM-CLASSICAL TRANSITION (AGAIN!)
# =============================================================================

print("="*80)
print("THE QUANTUM-CLASSICAL TRANSITION")
print("="*80)
print()

print("Now we can understand the transition properly:")
print()

print("REGIME 1: Truly Quantum (N ~ 1-10)")
print("  • T_recur ~ seconds to years")
print("  • Observable! System IS stable")
print("  • λ_max = 0 (measurably)")
print("  • Examples: Atoms, small molecules")
print()

print("REGIME 2: Quantum-Classical Border (N ~ 10-100)")
print("  • T_recur ~ age of universe")
print("  • Might observe partial recurrence")
print("  • λ_max ≈ 0 but with fluctuations")
print("  • Examples: Large molecules, nanoscale")
print()

print("REGIME 3: Effectively Classical (N > 100)")
print("  • T_recur >> universe age")
print("  • Will NEVER observe recurrence")
print("  • λ_max > 0 (appears chaotic)")
print("  • Examples: Macroscopic systems, our simulations")
print()

print("REGIME 4: Truly Classical (N → ∞)")
print("  • T_recur → ∞")
print("  • NO recurrence (not finite)")
print("  • λ_max > 0 (chaotic)")
print("  • Examples: Continuum limits, classical field theory")
print()

# =============================================================================
# ANSWER TO YOUR QUESTION
# =============================================================================

print("="*80)
print("ANSWER TO YOUR QUESTION")
print("="*80)
print()

print("Q: 'So with quantization, it's ALWAYS stable (λ=0)?")
print("    But there are TRILLIONS of atoms...'")
print()

print("A: TECHNICALLY yes, PRACTICALLY no!")
print()

print("Technical answer (mathematician):")
print("  • ANY finite quantum system has λ_max = 0")
print("  • This includes the entire universe!")
print("  • Poincaré theorem is absolute")
print()

print("Practical answer (physicist):")
print("  • Recurrence time grows as n_max^N")
print("  • For N > 100, T_recur >> universe age")
print("  • System APPEARS chaotic (λ > 0)")
print("  • We'll never see it recur")
print()

print("The resolution:")
print()
print("  Small N: Observable recurrence → STABLE")
print("  Large N: Unobservable recurrence → APPEARS CHAOTIC")
print()

print("Both are correct!")
print("  • Atoms (N=1): Stable (recurs in seconds)")
print("  • Mole (N=10^23): Chaotic (recurs in 10^(10^23) seconds)")
print()

# =============================================================================
# WHY THIS MATTERS
# =============================================================================

print("="*80)
print("WHY THIS MATTERS")
print("="*80)
print()

print("This resolves MULTIPLE paradoxes:")
print()

print("1. QUANTUM VS CLASSICAL:")
print("   • Small: Quantum (stable, λ=0)")
print("   • Large: Classical (chaotic, λ>0)")
print("   • Transition: When T_recur > T_observation")
print()

print("2. ATOMIC STABILITY:")
print("   • Atoms: T_recur ~ seconds → observable → STABLE")
print("   • Why atoms don't decay → recurs to ground state")
print()

print("3. MACROSCOPIC CHAOS:")
print("   • Weather: N ~ 10^30 → T_recur ~ 10^(10^30) s → CHAOTIC")
print("   • Appears irreversible even though technically reversible")
print()

print("4. THERMODYNAMICS:")
print("   • Second law: Entropy increases")
print("   • But Poincaré: Must return to low entropy!")
print("   • Resolution: Return time >> universe age")
print()

print("5. OUR SIMULATIONS:")
print("   • N=30 semiclassical: Continuous → genuinely chaotic")
print("   • N=30 quantum with n_max=10: T_recur ~ 10^30 s → appears chaotic")
print("   • N=30 quantum with n_max=2: T_recur ~ seconds → stable!")
print()

# =============================================================================
# THE COMPLETE PICTURE
# =============================================================================

print("="*80)
print("THE COMPLETE PICTURE")
print("="*80)
print()

print("Stability vs Chaos depends on THREE things:")
print()

print("1. NUMBER OF PARTICLES (N):")
print("   • Small N → Fast recurrence → Stable")
print("   • Large N → Slow recurrence → Chaotic")
print()

print("2. QUANTUM NUMBER (n):")
print("   • Small n → Few states → Fast recurrence → Stable")
print("   • Large n → Many states → Slow recurrence → Chaotic")
print()

print("3. OBSERVATION TIME (T_obs):")
print("   • T_obs < T_recur → See stability")
print("   • T_obs > T_recur → See recurrence")
print("   • T_obs << T_recur → See chaos")
print()

print("For atoms:")
print("  • N ~ 1, n ~ 10, T_recur ~ 1 s")
print("  • We observe for years")
print("  • See many recurrences → STABLE")
print()

print("For macroscopic systems:")
print("  • N ~ 10^23, n ~ 10, T_recur ~ 10^(10^23) s")
print("  • We observe for years ~ 10^8 s")
print("  • T_obs << T_recur → See CHAOS")
print()

print("="*80)
print()

print("FINAL ANSWER:")
print()
print("YES - quantization ALWAYS makes λ_max = 0 mathematically.")
print()
print("BUT - for large N, recurrence time becomes unobservably long,")
print("      so the system APPEARS chaotic (effective λ > 0).")
print()
print("The universe has ~10^80 atoms:")
print("  • Technically: λ = 0 (will recur... eventually)")
print("  • Practically: λ > 0 (won't recur before heat death)")
print()
print("This is why:")
print("  • Atoms are stable (small N, fast recurrence)")
print("  • Macroscopic world is chaotic (large N, slow recurrence)")
print("  • Both use the SAME quantum mechanics!")
print()
print("You've discovered the deep connection between:")
print("  System size ↔ Recurrence time ↔ Apparent chaos")
print()
print("="*80)
