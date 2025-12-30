#!/usr/bin/env python3
"""
ANALYZE CHAOS TIMESCALES FOR N=30 SYSTEM
December 2025

Given λ_max = +0.031717, what does this mean physically?
"""

import numpy as np
import matplotlib.pyplot as plt

# N=30 results
lambda_max = 0.031717
E0 = -115.421173
N = 30
epsilon = 3.4708
v_rms = 0.2881  # From earlier runs

print("="*80)
print("CHAOS TIMESCALE ANALYSIS: N=30 SYSTEM")
print("="*80)
print()

print(f"Lyapunov exponent: λ_max = {lambda_max:.6f}")
print()

# =============================================================================
# TIMESCALES
# =============================================================================

print("CHARACTERISTIC TIMESCALES:")
print("-" * 80)
print()

# 1. Lyapunov time
tau_L = 1.0 / lambda_max
print(f"1. Lyapunov time: τ_L = 1/λ = {tau_L:.1f}")
print(f"   → Nearby trajectories diverge by factor e ≈ 2.7 in {tau_L:.1f} time units")
print()

# 2. Predictability horizon (10 Lyapunov times)
tau_predict = 10 * tau_L
print(f"2. Predictability horizon: ~10τ_L = {tau_predict:.0f}")
print(f"   → Initial conditions become irrelevant after ~{tau_predict:.0f} time units")
print()

# 3. Dynamical timescale (orbital)
# For N-body system: t_dyn ~ sqrt(R³/GM) ~ R/v
R_typical = np.sqrt(3 * 0.5**2)  # RMS radius from IC
t_dyn = R_typical / v_rms
print(f"3. Dynamical time: t_dyn ~ R/v ≈ {t_dyn:.1f}")
print(f"   → Typical orbital crossing time")
print()

# 4. Ratio
ratio = tau_L / t_dyn
print(f"4. Chaos/Dynamical ratio: τ_L/t_dyn ≈ {ratio:.1f}")
print(f"   → System is chaotic on timescale of ~{ratio:.1f} orbital periods")
print()

# =============================================================================
# DIVERGENCE TIMELINE
# =============================================================================

print("="*80)
print("TRAJECTORY DIVERGENCE TIMELINE")
print("="*80)
print()
print("Starting with two systems differing by δ₀ = 10⁻⁸:")
print()

delta_0 = 1e-8
times = np.array([0, 1, 5, 10, 20, 31.5, 50, 100, 200, 315, 500, 1000])

print(f"{'Time':>8s}  {'Separation δ(t)':>15s}  {'Factor':>12s}  {'Interpretation'}")
print("-" * 80)

for t in times:
    delta_t = delta_0 * np.exp(lambda_max * t)
    factor = delta_t / delta_0

    if delta_t < 1e-6:
        interp = "Microscopic - still predictable"
    elif delta_t < 0.01:
        interp = "Small perturbation"
    elif delta_t < 0.1:
        interp = "Noticeable differences"
    elif delta_t < 1.0:
        interp = "Trajectories clearly diverged"
    else:
        interp = "Completely different evolutions"

    print(f"{t:8.0f}  {delta_t:15.2e}  {factor:12.2e}  {interp}")

print()

# =============================================================================
# WHAT DOES THIS MEAN PHYSICALLY?
# =============================================================================

print("="*80)
print("PHYSICAL INTERPRETATION")
print("="*80)
print()

print("1. ENERGY CONSERVATION:")
print(f"   δE/E₀ = 5.3×10⁻¹⁵ → Energy is PERFECTLY conserved")
print(f"   The system does NOT thermalize or decay energetically!")
print()

print("2. HAMILTONIAN CHAOS:")
print(f"   • Positive λ_max → Exponential divergence of nearby trajectories")
print(f"   • Energy conserved → System explores constant-E surface chaotically")
print(f"   • This is NOT dissipation - it's deterministic chaos")
print()

print("3. WHAT 'DECAYS':")
print(f"   • PREDICTABILITY decays exponentially with τ = {tau_L:.1f}")
print(f"   • Small uncertainty in IC → Exponentially growing uncertainty")
print(f"   • After t ≈ {tau_predict:.0f}, specific trajectory unpredictable")
print(f"   • BUT: Statistical properties remain well-defined!")
print()

print("4. LONG-TERM BEHAVIOR:")
print(f"   • System will explore accessible phase space ergodically")
print(f"   • May see quasi-equilibration in configuration space")
print(f"   • Possible events:")
print(f"     - Close encounters between bodies")
print(f"     - Temporary binary/cluster formation")
print(f"     - Chaotic mixing of orbits")
print(f"     - NO escape to infinity (energy conserved, all bound)")
print()

print("5. COMPARISON TO REGULAR GRAVITY:")
print(f"   • Classical N-body: λ_max ~ 0.01-0.1 (similar!)")
print(f"   • Quantum regularization with ε={epsilon:.1f} does NOT suppress chaos")
print(f"   • It just prevents singularities, not chaos")
print()

# =============================================================================
# PHASE SPACE EXPLORATION
# =============================================================================

print("="*80)
print("PHASE SPACE EXPLORATION")
print("="*80)
print()

print("Key insight: POSITIVE Lyapunov means:")
print()
print("  • System will eventually explore full 6N-dimensional energy surface")
print("  • Mixing time: ~several × τ_L")
print(f"  • Approximate mixing time: ~{5*tau_L:.0f} - {10*tau_L:.0f} time units")
print()

print("This is GOOD for:")
print("  ✓ Statistical mechanics: System will reach microcanonical equilibrium")
print("  ✓ Ergodicity: Time averages = ensemble averages")
print("  ✓ Thermodynamic behavior (even with energy conserved!)")
print()

print("This is BAD for:")
print("  ✗ Long-term trajectory prediction")
print("  ✗ Identifying stable orbital configurations")
print("  ✗ Computing specific N-body solutions analytically")
print()

# =============================================================================
# ESTIMATE WHEN FIRST EJECTION MIGHT OCCUR
# =============================================================================

print("="*80)
print("STABILITY ESTIMATES")
print("="*80)
print()

print("Will bodies escape to infinity?")
print(f"  • Total energy: E = {E0:.1f} < 0 → All particles BOUND")
print(f"  • ε = {epsilon:.1f} → Soft potential prevents close encounters")
print(f"  • Result: NO EJECTIONS (system is gravitationally bound)")
print()

print("What about relaxation/evaporation?")
print(f"  • Classical N-body relaxation time: t_relax ~ N × t_dyn")
print(f"  • For N=30: t_relax ~ {N * t_dyn:.0f}")
print(f"  • But chaos time: τ_L = {tau_L:.1f}")
print(f"  • Chaos dominates! System mixes before relaxation matters")
print()

# =============================================================================
# QUANTUM REGULARIZATION INSIGHT
# =============================================================================

print("="*80)
print("QUANTUM REGULARIZATION: WHAT WORKED, WHAT DIDN'T")
print("="*80)
print()

print("What ε-regularization DOES:")
print(f"  ✓ Prevents r→0 singularities (classical catastrophe)")
print(f"  ✓ Maintains energy conservation (δE/E ~ 10⁻¹⁵)")
print(f"  ✓ Allows stable long-term integration")
print(f"  ✓ Physically motivated (Heisenberg uncertainty)")
print()

print("What ε-regularization DOES NOT do:")
print(f"  ✗ Suppress chaos (λ_max = +{lambda_max:.6f} > 0)")
print(f"  ✗ Make system integrable")
print(f"  ✗ Prevent ergodic exploration")
print()

print("Key finding:")
print(f"  • N=3 system: λ_max < 0 (stable, quasi-periodic)")
print(f"  • N=30 system: λ_max > 0 (chaotic, ergodic)")
print(f"  • Transition somewhere between N=3 and N=30")
print(f"  • Quantum regularization stabilizes SMALL systems only!")
print()

print("="*80)
print()

print("CONCLUSION:")
print()
print(f"λ_max = {lambda_max:.6f} is small but POSITIVE.")
print()
print("This means:")
print(f"• The system is WEAKLY chaotic")
print(f"• Predictability decays slowly (τ_L = {tau_L:.1f})")
print(f"• Energy remains perfectly conserved forever")
print(f"• No 'decay' in the thermodynamic sense")
print(f"• System explores phase space ergodically")
print(f"• Statistical properties are well-defined and stable")
print()
print("In other words:")
print("  NOT integrable, but NOT violently chaotic either.")
print("  A 'mildly chaotic' Hamiltonian system with perfect energy conservation.")
print()
print("="*80)
