#!/usr/bin/env python3
"""
QUANTUM → CLASSICAL TRANSITION ANALYSIS
December 2025

Find the scaling regime where quantum regularization (ε) becomes negligible
and the system behaves classically.

Key question: When is ε << r_typical?
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

G = 1.0
HBAR = 1.0

print("="*80)
print("QUANTUM → CLASSICAL TRANSITION")
print("="*80)
print()

# =============================================================================
# CURRENT N=30 SYSTEM
# =============================================================================

print("CURRENT SYSTEM (N=30, 'atomic' units):")
print("-"*80)

N = 30
m = 1.0  # Particle mass
v_rms = 0.2881
epsilon = HBAR / (m * v_rms)
r_rms = np.sqrt(3 * 0.5**2)  # From IC: pos ~ randn * 0.5

print(f"  Mass per particle: m = {m}")
print(f"  Velocity scale: v_rms = {v_rms:.4f}")
print(f"  Position scale: r_rms = {r_rms:.4f}")
print(f"  Quantum length: ε = ℏ/(m·v) = {epsilon:.4f}")
print()

ratio = epsilon / r_rms
print(f"  Ratio: ε/r_rms = {ratio:.3f}")
print()

if ratio > 0.1:
    print("  → QUANTUM REGIME: ε is comparable to typical separation")
    print("  → Quantum regularization significantly affects dynamics")
elif ratio > 0.01:
    print("  → TRANSITION REGIME: ε matters but not dominant")
else:
    print("  → CLASSICAL REGIME: ε negligible compared to separations")

print()

# =============================================================================
# SCALING LAWS
# =============================================================================

print("="*80)
print("SCALING TO MACROSCOPIC REGIME")
print("="*80)
print()

print("Quantum length scale: ε = ℏ/(m·v_rms)")
print()
print("To make ε negligible (ε << r), we can:")
print("  1. Increase mass: m → M·m  ⇒  ε → ε/M")
print("  2. Increase velocity: v → V·v  ⇒  ε → ε/V")
print("  3. Both: (m,v) → (M·m, V·v)  ⇒  ε → ε/(M·V)")
print()

print("For classical limit, want ε/r_typical < 0.01")
print(f"Currently: ε/r = {ratio:.3f}")
print(f"Need scaling factor: M·V > {ratio/0.01:.1f}×")
print()

# =============================================================================
# MACROSCOPIC SCENARIOS
# =============================================================================

print("="*80)
print("EXAMPLE MACROSCOPIC SCALINGS")
print("="*80)
print()

scenarios = [
    ("Current (quantum)", 1, 1),
    ("10× heavier", 10, 1),
    ("10× faster", 1, 10),
    ("10× both", 10, 10),
    ("100× heavier", 100, 1),
    ("100× faster", 1, 100),
    ("Planetary (10³)", 1000, 1),
    ("Stellar (10⁶)", 1e6, 1),
    ("Galactic (10¹²)", 1e12, 1),
]

print(f"{'Scenario':<25s} {'M':<10s} {'V':<10s} {'ε':<12s} {'ε/r':<12s} {'Regime'}")
print("-"*80)

for name, M, V in scenarios:
    eps_scaled = epsilon / (M * V)
    ratio_scaled = eps_scaled / r_rms

    if ratio_scaled > 0.1:
        regime = "Quantum"
    elif ratio_scaled > 0.01:
        regime = "Transition"
    elif ratio_scaled > 1e-6:
        regime = "Classical"
    else:
        regime = "Pure classical"

    print(f"{name:<25s} {M:<10.0e} {V:<10.0e} {eps_scaled:<12.2e} {ratio_scaled:<12.2e} {regime}")

print()

# =============================================================================
# PHYSICAL INTERPRETATION
# =============================================================================

print("="*80)
print("PHYSICAL SYSTEMS")
print("="*80)
print()

print("Mapping to real physical systems:")
print()

# Atoms/molecules (quantum)
print("1. ATOMS/MOLECULES (Quantum regime):")
print("   • Electrons in atoms: m ~ 10⁻³⁰ kg, v ~ 10⁶ m/s")
print("   • ℏ/(mv) ~ 10⁻¹⁰ m (atomic size!)")
print("   • ε/r ~ 1 → Fully quantum")
print()

# Dust grains (transition)
print("2. DUST GRAINS (Transition regime):")
print("   • Micron-sized grains: m ~ 10⁻¹⁵ kg")
print("   • Thermal motion: v ~ 100 m/s")
print("   • ℏ/(mv) ~ 10⁻¹⁸ m << grain size")
print("   • ε/r ~ 10⁻³ → Nearly classical")
print()

# Planets (classical)
print("3. PLANETS (Classical regime):")
print("   • Earth mass: m ~ 6×10²⁴ kg")
print("   • Orbital velocity: v ~ 30 km/s")
print("   • ℏ/(mv) ~ 10⁻⁶³ m (unimaginably small!)")
print("   • ε/r → 0 → Pure classical gravity")
print()

# Stars (classical)
print("4. STARS (Classical regime):")
print("   • Solar mass: m ~ 2×10³⁰ kg")
print("   • Velocity: v ~ 100 km/s")
print("   • ℏ/(mv) ~ 10⁻⁶⁶ m")
print("   • ε/r → 0 → Pure classical")
print()

# Galaxies (classical)
print("5. GALAXIES (Classical regime):")
print("   • Galaxy mass: m ~ 10⁴² kg")
print("   • Velocity: v ~ 200 km/s")
print("   • ℏ/(mv) ~ 10⁻⁷⁸ m")
print("   • ε/r → 0 → Pure classical")
print()

# =============================================================================
# COMPUTATIONAL STRATEGY
# =============================================================================

print("="*80)
print("TESTING QUANTUM → CLASSICAL TRANSITION")
print("="*80)
print()

print("Strategy to probe the transition:")
print()
print("1. Run N=30 simulations at different mass/velocity scales:")
print("   • Keep r_rms, E/particle, N constant")
print("   • Scale (m,v) → (M·m, V·v) with M·V = {1, 3, 10, 30, 100}")
print("   • Measure λ_max as function of ε/r")
print()

print("2. Expected behavior:")
print("   • Quantum regime (ε/r ~ 1): λ_max stabilized by regularization")
print("   • Transition (ε/r ~ 0.01-0.1): λ_max starts increasing")
print("   • Classical (ε/r << 0.01): λ_max reaches classical N-body value")
print()

print("3. Hypothesis:")
print("   • Small ε/r → More violent chaos (classical singularities)")
print("   • Large ε/r → Milder chaos (quantum smoothing)")
print("   • Critical ε/r ~ 0.1 where transition occurs")
print()

# =============================================================================
# GENERATE TEST MATRIX
# =============================================================================

print("="*80)
print("PROPOSED TEST MATRIX")
print("="*80)
print()

print("Run N=30 Lyapunov calculations with scaling factors:")
print()

mass_scales = np.array([1, 3, 10, 30, 100, 300])
v_base = v_rms
m_base = m
r_base = r_rms

print(f"{'M_scale':<10s} {'v_rms':<10s} {'ε':<12s} {'ε/r':<12s} {'Expected regime':<20s} {'Est. λ_max'}")
print("-"*80)

for M in mass_scales:
    v_scaled = v_base  # Keep velocity same
    eps_scaled = HBAR / (M * m_base * v_scaled)
    ratio_scaled = eps_scaled / r_base

    # Rough estimate: classical N-body has λ ~ 0.1-1.0
    # Quantum smoothing reduces it
    if ratio_scaled > 0.5:
        regime = "Quantum"
        est_lambda = 0.03  # Like current
    elif ratio_scaled > 0.1:
        regime = "Transition"
        est_lambda = 0.1
    elif ratio_scaled > 0.01:
        regime = "Weakly classical"
        est_lambda = 0.3
    else:
        regime = "Classical"
        est_lambda = 0.5

    print(f"{M:<10.0f} {v_scaled:<10.4f} {eps_scaled:<12.2e} {ratio_scaled:<12.3f} {regime:<20s} ~{est_lambda:.2f}")

print()

# =============================================================================
# VISUALIZATION
# =============================================================================

# Create plot of ε/r vs expected behavior
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: ε/r vs M scaling
M_range = np.logspace(0, 3, 100)
eps_ratio = epsilon / (M_range * r_rms)

ax1.loglog(M_range, eps_ratio, 'b-', linewidth=2, label='ε/r')
ax1.axhline(0.1, color='orange', linestyle='--', label='Quantum threshold', alpha=0.7)
ax1.axhline(0.01, color='red', linestyle='--', label='Classical threshold', alpha=0.7)
ax1.fill_between(M_range, 0.1, 10, alpha=0.2, color='blue', label='Quantum regime')
ax1.fill_between(M_range, 0.01, 0.1, alpha=0.2, color='green', label='Transition')
ax1.fill_between(M_range, 1e-6, 0.01, alpha=0.2, color='red', label='Classical')

ax1.scatter([1], [epsilon/r_rms], s=200, c='black', marker='*',
            zorder=5, label='Current N=30')

ax1.set_xlabel('Mass scaling factor M', fontsize=12)
ax1.set_ylabel('ε/r (quantum importance)', fontsize=12)
ax1.set_title('Quantum vs Classical Regimes', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)
ax1.set_ylim(1e-6, 10)

# Right: Schematic of expected λ_max vs ε/r
eps_r_range = np.logspace(-3, 0, 100)
# Model: λ increases as ε/r decreases (more classical → more violent chaos)
lambda_model = 0.02 + 0.5 * (1 - np.tanh(5 * (np.log10(eps_r_range) + 1)))

ax2.semilogx(eps_r_range, lambda_model, 'r-', linewidth=2)
ax2.axvline(0.1, color='orange', linestyle='--', alpha=0.7)
ax2.axvline(0.01, color='red', linestyle='--', alpha=0.7)
ax2.scatter([epsilon/r_rms], [0.032], s=200, c='black', marker='*',
            zorder=5, label='Measured (N=30)')

ax2.fill_between([0.1, 10], 0, 1, alpha=0.2, color='blue')
ax2.fill_between([0.01, 0.1], 0, 1, alpha=0.2, color='green')
ax2.fill_between([1e-6, 0.01], 0, 1, alpha=0.2, color='red')

ax2.set_xlabel('ε/r (quantum importance)', fontsize=12)
ax2.set_ylabel('λ_max (chaos strength)', fontsize=12)
ax2.set_title('Expected Chaos vs Quantum Scale (Model)', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.set_xlim(1e-3, 2)
ax2.set_ylim(0, 0.6)

plt.tight_layout()
plt.savefig('/tmp/quantum_classical_transition.png', dpi=150, bbox_inches='tight')
print("="*80)
print()
print("Plot saved: /tmp/quantum_classical_transition.png")
print()

print("="*80)
print("SUMMARY")
print("="*80)
print()
print(f"Current N=30 system: ε/r = {epsilon/r_rms:.3f}")
print()
print("To reach classical regime (ε/r < 0.01):")
print(f"  • Need mass scaling M > {epsilon/(0.01*r_rms):.0f}×")
print(f"  • Or velocity scaling V > {epsilon/(0.01*r_rms):.0f}×")
print(f"  • Or combination M·V > {epsilon/(0.01*r_rms):.0f}")
print()
print("Prediction:")
print("  • As M increases (heavier particles), ε → 0")
print("  • System becomes more classically chaotic")
print("  • λ_max likely increases from 0.03 → 0.1-0.5")
print("  • More violent chaos, possible ejections")
print()
print("Next step: Run N=30 with M = {1, 3, 10, 30, 100}")
print("           to measure λ_max(ε/r) empirically!")
print()
print("="*80)
