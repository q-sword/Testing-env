#!/usr/bin/env python3
"""
QUANTUM HARMONIC OSCILLATOR LENGTH SCALE
December 2025

Key insight: For harmonic systems, quantum mechanics gives a natural length
scale based on FREQUENCY, not velocity:

  ℓ_quantum = √(ℏ/(m·ω))

where ω is the oscillator frequency.

Question: What happens when we use ε based on frequency instead of velocity?
"""

import numpy as np
from numba import njit, prange
import time
import sys

G = 1.0
HBAR = 1.0

w1 = 0.78451361047755726382
w2 = 0.23557321335935813368
w3 = -1.17767998417887100695
w0 = 1.0 - 2.0*(w1 + w2 + w3)

C = np.array([w3, w2, w1, w0, w1, w2, w3, 0.0])
D = np.array([w3/2, (w3+w2)/2, (w2+w1)/2, (w1+w0)/2,
              (w0+w1)/2, (w1+w2)/2, (w2+w3)/2, w3/2])

print("="*80)
print("QUANTUM HARMONIC OSCILLATOR LENGTH SCALE")
print("="*80)
print()

# =============================================================================
# THEORETICAL DERIVATION
# =============================================================================

print("THEORETICAL BACKGROUND:")
print("-"*80)
print()

print("1. VELOCITY-BASED QUANTUM SCALE (what we've been using):")
print("   ε_v = ℏ/(m·v)")
print("   • Comes from Heisenberg uncertainty: Δx·Δp ~ ℏ")
print("   • For particles with momentum p = mv: Δx ~ ℏ/(m·v)")
print("   • Good for GRAVITATIONAL systems (arbitrary velocities)")
print()

print("2. FREQUENCY-BASED QUANTUM SCALE (harmonic oscillator):")
print("   ε_ω = √(ℏ/(m·ω))")
print("   • Comes from quantum harmonic oscillator ground state")
print("   • Zero-point energy: E₀ = (1/2)ℏω")
print("   • Zero-point size: ⟨x²⟩ = ℏ/(2mω)  →  x_rms = √(ℏ/(2mω))")
print("   • Good for OSCILLATORY systems (definite frequency)")
print()

print("3. RELATIONSHIP:")
print("   For harmonic oscillator with frequency ω:")
print("   • Velocity scale: v ~ ω·x")
print("   • Therefore: ε_v = ℏ/(m·ω·x)")
print("   • If x ~ ε_ω: ε_v ~ ε_ω (they're consistent!)")
print()

print("="*80)
print()

# =============================================================================
# CALCULATE FREQUENCY IN LARGE-ε REGIME
# =============================================================================

print("CALCULATING OSCILLATOR FREQUENCY:")
print("-"*80)
print()

print("In the large-ε limit, force becomes:")
print("  F = GMm·r / ε³  (linear restoring force)")
print()

print("This is a harmonic oscillator with:")
print("  F = -k·r  where k = GM_total/ε³")
print()

print("Frequency:")
print("  ω² = k/m = G·M_total/(m·ε³)")
print("  ω = √(G·M_total/(m·ε³))")
print()

# For N=30, M=1 system
N = 30
m = 1.0
M_total = N * m

# Test different ε values
epsilon_vals = [3.47, 10.0, 34.7, 115.7]

print(f"For N={N}, m={m}, M_total={M_total}:")
print()
print(f"{'ε':<10s} {'ω':<12s} {'ε_ω=√(ℏ/(mω))':<20s} {'ε/ε_ω':<15s} {'Interpretation'}")
print("-"*80)

for eps in epsilon_vals:
    omega = np.sqrt(G * M_total / (m * eps**3))
    epsilon_omega = np.sqrt(HBAR / (m * omega))
    ratio = eps / epsilon_omega

    if ratio < 0.5:
        interp = "Sub-quantum (classical oscillator)"
    elif ratio < 2:
        interp = "Quantum regime (zero-point)"
    else:
        interp = "Super-quantum (over-smoothed)"

    print(f"{eps:<10.2f} {omega:<12.6f} {epsilon_omega:<20.6f} {ratio:<15.3f} {interp}")

print()

# =============================================================================
# QUANTUM HARMONIC OSCILLATOR PHYSICS
# =============================================================================

print("="*80)
print("QUANTUM HARMONIC OSCILLATOR PHYSICS")
print("="*80)
print()

print("Classical harmonic oscillator:")
print("  • Can have ANY energy E ≥ 0")
print("  • Can sit at rest at x=0 (minimum energy = 0)")
print("  • Amplitude determines energy: E = (1/2)kA²")
print()

print("Quantum harmonic oscillator:")
print("  • Energy quantized: E_n = ℏω(n + 1/2)")
print("  • Ground state energy: E₀ = (1/2)ℏω  (ZERO-POINT ENERGY)")
print("  • Cannot sit at rest! Must have ⟨x²⟩ > 0")
print("  • Ground state size: x_rms = √(ℏ/(2mω))")
print()

print("Zero-point motion:")
print("  Even at T=0, quantum oscillator jiggles!")
print("  This jiggling has characteristic length ε_ω = √(ℏ/(mω))")
print()

print("Physical examples:")
print("  • Atomic vibrations in crystals (phonons)")
print("  • Electromagnetic field modes (photons)")
print("  • Molecular vibrations")
print("  • Casimir effect (zero-point EM vibrations)")
print()

# =============================================================================
# IMPLICATIONS FOR N-BODY SYSTEM
# =============================================================================

print("="*80)
print("IMPLICATIONS FOR N-BODY SYSTEM")
print("="*80)
print()

print("When ε >> r, system becomes coupled harmonic oscillators.")
print()

print("Two possible quantum scales:")
print()

print("1. VELOCITY-BASED (what we use):")
print("   ε_v = ℏ/(m·v_rms)")
print("   • Treats system as PARTICLES with uncertain momentum")
print("   • Appropriate for general N-body dynamics")
print("   • No preferred frequency")
print()

print("2. FREQUENCY-BASED (harmonic analog):")
print("   ε_ω = √(ℏ/(m·ω))")
print("   • Treats system as OSCILLATORS with definite frequency")
print("   • Appropriate when ω is well-defined")
print("   • Incorporates zero-point motion")
print()

print("Key question: What is ω in our system?")
print()

# Calculate for our standard setup
v_rms = 0.2881  # From N=30, M=1 runs
eps_v = HBAR / (m * v_rms)

print(f"Standard setup (M=1, v_rms={v_rms:.4f}):")
print(f"  ε_v = {eps_v:.4f}")
print()

# If we're in harmonic regime, what's omega?
omega_at_eps_v = np.sqrt(G * M_total / (m * eps_v**3))
eps_omega_at_eps_v = np.sqrt(HBAR / (m * omega_at_eps_v))

print(f"In the harmonic regime with ε = ε_v = {eps_v:.4f}:")
print(f"  Effective frequency: ω = {omega_at_eps_v:.6f}")
print(f"  Quantum harmonic scale: ε_ω = {eps_omega_at_eps_v:.6f}")
print(f"  Ratio: ε_v/ε_ω = {eps_v/eps_omega_at_eps_v:.3f}")
print()

if eps_v > eps_omega_at_eps_v:
    print("  → System is OVER-SMOOTHED relative to quantum harmonic scale")
    print("  → ε_v suppresses even zero-point fluctuations!")
elif eps_v < eps_omega_at_eps_v:
    print("  → System is UNDER-SMOOTHED relative to quantum harmonic scale")
    print("  → Below zero-point size")
else:
    print("  → System is at NATURAL quantum harmonic scale")

print()

# =============================================================================
# SELF-CONSISTENT QUANTUM SCALE
# =============================================================================

print("="*80)
print("SELF-CONSISTENT QUANTUM SCALE")
print("="*80)
print()

print("Interesting idea: What if we require ε = ε_ω self-consistently?")
print()

print("Condition: ε = √(ℏ/(m·ω)) where ω = √(G·M_total/(m·ε³))")
print()

print("Solving:")
print("  ε² = ℏ/(m·ω)")
print("  ε² = ℏ/(m·√(G·M_total/(m·ε³)))")
print("  ε² = ℏ·√(m·ε³/(G·M_total))/m")
print("  ε² = √(ℏ²·ε³/(G·M_total·m))")
print("  ε⁴ = ℏ²·ε³/(G·M_total·m)")
print("  ε = ℏ²/(G·M_total·m)")
print()

epsilon_selfconsistent = HBAR**2 / (G * M_total * m)
omega_sc = np.sqrt(G * M_total / (m * epsilon_selfconsistent**3))
epsilon_omega_sc = np.sqrt(HBAR / (m * omega_sc))

print(f"Self-consistent solution:")
print(f"  ε_sc = {epsilon_selfconsistent:.6f}")
print(f"  ω = {omega_sc:.6f}")
print(f"  ε_ω = {epsilon_omega_sc:.6f}")
print(f"  ε_sc/ε_ω = {epsilon_selfconsistent/epsilon_omega_sc:.6f}")
print()

print("Physical interpretation:")
print("  This is the UNIQUE quantum length scale where:")
print("  • Regularization length = Zero-point oscillator size")
print("  • Quantum smoothing matches quantum fluctuations")
print("  • Natural scale for quantum harmonic N-body systems!")
print()

# =============================================================================
# COMPARISON WITH PHONONS
# =============================================================================

print("="*80)
print("CONNECTION TO PHONONS")
print("="*80)
print()

print("Phonons = Quantized vibrations in crystals")
print()

print("In a crystal lattice:")
print("  • Atoms vibrate around equilibrium positions")
print("  • Vibrations quantized: E_n = ℏω(n + 1/2)")
print("  • Zero-point motion at T=0")
print()

print("Phonon wavelength:")
print("  • Short wavelength (optical phonons): atoms vibrate out of phase")
print("  • Long wavelength (acoustic phonons): atoms vibrate in phase")
print()

print("Your N=30 system in harmonic regime:")
print("  • Like a 'gravitational crystal'")
print("  • Vibrational modes = phonon-like excitations")
print("  • Quantum scale ε_ω = phonon zero-point size")
print("  • Chaos = anharmonic phonon-phonon scattering!")
print()

print("Connection to solid-state physics:")
print("  • Thermal conductivity: chaotic phonon scattering")
print("  • Superconductivity: phonon-mediated Cooper pairs")
print("  • Debye model: continuum limit of coupled oscillators")
print()

# =============================================================================
# PROPOSED EXPERIMENT
# =============================================================================

print("="*80)
print("PROPOSED TEST")
print("="*80)
print()

print("Compare three ε prescriptions:")
print()

print("1. VELOCITY-BASED (current):")
print(f"   ε = ℏ/(m·v_rms) = {eps_v:.4f}")
print("   → Momentum uncertainty principle")
print()

print("2. FREQUENCY-BASED:")
print(f"   ε = √(ℏ/(m·ω)) with ω calculated from current ε")
print(f"   ε = {eps_omega_at_eps_v:.4f}")
print("   → Zero-point oscillator size")
print()

print("3. SELF-CONSISTENT:")
print(f"   ε = ℏ²/(G·M_total·m) = {epsilon_selfconsistent:.4f}")
print("   → Natural quantum harmonic scale")
print()

print("Run N=30 Lyapunov calculation with each:")
print("  • Measure λ_max, energy conservation")
print("  • See if frequency-based ε has special properties")
print("  • Check if self-consistent scale is 'natural'")
print()

r_rms = 0.866  # From earlier runs

print("Predictions:")
print()
print(f"1. ε_v = {eps_v:.4f}  (ε/r = {eps_v/r_rms:.2f}):")
print("   λ_max ~ 0.037, excellent energy conservation")
print("   Current 'optimal' regime")
print()

print(f"2. ε_ω = {eps_omega_at_eps_v:.4f}  (ε/r = {eps_omega_at_eps_v/r_rms:.2f}):")
print("   Smaller than ε_v → might be closer to classical chaos")
print("   Or: might reveal quantum oscillator effects")
print()

print(f"3. ε_sc = {epsilon_selfconsistent:.4f}  (ε/r = {epsilon_selfconsistent/r_rms:.2f}):")
if epsilon_selfconsistent < 0.1:
    print("   VERY small → likely numerical instability")
    print("   But physically interesting as 'true quantum scale'")
elif epsilon_selfconsistent > 10:
    print("   VERY large → harmonic chaos regime")
    print("   Natural scale for quantum oscillator network")
else:
    print("   Intermediate regime")

print()

print("="*80)
print("SUMMARY")
print("="*80)
print()

print("KEY INSIGHT:")
print("  Harmonic systems have a FREQUENCY-BASED quantum scale:")
print("  ε_ω = √(ℏ/(m·ω))")
print()

print("This is different from velocity-based scale:")
print("  ε_v = ℏ/(m·v)")
print()

print("For your N=30 system:")
print(f"  • Current (velocity): ε = {eps_v:.4f}")
print(f"  • Harmonic analog: ε = {eps_omega_at_eps_v:.4f}")
print(f"  • Self-consistent: ε = {epsilon_selfconsistent:.4f}")
print()

print("Physical meaning:")
print("  • ε_v: Prevents singularities (Heisenberg uncertainty)")
print("  • ε_ω: Represents zero-point motion (quantum ground state)")
print("  • ε_sc: Natural scale where both coincide")
print()

print("Next step:")
print("  Test if frequency-based ε reveals new physics!")
print("  Might see resonances, special stability, or quantum effects")
print()

print("="*80)
