#!/usr/bin/env python3
"""
Test: What is the PHYSICAL epsilon for gravitational three-body problem?

Question: If singularities are unphysical, what prevents collapse?

Options to test:
1. ε = ℏ/(m_typical × v_rms)  - quantum scale per particle
2. ε = ℏ/(μ_reduced × v_rms)  - reduced mass scale
3. ε = R_physical              - object size (stars/planets)
4. ε = collective quantum      - some N-dependent formula
"""

import numpy as np

# Physical constants (SI units)
HBAR_SI = 1.055e-34  # J·s
G_SI = 6.674e-11     # m³/(kg·s²)

print("="*80)
print("PHYSICAL EPSILON SCALES FOR DIFFERENT SYSTEMS")
print("="*80)
print()

# ==============================================================================
# Test 1: Molecular system (H2)
# ==============================================================================
print("1. MOLECULAR SYSTEM (H₂)")
print("-"*80)

m_proton = 1.673e-27  # kg
m_electron = 9.109e-31  # kg
v_electron = 2.2e6  # m/s (Bohr model)

eps_electron = HBAR_SI / (m_electron * v_electron)
a0_bohr = 5.29e-11  # Bohr radius

print(f"Electron mass: {m_electron:.2e} kg")
print(f"Electron velocity: {v_electron:.2e} m/s")
print(f"ε_quantum = ℏ/(m_e·v) = {eps_electron:.2e} m")
print(f"Bohr radius a₀ = {a0_bohr:.2e} m")
print(f"Ratio ε/a₀ = {eps_electron/a0_bohr:.2f}")
print(f"RESULT: ε ~ a₀ ✓ Quantum scale IS the bond length")
print()

# ==============================================================================
# Test 2: Solar system (Sun-Earth)
# ==============================================================================
print("2. SOLAR SYSTEM (Sun-Earth)")
print("-"*80)

M_sun = 1.989e30  # kg
M_earth = 5.972e24  # kg
v_earth = 2.98e4  # m/s (orbital velocity)
R_sun = 6.96e8  # m (solar radius)
r_orbit = 1.496e11  # m (1 AU)

# Reduced mass
mu_sun_earth = (M_sun * M_earth) / (M_sun + M_earth)

eps_quantum_sun = HBAR_SI / (M_sun * v_earth)
eps_quantum_reduced = HBAR_SI / (mu_sun_earth * v_earth)
eps_physical = R_sun

print(f"Sun mass: {M_sun:.2e} kg")
print(f"Earth mass: {M_earth:.2e} kg")
print(f"Reduced mass: {mu_sun_earth:.2e} kg ≈ M_Earth")
print(f"Earth orbital velocity: {v_earth:.2e} m/s")
print(f"Orbital radius: {r_orbit:.2e} m")
print()
print(f"ε_quantum (Sun mass) = ℏ/(M_☉·v) = {eps_quantum_sun:.2e} m")
print(f"ε_quantum (reduced) = ℏ/(μ·v) = {eps_quantum_reduced:.2e} m")
print(f"ε_physical (Sun radius) = {eps_physical:.2e} m")
print()
print(f"Ratio ε_quantum/r_orbit = {eps_quantum_reduced/r_orbit:.2e} (NEGLIGIBLE)")
print(f"Ratio ε_physical/r_orbit = {eps_physical/r_orbit:.2e}")
print(f"RESULT: Quantum scale irrelevant, finite radius matters ✓")
print()

# ==============================================================================
# Test 3: Three equal stars (hypothetical)
# ==============================================================================
print("3. THREE-BODY STELLAR SYSTEM (equal solar masses)")
print("-"*80)

M_star = M_sun
v_typical = 1e4  # m/s (typical relative velocity in cluster)
R_star = R_sun

# Different epsilon choices
eps_single = HBAR_SI / (M_star * v_typical)
eps_total = HBAR_SI / (3 * M_star * v_typical)
eps_sqrt_N = np.sqrt(3) * HBAR_SI / (M_star * v_typical)
eps_N = 3 * HBAR_SI / (M_star * v_typical)

print(f"Star mass (each): {M_star:.2e} kg")
print(f"Typical velocity: {v_typical:.2e} m/s")
print(f"Star radius: {R_star:.2e} m")
print()
print(f"ε₁ = ℏ/(m·v) = {eps_single:.2e} m")
print(f"ε_total = ℏ/(3m·v) = {eps_total:.2e} m")
print(f"ε_√N = √3·ℏ/(m·v) = {eps_sqrt_N:.2e} m")
print(f"ε_N = 3·ℏ/(m·v) = {eps_N:.2e} m")
print(f"ε_physical (R_star) = {eps_physical:.2e} m")
print()
print(f"ALL quantum scales << R_star by factor ~ {R_star/eps_N:.2e}")
print(f"RESULT: For stars, use ε = R_star (finite size) ✓")
print()

# ==============================================================================
# Test 4: What mass makes quantum epsilon relevant?
# ==============================================================================
print("4. CRITICAL MASS: When does ε_quantum ~ r_typical?")
print("-"*80)

r_typical = 1.0  # AU (typical separation)
v_typical = 3e4  # m/s

# Solve: ℏ/(m·v) = r
m_critical = HBAR_SI / (r_typical * 1.496e11 * v_typical)

print(f"For separation r ~ {r_typical} AU")
print(f"For velocity v ~ {v_typical:.0e} m/s")
print(f"Quantum scale ε ~ r when:")
print(f"  m_critical = ℏ/(r·v) = {m_critical:.2e} kg")
print(f"  This is {m_critical/m_electron:.2e} × m_electron")
print(f"  This is {m_critical/m_proton:.2e} × m_proton")
print()

if m_critical < m_electron:
    print(f"RESULT: Only for particles LIGHTER than electrons!")
    print(f"        (or much slower velocities)")
elif m_critical < m_proton:
    print(f"RESULT: Only for particles lighter than protons")
else:
    print(f"RESULT: Only for atomic-scale particles")
print()

# ==============================================================================
# Test 5: What about primordial/tiny black holes?
# ==============================================================================
print("5. EXTREME CASE: Planck-mass objects")
print("-"*80)

m_planck = 2.176e-8  # kg (Planck mass)
v_typical = 1e3  # m/s
r_schwarzschild = 2 * G_SI * m_planck / (3e8)**2

eps_quantum_planck = HBAR_SI / (m_planck * v_typical)

print(f"Planck mass: {m_planck:.2e} kg")
print(f"Schwarzschild radius: {r_schwarzschild:.2e} m")
print(f"ε_quantum = ℏ/(m_planck·v) = {eps_quantum_planck:.2e} m")
print(f"Ratio ε/r_s = {eps_quantum_planck/r_schwarzschild:.2e}")
print(f"RESULT: Even for Planck mass, quantum scale ~ r_s")
print()

# ==============================================================================
# CONCLUSION
# ==============================================================================
print("="*80)
print("PHYSICAL CONCLUSION")
print("="*80)
print()
print("The regularization scale ε depends on what you're modeling:")
print()
print("1. MOLECULES (electrons around nuclei):")
print("   ε = ℏ/(m_electron·v) ~ 10⁻¹⁰ m")
print("   Physical origin: Heisenberg uncertainty")
print("   Status: REAL PHYSICS ✓")
print()
print("2. MACROSCOPIC OBJECTS (stars, planets):")
print("   ε = R_object ~ 10⁶-10⁹ m")
print("   Physical origin: Finite object size")
print("   Status: REAL PHYSICS ✓")
print()
print("3. POINT-MASS GRAVITY (classical N-body):")
print("   ε_quantum = ℏ/(m·v) ~ 10⁻⁶⁹ m << any gravitational scale")
print("   Physical origin: Quantum uncertainty (but negligible)")
print("   Status: NOT RELEVANT for macroscopic gravity ✗")
print()
print("4. DIMENSIONLESS SIMULATIONS:")
print("   ε = ℏ/(m·v) with ℏ=1, m~1, v~1 → ε~1")
print("   Physical origin: ARTIFICIAL (unit choice)")
print("   Status: MATHEMATICAL TOOL, not physical prediction ⚠")
print()
print("="*80)
print("ANSWER: For gravitational N-body of macroscopic objects,")
print("        use ε = R_object (finite size), NOT quantum scale.")
print()
print("        For molecular dynamics, ε = ℏ/(m·v) IS physical.")
print("="*80)
