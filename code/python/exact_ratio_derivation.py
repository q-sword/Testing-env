#!/usr/bin/env python3
"""
CAN WE DERIVE THE RATIO D(H₂⁺)/E(H) = 0.2 EXACTLY?

This is the key question: is there a closed-form expression
for the H₂⁺ dissociation energy that gives us 0.1026 Hartree?

NO FITTING. Only exact mathematics.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expi

# Fundamental
A0 = 52.9177210258e-12  # Bohr radius in meters
E_H = 27.211386  # Hartree in eV

print("=" * 70)
print("DERIVING THE EXACT RATIO D(H₂⁺)/E(H)")
print("=" * 70)

# ============================================================================
# THE EXACT ENERGY OF H₂⁺ AT R = 2a₀
# ============================================================================
print("\n" + "-" * 70)
print("THE EXACT H₂⁺ ENERGY AT EQUILIBRIUM")
print("-" * 70)

# The exact ground state energy of H₂⁺ at R = 2.0 a₀ is:
E_exact = -0.6026  # Hartree (from numerical solution of spheroidal equations)

# Dissociation energy (relative to H + H⁺)
# At infinite separation: E = -0.5 Hartree (just the H atom)
D_exact = 0.5 - (-E_exact)

print(f"""
From the EXACT numerical solution of the Schrödinger equation
in prolate spheroidal coordinates:

  E(H₂⁺, R=2a₀) = -0.6026 Hartree
  E(H + H⁺)     = -0.5 Hartree

  D_e = |E_eq| - 0.5 = {D_exact:.4f} Hartree

The ratio:
  D_e / E(H) = {D_exact:.4f} / 0.5 = {D_exact/0.5:.4f}
""")


# ============================================================================
# CAN WE GET 0.2 FROM SIMPLE PHYSICS?
# ============================================================================
print("\n" + "-" * 70)
print("ATTEMPT: DERIVE 0.2 FROM PHYSICAL ARGUMENTS")
print("-" * 70)

print("""
Consider H₂⁺: one electron shared between two protons.

At equilibrium (R = 2a₀):
1. The electron wavefunction is a superposition: |ψ⟩ = (|1s_a⟩ + |1s_b⟩)/√(2+2S)
   where S = ⟨1s_a|1s_b⟩ is the overlap integral

2. At R = 2a₀:
   S(R=2) = exp(-2)(1 + 2 + 4/3) = exp(-2) × (13/3) ≈ 0.586
""")

R = 2.0  # in units of a₀
S = np.exp(-R) * (1 + R + R**2/3)
print(f"   S(R=2) = {S:.4f}")

print(f"""
3. The bonding energy comes from:
   - Resonance integral: electron tunnels between nuclei
   - This stabilizes the bonding orbital

4. The resonance integral (in Hartree):
   H_ab = -S - (1+R)exp(-R) at R=2
""")

H_ab = -S - (1 + R) * np.exp(-R)
print(f"   H_ab(R=2) = {H_ab:.4f} Hartree")

# The kinetic energy penalty for delocalization
T_cost = 0.5 * (1 - S**2) / (1 + S)
print(f"\n5. Kinetic energy cost of sharing: ~{T_cost:.4f} Hartree")

print("""
The final binding is a balance of:
+ Resonance stabilization (electron feels both nuclei)
- Kinetic energy increase (larger effective "box")
- Electron-electron repulsion (none in H₂⁺!)
+ Nuclear repulsion at 1/R

This gives D_e ≈ 0.1 Hartree, but there's no simple closed formula.
""")


# ============================================================================
# THE EXACT VARIATIONAL CALCULATION
# ============================================================================
print("\n" + "-" * 70)
print("EXACT LCAO CALCULATION (VARIATIONAL UPPER BOUND)")
print("-" * 70)

def h2plus_energy_lcao(R, zeta=1.0):
    """
    LCAO energy for H₂⁺ with 1s orbitals, orbital exponent zeta.

    This is variational - gives upper bound to true energy.
    """
    if R < 0.1:
        return 10.0

    # Overlap
    rho = zeta * R
    S = np.exp(-rho) * (1 + rho + rho**2/3)

    # One-electron integrals (in Hartree, zeta=1)
    # H_aa = kinetic + nuclear-a + nuclear-b
    # For zeta=1: kinetic = 0.5, nuclear-a = -1
    # Nuclear-b attraction (electron at a, nucleus at b):
    Vab_on_a = -(1/R) * (1 - (1 + R) * np.exp(-2*R))

    H_aa = 0.5 - 1 + Vab_on_a  # = -0.5 + Vab

    # Resonance integral
    # H_ab = kinetic_ab + nuclear_ab
    H_ab = np.exp(-R) * (0.5*R - 1 - R - 1)  # Simplified

    # Actually use exact formulas
    H_aa = -0.5 - (1/R) * (1 - (1 + R) * np.exp(-2*R))
    H_ab = -0.5 * S - (1 + R) * np.exp(-R)

    # Bonding orbital energy
    E_elec = (H_aa + H_ab) / (1 + S)

    # Nuclear repulsion
    E_nuc = 1/R

    return E_elec + E_nuc

# Find LCAO minimum
result = minimize_scalar(h2plus_energy_lcao, bounds=(1.0, 4.0), method='bounded')
R_lcao = result.x
E_lcao = result.fun

print(f"LCAO result (zeta=1):")
print(f"  R_eq = {R_lcao:.4f} a₀")
print(f"  E_eq = {E_lcao:.4f} Hartree")
print(f"  D_e = {0.5 - (-E_lcao):.4f} Hartree")
print(f"  D_e / E(H) = {(0.5 - (-E_lcao))/0.5:.4f}")

# With optimized zeta
def h2plus_energy_opt(params):
    R, zeta = params
    if R < 0.1 or zeta < 0.5:
        return 10.0

    rho = zeta * R
    S = np.exp(-rho) * (1 + rho + rho**2/3)

    # Scaled integrals
    H_aa = zeta**2/2 - zeta - (zeta/R) * (1 - (1 + rho) * np.exp(-2*rho))
    H_ab = (zeta**2/2 - zeta) * S - zeta * (1 + rho) * np.exp(-rho)

    E_elec = (H_aa + H_ab) / (1 + S)
    E_nuc = 1/R

    return E_elec + E_nuc

# Grid search for optimal (R, zeta)
best = {'E': 0, 'R': 2.0, 'z': 1.0}
for R in np.linspace(1.5, 3.0, 50):
    for z in np.linspace(1.0, 1.5, 50):
        E = h2plus_energy_opt([R, z])
        if E < best['E']:
            best = {'E': E, 'R': R, 'z': z}

print(f"\nLCAO with optimized ζ:")
print(f"  R_eq = {best['R']:.4f} a₀")
print(f"  ζ_opt = {best['z']:.4f}")
print(f"  E_eq = {best['E']:.4f} Hartree")
print(f"  D_e = {0.5 - (-best['E']):.4f} Hartree")
print(f"  D_e / E(H) = {(0.5 - (-best['E']))/0.5:.4f}")


# ============================================================================
# THE CONCLUSION
# ============================================================================
print("\n" + "=" * 70)
print("CONCLUSION: IS THERE A CLOSED FORMULA FOR 0.2?")
print("=" * 70)

print("""
SHORT ANSWER: NO.

The exact dissociation energy D_e = 0.1026 Hartree comes from solving
transcendental equations in prolate spheroidal coordinates.

There is no simple closed-form expression.

However, we CAN say:

1. D_e ≈ 0.1 Hartree (order of magnitude from dimensional analysis)

2. The ratio D_e/E(H) ≈ 0.2 reflects the "sharing penalty":
   - One electron between two nuclei
   - Kinetic energy increases (delocalization cost)
   - Potential energy is more negative (two attractions)
   - Net: ~1/5 of atomic binding survives

3. The exact numerical value 0.1026 requires solving the full PDE.

WHAT WE DO KNOW EXACTLY:

  R(H₂⁺) = 2.0 a₀       (exact from spheroidal solution)
  D(H₂⁺) = 0.1026 E_H   (exact from spheroidal solution)

  The ratio 0.1026/0.5 = 0.2052... is NOT a simple fraction.
  It does not equal 1/5 = 0.2 exactly.
  It does not equal any simple closed form we can identify.

This is the fundamental limit: H₂⁺ requires numerical solution.
But it IS an exact numerical solution (not variational or approximate).
""")


# ============================================================================
# IS 2 EXACT?
# ============================================================================
print("\n" + "=" * 70)
print("IS R = 2.0 a₀ EXACT?")
print("=" * 70)

print("""
The exact equilibrium distance of H₂⁺ is:

  R_eq = 1.9972 a₀ (from high-precision calculations)

This is very close to 2.0 a₀, but NOT exactly 2.

However, the deviation is only:
  (2.0 - 1.9972) / 1.9972 = 0.14%

So R ≈ 2a₀ is an excellent approximation, but not exact.

WHY is it close to 2?

The electron wavefunction in H has maximum probability at r = a₀.
For the electron to effectively "span" both nuclei, they should be
separated by roughly twice this distance.

This gives R ~ 2a₀ as an order-of-magnitude estimate.
The exact answer 1.9972 a₀ is remarkably close.
""")


if __name__ == "__main__":
    pass
