#!/usr/bin/env python3
"""
H₂⁺: COUPLED EIGENVALUE SOLVER

Actually solve the damn thing properly.

The two ODEs:
  η-equation: eigenvalue problem in λ for given c²
  ξ-equation: check if λ gives bound state for given c²

Find E (hence c²) where both are satisfied.
"""

import numpy as np
from scipy.linalg import eigh
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("H₂⁺: COUPLED EIGENVALUE SOLVER")
print("=" * 70)


def compute_lambda(c2, n_basis=30):
    """
    Solve the η-equation to get separation constant λ.

    The spheroidal wave equation:
    d/dη[(1-η²)Y'] + [λ - c²η²]Y = 0

    Expand Y(η) = Σ a_n P_n(η) for even n (σg symmetry).
    This gives a tridiagonal matrix eigenvalue problem.
    """
    # For even functions, use n = 0, 2, 4, ...
    n_terms = n_basis

    # The recurrence relation for Legendre expansion gives:
    # Matrix elements connecting P_n to P_{n±2}

    A = np.zeros((n_terms, n_terms))

    for k in range(n_terms):
        n = 2 * k  # Even indices only

        # Diagonal: n(n+1)
        A[k, k] = n * (n + 1)

        # Off-diagonal from c²η² term
        # ⟨P_n|η²|P_m⟩ couples n to n, n±2

        # Coefficient for η² in Legendre basis:
        # η² P_n = α_n P_{n-2} + β_n P_n + γ_n P_{n+2}

        if k > 0:
            n_m = 2 * (k - 1)
            # Coupling P_n to P_{n-2}
            coef = c2 * (n * (n-1)) / ((2*n-1) * (2*n+1))
            A[k, k-1] = coef
            A[k-1, k] = coef

    # Eigenvalues are the allowed λ values
    eigenvalues = np.linalg.eigvalsh(A)

    # Ground state: lowest λ
    return eigenvalues[0]


def check_xi_equation(c2, lam, R, n_pts=500):
    """
    Check if ξ-equation has a bound state solution.

    d/dξ[(ξ²-1)X'] + [λ + 2pξ - c²ξ²]X = 0

    Integrate from ξ=1 outward and check decay.
    """
    p = R / 2

    def ode(xi, y):
        X, Xp = y  # X and dX/dξ
        f = xi**2 - 1
        if f < 1e-12:
            return [0, 0]

        # X'' = -2ξ/f * X' - (λ + 2pξ - c²ξ²)/f * X
        Xpp = -2*xi/f * Xp - (lam + 2*p*xi - c2*xi**2)/f * X
        return [Xp, Xpp]

    # Integrate from near ξ=1 to large ξ
    xi_span = (1.01, 30.0)
    xi_eval = np.linspace(1.01, 30.0, n_pts)

    # Initial conditions: X regular at ξ=1
    # Near ξ=1: X ≈ const, X' ≈ -(λ + 2p)/(2) * X
    X0 = 1.0
    Xp0 = -(lam + 2*p) / 2 * X0

    sol = solve_ivp(ode, xi_span, [X0, Xp0], t_eval=xi_eval, method='RK45')

    if not sol.success:
        return 1e10, None

    X_end = sol.y[0, -1]
    xi_end = sol.t[-1]

    # For bound state: X ~ exp(-κξ) where κ = sqrt(2|E|) = sqrt(c2*2)/R
    kappa = np.sqrt(2 * c2) / R

    # Expected value at xi_end
    X_expected = X0 * np.exp(-kappa * (xi_end - 1.01))

    # Log ratio tells us if we're decaying correctly
    if abs(X_end) < 1e-100:
        log_ratio = -100
    else:
        log_ratio = np.log(abs(X_end)) - np.log(abs(X_expected) + 1e-100)

    return log_ratio, sol


def find_energy(R, verbose=True):
    """
    Find ground state energy by solving coupled eigenvalue problem.
    """
    p = R / 2

    def objective(E):
        if E >= 0 or E < -2:
            return 1e10

        c2 = R**2 * abs(E) / 2

        # Get λ from η-equation
        lam = compute_lambda(c2)

        # Check ξ-equation
        log_ratio, sol = check_xi_equation(c2, lam, R)

        # For correct E: log_ratio ≈ 0 (solution decays as expected)
        return abs(log_ratio)

    # Grid search
    E_test = np.linspace(-0.75, -0.45, 100)
    residuals = [objective(E) for E in E_test]

    # Find minimum
    min_idx = np.argmin(residuals)
    E_coarse = E_test[min_idx]

    if verbose:
        print(f"  Coarse search: E ≈ {E_coarse:.4f}, residual = {residuals[min_idx]:.4f}")

    # Refine
    E_fine = np.linspace(E_coarse - 0.03, E_coarse + 0.03, 200)
    residuals_fine = [objective(E) for E in E_fine]
    min_idx_fine = np.argmin(residuals_fine)
    E_final = E_fine[min_idx_fine]

    if verbose:
        print(f"  Fine search: E = {E_final:.6f}, residual = {residuals_fine[min_idx_fine]:.4f}")

    return E_final


# ============================================================================
# MAIN: Compute binding curve
# ============================================================================

print("\nSolving coupled eigenvalue problem...\n")
print(f"{'R (a₀)':<10} {'E (Ha)':<14} {'D_e (eV)':<12} {'λ':<10}")
print("-" * 50)

results = []
for R in [1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 4.0]:
    print(f"\nR = {R} a₀:")
    E = find_energy(R, verbose=True)
    c2 = R**2 * abs(E) / 2
    lam = compute_lambda(c2)
    D = (-E - 0.5) * 27.211
    results.append((R, E, D, lam))
    print(f"  → E = {E:.6f} Ha, D_e = {D:.4f} eV, λ = {lam:.4f}")

# Summary
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"\n{'R (a₀)':<10} {'E (Ha)':<14} {'D_e (eV)':<12}")
print("-" * 40)
for R, E, D, lam in results:
    print(f"{R:<10.1f} {E:<14.6f} {D:<12.4f}")

# Find equilibrium
R_arr = np.array([r[0] for r in results])
E_arr = np.array([r[1] for r in results])
min_idx = np.argmin(E_arr)

print(f"\nEquilibrium: R ≈ {R_arr[min_idx]:.1f} a₀, D_e ≈ {results[min_idx][2]:.2f} eV")
print(f"Exact:       R = 2.0 a₀, D_e = 2.79 eV")


# ============================================================================
# VERIFY: Check the components
# ============================================================================

print("\n" + "=" * 70)
print("VERIFICATION: λ(c²) function")
print("=" * 70)

print("\nChecking λ values for known c²:")
print(f"{'c²':<10} {'λ computed':<15} {'λ expected':<15}")
print("-" * 40)

# Known values from literature
test_cases = [
    (0.0, 0.0),
    (0.5, 0.166),
    (1.0, 0.319),
    (2.0, 0.571),
    (4.0, 0.978),
]

for c2, lam_expected in test_cases:
    lam_computed = compute_lambda(c2)
    print(f"{c2:<10.1f} {lam_computed:<15.4f} {lam_expected:<15.3f}")


print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

E_best = E_arr[min_idx]
D_best = results[min_idx][2]

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    COUPLED EIGENVALUE RESULT                         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Method: Solve η-equation (matrix) → λ                               ║
║          Check ξ-equation (shooting) → bound state condition         ║
║          Iterate on E until both satisfied                           ║
║                                                                      ║
║  Our result:                                                         ║
║    R_eq ≈ {R_arr[min_idx]:.1f} a₀                                                       ║
║    D_e ≈ {D_best:.2f} eV                                                      ║
║                                                                      ║
║  Exact:                                                              ║
║    R_eq = 2.0 a₀                                                     ║
║    D_e = 2.79 eV                                                     ║
║                                                                      ║
║  Error: {abs(D_best - 2.79)/2.79 * 100:.1f}%                                                        ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

if __name__ == "__main__":
    pass
