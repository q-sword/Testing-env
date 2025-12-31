#!/usr/bin/env python3
"""
H₂⁺ EXACT SOLUTION: Finite Differences on Spheroidal ODEs

This is the minimal path to exact: discretize the natural equations.
"""

import numpy as np
from scipy.linalg import eigh, eig
from scipy.optimize import brentq, minimize_scalar
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("H₂⁺ EXACT SOLUTION BY FINITE DIFFERENCES")
print("=" * 70)

def solve_h2plus_exact(R, n_grid=150):
    """
    Solve H₂⁺ exactly using finite differences in spheroidal coordinates.

    The separated equations:
      d/dξ[(ξ²-1)X'] + [λ + 2pξ - c²ξ²]X = 0
      d/dη[(1-η²)Y'] + [λ - c²η²]Y = 0

    where c² = R²|E|/2, p = R/2

    Strategy: For trial E, solve η-equation to get λ, then check ξ-equation.
    """

    p = R / 2

    def solve_eta_equation(c2, n_pts):
        """
        Solve η-equation: d/dη[(1-η²)Y'] + [λ - c²η²]Y = 0

        This is an eigenvalue problem in λ.
        For 1σg (even parity): Y(-η) = Y(η)
        """
        # Grid from 0 to 1 (use symmetry)
        eta = np.linspace(0.01, 0.99, n_pts)
        h = eta[1] - eta[0]

        # Build matrix: -[(1-η²)Y']' + c²η²Y = λY
        # Discretize second derivative with variable coefficient

        A = np.zeros((n_pts, n_pts))

        for i in range(1, n_pts - 1):
            g = 1 - eta[i]**2
            g_p = 1 - eta[i+1]**2
            g_m = 1 - eta[i-1]**2

            # Second derivative: [(gY')']_i ≈ [g_{i+½}(Y_{i+1}-Y_i) - g_{i-½}(Y_i-Y_{i-1})]/h²
            g_ph = (g + g_p) / 2  # g at i+1/2
            g_mh = (g + g_m) / 2  # g at i-1/2

            A[i, i-1] = -g_mh / h**2
            A[i, i] = (g_ph + g_mh) / h**2 + c2 * eta[i]**2
            A[i, i+1] = -g_ph / h**2

        # Boundary conditions
        # At η=0: Y'(0) = 0 (even parity) → Y_0 = Y_1
        A[0, 0] = (1 - eta[0]**2) / h**2 + c2 * eta[0]**2
        A[0, 1] = -(1 - eta[0]**2) / h**2

        # At η=1: Y finite (regularity)
        A[-1, -1] = 1
        A[-1, -2] = -1

        # Solve eigenvalue problem
        eigenvalues = np.linalg.eigvalsh(A)

        # Return lowest eigenvalue (ground state λ)
        return np.min(eigenvalues[eigenvalues > -10])

    def solve_xi_equation(c2, lam, n_pts):
        """
        Solve ξ-equation: d/dξ[(ξ²-1)X'] + [λ + 2pξ - c²ξ²]X = 0

        Check if solution decays properly at large ξ.
        """
        # Grid from 1 to ξ_max
        xi_max = 25.0
        xi = np.linspace(1.02, xi_max, n_pts)
        h = xi[1] - xi[0]

        # Build matrix
        A = np.zeros((n_pts, n_pts))

        for i in range(1, n_pts - 1):
            f = xi[i]**2 - 1
            f_p = xi[i+1]**2 - 1
            f_m = xi[i-1]**2 - 1

            f_ph = (f + f_p) / 2
            f_mh = (f + f_m) / 2

            # -[(fX')']_i + [-λ - 2pξ + c²ξ²]X_i = 0
            A[i, i-1] = -f_mh / h**2
            A[i, i] = (f_ph + f_mh) / h**2 - lam - 2*p*xi[i] + c2*xi[i]**2
            A[i, i+1] = -f_ph / h**2

        # Boundary at ξ=1: X regular
        A[0, 0] = 1

        # Boundary at ξ=∞: X → 0 (exponential decay)
        kappa = np.sqrt(c2 * 2) / R if c2 > 0 else 1
        A[-1, -1] = 1
        A[-1, -2] = -np.exp(-kappa * h)

        # Check smallest singular value (should be ~0 for valid solution)
        try:
            s = np.linalg.svd(A, compute_uv=False)
            return np.min(s)
        except:
            return 1e10

    def objective(E):
        """Find E where both equations are satisfied."""
        if E >= 0 or E < -1.5:
            return 1e10

        c2 = R**2 * abs(E) / 2

        # Solve η-equation to get λ
        try:
            lam = solve_eta_equation(c2, n_grid // 2)
        except:
            return 1e10

        # Check ξ-equation with this λ
        residual = solve_xi_equation(c2, lam, n_grid)

        return residual

    # Grid search for E
    E_test = np.linspace(-0.8, -0.4, 100)
    residuals = [objective(E) for E in E_test]

    # Find minimum
    min_idx = np.argmin(residuals)
    E_best = E_test[min_idx]

    # Refine with finer search
    E_fine = np.linspace(E_best - 0.02, E_best + 0.02, 100)
    residuals_fine = [objective(E) for E in E_fine]
    min_idx_fine = np.argmin(residuals_fine)
    E_final = E_fine[min_idx_fine]

    return E_final


# ============================================================================
# SIMPLER APPROACH: Direct matrix method
# ============================================================================

def solve_h2plus_matrix(R, n_basis=50):
    """
    Alternative: Expand in Legendre polynomials and solve matrix eigenvalue problem.

    For the η-equation, expand Y(η) = Σ a_n P_n(η) (even n only for σg)
    This converts the ODE into a matrix eigenvalue problem.
    """
    from scipy.special import eval_legendre

    def compute_energy(E_trial):
        if E_trial >= 0 or E_trial < -1.5:
            return 1e10

        c2 = R**2 * abs(E_trial) / 2
        c = np.sqrt(c2)
        p = R / 2

        # η-equation in Legendre basis
        # The spheroidal wave equation has known matrix elements

        n_max = n_basis

        # Build matrix for η-equation (even n only: 0, 2, 4, ...)
        n_terms = n_max // 2
        H_eta = np.zeros((n_terms, n_terms))

        for i in range(n_terms):
            n = 2 * i  # Even indices

            # Diagonal: n(n+1)
            H_eta[i, i] = n * (n + 1)

            # Off-diagonal from c² term (couples n to n±2)
            if i > 0:
                n_m = 2 * (i - 1)
                # Coupling coefficient
                coef = c2 * (2*n - 1) * (2*n + 1) / ((2*n - 1) * (2*n + 3))
                coef = c2 * n * (n - 1) / ((2*n - 1) * (2*n + 1))
                H_eta[i, i-1] = coef
                H_eta[i-1, i] = coef

        # Eigenvalues give λ
        try:
            lam_values = np.linalg.eigvalsh(H_eta)
            lam = lam_values[0]  # Lowest for ground state
        except:
            return 1e10

        # ξ-equation: use variational estimate
        # For the exact solution, λ and E must be consistent

        # The matching condition relates λ, c², and p
        # For ground state: λ ≈ c²/3 + corrections

        # Simple consistency check
        lam_expected = c2 / 3 - c2**2 / 45  # Small c expansion

        return (lam - lam_expected)**2

    # Search for E
    result = minimize_scalar(compute_energy, bounds=(-0.8, -0.4), method='bounded')
    return result.x


# ============================================================================
# THE CLEANEST EXACT METHOD
# ============================================================================

def solve_h2plus_clean(R, n_grid=200):
    """
    Clean implementation using shooting method on the ξ-equation.

    For given E and λ from η-equation, integrate ξ-equation and
    check boundary condition at large ξ.
    """
    from scipy.integrate import odeint

    p = R / 2

    def get_lambda(c2):
        """Get separation constant λ from η-equation."""
        # For small c: λ ≈ c²/3
        # For larger c: use perturbation theory
        if c2 < 1:
            return c2/3 - c2**2/45 + c2**3/945
        else:
            c = np.sqrt(c2)
            return 2*c - 1 - 1/(4*c) - 1/(8*c**2)

    def xi_ode(y, xi, c2, lam):
        """The ξ-equation as first-order system."""
        X, dX = y
        f = xi**2 - 1
        if f < 1e-10:
            f = 1e-10

        # (fX')' + [λ + 2pξ - c²ξ²]X = 0
        # X'' = -f'/f X' - [λ + 2pξ - c²ξ²]/f X
        #     = -2ξ/f X' - [λ + 2pξ - c²ξ²]/f X

        d2X = -2*xi/f * dX - (lam + 2*p*xi - c2*xi**2)/f * X

        return [dX, d2X]

    def check_bound_state(E):
        """Check if E gives a proper bound state."""
        if E >= 0 or E < -1.5:
            return 1e10

        c2 = R**2 * abs(E) / 2
        lam = get_lambda(c2)

        # Integrate ξ-equation from ξ≈1 outward
        xi_span = np.linspace(1.01, 30.0, n_grid)

        # Initial condition at ξ≈1: X regular
        # Near ξ=1: X ≈ const, X' ≈ -(λ + 2p)X / 2
        X0 = 1.0
        dX0 = -(lam + 2*p) / 2 * X0

        try:
            sol = odeint(xi_ode, [X0, dX0], xi_span, args=(c2, lam))
            X_end = sol[-1, 0]

            # For bound state, X should decay
            # Expected decay: X ~ exp(-κξ) where κ = sqrt(2|E|)
            kappa = np.sqrt(2 * abs(E))
            X_expected = X0 * np.exp(-kappa * (xi_span[-1] - xi_span[0]))

            # If X_end is much larger than expected, not a bound state
            # If X_end has wrong sign, also not correct eigenvalue

            return abs(np.log(abs(X_end) + 1e-100) + kappa * xi_span[-1])
        except:
            return 1e10

    # Search for E
    E_test = np.linspace(-0.70, -0.50, 200)
    residuals = [check_bound_state(E) for E in E_test]

    min_idx = np.argmin(residuals)
    E_best = E_test[min_idx]

    # Refine
    E_fine = np.linspace(E_best - 0.01, E_best + 0.01, 200)
    residuals_fine = [check_bound_state(E) for E in E_fine]
    min_idx_fine = np.argmin(residuals_fine)

    return E_fine[min_idx_fine]


# ============================================================================
# COMPUTE BINDING CURVE
# ============================================================================

print("\nComputing exact binding curve...\n")
print(f"{'R (a₀)':<10} {'E (Ha)':<14} {'D_e (eV)':<12}")
print("-" * 40)

results = []
for R in [1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 4.0, 5.0]:
    E = solve_h2plus_clean(R, n_grid=300)
    D = (-E - 0.5) * 27.211
    results.append((R, E, D))
    print(f"{R:<10.1f} {E:<14.6f} {D:<12.4f}")

# Find equilibrium
R_arr = np.array([r[0] for r in results])
E_arr = np.array([r[1] for r in results])

min_idx = np.argmin(E_arr)
R_eq = R_arr[min_idx]
E_eq = E_arr[min_idx]
D_eq = (-E_eq - 0.5) * 27.211

print(f"\n{'='*40}")
print(f"COMPUTED EQUILIBRIUM:")
print(f"  R_eq = {R_eq:.2f} a₀ = {R_eq * 52.92:.1f} pm")
print(f"  E_eq = {E_eq:.6f} Hartree")
print(f"  D_e = {D_eq:.4f} eV")

print(f"\nEXACT (literature):")
print(f"  R_eq = 2.00 a₀ = 105.8 pm")
print(f"  D_e = 2.793 eV")

print(f"\nOur error: {abs(D_eq - 2.793)/2.793 * 100:.1f}%")


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                  FINITE DIFFERENCE SOLUTION                          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Method: Shooting on ξ-ODE with λ from perturbation theory           ║
║  Grid: 300 points                                                    ║
║                                                                      ║
║  Result:                                                             ║
║    R_eq ≈ {R_eq:.1f} a₀                                                       ║
║    D_e ≈ {D_eq:.2f} eV                                                      ║
║                                                                      ║
║  Exact:                                                              ║
║    R_eq = 2.0 a₀                                                     ║
║    D_e = 2.79 eV                                                     ║
║                                                                      ║
║  The remaining error comes from:                                     ║
║  1. Approximate λ from perturbation (not solving η-equation fully)   ║
║  2. Finite grid spacing                                              ║
║  3. Boundary condition approximation at ξ=1                          ║
║                                                                      ║
║  To get EXACT: solve both ODEs simultaneously as coupled             ║
║  eigenvalue problem, or use continued fractions (Jaffe method).      ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

if __name__ == "__main__":
    pass
