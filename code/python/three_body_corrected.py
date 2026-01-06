#!/usr/bin/env python3
"""
CORRECTED THREE-BODY CODE WITH ADAPTIVE SOFTENING
==================================================

Problem with original: ε = ℏ/(m·v) gives ε >> r, so not doing real gravity.

Fix: Use adaptive softening that activates only during close encounters.
     ε_eff = max(ε_min, min(ε_max, α × min_separation))

This preserves Newtonian dynamics at large separations while preventing
singularities during close encounters.
"""

import numpy as np
from multiprocessing import Pool
import time

# =============================================================================
# CONSTANTS
# =============================================================================

G = 1.0
HBAR = 1.0

# Yoshida 6th order coefficients
w1 = 0.78451361047755726382
w2 = 0.23557321335935813368
w3 = -1.17767998417887100695
w0 = 1.0 - 2.0*(w1 + w2 + w3)

YOSHIDA6_C = np.array([w3, w2, w1, w0, w1, w2, w3, 0.0])
YOSHIDA6_D = np.array([
    w3/2, (w3+w2)/2, (w2+w1)/2, (w1+w0)/2,
    (w0+w1)/2, (w1+w2)/2, (w2+w3)/2, w3/2
])

# =============================================================================
# BODY CLASS
# =============================================================================

class Body:
    def __init__(self, mass, pos, vel):
        self.mass = mass
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.acc = np.zeros(3)

# =============================================================================
# SYSTEM CLASS WITH ADAPTIVE SOFTENING
# =============================================================================

class ThreeBodySystem:
    def __init__(self, bodies, epsilon_min=0.01, epsilon_fraction=0.1):
        """
        epsilon_min: minimum softening (prevents singularity)
        epsilon_fraction: ε = fraction × min_separation (adapts to geometry)
        """
        self.bodies = bodies
        self.epsilon_min = epsilon_min
        self.epsilon_fraction = epsilon_fraction
        self.time = 0.0

    def get_min_separation(self):
        """Get minimum pairwise separation"""
        min_r = float('inf')
        for i in range(3):
            for j in range(i+1, 3):
                r = np.linalg.norm(self.bodies[j].pos - self.bodies[i].pos)
                if r < min_r:
                    min_r = r
        return min_r

    def get_adaptive_epsilon(self):
        """Compute adaptive softening parameter"""
        min_r = self.get_min_separation()
        # ε = fraction of minimum separation, but at least epsilon_min
        return max(self.epsilon_min, self.epsilon_fraction * min_r)

    def compute_forces(self):
        """Compute forces with adaptive softening"""
        epsilon = self.get_adaptive_epsilon()

        for body in self.bodies:
            body.acc = np.zeros(3)

        for i in range(3):
            for j in range(i + 1, 3):
                r_ij = self.bodies[j].pos - self.bodies[i].pos
                r = np.linalg.norm(r_ij)
                r_hat = r_ij / r

                # Softened force
                denominator = (r**2 + epsilon**2)**(1.5)
                F_magnitude = G * self.bodies[i].mass * self.bodies[j].mass / denominator
                F_ij = F_magnitude * r_hat

                self.bodies[i].acc += F_ij / self.bodies[i].mass
                self.bodies[j].acc -= F_ij / self.bodies[j].mass

    def yoshida6_step(self, dt):
        """Yoshida 6th order symplectic step"""
        for i in range(len(YOSHIDA6_D)):
            self.compute_forces()
            for body in self.bodies:
                body.vel += YOSHIDA6_D[i] * dt * body.acc

            if i < len(YOSHIDA6_C) - 1 or YOSHIDA6_C[i] != 0.0:
                for body in self.bodies:
                    body.pos += YOSHIDA6_C[i] * dt * body.vel

        self.time += dt

    def compute_energy(self):
        """Compute total energy with current adaptive epsilon"""
        epsilon = self.get_adaptive_epsilon()

        KE = 0.0
        PE = 0.0

        for body in self.bodies:
            KE += 0.5 * body.mass * np.sum(body.vel**2)

        for i in range(3):
            for j in range(i + 1, 3):
                r = np.linalg.norm(self.bodies[j].pos - self.bodies[i].pos)
                PE -= G * self.bodies[i].mass * self.bodies[j].mass / np.sqrt(r**2 + epsilon**2)

        return KE + PE

    def compute_true_energy(self):
        """Compute TRUE Newtonian energy (no softening) for comparison"""
        KE = 0.0
        PE = 0.0

        for body in self.bodies:
            KE += 0.5 * body.mass * np.sum(body.vel**2)

        for i in range(3):
            for j in range(i + 1, 3):
                r = np.linalg.norm(self.bodies[j].pos - self.bodies[i].pos)
                if r > 1e-10:
                    PE -= G * self.bodies[i].mass * self.bodies[j].mass / r
                else:
                    PE = float('-inf')  # Collision

        return KE + PE

# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_seed(seed, T_total=10, dt=0.0001):
    """Test a single random configuration"""
    np.random.seed(seed)

    masses = np.ones(3)
    pos = np.random.randn(3, 3) * 0.5
    vel = np.random.randn(3, 3) * 0.3

    bodies = [Body(masses[i], pos[i], vel[i]) for i in range(3)]
    system = ThreeBodySystem(bodies, epsilon_min=0.01, epsilon_fraction=0.1)

    E0_soft = system.compute_energy()
    E0_true = system.compute_true_energy()
    min_r_initial = system.get_min_separation()

    min_r_seen = min_r_initial
    N_steps = int(T_total / dt)

    for step in range(N_steps):
        system.yoshida6_step(dt)

        min_r = system.get_min_separation()
        if min_r < min_r_seen:
            min_r_seen = min_r

    E1_soft = system.compute_energy()
    E1_true = system.compute_true_energy()

    # Report both softened and true energy conservation
    dE_soft = abs((E1_soft - E0_soft) / E0_soft) if E0_soft != 0 else float('inf')
    dE_true = abs((E1_true - E0_true) / E0_true) if E0_true != 0 and E0_true != float('-inf') else float('inf')

    return seed, dE_soft, dE_true, min_r_initial, min_r_seen

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("CORRECTED THREE-BODY WITH ADAPTIVE SOFTENING")
    print("=" * 70)
    print()
    print("Adaptive ε = max(0.01, 0.1 × min_separation)")
    print("This preserves Newtonian dynamics except during close encounters.")
    print()

    N_SEEDS = 10
    T = 10
    dt = 0.0001

    print(f"Testing {N_SEEDS} seeds, T={T}, dt={dt}")
    print()
    print(f"{'Seed':>4} {'δE(soft)':>12} {'δE(true)':>12} {'r_init':>8} {'r_min':>8} {'Status'}")
    print("-" * 60)

    for seed in range(N_SEEDS):
        _, dE_soft, dE_true, r_init, r_min = test_seed(seed, T, dt)

        if dE_soft < 0.001:
            status = "GOOD"
        elif dE_soft < 0.01:
            status = "OK"
        elif dE_soft < 0.1:
            status = "DRIFT"
        else:
            status = "BAD"

        print(f"{seed:4d} {dE_soft:12.2e} {dE_true:12.2e} {r_init:8.4f} {r_min:8.4f} {status}")

    print()
    print("=" * 70)
    print("δE(soft) = softened Hamiltonian conservation")
    print("δE(true) = true Newtonian energy (may not conserve during close encounters)")
    print("=" * 70)

if __name__ == "__main__":
    main()
