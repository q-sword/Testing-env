#!/usr/bin/env python3
"""
Glycine Formation with High Density + Catalysis

Fixes the sparse simulation by:
1. Higher density (smaller box, more collisions)
2. Catalysis (mineral surface effect - lowers barriers)
3. Pre-concentration of reactants (like hydrothermal vents)

Glycine structure: NH₂-CH₂-COOH
- Amino group: N bonded to 2 H
- Alpha carbon: C bonded to 2 H, 1 N, 1 C
- Carboxyl group: C bonded to =O and -OH

Key insight: Catalysis is equivalent to INCREASING EFFECTIVE HIERARCHY
Without catalysis: H = D/(kT) ≈ 155 at 300K (insufficient for 10-atom molecule)
With catalysis: H_eff = D_eff/(kT) where D_eff >> D (barriers lowered)
"""

import numpy as np
import json
from pathlib import Path

KB = 8.617e-5  # eV/K

# Bond parameters
BONDS = {
    'N-H': {'D': 4.0, 'r0': 1.01, 'alpha': 2.0},
    'C-H': {'D': 4.3, 'r0': 1.09, 'alpha': 2.0},
    'C-N': {'D': 3.0, 'r0': 1.47, 'alpha': 1.8},
    'C-C': {'D': 3.6, 'r0': 1.54, 'alpha': 1.8},
    'C-O': {'D': 3.6, 'r0': 1.43, 'alpha': 1.8},
    'C=O': {'D': 7.5, 'r0': 1.23, 'alpha': 2.2},
    'O-H': {'D': 4.8, 'r0': 0.96, 'alpha': 2.0},
}

# Masses (amu)
MASSES = {'H': 1.0, 'C': 12.0, 'N': 14.0, 'O': 16.0}


def get_bond_params(t1, t2):
    """Get bond parameters for atom pair."""
    for key in [f'{t1}-{t2}', f'{t2}-{t1}', f'{t1}={t2}', f'{t2}={t1}']:
        if key in BONDS:
            return BONDS[key]
    return {'D': 0.1, 'r0': 3.0, 'alpha': 1.0}  # Weak default


def morse_energy(r, D, r0, alpha):
    """Morse potential energy."""
    if r < 0.3:
        return 1e6
    exp_term = np.exp(-alpha * (r - r0))
    return D * (1 - exp_term)**2 - D


class GlycineSim:
    """Monte Carlo simulation of glycine formation."""

    def __init__(self, n_glycine=4, box_size=5.0, catalysis=1.0):
        """
        Parameters:
        -----------
        n_glycine : int
            Target number of glycine molecules
        box_size : float
            Simulation box size (Å) - smaller = denser
        catalysis : float
            Catalysis factor (>1 lowers barriers, increases effective D)
        """
        self.n_glycine = n_glycine
        self.box_size = box_size
        self.catalysis = catalysis

        # Atoms per glycine: N(1) + C(2) + O(2) + H(5) = 10
        self.n_N = n_glycine
        self.n_C = 2 * n_glycine
        self.n_O = 2 * n_glycine
        self.n_H = 5 * n_glycine

        # Calculate density
        self.n_atoms = self.n_N + self.n_C + self.n_O + self.n_H
        self.density = self.n_atoms / box_size**3

        # Initialize atoms
        self.atoms = []
        np.random.seed(42)

        # Place atoms with initial separation
        for elem, count in [('N', self.n_N), ('C', self.n_C),
                            ('O', self.n_O), ('H', self.n_H)]:
            for _ in range(count):
                # Random position
                pos = np.random.uniform(0.5, box_size - 0.5, 3)
                self.atoms.append({'type': elem, 'pos': pos})

    def compute_energy(self):
        """Compute total potential energy with catalysis."""
        E = 0.0
        n = len(self.atoms)

        for i in range(n):
            for j in range(i + 1, n):
                t1 = self.atoms[i]['type']
                t2 = self.atoms[j]['type']
                p1 = self.atoms[i]['pos']
                p2 = self.atoms[j]['pos']

                # Minimum image
                dr = p2 - p1
                dr = dr - self.box_size * np.round(dr / self.box_size)
                r = np.linalg.norm(dr)

                # Get bond parameters
                params = get_bond_params(t1, t2)
                D = params['D'] * self.catalysis  # Catalysis increases effective D
                r0 = params['r0']
                alpha = params['alpha']

                E += morse_energy(r, D, r0, alpha)

        return E

    def count_bonds(self, r_cut=1.8):
        """Count bonds by type."""
        bonds = {}
        n = len(self.atoms)

        for i in range(n):
            for j in range(i + 1, n):
                t1 = self.atoms[i]['type']
                t2 = self.atoms[j]['type']
                p1 = self.atoms[i]['pos']
                p2 = self.atoms[j]['pos']

                dr = p2 - p1
                dr = dr - self.box_size * np.round(dr / self.box_size)
                r = np.linalg.norm(dr)

                if r < r_cut:
                    key = tuple(sorted([t1, t2]))
                    if key not in bonds:
                        bonds[key] = []
                    bonds[key].append({'i': i, 'j': j, 'r': r})

        return bonds

    def estimate_glycine(self, bonds):
        """Estimate glycine molecules from bond counts."""
        # Glycine needs: 2 N-H, 2 C-H, 1 C-N, 1 C-C, 2 C-O, 1 O-H
        counts = {k: len(v) for k, v in bonds.items()}

        n_nh = counts.get(('H', 'N'), 0)
        n_ch = counts.get(('C', 'H'), 0)
        n_cn = counts.get(('C', 'N'), 0)
        n_cc = counts.get(('C', 'C'), 0)
        n_co = counts.get(('C', 'O'), 0)
        n_oh = counts.get(('H', 'O'), 0)

        # Limiting factor
        glycine_est = min(
            n_nh // 2,  # 2 N-H per glycine
            n_ch // 2,  # 2 C-H per glycine
            n_cn,       # 1 C-N per glycine
            n_cc,       # 1 C-C per glycine
            n_co // 2,  # 2 C-O per glycine
            n_oh        # 1 O-H per glycine
        )

        return glycine_est, counts

    def mc_step(self, T, step_size=0.1):
        """Single Monte Carlo step."""
        E_old = self.compute_energy()

        # Pick random atom
        idx = np.random.randint(len(self.atoms))
        old_pos = self.atoms[idx]['pos'].copy()

        # Propose move
        new_pos = old_pos + np.random.uniform(-step_size, step_size, 3)
        new_pos = new_pos % self.box_size
        self.atoms[idx]['pos'] = new_pos

        E_new = self.compute_energy()

        # Metropolis criterion
        if E_new > E_old:
            dE = E_new - E_old
            if np.random.random() > np.exp(-dE / (KB * T)):
                # Reject
                self.atoms[idx]['pos'] = old_pos
                return E_old, False

        return E_new if E_new <= E_old else self.compute_energy(), True

    def run(self, T_high=3000, T_low=300, n_steps=100000):
        """Run simulation with cooling."""
        print("=" * 70)
        print("GLYCINE FORMATION WITH HIGH DENSITY + CATALYSIS")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Target: {self.n_glycine} glycine molecules")
        print(f"  Atoms: {self.n_N} N + {self.n_C} C + {self.n_O} O + {self.n_H} H = {self.n_atoms}")
        print(f"  Box size: {self.box_size} Å")
        print(f"  Density: {self.density:.2f} atoms/Å³")
        print(f"  Catalysis factor: {self.catalysis}×")
        print(f"  Cooling: {T_high} K → {T_low} K")

        # Effective hierarchy with catalysis
        avg_D = np.mean([b['D'] for b in BONDS.values()])
        D_eff = avg_D * self.catalysis
        H_high = D_eff / (KB * T_high)
        H_low = D_eff / (KB * T_low)
        print(f"  Effective hierarchy: {H_high:.0f} → {H_low:.0f}")

        history = {'T': [], 'E': [], 'glycine': [], 'bonds': []}

        print(f"\n{'Step':<10} {'T (K)':<10} {'E (eV)':<12} {'Glycine':<10} {'Bonds':<20}")
        print("-" * 65)

        accepted = 0
        for step in range(n_steps):
            # Temperature schedule
            T = T_high * (T_low / T_high) ** (step / n_steps)

            # Adaptive step size
            step_size = 0.3 * (T / T_high) + 0.03

            E, acc = self.mc_step(T, step_size)
            accepted += acc

            if step % (n_steps // 20) == 0:
                bonds = self.count_bonds()
                n_gly, counts = self.estimate_glycine(bonds)

                # Format bond summary
                bond_str = " ".join([f"{k[0]}-{k[1]}:{v}" for k, v in sorted(counts.items())])

                print(f"{step:<10} {T:<10.0f} {E:<12.2f} {n_gly:<10} {bond_str[:20]:<20}")

                history['T'].append(T)
                history['E'].append(E)
                history['glycine'].append(n_gly)
                history['bonds'].append(counts)

        # Final analysis
        print("-" * 65)
        bonds = self.count_bonds()
        n_glycine, counts = self.estimate_glycine(bonds)

        print(f"\n{'FINAL RESULTS':^65}")
        print("=" * 65)
        print(f"\nBond counts:")
        for bond_type, bond_list in sorted(bonds.items()):
            avg_r = np.mean([b['r'] for b in bond_list]) if bond_list else 0
            ideal_r = get_bond_params(bond_type[0], bond_type[1])['r0']
            print(f"  {bond_type[0]}-{bond_type[1]}: {len(bond_list)} bonds "
                  f"(avg r = {avg_r:.2f} Å, ideal = {ideal_r:.2f} Å)")

        print(f"\nEstimated glycine molecules: {n_glycine} / {self.n_glycine}")
        print(f"Formation efficiency: {100 * n_glycine / self.n_glycine:.0f}%")
        print(f"Acceptance rate: {100 * accepted / n_steps:.1f}%")

        return {
            'n_glycine': n_glycine,
            'target': self.n_glycine,
            'efficiency': n_glycine / self.n_glycine,
            'bonds': {f"{k[0]}-{k[1]}": len(v) for k, v in bonds.items()},
            'catalysis': self.catalysis,
            'density': self.density,
            'H_eff_low': H_low,
            'history': history
        }


def compare_conditions():
    """Compare different density and catalysis conditions."""
    print("\n" + "=" * 70)
    print("COMPARING CONDITIONS: DENSITY × CATALYSIS")
    print("=" * 70)

    conditions = [
        {'box': 8.0, 'cat': 1.0, 'label': 'Sparse, no catalysis'},
        {'box': 5.0, 'cat': 1.0, 'label': 'Dense, no catalysis'},
        {'box': 8.0, 'cat': 3.0, 'label': 'Sparse, with catalysis'},
        {'box': 5.0, 'cat': 3.0, 'label': 'Dense + catalysis (hydrothermal)'},
        {'box': 4.0, 'cat': 5.0, 'label': 'Very dense + strong catalysis'},
    ]

    results = []
    for cond in conditions:
        print(f"\n{'▶'*35}")
        print(f"Testing: {cond['label']}")
        sim = GlycineSim(n_glycine=4, box_size=cond['box'], catalysis=cond['cat'])
        result = sim.run(T_high=2500, T_low=350, n_steps=80000)
        result['label'] = cond['label']
        results.append(result)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: EFFECT OF DENSITY AND CATALYSIS")
    print("=" * 70)
    print(f"\n{'Condition':<35} {'Density':<10} {'Cat':<6} {'H_eff':<8} {'Glycine':<10}")
    print("-" * 70)

    for r in results:
        print(f"{r['label']:<35} {r['density']:<10.2f} {r['catalysis']:<6.1f} "
              f"{r['H_eff_low']:<8.0f} {r['n_glycine']}/{r['target']}")

    # Find best condition
    best = max(results, key=lambda x: x['efficiency'])
    print(f"\n→ BEST: {best['label']} ({best['efficiency']*100:.0f}% efficiency)")

    return results


def explain_catalysis():
    """Explain catalysis in terms of hierarchy."""
    print("\n" + "=" * 70)
    print("THE CATALYSIS = HIERARCHY EQUIVALENCE")
    print("=" * 70)
    print("""
PROBLEM: Complex molecules need H > N_atoms to form stably
- Glycine has 10 atoms
- At 300K with D_avg = 4.0 eV: H = 4.0/(8.6e-5 × 300) = 155
- H = 155 < 10? No! But effective H matters...

CATALYSIS EFFECT:
Catalysts (mineral surfaces, metal ions) lower activation barriers.
This is equivalent to INCREASING effective bond energy D_eff.

Without catalyst:     H = D/(kT)
With catalyst:        H_eff = D_eff/(kT) where D_eff = D × catalysis_factor

Example with 3× catalysis at 350K:
- D_eff = 4.0 × 3 = 12.0 eV
- H_eff = 12.0/(8.6e-5 × 350) = 398
- Now H_eff >> 10, so glycine can form!

BIOLOGICAL CATALYSIS:
- Enzymes are VERY good catalysts (factor ~10⁶)
- This is why life works at 300K despite needing H > 10⁶ for proteins
- Without enzymes, proteins could only form at ~0.001 K (impossible)

THE BOOTSTRAP:
1. Mineral catalysis (clay, FeS): factor ~3-10
2. Ribozymes (RNA catalysis): factor ~10²-10³
3. Enzymes (protein catalysis): factor ~10⁶

Each step enables the NEXT level of complexity!
This is how life climbed the hierarchy ladder.
""")


def main():
    print("╔" + "═" * 68 + "╗")
    print("║" + " GLYCINE FORMATION: DENSITY + CATALYSIS ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    # 1. Run comparison
    results = compare_conditions()

    # 2. Explain catalysis
    explain_catalysis()

    # 3. Theory connection
    print("\n" + "=" * 70)
    print("CONNECTION TO UNIFIED FRAMEWORK")
    print("=" * 70)
    print("""
The four selection mechanisms in amino acid formation:

1. DISSIPATION (Cooling):
   - High T → Low T removes kinetic energy
   - Drives toward low-energy (bonded) configurations

2. SURVIVAL BIAS (Bond stability):
   - Wrong geometries break apart
   - Only correct glycine structure persists

3. RESONANCE CAPTURE (Vibrational matching):
   - Atoms with matching frequencies get captured
   - Creates stable C-C, C-N, C-O bonds

4. HIERARCHICAL ASSEMBLY:
   - First: NH₂ and COOH groups form
   - Then: Connect via CH₂ linker
   - Sequential: simpler → complex

CATALYSIS enhances ALL four mechanisms:
- Faster equilibration (dissipation)
- Lower barriers to correct geometry (survival)
- Better frequency matching (resonance)
- Faster assembly steps (hierarchy)

This explains why life originated at HYDROTHERMAL VENTS:
- High concentration (density)
- Mineral surfaces (catalysis)
- Temperature gradients (dissipation)
- Protected from UV (survival)
""")

    # Save results
    output_dir = Path(__file__).parent.parent.parent / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_data = {
        'conditions': [
            {
                'label': r['label'],
                'density': r['density'],
                'catalysis': r['catalysis'],
                'H_eff': r['H_eff_low'],
                'n_glycine': r['n_glycine'],
                'efficiency': r['efficiency'],
                'bonds': r['bonds']
            }
            for r in results
        ]
    }

    with open(output_dir / "glycine_catalysis.json", 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to: {output_dir}/glycine_catalysis.json")

    return results


if __name__ == "__main__":
    results = main()
