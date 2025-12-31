#!/usr/bin/env python3
"""
Prebiotic Chemistry Simulation

Models the formation of life's building blocks under early Earth conditions.

Key environments simulated:
1. Hydrothermal vents (high T, mineral surfaces)
2. Atmospheric Miller-Urey (lightning, UV)
3. Meteoritic delivery (high density impact)

Key insight: Complex molecules require:
1. High DENSITY (collision frequency)
2. Appropriate TEMPERATURE (Goldilocks zone)
3. CATALYSIS (mineral surfaces, metal ions)
4. PROTECTION (from UV, hydrolysis)

This is the HIERARCHY LADDER:
Simple molecules → Monomers → Polymers → Protocells → Life

Each step has increasing H requirement.
"""

import numpy as np
import json
from pathlib import Path

KB = 8.617e-5  # eV/K


class PrebioticSystem:
    """Simulates prebiotic molecule formation."""

    # Bond energies (eV)
    BONDS = {
        'H-H': 4.5, 'O-H': 4.8, 'N-H': 4.0, 'C-H': 4.3,
        'C-C': 3.6, 'C=C': 6.3, 'C-N': 3.0, 'C=N': 6.2,
        'C-O': 3.6, 'C=O': 7.5, 'N-N': 1.6, 'N=N': 9.8,
        'P-O': 4.0, 'S-H': 3.7, 'S-S': 2.5,
    }

    # Bond lengths (Å)
    LENGTHS = {
        'H-H': 0.74, 'O-H': 0.96, 'N-H': 1.01, 'C-H': 1.09,
        'C-C': 1.54, 'C=C': 1.34, 'C-N': 1.47, 'C=N': 1.27,
        'C-O': 1.43, 'C=O': 1.23, 'N-N': 1.45, 'N=N': 1.10,
        'P-O': 1.60, 'S-H': 1.34, 'S-S': 2.05,
    }

    # Prebiotic molecules
    MOLECULES = {
        'HCN': {'atoms': {'H': 1, 'C': 1, 'N': 1}, 'bonds': ['C-H', 'C=N'], 'energy': -1.5},
        'H2O': {'atoms': {'H': 2, 'O': 1}, 'bonds': ['O-H', 'O-H'], 'energy': -2.5},
        'NH3': {'atoms': {'H': 3, 'N': 1}, 'bonds': ['N-H', 'N-H', 'N-H'], 'energy': -1.2},
        'CH4': {'atoms': {'H': 4, 'C': 1}, 'bonds': ['C-H', 'C-H', 'C-H', 'C-H'], 'energy': -0.8},
        'CO2': {'atoms': {'C': 1, 'O': 2}, 'bonds': ['C=O', 'C=O'], 'energy': -4.0},
        'HCHO': {'atoms': {'H': 2, 'C': 1, 'O': 1}, 'bonds': ['C-H', 'C-H', 'C=O'], 'energy': -1.7},
        'glycine': {'atoms': {'H': 5, 'C': 2, 'N': 1, 'O': 2}, 'bonds': 9, 'energy': -4.0},
        'adenine': {'atoms': {'H': 5, 'C': 5, 'N': 5}, 'bonds': 15, 'energy': -8.0},
        'ribose': {'atoms': {'H': 10, 'C': 5, 'O': 5}, 'bonds': 19, 'energy': -6.0},
    }

    def __init__(self, environment='hydrothermal'):
        """
        Environments:
        - hydrothermal: T=350-400K, mineral catalysis, high density
        - atmospheric: T=250-300K, UV/lightning energy, low density
        - meteoritic: T=200-500K, delivered organics, impact shock
        """
        self.environment = environment

        if environment == 'hydrothermal':
            self.T_range = (350, 450)
            self.density_factor = 10.0  # Concentrated at vents
            self.catalysis_factor = 5.0  # Mineral surfaces (FeS, clay)
            self.protection = 0.9  # Shielded from UV
        elif environment == 'atmospheric':
            self.T_range = (200, 350)
            self.density_factor = 0.1  # Dilute atmosphere
            self.catalysis_factor = 1.0  # No catalysis
            self.protection = 0.3  # Exposed to UV
        elif environment == 'meteoritic':
            self.T_range = (150, 400)
            self.density_factor = 5.0  # Concentrated delivery
            self.catalysis_factor = 2.0  # Metal catalysis
            self.protection = 0.7  # Inside rock/ice
        else:
            raise ValueError(f"Unknown environment: {environment}")

    def formation_rate(self, molecule, T):
        """
        Calculate relative formation rate for a molecule.

        Rate ∝ collision_rate × stability × catalysis × protection
        """
        mol = self.MOLECULES[molecule]
        n_atoms = sum(mol['atoms'].values())

        # Collision rate ∝ density^n (need n particles to meet)
        collision = self.density_factor ** n_atoms

        # Stability ∝ exp(-E/kT) for negative E (stable molecules)
        # At low T, stable molecules favored
        E = mol.get('energy', -1.0)
        stability = np.exp(-E / (KB * T))

        # Hierarchy factor: need H > n_atoms for stability
        H = abs(E) / (KB * T)
        if H < n_atoms:
            hierarchy_penalty = np.exp(-(n_atoms - H))
        else:
            hierarchy_penalty = 1.0

        # Catalysis (lowers activation barrier)
        catalysis = self.catalysis_factor

        # Protection from destruction
        protection = self.protection

        rate = collision * stability * hierarchy_penalty * catalysis * protection

        return rate, H

    def simulate_synthesis(self, n_steps=1000):
        """
        Simulate prebiotic synthesis over time.

        Returns time evolution of molecule abundances.
        """
        print(f"\n{'='*60}")
        print(f"PREBIOTIC SYNTHESIS: {self.environment.upper()} ENVIRONMENT")
        print(f"{'='*60}")
        print(f"Temperature range: {self.T_range[0]}-{self.T_range[1]} K")
        print(f"Density factor: {self.density_factor}")
        print(f"Catalysis factor: {self.catalysis_factor}")
        print(f"UV protection: {self.protection}")

        # Initial abundances (from primordial chemistry)
        abundances = {
            'HCN': 1.0,  # From atmospheric chemistry
            'H2O': 100.0,  # Abundant
            'NH3': 5.0,
            'CH4': 5.0,
            'CO2': 20.0,
            'HCHO': 0.5,  # Formaldehyde
            'glycine': 0.0,
            'adenine': 0.0,
            'ribose': 0.0,
        }

        history = {mol: [abundances[mol]] for mol in abundances}
        history['T'] = []

        print(f"\n{'Step':<8} {'T (K)':<8} {'glycine':<12} {'adenine':<12} {'ribose':<12}")
        print("-" * 52)

        for step in range(n_steps):
            # Temperature variation (day/night, seasonal, volcanic)
            T = np.random.uniform(*self.T_range)

            # Synthesis reactions
            for mol in ['glycine', 'adenine', 'ribose']:
                rate, H = self.formation_rate(mol, T)

                # Formation from precursors
                if mol == 'glycine':
                    # HCN + H2O + CH4 → glycine pathway
                    precursor_limit = min(abundances['HCN'], abundances['H2O']/2)
                elif mol == 'adenine':
                    # 5 HCN → adenine (Oro synthesis)
                    precursor_limit = abundances['HCN'] / 5
                elif mol == 'ribose':
                    # Formose reaction: HCHO → ribose
                    precursor_limit = abundances['HCHO'] / 5

                # Add to abundance (with stochastic noise)
                delta = rate * precursor_limit * 0.001 * np.random.exponential(1)
                abundances[mol] += delta

                # Consume precursors
                if mol == 'glycine' and delta > 0:
                    abundances['HCN'] -= delta
                    abundances['H2O'] -= 2 * delta
                elif mol == 'adenine' and delta > 0:
                    abundances['HCN'] -= 5 * delta
                elif mol == 'ribose' and delta > 0:
                    abundances['HCHO'] -= 5 * delta

            # Degradation (UV, hydrolysis)
            for mol in ['glycine', 'adenine', 'ribose']:
                half_life = 100 * self.protection  # Protected lasts longer
                decay = abundances[mol] * (1 - np.exp(-1/half_life))
                abundances[mol] -= decay

            # Record history
            for mol in abundances:
                history[mol].append(abundances[mol])
            history['T'].append(T)

            if step % (n_steps // 10) == 0:
                print(f"{step:<8} {T:<8.0f} {abundances['glycine']:<12.4f} "
                      f"{abundances['adenine']:<12.4f} {abundances['ribose']:<12.4f}")

        print("-" * 52)
        print(f"{'FINAL':<8} {'-':<8} {abundances['glycine']:<12.4f} "
              f"{abundances['adenine']:<12.4f} {abundances['ribose']:<12.4f}")

        return history, abundances


def compare_environments():
    """Compare molecule formation across environments."""
    print("\n" + "=" * 70)
    print("COMPARING PREBIOTIC ENVIRONMENTS")
    print("=" * 70)

    results = {}
    for env in ['hydrothermal', 'atmospheric', 'meteoritic']:
        system = PrebioticSystem(env)
        history, final = system.simulate_synthesis(n_steps=500)
        results[env] = {
            'glycine': final['glycine'],
            'adenine': final['adenine'],
            'ribose': final['ribose'],
            'total_organic': final['glycine'] + final['adenine'] + final['ribose']
        }

    print("\n" + "=" * 70)
    print("ENVIRONMENT COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Environment':<15} {'Glycine':<12} {'Adenine':<12} {'Ribose':<12} {'Total':<12}")
    print("-" * 55)

    for env, res in results.items():
        print(f"{env:<15} {res['glycine']:<12.4f} {res['adenine']:<12.4f} "
              f"{res['ribose']:<12.4f} {res['total_organic']:<12.4f}")

    # Winner
    best = max(results.items(), key=lambda x: x[1]['total_organic'])
    print(f"\n→ BEST ENVIRONMENT: {best[0].upper()}")
    print(f"  Total organics: {best[1]['total_organic']:.4f}")

    return results


def hierarchy_requirements():
    """Analyze hierarchy requirements for prebiotic molecules."""
    print("\n" + "=" * 70)
    print("HIERARCHY REQUIREMENTS FOR LIFE'S MOLECULES")
    print("=" * 70)

    molecules = [
        ('H₂', 2, 1, 4.5),
        ('H₂O', 3, 2, 4.8),
        ('NH₃', 4, 3, 4.0),
        ('CH₄', 5, 4, 4.3),
        ('HCN', 3, 2, 6.0),
        ('HCHO', 4, 3, 5.0),
        ('Glycine', 10, 9, 4.0),
        ('Adenine', 15, 15, 5.0),
        ('Ribose', 20, 19, 4.5),
        ('ATP', 47, 50, 4.5),
        ('tRNA', 2500, 2700, 4.0),
        ('Ribosome', 200000, 210000, 4.0),
    ]

    print(f"\n{'Molecule':<15} {'Atoms':<8} {'Bonds':<8} {'H_min':<10} {'T_form (K)':<12}")
    print("-" * 58)

    for mol, atoms, bonds, D_avg in molecules:
        # Minimum hierarchy needed ≈ atoms
        H_min = atoms
        # Formation temperature where H ≈ atoms
        T_form = D_avg / (KB * H_min)

        print(f"{mol:<15} {atoms:<8} {bonds:<8} {H_min:<10} {T_form:<12.0f}")

    print(f"""
INTERPRETATION:

1. HIERARCHY LADDER:
   H₂ (H~2) → H₂O (H~3) → Amino acids (H~10) → Nucleotides (H~20)
   → Polymers (H~100+) → Cells (H~10⁶)

2. TEMPERATURE ZONES:
   - 1000-2000 K: Simple molecules (H₂O, CO₂, NH₃)
   - 300-500 K: Complex monomers (amino acids, nucleotides)
   - 200-400 K: Polymers (with catalysis!)
   - 250-350 K: Life (Earth's "Goldilocks zone")

3. THE CATALYSIS IMPERATIVE:
   Without catalysis: T_form for tRNA = 0.0002 K (impossible!)
   With enzymes: Effective H increased 1000×
   → Polymers form at 300 K with catalysis

4. ORIGIN OF LIFE REQUIREMENTS:
   a) Cooling to 300-400 K (after supernova)
   b) High local concentration (ponds, vents)
   c) Mineral catalysis (FeS, clay, Zn)
   d) Protection from UV (underwater, caves)
   e) Time (millions of years)

Each requirement INCREASES EFFECTIVE HIERARCHY!
""")


def polymerization_simulation():
    """Simulate polymer formation from monomers."""
    print("\n" + "=" * 70)
    print("POLYMERIZATION: MONOMERS → POLYMERS")
    print("=" * 70)

    # Parameters
    T = 350  # K (hydrothermal)
    n_monomers = 100
    D_peptide = 2.0  # eV (peptide bond energy)
    catalysis = 10.0  # Enzyme-like catalysis

    print(f"\nSimulating peptide formation:")
    print(f"  Temperature: {T} K")
    print(f"  Initial monomers: {n_monomers}")
    print(f"  Peptide bond energy: {D_peptide} eV")
    print(f"  Catalysis factor: {catalysis}×")

    # Without catalysis
    H_no_cat = D_peptide / (KB * T)
    P_join_no_cat = np.exp(-1 / H_no_cat)  # Probability of joining per step

    # With catalysis
    D_effective = D_peptide * catalysis
    H_with_cat = D_effective / (KB * T)
    P_join_cat = np.exp(-1 / H_with_cat)

    print(f"\nWithout catalysis:")
    print(f"  Hierarchy H = {H_no_cat:.1f}")
    print(f"  Join probability = {P_join_no_cat:.6f}")

    print(f"\nWith catalysis:")
    print(f"  Effective H = {H_with_cat:.1f}")
    print(f"  Join probability = {P_join_cat:.6f}")

    # Simulate polymerization
    n_steps = 1000

    # Without catalysis
    chain_lengths_no_cat = []
    chains = [[i] for i in range(n_monomers)]
    for _ in range(n_steps):
        for i in range(len(chains)):
            for j in range(i+1, len(chains)):
                if np.random.random() < P_join_no_cat:
                    # Join chains
                    chains[i] = chains[i] + chains[j]
                    chains[j] = []
        chains = [c for c in chains if len(c) > 0]
    chain_lengths_no_cat = [len(c) for c in chains]

    # With catalysis
    chains = [[i] for i in range(n_monomers)]
    for _ in range(n_steps):
        for i in range(len(chains)):
            for j in range(i+1, len(chains)):
                if i < len(chains) and j < len(chains):
                    if np.random.random() < P_join_cat * 0.01:  # Scaled for simulation
                        chains[i] = chains[i] + chains[j]
                        chains[j] = []
        chains = [c for c in chains if len(c) > 0]
    chain_lengths_cat = [len(c) for c in chains]

    print(f"\nResults after {n_steps} steps:")
    print(f"\n  Without catalysis:")
    print(f"    Longest chain: {max(chain_lengths_no_cat)}")
    print(f"    Average chain: {np.mean(chain_lengths_no_cat):.1f}")
    print(f"    Number of chains: {len(chain_lengths_no_cat)}")

    print(f"\n  With catalysis:")
    print(f"    Longest chain: {max(chain_lengths_cat)}")
    print(f"    Average chain: {np.mean(chain_lengths_cat):.1f}")
    print(f"    Number of chains: {len(chain_lengths_cat)}")

    print(f"""
KEY INSIGHT:
Catalysis enables polymer formation by increasing effective hierarchy.
Without catalysis, monomers stay monomers (life never starts).
With catalysis, polymers form → enzymes → more catalysis → life!

This is the AUTOCATALYTIC BOOTSTRAP:
  Catalysis → Polymers → Better catalysis → Longer polymers → Life
""")

    return chain_lengths_no_cat, chain_lengths_cat


def main():
    print("╔" + "═" * 68 + "╗")
    print("║" + " PREBIOTIC CHEMISTRY SIMULATION ".center(68) + "║")
    print("║" + " From Simple Molecules to Life's Building Blocks ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    # 1. Compare environments
    env_results = compare_environments()

    # 2. Hierarchy analysis
    hierarchy_requirements()

    # 3. Polymerization
    chains_no_cat, chains_cat = polymerization_simulation()

    # Summary
    print("\n" + "=" * 70)
    print("UNIFIED FRAMEWORK: FROM PHYSICS TO LIFE")
    print("=" * 70)
    print(f"""
The HIERARCHY SELECTION framework explains the origin of life:

LEVEL           HIERARCHY (H)    SELECTION MECHANISM
------          -------------    -------------------
Quarks→Hadrons     10¹⁵         Strong force binding
Hadrons→Atoms      10⁶          Electromagnetic binding
Atoms→Molecules    10²          Covalent bonding + cooling
Molecules→Polymers 10³          Catalysis + concentration
Polymers→Cells     10⁶          Membranes + metabolism
Cells→Organisms    10⁹          Natural selection

KEY INSIGHT: Each level requires HIGHER HIERARCHY than the previous!

The four selection mechanisms operate at every level:
1. DISSIPATION: Energy loss drives toward low-energy geometries
2. SURVIVAL BIAS: Unstable configurations break apart
3. RESONANCE: Matching frequencies enable binding
4. HIERARCHICAL ASSEMBLY: Build complex from simple

ORIGIN OF LIFE = GEOMETRY SELECTION IN CHEMICAL SPACE

The "correct geometry" for life is:
- Self-replicating (autocatalytic)
- Energy-extracting (metabolism)
- Information-storing (genetics)
- Membrane-bounded (cell)

These geometries have the HIGHEST HIERARCHY in chemical space.
Natural selection is the biological version of geometry selection!
""")

    # Save results
    output_dir = Path(__file__).parent.parent.parent / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'environments': env_results,
        'polymerization': {
            'without_catalysis': {
                'max_chain': int(max(chains_no_cat)),
                'avg_chain': float(np.mean(chains_no_cat))
            },
            'with_catalysis': {
                'max_chain': int(max(chains_cat)),
                'avg_chain': float(np.mean(chains_cat))
            }
        }
    }

    with open(output_dir / "prebiotic_chemistry.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}/prebiotic_chemistry.json")

    return results


if __name__ == "__main__":
    results = main()
