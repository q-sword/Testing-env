#!/usr/bin/env python3
"""
Polymerization with Hydrolysis Competition

More realistic simulation including:
1. Forward reaction: amino acids → peptide + H₂O (unfavorable in water)
2. Reverse reaction: peptide + H₂O → amino acids (hydrolysis)
3. Equilibrium depends on: concentration, temperature, catalysis

Key thermodynamics:
- ΔG_peptide ≈ +0.1 eV (unfavorable - water is a product in water!)
- With ATP: ΔG_net ≈ -0.4 eV (favorable)
- Equilibrium: K = exp(-ΔG/kT)

Without energy coupling or extreme conditions, hydrolysis wins!
This explains why life needs:
1. Energy currency (ATP)
2. Catalysis (ribozymes, enzymes)
3. Compartmentalization (membranes)
"""

import numpy as np
import json
from pathlib import Path

KB = 8.617e-5  # eV/K

# Thermodynamic parameters
DG_PEPTIDE = 0.10  # eV (unfavorable in water)
DG_ATP = -0.52     # eV (ATP hydrolysis releases energy)
D_PEPTIDE = 2.0    # eV (peptide bond energy)

AMINO_ACIDS = ['Gly', 'Ala', 'Val', 'Leu', 'Ser', 'Asp', 'Lys', 'Phe']


class RealisticPolymerization:
    """
    Simulates polymerization with competing hydrolysis.

    Without energy input: equilibrium favors monomers
    With ATP coupling: equilibrium shifts to polymers
    """

    def __init__(self, n_monomers=100, T=310, catalysis=1.0,
                 atp_coupling=False, concentration=0.01):
        """
        Parameters:
        -----------
        n_monomers : int
            Initial number of amino acid monomers
        T : float
            Temperature (K) - 310K is body temperature
        catalysis : float
            Catalysis factor (enzyme ~ 10^6)
        atp_coupling : bool
            Whether ATP is available for energy coupling
        concentration : float
            Effective concentration (M) - affects collision rate
        """
        self.n_monomers = n_monomers
        self.T = T
        self.catalysis = catalysis
        self.atp_coupling = atp_coupling
        self.concentration = concentration

        # Initialize chains
        self.chains = [[i] for i in range(n_monomers)]
        self.sequences = [[np.random.choice(AMINO_ACIDS)] for _ in range(n_monomers)]

        # Calculate rate constants
        self._calculate_rates()

    def _calculate_rates(self):
        """Calculate forward and reverse rate constants."""
        # Free energy change for peptide bond formation
        if self.atp_coupling:
            # ATP provides energy: ΔG_net = ΔG_peptide + ΔG_ATP
            self.dG = DG_PEPTIDE + DG_ATP  # ≈ -0.42 eV (favorable!)
        else:
            # No ATP: thermodynamically unfavorable
            self.dG = DG_PEPTIDE  # +0.10 eV (unfavorable)

        # Equilibrium constant K = exp(-ΔG/kT)
        self.K_eq = np.exp(-self.dG / (KB * self.T))

        # Base rates (catalysis affects both equally, preserving K_eq)
        k0 = 1e-6  # Base rate constant

        # Forward rate enhanced by catalysis and concentration
        self.k_forward = k0 * self.catalysis * self.concentration

        # Reverse rate (hydrolysis) - water is always ~55M
        self.k_reverse = self.k_forward / self.K_eq

        # Effective hierarchy
        D_eff = D_PEPTIDE * self.catalysis
        self.H_eff = D_eff / (KB * self.T)

    def attempt_polymerization(self):
        """Try to form a peptide bond between two chains."""
        if len(self.chains) < 2:
            return False

        # Pick two random chains
        i, j = np.random.choice(len(self.chains), 2, replace=False)
        len_i = len(self.chains[i])
        len_j = len(self.chains[j])

        # Probability depends on chain lengths (diffusion)
        # Longer chains diffuse slower
        diffusion_factor = 1.0 / np.sqrt(len_i * len_j)

        # Forward rate
        P_forward = self.k_forward * diffusion_factor

        if np.random.random() < P_forward:
            # Join chains
            new_chain = self.chains[i] + self.chains[j]
            new_seq = self.sequences[i] + self.sequences[j]

            # Remove old chains
            if i > j:
                del self.chains[i], self.sequences[i]
                del self.chains[j], self.sequences[j]
            else:
                del self.chains[j], self.sequences[j]
                del self.chains[i], self.sequences[i]

            self.chains.append(new_chain)
            self.sequences.append(new_seq)
            return True

        return False

    def attempt_hydrolysis(self):
        """Try to break a peptide bond (hydrolysis)."""
        if not self.chains:
            return False

        # Pick random chain
        idx = np.random.randint(len(self.chains))
        chain = self.chains[idx]

        if len(chain) < 2:
            return False

        # Hydrolysis rate (water attacks peptide bonds)
        # Shorter chains are more exposed
        exposure_factor = 1.0 / len(chain)

        P_reverse = self.k_reverse * exposure_factor

        if np.random.random() < P_reverse:
            # Break at random position
            pos = np.random.randint(1, len(chain))

            chain1, chain2 = chain[:pos], chain[pos:]
            seq1, seq2 = self.sequences[idx][:pos], self.sequences[idx][pos:]

            # Replace with fragments
            del self.chains[idx], self.sequences[idx]
            self.chains.extend([chain1, chain2])
            self.sequences.extend([seq1, seq2])
            return True

        return False

    def get_stats(self):
        """Calculate chain length statistics."""
        lengths = [len(c) for c in self.chains]
        if not lengths:
            return {'mean': 0, 'max': 0, 'n_chains': 0, 'n_bonds': 0}

        n_bonds = sum(l - 1 for l in lengths)  # Peptide bonds

        return {
            'mean': np.mean(lengths),
            'max': max(lengths),
            'n_chains': len(lengths),
            'n_bonds': n_bonds,
            'n_monomers': sum(lengths)
        }

    def run(self, n_steps=100000, equilibrate=True):
        """Run simulation."""
        print("=" * 70)
        print(f"POLYMERIZATION WITH HYDROLYSIS COMPETITION")
        print("=" * 70)
        print(f"\nParameters:")
        print(f"  Monomers: {self.n_monomers}")
        print(f"  Temperature: {self.T} K")
        print(f"  Catalysis: {self.catalysis}×")
        print(f"  ATP coupling: {self.atp_coupling}")
        print(f"  Concentration: {self.concentration} M")
        print(f"\nThermodynamics:")
        print(f"  ΔG = {self.dG:.3f} eV ({'favorable' if self.dG < 0 else 'unfavorable'})")
        print(f"  K_eq = {self.K_eq:.2e}")
        print(f"  k_forward = {self.k_forward:.2e}")
        print(f"  k_reverse = {self.k_reverse:.2e}")
        print(f"  H_eff = {self.H_eff:.0f}")

        history = []

        print(f"\n{'Step':<10} {'Chains':<10} {'Mean L':<10} {'Max L':<10} {'Bonds':<10}")
        print("-" * 55)

        polymerizations = 0
        hydrolyses = 0

        for step in range(n_steps):
            # Try polymerization
            if self.attempt_polymerization():
                polymerizations += 1

            # Try hydrolysis (more frequent since water is abundant)
            for _ in range(5):  # Water is ~55M vs ~0.01M amino acids
                if self.attempt_hydrolysis():
                    hydrolyses += 1

            if step % (n_steps // 20) == 0:
                stats = self.get_stats()
                print(f"{step:<10} {stats['n_chains']:<10} {stats['mean']:<10.1f} "
                      f"{stats['max']:<10} {stats['n_bonds']:<10}")
                history.append(stats)

        # Final stats
        print("-" * 55)
        stats = self.get_stats()
        print(f"{'FINAL':<10} {stats['n_chains']:<10} {stats['mean']:<10.1f} "
              f"{stats['max']:<10} {stats['n_bonds']:<10}")

        print(f"\nReaction counts:")
        print(f"  Polymerizations: {polymerizations}")
        print(f"  Hydrolyses: {hydrolyses}")
        print(f"  Net: {polymerizations - hydrolyses}")

        # Theoretical equilibrium
        # At equilibrium: <n_bonds> / <n_monomers> ≈ K_eq * concentration
        expected_bond_fraction = self.K_eq * self.concentration / (1 + self.K_eq * self.concentration)
        print(f"\nTheoretical equilibrium:")
        print(f"  Expected bond fraction: {expected_bond_fraction:.4f}")
        print(f"  Observed bond fraction: {stats['n_bonds'] / self.n_monomers:.4f}")

        return {
            'final_stats': stats,
            'history': history,
            'polymerizations': polymerizations,
            'hydrolyses': hydrolyses,
            'K_eq': self.K_eq,
            'dG': self.dG,
            'atp_coupling': self.atp_coupling
        }


def compare_conditions():
    """Compare different polymerization conditions."""
    print("\n" + "=" * 70)
    print("COMPARING POLYMERIZATION CONDITIONS")
    print("=" * 70)

    conditions = [
        {'cat': 1.0, 'atp': False, 'conc': 0.001, 'label': 'Dilute, no ATP'},
        {'cat': 1.0, 'atp': False, 'conc': 0.1, 'label': 'Concentrated, no ATP'},
        {'cat': 1.0, 'atp': True, 'conc': 0.001, 'label': 'Dilute + ATP'},
        {'cat': 100.0, 'atp': False, 'conc': 0.01, 'label': 'Ribozyme, no ATP'},
        {'cat': 100.0, 'atp': True, 'conc': 0.01, 'label': 'Ribozyme + ATP'},
        {'cat': 1e6, 'atp': True, 'conc': 0.01, 'label': 'Enzyme + ATP (life)'},
    ]

    results = []
    for cond in conditions:
        print(f"\n{'▶' * 35}")
        print(f"Testing: {cond['label']}")
        sim = RealisticPolymerization(
            n_monomers=50, T=310,
            catalysis=cond['cat'],
            atp_coupling=cond['atp'],
            concentration=cond['conc']
        )
        result = sim.run(n_steps=50000)
        result['label'] = cond['label']
        result['catalysis'] = cond['cat']
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: HYDROLYSIS COMPETITION")
    print("=" * 70)
    print(f"\n{'Condition':<25} {'ΔG (eV)':<10} {'K_eq':<12} {'Max L':<8} {'Bonds':<8}")
    print("-" * 65)

    for r in results:
        print(f"{r['label']:<25} {r['dG']:<10.3f} {r['K_eq']:<12.2e} "
              f"{r['final_stats']['max']:<8} {r['final_stats']['n_bonds']:<8}")

    print("""
KEY INSIGHTS:

1. WITHOUT ATP: Hydrolysis wins!
   - ΔG = +0.10 eV means K_eq << 1
   - Most amino acids stay as monomers
   - Only very high concentration shifts equilibrium

2. WITH ATP: Polymerization wins!
   - ΔG = -0.42 eV means K_eq >> 1
   - Equilibrium strongly favors polymers
   - This is how ribosomes work

3. CATALYSIS SPEEDS UP BOTH DIRECTIONS
   - Enzymes don't change K_eq
   - But they allow equilibrium to be reached faster
   - Important for cellular timescales

4. LIFE'S SOLUTION:
   - ATP provides energy (ΔG < 0)
   - Enzymes provide speed (high catalysis)
   - Membranes provide concentration
   - All three are necessary!
""")

    return results


def main():
    print("╔" + "═" * 68 + "╗")
    print("║" + " POLYMERIZATION WITH HYDROLYSIS ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    results = compare_conditions()

    # Save
    output_dir = Path(__file__).parent.parent.parent / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_data = {
        'conditions': [
            {
                'label': r['label'],
                'dG': float(r['dG']),
                'K_eq': float(r['K_eq']),
                'max_length': r['final_stats']['max'],
                'n_bonds': r['final_stats']['n_bonds'],
                'catalysis': float(r['catalysis'])
            }
            for r in results
        ]
    }

    with open(output_dir / "polymerization_hydrolysis.json", 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to: {output_dir}/polymerization_hydrolysis.json")
    return results


if __name__ == "__main__":
    main()
