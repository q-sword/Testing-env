#!/usr/bin/env python3
"""
Polymerization Study: From Monomers to Proteins

Simulates the hierarchical assembly of:
1. Amino acids (monomers)
2. Dipeptides (dimers)
3. Oligopeptides (short chains)
4. Polypeptides (long chains)
5. Functional proteins (folded chains)

Key insight: Each polymerization step requires HIGHER HIERARCHY
- Peptide bond formation: H_min ~ 10 (with catalysis)
- Chain extension: H_min ~ chain_length
- Protein folding: H_min ~ 100-1000

Without catalysis, polymerization is thermodynamically unfavorable!
This explains why life needs enzymes.
"""

import numpy as np
import json
from pathlib import Path

KB = 8.617e-5  # eV/K

# Peptide bond parameters
D_PEPTIDE = 2.0  # eV (peptide bond energy)
DG_HYDROLYSIS = 0.1  # eV (free energy of hydrolysis - makes bonds unfavorable in water)

# Amino acid properties
AMINO_ACIDS = [
    'Gly', 'Ala', 'Val', 'Leu', 'Ile',  # Nonpolar
    'Ser', 'Thr', 'Cys', 'Met',          # Polar uncharged
    'Asp', 'Glu',                         # Negative
    'Lys', 'Arg', 'His',                  # Positive
    'Phe', 'Tyr', 'Trp',                  # Aromatic
    'Asn', 'Gln', 'Pro'                   # Special
]


class PolymerizationSim:
    """Simulate polymerization from monomers to proteins."""

    def __init__(self, n_monomers=100, T=350, catalysis=1.0):
        """
        Parameters:
        -----------
        n_monomers : int
            Initial number of amino acid monomers
        T : float
            Temperature (K)
        catalysis : float
            Catalysis factor (ribozyme ~ 100, enzyme ~ 10^6)
        """
        self.n_monomers = n_monomers
        self.T = T
        self.catalysis = catalysis

        # Initialize: all monomers are free
        self.chains = [[i] for i in range(n_monomers)]
        self.sequences = [[np.random.choice(AMINO_ACIDS)] for _ in range(n_monomers)]

        # Compute effective hierarchy
        self.D_eff = D_PEPTIDE * catalysis
        self.H_eff = self.D_eff / (KB * T)

        # Probability of bond formation per encounter
        # P = exp(-ΔG/kT) where ΔG = DG_HYDROLYSIS (unfavorable in water)
        # Catalysis lowers this barrier
        self.P_bond = np.exp(-DG_HYDROLYSIS / (KB * T * catalysis))

        # Probability of hydrolysis (bond breaking)
        # Lower for longer chains (cooperative stabilization)
        self.P_break_base = np.exp(-D_PEPTIDE / (KB * T))

    def chain_stability(self, length):
        """
        Calculate chain stability factor.
        Longer chains are more stable (cooperative effects).
        """
        # Stability increases with length (folding entropy, hydrophobic core)
        return 1.0 + 0.1 * np.log(length + 1)

    def attempt_polymerization(self):
        """
        Attempt one polymerization step.
        Two random chains may join end-to-end.
        """
        if len(self.chains) < 2:
            return False

        # Pick two random chains
        i, j = np.random.choice(len(self.chains), 2, replace=False)
        chain_i = self.chains[i]
        chain_j = self.chains[j]

        # Probability of joining depends on catalysis
        # Longer chains are harder to bring together (diffusion)
        diffusion_factor = 1.0 / np.sqrt(len(chain_i) * len(chain_j))
        P_join = self.P_bond * diffusion_factor

        if np.random.random() < P_join:
            # Join chains
            new_chain = chain_i + chain_j
            new_seq = self.sequences[i] + self.sequences[j]

            # Remove old chains and add new one
            # (remove higher index first to preserve indices)
            if i > j:
                del self.chains[i]
                del self.sequences[i]
                del self.chains[j]
                del self.sequences[j]
            else:
                del self.chains[j]
                del self.sequences[j]
                del self.chains[i]
                del self.sequences[i]

            self.chains.append(new_chain)
            self.sequences.append(new_seq)
            return True

        return False

    def attempt_hydrolysis(self):
        """
        Attempt hydrolysis (chain breaking).
        Shorter chains and ends are more susceptible.
        """
        if not self.chains:
            return False

        # Pick random chain
        idx = np.random.randint(len(self.chains))
        chain = self.chains[idx]

        if len(chain) < 2:
            return False

        # Probability of breaking depends on chain length
        stability = self.chain_stability(len(chain))
        P_break = self.P_break_base / stability

        if np.random.random() < P_break:
            # Break at random position
            pos = np.random.randint(1, len(chain))

            chain1 = chain[:pos]
            chain2 = chain[pos:]
            seq1 = self.sequences[idx][:pos]
            seq2 = self.sequences[idx][pos:]

            # Replace with two shorter chains
            del self.chains[idx]
            del self.sequences[idx]
            self.chains.extend([chain1, chain2])
            self.sequences.extend([seq1, seq2])
            return True

        return False

    def get_statistics(self):
        """Calculate chain length statistics."""
        lengths = [len(c) for c in self.chains]

        if not lengths:
            return {'mean': 0, 'max': 0, 'n_chains': 0, 'distribution': {}}

        # Length distribution
        dist = {}
        for l in lengths:
            if l <= 5:
                key = str(l)
            elif l <= 10:
                key = '6-10'
            elif l <= 20:
                key = '11-20'
            elif l <= 50:
                key = '21-50'
            else:
                key = '>50'
            dist[key] = dist.get(key, 0) + 1

        return {
            'mean': np.mean(lengths),
            'max': max(lengths),
            'min': min(lengths),
            'n_chains': len(lengths),
            'distribution': dist
        }

    def run(self, n_steps=50000):
        """Run polymerization simulation."""
        print("=" * 70)
        print("POLYMERIZATION SIMULATION")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Initial monomers: {self.n_monomers}")
        print(f"  Temperature: {self.T} K")
        print(f"  Catalysis factor: {self.catalysis}×")
        print(f"  Effective hierarchy H: {self.H_eff:.0f}")
        print(f"  Bond formation probability: {self.P_bond:.4f}")
        print(f"  Bond breaking probability: {self.P_break_base:.2e}")

        history = []

        print(f"\n{'Step':<10} {'Chains':<10} {'Mean L':<10} {'Max L':<10} {'Distribution':<30}")
        print("-" * 70)

        for step in range(n_steps):
            # Attempt polymerization
            self.attempt_polymerization()

            # Attempt hydrolysis (less frequent)
            if np.random.random() < 0.1:
                self.attempt_hydrolysis()

            if step % (n_steps // 20) == 0:
                stats = self.get_statistics()
                dist_str = " ".join([f"{k}:{v}" for k, v in sorted(stats['distribution'].items())])
                print(f"{step:<10} {stats['n_chains']:<10} {stats['mean']:<10.1f} "
                      f"{stats['max']:<10} {dist_str[:30]:<30}")
                history.append(stats)

        # Final statistics
        print("-" * 70)
        stats = self.get_statistics()
        print(f"\nFinal state:")
        print(f"  Number of chains: {stats['n_chains']}")
        print(f"  Mean length: {stats['mean']:.1f}")
        print(f"  Max length: {stats['max']}")
        print(f"  Length distribution: {stats['distribution']}")

        # Show longest chains
        if self.chains:
            sorted_chains = sorted(zip(self.chains, self.sequences),
                                  key=lambda x: len(x[0]), reverse=True)
            print(f"\nLongest chains:")
            for chain, seq in sorted_chains[:3]:
                seq_str = '-'.join(seq[:10])
                if len(seq) > 10:
                    seq_str += f"... ({len(seq)} total)"
                print(f"  Length {len(chain)}: {seq_str}")

        return {
            'final_stats': stats,
            'history': history,
            'catalysis': self.catalysis,
            'H_eff': self.H_eff,
            'longest_sequence': self.sequences[np.argmax([len(c) for c in self.chains])] if self.chains else []
        }


def compare_catalysis_levels():
    """Compare polymerization with different catalysis levels."""
    print("\n" + "=" * 70)
    print("EFFECT OF CATALYSIS ON POLYMERIZATION")
    print("=" * 70)

    catalysis_levels = [
        (1.0, "No catalysis (abiotic)"),
        (10.0, "Mineral catalysis (clay, FeS)"),
        (100.0, "Ribozyme (RNA catalysis)"),
        (1000.0, "Simple enzyme"),
        (1e6, "Modern enzyme (fully evolved)"),
    ]

    results = []
    for cat, label in catalysis_levels:
        print(f"\n{'▶' * 35}")
        print(f"Testing: {label} (catalysis = {cat}×)")
        sim = PolymerizationSim(n_monomers=50, T=350, catalysis=cat)
        result = sim.run(n_steps=30000)
        result['label'] = label
        result['catalysis'] = cat
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: CATALYSIS ENABLES POLYMERIZATION")
    print("=" * 70)
    print(f"\n{'Catalysis Level':<30} {'Factor':<12} {'H_eff':<10} {'Max Length':<12} {'Mean L':<10}")
    print("-" * 75)

    for r in results:
        print(f"{r['label']:<30} {r['catalysis']:<12.0e} {r['H_eff']:<10.0f} "
              f"{r['final_stats']['max']:<12} {r['final_stats']['mean']:<10.1f}")

    return results


def protein_folding_analysis():
    """Analyze hierarchy requirements for protein folding."""
    print("\n" + "=" * 70)
    print("PROTEIN FOLDING: THE ULTIMATE HIERARCHY CHALLENGE")
    print("=" * 70)

    # Protein folding energetics
    proteins = [
        {'name': 'Insulin', 'length': 51, 'folds': 1, 'D_fold': 0.5},
        {'name': 'Lysozyme', 'length': 129, 'folds': 5, 'D_fold': 1.0},
        {'name': 'Hemoglobin', 'length': 574, 'folds': 8, 'D_fold': 2.0},
        {'name': 'ATP Synthase', 'length': 5000, 'folds': 50, 'D_fold': 10.0},
    ]

    print(f"\n{'Protein':<15} {'Length':<10} {'Fold E (eV)':<12} {'H_fold':<10} {'Possible configs':<20}")
    print("-" * 70)

    T = 300  # Body temperature

    for p in proteins:
        H_fold = p['D_fold'] / (KB * T)
        # Number of possible configurations ~ 2^N (very rough)
        configs = 2 ** (p['length'] // 10)
        print(f"{p['name']:<15} {p['length']:<10} {p['D_fold']:<12.1f} {H_fold:<10.0f} ~2^{p['length']//10}")

    print("""
FOLDING PARADOX (Levinthal):
A 100-amino-acid protein has ~10^100 possible configurations.
Sampling all at 10^13/s would take 10^77 years!

Yet proteins fold in MILLISECONDS. How?

ANSWER: HIERARCHY!
1. Local structure forms first (α-helices, β-sheets)
2. Secondary structures assemble into domains
3. Domains pack into tertiary structure
4. Quaternary structure (multi-protein complexes)

Each level REDUCES the search space for the next level.
This is HIERARCHICAL ASSEMBLY in action!

The folding hierarchy:
  H_backbone >> H_secondary >> H_tertiary >> H_quaternary

Chaperones (molecular assistants) provide CATALYSIS for folding:
- Prevent misfolding (survival bias)
- Speed up correct folding (dissipation)
- Guide assembly (hierarchy)
""")


def origin_of_polymerization():
    """Explain the origin of biological polymerization."""
    print("\n" + "=" * 70)
    print("THE ORIGIN OF POLYMERIZATION: FROM CHEMISTRY TO LIFE")
    print("=" * 70)
    print("""
THE POLYMERIZATION PROBLEM:
- Peptide bond formation is thermodynamically UNFAVORABLE in water
- ΔG ≈ +0.1 eV per bond (costs energy)
- Without catalysis, equilibrium strongly favors hydrolysis
- Yet life is MADE of polymers (proteins, DNA, RNA)

SOLUTION: COUPLED REACTIONS + CATALYSIS

1. ENERGY COUPLING (ATP hydrolysis):
   ATP → ADP + Pi releases ~0.5 eV
   This drives unfavorable peptide bond formation
   Net ΔG = -0.5 + 0.1 = -0.4 eV (now favorable!)

2. MINERAL CATALYSIS (prebiotic):
   Clay surfaces (montmorillonite) catalyze peptide bonds
   FeS clusters provide electron transfer
   Catalysis factor ~10-100

3. RIBOZYMES (RNA world):
   RNA can catalyze its own polymerization
   First self-replicating molecules
   Catalysis factor ~100-1000

4. ENZYMES (modern life):
   Ribosomes: giant RNA-protein machines
   Polymerases: copy DNA/RNA with high fidelity
   Catalysis factor ~10^6

THE BOOTSTRAP:
Minerals → RNA → Proteins → Modern enzymes
Each step enables HIGHER catalysis → LONGER polymers → MORE complexity

This is the AUTOCATALYTIC HIERARCHY:
  H_minerals < H_ribozymes < H_enzymes < H_cells

Life is a SELF-AMPLIFYING hierarchy generator!
""")


def main():
    print("╔" + "═" * 68 + "╗")
    print("║" + " POLYMERIZATION: FROM MONOMERS TO PROTEINS ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    # 1. Compare catalysis levels
    results = compare_catalysis_levels()

    # 2. Protein folding analysis
    protein_folding_analysis()

    # 3. Origin of polymerization
    origin_of_polymerization()

    # 4. Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS: POLYMERIZATION AND HIERARCHY")
    print("=" * 70)
    print("""
1. POLYMERIZATION REQUIRES CATALYSIS
   Without catalysis: only monomers and short oligomers
   With catalysis: long chains can form and persist

2. CATALYSIS = INCREASED EFFECTIVE HIERARCHY
   H_eff = D_eff / (kT) where D_eff = D × catalysis
   Enzymes (cat = 10^6) → H_eff ~ 10^8 at 300K
   This enables proteins with >1000 amino acids

3. HIERARCHICAL ASSEMBLY IS ESSENTIAL
   Long polymers can't form in one step
   Must build: monomer → dimer → oligomer → polymer
   Each step prepares for the next

4. FOLDING IS GEOMETRY SELECTION
   Correct fold = lowest energy = highest hierarchy
   Wrong folds are unstable → hydrolysis (survival bias)
   Chaperones accelerate correct folding (catalysis)

5. LIFE BOOTSTRAPS ITS OWN CATALYSIS
   Minerals → RNA → Proteins → Better proteins
   Each generation of catalysts enables the next
   This is the AUTOCATALYTIC ORIGIN OF LIFE
""")

    # Save results
    output_dir = Path(__file__).parent.parent.parent / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_data = {
        'catalysis_comparison': [
            {
                'label': r['label'],
                'catalysis': float(r['catalysis']),
                'H_eff': float(r['H_eff']),
                'max_length': r['final_stats']['max'],
                'mean_length': r['final_stats']['mean'],
                'n_chains': r['final_stats']['n_chains']
            }
            for r in results
        ]
    }

    with open(output_dir / "polymerization_study.json", 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to: {output_dir}/polymerization_study.json")

    return results


if __name__ == "__main__":
    results = main()
