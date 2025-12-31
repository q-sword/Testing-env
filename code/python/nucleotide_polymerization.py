#!/usr/bin/env python3
"""
Nucleotide Polymerization: DNA and RNA Synthesis

Simulates the hierarchical assembly of nucleic acids:
1. Nucleotides (monomers): A, T/U, G, C
2. Oligonucleotides (short chains)
3. Polynucleotides (DNA/RNA)
4. Base pairing and double helix formation

Key insight: Information storage requires EVEN HIGHER HIERARCHY than proteins
- Phosphodiester bond: D ~ 3.6 eV (stronger than peptide bond)
- Base stacking: D ~ 0.3-0.5 eV (cooperative stability)
- Hydrogen bonds (base pairs): D ~ 0.1-0.3 eV
- Double helix: emergent stability from hierarchy

The RNA World hypothesis: RNA came first because it can:
1. Store information (like DNA)
2. Catalyze reactions (like proteins)
"""

import numpy as np
import json
from pathlib import Path

KB = 8.617e-5  # eV/K

# Nucleotide bond energies
D_PHOSPHODIESTER = 3.6   # eV (backbone bond)
D_BASE_STACK = 0.4       # eV (average stacking energy)
D_HYDROGEN_BOND = 0.15   # eV (per H-bond in base pair)

# Free energy of phosphodiester hydrolysis
DG_HYDROLYSIS = 0.15     # eV (unfavorable in water, like peptides)

# Base pairing rules
BASE_PAIRS = {
    'A': 'T',  # DNA
    'T': 'A',
    'G': 'C',
    'C': 'G',
    'U': 'A',  # RNA (Uracil pairs with Adenine)
}

# Number of H-bonds per base pair
H_BONDS = {
    ('A', 'T'): 2,
    ('T', 'A'): 2,
    ('A', 'U'): 2,
    ('U', 'A'): 2,
    ('G', 'C'): 3,
    ('C', 'G'): 3,
}


class NucleotidePolymerization:
    """
    Simulate nucleotide polymerization with template-directed synthesis.

    Key difference from proteins:
    - Templates enable INFORMATION COPYING
    - Base pairing provides SPECIFICITY
    - This is the foundation of heredity
    """

    def __init__(self, n_nucleotides=100, T=350, catalysis=1.0,
                 is_rna=True, has_template=False):
        """
        Parameters:
        -----------
        n_nucleotides : int
            Initial number of nucleotide monomers
        T : float
            Temperature (K)
        catalysis : float
            Catalysis factor (ribozyme ~ 10^4, polymerase ~ 10^8)
        is_rna : bool
            True for RNA (uses U), False for DNA (uses T)
        has_template : bool
            If True, simulate template-directed synthesis
        """
        self.n_nucleotides = n_nucleotides
        self.T = T
        self.catalysis = catalysis
        self.is_rna = is_rna
        self.has_template = has_template

        # Base alphabet
        self.bases = ['A', 'U' if is_rna else 'T', 'G', 'C']

        # Initialize: all monomers are free
        self.chains = [[i] for i in range(n_nucleotides)]
        self.sequences = [[np.random.choice(self.bases)] for _ in range(n_nucleotides)]

        # Template strand (if using template-directed synthesis)
        if has_template:
            template_length = n_nucleotides // 2
            self.template = [np.random.choice(self.bases) for _ in range(template_length)]
            self.template_position = 0  # Current copying position

        # Calculate rate constants
        self._calculate_rates()

    def _calculate_rates(self):
        """Calculate polymerization and hydrolysis rates."""
        # Effective bond energy with catalysis
        self.D_eff = D_PHOSPHODIESTER * np.sqrt(self.catalysis)
        self.H_eff = self.D_eff / (KB * self.T)

        # Bond formation probability (catalysis lowers activation barrier)
        self.P_bond = np.exp(-DG_HYDROLYSIS / (KB * self.T * self.catalysis))

        # Hydrolysis probability (base rate)
        self.P_break = np.exp(-D_PHOSPHODIESTER / (KB * self.T))

        # Template fidelity (correct base pairing probability)
        if self.has_template:
            # With template, correct base is favored
            self.P_correct = 0.99 ** (1 / np.sqrt(self.catalysis))  # Higher catalysis = more errors!
            # Modern polymerases have proofreading which corrects this

    def chain_stability(self, chain, sequence):
        """
        Calculate stability of a nucleotide chain.
        Includes backbone + stacking + (if paired) H-bonds.
        """
        length = len(chain)
        if length < 2:
            return 1.0

        # Backbone stability (phosphodiester bonds)
        backbone = (length - 1) * D_PHOSPHODIESTER

        # Base stacking (sequence-dependent)
        stacking = (length - 1) * D_BASE_STACK

        # Total binding energy
        E_total = backbone + stacking

        return E_total / (KB * self.T)

    def attempt_polymerization(self):
        """Attempt to form a phosphodiester bond between two chains."""
        if len(self.chains) < 2:
            return False

        # Pick two random chains
        i, j = np.random.choice(len(self.chains), 2, replace=False)
        chain_i, chain_j = self.chains[i], self.chains[j]

        # Diffusion factor (longer chains are slower)
        diffusion = 1.0 / np.sqrt(len(chain_i) * len(chain_j))

        P_join = self.P_bond * diffusion

        if np.random.random() < P_join:
            # Join chains (5' to 3' direction)
            new_chain = chain_i + chain_j
            new_seq = self.sequences[i] + self.sequences[j]

            # Remove old, add new
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

    def attempt_template_extension(self):
        """
        Attempt template-directed synthesis.
        This is how DNA/RNA replication works!
        """
        if not self.has_template:
            return False

        if self.template_position >= len(self.template):
            return False  # Template fully copied

        # Find a monomer to add
        monomers = [i for i, c in enumerate(self.chains) if len(c) == 1]
        if not monomers:
            return False

        # Template specifies which base should be added
        template_base = self.template[self.template_position]
        correct_base = BASE_PAIRS.get(template_base, 'A')

        # Check if any monomer has the correct base
        for idx in monomers:
            if self.sequences[idx][0] == correct_base:
                # Found correct base - high probability of incorporation
                if np.random.random() < self.P_correct * self.P_bond:
                    self.template_position += 1
                    # Mark this monomer as "incorporated"
                    self.chains[idx] = [-1]  # Sentinel for incorporated
                    return True
            else:
                # Wrong base - can still incorporate with low probability (error)
                if np.random.random() < (1 - self.P_correct) * self.P_bond * 0.1:
                    self.template_position += 1
                    self.chains[idx] = [-1]
                    return True  # Misincorporation!

        return False

    def attempt_hydrolysis(self):
        """Attempt to break a phosphodiester bond."""
        if not self.chains:
            return False

        # Pick random chain
        idx = np.random.randint(len(self.chains))
        chain = self.chains[idx]

        if len(chain) < 2:
            return False

        # Stability increases with chain length (stacking)
        stability = self.chain_stability(chain, self.sequences[idx])
        P_break = self.P_break / (1 + stability / 100)

        if np.random.random() < P_break:
            # Break at random position
            pos = np.random.randint(1, len(chain))

            chain1, chain2 = chain[:pos], chain[pos:]
            seq1, seq2 = self.sequences[idx][:pos], self.sequences[idx][pos:]

            del self.chains[idx], self.sequences[idx]
            self.chains.extend([chain1, chain2])
            self.sequences.extend([seq1, seq2])
            return True

        return False

    def get_stats(self):
        """Calculate chain statistics."""
        # Filter out incorporated monomers
        valid_chains = [(c, s) for c, s in zip(self.chains, self.sequences)
                        if c[0] != -1]

        if not valid_chains:
            return {'mean': 0, 'max': 0, 'n_chains': 0, 'gc_content': 0}

        chains, seqs = zip(*valid_chains)
        lengths = [len(c) for c in chains]

        # GC content (higher = more stable)
        all_bases = [b for s in seqs for b in s]
        gc = sum(1 for b in all_bases if b in ['G', 'C']) / len(all_bases) if all_bases else 0

        return {
            'mean': np.mean(lengths),
            'max': max(lengths),
            'min': min(lengths),
            'n_chains': len(lengths),
            'gc_content': gc,
            'total_nucleotides': sum(lengths)
        }

    def run(self, n_steps=50000):
        """Run nucleotide polymerization simulation."""
        mol_type = "RNA" if self.is_rna else "DNA"

        print("=" * 70)
        print(f"{mol_type} NUCLEOTIDE POLYMERIZATION")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Nucleotides: {self.n_nucleotides}")
        print(f"  Temperature: {self.T} K")
        print(f"  Catalysis: {self.catalysis}×")
        print(f"  Template: {self.has_template}")
        print(f"  H_eff: {self.H_eff:.0f}")
        print(f"  P_bond: {self.P_bond:.4f}")

        if self.has_template:
            print(f"  Template length: {len(self.template)}")
            print(f"  Template: {''.join(self.template[:20])}...")

        history = []
        polymerizations = 0
        hydrolyses = 0
        template_extensions = 0

        print(f"\n{'Step':<10} {'Chains':<10} {'Mean':<8} {'Max':<8} {'GC%':<8}")
        print("-" * 50)

        for step in range(n_steps):
            # Try polymerization
            if self.attempt_polymerization():
                polymerizations += 1

            # Try template extension
            if self.has_template:
                if self.attempt_template_extension():
                    template_extensions += 1

            # Try hydrolysis (water is always present)
            for _ in range(3):  # Water attacks
                if self.attempt_hydrolysis():
                    hydrolyses += 1

            if step % (n_steps // 20) == 0:
                stats = self.get_stats()
                print(f"{step:<10} {stats['n_chains']:<10} {stats['mean']:<8.1f} "
                      f"{stats['max']:<8} {stats['gc_content']*100:<8.1f}")
                history.append(stats)

        # Final stats
        print("-" * 50)
        stats = self.get_stats()
        print(f"{'FINAL':<10} {stats['n_chains']:<10} {stats['mean']:<8.1f} "
              f"{stats['max']:<8} {stats['gc_content']*100:<8.1f}")

        print(f"\nReaction counts:")
        print(f"  Polymerizations: {polymerizations}")
        print(f"  Hydrolyses: {hydrolyses}")
        if self.has_template:
            print(f"  Template extensions: {template_extensions}")
            print(f"  Template copied: {self.template_position}/{len(self.template)}")

        # Show longest sequences
        valid = [(c, s) for c, s in zip(self.chains, self.sequences) if c[0] != -1]
        if valid:
            sorted_chains = sorted(valid, key=lambda x: len(x[0]), reverse=True)
            print(f"\nLongest sequences:")
            for chain, seq in sorted_chains[:3]:
                seq_str = ''.join(seq[:30])
                if len(seq) > 30:
                    seq_str += f"... ({len(seq)} nt)"
                print(f"  {seq_str}")

        return {
            'final_stats': stats,
            'history': history,
            'polymerizations': polymerizations,
            'hydrolyses': hydrolyses,
            'template_extensions': template_extensions,
            'is_rna': self.is_rna,
            'catalysis': self.catalysis
        }


def compare_rna_dna():
    """Compare RNA vs DNA polymerization."""
    print("\n" + "=" * 70)
    print("RNA WORLD vs DNA WORLD")
    print("=" * 70)

    results = []

    # RNA without catalysis (primordial soup)
    print("\n" + "▶" * 35)
    print("RNA - No catalysis (primordial)")
    sim = NucleotidePolymerization(n_nucleotides=50, T=350, catalysis=1.0,
                                    is_rna=True, has_template=False)
    r = sim.run(n_steps=30000)
    r['label'] = 'RNA primordial'
    results.append(r)

    # RNA with ribozyme catalysis
    print("\n" + "▶" * 35)
    print("RNA - Ribozyme catalysis")
    sim = NucleotidePolymerization(n_nucleotides=50, T=350, catalysis=100.0,
                                    is_rna=True, has_template=False)
    r = sim.run(n_steps=30000)
    r['label'] = 'RNA + ribozyme'
    results.append(r)

    # RNA with template (self-replication!)
    print("\n" + "▶" * 35)
    print("RNA - Template-directed (self-replication)")
    sim = NucleotidePolymerization(n_nucleotides=50, T=350, catalysis=100.0,
                                    is_rna=True, has_template=True)
    r = sim.run(n_steps=30000)
    r['label'] = 'RNA replication'
    results.append(r)

    # DNA with polymerase
    print("\n" + "▶" * 35)
    print("DNA - Polymerase (modern life)")
    sim = NucleotidePolymerization(n_nucleotides=50, T=310, catalysis=1e6,
                                    is_rna=False, has_template=True)
    r = sim.run(n_steps=30000)
    r['label'] = 'DNA + polymerase'
    results.append(r)

    return results


def information_hierarchy():
    """Explain the information hierarchy in nucleic acids."""
    print("\n" + "=" * 70)
    print("INFORMATION HIERARCHY: FROM CHEMISTRY TO HEREDITY")
    print("=" * 70)
    print("""
THE INFORMATION PROBLEM:
Life requires not just polymers, but INFORMATION-CARRYING polymers.
- Proteins: 20 amino acids → specific 3D structure → function
- DNA/RNA: 4 bases → sequence → protein blueprint

HIERARCHY OF INFORMATION STORAGE:

Level 1: SEQUENCE (Primary Structure)
  - Linear order of nucleotides: ATGCAGTC...
  - 4^N possible sequences for N nucleotides
  - Information capacity: 2 bits per nucleotide

Level 2: BASE PAIRING (Secondary Structure)
  - A pairs with T/U (2 H-bonds)
  - G pairs with C (3 H-bonds)
  - Creates double helix stability
  - Enables COPYING (template-directed synthesis)

Level 3: 3D STRUCTURE (Tertiary Structure)
  - RNA folds into complex shapes
  - Ribozymes: RNA enzymes
  - Ribosomes: RNA machines

Level 4: GENETIC CODE (Information → Function)
  - 3 nucleotides = 1 codon = 1 amino acid
  - 64 codons → 20 amino acids + stop signals
  - Universal code: same in all life

THE HIERARCHY INSIGHT:
  H_sequence < H_pairing < H_folding < H_code

Each level provides CONTEXT for the next:
- Sequence enables pairing
- Pairing enables copying
- Copying enables evolution
- Evolution enables complexity
""")


def replication_fidelity():
    """Analyze replication fidelity requirements."""
    print("\n" + "=" * 70)
    print("REPLICATION FIDELITY: THE ERROR CATASTROPHE")
    print("=" * 70)
    print("""
THE ERROR PROBLEM:
For information to persist, copying must be accurate enough.
Error rate ε, genome length L:
  - Probability of perfect copy: (1-ε)^L
  - For survival: (1-ε)^L > threshold

EIGEN'S ERROR THRESHOLD:
Maximum genome length: L_max ≈ 1/ε

Without enzymes (ε ~ 0.01):
  L_max ~ 100 nucleotides
  → Only short RNA can replicate!

With ribozymes (ε ~ 0.001):
  L_max ~ 1000 nucleotides
  → Longer RNA, more complex ribozymes possible

With enzymes + proofreading (ε ~ 10^-9):
  L_max ~ 10^9 nucleotides
  → Human genome (3×10^9 bp) is possible!

THE BOOTSTRAP PROBLEM:
- Need accurate replication for long genomes
- Need long genomes for accurate replicases
- How did life cross this barrier?

SOLUTION: HIERARCHICAL EVOLUTION
1. Short RNA (error-prone, no enzymes)
2. Ribozymes improve fidelity
3. Protein enzymes emerge
4. DNA replaces RNA (more stable)
5. Proofreading evolves
6. Long genomes become possible

Each step ENABLES the next through INCREASED HIERARCHY!
""")

    # Calculate error thresholds
    print("\nError Threshold Calculations:")
    print(f"{'Replicator':<25} {'Error Rate':<15} {'Max Genome':<15}")
    print("-" * 55)

    replicators = [
        ("No enzyme", 0.01),
        ("Simple ribozyme", 0.001),
        ("RNA polymerase", 1e-4),
        ("DNA polymerase", 1e-6),
        ("+ Proofreading", 1e-9),
    ]

    for name, error in replicators:
        L_max = int(1 / error)
        print(f"{name:<25} {error:<15.0e} {L_max:<15,}")


def main():
    print("╔" + "═" * 68 + "╗")
    print("║" + " NUCLEOTIDE POLYMERIZATION: DNA AND RNA ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    # 1. Compare RNA vs DNA
    results = compare_rna_dna()

    # 2. Information hierarchy
    information_hierarchy()

    # 3. Replication fidelity
    replication_fidelity()

    # Summary
    print("\n" + "=" * 70)
    print("KEY INSIGHTS: NUCLEIC ACID HIERARCHY")
    print("=" * 70)
    print("""
1. NUCLEIC ACIDS ARE INFORMATION POLYMERS
   - Unlike proteins (structure → function)
   - DNA/RNA: sequence → information → heredity
   - The basis of Darwinian evolution

2. RNA WORLD WAS LIKELY FIRST
   - RNA can store information AND catalyze
   - Ribozymes enable self-replication
   - H_RNA << H_DNA but RNA is more versatile

3. TEMPLATE COPYING IS KEY
   - Base pairing provides specificity
   - Enables accurate copying
   - Information can persist across generations

4. ERROR THRESHOLD LIMITS COMPLEXITY
   - Without enzymes: only ~100 nt genomes
   - Protein enzymes enable longer genomes
   - Proofreading enables human-scale genomes

5. HIERARCHY BOOTSTRAP IN ACTION
   - Short RNA → ribozymes → proteins → DNA
   - Each level enables higher fidelity
   - This is the origin of genetic information!

THE DEEP INSIGHT:
Information storage requires the HIGHEST hierarchy
because errors are CATASTROPHIC for information.

H_protein < H_RNA < H_DNA

This is why DNA is life's memory:
- Most stable (no 2'-OH, no U→C deamination)
- Most accurately copied (10^-9 error rate)
- Highest information capacity (10^9 bp genomes)
""")

    # Save results
    output_dir = Path(__file__).parent.parent.parent / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_data = {
        'nucleotide_comparison': [
            {
                'label': r['label'],
                'is_rna': r['is_rna'],
                'catalysis': float(r['catalysis']),
                'max_length': r['final_stats']['max'],
                'mean_length': r['final_stats']['mean'],
                'gc_content': r['final_stats']['gc_content'],
                'polymerizations': r['polymerizations'],
                'hydrolyses': r['hydrolyses']
            }
            for r in results
        ]
    }

    with open(output_dir / "nucleotide_polymerization.json", 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to: {output_dir}/nucleotide_polymerization.json")

    return results


if __name__ == "__main__":
    main()
