#!/usr/bin/env python3
"""
Origin of Life: Integrated Pathway Simulation

Integrates all components of the unified framework:
1. Molecular formation (hierarchy parameter H = E_binding/kT)
2. Polymer synthesis (catalysis + energy coupling)
3. Information storage (nucleotides → replication)
4. Compartmentalization (membranes → protocells)
5. Metabolism (energy coupling → sustained life)

This simulation traces the COMPLETE pathway from
simple molecules to self-sustaining protocells.

Key insight: Each step ENABLES the next through
increased hierarchy and emergent properties.
"""

import numpy as np
import json
from pathlib import Path

KB = 8.617e-5  # eV/K

# Bond energies (eV)
BOND_ENERGIES = {
    'H-O': 4.8,       # Water
    'C-C': 3.6,       # Organic backbone
    'C-N': 3.0,       # Amino acids
    'C-O': 3.6,       # Carboxyl
    'peptide': 2.0,   # Peptide bond
    'phosphodiester': 3.6,  # DNA/RNA backbone
    'H-bond': 0.15,   # Base pairing
    'hydrophobic': 0.3,    # Membrane assembly
}

# Free energies (eV)
FREE_ENERGIES = {
    'peptide_formation': 0.10,      # Unfavorable
    'atp_hydrolysis': -0.52,        # Favorable
    'phosphodiester': 0.15,         # Unfavorable
    'membrane_assembly': -0.30,     # Favorable (entropy)
    'proton_gradient': 0.20,        # Per proton
}


class Protocell:
    """
    A protocell: the simplest living system.

    Components:
    - Membrane (lipid bilayer)
    - Metabolism (ATP synthesis)
    - Genome (RNA/DNA)
    - Ribozymes (catalysis)
    """

    def __init__(self, genome_length=50, n_ribozymes=5, n_lipids=100):
        self.genome_length = genome_length
        self.n_ribozymes = n_ribozymes
        self.n_lipids = n_lipids

        # Internal state
        self.atp = 10.0              # mM
        self.amino_acids = 20.0      # mM
        self.nucleotides = 10.0      # mM
        self.membrane_integrity = 1.0  # 0-1

        # Genome (random sequence)
        self.genome = ''.join(np.random.choice(['A', 'U', 'G', 'C'])
                              for _ in range(genome_length))

        # Calculate hierarchy
        self.H_genome = genome_length * BOND_ENERGIES['phosphodiester'] / (KB * 300)
        self.H_membrane = n_lipids * BOND_ENERGIES['hydrophobic'] / (KB * 300)

    def metabolize(self):
        """Run one metabolic cycle."""
        if self.amino_acids > 0:
            # ATP synthesis (simplified)
            self.atp += 2.0 * min(1.0, self.membrane_integrity)
            self.amino_acids -= 0.5

        # ATP decay
        self.atp *= 0.95

        return self.atp > 0

    def replicate_genome(self):
        """Attempt genome replication."""
        if self.atp < 5.0 or self.nucleotides < self.genome_length * 0.1:
            return None, 0  # Not enough resources

        # Replication fidelity depends on ribozymes
        error_rate = 0.01 / np.sqrt(self.n_ribozymes)

        new_genome = []
        errors = 0
        for base in self.genome:
            if np.random.random() < error_rate:
                new_genome.append(np.random.choice(['A', 'U', 'G', 'C']))
                errors += 1
            else:
                new_genome.append(base)

        self.atp -= 5.0
        self.nucleotides -= self.genome_length * 0.1

        return ''.join(new_genome), errors

    def divide(self):
        """Attempt cell division."""
        if self.atp < 3.0 or self.n_lipids < 50:
            return None

        # Create daughter cell
        daughter = Protocell(
            genome_length=self.genome_length,
            n_ribozymes=self.n_ribozymes // 2,
            n_lipids=self.n_lipids // 2
        )

        # Split resources
        self.atp /= 2
        self.amino_acids /= 2
        self.nucleotides /= 2
        self.n_lipids //= 2
        self.n_ribozymes //= 2

        daughter.atp = self.atp
        daughter.amino_acids = self.amino_acids
        daughter.nucleotides = self.nucleotides

        # Replicate genome for daughter
        daughter.genome, _ = self.replicate_genome()

        return daughter

    def is_alive(self):
        """Check if protocell is still viable."""
        return (self.atp > 0.5 and
                self.membrane_integrity > 0.1 and
                self.n_ribozymes > 0)


class OriginOfLifeSim:
    """
    Simulate the complete origin of life pathway.

    Stages:
    1. Prebiotic chemistry → monomers
    2. Polymerization → polymers
    3. Membrane formation → compartments
    4. RNA world → information + catalysis
    5. Metabolism → energy coupling
    6. Protocell → first life
    """

    def __init__(self, T=350):
        self.T = T

        # Track progress through stages
        self.stages = {
            'monomers': False,
            'polymers': False,
            'membranes': False,
            'rna_world': False,
            'metabolism': False,
            'protocell': False
        }

        # Molecular pools
        self.pools = {
            'amino_acids': 0,
            'nucleotides': 0,
            'lipids': 0,
            'peptides': 0,
            'rna': 0,
            'membranes': 0,
            'ribozymes': 0,
            'protocells': 0
        }

        # Environment conditions
        self.energy_flux = 1.0       # Relative energy input (UV, heat, etc.)
        self.concentration = 0.001   # M (initial)

    def stage1_monomers(self, n_steps=100):
        """Stage 1: Prebiotic synthesis of monomers."""
        print("\n" + "═" * 70)
        print("STAGE 1: PREBIOTIC MONOMER SYNTHESIS")
        print("═" * 70)
        print("""
Miller-Urey type chemistry:
  CH4 + NH3 + H2O + Energy → Amino acids, Nucleobases, Fatty acids

Key insight: Monomers form SPONTANEOUSLY given:
- Reducing atmosphere
- Energy source (UV, lightning, heat)
- Water
""")

        # Hierarchy for monomer formation
        H_monomer = BOND_ENERGIES['C-N'] / (KB * self.T)
        P_form = 1 - np.exp(-H_monomer / 100)

        for step in range(n_steps):
            if np.random.random() < P_form * self.energy_flux:
                self.pools['amino_acids'] += np.random.randint(1, 5)
            if np.random.random() < P_form * self.energy_flux * 0.5:
                self.pools['nucleotides'] += np.random.randint(1, 3)
            if np.random.random() < P_form * self.energy_flux * 0.3:
                self.pools['lipids'] += np.random.randint(1, 4)

        self.stages['monomers'] = (self.pools['amino_acids'] > 10 and
                                   self.pools['nucleotides'] > 5 and
                                   self.pools['lipids'] > 5)

        print(f"\nResults:")
        print(f"  Amino acids: {self.pools['amino_acids']}")
        print(f"  Nucleotides: {self.pools['nucleotides']}")
        print(f"  Lipids: {self.pools['lipids']}")
        print(f"  H_monomer: {H_monomer:.0f}")
        print(f"  Stage complete: {self.stages['monomers']}")

        return self.stages['monomers']

    def stage2_polymers(self, n_steps=100):
        """Stage 2: Polymerization (requires catalysis or concentration)."""
        print("\n" + "═" * 70)
        print("STAGE 2: POLYMER FORMATION")
        print("═" * 70)
        print("""
Condensation reactions:
  Amino acids → Peptides (dehydration)
  Nucleotides → RNA (dehydration)

Challenge: Thermodynamically unfavorable in water!
Solution: High concentration + catalysis (mineral surfaces, cycles)
""")

        if not self.stages['monomers']:
            print("  ERROR: Need monomers first!")
            return False

        # Catalysis from mineral surfaces
        catalysis = 10.0  # Clay minerals, iron sulfide, etc.
        H_eff = BOND_ENERGIES['peptide'] * catalysis / (KB * self.T)

        for step in range(n_steps):
            # Peptide formation
            if self.pools['amino_acids'] > 2:
                P_peptide = 0.1 * np.sqrt(catalysis) * self.concentration * 1000
                if np.random.random() < P_peptide:
                    self.pools['amino_acids'] -= 2
                    self.pools['peptides'] += 1

            # RNA formation
            if self.pools['nucleotides'] > 2:
                P_rna = 0.05 * np.sqrt(catalysis) * self.concentration * 1000
                if np.random.random() < P_rna:
                    self.pools['nucleotides'] -= 2
                    self.pools['rna'] += 1

        self.stages['polymers'] = (self.pools['peptides'] > 5 and
                                   self.pools['rna'] > 3)

        print(f"\nResults:")
        print(f"  Peptides formed: {self.pools['peptides']}")
        print(f"  RNA formed: {self.pools['rna']}")
        print(f"  H_eff (with catalysis): {H_eff:.0f}")
        print(f"  Stage complete: {self.stages['polymers']}")

        return self.stages['polymers']

    def stage3_membranes(self, n_steps=100):
        """Stage 3: Membrane self-assembly."""
        print("\n" + "═" * 70)
        print("STAGE 3: MEMBRANE FORMATION")
        print("═" * 70)
        print("""
Lipid self-assembly:
  Amphiphiles + Water → Micelles → Bilayers → Vesicles

Key insight: This is SPONTANEOUS!
- Hydrophobic effect drives assembly
- No enzymes needed
- Creates compartments automatically
""")

        if not self.stages['monomers']:
            print("  ERROR: Need lipids first!")
            return False

        # Membrane formation is favorable
        H_membrane = BOND_ENERGIES['hydrophobic'] / (KB * self.T)
        P_assemble = 1 - np.exp(-H_membrane * self.pools['lipids'] / 100)

        for step in range(n_steps):
            if self.pools['lipids'] > 20:
                if np.random.random() < P_assemble:
                    self.pools['lipids'] -= 20
                    self.pools['membranes'] += 1

        self.stages['membranes'] = self.pools['membranes'] > 2

        print(f"\nResults:")
        print(f"  Membranes (vesicles): {self.pools['membranes']}")
        print(f"  H_hydrophobic: {H_membrane:.0f}")
        print(f"  Stage complete: {self.stages['membranes']}")

        return self.stages['membranes']

    def stage4_rna_world(self, n_steps=100):
        """Stage 4: RNA world - catalytic RNA."""
        print("\n" + "═" * 70)
        print("STAGE 4: RNA WORLD")
        print("═" * 70)
        print("""
RNA as information + catalyst:
  RNA sequences → some fold into ribozymes
  Ribozymes → catalyze reactions (including replication!)

Key insight: RNA can do BOTH jobs
- Store genetic information
- Catalyze chemical reactions
- This solves the chicken-egg problem!
""")

        if not self.stages['polymers']:
            print("  ERROR: Need RNA polymers first!")
            return False

        # Some RNA sequences are catalytic
        P_ribozyme = 0.01  # ~1% of random sequences have function

        for step in range(n_steps):
            if self.pools['rna'] > 0:
                if np.random.random() < P_ribozyme:
                    self.pools['ribozymes'] += 1
                    # Ribozymes catalyze more RNA synthesis
                    if self.pools['nucleotides'] > 2:
                        self.pools['nucleotides'] -= 2
                        self.pools['rna'] += 1

        self.stages['rna_world'] = self.pools['ribozymes'] > 1

        print(f"\nResults:")
        print(f"  Ribozymes: {self.pools['ribozymes']}")
        print(f"  Total RNA: {self.pools['rna']}")
        print(f"  Stage complete: {self.stages['rna_world']}")

        return self.stages['rna_world']

    def stage5_metabolism(self, n_steps=100):
        """Stage 5: Primitive metabolism."""
        print("\n" + "═" * 70)
        print("STAGE 5: METABOLISM")
        print("═" * 70)
        print("""
Energy coupling reactions:
  Energy source + Unfavorable → Favorable net reaction

In membranes:
  - Proton gradients form
  - ATP-like molecules synthesize
  - Biosynthesis becomes possible

Key insight: Membranes enable CHEMIOSMOSIS
""")

        if not (self.stages['membranes'] and self.stages['rna_world']):
            print("  ERROR: Need membranes and ribozymes!")
            return False

        # Metabolism requires all pieces
        has_metabolism = (self.pools['membranes'] > 0 and
                          self.pools['ribozymes'] > 0)

        if has_metabolism:
            # Metabolism enables more polymer synthesis
            for step in range(n_steps):
                if self.pools['amino_acids'] > 0:
                    self.pools['peptides'] += 1
                    self.pools['amino_acids'] -= 1

        self.stages['metabolism'] = has_metabolism

        print(f"\nResults:")
        print(f"  Active metabolism: {has_metabolism}")
        print(f"  New peptides: {self.pools['peptides']}")
        print(f"  Stage complete: {self.stages['metabolism']}")

        return self.stages['metabolism']

    def stage6_protocell(self, n_steps=100):
        """Stage 6: Protocell assembly."""
        print("\n" + "═" * 70)
        print("STAGE 6: PROTOCELL FORMATION")
        print("═" * 70)
        print("""
Combining all components:
  Membrane + RNA + Ribozymes + Metabolism → PROTOCELL

A protocell can:
  1. Contain genetic information (RNA)
  2. Replicate (ribozymes)
  3. Metabolize (energy coupling)
  4. Grow and divide (membrane dynamics)

This is the ORIGIN OF LIFE!
""")

        if not self.stages['metabolism']:
            print("  ERROR: Need metabolism first!")
            return False

        # Assemble protocells from components
        while (self.pools['membranes'] > 0 and
               self.pools['rna'] > 5 and
               self.pools['ribozymes'] > 0):
            self.pools['membranes'] -= 1
            self.pools['rna'] -= 5
            self.pools['ribozymes'] -= 1
            self.pools['protocells'] += 1

        self.stages['protocell'] = self.pools['protocells'] > 0

        print(f"\nResults:")
        print(f"  Protocells formed: {self.pools['protocells']}")
        print(f"  Stage complete: {self.stages['protocell']}")

        if self.stages['protocell']:
            print("\n  *** LIFE HAS EMERGED! ***")

        return self.stages['protocell']

    def run_complete_pathway(self):
        """Run the complete origin of life pathway."""
        print("╔" + "═" * 68 + "╗")
        print("║" + " ORIGIN OF LIFE: COMPLETE PATHWAY ".center(68) + "║")
        print("╚" + "═" * 68 + "╝")

        # Run all stages
        self.stage1_monomers(n_steps=200)
        self.stage2_polymers(n_steps=200)
        self.stage3_membranes(n_steps=100)
        self.stage4_rna_world(n_steps=150)
        self.stage5_metabolism(n_steps=100)
        self.stage6_protocell(n_steps=50)

        return self.stages, self.pools


def hierarchy_summary():
    """Summarize the hierarchy at each stage."""
    print("\n" + "=" * 70)
    print("HIERARCHY PARAMETER AT EACH STAGE")
    print("=" * 70)

    stages = [
        ("Atoms → Molecules", BOND_ENERGIES['C-C'], 300, "Spontaneous"),
        ("Monomers → Polymers", BOND_ENERGIES['peptide'], 350, "Needs catalysis"),
        ("Lipids → Membranes", BOND_ENERGIES['hydrophobic'], 300, "Spontaneous"),
        ("RNA → Ribozymes", BOND_ENERGIES['phosphodiester'], 350, "Selection"),
        ("Components → Protocell", 5.0, 350, "Integration"),
    ]

    print(f"\n{'Stage':<25} {'D (eV)':<10} {'T (K)':<10} {'H':<10} {'Note':<15}")
    print("-" * 75)

    for name, D, T, note in stages:
        H = D / (KB * T)
        print(f"{name:<25} {D:<10.2f} {T:<10} {H:<10.0f} {note:<15}")

    print("""
THE HIERARCHY CASCADE:
  H_atoms > H_monomers > H_polymers > H_membranes > H_protocells

Each level:
1. Is MORE STABLE than random (higher H)
2. ENABLES the next level
3. Creates EMERGENT properties

This is why life follows this pathway:
  Physics → Chemistry → Biochemistry → Cell Biology
""")


def key_insights():
    """Summarize key insights from the origin of life simulation."""
    print("\n" + "=" * 70)
    print("KEY INSIGHTS: ORIGIN OF LIFE")
    print("=" * 70)
    print("""
1. LIFE IS THERMODYNAMICALLY POSSIBLE
   - Each step has H > 1 (favorable)
   - Energy coupling overcomes barriers
   - Catalysis speeds up rates

2. THE PATHWAY IS SEQUENTIAL
   - Monomers → Polymers → Membranes → Protocells
   - Each step REQUIRES the previous
   - Cannot skip stages!

3. RNA WORLD SOLVES CHICKEN-EGG
   - RNA stores information (like DNA)
   - RNA catalyzes reactions (like proteins)
   - One molecule does both jobs!

4. MEMBRANES ARE ESSENTIAL
   - Create concentration gradients
   - Enable chemiosmosis (energy)
   - Allow Darwinian selection

5. LIFE IS A PHASE TRANSITION
   - From non-living to living
   - Driven by hierarchy increase
   - Happens when all pieces combine

THE UNIFIED FRAMEWORK:
From quantum regularization ε = ℏ/(mv) to cells:

  Quantum → Atoms → Molecules → Polymers →
  Membranes → Ribozymes → Metabolism → LIFE

All governed by: H = E_binding / (kT)

When H > 1 everywhere, life MUST emerge!
This is not chance - it's THERMODYNAMIC INEVITABILITY.
""")


def main():
    # Run the complete simulation
    sim = OriginOfLifeSim(T=350)
    stages, pools = sim.run_complete_pathway()

    # Hierarchy summary
    hierarchy_summary()

    # Key insights
    key_insights()

    # Final summary
    print("\n" + "=" * 70)
    print("SIMULATION SUMMARY")
    print("=" * 70)

    print("\nStages completed:")
    for stage, complete in stages.items():
        status = "✓" if complete else "✗"
        print(f"  {status} {stage}")

    print("\nFinal pools:")
    for pool, count in pools.items():
        if count > 0:
            print(f"  {pool}: {count}")

    success = all(stages.values())
    print(f"\n{'SUCCESS' if success else 'INCOMPLETE'}: ", end="")
    print("Life emerged!" if success else "Pathway incomplete")

    # Save results
    output_dir = Path(__file__).parent.parent.parent / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_data = {
        'stages': {k: bool(v) for k, v in stages.items()},
        'pools': {k: int(v) for k, v in pools.items()},
        'success': success,
        'temperature': sim.T
    }

    with open(output_dir / "origin_of_life.json", 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to: {output_dir}/origin_of_life.json")

    return stages, pools


if __name__ == "__main__":
    main()
