#!/usr/bin/env python3
"""
Metabolism Simulation: Energy Coupling Cycles

Simulates the fundamental energy transformations in life:
1. ATP synthesis/hydrolysis cycle
2. Redox reactions (electron transfer)
3. Proton gradients (chemiosmosis)
4. Coupled reactions (thermodynamically unfavorable + favorable)

Key insight: Life is a DISSIPATIVE STRUCTURE
- Constantly consumes energy
- Maintains low entropy internally
- Exports entropy to environment

The hierarchy: H_coupling > H_single determines if life can sustain itself
"""

import numpy as np
import json
from pathlib import Path

KB = 8.617e-5  # eV/K

# Thermodynamic constants (at 300K, pH 7)
DG_ATP_HYDROLYSIS = -0.52  # eV (-50 kJ/mol) - ATP → ADP + Pi
DG_NADH_OXIDATION = -2.2   # eV (-220 kJ/mol) - NADH → NAD+ (full chain)
DG_PROTON_GRADIENT = 0.2   # eV per proton (across membrane)
DG_PEPTIDE_BOND = 0.10     # eV (+10 kJ/mol) - amino acid condensation

# Reaction rates (relative)
K_BASAL = 1e-6  # Uncatalyzed rate
K_ENZYME = 1e6  # Enzyme-catalyzed rate (10^12 enhancement)


class Metabolite:
    """A metabolite with energy content and concentration."""

    def __init__(self, name, dG_formation, concentration):
        self.name = name
        self.dG = dG_formation  # Free energy of formation (eV)
        self.conc = concentration  # Concentration (M)


class Reaction:
    """A metabolic reaction with substrates, products, and coupling."""

    def __init__(self, name, substrates, products, dG, catalyzed=False):
        self.name = name
        self.substrates = substrates  # List of (metabolite_name, stoichiometry)
        self.products = products
        self.dG = dG  # Free energy change (eV)
        self.catalyzed = catalyzed
        self.rate = K_ENZYME if catalyzed else K_BASAL

    def is_favorable(self):
        return self.dG < 0

    def equilibrium_constant(self, T=300):
        return np.exp(-self.dG / (KB * T))


class MetabolismSim:
    """
    Simulate metabolic energy coupling.

    Models:
    1. ATP cycle (synthesis/hydrolysis)
    2. Electron transport (NADH → O2)
    3. Proton pumping (chemiosmosis)
    4. Biosynthesis (using ATP)
    """

    def __init__(self, T=300):
        self.T = T

        # Initialize metabolite pools
        self.metabolites = {
            'ATP': 5.0,      # mM (high energy)
            'ADP': 0.5,      # mM
            'Pi': 5.0,       # mM (phosphate)
            'NADH': 0.5,     # mM (reduced)
            'NAD+': 5.0,     # mM (oxidized)
            'H+_in': 0.1,    # μM (inside cell, pH 7)
            'H+_out': 1.0,   # μM (outside, pH 6)
            'glucose': 5.0,  # mM
            'pyruvate': 0.1, # mM
            'amino_acids': 10.0,  # mM
            'peptides': 0.1,      # mM (polymer units)
        }

        # Define reactions
        self.reactions = {
            'glycolysis': Reaction(
                'Glucose → 2 Pyruvate + 2 ATP + 2 NADH',
                [('glucose', 1), ('NAD+', 2), ('ADP', 2), ('Pi', 2)],
                [('pyruvate', 2), ('NADH', 2), ('ATP', 2)],
                dG=-0.35,  # eV (favorable)
                catalyzed=True
            ),
            'respiration': Reaction(
                'NADH + O2 → NAD+ + H2O (+ proton pumping)',
                [('NADH', 1)],
                [('NAD+', 1)],
                dG=-2.2,  # eV (very favorable)
                catalyzed=True
            ),
            'atp_synthesis': Reaction(
                'ADP + Pi + 3H+_out → ATP + 3H+_in',
                [('ADP', 1), ('Pi', 1), ('H+_out', 3)],
                [('ATP', 1), ('H+_in', 3)],
                dG=-0.08,  # eV (favorable with gradient)
                catalyzed=True
            ),
            'atp_hydrolysis': Reaction(
                'ATP → ADP + Pi (releases energy)',
                [('ATP', 1)],
                [('ADP', 1), ('Pi', 1)],
                dG=-0.52,  # eV (very favorable)
                catalyzed=True
            ),
            'biosynthesis': Reaction(
                'Amino acids + ATP → Peptides + ADP',
                [('amino_acids', 1), ('ATP', 1)],
                [('peptides', 1), ('ADP', 1), ('Pi', 1)],
                dG=-0.42,  # eV (favorable with ATP coupling)
                catalyzed=True
            ),
            'uncoupled_synthesis': Reaction(
                'Amino acids → Peptides (no ATP)',
                [('amino_acids', 1)],
                [('peptides', 1)],
                dG=+0.10,  # eV (unfavorable without ATP!)
                catalyzed=False
            ),
        }

        # Track energy flow
        self.energy_consumed = 0
        self.energy_stored = 0
        self.entropy_exported = 0

    def calculate_reaction_quotient(self, reaction):
        """Calculate Q = [products]/[substrates] for mass action."""
        Q_num = 1.0
        Q_den = 1.0

        for met, stoich in reaction.products:
            Q_num *= self.metabolites.get(met, 1e-6) ** stoich

        for met, stoich in reaction.substrates:
            Q_den *= self.metabolites.get(met, 1e-6) ** stoich

        return Q_num / Q_den if Q_den > 0 else 1e10

    def actual_dG(self, reaction):
        """Calculate actual ΔG including concentration effects."""
        Q = self.calculate_reaction_quotient(reaction)
        return reaction.dG + KB * self.T * np.log(Q + 1e-20)

    def attempt_reaction(self, reaction_name):
        """Attempt to run a metabolic reaction."""
        reaction = self.reactions[reaction_name]

        # Check if substrates available
        for met, stoich in reaction.substrates:
            if self.metabolites.get(met, 0) < stoich * 0.01:
                return False, 0  # Not enough substrate

        # Calculate actual ΔG
        dG_actual = self.actual_dG(reaction)

        # Probability of forward reaction
        if dG_actual < 0:
            P_forward = reaction.rate
        else:
            P_forward = reaction.rate * np.exp(-dG_actual / (KB * self.T))

        if np.random.random() < P_forward:
            # Execute reaction
            for met, stoich in reaction.substrates:
                self.metabolites[met] -= stoich * 0.01

            for met, stoich in reaction.products:
                self.metabolites[met] = self.metabolites.get(met, 0) + stoich * 0.01

            # Track energy
            if dG_actual < 0:
                self.energy_consumed += abs(dG_actual)
                self.entropy_exported += abs(dG_actual) / self.T

            return True, dG_actual

        return False, 0

    def run_metabolism_cycle(self):
        """Run one cycle of metabolism."""
        results = {}

        # 1. Glycolysis (glucose → pyruvate + ATP)
        success, dG = self.attempt_reaction('glycolysis')
        results['glycolysis'] = (success, dG)

        # 2. Respiration (NADH → NAD+ + proton gradient)
        for _ in range(3):  # Multiple rounds per glucose
            success, dG = self.attempt_reaction('respiration')
            results['respiration'] = (success, dG)

        # 3. ATP synthesis (proton gradient → ATP)
        for _ in range(10):  # ~30 ATP per glucose
            success, dG = self.attempt_reaction('atp_synthesis')
            results['atp_synthesis'] = (success, dG)

        # 4. Biosynthesis (ATP → polymers)
        for _ in range(5):
            success, dG = self.attempt_reaction('biosynthesis')
            results['biosynthesis'] = (success, dG)

        return results

    def get_state(self):
        """Get current metabolic state."""
        return {
            'ATP': self.metabolites['ATP'],
            'ADP': self.metabolites['ADP'],
            'ATP_ratio': self.metabolites['ATP'] / (self.metabolites['ADP'] + 0.01),
            'NADH': self.metabolites['NADH'],
            'NAD+': self.metabolites['NAD+'],
            'NADH_ratio': self.metabolites['NADH'] / (self.metabolites['NAD+'] + 0.01),
            'glucose': self.metabolites['glucose'],
            'peptides': self.metabolites['peptides'],
            'energy_consumed': self.energy_consumed,
            'entropy_exported': self.entropy_exported
        }

    def run(self, n_cycles=500):
        """Run metabolic simulation."""
        print("=" * 70)
        print("METABOLIC SIMULATION")
        print("=" * 70)
        print(f"\nTemperature: {self.T} K")
        print(f"Initial ATP: {self.metabolites['ATP']:.2f} mM")
        print(f"Initial glucose: {self.metabolites['glucose']:.2f} mM")

        print(f"\n{'Cycle':<8} {'ATP':<8} {'ATP/ADP':<10} {'Glucose':<10} {'Peptides':<10}")
        print("-" * 50)

        history = []

        for cycle in range(n_cycles):
            self.run_metabolism_cycle()

            if cycle % (n_cycles // 10) == 0:
                state = self.get_state()
                print(f"{cycle:<8} {state['ATP']:<8.2f} {state['ATP_ratio']:<10.1f} "
                      f"{state['glucose']:<10.2f} {state['peptides']:<10.2f}")
                history.append(state)

        # Final state
        print("-" * 50)
        state = self.get_state()
        print(f"{'FINAL':<8} {state['ATP']:<8.2f} {state['ATP_ratio']:<10.1f} "
              f"{state['glucose']:<10.2f} {state['peptides']:<10.2f}")

        print(f"\nEnergy accounting:")
        print(f"  Total energy consumed: {self.energy_consumed:.2f} eV")
        print(f"  Entropy exported: {self.entropy_exported:.2f} eV/K")
        print(f"  Peptides synthesized: {state['peptides']:.2f} mM")

        return {
            'final_state': state,
            'history': history
        }


def compare_coupled_vs_uncoupled():
    """Compare ATP-coupled vs uncoupled biosynthesis."""
    print("\n" + "=" * 70)
    print("COUPLED vs UNCOUPLED REACTIONS")
    print("=" * 70)

    print("""
THE COUPLING PROBLEM:
Biosynthesis (making polymers) is UNFAVORABLE:
  Amino acids → Peptides    ΔG = +0.10 eV (won't happen!)

But ATP hydrolysis is FAVORABLE:
  ATP → ADP + Pi            ΔG = -0.52 eV (releases energy)

COUPLING SOLUTION:
  Amino acids + ATP → Peptides + ADP + Pi
  ΔG_total = +0.10 + (-0.52) = -0.42 eV (favorable!)

This is how life makes polymers thermodynamically possible.
""")

    # Run uncoupled simulation
    print("\n▶ Uncoupled biosynthesis (no ATP)")
    sim_uncoupled = MetabolismSim(T=300)
    for _ in range(100):
        sim_uncoupled.attempt_reaction('uncoupled_synthesis')
    state_uncoupled = sim_uncoupled.get_state()
    print(f"  Peptides formed: {state_uncoupled['peptides']:.3f} mM")

    # Run coupled simulation
    print("\n▶ ATP-coupled biosynthesis")
    sim_coupled = MetabolismSim(T=300)
    for _ in range(100):
        sim_coupled.attempt_reaction('biosynthesis')
    state_coupled = sim_coupled.get_state()
    print(f"  Peptides formed: {state_coupled['peptides']:.3f} mM")
    print(f"  ATP consumed: {5.0 - state_coupled['ATP']:.3f} mM")

    print(f"\nRatio (coupled/uncoupled): {state_coupled['peptides']/max(state_uncoupled['peptides'], 0.001):.0f}×")

    return state_uncoupled, state_coupled


def energy_currency():
    """Explain ATP as universal energy currency."""
    print("\n" + "=" * 70)
    print("ATP: THE UNIVERSAL ENERGY CURRENCY")
    print("=" * 70)
    print("""
WHY ATP?
ATP hydrolysis releases just the right amount of energy:
  ΔG = -0.52 eV (-50 kJ/mol)

This is:
- Enough to drive most biosynthesis (+0.1 to +0.4 eV)
- Not so much that energy is wasted
- Goldilocks energy currency!

ATP CYCLE:
  Energy source (glucose/light)
       ↓
    NADH + O2
       ↓
  Proton gradient
       ↓
     ATP synthesis ←──┐
       ↓              │
     ATP pool         │
       ↓              │
  Biosynthesis ───────┘
  (makes polymers)

THE HIERARCHY:
  H_glucose > H_NADH > H_gradient > H_ATP > H_polymer

Each step:
1. Captures energy
2. Stores in more stable form
3. Loses some to entropy

This is WHY life needs constant energy input:
Entropy always increases, so energy must flow!

LIFE = ORGANIZED ENERGY DISSIPATION
""")


def proton_motive_force():
    """Explain chemiosmosis and the proton gradient."""
    print("\n" + "=" * 70)
    print("CHEMIOSMOSIS: THE PROTON GRADIENT")
    print("=" * 70)
    print("""
THE PROTON GRADIENT:
Across the membrane:
  Outside (intermembrane): high [H+] (pH ~6)
  Inside (matrix): low [H+] (pH ~8)

This creates:
1. Chemical gradient (concentration)
2. Electrical gradient (charge)
Combined: PROTON MOTIVE FORCE (PMF)

PMF = ΔG = 2.3 RT (ΔpH) + F(Δψ)
    ≈ 0.2 eV per proton

HOW IT WORKS:

1. ELECTRON TRANSPORT
   NADH → Complex I → Q → Complex III → Cyt c → Complex IV → O2
   Each step PUMPS protons out

2. PROTON ACCUMULATION
   10 H+ pumped per NADH oxidized
   Creates ~200 mV potential

3. ATP SYNTHESIS
   H+ flows back through ATP synthase
   3 H+ per ATP made
   Mechanical rotation → chemical synthesis

WHY THIS MATTERS:
- Decouples energy source from ATP synthesis
- Stores energy as gradient (capacitor)
- Universal mechanism (all life!)
- Evolved once, kept forever

THE HIERARCHY:
  H_electron_carrier > H_proton_gradient > H_ATP

This is the core of BIOENERGETICS!
""")


def main():
    print("╔" + "═" * 68 + "╗")
    print("║" + " METABOLISM: ENERGY COUPLING CYCLES ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    # 1. Run basic metabolism
    print("\n▶ Running metabolic simulation")
    sim = MetabolismSim(T=300)
    result = sim.run(n_cycles=500)

    # 2. Compare coupled vs uncoupled
    uncoupled, coupled = compare_coupled_vs_uncoupled()

    # 3. Explain ATP
    energy_currency()

    # 4. Explain chemiosmosis
    proton_motive_force()

    # Summary
    print("\n" + "=" * 70)
    print("KEY INSIGHTS: METABOLIC HIERARCHY")
    print("=" * 70)
    print("""
1. LIFE REQUIRES ENERGY COUPLING
   - Unfavorable reactions (biosynthesis) need help
   - Favorable reactions (ATP hydrolysis) provide it
   - Net result: thermodynamically possible!

2. ATP IS THE UNIVERSAL CURRENCY
   - Right energy quantum (~0.5 eV)
   - Kinetically stable (needs enzyme to hydrolyze)
   - Abundant (mM concentrations)

3. PROTON GRADIENTS STORE ENERGY
   - Chemiosmosis is universal
   - Membranes are essential
   - Coupling through ATP synthase

4. LIFE IS DISSIPATIVE
   - Constant energy throughput
   - Local entropy decrease
   - Global entropy increase
   - Satisfies 2nd law!

THE DEEP INSIGHT:
Life is not about having energy.
Life is about COUPLING energy flow to work.

H_coupling = E_favorable + E_unfavorable > 0

This is why ATP coupling was revolutionary:
It made life thermodynamically POSSIBLE.
""")

    # Save results
    output_dir = Path(__file__).parent.parent.parent / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_data = {
        'metabolism': {
            'final_ATP': result['final_state']['ATP'],
            'final_peptides': result['final_state']['peptides'],
            'energy_consumed': result['final_state']['energy_consumed'],
            'entropy_exported': result['final_state']['entropy_exported']
        },
        'coupling_comparison': {
            'uncoupled_peptides': uncoupled['peptides'],
            'coupled_peptides': coupled['peptides'],
            'enhancement': coupled['peptides'] / max(uncoupled['peptides'], 0.001)
        }
    }

    with open(output_dir / "metabolism.json", 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to: {output_dir}/metabolism.json")

    return result


if __name__ == "__main__":
    main()
