#!/usr/bin/env python3
"""
Lipid Membrane Simulation: Compartmentalization and Protocells

Simulates the self-assembly of amphiphilic molecules into membranes:
1. Lipid monomers (head + tail structure)
2. Micelle formation (spherical aggregates)
3. Bilayer formation (membrane sheets)
4. Vesicle formation (protocells)

Key insight: Membranes create CONCENTRATION GRADIENTS
- Inside vs outside separation
- Enable chemical accumulation
- This is why life needs compartments!

The hierarchy: H_lipid > H_water means lipids prefer each other
"""

import numpy as np
import json
from pathlib import Path

KB = 8.617e-5  # eV/K

# Interaction energies
E_HYDROPHOBIC = 0.3    # eV - tail-tail attraction
E_HYDROPHILIC = 0.2    # eV - head-water attraction
E_HEAD_HEAD = 0.1      # eV - head-head repulsion (electrostatic)
E_TAIL_WATER = 0.4     # eV - tail-water repulsion (hydrophobic effect)


class Lipid:
    """
    Simple amphiphilic lipid model.

    Structure: HEAD---TAIL
    - Head: hydrophilic (loves water)
    - Tail: hydrophobic (hates water)
    """

    def __init__(self, pos, orientation):
        self.pos = np.array(pos)  # Position of center
        self.orientation = np.array(orientation)  # Unit vector head→tail
        self.orientation = self.orientation / np.linalg.norm(self.orientation)
        self.length = 2.0  # nm (typical lipid length)

    @property
    def head_pos(self):
        return self.pos - 0.5 * self.length * self.orientation

    @property
    def tail_pos(self):
        return self.pos + 0.5 * self.length * self.orientation


class LipidMembraneSim:
    """
    Monte Carlo simulation of lipid self-assembly.

    The hydrophobic effect drives membrane formation:
    - Tails avoid water → cluster together
    - Heads face water → form bilayer surfaces
    """

    def __init__(self, n_lipids=100, box_size=30.0, T=300):
        """
        Parameters:
        -----------
        n_lipids : int
            Number of lipid molecules
        box_size : float
            Simulation box size (nm)
        T : float
            Temperature (K)
        """
        self.n_lipids = n_lipids
        self.box_size = box_size
        self.T = T

        # Initialize lipids randomly
        np.random.seed(42)
        self.lipids = []
        for _ in range(n_lipids):
            pos = np.random.uniform(2, box_size - 2, 3)
            orientation = np.random.randn(3)
            self.lipids.append(Lipid(pos, orientation))

        # Calculate hierarchy parameter
        self.H = E_HYDROPHOBIC / (KB * T)

    def compute_energy(self):
        """
        Calculate total system energy.

        Interactions:
        1. Tail-tail: attractive (hydrophobic bonding)
        2. Head-head: repulsive (electrostatic)
        3. Tail exposure: penalty for tails near "water" (box boundary)
        """
        E = 0.0

        for i, lip_i in enumerate(self.lipids):
            # Tail exposure penalty (hydrophobic effect)
            # Tails near edges are "exposed to water"
            tail_i = lip_i.tail_pos
            dist_to_edge = min(
                tail_i[0], self.box_size - tail_i[0],
                tail_i[1], self.box_size - tail_i[1],
                tail_i[2], self.box_size - tail_i[2]
            )
            if dist_to_edge < 3.0:
                E += E_TAIL_WATER * (3.0 - dist_to_edge) / 3.0

            for j in range(i + 1, len(self.lipids)):
                lip_j = self.lipids[j]

                # Tail-tail interaction (attractive)
                dr_tail = lip_j.tail_pos - tail_i
                dr_tail = dr_tail - self.box_size * np.round(dr_tail / self.box_size)
                r_tail = np.linalg.norm(dr_tail)

                if r_tail < 5.0:
                    # Lennard-Jones-like attraction
                    E -= E_HYDROPHOBIC * np.exp(-(r_tail - 1.0)**2 / 2.0)

                    # Alignment bonus (parallel tails more stable)
                    alignment = abs(np.dot(lip_i.orientation, lip_j.orientation))
                    E -= 0.5 * E_HYDROPHOBIC * alignment * np.exp(-r_tail / 2.0)

                # Head-head interaction (repulsive)
                dr_head = lip_j.head_pos - lip_i.head_pos
                dr_head = dr_head - self.box_size * np.round(dr_head / self.box_size)
                r_head = np.linalg.norm(dr_head)

                if r_head < 3.0:
                    E += E_HEAD_HEAD / (r_head + 0.5)

        return E

    def mc_step(self, step_size_pos=0.5, step_size_rot=0.3):
        """Perform one Monte Carlo step."""
        E_old = self.compute_energy()

        # Choose random lipid
        idx = np.random.randint(len(self.lipids))
        lip = self.lipids[idx]
        old_pos = lip.pos.copy()
        old_orient = lip.orientation.copy()

        # Random move (translation + rotation)
        if np.random.random() < 0.7:
            # Translation
            lip.pos = old_pos + np.random.uniform(-step_size_pos, step_size_pos, 3)
            lip.pos = lip.pos % self.box_size
        else:
            # Rotation
            delta = np.random.randn(3) * step_size_rot
            new_orient = old_orient + delta
            lip.orientation = new_orient / np.linalg.norm(new_orient)

        E_new = self.compute_energy()

        # Metropolis criterion
        if E_new > E_old:
            dE = E_new - E_old
            if np.random.random() > np.exp(-dE / (KB * self.T)):
                # Reject
                lip.pos = old_pos
                lip.orientation = old_orient
                return E_old, False

        return E_new, True

    def analyze_structure(self):
        """
        Analyze lipid organization.

        Metrics:
        - Clustering: how many lipids are in aggregates
        - Alignment: how aligned are neighboring tails
        - Bilayer order: heads on outside, tails inside
        """
        # Build neighbor list
        clusters = []
        visited = set()

        for i in range(len(self.lipids)):
            if i in visited:
                continue

            cluster = [i]
            queue = [i]
            visited.add(i)

            while queue:
                current = queue.pop(0)
                for j in range(len(self.lipids)):
                    if j in visited:
                        continue

                    dr = self.lipids[j].tail_pos - self.lipids[current].tail_pos
                    dr = dr - self.box_size * np.round(dr / self.box_size)
                    r = np.linalg.norm(dr)

                    if r < 3.0:  # Within clustering distance
                        cluster.append(j)
                        queue.append(j)
                        visited.add(j)

            clusters.append(cluster)

        # Cluster statistics
        cluster_sizes = [len(c) for c in clusters]
        largest_cluster = max(cluster_sizes)
        n_aggregated = sum(1 for s in cluster_sizes if s > 3)

        # Alignment in largest cluster
        if largest_cluster > 1:
            largest = max(clusters, key=len)
            alignments = []
            for i in range(len(largest)):
                for j in range(i + 1, len(largest)):
                    lip_i = self.lipids[largest[i]]
                    lip_j = self.lipids[largest[j]]
                    alignment = abs(np.dot(lip_i.orientation, lip_j.orientation))
                    alignments.append(alignment)
            avg_alignment = np.mean(alignments) if alignments else 0
        else:
            avg_alignment = 0

        return {
            'n_clusters': len(clusters),
            'largest_cluster': largest_cluster,
            'cluster_sizes': cluster_sizes,
            'aggregated_fraction': sum(s for s in cluster_sizes if s > 3) / self.n_lipids,
            'avg_alignment': avg_alignment
        }

    def run(self, n_steps=30000):
        """Run lipid self-assembly simulation."""
        print("=" * 70)
        print("LIPID MEMBRANE SELF-ASSEMBLY")
        print("=" * 70)
        print(f"\nParameters:")
        print(f"  Lipids: {self.n_lipids}")
        print(f"  Box size: {self.box_size} nm")
        print(f"  Temperature: {self.T} K")
        print(f"  H_hydrophobic: {self.H:.0f}")

        print(f"\n{'Step':<10} {'E (eV)':<12} {'Clusters':<10} {'Largest':<10} {'Aligned':<10}")
        print("-" * 55)

        history = []

        for step in range(n_steps):
            E, _ = self.mc_step()

            if step % (n_steps // 15) == 0:
                stats = self.analyze_structure()
                print(f"{step:<10} {E:<12.2f} {stats['n_clusters']:<10} "
                      f"{stats['largest_cluster']:<10} {stats['avg_alignment']:<10.2f}")
                history.append({
                    'step': step,
                    'energy': E,
                    **stats
                })

        # Final analysis
        print("-" * 55)
        stats = self.analyze_structure()
        print(f"{'FINAL':<10} {E:<12.2f} {stats['n_clusters']:<10} "
              f"{stats['largest_cluster']:<10} {stats['avg_alignment']:<10.2f}")

        return {
            'final_stats': stats,
            'history': history,
            'n_lipids': self.n_lipids,
            'T': self.T
        }


def compare_temperatures():
    """Compare membrane formation at different temperatures."""
    print("\n" + "=" * 70)
    print("TEMPERATURE DEPENDENCE OF MEMBRANE FORMATION")
    print("=" * 70)

    results = []

    for T in [250, 300, 350, 400, 500]:
        print(f"\n▶ T = {T} K (H = {E_HYDROPHOBIC / (KB * T):.0f})")
        sim = LipidMembraneSim(n_lipids=30, box_size=20.0, T=T)
        r = sim.run(n_steps=5000)
        r['T'] = T
        r['H'] = E_HYDROPHOBIC / (KB * T)
        results.append(r)

    # Summary
    print("\n" + "=" * 70)
    print("TEMPERATURE SUMMARY")
    print("=" * 70)
    print(f"\n{'T (K)':<10} {'H':<10} {'Clusters':<12} {'Largest':<12} {'Aggregated':<12}")
    print("-" * 60)

    for r in results:
        stats = r['final_stats']
        print(f"{r['T']:<10} {r['H']:<10.0f} {stats['n_clusters']:<12} "
              f"{stats['largest_cluster']:<12} {stats['aggregated_fraction']*100:<10.0f}%")

    return results


def membrane_hierarchy():
    """Explain the hierarchy of membrane organization."""
    print("\n" + "=" * 70)
    print("MEMBRANE HIERARCHY: FROM LIPIDS TO CELLS")
    print("=" * 70)
    print("""
THE COMPARTMENTALIZATION PROBLEM:
Life requires concentration gradients:
- High [ATP] inside, low outside
- High [K+] inside, low outside
- This requires BARRIERS = membranes

HIERARCHY OF MEMBRANE ORGANIZATION:

Level 1: AMPHIPHILE STRUCTURE
  - Head: hydrophilic (COO-, PO4-, NH3+)
  - Tail: hydrophobic (hydrocarbon chains)
  - This dual nature drives self-assembly

Level 2: MICELLE FORMATION
  - Spherical aggregates in water
  - Tails inside, heads outside
  - Forms spontaneously above CMC (critical micelle concentration)

Level 3: BILAYER SHEETS
  - Two layers, tails facing each other
  - Heads on both surfaces (water-facing)
  - This is the membrane structure

Level 4: VESICLE CLOSURE
  - Bilayer closes into sphere
  - Creates inside/outside separation
  - PROTOCELL: first compartment!

Level 5: FUNCTIONAL MEMBRANE
  - Embedded proteins (channels, pumps)
  - Selective permeability
  - Energy transduction

THE HIERARCHY INSIGHT:
  H_tail-tail > H_tail-water → spontaneous aggregation

The hydrophobic effect DRIVES membrane formation:
- Water doesn't want to touch tails
- This is an ENTROPIC effect (water ordering)
- ΔG_transfer ≈ 3.5 kJ/mol per CH2 group
""")


def why_membranes_matter():
    """Explain why membranes are essential for life."""
    print("\n" + "=" * 70)
    print("WHY MEMBRANES ARE ESSENTIAL FOR LIFE")
    print("=" * 70)
    print("""
THE CONCENTRATION PROBLEM:
Prebiotic chemistry is DILUTE:
- Ocean concentration: ~10^-9 M
- Cell concentration: ~10^-3 M
- Need 10^6 × concentration!

MEMBRANES SOLVE THIS:
1. Create small volumes (femtoliters)
2. Accumulate molecules inside
3. Enable high concentrations

WHAT MEMBRANES ENABLE:

1. METABOLISM
   - Proton gradients (chemiosmosis)
   - ATP synthesis
   - Energy coupling

2. HEREDITY
   - Keep genome inside
   - Prevent dilution
   - Enable replication

3. DARWINIAN SELECTION
   - Individual protocells compete
   - Better protocells grow faster
   - Evolution can begin!

THE ORIGIN OF LIFE SEQUENCE:
  Lipids → Membranes → Protocells → Cells

Without membranes:
- No concentration
- No metabolism
- No heredity
- No evolution
- NO LIFE

WITH membranes:
- High concentration
- Energy coupling
- Genome containment
- Competition
- LIFE!

THE HIERARCHY OF LIFE:
  H_membrane > H_polymer > H_monomer

Each level enables the next:
- Monomers form polymers (information)
- Polymers form catalysts (function)
- Membranes contain both (compartment)
- CELL = information + function + compartment
""")


def main():
    print("╔" + "═" * 68 + "╗")
    print("║" + " LIPID MEMBRANE: COMPARTMENTALIZATION ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    # 1. Basic membrane formation
    print("\n▶ Basic membrane self-assembly")
    sim = LipidMembraneSim(n_lipids=40, box_size=20.0, T=300)
    basic_result = sim.run(n_steps=8000)

    # 2. Temperature comparison
    temp_results = compare_temperatures()

    # 3. Hierarchy explanation
    membrane_hierarchy()

    # 4. Why membranes matter
    why_membranes_matter()

    # Summary
    print("\n" + "=" * 70)
    print("KEY INSIGHTS: MEMBRANE HIERARCHY")
    print("=" * 70)
    print("""
1. MEMBRANES SELF-ASSEMBLE
   - Hydrophobic effect drives aggregation
   - No enzymes needed!
   - H_hydrophobic ≈ 10-15 at room temperature

2. TEMPERATURE CONTROLS ASSEMBLY
   - Low T: frozen, no dynamics
   - Optimal T: fluid bilayer (membrane)
   - High T: disordered, no structure

3. COMPARTMENTALIZATION IS KEY
   - Creates concentration gradients
   - Enables metabolism (chemiosmosis)
   - Contains genome (heredity)
   - Allows competition (evolution)

4. PROTOCELLS ARE INEVITABLE
   - Given amphiphiles + water
   - Membranes form spontaneously
   - Vesicles close automatically
   - Life's container assembles itself!

THE DEEP INSIGHT:
The hydrophobic effect is ENTROPIC, not energetic:
- Water molecules order around hydrophobes
- This decreases entropy (unfavorable)
- Aggregating hydrophobes releases water
- Entropy increases → membrane forms!

This is why membranes are THERMODYNAMICALLY FAVORED.
""")

    # Save results
    output_dir = Path(__file__).parent.parent.parent / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_data = {
        'basic_assembly': {
            'n_lipids': basic_result['n_lipids'],
            'T': basic_result['T'],
            'final_clusters': basic_result['final_stats']['n_clusters'],
            'largest_cluster': basic_result['final_stats']['largest_cluster'],
            'aggregated_fraction': basic_result['final_stats']['aggregated_fraction']
        },
        'temperature_comparison': [
            {
                'T': r['T'],
                'H': r['H'],
                'clusters': r['final_stats']['n_clusters'],
                'largest': r['final_stats']['largest_cluster'],
                'aggregated': r['final_stats']['aggregated_fraction']
            }
            for r in temp_results
        ]
    }

    with open(output_dir / "lipid_membrane.json", 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to: {output_dir}/lipid_membrane.json")

    return basic_result, temp_results


if __name__ == "__main__":
    main()
