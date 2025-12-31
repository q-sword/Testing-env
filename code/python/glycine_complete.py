#!/usr/bin/env python3
"""
Complete Glycine Simulation - Remaining Conditions

Runs the final 2 conditions:
1. Dense + catalysis (hydrothermal) - box=5.0, cat=3.0
2. Very dense + strong catalysis - box=4.0, cat=5.0

These represent optimal prebiotic conditions.
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

MASSES = {'H': 1.0, 'C': 12.0, 'N': 14.0, 'O': 16.0}


def get_bond_params(t1, t2):
    for key in [f'{t1}-{t2}', f'{t2}-{t1}', f'{t1}={t2}', f'{t2}={t1}']:
        if key in BONDS:
            return BONDS[key]
    return {'D': 0.1, 'r0': 3.0, 'alpha': 1.0}


def morse_energy(r, D, r0, alpha):
    if r < 0.3:
        return 1e6
    exp_term = np.exp(-alpha * (r - r0))
    return D * (1 - exp_term)**2 - D


class GlycineSim:
    def __init__(self, n_glycine=4, box_size=5.0, catalysis=1.0):
        self.n_glycine = n_glycine
        self.box_size = box_size
        self.catalysis = catalysis
        self.n_N = n_glycine
        self.n_C = 2 * n_glycine
        self.n_O = 2 * n_glycine
        self.n_H = 5 * n_glycine
        self.n_atoms = self.n_N + self.n_C + self.n_O + self.n_H
        self.density = self.n_atoms / box_size**3

        self.atoms = []
        np.random.seed(42)
        for elem, count in [('N', self.n_N), ('C', self.n_C),
                            ('O', self.n_O), ('H', self.n_H)]:
            for _ in range(count):
                pos = np.random.uniform(0.5, box_size - 0.5, 3)
                self.atoms.append({'type': elem, 'pos': pos})

    def compute_energy(self):
        E = 0.0
        n = len(self.atoms)
        for i in range(n):
            for j in range(i + 1, n):
                t1, t2 = self.atoms[i]['type'], self.atoms[j]['type']
                p1, p2 = self.atoms[i]['pos'], self.atoms[j]['pos']
                dr = p2 - p1
                dr = dr - self.box_size * np.round(dr / self.box_size)
                r = np.linalg.norm(dr)
                params = get_bond_params(t1, t2)
                D = params['D'] * self.catalysis
                E += morse_energy(r, D, params['r0'], params['alpha'])
        return E

    def count_bonds(self, r_cut=1.8):
        bonds = {}
        n = len(self.atoms)
        for i in range(n):
            for j in range(i + 1, n):
                t1, t2 = self.atoms[i]['type'], self.atoms[j]['type']
                p1, p2 = self.atoms[i]['pos'], self.atoms[j]['pos']
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
        counts = {k: len(v) for k, v in bonds.items()}
        n_nh = counts.get(('H', 'N'), 0)
        n_ch = counts.get(('C', 'H'), 0)
        n_cn = counts.get(('C', 'N'), 0)
        n_cc = counts.get(('C', 'C'), 0)
        n_co = counts.get(('C', 'O'), 0)
        n_oh = counts.get(('H', 'O'), 0)
        glycine_est = min(n_nh // 2, n_ch // 2, n_cn, n_cc, n_co // 2, n_oh)
        return glycine_est, counts

    def mc_step(self, T, step_size=0.1):
        E_old = self.compute_energy()
        idx = np.random.randint(len(self.atoms))
        old_pos = self.atoms[idx]['pos'].copy()
        new_pos = old_pos + np.random.uniform(-step_size, step_size, 3)
        new_pos = new_pos % self.box_size
        self.atoms[idx]['pos'] = new_pos
        E_new = self.compute_energy()
        if E_new > E_old:
            dE = E_new - E_old
            if np.random.random() > np.exp(-dE / (KB * T)):
                self.atoms[idx]['pos'] = old_pos
                return E_old, False
        return E_new if E_new <= E_old else self.compute_energy(), True

    def run(self, T_high=2500, T_low=350, n_steps=80000):
        print("=" * 70)
        print(f"GLYCINE: box={self.box_size}Å, density={self.density:.2f}, cat={self.catalysis}×")
        print("=" * 70)

        avg_D = np.mean([b['D'] for b in BONDS.values()])
        D_eff = avg_D * self.catalysis
        H_low = D_eff / (KB * T_low)
        print(f"  Effective H at {T_low}K: {H_low:.0f}")

        print(f"\n{'Step':<10} {'T (K)':<10} {'E (eV)':<12} {'Glycine':<10}")
        print("-" * 45)

        for step in range(n_steps):
            T = T_high * (T_low / T_high) ** (step / n_steps)
            step_size = 0.3 * (T / T_high) + 0.03
            E, _ = self.mc_step(T, step_size)

            if step % (n_steps // 10) == 0:
                bonds = self.count_bonds()
                n_gly, _ = self.estimate_glycine(bonds)
                print(f"{step:<10} {T:<10.0f} {E:<12.2f} {n_gly:<10}")

        bonds = self.count_bonds()
        n_glycine, counts = self.estimate_glycine(bonds)
        print("-" * 45)
        print(f"FINAL: {n_glycine} glycine ({100*n_glycine/self.n_glycine:.0f}%)")

        return {
            'n_glycine': n_glycine,
            'target': self.n_glycine,
            'efficiency': n_glycine / self.n_glycine,
            'density': self.density,
            'catalysis': self.catalysis,
            'H_eff': H_low
        }


def main():
    print("╔" + "═" * 68 + "╗")
    print("║" + " GLYCINE: REMAINING CONDITIONS ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    results = []

    # Condition 4: Dense + catalysis (hydrothermal)
    print("\n" + "▶" * 35)
    print("Condition 4: Dense + catalysis (hydrothermal)")
    sim1 = GlycineSim(n_glycine=4, box_size=5.0, catalysis=3.0)
    r1 = sim1.run(T_high=2500, T_low=350, n_steps=20000)
    r1['label'] = 'Dense + catalysis (hydrothermal)'
    results.append(r1)

    # Condition 5: Very dense + strong catalysis
    print("\n" + "▶" * 35)
    print("Condition 5: Very dense + strong catalysis")
    sim2 = GlycineSim(n_glycine=4, box_size=4.0, catalysis=5.0)
    r2 = sim2.run(T_high=2500, T_low=350, n_steps=20000)
    r2['label'] = 'Very dense + strong catalysis'
    results.append(r2)

    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE GLYCINE RESULTS (ALL 5 CONDITIONS)")
    print("=" * 70)

    # Include previous results
    all_results = [
        {'label': 'Sparse, no catalysis', 'density': 0.08, 'catalysis': 1.0,
         'H_eff': 146, 'n_glycine': 11, 'efficiency': 2.75},
        {'label': 'Dense, no catalysis', 'density': 0.32, 'catalysis': 1.0,
         'H_eff': 146, 'n_glycine': 10, 'efficiency': 2.50},
        {'label': 'Sparse, with catalysis', 'density': 0.08, 'catalysis': 3.0,
         'H_eff': 438, 'n_glycine': 10, 'efficiency': 2.50},
    ] + results

    print(f"\n{'Condition':<35} {'Density':<10} {'Cat':<6} {'H_eff':<8} {'Glycine':<10}")
    print("-" * 70)
    for r in all_results:
        eff = r.get('efficiency', r['n_glycine']/4)
        print(f"{r['label']:<35} {r['density']:<10.2f} {r['catalysis']:<6.1f} "
              f"{r['H_eff']:<8.0f} {r['n_glycine']}/4 ({100*eff:.0f}%)")

    # Save results
    output_dir = Path(__file__).parent.parent.parent / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_data = {'conditions': [
        {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
         for k, v in r.items()} for r in all_results
    ]}

    with open(output_dir / "glycine_complete.json", 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to: {output_dir}/glycine_complete.json")
    return results


if __name__ == "__main__":
    main()
