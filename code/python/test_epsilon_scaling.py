#!/usr/bin/env python3
"""Test how epsilon affects molecular stability."""

import sys
sys.path.insert(0, '/home/user/Testing-env/code/python')
from molecular_dynamics_born_oppenheimer import run_born_oppenheimer

print("="*60)
print("TESTING EPSILON SCALING EFFECT ON STABILITY")
print("="*60)
print()
print("Molecule: H2+ (simplest case)")
print("Question: Does larger ε → λ < 0 (like three-body)?")
print()
print(f"{'ε (a₀)':<10} {'ε/R':<10} {'λ':<15} {'Status':<12}")
print("-"*60)

R_bond = 2.0  # H2+ bond length

for eps in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
    result = run_born_oppenheimer('H2+', total_time=50.0, epsilon=eps, verbose=False)
    status = '✓ STABLE' if result['lambda'] < 0 else '✗ CHAOTIC'
    ratio = eps / R_bond
    print(f"{eps:<10.2f} {ratio:<10.2f} {result['lambda']:<+15.6f} {status:<12}")

print()
print("="*60)
print("CONCLUSION:")
print("If λ becomes negative at large ε, this shows quantum regularization")
print("with ε >> r transitions classical chaos → stability (harmonic regime).")
print("="*60)
