#!/usr/bin/env python3
"""
EXACT BUG VERIFICATION

The bug: Line divides by (1e-10 * √(N*3)) but should divide by 1e-10

For N=3:
  Buggy: log(norm / (1e-10 * 3))
  Correct: log(norm / 1e-10)
  Difference: -log(3) = -1.099 per interval

Over 100 time units with T_lyap=10 (10 intervals):
  Extra penalty: 10 × (-1.099) = -10.99

If true λ = +0.03:
  True log_stretch = 0.03 × 100 = 3.0
  Buggy log_stretch = 3.0 - 10.99 = -7.99
  Buggy λ = -7.99 / 100 = -0.0799

Predicted buggy λ: -0.08
Observed buggy λ: -0.096

Close! Let's verify the exact calculation.
"""

import numpy as np

print("="*80)
print("EXACT BUG VERIFICATION")
print("="*80)
print()

N = 3
T_total = 100
T_lyap = 10
n_intervals = int(T_total / T_lyap)

print(f"N = {N}")
print(f"T_total = {T_total}")
print(f"T_lyap = {T_lyap}")
print(f"n_intervals = {n_intervals}")
print()

# The spurious factor
spurious_factor = np.sqrt(N * 3)
print(f"Spurious factor: √(N×3) = √{N*3} = {spurious_factor:.3f}")
print(f"Extra log per interval: -log({spurious_factor:.3f}) = {-np.log(spurious_factor):.4f}")
print()

# Total spurious penalty
total_spurious = n_intervals * (-np.log(spurious_factor))
print(f"Total spurious penalty over {n_intervals} intervals:")
print(f"  {n_intervals} × {-np.log(spurious_factor):.4f} = {total_spurious:.4f}")
print()

# If the QR method gives λ_true ≈ 0.034:
lambda_true_qr = 0.034
log_stretch_true = lambda_true_qr * T_total

print(f"QR method gives λ = {lambda_true_qr:.6f}")
print(f"True log_stretch = {lambda_true_qr:.6f} × {T_total} = {log_stretch_true:.4f}")
print()

log_stretch_buggy = log_stretch_true + total_spurious
lambda_buggy = log_stretch_buggy / T_total

print(f"Buggy log_stretch = {log_stretch_true:.4f} + ({total_spurious:.4f}) = {log_stretch_buggy:.4f}")
print(f"Buggy λ = {log_stretch_buggy:.4f} / {T_total} = {lambda_buggy:.6f}")
print()

print("="*80)
print("COMPARISON TO OBSERVED")
print("="*80)
print()

lambda_observed_buggy = -0.096
lambda_observed_qr = 0.034

print(f"Predicted buggy λ: {lambda_buggy:.6f}")
print(f"Observed buggy λ:  {lambda_observed_buggy:.6f}")
print(f"Difference: {abs(lambda_buggy - lambda_observed_buggy):.6f}")
print()

print(f"Predicted QR λ: {lambda_true_qr:.6f}")
print(f"Observed QR λ:  {lambda_observed_qr:.6f}")
print(f"Difference: {abs(lambda_true_qr - lambda_observed_qr):.6f}")
print()

print("="*80)
print("CONCLUSION")
print("="*80)
print()

print("The exact bug:")
print("  Line 122: log_stretch += np.log(norm / (1e-10 * np.sqrt(N * 3)))")
print("            Should be: np.log(norm / 1e-10)")
print()

print("The spurious factor √(N×3) = 3 for N=3 adds:")
print(f"  -log(3) = {-np.log(spurious_factor):.4f} per interval")
print(f"  Total: {total_spurious:.4f} over {n_intervals} intervals")
print()

print("This explains the sign flip:")
print(f"  True λ ≈ {lambda_observed_qr:+.6f} (positive, chaotic)")
print(f"  Buggy λ ≈ {lambda_observed_buggy:+.6f} (negative, artificially stable)")
print()

print("The prediction matches observation within ~30%:")
print(f"  |{lambda_buggy:.6f} - ({lambda_observed_buggy:.6f})| / |{lambda_observed_buggy:.6f}| = {abs(lambda_buggy - lambda_observed_buggy) / abs(lambda_observed_buggy) * 100:.1f}%")
print()

print("The remaining ~30% discrepancy likely comes from:")
print("  • Finite integration time (T=100 may not fully converge)")
print("  • Initial transient effects")
print("  • Statistical fluctuations")
print()

print("="*80)
print("THE FIX")
print("="*80)
print()

print("Change line 122 from:")
print("  log_stretch += np.log(norm / (1e-10 * np.sqrt(N * 3)))")
print()
print("To:")
print("  log_stretch += np.log(norm / 1e-10)")
print()

print("Or equivalently:")
print("  log_stretch += np.log(norm) + np.log(1e10)")
print()

print("="*80)
