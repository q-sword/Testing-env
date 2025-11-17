# Timestep Optimization Results

**Date:** 2025-11-17
**System:** 30-body quantum-regularized N-body simulation
**Integrator:** Yoshida 6th order symplectic integrator
**Test Duration:** T=50 (extrapolated to T=1000)

## Executive Summary

The timestep testing protocol has identified that **dt=0.00100** provides optimal performance with excellent energy conservation, achieving a **10× speedup** compared to the conservative baseline (dt=0.00010).

## Test Results

| dt      | Steps   | δE/E₀       | Time (s) | T=1000 (min) | Speedup | Status      |
|---------|---------|-------------|----------|--------------|---------|-------------|
| 0.00010 | 500000  | 8.454e-16   | 401.80   | 133.9        | 1.0×    | ✅ EXCELLENT |
| 0.00020 | 250000  | 6.522e-15   | 202.14   | 67.4         | 2.0×    | ✅ EXCELLENT |
| 0.00025 | 200000  | 1.208e-15   | 161.45   | 53.8         | 2.5×    | ✅ EXCELLENT |
| 0.00030 | 166666  | 1.691e-15   | 134.94   | 45.0         | 3.0×    | ✅ EXCELLENT |
| 0.00040 | 125000  | 6.039e-16   | 100.10   | 33.4         | 4.0×    | ✅ EXCELLENT |
| 0.00050 | 100000  | 4.831e-16   | 79.95    | 26.7         | 5.0×    | ✅ EXCELLENT |
| 0.00080 | 62500   | 2.416e-16   | 50.78    | 16.9         | 8.0×    | ✅ EXCELLENT |
| **0.00100** | **50000**   | **1.329e-15**   | **40.33**    | **13.4**         | **10.0×**   | **✅ EXCELLENT** |

## Recommendations

### Primary Recommendation: dt = 0.00100

**Rationale:**
- **Performance:** 10× speedup reduces T=1000 runtime from 134 min to 13.4 min
- **Accuracy:** Energy drift of 1.329e-15 is well within machine precision
- **Safety:** Excellent margin below the δE/E₀ < 1e-13 threshold
- **Scalability:** The Yoshida 6th order integrator's O(dt⁷) error scaling enables large timesteps

**Expected Performance for T=1000:**
- Number of steps: 1,000,000
- Runtime: ~13.4 minutes
- Energy conservation: δE/E₀ ≈ 1e-15

### Conservative Alternative: dt = 0.00050

For applications requiring extra safety margin:
- **Performance:** 5× speedup (T=1000 in 26.7 min)
- **Accuracy:** Energy drift of 4.831e-16 (even better than dt=0.00100)
- **Use case:** Long-term integrations or high-precision requirements

## Key Insights

1. **Yoshida 6th Order Stability:** The high-order symplectic integrator exhibits exceptional stability, maintaining machine-precision energy conservation even with dt=0.001

2. **System-Specific Behavior:** This particular system (ε/⟨r⟩ ratio, energy scales) allows for unusually large timesteps while preserving accuracy

3. **No Accuracy Degradation:** All tested timesteps maintained δE/E₀ < 1e-13 (EXCELLENT threshold), suggesting the system is well-regularized

4. **Linear Speedup:** Runtime scales linearly with timestep as expected (10× larger dt → 10× fewer steps → 10× faster)

## Energy Conservation Criteria

- **δE/E₀ < 1e-13:** EXCELLENT (machine precision) ← All tests achieved this!
- **δE/E₀ < 1e-11:** GOOD (publication quality)
- **δE/E₀ < 1e-10:** ACCEPTABLE (still very accurate)
- **δE/E₀ < 1e-8:**  MARGINAL (use with caution)
- **δE/E₀ > 1e-8:**  TOO LARGE (inaccurate)

## Implementation Notes

1. **Before Production:** Always validate timestep choice on your specific initial conditions
2. **Monitoring:** Track energy drift during long simulations to detect any degradation
3. **Further Optimization:** Results suggest dt could potentially be pushed even larger (e.g., dt=0.0015)

## Yoshida 6th Order Properties

- **Error Scaling:** O(dt⁷) - very favorable for large timesteps
- **Symplectic:** Exactly conserves phase space volume
- **Time-Reversible:** Additional numerical stability
- **Cost per Step:** 8 force evaluations (but allows much larger dt)

## System Parameters (N=30 test)

- **Masses:** m_i = 1.0 (uniform)
- **Initial positions:** Gaussian distribution, σ = 0.5
- **Initial velocities:** Gaussian distribution, σ = 0.3
- **Regularization:** ε = ℏ/v_rms (quantum-inspired)
- **Constants:** G = 1.0, ℏ = 1.0

## Conclusion

The testing demonstrates that **dt=0.00100 is optimal** for this system, providing:
- ✅ 10× performance improvement
- ✅ Machine-precision energy conservation
- ✅ Excellent safety margin
- ✅ Validated over T=50 integration

**Estimated production runtime for T=1000:** ~13-14 minutes (vs. 134 minutes with dt=0.0001)
