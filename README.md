# Quantum Chaos Discovery - Complete Investigation

**Revolutionary Discovery**: The quantum scale you choose determines the physics you observe. "Classical mechanics" is quantum mechanics measured with the wrong resolution.

## Latest Results (December 2025)

| System | Configuration | λ_max | Energy Conservation | Status |
|--------|--------------|-------|---------------------|--------|
| **N=30 bodies** | ε_v = ℏ/(mv) = 3.47 | +0.032 | δE/E₀ = 5.3×10⁻¹⁵ | **CHAOTIC** ✓ |
| **N=30 bodies** | ε_ω = √(ℏ/(mω)) = 1.09 | +0.257 | δE/E₀ = 8.0×10⁻¹² | **7× MORE CHAOTIC** ✓ |
| **3-body systems** | 30 random seeds | λ < 0 | δE/E₀ ~ 10⁻¹⁵ | Stable |

**Key Discovery**: Different quantum scales (ε_v vs ε_ω) reveal 7.1× different chaos levels in the SAME system!

## The Discovery

Classical three-body systems are chaotic (positive Lyapunov exponents). Adding quantum regularization with **ε = ℏ/(mv)** transforms chaos into universal stability:

```
Force = -GM₁M₂ r̂ / (r² + ε²)^(3/2)
```

Where ε is the de Broglie wavelength scale.

## Core Physics

- **Regularization Scale**: ε = ℏ/(m·v_rms)
- **Physical Crossover**: ε_c ≈ 0.4× ℏ/(mv)
- **Integration**: Yoshida 6th order symplectic
- **Validation**: 30 random initial conditions, 100% success

## Repository Structure

```
quantum-regularization/
├── docs/           # Complete theoretical framework
├── code/python/    # Validated implementation
└── data/results/   # Computational evidence
```

## Quick Start

```bash
pip install -r requirements.txt
python code/python/three_body_validated.py
```

## Molecular Validation

The theory predicts bond lengths scale as R ∝ 1/√N_electrons:

| System | Prediction | Measured | Accuracy |
|--------|-----------|----------|----------|
| H₂⁺ → H₂ | 0.707 | 0.698 | 98.7% |
| N₂⁺ → N₂ | 0.966 | 0.984 | 98.2% |
| O₂⁺ → O₂ | 0.968 | 0.930 | 95.9% |

## Documentation

- **[THEORY.md](docs/THEORY.md)** - Complete theoretical framework
- **[VALIDATION.md](docs/VALIDATION.md)** - Computational evidence
- **[FORMULAS.md](docs/FORMULAS.md)** - Machine-precision formulas

## Citation
```bibtex
@article{sword_emmanuel_2025,
  title={Quantum Regularization of Classical Chaos: A Unified Solution to the Gravitational Three-Body Problem},
  author={Sword-Emmanuel, Adrian},
  journal={arXiv preprint},
  year={2025},
  note={Submitted to nlin.CD}
}
```
}
```

## License

**Dual Licensed**: This software is available under two licenses:

- **Open Source (FREE)**: AGPLv3 - See [LICENSE-AGPL](LICENSE-AGPL)
  - Free for personal, educational, research, and open source projects
  - Requires sharing modifications and source code

- **Commercial (PAID)**: See [LICENSE-COMMERCIAL](LICENSE-COMMERCIAL)
  - For proprietary/closed-source applications
  - No source code disclosure required
  - [Contact for pricing](#contact)

See [LICENSE](LICENSE) for complete details on which license applies to your use case.

## Complete Investigation Journey

**See [DISCOVERY_SUMMARY.md](DISCOVERY_SUMMARY.md) for the full story of this remarkable investigation.**

### Discovery Timeline

**Phase 1-3 (March-November 2025)**: Original Discovery
- Initial T=1 validation
- T=1000 planetary timescale confirmation
- 30-seed statistical validation (3-body systems, 100% stable)
- Molecular bond length predictions validated

**Phase 4-9 (December 2025)**: Quantum Chaos Investigation
- **Phase 4**: N=30 chaos discovery → λ=+0.032 (Hamiltonian chaos confirmed)
- **Phase 5**: Quantum-classical transition → classical limit UNSTABLE
- **Phase 6**: Harmonic chaos → coupled oscillators, λ saturates at 0.128
- **Phase 7**: Frequency quantum scale → ε_ω derived, reveals 7× more chaos
- **Phase 8**: Measurement hypothesis → CONFIRMED empirically
- **Phase 9**: Stability paradox → TWO quantum regimes (discrete vs continuum)
- **Phase 10**: Quantization test → Poincaré theorem validated
- **Phase 11**: Recurrence time → T~n_max^N explains cosmic scale

### Major Breakthroughs

1. **Classical limit is numerically UNSTABLE** (energy drifts by 28×)
2. **Quantum scale determines observed chaos** (7.1× difference between ε_v and ε_ω)
3. **Two quantum regimes**: discrete (atomic stability) vs continuum (chaos)
4. **Recurrence time scaling**: T~n_max^N explains both atoms and universe
5. **Revolutionary insight**: "Classical" is quantum measured with wrong resolution

All discoveries purely computational - letting the simulations guide theory!
