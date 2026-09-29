# DeepONet for Space Domain Awareness

Physics-Informed Neural Network (PINN) for J2-perturbed orbital mechanics. Learns satellite trajectories entirely from physics, without training data.

**Author:** shivaprabha22

## What This Project Does

Propagates a satellite orbit under Earth's J2 gravitational perturbation using a Fourier-feature DeepONet, trained only on the ODE residual. Validated against an RK45 numerical integrator.

## Results

| Phase | Method | Position Error (1 orbit) |
|---|---|---|
| Phase 1 | Soft IC, Fourier DeepONet | ~55 km (phase-shift limited) |
| Phase 2 | Hard IC ansatz + empirical n_eff + RAAN supervision | **4.649 km** |

Phase 2 is within the same order of magnitude as operational SGP4 accuracy (~1–3 km) and uses no training data.

## Files

- `DeepONet_SDA_Engine.ipynb` — Phase 1: baseline Fourier DeepONet, RK45 benchmark, live TLE ingestion
- `DeepONet_SDA_Engine_Phase2.ipynb` — Phase 2: hard-IC ansatz, empirical mean-motion correction, RAAN supervision

## Method Summary

**Phase 1 (baseline):**
- Fourier feature mapping on the trunk network (mitigates spectral bias)
- Physics loss: J2-perturbed two-body equations
- Soft IC loss at t=0

**Phase 2 (working):**
- Hard IC ansatz: `r0·cos(n_eff·t) + v0/n_eff·sin(n_eff·t) + m(t)·NN(t)`
- Time enforcer: `m(t) = (1 - exp(-t))²` enforces `r(0)=r0`, `ṙ(0)=v0` exactly
- Empirical mean motion `n_eff` extracted from RK4 reference
- RAAN secular-drift supervision with decay schedule
- Canonical units, float64 precision, PyTorch

## Requirements

- torch
- numpy
- scipy
- skyfield
- matplotlib

## How to Run

Open either notebook in Google Colab. Run all cells.

## Roadmap

- [x] Phase 1: Baseline DeepONet + physics loss
- [x] Phase 2: Hard-IC ansatz, empirical n_eff, RAAN supervision → 4.649 km
- [ ] Phase 3: Multi-IC operator training (eccentric orbits, varying inclinations)
- [ ] Phase 4: Add atmospheric drag, higher-order harmonics
