# DeepONet for Space Domain Awareness

Physics-Informed Neural Network (PINN) for J2-perturbed orbital mechanics. Learns satellite trajectories entirely from physics, without training data.

**Author:** shivaprabha22

## What This Project Does

Propagates a satellite orbit under Earth's J2 gravitational perturbation using a Fourier-feature DeepONet, trained only on the ODE residual. Validated against an RK45 numerical integrator.

## Files

- `DeepONet_SDA_Engine.ipynb` — Phase 1: Architecture, training, RK45 benchmark, live TLE ingestion
- *Phase 2 (in progress)* — Hard-IC ansatz with empirical mean-motion correction, targeting sub-5 km accuracy

## Phase 1 Results

The Fourier-feature DeepONet learns the closed orbital geometry but exhibits phase-shift error — the trajectory is correct in shape but out of sync with the true orbit over long horizons. This is a known limitation of soft-IC, acceleration-only PINN losses and is the motivation for Phase 2.

## Method (Phase 1)

- Fourier feature mapping on the trunk network (mitigates spectral bias)
- Branch/trunk DeepONet architecture
- Physics loss: J2-perturbed two-body equations
- Soft IC loss at t=0
- Validated against RK45 with rtol=1e-9

## Requirements

- torch
- numpy
- scipy
- skyfield
- matplotlib

## How to Run

Open `DeepONet_SDA_Engine.ipynb` in Google Colab. Run all cells.

## Roadmap

- [x] Phase 1: Baseline DeepONet + physics loss
- [ ] Phase 2: Hard-IC ansatz, empirical n_eff, RAAN supervision → target <5 km
- [ ] Phase 3: Multi-IC operator training (eccentric orbits, varying inclinations)
