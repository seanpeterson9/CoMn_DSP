# Spin Polarization Calculator for Body-Centered Tetragonal CoMn Alloys

A Python pipeline for computing the degree of spin polarization in CoMn alloys from first-principles band structures, using a custom Fermi-surface integration scheme built on top of DFT output (Abinit / Quantum ATK).

## Overview

This project estimates the spin-resolved current density ratio (a good metric for spintronic materials) for body-centered tetragonal (BCT) CoMn alloys across a range of c/a lattice ratios. Rather than relying on a fixed DFT k-point mesh for transport-relevant quantities, the pipeline reconstructs a continuous representation of each band near the Fermi level and performs its own adaptive Fermi-surface integration, giving finer control over the energy window and conductivity model than typical DFT post-processing tools provide.

## What it does

1. **Imports DFT band structures** from Quantum ATK outputs
2. **Builds continuous interpolated band functions** `E(kx, ky, kz)` for every band crossing the Fermi level, using regular-grid interpolation over the irreducible Brillouin zone
3. **Solves for Fermi-surface "sheets"** by root-finding `kz` self-consistently for fixed `(kx, ky)` and a target energy, sweeping a window of energies around the Fermi level (±0.2 eV)
4. **Computes group velocities** analytically from the interpolated band functions and uses them to evaluate the differential surface area element on each sheet
5. **Integrates the density of states (DOS)** and spin-resolved conductivity over the reconstructed Fermi surface, using a relaxation-time approximation where the lifetime is set by the inverse DOS at the Fermi level — including a tunable spin-flip mixing parameter between the two spin channels
6. **Cross-validates** the integrated DOS against the native DFT-code DOS as an internal consistency check
7. **Sweeps lattice parameters** (a, c) to track how spin polarization evolves with tetragonal distortion (c/a ratio)

## Computational methods

- **Band structure interpolation**: Regular-grid trilinear interpolation (`scipy.interpolate.RegularGridInterpolator`) over dense, symmetry-expanded k-point grids, after evaluating tradeoffs against scattered-data interpolators (`LinearNDInterpolator`, RBF) for accuracy and memory scaling
- **Fermi surface reconstruction**: A self-consistent root-finding approach (`scipy.optimize.fsolve`) that solves for the third k-component on a 2D grid, generating equipotential "sheets" for each energy in the integration window, with explicit duplicate-sheet detection for bands with multiple Fermi crossings
- **Numerical integration**: Midpoint-rule surface integration with velocity and surface-area-element evaluation kept self-consistent at the same sample points
- **DOS smoothing**: Gauss-Hermite broadening kernels for converting discrete k-space sums into smooth, differentiable density-of-states curves
- **Symmetry exploitation**: Full use of tetragonal point-group symmetry to reduce the DFT calculation to the irreducible wedge of the Brillouin zone while reconstructing the full zone for integration
- **Validation loop**: Independent comparison between the custom-integrated DOS and the DFT code's native DOS output, used to iteratively diagnose and correct sources of numerical error (interpolation density, broadening width, sampling resolution)

## Tech stack

- **Python**: NumPy, SciPy (interpolation, root-finding, optimization), Matplotlib
- **DFT codes**: Abinit, Quantum ATK
- **Workflow**: Designed for both local iteration and HPC cluster batch submission (see companion MPI README for parallelized multi-configuration runs)

## Project structure

| Component | Purpose |
|---|---|
| Band structure reader | Parses Abinit/Quantum ATK output into structured k-point and eigenvalue arrays |
| Interpolation engine | Builds per-band 3D interpolators and analytic velocity functions over the BZ |
| Fermi surface solver | Generates energy-resolved equipotential sheets via constrained root-finding |
| DOS / conductivity integrator | Performs the surface integration and relaxation-time transport calculation |
| Lattice sweep driver | Automates the calculation across a series of (a, c) lattice parameter pairs |

## Background

Spin polarization quantifies the imbalance between spin-up and spin-down conduction electrons at the Fermi level, and is central to designing materials for spintronic devices such as magnetic tunnel junctions and spin-transfer-torque memory. BCT CoMn alloys are of interest because tetragonal distortion (tuning the c/a ratio) can be used to engineer the electronic structure near the Fermi level and, in turn, the spin polarization — making a fast, flexible, and independently-validated computational pipeline for sweeping this parameter space valuable for materials screening.
