# Author: Abdallah Kamhawi <Kamhawi@umich.edu>
# Package Maintainer: Abdallah Kamhawi <Kamhawi@umich.edu>

"""Solver subpackage for transient, nonlinear, and kinematic 3DCP routines.

This package contains:
- kinematic helper operators and activation-aware displacement utilities,
- Newton nonlinear solve utilities with MPI-safe DOF pinning,
- the global time-stepping driver that couples solve, projection, constitutive
  updates, diagnostics, and output.
"""
