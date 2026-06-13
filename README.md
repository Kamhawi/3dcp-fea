# 3DCP-FEA

A distributed-memory, MPI-parallel finite element framework for simulating layer-by-layer **3D concrete printing (3DCP)**, built on [FEniCSx/DOLFINx](https://fenicsproject.org/). The framework models the printing process as a sequence of element activations and captures the competing failure modes of fresh-state printed concrete: plastic yielding of the young material and delamination between deposited layers.

## Physics

- **Implicit displacement solve** on DG1 elements with a symmetric interior penalty (SIPG) formulation by default, plus a CG path for cost/accuracy comparison.
- **Explicit J2–Perzyna viscoplastic** constitutive updates with Bingham-type rheology (`tau_0`, `A_thix`, `mu_p`, `gamma_c`).
- **Age-dependent hardening** — time-varying Young's modulus and yield stress as the material sets (`t_set`, `E_inf`, time-varying ν).
- **Dual interface treatment** (DG): SIPG/Nitsche penalties bond *intra*-layer facets, while a **mixed-mode cohesive law** (`G_Ic`, `G_IIc`, Benzeggagh–Kenane mixing) governs *inter*-layer delamination.
- **Element activation** driven by toolpath kinematics: cell birth times computed from `tcp_speed`, with a tanh stiffness ramp for inactive cells.

**Units convention (everywhere):** lengths in mm, stresses in Pa, density in kg/m³, g = 9810 mm/s², fracture energies in N/mm.

## Installation

There is no `requirements.txt`; the stack is conda-based. Create an environment named `fea`:

```bash
conda create -n fea -c conda-forge fenics-dolfinx pyyaml tqdm scipy matplotlib
# Optional but recommended:
conda install -n fea -c conda-forge numba psutil
```

`fenics-dolfinx` pulls in PETSc, mpi4py, NumPy, and ADIOS2. `numba` JIT-accelerates the Perzyna update (the code degrades gracefully without it); `psutil` enables memory tracking in the Newton solver.

> **macOS note:** the FFCx JIT compiler links C kernels with conda's clang, which needs the env vars set by `conda activate` (e.g. `SDKROOT`). Always `conda activate fea` first — invoking the env's `python` binary directly fails at the JIT link step.

## Quick start

Run everything from the repository root so package imports resolve.

### Barrel vault (primary geometry)

```bash
conda activate fea
python main.py                  # serial
mpirun -np 4 python main.py     # MPI-parallel
```

`main.py` takes **no CLI arguments** — all run-time choices live in [config/config.yaml](config/config.yaml); edit the YAML before running. The main sections:

| Section | Controls |
| --- | --- |
| `geometry` / `mesh` | barrel-vault dimensions, hex divisions, partitioner |
| `boundary_conditions` | Dirichlet support during printing |
| `material` / `hardening` | Bingham/Perzyna rheology, age-dependent stiffening |
| `interface` | SIPG/Nitsche bonding penalties + cohesive fracture parameters |
| `activation` | toolpath `tcp_speed` → birth times, stiffness ramp |
| `solver` | `direct` (MUMPS) or `iterative` (GMRES + ASM/ILU), active-DOF reduction |
| `time_stepping` / `checkpoint` / `output` | step count, resume, output paths |
| `mesh_quality` / `debug` | element-quality gates, opt-in MPI/Newton/IO probes |

### Collapse-print validation case

A designed-to-fail non-planar printed cylinder, compared against an instrumented print experiment (per-layer settlement extracted from video via optical flow). This case has its own runner with CLI arguments:

```bash
python -m validation.collapse_print.fea.run --element DG   # or CG
python -m validation.collapse_print.fea.run --bonded-control  # bonded-DG control
# also: --config, --run-tag, --max-steps (smoke tests); MPI-capable via mpirun
python -m validation.collapse_print.fea.mesh_qa             # mesh QA gate (serial)
```

Figures (after canonical CG, DG, and bonded-DG runs):

```bash
python -m validation.collapse_print.fea.cg_dg_figure
python -m validation.collapse_print.fea.cost_figure
python -m validation.collapse_print.fea.simulation_progress
```

## CG vs DG dual-element support

The two element families share every numeric setting for cost-comparison parity. The structural differences are deliberately confined to three places:

1. [physics/weak_form.py](physics/weak_form.py) — DG assembles the cohesive, bonded-SIPG, and jump-stabilization interior-facet terms; CG skips every `dS` term, leaving bulk EVP + gravity + the Nitsche weak-Dirichlet boundary.
2. [solver/kinematics.py](solver/kinematics.py) — inactive-DOF pinning subtracts active-cell DOFs so CG activation-front nodes shared with an already-born neighbor stay free (an exact no-op for DG).
3. Activation seeding — CG starts newly activated DOFs at zero; DG inherits the support cell's state.

CG additionally requires `mesh.partitioner: strip` for ghost-visibility safety.

## Repository layout

| Path | Purpose |
| --- | --- |
| [main.py](main.py) | Orchestrator (11-step pipeline: config → mesh → activation → state → tagging → spaces → weak form → time loop). No numerics. |
| [config/](config/) | YAML loading, run tags, output paths, config snapshots |
| [mesh/](mesh/) | Geometry-agnostic primitives, parametric barrel-vault and non-planar cylinder hex meshes, DOLFINx partitioning/mapping, mesh quality |
| [materials/](materials/) | `MaterialStateManager` (cell-wise DG0 state), age-dependent property models, damage evolution |
| [physics/](physics/) | `build_evp_cohesive_weak_form` — residual and Jacobian assembly |
| [solver/](solver/) | Time stepper (activation → Newton → Perzyna → output, checkpointing), Newton with Armijo line search and active-DOF reduction, kinematics helpers, writers |
| [validation/collapse_print/](validation/collapse_print/) | Validation case: experiment photos, measured slicing centerlines, FEA runner, settlement extraction, figure scripts |
| [diagnostics/](diagnostics/) | Ad-hoc analysis scripts and the phased `python -m diagnostics` runner |
| `verification/`, `figures/` | Convergence/consistency studies (MMS, patch test, cohesive benchmarks) and paper-figure generators — local only, not tracked in git |

## Outputs

Each run writes to `output/run_<YYMMDD_HHMMSS>/` (collapse-print runs prefix the element family, e.g. `run_DG_<tag>/`):

- `disp.bp`, `cell_data.bp` — VTX (ADIOS2) displacement and cell-state fields, viewable in ParaView
- `step_metrics.csv` — per-step scalars (Newton iterations, max displacement, yielding count, …)
- `results.json` — structured metadata and layer-by-layer snapshots
- `sim_run.log`, `settings_used.yaml` — rank-0 log and config snapshot for reproducibility

Checkpoints (`checkpoint_latest.npz`) support resuming long runs via `checkpoint.resume_enabled`.

## Author

Abdallah Kamhawi — PhD research, University of Michigan.
