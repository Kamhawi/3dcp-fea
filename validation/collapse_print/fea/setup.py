# Author: Abdallah Kamhawi <Kamhawi@umich.edu>

"""Standalone setup for the non-planar collapse-print validation case.

Builds the full FEA state (mesh, function space, materials, weak form,
settlement-extraction metadata) for either a CG or DG displacement space,
reusing the framework pipeline without going through ``main.py``. The element
family is the only structural difference between the two runs: DG assembles the
cohesive + SIPG interior-facet terms, CG drops every ``dS`` term and keeps the
bulk EVP + gravity + Nitsche weak-Dirichlet boundary.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from dolfinx import fem
from mpi4py import MPI

from config.config_utils import load_config
from materials.material_state import MaterialStateManager
from mesh.dolfinx_mapping import compute_cell_permutation, reorder_cell_data
from mesh.dolfinx_setup import (
    build_partitioned_dolfinx_mesh,
    tag_interfaces_and_boundaries,
)
from mesh.mesh_core import FaceLocation
from mesh.mesh_quality import evaluate_mesh_quality
from mesh.nonplanar_cylinder import NonPlanarCylinderMesh
from physics.weak_form import build_evp_cohesive_weak_form
from solver.time_stepper import build_cell_to_dofs_and_support


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
CASE_DIR = Path(__file__).resolve().parents[1]

# Layer-top sheet of a hexahedral cell = local nodes 4,5,6,7 (the BACK face,
# i.e. the k+1 node sheet). Their vertical (z) vector-DOF positions within the
# 24-entry per-cell DG1/CG1 vector-DOF block are 3*node + 2.
LAYER_TOP_LOCAL_NODES = np.array([4, 5, 6, 7], dtype=np.int32)
LAYER_TOP_Z_DOF_POSITIONS = np.array(
    [3 * int(node) + 2 for node in LAYER_TOP_LOCAL_NODES], dtype=np.int32
)


@dataclass
class CollapseFEAState:
    """Container for all objects required by the collapse-print runner."""

    cfg: dict
    element: str
    msh: object
    V: object
    u: object
    materials: object
    birth_times_dolfinx: np.ndarray
    birth_time_func: object
    is_active_func: object
    cells_layers_dolfinx: np.ndarray
    cells_layers_func: object
    mpi_rank_func: object
    interior_facet_tags: object
    dirichlet_tags: object
    cell_to_dofs: np.ndarray
    support: np.ndarray
    F_form: object
    J_form: object
    cylinder_mesh: object
    cells_lst: object
    perm: np.ndarray
    ring_time_s: float
    layer_completion_times_s: np.ndarray
    layer_coverage_times_s: np.ndarray
    span_indices_dolfinx: np.ndarray
    front_arc_cell_mask_dolfinx: np.ndarray
    h0_mm: float
    layer_top_z_dof_positions: np.ndarray = field(
        default_factory=lambda: LAYER_TOP_Z_DOF_POSITIONS.copy()
    )
    comm: object = field(default=None)


def load_validation_config(
    config_path: Optional[Path] = None,
    element: Optional[str] = None,
    bonded_control: bool = False,
) -> dict:
    """Load the collapse-print config with local defaults applied."""
    cfg = load_config(config_path or DEFAULT_CONFIG_PATH)
    cfg = deepcopy(cfg)

    if element is not None:
        cfg["element"] = element
    cfg["element"] = str(cfg.get("element", "DG")).strip().upper()
    if cfg["element"] not in ("CG", "DG"):
        raise ValueError(f"element must be CG or DG, got {cfg['element']!r}.")
    if bonded_control:
        if cfg["element"] != "DG":
            raise ValueError("bonded_control requires the DG displacement space.")
        cfg.setdefault("interface", {})["bonded_only"] = True

    checkpoint_cfg = {
        "save_every": 25,
        "resume_enabled": False,
        "directory": "paper/output/collapse_print/checkpoints",
    }
    checkpoint_cfg.update(cfg.get("checkpoint", {}))
    cfg["checkpoint"] = checkpoint_cfg

    output_cfg = {
        "directory": "paper/output/collapse_print/runs",
        "displacement_file": "",
        "cell_data_file": "",
        "log_file": "",
    }
    output_cfg.update(cfg.get("output", {}))
    cfg["output"] = output_cfg

    return cfg


def _front_arc_mask(theta, center_deg, half_width_deg):
    """Boolean mask of azimuth angles within the camera-visible sector."""
    center = np.radians(center_deg)
    half = np.radians(half_width_deg)
    delta = np.angle(np.exp(1j * (theta - center)))  # wrapped to (-pi, pi]
    return np.abs(delta) <= half


def build_validation_state(
    config_path: Optional[Path] = None,
    comm=None,
    element: Optional[str] = None,
    bonded_control: bool = False,
):
    """Build the full collapse-print FEA state without using main.py."""
    if comm is None:
        comm = MPI.COMM_WORLD

    cfg = load_validation_config(
        config_path,
        element=element,
        bonded_control=bonded_control,
    )
    element_family = cfg["element"]

    geom = cfg["geometry"]
    mesh_cfg = cfg["mesh"]
    activation_cfg = cfg["activation"]
    bc_cfg = cfg.get("boundary_conditions", {})

    if str(geom.get("type", "")).strip().lower() != "nonplanar_cylinder":
        raise ValueError(
            "validation/collapse_print/fea requires geometry.type = nonplanar_cylinder."
        )

    slicing_path = CASE_DIR / geom["slicing_json"]
    n_layers = int(geom["n_layers"])
    h0_mm = float(geom["layer_height"])

    cylinder_mesh = NonPlanarCylinderMesh(
        slicing_path,
        n_layers=n_layers,
        layer_height=h0_mm,
        n_thickness=int(mesh_cfg["n_thickness"]),
    )
    cylinder_mesh.assign_dirichlet_boundary_conditions(
        {"base": bc_cfg.get("base_dirichlet_value", 0.0)}
    )

    birth_times_array = cylinder_mesh.compute_birth_times(
        tcp_speed=activation_cfg["tcp_speed"]
    )

    msh, cells_lst, cells_layers = build_partitioned_dolfinx_mesh(
        cylinder_mesh,
        comm,
        partitioner_mode=mesh_cfg.get("partitioner", "strip"),
    )

    quality_report = evaluate_mesh_quality(msh, cfg, comm)
    if not quality_report.is_valid:
        raise RuntimeError("Collapse-print mesh quality check failed.")

    perm = compute_cell_permutation(msh, cylinder_mesh)
    birth_times_dolfinx = reorder_cell_data(birth_times_array, perm)
    cells_layers_dolfinx = reorder_cell_data(cells_layers, perm)

    span_indices_custom = np.array(
        [float(cell.span_index) for cell in cells_lst], dtype=float
    )
    span_indices_dolfinx = reorder_cell_data(span_indices_custom, perm)

    materials = MaterialStateManager(
        msh,
        cfg["material"],
        cfg["hardening"],
        birth_times_dolfinx,
    )

    birth_time_func = fem.Function(materials.V_DG0, name="birth_time")
    birth_time_func.x.array[:] = birth_times_dolfinx
    birth_time_func.x.scatter_forward()

    cells_layers_func = fem.Function(materials.V_DG0, name="layer_id")
    cells_layers_func.x.array[:] = cells_layers_dolfinx
    cells_layers_func.x.scatter_forward()

    is_active_func = fem.Function(materials.V_DG0, name="is_active")
    is_active_func.x.array[:] = 0.0
    is_active_func.x.scatter_forward()

    mpi_rank_func = fem.Function(materials.V_DG0, name="mpi_rank")
    num_owned_cells = msh.topology.index_map(msh.topology.dim).size_local
    mpi_rank_func.x.array[:num_owned_cells] = float(comm.rank)
    mpi_rank_func.x.scatter_forward()

    (
        _interlayer_facet_indices,
        _intralayer_facet_indices,
        interior_facet_tags,
        dirichlet_tags,
    ) = tag_interfaces_and_boundaries(msh, cylinder_mesh)

    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)

    family = "Lagrange" if element_family == "CG" else "Discontinuous Lagrange"
    V = fem.functionspace(msh, (family, 1, (3,)))
    u = fem.Function(V, name="displacement")
    materials.u = u

    cell_to_dofs, support, support_missing = build_cell_to_dofs_and_support(
        msh,
        V,
        cylinder_mesh,
        cells_lst,
        perm,
    )

    if comm.rank == 0:
        interlayer_count = int(np.count_nonzero(interior_facet_tags.values == 1))
        intralayer_count = int(np.count_nonzero(interior_facet_tags.values == 2))
        print(f"  Element family: {element_family}", flush=True)
        interlayer_role = (
            "bonded control"
            if cfg.get("interface", {}).get("bonded_only", False)
            else "cohesive"
        )
        print(f"  Interlayer facets ({interlayer_role}): {interlayer_count}", flush=True)
        print(f"  Intralayer facets (bonded): {intralayer_count}", flush=True)
        print(
            f"  Global DOFs: {V.dofmap.index_map.size_global * V.dofmap.index_map_bs}",
            flush=True,
        )
        if support_missing > 0:
            print(
                f"  WARNING: {support_missing} interlayer supports unmapped.",
                flush=True,
            )

    F_form, J_form = build_evp_cohesive_weak_form(
        msh,
        V,
        materials,
        birth_time_func,
        interior_facet_tags,
        dirichlet_tags,
        cfg,
        include_facet_terms=(element_family == "DG"),
    )

    # Layer completion / coverage timing from the measured wavy path lengths.
    tcp_speed = float(activation_cfg["tcp_speed"])
    ring_periods_s = cylinder_mesh.ring_path_length_mm / tcp_speed
    ring_time_s = float(np.mean(ring_periods_s))
    layer_completion_times_s = np.asarray(
        cylinder_mesh.layer_last_birth_s, dtype=float
    )

    # Coverage reference per layer = mean front-arc birth time of that layer plus
    # one ring period (the nozzle returns one layer up over the same azimuths).
    front_cfg = cfg.get("front_arc", {})
    center_deg = float(front_cfg.get("center_deg", -90.0))
    half_width_deg = float(front_cfg.get("half_width_deg", 60.0))
    theta = cylinder_mesh.azimuth_theta  # (n_span,) reference azimuths
    front_span_mask = _front_arc_mask(theta, center_deg, half_width_deg)

    layer_coverage_times_s = np.zeros(n_layers, dtype=float)
    for layer_idx, layer in enumerate(cylinder_mesh.layers):
        front_births = [
            cell.birth_time
            for cell in layer.cells
            if front_span_mask[cell.span_index]
        ]
        mean_birth = float(np.mean(front_births)) if front_births else float(
            np.mean([c.birth_time for c in layer.cells])
        )
        layer_coverage_times_s[layer_idx] = mean_birth + ring_periods_s[layer_idx]

    front_arc_cell_mask_dolfinx = front_span_mask[
        span_indices_dolfinx.astype(int)
    ]

    return CollapseFEAState(
        cfg=cfg,
        element=element_family,
        msh=msh,
        V=V,
        u=u,
        materials=materials,
        birth_times_dolfinx=birth_times_dolfinx,
        birth_time_func=birth_time_func,
        is_active_func=is_active_func,
        cells_layers_dolfinx=cells_layers_dolfinx,
        cells_layers_func=cells_layers_func,
        mpi_rank_func=mpi_rank_func,
        interior_facet_tags=interior_facet_tags,
        dirichlet_tags=dirichlet_tags,
        cell_to_dofs=cell_to_dofs,
        support=support,
        F_form=F_form,
        J_form=J_form,
        cylinder_mesh=cylinder_mesh,
        cells_lst=cells_lst,
        perm=perm,
        ring_time_s=ring_time_s,
        layer_completion_times_s=layer_completion_times_s,
        layer_coverage_times_s=layer_coverage_times_s,
        span_indices_dolfinx=span_indices_dolfinx,
        front_arc_cell_mask_dolfinx=front_arc_cell_mask_dolfinx,
        h0_mm=h0_mm,
        comm=comm,
    )
