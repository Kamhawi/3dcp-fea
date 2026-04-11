"""Standalone setup for the hollow-cylinder physical validation case."""

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
from mesh.hollow_cylinder import HollowCylinderVolumetricMesh
from mesh.mesh_core import FaceLocation
from mesh.mesh_quality import evaluate_mesh_quality
from physics.weak_form import build_evp_cohesive_weak_form
from solver.time_stepper import build_cell_to_dofs_and_support


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
TOP_FACE_LOCAL_NODES = np.array([2, 6, 7, 3], dtype=np.int32)
TOP_FACE_VECTOR_DOF_POSITIONS = np.array(
    [3 * int(node) + comp for node in TOP_FACE_LOCAL_NODES for comp in range(3)],
    dtype=np.int32,
)


@dataclass
class ValidationState:
    """Container for all objects required by the validation runner."""

    cfg: dict
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
    span_indices_dolfinx: np.ndarray
    thickness_indices_dolfinx: np.ndarray
    outer_cell_mask_dolfinx: np.ndarray
    outer_face_centroids_dolfinx: np.ndarray
    top_face_vector_dof_positions: np.ndarray = field(
        default_factory=lambda: TOP_FACE_VECTOR_DOF_POSITIONS.copy()
    )
    comm: object = field(default=None)


def load_validation_config(config_path: Optional[Path] = None) -> dict:
    """Load the cylinder-validation config with local defaults applied."""
    cfg = load_config(config_path or DEFAULT_CONFIG_PATH)
    cfg = deepcopy(cfg)

    checkpoint_cfg = {
        "save_every": 5,
        "resume_enabled": False,
        "directory": "validation/cylinder_print/checkpoints",
    }
    checkpoint_cfg.update(cfg.get("checkpoint", {}))
    cfg["checkpoint"] = checkpoint_cfg

    output_cfg = {
        "directory": "validation/cylinder_print/output",
        "displacement_file": "",
        "cell_data_file": "",
        "log_file": "",
    }
    output_cfg.update(cfg.get("output", {}))
    cfg["output"] = output_cfg

    return cfg


def build_validation_state(
    config_path: Optional[Path] = None,
    comm=None,
):
    """Build the full cylinder-validation state without using main.py."""
    if comm is None:
        comm = MPI.COMM_WORLD

    cfg = load_validation_config(config_path)

    geom = cfg["geometry"]
    mesh_cfg = cfg["mesh"]
    activation_cfg = cfg["activation"]
    bc_cfg = cfg.get("boundary_conditions", {})

    if str(geom.get("type", "")).strip().lower() != "hollow_cylinder":
        raise ValueError(
            "validation/cylinder_print requires geometry.type = hollow_cylinder."
        )

    cylinder_mesh = HollowCylinderVolumetricMesh(
        heartline_radius=geom["heartline_radius"],
        height=geom["height"],
        thickness=geom["thickness"],
        n_span=mesh_cfg["n_span"],
        n_length=mesh_cfg["n_length"],
        n_thickness=mesh_cfg["n_thickness"],
        layer_height=geom.get("layer_height"),
        imperfection_amplitude=geom.get("imperfection_amplitude", 1.0),
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
        raise RuntimeError("Cylinder mesh quality check failed.")

    perm = compute_cell_permutation(msh, cylinder_mesh)
    birth_times_dolfinx = reorder_cell_data(birth_times_array, perm)
    cells_layers_dolfinx = reorder_cell_data(cells_layers, perm)

    span_indices_custom = np.array(
        [float(cell.span_index) for cell in cells_lst], dtype=float
    )
    thickness_indices_custom = np.array(
        [float(cell.thickness_index) for cell in cells_lst], dtype=float
    )
    outer_face_centroids_custom = np.zeros((len(cells_lst), 3), dtype=float)
    for idx, cell in enumerate(cells_lst):
        outer_face_centroids_custom[idx] = cell.get_face(FaceLocation.TOP).compute_centroid(
            cylinder_mesh.nodes
        )

    span_indices_dolfinx = reorder_cell_data(span_indices_custom, perm)
    thickness_indices_dolfinx = reorder_cell_data(thickness_indices_custom, perm)
    outer_face_centroids_dolfinx = reorder_cell_data(outer_face_centroids_custom, perm)

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

    V = fem.functionspace(msh, ("DG", 1, (3,)))
    u = fem.Function(V, name="displacement")
    materials.u = u

    cell_to_dofs, support, _support_missing = build_cell_to_dofs_and_support(
        msh,
        V,
        cylinder_mesh,
        cells_lst,
        perm,
    )

    if comm.rank == 0:
        interlayer_count = int(np.count_nonzero(interior_facet_tags.values == 1))
        intralayer_count = int(np.count_nonzero(interior_facet_tags.values == 2))
        print(f"  Interlayer facets (cohesive): {interlayer_count}", flush=True)
        print(f"  Intralayer facets (bonded): {intralayer_count}", flush=True)
        print("Creating DG function space...", flush=True)
        print(
            f"  Global DOFs: {V.dofmap.index_map.size_global * V.dofmap.index_map_bs}",
            flush=True,
        )
        if _support_missing > 0:
            print(
                f"  WARNING: {_support_missing} interlayer supports could not be mapped.",
                flush=True,
            )
        print(
            "Building nonlinear weak form with EVP bulk and mixed-mode cohesive law...",
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
    )

    ring_time_s = (
        2.0 * np.pi * float(geom["heartline_radius"]) / float(activation_cfg["tcp_speed"])
    )
    layer_completion_times_s = ring_time_s * np.arange(1, mesh_cfg["n_length"] + 1)

    outer_cell_mask_dolfinx = thickness_indices_dolfinx >= (mesh_cfg["n_thickness"] - 1)

    return ValidationState(
        cfg=cfg,
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
        span_indices_dolfinx=span_indices_dolfinx,
        thickness_indices_dolfinx=thickness_indices_dolfinx,
        outer_cell_mask_dolfinx=outer_cell_mask_dolfinx,
        outer_face_centroids_dolfinx=outer_face_centroids_dolfinx,
        comm=comm,
    )
