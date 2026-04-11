"""Standalone physical validation runner for the hollow-cylinder print case."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Iterable, Optional

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from dolfinx.io import VTXWriter
from mpi4py import MPI
from petsc4py import PETSc

from config.config_utils import (
    build_output_paths,
    build_run_tag,
    get_checkpoint_dir,
    save_run_config_snapshot,
)
from materials.damage_update import update_damage_max_numpy
from materials.material_state import update_perzyna_state_cellwise
from mesh.dolfinx_setup import configure_streaming_stdio
from solver.kinematics import (
    epsilon,
    init_newly_activated_displacement,
    update_active_indicator,
    zero_inactive_cells,
)
from solver.newton import NewtonLinearWorkspace, solve_newton
from solver.time_stepper import _CSV_COLUMNS, _current_rss_gib, _print_step_diagnostics
from validation.cylinder_print.figure import write_comparison_figure
from validation.cylinder_print.setup import build_validation_state, load_validation_config


class _TeeStream:
    """Mirror writes to both the original stream and a log file."""

    def __init__(self, original_stream, log_file):
        self._original_stream = original_stream
        self._log_file = log_file

    def write(self, data):
        self._original_stream.write(data)
        self._log_file.write(data)
        return len(data)

    def flush(self):
        self._original_stream.flush()
        self._log_file.flush()

    def isatty(self):
        return self._original_stream.isatty()

    @property
    def encoding(self):
        return getattr(self._original_stream, "encoding", "utf-8")

    def fileno(self):
        return self._original_stream.fileno()


def _checkpoint_field_layout(field):
    index_map = field.function_space.dofmap.index_map
    bs = field.function_space.dofmap.index_map_bs
    num_blocks_local = index_map.size_local
    size_local = num_blocks_local * bs
    size_global = index_map.size_global * bs
    local_block_indices = np.arange(num_blocks_local, dtype=np.int32)
    global_block_indices = index_map.local_to_global(local_block_indices)
    global_indices = (
        global_block_indices[:, None] * bs + np.arange(bs, dtype=np.int64)[None, :]
    ).reshape(-1)
    return size_local, size_global, global_indices


def _assemble_global_from_owned_chunks(
    gathered_indices,
    gathered_values,
    expected_size,
    field_label,
):
    all_indices = np.concatenate(gathered_indices).astype(np.int64, copy=False)
    all_values = np.concatenate(gathered_values)
    order = np.argsort(all_indices)
    sorted_indices = all_indices[order]
    sorted_values = all_values[order]
    if sorted_indices.size != expected_size:
        raise RuntimeError(
            f"Checkpoint gather size mismatch for {field_label}: "
            f"{sorted_indices.size} vs expected {expected_size}."
        )
    if sorted_indices.size > 0:
        unique_indices = np.unique(sorted_indices)
        if unique_indices.size != expected_size:
            raise RuntimeError(
                f"Checkpoint gather contains duplicate or missing indices for "
                f"{field_label}: unique={unique_indices.size}, expected={expected_size}."
            )
    global_values = np.empty(expected_size, dtype=sorted_values.dtype)
    global_values[sorted_indices] = sorted_values
    return global_values


def _precompute_interlayer_damage_data(msh, interior_facet_tags):
    n_il = 0
    il_cell_plus = np.empty(0, dtype=np.int32)
    il_cell_minus = np.empty(0, dtype=np.int32)
    il_facet_normals = np.empty((0, 3), dtype=np.float64)
    il_h_cells = np.empty(0, dtype=np.float64)

    if interior_facet_tags is None:
        return n_il, il_cell_plus, il_cell_minus, il_facet_normals, il_h_cells

    il_mask = interior_facet_tags.values == 1
    il_facet_indices = interior_facet_tags.indices[il_mask]
    n_il = len(il_facet_indices)
    if n_il == 0:
        return n_il, il_cell_plus, il_cell_minus, il_facet_normals, il_h_cells

    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    msh.topology.create_connectivity(fdim, 0)
    f_to_c = msh.topology.connectivity(fdim, tdim)
    f_to_v = msh.topology.connectivity(fdim, 0)

    il_cell_plus = np.empty(n_il, dtype=np.int32)
    il_cell_minus = np.empty(n_il, dtype=np.int32)
    for i, facet in enumerate(il_facet_indices):
        cells = f_to_c.links(facet)
        il_cell_plus[i] = cells[0]
        il_cell_minus[i] = cells[1] if len(cells) > 1 else cells[0]

    from dolfinx.mesh import entities_to_geometry

    geom_x = msh.geometry.x
    il_facet_normals = np.empty((n_il, 3), dtype=np.float64)
    for i, facet in enumerate(il_facet_indices):
        verts = f_to_v.links(facet)
        geom_dofs = entities_to_geometry(
            msh, 0, verts.astype(np.int32), False
        ).reshape(-1)
        coords = geom_x[geom_dofs]
        normal = np.cross(coords[1] - coords[0], coords[3] - coords[0])
        norm = np.linalg.norm(normal)
        il_facet_normals[i] = normal / norm if norm > 1.0e-12 else normal

    from dolfinx.cpp.mesh import h as dolfinx_h

    map_c = msh.topology.index_map(tdim)
    all_cells = np.arange(map_c.size_local + map_c.num_ghosts, dtype=np.int32)
    il_h_cells = np.asarray(dolfinx_h(msh._cpp_object, tdim, all_cells))
    return n_il, il_cell_plus, il_cell_minus, il_facet_normals, il_h_cells


def _outer_wall_snapshot(state, t_val):
    outer_cells = np.where(
        state.outer_cell_mask_dolfinx & (state.birth_times_dolfinx <= t_val)
    )[0]
    if outer_cells.size == 0:
        local_payload = (
            np.empty((0, 3), dtype=float),
            np.empty((0, 3), dtype=float),
            np.empty(0, dtype=np.int32),
        )
    else:
        face_dofs = state.cell_to_dofs[outer_cells][:, state.top_face_vector_dof_positions]
        u_face = state.u.x.array[face_dofs].reshape(-1, 4, 3).mean(axis=1)
        x0_face = state.outer_face_centroids_dolfinx[outer_cells]
        layer_ids = state.cells_layers_dolfinx[outer_cells].astype(np.int32, copy=False)
        local_payload = (x0_face, u_face, layer_ids)

    gathered = state.comm.gather(local_payload, root=0)
    if state.comm.rank != 0:
        return None

    x0_list = [payload[0] for payload in gathered if payload[0].size > 0]
    u_list = [payload[1] for payload in gathered if payload[1].size > 0]
    layer_list = [payload[2] for payload in gathered if payload[2].size > 0]
    if not x0_list:
        return {
            "max_radial_bulge_mm": 0.0,
            "bulge_height_mm": 0.0,
            "z_profile_mm": [],
            "x_nominal_profile_mm": [],
            "x_deformed_profile_mm": [],
            "r_bulge_profile_mm": [],
        }

    x0_all = np.vstack(x0_list)
    u_all = np.vstack(u_list)
    layer_all = np.concatenate(layer_list)
    x_def = x0_all + u_all
    r_def = np.linalg.norm(x_def[:, :2], axis=1)
    bulge = r_def - float(state.cylinder_mesh.outer_radius)
    z_vals = x0_all[:, 2]

    idx_max = int(np.argmax(bulge))
    max_bulge = float(bulge[idx_max])
    bulge_height = float(z_vals[idx_max])

    z_profile = []
    x_nominal_profile = []
    x_deformed_profile = []
    r_bulge_profile = []
    for layer in sorted(np.unique(layer_all).tolist()):
        mask = layer_all == int(layer)
        z_profile.append(float(np.mean(z_vals[mask])))
        x_nominal_profile.append(float(np.max(x0_all[mask, 0])))
        x_deformed_profile.append(float(np.max(x_def[mask, 0])))
        r_bulge_profile.append(float(np.max(bulge[mask])))

    return {
        "max_radial_bulge_mm": max_bulge,
        "bulge_height_mm": bulge_height,
        "z_profile_mm": z_profile,
        "x_nominal_profile_mm": x_nominal_profile,
        "x_deformed_profile_mm": x_deformed_profile,
        "r_bulge_profile_mm": r_bulge_profile,
    }


def _build_results_template(state, run_tag):
    return {
        "case": "cylinder_print_validation",
        "run_tag": run_tag,
        "geometry": {
            "heartline_radius_mm": float(state.cfg["geometry"]["heartline_radius"]),
            "inner_radius_mm": float(state.cylinder_mesh.inner_radius),
            "outer_radius_mm": float(state.cylinder_mesh.outer_radius),
            "thickness_mm": float(state.cfg["geometry"]["thickness"]),
            "height_mm": float(state.cfg["geometry"]["height"]),
            "layer_height_mm": float(state.cfg["geometry"]["layer_height"]),
            "imperfection_amplitude_mm": float(
                state.cfg["geometry"].get("imperfection_amplitude", 0.0)
            ),
            "n_length": int(state.cfg["mesh"]["n_length"]),
            "n_span": int(state.cfg["mesh"]["n_span"]),
            "n_thickness": int(state.cfg["mesh"]["n_thickness"]),
        },
        "material_mapping": {
            "rho_kg_m3": float(state.cfg["material"]["rho"]),
            "tau_0_pa": float(state.cfg["material"]["tau_0"]),
            "A_thix_pa_s": float(state.cfg["material"]["A_thix"]),
            "mu_p_pa_s": float(state.cfg["material"]["mu_p"]),
            "gamma_c": float(state.cfg["material"]["gamma_c"]),
            "nu_fresh": float(state.cfg["hardening"]["nu_fresh"]),
            "nu_hard": float(state.cfg["hardening"]["nu_hard"]),
            "t_set_s": float(state.cfg["hardening"]["t_set"]),
            "E_inf_MPa": float(state.cfg["hardening"]["E_inf"]),
            "n_h": float(state.cfg["hardening"]["n_h"]),
        },
        "layers": [],
        "milestones": {},
    }


def _run_validation_case(
    state,
    run_tag,
    output_paths,
    simulation_start_time=None,
):
    comm = state.comm
    cfg = state.cfg
    log_path = output_paths["log_path"]
    checkpoint_cfg = cfg["checkpoint"]
    checkpoint_dir = get_checkpoint_dir(cfg)
    checkpoint_save_every = max(1, int(checkpoint_cfg["save_every"]))

    def log_message(message=""):
        if comm.rank != 0:
            return
        print(message, flush=True)
        tee_active = os.environ.get("SIM_TEE_ACTIVE") == "1"
        if tee_active:
            return
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"{message}\n")

    dx = ufl.Measure("dx", domain=state.msh)
    proj_trial = ufl.TrialFunction(state.materials.V_DG0_tensor)
    proj_test = ufl.TestFunction(state.materials.V_DG0_tensor)
    a_proj_form = fem.form(ufl.inner(proj_trial, proj_test) * dx)
    A_proj = assemble_matrix(a_proj_form)
    A_proj.assemble()
    diag_proj = A_proj.getDiagonal()
    inv_diag_proj = diag_proj.copy()
    inv_diag_proj.array[:] = 1.0 / np.maximum(inv_diag_proj.array, 1.0e-30)
    L_strain_form = fem.form(ufl.inner(epsilon(state.u), proj_test) * dx)
    b_strain = create_vector(L_strain_form)

    def project_tensor_to_dg0(out_func):
        with b_strain.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(b_strain, L_strain_form)
        b_strain.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        out_func.x.petsc_vec.pointwiseMult(inv_diag_proj, b_strain)
        out_func.x.scatter_forward()

    (
        n_il,
        il_cell_plus,
        il_cell_minus,
        il_facet_normals,
        il_h_cells,
    ) = _precompute_interlayer_damage_data(state.msh, state.interior_facet_tags)

    u_size_local, u_size_global, u_global_indices = _checkpoint_field_layout(state.u)
    eps_vp = state.materials.eps_vp
    eps_vp_size_local, eps_vp_size_global, eps_vp_global_indices = _checkpoint_field_layout(
        eps_vp
    )
    damage_max_field = state.materials.damage_max
    dmg_size_local, dmg_size_global, dmg_global_indices = _checkpoint_field_layout(
        damage_max_field
    )

    latest_checkpoint_filename = "checkpoint_latest.npz"

    def write_checkpoint_metadata(step, t_val, checkpoint_filename):
        if comm.rank != 0:
            return
        metadata = {
            "step": int(step),
            "time": float(t_val),
            "file_name": Path(checkpoint_filename).name,
            "file_prefix": Path(checkpoint_filename).stem,
        }
        metadata_path = checkpoint_dir / "latest_checkpoint.json"
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")

    def save_checkpoint(step, t_val):
        u_local_values = state.u.x.array[:u_size_local].copy()
        eps_vp_local_values = eps_vp.x.array[:eps_vp_size_local].copy()
        dmg_local_values = damage_max_field.x.array[:dmg_size_local].copy()
        gathered_u_indices = comm.gather(u_global_indices, root=0)
        gathered_u_values = comm.gather(u_local_values, root=0)
        gathered_eps_indices = comm.gather(eps_vp_global_indices, root=0)
        gathered_eps_values = comm.gather(eps_vp_local_values, root=0)
        gathered_dmg_indices = comm.gather(dmg_global_indices, root=0)
        gathered_dmg_values = comm.gather(dmg_local_values, root=0)

        if comm.rank == 0:
            u_global = _assemble_global_from_owned_chunks(
                gathered_u_indices,
                gathered_u_values,
                u_size_global,
                "u",
            )
            eps_vp_global = _assemble_global_from_owned_chunks(
                gathered_eps_indices,
                gathered_eps_values,
                eps_vp_size_global,
                "eps_vp",
            )
            dmg_global = _assemble_global_from_owned_chunks(
                gathered_dmg_indices,
                gathered_dmg_values,
                dmg_size_global,
                "damage_max",
            )
            checkpoint_file = checkpoint_dir / latest_checkpoint_filename
            np.savez(
                checkpoint_file,
                u=u_global,
                eps_vp=eps_vp_global,
                damage_max=dmg_global,
            )
            write_checkpoint_metadata(step, t_val, checkpoint_file.name)
        comm.Barrier()

    time_cfg = cfg.get("time_stepping", {})
    n_steps_cfg = int(time_cfg.get("n_steps", 0))
    start_offset = float(time_cfg.get("start_offset", 0.0))
    end_multiplier = float(time_cfg.get("end_multiplier", 1.0))
    max_steps_cfg = int(time_cfg.get("max_steps", 0))
    t_end = state.layer_completion_times_s[-1] * end_multiplier
    t_start = state.layer_completion_times_s[0] * start_offset if start_offset > 0.0 else 0.0
    if n_steps_cfg > 0 and n_steps_cfg != len(state.layer_completion_times_s):
        sample_times = np.linspace(
            t_start + (t_end - t_start) / n_steps_cfg, t_end, n_steps_cfg
        )
    else:
        sample_times = np.asarray(state.layer_completion_times_s, dtype=float)
    if max_steps_cfg > 0 and len(sample_times) > max_steps_cfg:
        sample_times = sample_times[:max_steps_cfg]
    n_steps = len(sample_times)
    solver_cfg = cfg["solver"]
    debug_cfg = cfg.get("debug", {})
    collective_debug = bool(debug_cfg.get("collective_debug", False))
    collective_debug_max_iter = max(
        1, int(debug_cfg.get("collective_debug_max_iter", 2))
    )
    collective_debug_barrier = bool(debug_cfg.get("collective_debug_barrier", False))
    output_io_debug = bool(debug_cfg.get("output_io", False))
    newton_memory_debug = bool(debug_cfg.get("newton_memory_tracking", False))
    newton_memory_every_iter = max(
        1, int(debug_cfg.get("newton_memory_every_iter", 1))
    )
    newton_memory_collect_garbage = bool(
        debug_cfg.get("newton_memory_collect_garbage", False)
    )
    newton_memory_track_mumps = bool(debug_cfg.get("newton_memory_track_mumps", False))

    newton_workspace = NewtonLinearWorkspace(
        state.V,
        state.msh,
        state.F_form,
        solver_cfg=solver_cfg,
    )
    track_mumps_memory_primary = (
        newton_memory_track_mumps
        and (newton_workspace.linear_solver_mode == "direct")
    )
    newton_workspace.ensure_jacobian(state.J_form)

    n_length = int(cfg["mesh"]["n_length"])
    milestone_layers = set(range(5, n_length + 1, 5))
    results = _build_results_template(state, run_tag=run_tag)

    total_newton_iters = 0
    total_ksp_its = 0
    total_perzyna_steps = 0
    total_time_newton = 0.0
    total_time_perzyna = 0.0
    total_time_proj = 0.0
    total_time_io = 0.0
    final_global_max_u = 0.0
    final_global_min_z = 0.0
    final_global_max_vm = 0.0
    final_global_max_ps = 0.0
    final_global_yielding = 0
    num_owned_cells = state.msh.topology.index_map(state.msh.topology.dim).size_local

    if comm.rank == 0:
        log_message("")
        log_message("Cylinder validation case:")
        log_message("  Geometry: hollow_cylinder")
        log_message(f"  Ring time: {state.ring_time_s:.6f} s")
        log_message(f"  tau_0: {float(cfg['material']['tau_0']):.3f} Pa")
        log_message(f"  A_thix: {float(cfg['material']['A_thix']):.6f} Pa/s")
        log_message(f"  mu_p: {float(cfg['material']['mu_p']):.3f} Pa*s")
        log_message(f"  gamma_c: {float(cfg['material']['gamma_c']):.6f}")
        log_message(
            f"  Imperfection amplitude: "
            f"{float(cfg['geometry'].get('imperfection_amplitude', 0.0)):.3f} mm"
        )

    csv_file = None
    csv_writer = None
    vtx_disp = None
    vtx_cell = None
    loop_start_time = time.time()

    try:
        if comm.rank == 0:
            settings_snapshot_path = save_run_config_snapshot(
                state.cfg,
                output_paths["run_dir"],
            )
            log_message(f"  Config snapshot: {settings_snapshot_path}")

        vtx_disp = VTXWriter(state.msh.comm, str(output_paths["disp_path"]), [state.u])
        vtx_cell = VTXWriter(
            state.msh.comm,
            str(output_paths["cell_path"]),
            [
                state.birth_time_func,
                state.is_active_func,
                state.cells_layers_func,
                state.mpi_rank_func,
                state.materials.E,
                state.materials.G,
                state.materials.nu,
                state.materials.rho,
                state.materials.eta,
                state.materials.tau_y,
                state.materials.sigma_y,
                state.materials.yield_function_trial,
                state.materials.von_mises,
                state.materials.max_principal_stress,
                state.materials.strain,
                state.materials.stress,
                state.materials.eps_vp,
                state.materials.damage_max,
            ],
        )
        vtx_disp.__enter__()
        vtx_cell.__enter__()

        csv_path = output_paths["disp_path"].parent / "step_metrics.csv"
        if comm.rank == 0:
            csv_file = open(csv_path, "w", newline="", encoding="utf-8")
            csv_writer = csv.DictWriter(csv_file, fieldnames=_CSV_COLUMNS)
            csv_writer.writeheader()
            csv_file.flush()

        dt_default = (
            sample_times[1] - sample_times[0] if n_steps > 1 else state.ring_time_s
        )
        if comm.rank == 0:
            log_message(f"\nTime stepping: {len(sample_times)} steps")
            log_message(f"  MPI ranks: {comm.size}")
            log_message(
                f"  Global time range: [{sample_times[0]:.4f}, {sample_times[-1]:.4f}] s"
            )
            log_message(
                f"  Linear solver mode: {newton_workspace.linear_solver_mode}"
            )

        t_prev = 0.0
        for step, t_val in enumerate(sample_times):
            if hasattr(state.materials, "t_current"):
                state.materials.t_current.value = float(t_val)
            dt = float(dt_default if step == 0 else max(float(t_val - t_prev), 0.0))
            t0 = time.time()

            if step == 0:
                state.u.x.array[:] = 0.0
                state.u.x.scatter_forward()
            else:
                newly_active_cells = np.where(
                    (state.birth_times_dolfinx <= t_val)
                    & (state.birth_times_dolfinx > t_prev)
                )[0]
                init_newly_activated_displacement(
                    state.u,
                    newly_active_cells,
                    state.support,
                    state.cell_to_dofs,
                    birth_times=state.birth_times_dolfinx,
                    t_prev=t_prev,
                )
                newly_active_mask = np.zeros_like(state.birth_times_dolfinx, dtype=bool)
                newly_active_mask[newly_active_cells] = True
                if np.any(newly_active_mask):
                    eps_vp_arr = state.materials.eps_vp.x.array.reshape((-1, 3, 3))
                    eps_vp_arr[newly_active_mask, :, :] = 0.0
                    state.materials.damage_max.x.array[newly_active_mask] = 0.0
                state.materials.eps_vp.x.scatter_forward()
                state.materials.damage_max.x.scatter_forward()
                if newly_active_cells.size > 0 and comm.rank == 0:
                    new_dofs = state.cell_to_dofs[newly_active_cells].reshape(-1)
                    u_new = state.u.x.array[new_dofs]
                    log_message(
                        f"  Init disp: {len(newly_active_cells)} new cells, "
                        f"max|u|={np.max(np.abs(u_new)):.4e}, "
                        f"mean|u|={np.mean(np.abs(u_new)):.4e}"
                    )

            state.materials.update_properties(float(t_val))
            _print_step_diagnostics(
                comm,
                num_owned_cells,
                step,
                float(t_val),
                state.birth_times_dolfinx,
                log_message=log_message,
            )
            active_mask = state.birth_times_dolfinx <= t_val
            dt_stable_limit = np.inf
            if np.any(active_mask):
                dt_stable_limit = np.min(
                    state.materials.eta_arr[active_mask]
                    / np.maximum(state.materials.e_arr[active_mask], 1.0e-12)
                )

            u_backup = state.u.x.array.copy()
            t_newton_start = time.time()
            n_iter, converged, msg, newton_stats = solve_newton(
                state.u,
                state.V,
                state.msh,
                state.F_form,
                state.J_form,
                t_val=float(t_val),
                birth_times=state.birth_times_dolfinx,
                cell_to_dofs=state.cell_to_dofs,
                max_iter=solver_cfg["max_iterations"],
                rtol=solver_cfg["rtol"],
                atol=solver_cfg["atol"],
                debug=(comm.rank == 0),
                collective_debug=collective_debug,
                collective_debug_max_iter=collective_debug_max_iter,
                collective_debug_barrier=collective_debug_barrier,
                workspace=newton_workspace,
                memory_tracking=newton_memory_debug,
                memory_tracking_every_iter=newton_memory_every_iter,
                memory_tracking_collect_garbage=newton_memory_collect_garbage,
                memory_tracking_mumps=track_mumps_memory_primary,
            )
            time_newton = time.time() - t_newton_start

            if not converged:
                if comm.rank == 0:
                    log_message("  REVERTING: restoring displacement from before failed step")
                state.u.x.array[:] = u_backup
                state.u.x.scatter_forward()

            zero_inactive_cells(
                state.u,
                float(t_val),
                state.birth_times_dolfinx,
                state.cell_to_dofs,
            )
            update_active_indicator(
                state.is_active_func,
                float(t_val),
                state.birth_times_dolfinx,
            )

            t_proj_start = time.time()
            project_tensor_to_dg0(state.materials.strain)
            time_proj = time.time() - t_proj_start

            strain_arr = state.materials.strain.x.array.reshape((-1, 3, 3))
            eps_vp_prev_arr = state.materials.eps_vp.x.array.reshape((-1, 3, 3))
            dt_eff = float(dt if converged else 0.0)
            if dt_eff > 0.0 and np.isfinite(dt_stable_limit):
                dt_limit = max(float(dt_stable_limit), 1.0e-12)
                n_sub = max(1, int(np.ceil(dt_eff / dt_limit)))
            else:
                n_sub = 1
            sub_dt = dt_eff / n_sub if n_sub > 0 else 0.0

            if comm.rank == 0:
                log_message(
                    f"  Constitutive Update: integrating {n_sub} sub-steps "
                    f"(dt_sub = {sub_dt:.2e}s)..."
                )

            eps_vp_tmp = eps_vp_prev_arr.copy()
            t_perzyna_start = time.time()
            (
                eps_vp_tmp,
                sigma_new_arr,
                vm_new_arr,
                max_ps_new_arr,
                f_trial_arr,
            ) = update_perzyna_state_cellwise(
                strain_total=strain_arr,
                eps_vp_prev=eps_vp_tmp,
                e_arr=state.materials.e_arr,
                nu_arr=state.materials.nu_arr,
                sigma_y_arr=state.materials.sigma_y_arr,
                eta_arr=state.materials.eta_arr,
                dt=sub_dt,
                active_mask=active_mask,
            )

            yielding_mask = (f_trial_arr > 0.0) & active_mask
            n_yielding = int(np.sum(yielding_mask))
            n_active = int(np.sum(active_mask))
            n_sub_computed = n_sub if n_yielding > 0 else 1

            if comm.rank == 0:
                log_message(
                    "  Constitutive Update: integrating "
                    f"{n_sub_computed} sub-steps (dt_sub = {sub_dt:.2e}s), "
                    f"{n_yielding} yielding cells out of {n_active} active"
                )

            if n_yielding > 0 and n_sub > 1:
                strain_yielding = strain_arr[yielding_mask]
                eps_vp_yielding = eps_vp_tmp[yielding_mask]
                e_yielding = state.materials.e_arr[yielding_mask]
                nu_yielding = state.materials.nu_arr[yielding_mask]
                sigma_y_yielding = state.materials.sigma_y_arr[yielding_mask]
                eta_yielding = state.materials.eta_arr[yielding_mask]
                f_trial_yielding = f_trial_arr[yielding_mask].copy()
                for _ in range(2, n_sub + 1):
                    (
                        eps_vp_yielding,
                        _sigma_new_yielding,
                        _vm_new_yielding,
                        _max_ps_new_yielding,
                        f_trial_yielding,
                    ) = update_perzyna_state_cellwise(
                        strain_total=strain_yielding,
                        eps_vp_prev=eps_vp_yielding,
                        e_arr=e_yielding,
                        nu_arr=nu_yielding,
                        sigma_y_arr=sigma_y_yielding,
                        eta_arr=eta_yielding,
                        dt=sub_dt,
                        active_mask=None,
                    )
                eps_vp_tmp[yielding_mask] = eps_vp_yielding
                (
                    eps_vp_tmp,
                    sigma_new_arr,
                    vm_new_arr,
                    max_ps_new_arr,
                    f_trial_dt0_arr,
                ) = update_perzyna_state_cellwise(
                    strain_total=strain_arr,
                    eps_vp_prev=eps_vp_tmp,
                    e_arr=state.materials.e_arr,
                    nu_arr=state.materials.nu_arr,
                    sigma_y_arr=state.materials.sigma_y_arr,
                    eta_arr=state.materials.eta_arr,
                    dt=0.0,
                    active_mask=active_mask,
                )
                f_trial_arr = f_trial_dt0_arr
                f_trial_arr[yielding_mask] = f_trial_yielding

            n_sub = n_sub_computed
            time_perzyna = time.time() - t_perzyna_start

            inactive_mask = ~active_mask
            eps_vp_tmp[inactive_mask, :, :] = 0.0
            sigma_new_arr[inactive_mask, :, :] = 0.0
            vm_new_arr[inactive_mask] = 0.0
            max_ps_new_arr[inactive_mask] = 0.0
            f_trial_arr[inactive_mask] = 0.0

            state.materials.eps_vp.x.array[:] = eps_vp_tmp.reshape(-1)
            state.materials.stress.x.array[:] = sigma_new_arr.reshape(-1)
            state.materials.von_mises.x.array[:] = vm_new_arr
            state.materials.max_principal_stress.x.array[:] = max_ps_new_arr
            state.materials.yield_function_trial.x.array[:] = f_trial_arr
            state.materials.eps_vp.x.scatter_forward()
            state.materials.stress.x.scatter_forward()
            state.materials.von_mises.x.scatter_forward()
            state.materials.max_principal_stress.x.scatter_forward()
            state.materials.yield_function_trial.x.scatter_forward()

            if n_il > 0:
                update_damage_max_numpy(
                    state.u,
                    state.materials,
                    state.cell_to_dofs,
                    state.birth_times_dolfinx,
                    cfg,
                    il_cell_plus,
                    il_cell_minus,
                    il_facet_normals,
                    il_h_cells,
                    active_mask=active_mask,
                )

            u_vec = state.u.x.array.reshape((-1, 3))
            local_max_u = (
                float(np.max(np.linalg.norm(u_vec, axis=1))) if u_vec.size > 0 else 0.0
            )
            local_min_z = (
                float(np.min(state.u.x.array[2::3])) if state.u.x.array[2::3].size > 0 else np.inf
            )
            local_max_vm = (
                float(np.max(state.materials.von_mises.x.array))
                if state.materials.von_mises.x.array.size > 0
                else 0.0
            )
            local_max_ps = (
                float(np.max(state.materials.max_principal_stress.x.array))
                if state.materials.max_principal_stress.x.array.size > 0
                else 0.0
            )
            epsvp_vec = state.materials.eps_vp.x.array.reshape((-1, 9))
            local_max_epsvp = (
                float(np.max(np.linalg.norm(epsvp_vec, axis=1))) if epsvp_vec.size > 0 else 0.0
            )
            local_yielding = int(np.sum((f_trial_arr > 0.0) & active_mask))
            local_max_f = (
                float(np.max(f_trial_arr[active_mask])) if np.any(active_mask) else 0.0
            )

            global_max_u = comm.allreduce(local_max_u, op=MPI.MAX)
            global_min_z = comm.allreduce(local_min_z, op=MPI.MIN)
            global_max_vm = comm.allreduce(local_max_vm, op=MPI.MAX)
            global_max_ps = comm.allreduce(local_max_ps, op=MPI.MAX)
            global_max_epsvp = comm.allreduce(local_max_epsvp, op=MPI.MAX)
            global_yielding = comm.allreduce(local_yielding, op=MPI.SUM)
            global_max_f = comm.allreduce(local_max_f, op=MPI.MAX)

            total_newton_iters += int(n_iter)
            total_ksp_its += int(newton_stats.get("total_ksp_its", 0))
            total_perzyna_steps += int(n_sub)
            total_time_newton += float(time_newton)
            total_time_perzyna += float(time_perzyna)
            total_time_proj += float(time_proj)
            final_global_max_u = float(global_max_u)
            final_global_min_z = float(global_min_z)
            final_global_max_vm = float(global_max_vm)
            final_global_max_ps = float(global_max_ps)
            final_global_yielding = int(global_yielding)

            time_io = 0.0
            t_io_start = time.time()
            vtx_disp.write(float(t_val))
            time_io += time.time() - t_io_start
            t_io_start = time.time()
            vtx_cell.write(float(t_val))
            time_io += time.time() - t_io_start

            is_final_step = step == (n_steps - 1)
            should_save_checkpoint = (
                (step % checkpoint_save_every == 0) or is_final_step
            )
            if should_save_checkpoint:
                t_io_start = time.time()
                save_checkpoint(step, t_val)
                time_io += time.time() - t_io_start
            total_time_io += float(time_io)

            snapshot = _outer_wall_snapshot(state, float(t_val))
            if comm.rank == 0:
                layer_number = int(np.searchsorted(state.layer_completion_times_s, t_val)) + 1
                results["layers"].append(
                    {
                        "step": int(step),
                        "layer": int(layer_number),
                        "time_s": float(t_val),
                        "dt_s": float(dt),
                        "newton_iters": int(n_iter),
                        "newton_converged": bool(converged),
                        "newton_msg": msg,
                        "max_radial_bulge_mm": float(snapshot["max_radial_bulge_mm"]),
                        "bulge_height_mm": float(snapshot["bulge_height_mm"]),
                        "max_downward_disp_mm": float(global_min_z),
                        "max_disp_mm": float(global_max_u),
                        "max_von_mises_MPa": float(global_max_vm),
                        "max_principal_MPa": float(global_max_ps),
                        "max_plastic_strain": float(global_max_epsvp),
                        "yielding_cells": int(global_yielding),
                    }
                )
                if layer_number in milestone_layers and str(layer_number) not in results["milestones"]:
                    results["milestones"][str(layer_number)] = snapshot

            local_active_count = int(np.sum(state.birth_times_dolfinx[:num_owned_cells] <= t_val))
            n_active_csv = comm.allreduce(local_active_count, op=MPI.SUM)
            n_total_csv = comm.allreduce(int(num_owned_cells), op=MPI.SUM)
            n_active_ranks = comm.allreduce(1 if local_active_count > 0 else 0, op=MPI.SUM)

            if csv_writer is not None:
                elapsed_step = time.time() - t0
                csv_writer.writerow(
                    {
                        "step": step,
                        "time_s": f"{t_val:.6f}",
                        "dt_s": f"{dt:.6e}",
                        "mpi_ranks_total": comm.size,
                        "mpi_ranks_active": n_active_ranks,
                        "active_cells": n_active_csv,
                        "total_cells": n_total_csv,
                        "active_dofs": newton_stats.get("n_active_dofs", ""),
                        "inactive_dofs": newton_stats.get("n_inactive_dofs", ""),
                        "newton_iters": n_iter,
                        "converged": int(converged),
                        "newton_msg": msg,
                        "res0": f"{newton_stats.get('res0', float('nan')):.6e}",
                        "res_final": f"{newton_stats.get('res_final', float('nan')):.6e}",
                        "rel_res_final": f"{newton_stats.get('rel_res_final', float('nan')):.6e}",
                        "total_ksp_its": newton_stats.get("total_ksp_its", 0),
                        "total_ls_its": newton_stats.get("total_ls_its", 0),
                        "final_omega": f"{newton_stats.get('final_omega', 1.0):.4f}",
                        "time_newton_s": f"{time_newton:.4f}",
                        "time_perzyna_s": f"{time_perzyna:.4f}",
                        "time_proj_s": f"{time_proj:.4f}",
                        "time_io_s": f"{time_io:.4f}",
                        "time_step_total_s": f"{time.time() - t0:.4f}",
                        "n_sub_steps": n_sub,
                        "yielding_cells": global_yielding,
                        "max_yield_f_MPa": f"{global_max_f:.6e}",
                        "max_disp_mm": f"{global_max_u:.6e}",
                        "z_sag_mm": f"{global_min_z:.6e}",
                        "max_von_mises_MPa": f"{global_max_vm:.6e}",
                        "max_principal_MPa": f"{global_max_ps:.6e}",
                        "max_plastic_strain": f"{global_max_epsvp:.6e}",
                        "dt_stable_limit_s": (
                            f"{dt_stable_limit:.6e}" if np.isfinite(dt_stable_limit) else ""
                        ),
                        "rss_GiB": f"{_current_rss_gib():.3f}",
                        "cumul_newton_iters": total_newton_iters,
                        "cumul_ksp_its": total_ksp_its,
                        "cumul_wall_s": f"{time.time() - loop_start_time:.2f}",
                    }
                )
                csv_file.flush()

            if comm.rank == 0:
                elapsed_step = time.time() - t0
                log_message(f"\n  RESULT: {msg} in {n_iter} iters ({elapsed_step:.2f}s)")
                log_message(f"  Step Summary (t = {t_val:.4f} s):")
                log_message("  \u251c\u2500 Timings:")
                log_message(f"  \u2502    Newton Solver:   {time_newton:.2f} s")
                log_message(f"  \u2502    Perzyna Update:  {time_perzyna:.2f} s")
                log_message(f"  \u2502    DG0 Projection:  {time_proj:.2f} s")
                log_message(f"  \u2502    File I/O:        {time_io:.2f} s")
                log_message("  \u2514\u2500 Global Statistics:")
                log_message(
                    f"       Yielding Cells:  {global_yielding} "
                    f"(Max f_trial: {global_max_f:.2e} MPa)"
                )
                log_message(
                    f"       Max Disp (|u|):  {global_max_u:.2e} mm "
                    f"(Z-sag: {global_min_z:.2e} mm)"
                )
                log_message(f"       Max VM Stress:   {global_max_vm:.2e} MPa")
                log_message(
                    f"       Max Principal:   {global_max_ps:.2e} MPa (Tension)"
                )
                log_message(f"       Max Plastic Str: {global_max_epsvp:.2e}")
                log_message(
                    f"       Max Bulge:       {snapshot['max_radial_bulge_mm']:.2e} mm "
                    f"at z={snapshot['bulge_height_mm']:.2e} mm"
                )
                if dt_eff > 0.0 and np.isfinite(dt_stable_limit):
                    log_message(
                        f"       Explicit limit:  dt_sub <= {dt_stable_limit:.2e} s (eta/E)"
                    )

            t_prev = float(t_val)

        results_path = output_paths["run_dir"] / "results.json"
        png_path = output_paths["run_dir"] / "comparison_layers_5_10_15_20_25_30.png"
        pdf_path = output_paths["run_dir"] / "comparison_layers_5_10_15_20_25_30.pdf"

        if comm.rank == 0:
            with open(results_path, "w", encoding="utf-8") as handle:
                json.dump(results, handle, indent=2)
                handle.write("\n")
            try:
                write_comparison_figure(results, png_path=png_path, pdf_path=pdf_path)
            except Exception as exc:
                log_message(f"  WARNING: comparison figure generation failed: {exc}")

            total_loop_time = time.time() - loop_start_time
            total_simulation_time = (
                total_loop_time
                if simulation_start_time is None
                else (time.time() - float(simulation_start_time))
            )
            total_overhead = total_loop_time - (
                total_time_newton + total_time_perzyna + total_time_proj + total_time_io
            )
            log_message("")
            log_message("  ========================================================================")
            log_message("  SIMULATION COMPLETE")
            log_message("  ========================================================================")
            log_message("  Final Structural State:")
            log_message(
                f"    Max Disp (|u|):      {final_global_max_u:.2e} mm "
                f"(Z-sag: {final_global_min_z:.2e} mm)"
            )
            log_message(f"    Max VM Stress:       {final_global_max_vm:.2e} MPa")
            log_message(f"    Max Principal:       {final_global_max_ps:.2e} MPa")
            log_message(f"    Yielding Cells:      {final_global_yielding}")
            if results["layers"]:
                final_layer = results["layers"][-1]
                log_message(
                    f"    Max Bulge:           {final_layer['max_radial_bulge_mm']:.2e} mm "
                    f"at z={final_layer['bulge_height_mm']:.2e} mm"
                )
            log_message("  ")
            log_message("  Computational Effort:")
            log_message(f"    Total Time Steps:    {n_steps}")
            log_message(f"    Total Newton Iters:  {total_newton_iters}")
            log_message(f"    Total Perzyna Steps: {total_perzyna_steps}")
            log_message("  ")
            log_message("  Timing Breakdown:")
            log_message(f"    Total Simulation Time:  {total_simulation_time:.2f} s")
            log_message(f"    Total Loop Time:     {total_loop_time:.2f} s")
            log_message(f"    \u251c\u2500 Newton Solver:    {total_time_newton:.2f} s")
            log_message(f"    \u251c\u2500 Perzyna Update:   {total_time_perzyna:.2f} s")
            log_message(f"    \u251c\u2500 DG0 Projection:   {total_time_proj:.2f} s")
            log_message(f"    \u251c\u2500 File I/O:         {total_time_io:.2f} s")
            log_message(f"    \u2514\u2500 Overhead (MPI/Misc):  {total_overhead:.2f} s")
            log_message("  ========================================================================")
            log_message(f"  Results JSON:        {results_path}")
            log_message(f"  Comparison PNG:      {png_path}")
            log_message(f"  Log saved to: {log_path}")
            log_message(f"  CSV metrics saved to: {csv_path}")

        comm.Barrier()
        return results if comm.rank == 0 else None
    finally:
        if csv_file is not None:
            csv_file.close()
        if vtx_cell is not None:
            vtx_cell.__exit__(None, None, None)
        if vtx_disp is not None:
            vtx_disp.__exit__(None, None, None)
        newton_workspace.destroy()
        A_proj.destroy()
        b_strain.destroy()
        inv_diag_proj.destroy()
        diag_proj.destroy()


def _prepare_output_bundle(cfg, run_tag, comm):
    disp_path, cell_path, log_path = build_output_paths(cfg, run_tag=run_tag)
    run_dir = disp_path.parent
    checkpoint_dir = get_checkpoint_dir(cfg)

    if comm.rank == 0:
        if not cfg["checkpoint"]["resume_enabled"] and checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    return {
        "disp_path": disp_path,
        "cell_path": cell_path,
        "log_path": log_path,
        "run_dir": run_dir,
    }


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run the hollow-cylinder physical validation case."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="Path to the validation config file.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional fixed run tag. Defaults to a timestamp.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None):
    configure_streaming_stdio()
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    comm = MPI.COMM_WORLD
    simulation_start_time = time.time()

    run_tag = args.run_tag if args.run_tag is not None else (build_run_tag() if comm.rank == 0 else None)
    run_tag = comm.bcast(run_tag, root=0)

    log_file = None
    original_stdout = None
    original_stderr = None

    state = build_validation_state(config_path=args.config, comm=comm)
    output_paths = _prepare_output_bundle(state.cfg, run_tag, comm)

    try:
        if comm.rank == 0:
            log_file = open(
                output_paths["log_path"],
                "w",
                encoding="utf-8",
                buffering=1,
            )
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = _TeeStream(original_stdout, log_file)
            sys.stderr = _TeeStream(original_stderr, log_file)
            os.environ["SIM_TEE_ACTIVE"] = "1"
            print(f"  Run output directory: {output_paths['run_dir']}", flush=True)

        _run_validation_case(
            state,
            run_tag=run_tag,
            output_paths=output_paths,
            simulation_start_time=simulation_start_time,
        )
    finally:
        if comm.rank == 0:
            if original_stdout is not None:
                sys.stdout = original_stdout
            if original_stderr is not None:
                sys.stderr = original_stderr
            if log_file is not None:
                log_file.close()
            os.environ.pop("SIM_TEE_ACTIVE", None)


if __name__ == "__main__":
    main()
