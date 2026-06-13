# Author: Abdallah Kamhawi <Kamhawi@umich.edu>

"""Standalone runner for the non-planar collapse-print validation case.

Drives the layer-by-layer printing simulation for either a CG or DG
displacement space, reusing the framework's activation -> Newton -> explicit
Perzyna update -> output loop. The two element families share every numeric
setting (parity for the cost/accuracy comparison); the only differences are
the weak form (CG drops the interior-facet terms) and the activation
displacement seeding (CG is affine, so newly activated DOFs start at zero
instead of inheriting the support cell's state, which would corrupt shared
interface nodes).

Run from the repo root inside the ``fea`` conda env:

    python -m validation.collapse_print.fea.run --element DG
    mpirun -np 4 python -m validation.collapse_print.fea.run --element CG
"""

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
from validation.collapse_print.fea.setup import (
    build_validation_state,
    load_validation_config,
)
from validation.collapse_print.fea.settlement import (
    assert_layer_top_dofs,
    extract_layer_uz,
    reference_to_coverage,
)


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
    gathered_indices, gathered_values, expected_size, field_label
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


def _max_interlayer_damage(state):
    """Global maximum inter-layer damage history variable."""
    dmg = state.materials.damage_max.x.array
    local = float(np.max(dmg)) if dmg.size > 0 else 0.0
    return state.comm.allreduce(local, op=MPI.MAX)


def _build_sample_times(state, cfg):
    """Uniform step grid unioned with the per-layer coverage reference times."""
    time_cfg = cfg.get("time_stepping", {})
    n_steps_cfg = int(time_cfg.get("n_steps", 0))
    end_multiplier = float(time_cfg.get("end_multiplier", 1.0))
    max_steps_cfg = int(time_cfg.get("max_steps", 0))

    t_end = float(state.layer_completion_times_s[-1]) * end_multiplier
    if n_steps_cfg > 0:
        grid = np.linspace(t_end / n_steps_cfg, t_end, n_steps_cfg)
    else:
        grid = np.asarray(state.layer_completion_times_s, dtype=float)

    # Snap clean references: include layer completion and coverage instants.
    extras = np.concatenate(
        [state.layer_completion_times_s, state.layer_coverage_times_s]
    )
    extras = extras[(extras > 0.0) & (extras <= t_end)]
    sample_times = np.unique(np.concatenate([grid, extras]))
    if max_steps_cfg > 0 and len(sample_times) > max_steps_cfg:
        sample_times = sample_times[:max_steps_cfg]
    return sample_times


def _run_case(state, run_tag, output_paths, simulation_start_time=None):
    comm = state.comm
    cfg = state.cfg
    element = state.element
    log_path = output_paths["log_path"]
    checkpoint_cfg = cfg["checkpoint"]
    checkpoint_dir = get_checkpoint_dir(cfg)
    checkpoint_save_every = max(1, int(checkpoint_cfg["save_every"]))

    def log_message(message=""):
        if comm.rank != 0:
            return
        print(message, flush=True)
        if os.environ.get("SIM_TEE_ACTIVE") == "1":
            return
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"{message}\n")

    # DG0 strain projection (lumped mass) for the explicit Perzyna update.
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

    # Inter-layer damage diagnostics (DG cohesive only: CG and bonded-DG
    # controls have no cohesive interfaces).
    track_damage = element == "DG" and not bool(
        cfg.get("interface", {}).get("bonded_only", False)
    )
    if track_damage:
        (
            n_il,
            il_cell_plus,
            il_cell_minus,
            il_facet_normals,
            il_h_cells,
        ) = _precompute_interlayer_damage_data(state.msh, state.interior_facet_tags)
    else:
        n_il = 0

    u_size_local, u_size_global, u_global_indices = _checkpoint_field_layout(state.u)
    eps_vp = state.materials.eps_vp
    eps_vp_size_local, eps_vp_size_global, eps_vp_global_indices = (
        _checkpoint_field_layout(eps_vp)
    )
    damage_max_field = state.materials.damage_max
    dmg_size_local, dmg_size_global, dmg_global_indices = _checkpoint_field_layout(
        damage_max_field
    )

    def save_checkpoint(step, t_val):
        u_local = state.u.x.array[:u_size_local].copy()
        eps_local = eps_vp.x.array[:eps_vp_size_local].copy()
        dmg_local = damage_max_field.x.array[:dmg_size_local].copy()
        g_ui = comm.gather(u_global_indices, root=0)
        g_uv = comm.gather(u_local, root=0)
        g_ei = comm.gather(eps_vp_global_indices, root=0)
        g_ev = comm.gather(eps_local, root=0)
        g_di = comm.gather(dmg_global_indices, root=0)
        g_dv = comm.gather(dmg_local, root=0)
        if comm.rank == 0:
            u_global = _assemble_global_from_owned_chunks(g_ui, g_uv, u_size_global, "u")
            eps_global = _assemble_global_from_owned_chunks(
                g_ei, g_ev, eps_vp_size_global, "eps_vp"
            )
            dmg_global = _assemble_global_from_owned_chunks(
                g_di, g_dv, dmg_size_global, "damage_max"
            )
            np.savez(
                checkpoint_dir / "checkpoint_latest.npz",
                u=u_global,
                eps_vp=eps_global,
                damage_max=dmg_global,
            )
            with open(checkpoint_dir / "latest_checkpoint.json", "w") as handle:
                json.dump({"step": int(step), "time": float(t_val)}, handle, indent=2)
        comm.Barrier()

    sample_times = _build_sample_times(state, cfg)
    n_steps = len(sample_times)
    solver_cfg = cfg["solver"]

    newton_workspace = NewtonLinearWorkspace(
        state.V, state.msh, state.F_form, solver_cfg=solver_cfg
    )
    newton_workspace.ensure_jacobian(state.J_form)

    num_owned_cells = state.msh.topology.index_map(state.msh.topology.dim).size_local
    n_layers = int(cfg["geometry"]["n_layers"])

    # Per-step records for fea_uz.csv and the cost figures.
    uz_full_series = np.full((n_steps, n_layers), np.nan)
    uz_front_series = np.full((n_steps, n_layers), np.nan)
    step_records = []

    total_newton_iters = 0
    total_ksp_its = 0

    if comm.rank == 0:
        log_message("")
        log_message(f"Collapse-print validation case ({element}):")
        log_message(f"  Geometry: nonplanar_cylinder ({state.cylinder_mesh.n_cells} cells)")
        log_message(f"  Mean ring time: {state.ring_time_s:.4f} s")
        log_message(f"  tau_0: {float(cfg['material']['tau_0']):.3f} Pa")
        log_message(f"  Time steps: {n_steps} in [{sample_times[0]:.3f}, {sample_times[-1]:.3f}] s")
        log_message(f"  Linear solver: {newton_workspace.linear_solver_mode}")

    assert_layer_top_dofs(state)

    csv_file = None
    csv_writer = None
    vtx_disp = None
    vtx_cell = None
    loop_start_time = time.time()

    try:
        if comm.rank == 0:
            settings_snapshot_path = save_run_config_snapshot(
                state.cfg, output_paths["run_dir"]
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
                state.materials.nu,
                state.materials.von_mises,
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

        dt_default = sample_times[1] - sample_times[0] if n_steps > 1 else state.ring_time_s
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
                # CG is affine in u, so zero-init of new DOFs is exact and
                # avoids corrupting interface nodes shared with active cells.
                # DG inherits the support cell's deformed state.
                if element == "DG":
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
                workspace=newton_workspace,
            )
            time_newton = time.time() - t_newton_start

            if not converged:
                if comm.rank == 0:
                    log_message("  REVERTING: restoring displacement from before failed step")
                state.u.x.array[:] = u_backup
                state.u.x.scatter_forward()

            zero_inactive_cells(
                state.u, float(t_val), state.birth_times_dolfinx, state.cell_to_dofs
            )
            update_active_indicator(
                state.is_active_func, float(t_val), state.birth_times_dolfinx
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

            t_perzyna_start = time.time()
            eps_vp_tmp = eps_vp_prev_arr.copy()
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

            if n_yielding > 0 and n_sub > 1:
                strain_y = strain_arr[yielding_mask]
                eps_vp_y = eps_vp_tmp[yielding_mask]
                e_y = state.materials.e_arr[yielding_mask]
                nu_y = state.materials.nu_arr[yielding_mask]
                sig_y = state.materials.sigma_y_arr[yielding_mask]
                eta_y = state.materials.eta_arr[yielding_mask]
                for _ in range(2, n_sub + 1):
                    (eps_vp_y, _s, _v, _m, _f) = update_perzyna_state_cellwise(
                        strain_total=strain_y,
                        eps_vp_prev=eps_vp_y,
                        e_arr=e_y,
                        nu_arr=nu_y,
                        sigma_y_arr=sig_y,
                        eta_arr=eta_y,
                        dt=sub_dt,
                        active_mask=None,
                    )
                eps_vp_tmp[yielding_mask] = eps_vp_y
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
                    dt=0.0,
                    active_mask=active_mask,
                )
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

            if track_damage and n_il > 0:
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

            # Global diagnostics.
            local_min_z = (
                float(np.min(state.u.x.array[2::3]))
                if state.u.x.array[2::3].size > 0
                else np.inf
            )
            u_vec = state.u.x.array.reshape((-1, 3))
            local_max_u = (
                float(np.max(np.linalg.norm(u_vec, axis=1))) if u_vec.size > 0 else 0.0
            )
            local_max_vm = (
                float(np.max(state.materials.von_mises.x.array))
                if state.materials.von_mises.x.array.size > 0
                else 0.0
            )
            local_yielding = int(np.sum((f_trial_arr > 0.0) & active_mask))
            global_max_u = comm.allreduce(local_max_u, op=MPI.MAX)
            global_min_z = comm.allreduce(local_min_z, op=MPI.MIN)
            global_max_vm = comm.allreduce(local_max_vm, op=MPI.MAX)
            global_yielding = comm.allreduce(local_yielding, op=MPI.SUM)
            max_damage = _max_interlayer_damage(state) if track_damage else 0.0

            total_newton_iters += int(n_iter)
            total_ksp_its += int(newton_stats.get("total_ksp_its", 0))

            time_io = 0.0
            t_io_start = time.time()
            vtx_disp.write(float(t_val))
            vtx_cell.write(float(t_val))
            time_io += time.time() - t_io_start

            is_final_step = step == (n_steps - 1)
            if (step % checkpoint_save_every == 0) or is_final_step:
                t_io_start = time.time()
                save_checkpoint(step, t_val)
                time_io += time.time() - t_io_start

            # Per-layer settlement extraction.
            uz_full, uz_front = extract_layer_uz(state, float(t_val))
            if comm.rank == 0:
                uz_full_series[step] = uz_full
                uz_front_series[step] = uz_front

            local_active = int(np.sum(state.birth_times_dolfinx[:num_owned_cells] <= t_val))
            n_active_csv = comm.allreduce(local_active, op=MPI.SUM)
            n_total_csv = comm.allreduce(int(num_owned_cells), op=MPI.SUM)
            n_active_ranks = comm.allreduce(1 if local_active > 0 else 0, op=MPI.SUM)

            if csv_writer is not None:
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
                        "max_yield_f_MPa": "",
                        "max_disp_mm": f"{global_max_u:.6e}",
                        "z_sag_mm": f"{global_min_z:.6e}",
                        "max_von_mises_MPa": f"{global_max_vm:.6e}",
                        "max_principal_MPa": "",
                        "max_plastic_strain": "",
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

                step_records.append(
                    {
                        "step": int(step),
                        "time_s": float(t_val),
                        "newton_iters": int(n_iter),
                        "converged": bool(converged),
                        "total_ksp_its": int(newton_stats.get("total_ksp_its", 0)),
                        "active_dofs": newton_stats.get("n_active_dofs", None),
                        "active_cells": int(n_active_csv),
                        "time_newton_s": float(time_newton),
                        "time_step_total_s": float(time.time() - t0),
                        "max_disp_mm": float(global_max_u),
                        "z_sag_mm": float(global_min_z),
                        "yielding_cells": int(global_yielding),
                        "max_interface_damage": float(max_damage),
                    }
                )
                log_message(
                    f"  [{step+1}/{n_steps}] t={t_val:.2f}s  {msg} in {n_iter} it  "
                    f"|u|max={global_max_u:.3e}mm  zsag={global_min_z:.3e}mm  "
                    f"yield={global_yielding}  dmg={max_damage:.3f}  "
                    f"({time.time()-t0:.2f}s)"
                )

            t_prev = float(t_val)

        # Post-process settlement to coverage-referenced percent of h0.
        if comm.rank == 0:
            uz_full_pct, cov_steps = reference_to_coverage(
                uz_full_series,
                sample_times,
                state.layer_coverage_times_s,
                state.h0_mm,
            )
            uz_front_pct, _ = reference_to_coverage(
                uz_front_series,
                sample_times,
                state.layer_coverage_times_s,
                state.h0_mm,
            )
            _write_fea_uz_csv(
                output_paths["run_dir"] / "fea_uz.csv",
                sample_times,
                uz_full_pct,
                uz_front_pct,
                n_layers,
            )
            _write_results_json(
                output_paths["run_dir"] / "results.json",
                state,
                run_tag,
                sample_times,
                uz_full_pct,
                uz_front_pct,
                cov_steps,
                step_records,
                total_newton_iters,
                total_ksp_its,
                time.time() - loop_start_time,
                simulation_start_time,
            )
            _log_summary(
                log_message,
                element,
                n_steps,
                total_newton_iters,
                uz_full_pct,
                uz_front_pct,
                n_layers,
                time.time() - loop_start_time,
            )

        comm.Barrier()
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


def _write_fea_uz_csv(path, sample_times, uz_full_pct, uz_front_pct, n_layers):
    """Write the coverage-referenced per-layer settlement (% of h0)."""
    header = ["time_s"]
    header += [f"uz_full_pct_L{l+1}" for l in range(n_layers)]
    header += [f"uz_front_pct_L{l+1}" for l in range(n_layers)]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for step, t_val in enumerate(sample_times):
            row = [f"{t_val:.6f}"]
            row += [
                "" if not np.isfinite(uz_full_pct[step, l]) else f"{uz_full_pct[step, l]:.4f}"
                for l in range(n_layers)
            ]
            row += [
                "" if not np.isfinite(uz_front_pct[step, l]) else f"{uz_front_pct[step, l]:.4f}"
                for l in range(n_layers)
            ]
            writer.writerow(row)


def _final_per_layer(uz_pct, n_layers):
    """Final finite coverage-referenced settlement (% of h0) per layer."""
    finals = []
    for l in range(n_layers):
        col = uz_pct[:, l]
        finite = col[np.isfinite(col)]
        finals.append(float(finite[-1]) if finite.size > 0 else float("nan"))
    return finals


def _write_results_json(
    path,
    state,
    run_tag,
    sample_times,
    uz_full_pct,
    uz_front_pct,
    cov_steps,
    step_records,
    total_newton_iters,
    total_ksp_its,
    loop_wall_s,
    simulation_start_time,
):
    cfg = state.cfg
    results = {
        "case": "collapse_print_nonplanar",
        "element": state.element,
        "run_tag": run_tag,
        "geometry": {
            "n_layers": int(cfg["geometry"]["n_layers"]),
            "n_span": int(cfg["mesh"]["n_span"]),
            "n_thickness": int(cfg["mesh"]["n_thickness"]),
            "n_cells": int(state.cylinder_mesh.n_cells),
            "layer_height_mm": float(state.h0_mm),
            "wall_thickness_mm": float(state.cylinder_mesh.wall_thickness),
        },
        "material": {
            k: float(cfg["material"][k])
            for k in ("tau_0", "A_thix", "mu_p", "gamma_c", "rho", "g")
        },
        "hardening": {
            k: float(cfg["hardening"][k])
            for k in ("t_set", "E_inf", "nu_fresh", "nu_hard", "n_h")
        },
        "timing": {
            "loop_wall_s": float(loop_wall_s),
            "total_wall_s": (
                float(loop_wall_s)
                if simulation_start_time is None
                else float(time.time() - simulation_start_time)
            ),
            "total_newton_iters": int(total_newton_iters),
            "total_ksp_its": int(total_ksp_its),
            "n_steps": int(len(sample_times)),
        },
        "coverage_times_s": [float(t) for t in state.layer_coverage_times_s],
        "coverage_steps": [int(s) for s in cov_steps],
        "final_uz_full_pct": _final_per_layer(uz_full_pct, int(cfg["geometry"]["n_layers"])),
        "final_uz_front_pct": _final_per_layer(uz_front_pct, int(cfg["geometry"]["n_layers"])),
        "steps": step_records,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")


def _log_summary(
    log_message, element, n_steps, total_newton_iters, uz_full_pct, uz_front_pct, n_layers, wall_s
):
    full = _final_per_layer(uz_full_pct, n_layers)
    front = _final_per_layer(uz_front_pct, n_layers)
    log_message("")
    log_message("  ====================================================")
    log_message(f"  SIMULATION COMPLETE ({element})")
    log_message("  ====================================================")
    log_message(f"    Steps: {n_steps}   Newton iters: {total_newton_iters}   Wall: {wall_s:.1f}s")
    log_message("    Final per-layer u_z [% of h0] (full ring | front arc):")
    for l in range(n_layers):
        log_message(f"      L{l+1:2d}:  {full[l]:7.2f}  |  {front[l]:7.2f}")


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
        description="Run the non-planar collapse-print validation case."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="Path to the validation config file.",
    )
    parser.add_argument(
        "--element",
        type=str,
        default=None,
        choices=["CG", "DG", "cg", "dg"],
        help="Displacement element family (overrides config).",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional fixed run tag. Defaults to <element>_<timestamp>.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Cap the number of steps (smoke testing).",
    )
    parser.add_argument(
        "--bonded-control",
        action="store_true",
        help=(
            "Run the DG bonded-interface control: inter-layer facets receive "
            "the intra-layer SIPG coupling instead of the cohesive law."
        ),
    )
    return parser


def main(argv: Optional[Iterable[str]] = None):
    configure_streaming_stdio()
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    comm = MPI.COMM_WORLD
    simulation_start_time = time.time()

    element = args.element.upper() if args.element else None
    if args.bonded_control:
        element = "DG"
    state = build_validation_state(
        config_path=args.config,
        comm=comm,
        element=element,
        bonded_control=args.bonded_control,
    )
    if args.max_steps is not None:
        state.cfg["time_stepping"]["max_steps"] = int(args.max_steps)

    if args.run_tag is not None:
        run_tag = args.run_tag
    else:
        base = build_run_tag() if comm.rank == 0 else None
        base = comm.bcast(base, root=0)
        family_tag = (
            "DGB"
            if state.cfg.get("interface", {}).get("bonded_only", False)
            else state.element
        )
        run_tag = f"{family_tag}_{base}"

    output_paths = _prepare_output_bundle(state.cfg, run_tag, comm)

    log_file = None
    original_stdout = None
    original_stderr = None
    try:
        if comm.rank == 0:
            log_file = open(output_paths["log_path"], "w", encoding="utf-8", buffering=1)
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = _TeeStream(original_stdout, log_file)
            sys.stderr = _TeeStream(original_stderr, log_file)
            os.environ["SIM_TEE_ACTIVE"] = "1"
            print(f"  Run output directory: {output_paths['run_dir']}", flush=True)

        _run_case(
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
