# Author: Abdallah Kamhawi <Kamhawi@umich.edu>

"""Per-layer vertical-settlement extraction for the collapse-print case.

The measurand mirrors the optical-flow ground truth: the vertical displacement
u_z (positive downward) of each printed layer's top sheet, referenced to the
instant that layer is covered by the next one, expressed as a percentage of the
nominal layer height h0. The layer-top sheet of a hexahedral cell is its local
nodes 4,5,6,7 (the BACK face); each cell contributes one azimuth sample, and we
reduce by the median over azimuths for both the full ring and the camera-facing
front arc.
"""

from __future__ import annotations

import numpy as np
from mpi4py import MPI


def assert_layer_top_dofs(state) -> None:
    """One-time check that the layer-top z-DOFs sit on the layer-top sheet.

    Confirms the local-node-4567 / z-component convention by comparing the
    tabulated DOF coordinates against the mesh's measured top-sheet elevations.
    """
    comm = state.comm
    msh = state.msh
    bs = state.V.dofmap.bs
    dof_coords = state.V.tabulate_dof_coordinates()  # (n_blocks, 3)
    num_owned = msh.topology.index_map(msh.topology.dim).size_local
    sheet_z = state.cylinder_mesh.sheet_z  # (n_sheets, n_span)

    n_span = sheet_z.shape[1]
    n_check = min(num_owned, 8)
    max_err = 0.0
    for c in range(n_check):
        layer = int(state.cells_layers_dolfinx[c])
        azim = int(round(state.span_indices_dolfinx[c]))
        z_dofs = state.cell_to_dofs[c][state.layer_top_z_dof_positions]
        blocks = (z_dofs // bs).astype(int)
        z_mean = float(np.mean(dof_coords[blocks, 2]))
        # The cell's four top-sheet nodes span azimuths ``azim`` and
        # ``azim+1``; the wave makes their mean z the average of both.
        z_expected = 0.5 * (
            float(sheet_z[layer + 1, azim])
            + float(sheet_z[layer + 1, (azim + 1) % n_span])
        )
        max_err = max(max_err, abs(z_mean - z_expected))

    global_max_err = comm.allreduce(max_err, op=MPI.MAX)
    if global_max_err > 1.0e-6:
        raise AssertionError(
            "Layer-top z-DOF positions do not lie on the layer-top sheet "
            f"(max |z_dof - sheet_z| = {global_max_err:.3e} mm)."
        )
    if comm.rank == 0:
        print(
            f"  Layer-top DOF assertion passed (max coord error "
            f"{global_max_err:.2e} mm).",
            flush=True,
        )


def extract_layer_uz(state, t_val):
    """Return per-layer median settlement (mm, +down) at time ``t_val``.

    Args:
        state: ``CollapseFEAState`` with the current displacement.
        t_val: Current physical time [s].

    Returns:
        On rank 0, a tuple ``(uz_full, uz_front)`` of length-``n_layers`` arrays
        (mm, positive downward; ``nan`` where a layer has no active cells). On
        other ranks, ``(None, None)``.
    """
    comm = state.comm
    msh = state.msh
    n_layers = int(state.cfg["geometry"]["n_layers"])
    num_owned = msh.topology.index_map(msh.topology.dim).size_local

    owned = slice(0, num_owned)
    z_dofs = state.cell_to_dofs[owned][:, state.layer_top_z_dof_positions]
    # Settlement is downward, so flip the sign of the vertical displacement.
    settlement = -state.u.x.array[z_dofs].reshape(num_owned, -1).mean(axis=1)
    layer_ids = state.cells_layers_dolfinx[:num_owned].astype(np.int32)
    front = state.front_arc_cell_mask_dolfinx[:num_owned]
    active = state.birth_times_dolfinx[:num_owned] <= t_val

    payload = (
        settlement[active],
        layer_ids[active],
        front[active],
    )
    gathered = comm.gather(payload, root=0)
    if comm.rank != 0:
        return None, None

    s_all = np.concatenate([g[0] for g in gathered]) if gathered else np.empty(0)
    l_all = np.concatenate([g[1] for g in gathered]) if gathered else np.empty(0, int)
    f_all = np.concatenate([g[2] for g in gathered]) if gathered else np.empty(0, bool)

    uz_full = np.full(n_layers, np.nan)
    uz_front = np.full(n_layers, np.nan)
    for layer in range(n_layers):
        in_layer = l_all == layer
        if np.any(in_layer):
            uz_full[layer] = float(np.median(s_all[in_layer]))
            in_front = in_layer & f_all
            if np.any(in_front):
                uz_front[layer] = float(np.median(s_all[in_front]))
    return uz_full, uz_front


def reference_to_coverage(uz_series, sample_times, coverage_times, h0_mm):
    """Convert raw settlement series to coverage-referenced percent of h0.

    Args:
        uz_series: ``(n_steps, n_layers)`` raw settlement [mm, +down].
        sample_times: ``(n_steps,)`` stored step times [s].
        coverage_times: ``(n_layers,)`` per-layer coverage reference times [s].
        h0_mm: Nominal layer height [mm].

    Returns:
        tuple ``(uz_pct, coverage_steps)`` where ``uz_pct`` is the
        coverage-referenced settlement in percent of h0 (``nan`` before each
        layer's coverage step) and ``coverage_steps`` are the snapped step
        indices.
    """
    uz_series = np.asarray(uz_series, dtype=float)
    sample_times = np.asarray(sample_times, dtype=float)
    n_steps, n_layers = uz_series.shape

    uz_pct = np.full_like(uz_series, np.nan)
    coverage_steps = np.zeros(n_layers, dtype=int)
    for layer in range(n_layers):
        cov_step = int(np.argmin(np.abs(sample_times - coverage_times[layer])))
        coverage_steps[layer] = cov_step
        ref = uz_series[cov_step, layer]
        if not np.isfinite(ref):
            # Fall back to the first finite sample for this layer.
            finite = np.where(np.isfinite(uz_series[:, layer]))[0]
            if finite.size == 0:
                continue
            cov_step = int(finite[0])
            coverage_steps[layer] = cov_step
            ref = uz_series[cov_step, layer]
        rel = uz_series[:, layer] - ref
        rel[:cov_step] = np.nan
        uz_pct[:, layer] = 100.0 * rel / h0_mm
    return uz_pct, coverage_steps
