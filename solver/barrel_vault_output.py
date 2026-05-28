"""Barrel-vault output helpers for normal production runs.

This module provides:
- cell/facet metadata builders for barrel-vault meshes,
- per-step inter-layer opening/damage summaries,
- CSV/JSON writers for run outputs,
- post-processing figure generation from saved run outputs.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


REGION_UNKNOWN = -1
REGION_SPRINGER = 0
REGION_HAUNCH = 1
REGION_CROWN = 2

REGION_LABELS = {
    REGION_UNKNOWN: "unknown",
    REGION_SPRINGER: "springer",
    REGION_HAUNCH: "haunch",
    REGION_CROWN: "crown",
}


@dataclass
class CellOutputData:
    """Cell metadata in DOLFINx local+ghost ordering."""

    span_index: np.ndarray
    thickness_index: np.ndarray
    length_index: np.ndarray
    region_code: np.ndarray
    centroid_xyz: np.ndarray


@dataclass
class InterlayerOutputData:
    """Inter-layer facet metadata in local facet ownership ordering."""

    facet_key: np.ndarray
    lower_cell: np.ndarray
    upper_cell: np.ndarray
    normal_xyz: np.ndarray
    centroid_xyz: np.ndarray
    span_index: np.ndarray
    thickness_index: np.ndarray
    lower_layer: np.ndarray
    upper_layer: np.ndarray
    region_code: np.ndarray
    junction_flag: np.ndarray


def get_barrel_vault_output_config(cfg: dict) -> dict:
    """Return normalized barrel-vault output configuration."""
    defaults = {
        "enabled": True,
        "write_facet_history": False,
        "write_span_profiles": True,
        "sample_every_steps": 1,
        "damage_threshold": 0.05,
        "gap_threshold": 0.50,
        "front_window_time_mult": 1.0,
    }
    user_cfg = dict(cfg.get("barrel_vault_output", {}))
    merged = dict(defaults)
    merged.update(user_cfg)
    merged["enabled"] = bool(merged.get("enabled", True))
    merged["write_facet_history"] = bool(merged.get("write_facet_history", False))
    merged["write_span_profiles"] = bool(merged.get("write_span_profiles", True))
    merged["sample_every_steps"] = max(1, int(merged.get("sample_every_steps", 1)))
    merged["damage_threshold"] = float(merged.get("damage_threshold", 0.05))
    merged["gap_threshold"] = float(merged.get("gap_threshold", 0.50))
    merged["front_window_time_mult"] = max(
        0.0, float(merged.get("front_window_time_mult", 1.0))
    )
    return merged


def build_cell_output_data(vault_mesh, cells_lst, perm) -> CellOutputData:
    """Build cell metadata arrays in DOLFINx local+ghost ordering."""
    nodes = vault_mesh.nodes
    n_cells = len(perm)

    span_index = np.empty(n_cells, dtype=np.int32)
    thickness_index = np.empty(n_cells, dtype=np.int32)
    length_index = np.empty(n_cells, dtype=np.int32)
    region_code = np.empty(n_cells, dtype=np.int8)
    centroid_xyz = np.empty((n_cells, 3), dtype=np.float64)

    for dolfinx_idx, original_idx in enumerate(np.asarray(perm, dtype=np.int64)):
        cell = cells_lst[int(original_idx)]
        span_index[dolfinx_idx] = int(getattr(cell, "span_index", -1))
        thickness_index[dolfinx_idx] = int(getattr(cell, "thickness_index", -1))
        layer = getattr(cell, "layer", None)
        length_index[dolfinx_idx] = int(
            getattr(layer, "layer_id", getattr(cell, "length_index", -1))
        )
        centroid_xyz[dolfinx_idx, :] = np.asarray(
            cell.compute_centroid(nodes), dtype=np.float64
        )
        region_name = getattr(getattr(cell, "vault_cell_type", None), "value", None)
        if region_name == "springer":
            region_code[dolfinx_idx] = REGION_SPRINGER
        elif region_name == "haunch":
            region_code[dolfinx_idx] = REGION_HAUNCH
        elif region_name == "crown":
            region_code[dolfinx_idx] = REGION_CROWN
        else:
            region_code[dolfinx_idx] = REGION_UNKNOWN

    return CellOutputData(
        span_index=span_index,
        thickness_index=thickness_index,
        length_index=length_index,
        region_code=region_code,
        centroid_xyz=centroid_xyz,
    )


def build_interlayer_output_data(
    msh,
    interior_facet_tags,
    cell_data: CellOutputData,
    cfg: dict,
) -> InterlayerOutputData:
    """Build inter-layer facet metadata in local facet ownership ordering."""
    empty = InterlayerOutputData(
        facet_key=np.empty(0, dtype=np.int64),
        lower_cell=np.empty(0, dtype=np.int32),
        upper_cell=np.empty(0, dtype=np.int32),
        normal_xyz=np.empty((0, 3), dtype=np.float64),
        centroid_xyz=np.empty((0, 3), dtype=np.float64),
        span_index=np.empty(0, dtype=np.int32),
        thickness_index=np.empty(0, dtype=np.int32),
        lower_layer=np.empty(0, dtype=np.int32),
        upper_layer=np.empty(0, dtype=np.int32),
        region_code=np.empty(0, dtype=np.int8),
        junction_flag=np.empty(0, dtype=bool),
    )
    if interior_facet_tags is None:
        return empty

    il_mask = np.asarray(interior_facet_tags.values == 1, dtype=bool)
    il_facets = np.asarray(interior_facet_tags.indices[il_mask], dtype=np.int32)
    n_il = int(il_facets.size)
    if n_il == 0:
        return empty

    n_span = int(cfg["mesh"]["n_span"])
    n_thickness = int(cfg["mesh"]["n_thickness"])
    support_upto = int(cfg["boundary_conditions"]["intrados_dirichlet_upto_layer"])

    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    msh.topology.create_connectivity(fdim, 0)
    f_to_c = msh.topology.connectivity(fdim, tdim)
    f_to_v = msh.topology.connectivity(fdim, 0)

    from dolfinx.mesh import entities_to_geometry

    geom_x = msh.geometry.x

    facet_key = np.empty(n_il, dtype=np.int64)
    lower_cell = np.empty(n_il, dtype=np.int32)
    upper_cell = np.empty(n_il, dtype=np.int32)
    normal_xyz = np.empty((n_il, 3), dtype=np.float64)
    centroid_xyz = np.empty((n_il, 3), dtype=np.float64)
    span_index = np.empty(n_il, dtype=np.int32)
    thickness_index = np.empty(n_il, dtype=np.int32)
    lower_layer = np.empty(n_il, dtype=np.int32)
    upper_layer = np.empty(n_il, dtype=np.int32)
    region_code = np.empty(n_il, dtype=np.int8)
    junction_flag = np.empty(n_il, dtype=bool)

    for idx, facet in enumerate(il_facets):
        cells = np.asarray(f_to_c.links(int(facet)), dtype=np.int32)
        if cells.size == 0:
            continue
        if cells.size == 1:
            c0 = c1 = int(cells[0])
        else:
            c0 = int(cells[0])
            c1 = int(cells[1])

        l0 = int(cell_data.length_index[c0])
        l1 = int(cell_data.length_index[c1])
        if l0 <= l1:
            lo = c0
            up = c1
        else:
            lo = c1
            up = c0

        lower_cell[idx] = lo
        upper_cell[idx] = up
        lower_layer[idx] = int(cell_data.length_index[lo])
        upper_layer[idx] = int(cell_data.length_index[up])
        span_index[idx] = int(cell_data.span_index[lo])
        thickness_index[idx] = int(cell_data.thickness_index[lo])
        region_code[idx] = int(cell_data.region_code[lo])
        junction_flag[idx] = lower_layer[idx] == support_upto
        facet_key[idx] = (
            int(lower_layer[idx]) * n_span * n_thickness
            + int(thickness_index[idx]) * n_span
            + int(span_index[idx])
        )

        verts = np.asarray(f_to_v.links(int(facet)), dtype=np.int32)
        geom_dofs = entities_to_geometry(msh, 0, verts, False).reshape(-1)
        coords = np.asarray(geom_x[geom_dofs], dtype=np.float64)
        centroid_xyz[idx, :] = np.mean(coords, axis=0)
        if coords.shape[0] >= 4:
            normal = np.cross(coords[1] - coords[0], coords[3] - coords[0])
        elif coords.shape[0] >= 3:
            normal = np.cross(coords[1] - coords[0], coords[2] - coords[0])
        else:
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        orient = cell_data.centroid_xyz[up] - cell_data.centroid_xyz[lo]
        if np.dot(normal, orient) < 0.0:
            normal *= -1.0
        norm = float(np.linalg.norm(normal))
        if norm > 1.0e-12:
            normal /= norm
        else:
            normal[:] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        normal_xyz[idx, :] = normal

    return InterlayerOutputData(
        facet_key=facet_key,
        lower_cell=lower_cell,
        upper_cell=upper_cell,
        normal_xyz=normal_xyz,
        centroid_xyz=centroid_xyz,
        span_index=span_index,
        thickness_index=thickness_index,
        lower_layer=lower_layer,
        upper_layer=upper_layer,
        region_code=region_code,
        junction_flag=junction_flag,
    )


def mean_birth_interval(birth_times: np.ndarray) -> float:
    """Return the mean positive birth-time increment."""
    uniq = np.unique(np.asarray(birth_times, dtype=np.float64))
    diffs = np.diff(np.sort(uniq))
    diffs = diffs[diffs > 1.0e-12]
    if diffs.size == 0:
        return 0.0
    return float(np.mean(diffs))


def safe_ratio(numerator: float, denominator: float, large: float = 1.0e12) -> float:
    """Return a bounded ratio that stays finite when the denominator vanishes."""
    num = float(numerator)
    den = float(denominator)
    if abs(den) > 1.0e-15:
        return num / den
    if abs(num) > 1.0e-15:
        return large if num > 0.0 else -large
    return 0.0


def _max_or_zero(values: np.ndarray, mask: np.ndarray) -> float:
    if values.size == 0 or not np.any(mask):
        return 0.0
    return float(np.max(values[mask]))


def _pick_event_index(score: np.ndarray, mask: np.ndarray) -> Optional[int]:
    if score.size == 0 or not np.any(mask):
        return None
    candidates = np.flatnonzero(mask)
    return int(candidates[np.argmax(score[candidates])])


def summarize_region(
    region_code: int,
    junction_flag: bool,
    front_band_flag: bool,
) -> str:
    """Return a coarse region summary used in event JSON output."""
    if bool(front_band_flag):
        return "deposition_front"
    if bool(junction_flag):
        return "junction"
    return REGION_LABELS.get(int(region_code), "unknown")


def _format_interface_label(
    lower_layer_zero_based: int, upper_layer_zero_based: int
) -> Dict[str, str]:
    return {
        "interface_zero_based": (
            f"{int(lower_layer_zero_based)}-{int(upper_layer_zero_based)}"
        ),
        "interface_one_based": (
            f"{int(lower_layer_zero_based) + 1}-{int(upper_layer_zero_based) + 1}"
        ),
    }


def _empty_step_summary() -> dict:
    return {
        "max_opening_mm": 0.0,
        "max_damage": 0.0,
        "junction_opening_mm": 0.0,
        "junction_damage": 0.0,
        "crown_opening_mm": 0.0,
        "crown_damage": 0.0,
        "front_opening_mm": 0.0,
        "front_damage": 0.0,
        "front_active_facets": 0,
        "front_localization_ratio": 0.0,
        "front_vs_crown_opening_ratio": 0.0,
        "crown_vs_junction_opening_ratio": 0.0,
        "junction_opening_ratio": 0.0,
        "junction_damage_ratio": 0.0,
        "cantilever_sag_mm": 0.0,
    }


def compute_step_output_summary(
    u,
    materials,
    cell_to_dofs: np.ndarray,
    birth_times: np.ndarray,
    cfg: dict,
    output_cfg: dict,
    cell_data: CellOutputData,
    interlayer_data: InterlayerOutputData,
    t_val: float,
    active_layers: int,
    front_window_time: float,
) -> dict:
    """Compute inter-layer opening and damage summaries for one step."""
    n_il = int(interlayer_data.facet_key.size)
    if n_il == 0:
        return {
            "summary": _empty_step_summary(),
            "active_rows": {
                "facet_key": np.empty(0, dtype=np.int64),
                "span_index": np.empty(0, dtype=np.int32),
                "thickness_index": np.empty(0, dtype=np.int32),
                "lower_layer_zero_based": np.empty(0, dtype=np.int32),
                "upper_layer_zero_based": np.empty(0, dtype=np.int32),
                "region_code": np.empty(0, dtype=np.int8),
                "junction_flag": np.empty(0, dtype=bool),
                "front_band_flag": np.empty(0, dtype=bool),
                "centroid_x_mm": np.empty(0, dtype=np.float64),
                "centroid_y_mm": np.empty(0, dtype=np.float64),
                "centroid_z_mm": np.empty(0, dtype=np.float64),
                "jump_n_mm": np.empty(0, dtype=np.float64),
                "jump_n_open_mm": np.empty(0, dtype=np.float64),
                "jump_t_mm": np.empty(0, dtype=np.float64),
                "mode_i_drive": np.empty(0, dtype=np.float64),
                "mode_ii_drive": np.empty(0, dtype=np.float64),
                "facet_damage": np.empty(0, dtype=np.float64),
                "damage_max_pair": np.empty(0, dtype=np.float64),
            },
            "events": {"damage_idx": None, "gap_idx": None, "junction_idx": None},
        }

    lower = interlayer_data.lower_cell
    upper = interlayer_data.upper_cell
    normals = interlayer_data.normal_xyz

    u_arr = u.x.array
    dofs_lower = cell_to_dofs[lower]
    dofs_upper = cell_to_dofs[upper]
    u_lower = u_arr[dofs_lower].reshape(n_il, -1, 3).mean(axis=1)
    u_upper = u_arr[dofs_upper].reshape(n_il, -1, 3).mean(axis=1)
    jump_vec = u_upper - u_lower

    jump_n = np.sum(jump_vec * normals, axis=1)
    jump_n_open = np.maximum(jump_n, 0.0)
    jump_t_vec = jump_vec - jump_n[:, None] * normals
    jump_t_mag = np.linalg.norm(jump_t_vec, axis=1)

    active_mask = (birth_times[lower] <= t_val) & (birth_times[upper] <= t_val)
    recent_birth = np.maximum(birth_times[lower], birth_times[upper])
    front_band_flag = (
        active_mask
        & (interlayer_data.upper_layer == max(active_layers - 1, -1))
        & ((t_val - recent_birth) >= -1.0e-12)
        & ((t_val - recent_birth) <= front_window_time + 1.0e-12)
    )

    sigma_y_arr = materials.sigma_y.x.array
    tau_y_arr = materials.tau_y.x.array
    e_arr = materials.E.x.array
    dmg_arr = materials.damage_max.x.array

    sigma_y_avg = 0.5 * (sigma_y_arr[upper] + sigma_y_arr[lower])
    tau_y_avg = 0.5 * (tau_y_arr[upper] + tau_y_arr[lower])
    e_avg = 0.5 * (e_arr[upper] + e_arr[lower])

    tau_0_pa = float(cfg["material"]["tau_0"])
    a_thix_pa_s = float(cfg["material"]["A_thix"])
    nozzle_pressure_pa = float(cfg["interface"]["nozzle_pressure"])
    g_i_c = float(cfg["interface"]["G_Ic"])
    g_ii_c = float(cfg["interface"]["G_IIc"])
    k_min_mult = float(cfg["interface"].get("K_min_mult", 0.1))

    t_open = np.abs(birth_times[upper] - birth_times[lower])
    tau_sub = tau_0_pa + a_thix_pa_s * t_open
    phi = nozzle_pressure_pa / np.maximum(tau_sub, 1.0e-12)
    beta = np.clip(1.0 - np.exp(-phi), 1.0e-8, 1.0)

    g_i_c_eff = np.maximum(beta * g_i_c, 1.0e-12)
    g_ii_c_eff = np.maximum(beta * g_ii_c, 1.0e-12)

    h_upper = np.linalg.norm(
        cell_data.centroid_xyz[upper] - interlayer_data.centroid_xyz, axis=1
    )
    h_lower = np.linalg.norm(
        cell_data.centroid_xyz[lower] - interlayer_data.centroid_xyz, axis=1
    )
    h_avg = np.maximum(h_upper + h_lower, 1.0e-6)

    k_n = beta * sigma_y_avg ** 2 / (2.0 * g_i_c_eff)
    k_t = beta * tau_y_avg ** 2 / (2.0 * g_ii_c_eff)
    k_min = k_min_mult * e_avg / h_avg
    k_n = np.maximum(k_n, k_min)
    k_t = np.maximum(k_t, k_min)

    mode_i_drive = jump_n_open ** 2 / np.maximum(
        2.0 * g_i_c_eff / np.maximum(k_n, 1.0e-12), 1.0e-12
    )
    mode_ii_drive = jump_t_mag ** 2 / np.maximum(
        2.0 * g_ii_c_eff / np.maximum(k_t, 1.0e-12), 1.0e-12
    )
    facet_damage = np.clip(1.0 - np.exp(-(mode_i_drive + mode_ii_drive)), 0.0, 1.0)
    damage_max_pair = np.maximum(dmg_arr[upper], dmg_arr[lower])

    jump_n = np.where(active_mask, jump_n, 0.0)
    jump_n_open = np.where(active_mask, jump_n_open, 0.0)
    jump_t_mag = np.where(active_mask, jump_t_mag, 0.0)
    mode_i_drive = np.where(active_mask, mode_i_drive, 0.0)
    mode_ii_drive = np.where(active_mask, mode_ii_drive, 0.0)
    facet_damage = np.where(active_mask, facet_damage, 0.0)
    damage_max_pair = np.where(active_mask, damage_max_pair, 0.0)
    front_band_flag = np.where(active_mask, front_band_flag, False)

    region_is_crown = interlayer_data.region_code == REGION_CROWN
    junction_flag = interlayer_data.junction_flag

    summary = {
        "max_opening_mm": _max_or_zero(jump_n_open, active_mask),
        "max_damage": _max_or_zero(facet_damage, active_mask),
        "junction_opening_mm": _max_or_zero(jump_n_open, active_mask & junction_flag),
        "junction_damage": _max_or_zero(facet_damage, active_mask & junction_flag),
        "crown_opening_mm": _max_or_zero(jump_n_open, active_mask & region_is_crown),
        "crown_damage": _max_or_zero(facet_damage, active_mask & region_is_crown),
        "front_opening_mm": _max_or_zero(jump_n_open, front_band_flag),
        "front_damage": _max_or_zero(facet_damage, front_band_flag),
        "front_active_facets": int(np.sum(front_band_flag)),
    }
    summary["front_localization_ratio"] = safe_ratio(
        summary["front_opening_mm"], summary["crown_opening_mm"]
    )
    summary["front_vs_crown_opening_ratio"] = summary["front_localization_ratio"]
    summary["crown_vs_junction_opening_ratio"] = safe_ratio(
        summary["crown_opening_mm"], summary["junction_opening_mm"]
    )
    summary["junction_opening_ratio"] = safe_ratio(
        summary["junction_opening_mm"], summary["max_opening_mm"]
    )
    summary["junction_damage_ratio"] = safe_ratio(
        summary["junction_damage"], summary["max_damage"]
    )

    cell_active_mask = birth_times <= t_val
    unsupported_mask = cell_active_mask & (
        cell_data.length_index
        > int(cfg["boundary_conditions"]["intrados_dirichlet_upto_layer"])
    )
    if np.any(unsupported_mask):
        cell_u_z = u_arr[cell_to_dofs].reshape(cell_to_dofs.shape[0], -1, 3).mean(
            axis=1
        )[:, 2]
        summary["cantilever_sag_mm"] = float(np.min(cell_u_z[unsupported_mask]))
    else:
        summary["cantilever_sag_mm"] = 0.0

    damage_idx = _pick_event_index(
        facet_damage,
        active_mask & (facet_damage >= float(output_cfg["damage_threshold"])),
    )
    gap_idx = _pick_event_index(
        jump_n_open,
        active_mask & (jump_n_open >= float(output_cfg["gap_threshold"])),
    )
    junction_idx = _pick_event_index(
        jump_n_open,
        active_mask & junction_flag & (jump_n_open > 0.0),
    )

    active_rows_mask = active_mask
    active_rows = {
        "facet_key": interlayer_data.facet_key[active_rows_mask].copy(),
        "span_index": interlayer_data.span_index[active_rows_mask].copy(),
        "thickness_index": interlayer_data.thickness_index[active_rows_mask].copy(),
        "lower_layer_zero_based": interlayer_data.lower_layer[active_rows_mask].copy(),
        "upper_layer_zero_based": interlayer_data.upper_layer[active_rows_mask].copy(),
        "region_code": interlayer_data.region_code[active_rows_mask].copy(),
        "junction_flag": interlayer_data.junction_flag[active_rows_mask].copy(),
        "front_band_flag": front_band_flag[active_rows_mask].copy(),
        "centroid_x_mm": interlayer_data.centroid_xyz[active_rows_mask, 0].copy(),
        "centroid_y_mm": interlayer_data.centroid_xyz[active_rows_mask, 1].copy(),
        "centroid_z_mm": interlayer_data.centroid_xyz[active_rows_mask, 2].copy(),
        "jump_n_mm": jump_n[active_rows_mask].copy(),
        "jump_n_open_mm": jump_n_open[active_rows_mask].copy(),
        "jump_t_mm": jump_t_mag[active_rows_mask].copy(),
        "mode_i_drive": mode_i_drive[active_rows_mask].copy(),
        "mode_ii_drive": mode_ii_drive[active_rows_mask].copy(),
        "facet_damage": facet_damage[active_rows_mask].copy(),
        "damage_max_pair": damage_max_pair[active_rows_mask].copy(),
    }

    return {
        "summary": summary,
        "active_rows": active_rows,
        "events": {
            "damage_idx": damage_idx,
            "gap_idx": gap_idx,
            "junction_idx": junction_idx,
        },
    }


def build_event_record(
    root_rows: dict, row_index: int, step: int, time_s: float, active_layers: int
) -> dict:
    """Build a JSON-serializable event record from gathered facet data."""
    region_code = int(root_rows["region_code"][row_index])
    junction_flag = bool(root_rows["junction_flag"][row_index])
    front_band_flag = bool(root_rows["front_band_flag"][row_index])
    lower_layer = int(root_rows["lower_layer_zero_based"][row_index])
    upper_layer = int(root_rows["upper_layer_zero_based"][row_index])
    record = {
        "step": int(step),
        "time_s": float(time_s),
        "active_layers": int(active_layers),
        "facet_key": int(root_rows["facet_key"][row_index]),
        "summary_region": summarize_region(region_code, junction_flag, front_band_flag),
        "region_label": REGION_LABELS.get(region_code, "unknown"),
        "front_band_flag": front_band_flag,
        "junction_flag": junction_flag,
        "span_index": int(root_rows["span_index"][row_index]),
        "thickness_index": int(root_rows["thickness_index"][row_index]),
        "lower_layer_zero_based": lower_layer,
        "upper_layer_zero_based": upper_layer,
        "centroid_x_mm": float(root_rows["centroid_x_mm"][row_index]),
        "centroid_y_mm": float(root_rows["centroid_y_mm"][row_index]),
        "centroid_z_mm": float(root_rows["centroid_z_mm"][row_index]),
        "jump_n_mm": float(root_rows["jump_n_mm"][row_index]),
        "jump_n_open_mm": float(root_rows["jump_n_open_mm"][row_index]),
        "jump_t_mm": float(root_rows["jump_t_mm"][row_index]),
        "mode_i_drive": float(root_rows["mode_i_drive"][row_index]),
        "mode_ii_drive": float(root_rows["mode_ii_drive"][row_index]),
        "facet_damage": float(root_rows["facet_damage"][row_index]),
        "damage_max_pair": float(root_rows["damage_max_pair"][row_index]),
    }
    record.update(_format_interface_label(lower_layer, upper_layer))
    return record


def concat_gathered_rows(gathered_rows: Iterable[dict]) -> dict:
    """Concatenate gathered rank-local active facet arrays on rank 0."""
    gathered_rows = list(gathered_rows)
    if not gathered_rows:
        return {}
    keys = list(gathered_rows[0].keys())
    out = {}
    for key in keys:
        parts = [np.asarray(row[key]) for row in gathered_rows if len(row[key]) > 0]
        if parts:
            out[key] = np.concatenate(parts)
        else:
            sample = np.asarray(gathered_rows[0][key])
            out[key] = np.empty(
                0,
                dtype=sample.dtype if sample.dtype != object else np.float64,
            )
    return out


def write_facet_history_rows(
    writer: csv.DictWriter,
    root_rows: dict,
    step: int,
    time_s: float,
    active_layers: int,
) -> None:
    """Append one sampled step of active-facet history rows."""
    n_rows = int(root_rows.get("facet_key", np.empty(0)).size)
    for idx in range(n_rows):
        writer.writerow(
            {
                "step": step,
                "time_s": f"{time_s:.6f}",
                "active_layers": active_layers,
                "facet_key": int(root_rows["facet_key"][idx]),
                "span_index": int(root_rows["span_index"][idx]),
                "thickness_index": int(root_rows["thickness_index"][idx]),
                "lower_layer_zero_based": int(root_rows["lower_layer_zero_based"][idx]),
                "upper_layer_zero_based": int(root_rows["upper_layer_zero_based"][idx]),
                "region_label": REGION_LABELS.get(
                    int(root_rows["region_code"][idx]), "unknown"
                ),
                "junction_flag": int(bool(root_rows["junction_flag"][idx])),
                "front_band_flag": int(bool(root_rows["front_band_flag"][idx])),
                "centroid_x_mm": f"{float(root_rows['centroid_x_mm'][idx]):.6e}",
                "centroid_y_mm": f"{float(root_rows['centroid_y_mm'][idx]):.6e}",
                "centroid_z_mm": f"{float(root_rows['centroid_z_mm'][idx]):.6e}",
                "jump_n_mm": f"{float(root_rows['jump_n_mm'][idx]):.6e}",
                "jump_n_open_mm": f"{float(root_rows['jump_n_open_mm'][idx]):.6e}",
                "jump_t_mm": f"{float(root_rows['jump_t_mm'][idx]):.6e}",
                "mode_i_drive": f"{float(root_rows['mode_i_drive'][idx]):.6e}",
                "mode_ii_drive": f"{float(root_rows['mode_ii_drive'][idx]):.6e}",
                "facet_damage": f"{float(root_rows['facet_damage'][idx]):.6e}",
                "damage_max_pair": f"{float(root_rows['damage_max_pair'][idx]):.6e}",
            }
        )


def write_span_profile_rows(
    writer: csv.DictWriter,
    root_rows: dict,
    step: int,
    time_s: float,
    active_layers: int,
) -> None:
    """Append span-wise, thickness-collapsed profile rows for one sampled step."""
    n_rows = int(root_rows.get("facet_key", np.empty(0)).size)
    if n_rows == 0:
        return

    lower_layer = np.asarray(root_rows["lower_layer_zero_based"], dtype=np.int32)
    span_index = np.asarray(root_rows["span_index"], dtype=np.int32)
    group_key = lower_layer.astype(np.int64) * 1_000_000 + span_index.astype(np.int64)
    order = np.argsort(group_key, kind="mergesort")
    group_key = group_key[order]

    _, starts = np.unique(group_key, return_index=True)
    ends = np.concatenate([starts[1:], np.array([group_key.size])])

    jump_open = np.asarray(root_rows["jump_n_open_mm"], dtype=np.float64)[order]
    facet_damage = np.asarray(root_rows["facet_damage"], dtype=np.float64)[order]
    region_code = np.asarray(root_rows["region_code"], dtype=np.int8)[order]
    junction_flag = np.asarray(root_rows["junction_flag"], dtype=bool)[order]
    lower_layer_sorted = lower_layer[order]
    span_index_sorted = span_index[order]

    for start, end in zip(starts, ends):
        writer.writerow(
            {
                "step": step,
                "time_s": f"{time_s:.6f}",
                "active_layers": active_layers,
                "lower_layer_zero_based": int(lower_layer_sorted[start]),
                "span_index": int(span_index_sorted[start]),
                "region_label": REGION_LABELS.get(int(region_code[start]), "unknown"),
                "junction_flag": int(bool(np.any(junction_flag[start:end]))),
                "max_opening_mm": f"{float(np.max(jump_open[start:end])):.6e}",
                "max_damage": f"{float(np.max(facet_damage[start:end])):.6e}",
                "mean_opening_mm": f"{float(np.mean(jump_open[start:end])):.6e}",
                "mean_damage": f"{float(np.mean(facet_damage[start:end])):.6e}",
            }
        )


def finalize_barrel_vault_results(
    run_dir: Path,
    cfg: dict,
    output_cfg: dict,
    event_state: dict,
    peak_state: dict,
    final_step_summary: dict,
) -> Path:
    """Write the run-level barrel-vault results JSON."""
    first_damage = event_state.get("first_damage")
    first_visible_gap = event_state.get("first_visible_gap")
    first_junction_opening = event_state.get("first_junction_opening")

    hypotheses = {
        "front_localizes_but_gap_forms_at_crown": bool(
            peak_state.get("front_localization_ratio", 0.0) > 1.0
            and first_visible_gap is not None
            and first_visible_gap.get("summary_region") == "crown"
        ),
        "damage_initiates_at_deposition_front": bool(
            first_damage is not None and first_damage.get("front_band_flag", False)
        ),
        "junction_opens_before_crown": bool(
            first_junction_opening is not None
            and first_visible_gap is not None
            and int(first_junction_opening["step"]) <= int(first_visible_gap["step"])
        ),
    }

    results = {
        "output_version": 1,
        "events": {
            "first_damage": first_damage,
            "first_visible_gap": first_visible_gap,
            "first_junction_opening": first_junction_opening,
        },
        "derived": {
            "first_damage_region": (
                None if first_damage is None else first_damage.get("summary_region")
            ),
            "front_localization_ratio": float(
                peak_state.get("front_localization_ratio", 0.0)
            ),
            "junction_damage_ratio": float(
                peak_state.get("junction_damage_ratio", 0.0)
            ),
            "junction_opening_ratio": float(
                peak_state.get("junction_opening_ratio", 0.0)
            ),
            "front_vs_crown_opening_ratio": float(
                peak_state.get("front_vs_crown_opening_ratio", 0.0)
            ),
            "crown_vs_junction_opening_ratio": float(
                peak_state.get("crown_vs_junction_opening_ratio", 0.0)
            ),
        },
        "hypotheses": hypotheses,
        "summary": {
            "final": dict(final_step_summary),
        },
        "config": {
            "boundary_conditions": {
                "intrados_dirichlet_upto_layer": int(
                    cfg["boundary_conditions"]["intrados_dirichlet_upto_layer"]
                )
            },
            "barrel_vault_output": dict(output_cfg),
        },
    }

    results_path = Path(run_dir) / "barrel_vault_results.json"
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results_path


def _load_csv_rows(csv_path: Path) -> List[dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if value is None or value == "":
                    parsed[key] = value
                    continue
                try:
                    parsed[key] = int(value)
                    continue
                except (TypeError, ValueError):
                    pass
                try:
                    parsed[key] = float(value)
                    continue
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def _is_truthy(value) -> bool:
    """Interpret CSV-style values as booleans."""
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(value)


def _event_from_step_row(row: Optional[dict]) -> Optional[dict]:
    """Build a lightweight event record from a step_metrics row."""
    if row is None:
        return None
    return {
        "step": int(row.get("step", 0)),
        "time_s": float(row.get("time_s", 0.0)),
        "active_layers": int(row.get("active_layers", row.get("step", 0))),
    }


def _find_first_flagged_event(rows: List[dict], flag_key: str) -> Optional[dict]:
    """Return the first row flagged by a step_metrics event column."""
    for row in rows:
        if _is_truthy(row.get(flag_key, 0)):
            return _event_from_step_row(row)
    return None


def _load_results_json(results_path: Optional[Path]) -> dict:
    """Load results JSON when it exists; otherwise return an empty dict."""
    if results_path is None or not Path(results_path).exists():
        return {}
    with open(results_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_event_records(results: dict, rows: List[dict]) -> Dict[str, Optional[dict]]:
    """Resolve event records from JSON when present, otherwise from CSV flags."""
    events = dict(results.get("events", {}))
    if events.get("first_damage") is None:
        events["first_damage"] = _find_first_flagged_event(rows, "first_damage_seen")
    if events.get("first_visible_gap") is None:
        events["first_visible_gap"] = _find_first_flagged_event(rows, "first_gap_seen")
    if events.get("first_junction_opening") is None:
        events["first_junction_opening"] = _find_first_flagged_event(
            rows, "first_junction_opening_seen"
        )
    return events


def _pick_heatmap_steps(events: dict, rows: List[dict]) -> List[tuple]:
    final_step = int(rows[-1]["step"]) if rows else 0
    selections = []
    first_damage = events.get("first_damage")
    first_gap = events.get("first_visible_gap")
    if first_damage is not None:
        selections.append(("First Damage", int(first_damage["step"])))
    if first_gap is not None:
        gap_step = int(first_gap["step"])
        if not selections or selections[-1][1] != gap_step:
            selections.append(("First Visible Gap", gap_step))
    if not selections or selections[-1][1] != final_step:
        selections.append(("Final State", final_step))
    return selections[:2] if len(selections) > 2 else selections


def _build_heatmap_matrix(
    profile_rows: List[dict], step: int, value_key: str
) -> np.ndarray:
    step_rows = [row for row in profile_rows if int(row["step"]) == int(step)]
    if not step_rows:
        return np.zeros((1, 1), dtype=np.float64)
    max_layer = max(int(row["lower_layer_zero_based"]) for row in step_rows)
    max_span = max(int(row["span_index"]) for row in step_rows)
    matrix = np.zeros((max_layer + 1, max_span + 1), dtype=np.float64)
    for row in step_rows:
        matrix[int(row["lower_layer_zero_based"]), int(row["span_index"])] = float(
            row[value_key]
        )
    return matrix


def _generate_heatmap_figure(
    profile_rows: List[dict],
    rows: List[dict],
    events: dict,
    output_dir: Path,
    plt,
) -> List[Path]:
    """Write the barrel-vault heatmap figure in horizontal B&W + Colorful Heatmap style."""
    out_paths: List[Path] = []
    selections = _pick_heatmap_steps(events, rows)
    if not selections:
        return out_paths

    FONT_SIZE = 8
    LABEL_SIZE = 10
    
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
    })

    n_sel = len(selections)
    n_panels = 2 * n_sel
    
    # Very tight, horizontal layout suitable for the standard width
    fig = plt.figure(figsize=(8.5, 2.5), facecolor='white')
    panel_width = 0.70 / n_panels
    axes = []
    
    for i in range(n_panels):
        x0 = 0.05 + i * (panel_width + 0.08)
        # Setup main axis and thin colorbar axis
        ax = fig.add_axes([x0, 0.20, panel_width * 0.85, 0.65])
        cax = fig.add_axes([x0 + panel_width * 0.88, 0.20, panel_width * 0.08, 0.65])
        axes.append((ax, cax))

    panel_labels = ["a", "b", "c", "d", "e", "f"]

    for col, (title, step) in enumerate(selections):
        open_matrix = _build_heatmap_matrix(profile_rows, step, "max_opening_mm")
        dmg_matrix = _build_heatmap_matrix(profile_rows, step, "max_damage")

        # Opening panel (Blues mapping for contrast against B&W)
        ax_open, cax_open = axes[2 * col]
        im0 = ax_open.imshow(open_matrix, origin="lower", aspect="auto", cmap="viridis")
        ax_open.text(0.0, 1.05, panel_labels[2 * col], ha="left", va="bottom", transform=ax_open.transAxes, fontsize=LABEL_SIZE, fontweight="bold")
        # ax_open.text(0.05, 0.90, f"{title}\nOpening", ha="left", va="top", transform=ax_open.transAxes, fontsize=FONT_SIZE, color="black", bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
        
        ax_open.set_xlabel("Span")
        if col == 0:
            ax_open.set_ylabel("Length")
        fig.colorbar(im0, cax=cax_open)

        # Damage panel (Reds mapping for contrast against B&W, fits the Red accent)
        ax_dmg, cax_dmg = axes[2 * col + 1]
        im1 = ax_dmg.imshow(dmg_matrix, origin="lower", aspect="auto", cmap="magma")
        ax_dmg.text(0.0, 1.05, panel_labels[2 * col + 1], ha="left", va="bottom", transform=ax_dmg.transAxes, fontsize=LABEL_SIZE, fontweight="bold")
        # ax_dmg.text(0.05, 0.90, f"{title}\nDamage", ha="left", va="top", transform=ax_dmg.transAxes, fontsize=FONT_SIZE, color="black", bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

        ax_dmg.set_xlabel("Span")
        fig.colorbar(im1, cax=cax_dmg)

    heatmap_pdf = output_dir / "barrel_vault_heatmaps_bw_red.pdf"
    heatmap_png = output_dir / "barrel_vault_heatmaps_bw_red.png"
    fig.savefig(heatmap_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(heatmap_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    out_paths.extend([heatmap_pdf, heatmap_png])
    return out_paths


def _resolve_figure_paths(
    run_dir: Optional[Path],
    step_metrics_path: Optional[Path],
    span_profiles_path: Optional[Path],
    output_dir: Optional[Path],
    results_path: Optional[Path],
) -> tuple[Path, Path, Path, Optional[Path]]:
    """Resolve figure input/output paths from a run dir or explicit CSV paths."""
    if run_dir is not None:
        run_dir = Path(run_dir)
        if step_metrics_path is None:
            step_metrics_path = run_dir / "step_metrics.csv"
        if span_profiles_path is None:
            span_profiles_path = run_dir / "span_profiles.csv"
        if output_dir is None:
            output_dir = run_dir
        if results_path is None:
            candidate = run_dir / "barrel_vault_results.json"
            results_path = candidate if candidate.exists() else None

    if step_metrics_path is None or span_profiles_path is None:
        raise ValueError("step_metrics.csv and span_profiles.csv are required")

    step_metrics_path = Path(step_metrics_path)
    span_profiles_path = Path(span_profiles_path)
    output_dir = Path(output_dir) if output_dir is not None else step_metrics_path.parent
    results_path = None if results_path is None else Path(results_path)
    return step_metrics_path, span_profiles_path, output_dir, results_path


def generate_barrel_vault_figures(
    run_dir: Optional[Path] = None,
    cfg: Optional[dict] = None,
    *,
    step_metrics_path: Optional[Path] = None,
    span_profiles_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    results_path: Optional[Path] = None,
) -> List[Path]:
    """Generate barrel-vault figures from saved CSV outputs in horizontal B&W + Red Accent style."""
    del cfg  # Kept for backward compatibility with older callers.

    (
        step_metrics_path,
        span_profiles_path,
        output_dir,
        results_path,
    ) = _resolve_figure_paths(
        run_dir,
        step_metrics_path,
        span_profiles_path,
        output_dir,
        results_path,
    )

    if not step_metrics_path.exists() or not span_profiles_path.exists():
        return []

    os.environ["MPLCONFIGDIR"] = str(output_dir / ".mplconfig")
    os.environ["XDG_CACHE_HOME"] = str(output_dir / ".cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    rows = _load_csv_rows(step_metrics_path)
    profile_rows = _load_csv_rows(span_profiles_path)
    results = _load_results_json(results_path)

    if not rows:
        return []

    x_vals = np.array(
        [row.get("active_layers", row.get("step", 0)) for row in rows], dtype=float
    )
    max_disp = np.array([float(row.get("max_disp_mm", 0.0)) for row in rows], dtype=float)
    cantilever_sag = np.array(
        [float(row.get("cantilever_sag_mm", 0.0)) for row in rows], dtype=float
    )
    max_open = np.array([float(row.get("max_opening_mm", 0.0)) for row in rows], dtype=float)
    junction_open = np.array(
        [float(row.get("junction_opening_mm", 0.0)) for row in rows], dtype=float
    )
    crown_open = np.array(
        [float(row.get("crown_opening_mm", 0.0)) for row in rows], dtype=float
    )
    front_open = np.array(
        [float(row.get("front_opening_mm", 0.0)) for row in rows], dtype=float
    )
    max_damage = np.array([float(row.get("max_damage", 0.0)) for row in rows], dtype=float)
    junction_damage = np.array(
        [float(row.get("junction_damage", 0.0)) for row in rows], dtype=float
    )
    crown_damage = np.array(
        [float(row.get("crown_damage", 0.0)) for row in rows], dtype=float
    )
    front_damage = np.array(
        [float(row.get("front_damage", 0.0)) for row in rows], dtype=float
    )
    front_ratio = np.array(
        [float(row.get("front_vs_crown_opening_ratio", 0.0)) for row in rows], dtype=float
    )
    junction_ratio = np.array(
        [float(row.get("junction_opening_ratio", 0.0)) for row in rows], dtype=float
    )

    events = _resolve_event_records(results, rows)
    event_damage = events.get("first_damage")
    event_gap = events.get("first_visible_gap")
    event_junction = events.get("first_junction_opening")

    out_paths: List[Path] = []

    # ---------------------------------------------------------------------------
    # Horizontal tight 4-panel layout (B&W + Red Accent)
    # ---------------------------------------------------------------------------
    FONT_SIZE = 8
    LABEL_SIZE = 10

    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "figure.titlesize": FONT_SIZE,
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
    })

    fig = plt.figure(figsize=(8.5, 2.5), facecolor='white')

    ax1 = fig.add_axes([0.05, 0.20, 0.18, 0.70])
    ax2 = fig.add_axes([0.30, 0.20, 0.18, 0.70])
    ax3 = fig.add_axes([0.55, 0.20, 0.18, 0.70])
    ax4 = fig.add_axes([0.80, 0.20, 0.18, 0.70])

    fig.text(0.01, 0.95, "a", fontsize=LABEL_SIZE, fontweight="bold", ha="left", va="top")
    fig.text(0.26, 0.95, "b", fontsize=LABEL_SIZE, fontweight="bold", ha="left", va="top")
    fig.text(0.51, 0.95, "c", fontsize=LABEL_SIZE, fontweight="bold", ha="left", va="top")
    fig.text(0.76, 0.95, "d", fontsize=LABEL_SIZE, fontweight="bold", ha="left", va="top")

    # Panel a: Displacement
    ax1.plot(x_vals, max_disp, ls='-', lw=2.0, color="#000000", label="Max displacement")
    ax1.plot(x_vals, cantilever_sag, ls='--', lw=2.0, color="#555555", label="Cantilever sag")
    ax1.set_xlabel("Active layers")
    ax1.set_ylabel("Displacement [mm]")
    ax1.legend(frameon=True, borderpad=0.3)
    ax1.grid(alpha=0.4)

    # Panel b: Opening
    ax2.plot(x_vals, max_open, ls='-', lw=2.0, color="#000000", label="Max opening")
    ax2.plot(x_vals, crown_open, ls='--', lw=1.8, color="#555555", label="Crown opening")
    ax2.plot(x_vals, junction_open, ls=':', lw=1.8, color="#888888", label="Junction opening")
    ax2.plot(x_vals, front_open, ls='-.', lw=1.8, color="#CC2222", label="Front opening")
    ax2.set_xlabel("Active layers")
    ax2.set_ylabel("Opening [mm]")
    ax2.legend(frameon=True, borderpad=0.3)
    ax2.grid(alpha=0.4)

    # Panel c: Damage
    ax3.plot(x_vals, max_damage, ls='-', lw=2.0, color="#000000", label="Max damage")
    ax3.plot(x_vals, crown_damage, ls='--', lw=1.8, color="#555555", label="Crown damage")
    ax3.plot(x_vals, junction_damage, ls=':', lw=1.8, color="#888888", label="Junction damage")
    ax3.plot(x_vals, front_damage, ls='-.', lw=1.8, color="#CC2222", label="Front damage")
    ax3.set_xlabel("Active layers")
    ax3.set_ylabel("Damage [-]")
    ax3.legend(frameon=True, borderpad=0.3)
    ax3.grid(alpha=0.4)

    # Panel d: Ratios
    ax4.plot(x_vals, front_ratio, ls='-.', lw=2.0, color="#CC2222", label="Front / Crown")
    ax4.plot(x_vals, junction_ratio, ls=':', lw=2.0, color="#888888", label="Junction / Max")
    ax4.set_xlabel("Active layers")
    ax4.set_ylabel("Ratio [-]")
    ax4.legend(frameon=True, borderpad=0.3)
    ax4.grid(alpha=0.4)

    for event in (event_junction, event_damage, event_gap):
        if event is None:
            continue
        x_event = float(event.get("active_layers", event.get("step", 0)))
        for ax in (ax1, ax2, ax3, ax4):
            ax.axvline(x_event, color="#777777", lw=1.0, ls=":", alpha=0.8)

    progress_pdf = output_dir / "barrel_vault_progress_bw_red.pdf"
    progress_png = output_dir / "barrel_vault_progress_bw_red.png"
    fig.savefig(progress_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(progress_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    out_paths.extend([progress_pdf, progress_png])

    out_paths.extend(
        _generate_heatmap_figure(profile_rows, rows, events, output_dir, plt)
    )

    return out_paths


def generate_barrel_vault_heatmaps(
    run_dir: Optional[Path] = None,
    *,
    step_metrics_path: Optional[Path] = None,
    span_profiles_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    results_path: Optional[Path] = None,
) -> List[Path]:
    """Generate only the barrel-vault heatmap figure from saved CSV outputs."""
    (
        step_metrics_path,
        span_profiles_path,
        output_dir,
        results_path,
    ) = _resolve_figure_paths(
        run_dir,
        step_metrics_path,
        span_profiles_path,
        output_dir,
        results_path,
    )

    if not step_metrics_path.exists() or not span_profiles_path.exists():
        return []

    os.environ["MPLCONFIGDIR"] = str(output_dir / ".mplconfig")
    os.environ["XDG_CACHE_HOME"] = str(output_dir / ".cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    rows = _load_csv_rows(step_metrics_path)
    profile_rows = _load_csv_rows(span_profiles_path)
    if not rows or not profile_rows:
        return []

    results = _load_results_json(results_path)
    events = _resolve_event_records(results, rows)
    return _generate_heatmap_figure(profile_rows, rows, events, output_dir, plt)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for figure regeneration from saved outputs."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        help="Completed run directory containing step_metrics.csv and span_profiles.csv",
    )
    parser.add_argument(
        "--step-metrics",
        type=Path,
        help="Path to step_metrics.csv",
    )
    parser.add_argument(
        "--span-profiles",
        type=Path,
        help="Path to span_profiles.csv",
    )
    parser.add_argument(
        "--results",
        type=Path,
        help="Optional path to barrel_vault_results.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to write figure files into",
    )
    args = parser.parse_args(argv)

    out_paths = generate_barrel_vault_figures(
        run_dir=args.run_dir,
        step_metrics_path=args.step_metrics,
        span_profiles_path=args.span_profiles,
        results_path=args.results,
        output_dir=args.output_dir,
    )
    if out_paths:
        for path in out_paths:
            print(path)
        return 0
    print("No figures generated.")
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI convenience.
    raise SystemExit(main())