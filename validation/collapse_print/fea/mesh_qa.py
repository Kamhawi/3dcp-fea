# Author: Abdallah Kamhawi <Kamhawi@umich.edu>

"""Mesh QA for the non-planar conformal collapse-print cylinder (GATE B).

Builds ``NonPlanarCylinderMesh`` from the measured slicing export, verifies
topology (interface counts, sheet monotonicity, birth times), runs the
framework mesh-quality gate on the DOLFINx mesh, persists the structured
node table, and renders the inspection figure ``output/mesh_qa.png``
(front view + 3D, colored by printed layer).

Run serially from the repo root inside the ``fea`` conda env:

    python -m validation.collapse_print.fea.mesh_qa
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from mpi4py import MPI
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mesh import (
    FaceLocation,
    NonPlanarCylinderMesh,
    build_partitioned_dolfinx_mesh,
    evaluate_mesh_quality,
)

CASE_DIR = Path(__file__).resolve().parents[1]
SLICING_JSON = CASE_DIR / "slicing.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

N_LAYERS = 11
LAYER_HEIGHT_MM = 9.0
TCP_SPEED_MM_S = 50.0

# Same thresholds as the shipped config/config.yaml mesh_quality block.
MESH_QUALITY_CFG = {
    "mesh_quality": {
        "enabled": True,
        "aspect_ratio_threshold": 10.0,
        "min_scaled_jacobian": 0.1,
        "jacobian_near_zero_tol": 1.0e-10,
        "volume_variation_threshold": 20.0,
        "max_skewness": 0.5,
        "max_warping": 0.1,
        "fail_on_negative_jacobian": True,
    }
}


def build_mesh() -> NonPlanarCylinderMesh:
    """Build the validation mesh with the base Dirichlet label applied."""
    mesh = NonPlanarCylinderMesh(
        SLICING_JSON,
        n_layers=N_LAYERS,
        layer_height=LAYER_HEIGHT_MM,
        n_thickness=1,
        dirichlet_boundary_conditions={"base": 0.0},
    )
    mesh.compute_birth_times(tcp_speed=TCP_SPEED_MM_S)
    return mesh


def report_topology(mesh: NonPlanarCylinderMesh) -> None:
    """Print and assert the structural QA checks."""
    n_inter = len(mesh.inter_layer_interfaces)
    n_intra = len(mesh.intra_layer_interfaces)
    expected_inter = mesh.n_span * (mesh.n_length - 1)
    expected_intra = mesh.n_span * mesh.n_length

    print(f"Mesh: {mesh}")
    print(f"  Nodes: {mesh.n_nodes} (expected {96 * 2 * (N_LAYERS + 1)})")
    print(f"  Cells: {mesh.n_cells} (expected {96 * N_LAYERS})")
    print(f"  Inter-layer interfaces: {n_inter} (expected {expected_inter})")
    print(f"  Intra-layer interfaces: {n_intra} (expected {expected_intra})")
    assert n_inter == expected_inter, "inter-layer interface count mismatch"
    assert n_intra == expected_intra, "intra-layer interface count mismatch"

    n_base = sum(
        1
        for cell in mesh.layers[0].cells
        if cell.get_face(FaceLocation.FRONT).dirichlet_value is not None
    )
    print(f"  Dirichlet base faces (sheet 0): {n_base} (expected {mesh.n_span})")
    assert n_base == mesh.n_span, "base Dirichlet faces missing"

    dz = np.diff(mesh.sheet_z, axis=0)
    print(
        f"  Sheet spacing dz: min {dz.min():.3f} mm, max {dz.max():.3f} mm "
        f"(strictly monotone: {(dz > 0).all()})"
    )
    print(
        f"  Sheet 1 z: [{mesh.sheet_z[1].min():.2f}, {mesh.sheet_z[1].max():.2f}] mm; "
        f"sheet {N_LAYERS} z: [{mesh.sheet_z[-1].min():.2f}, "
        f"{mesh.sheet_z[-1].max():.2f}] mm"
    )
    print(
        f"  Axis fit (Rhino coords): ({mesh.axis_center_xy[0]:.3f}, "
        f"{mesh.axis_center_xy[1]:.3f}); exported axis delta "
        f"{np.linalg.norm(mesh.axis_center_xy - mesh.axis_center_json):.3f} mm; "
        f"z_plate {mesh.z_plate:.3f} mm"
    )

    periods = mesh.ring_path_length_mm / TCP_SPEED_MM_S
    print(
        f"  Ring path lengths: {mesh.ring_path_length_mm.min():.1f}-"
        f"{mesh.ring_path_length_mm.max():.1f} mm -> periods "
        f"{periods.min():.2f}-{periods.max():.2f} s @ {TCP_SPEED_MM_S:.0f} mm/s"
    )
    print(
        f"  Birth times: first {mesh.layer_first_birth_s[0]:.2f} s, "
        f"last {mesh.layer_last_birth_s[-1]:.2f} s "
        f"(total print {mesh.layer_last_birth_s[-1]:.1f} s)"
    )


def report_quality(mesh: NonPlanarCylinderMesh) -> bool:
    """Run the framework mesh-quality gate on the DOLFINx mesh."""
    comm = MPI.COMM_WORLD
    msh, _cells_lst, _cells_layers = build_partitioned_dolfinx_mesh(
        mesh, comm, partitioner_mode="strip"
    )
    report = evaluate_mesh_quality(msh, MESH_QUALITY_CFG, comm)

    print("Mesh quality (DOLFINx gate):")
    for name, stats in [
        ("aspect ratio", report.aspect_ratio_stats),
        ("scaled Jacobian", report.scaled_jacobian_stats),
        ("skewness", report.skewness_stats),
        ("warping", report.warping_stats),
        ("volume [mm^3]", report.volume_stats),
        ("edge length [mm]", report.edge_length_stats),
    ]:
        print(
            f"  {name:18s} min {stats['min']:.4g}  max {stats['max']:.4g}  "
            f"mean {stats['mean']:.4g}"
        )
    print(f"  Negative-Jacobian cells: {len(report.negative_jacobian_cells)}")
    print(f"  Flagged (aspect/skew/warp): "
          f"{len(report.high_aspect_ratio_cells)}/"
          f"{len(report.high_skewness_cells)}/"
          f"{len(report.high_warping_cells)}")
    for warning in report.warnings_issued:
        print(f"  WARNING: {warning}")
    print(f"  Verdict: {'PASS' if report.is_valid else 'FAIL'}")
    return report.is_valid


def save_node_table(mesh: NonPlanarCylinderMesh) -> Path:
    """Persist the structured node table for the settlement extractor."""
    cells = mesh.get_all_cells()
    path = OUTPUT_DIR / "mesh_node_table.npz"
    np.savez(
        path,
        node_table=mesh.node_table,
        sheet_centerline=mesh.sheet_centerline,
        sheet_z=mesh.sheet_z,
        azimuth_theta=mesh.azimuth_theta,
        nodes=mesh.nodes,
        span_index=np.array([c.span_index for c in cells], dtype=int),
        layer_index=np.array([c.length_index for c in cells], dtype=int),
        birth_time_s=np.array([c.birth_time for c in cells], dtype=float),
        ring_path_length_mm=mesh.ring_path_length_mm,
        layer_first_birth_s=mesh.layer_first_birth_s,
        layer_last_birth_s=mesh.layer_last_birth_s,
        axis_center_xy=mesh.axis_center_xy,
        axis_center_json=mesh.axis_center_json,
        z_plate=np.array(mesh.z_plate),
        layer_height_mm=np.array(mesh.layer_height),
        wall_thickness_mm=np.array(mesh.wall_thickness),
        tcp_speed_mm_s=np.array(TCP_SPEED_MM_S),
    )
    return path


def _outer_faces_by_layer(mesh: NonPlanarCylinderMesh):
    """Return (quad vertex arrays, layer ids, mean y) of outer-wall faces."""
    quads, layer_ids, y_means = [], [], []
    for layer in mesh.layers:
        for cell in layer.cells:
            face = cell.get_face(FaceLocation.TOP)  # outer wall
            verts = mesh.nodes[face.node_indices]
            quads.append(verts)
            layer_ids.append(layer.layer_id)
            y_means.append(verts[:, 1].mean())
    return np.array(quads), np.array(layer_ids), np.array(y_means)


def render_qa_figure(mesh: NonPlanarCylinderMesh, quality_ok: bool) -> Path:
    """Render the GATE-B inspection figure (front view, 3D, sheet profiles)."""
    quads, layer_ids, y_means = _outer_faces_by_layer(mesh)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, mesh.n_length))

    fig = plt.figure(figsize=(12.5, 4.4), dpi=200)

    # (a) Front view: orthographic x-z projection of the outer wall,
    # painter-sorted far-to-near (camera looking along -y).
    ax_front = fig.add_axes([0.05, 0.10, 0.30, 0.78])
    order = np.argsort(-y_means)
    polys = quads[order][:, :, [0, 2]]
    face_colors = colors[layer_ids[order]]
    ax_front.add_collection(
        PolyCollection(
            polys, facecolors=face_colors, edgecolors="k", linewidths=0.25
        )
    )
    ax_front.axhline(0.0, color="k", linewidth=0.8)
    ax_front.set_xlim(-125, 125)
    ax_front.set_ylim(-8, mesh.sheet_z.max() + 8)
    ax_front.set_aspect("equal")
    ax_front.set_xlabel("x [mm]")
    ax_front.set_ylabel("z [mm]")
    ax_front.set_title("(a) front view, outer wall")

    # (b) 3D view of the outer wall colored by layer.
    ax_3d = fig.add_axes([0.35, 0.02, 0.30, 0.94], projection="3d")
    ax_3d.add_collection3d(
        Poly3DCollection(
            quads, facecolors=colors[layer_ids], edgecolors="k", linewidths=0.15
        )
    )
    rmax = 125.0
    ax_3d.set_xlim(-rmax, rmax)
    ax_3d.set_ylim(-rmax, rmax)
    ax_3d.set_zlim(0, 2 * rmax * 0.55)
    ax_3d.set_box_aspect((1, 1, 0.55))
    ax_3d.view_init(elev=18, azim=-60)
    ax_3d.set_xlabel("x [mm]")
    ax_3d.set_ylabel("y [mm]")
    ax_3d.set_zlabel("z [mm]")
    ax_3d.set_title("(b) printed layers 1-11", y=0.98)

    # (c) Per-azimuth sheet elevations: the measured non-planar wave.
    ax_sheet = fig.add_axes([0.79, 0.10, 0.19, 0.78])
    theta_deg = np.degrees(mesh.azimuth_theta)
    order_th = np.argsort(theta_deg)
    ax_sheet.plot(
        theta_deg[order_th], mesh.sheet_z[0][order_th], color="k", linewidth=1.0
    )
    for k in range(1, mesh.n_length + 1):
        ax_sheet.plot(
            theta_deg[order_th],
            mesh.sheet_z[k][order_th],
            color=colors[k - 1],
            linewidth=1.0,
        )
    ax_sheet.set_xlabel("azimuth [deg]")
    ax_sheet.set_ylabel("sheet z [mm]")
    ax_sheet.set_xlim(-180, 180)
    ax_sheet.set_title("(c) node sheets 0-11")

    fig.suptitle(
        f"Non-planar conformal mesh QA - {mesh.n_cells} cells "
        f"(96 x 1 x {mesh.n_length}), {mesh.n_nodes} nodes, "
        f"quality gate: {'PASS' if quality_ok else 'FAIL'}",
        fontsize=11,
    )

    path = OUTPUT_DIR / "mesh_qa.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mesh = build_mesh()
    report_topology(mesh)
    quality_ok = report_quality(mesh)
    table_path = save_node_table(mesh)
    print(f"Node table saved: {table_path}")
    fig_path = render_qa_figure(mesh, quality_ok)
    print(f"QA figure saved: {fig_path}")

    if not quality_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
