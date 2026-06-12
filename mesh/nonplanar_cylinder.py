# Author: Abdallah Kamhawi <Kamhawi@umich.edu>

"""Non-planar conformal cylinder mesh built from measured slicing data.

This module generates the collapse-print validation geometry directly from
the Rhino slicing export (``validation/collapse_print/slicing.json``): 30
closed wavy centerline rings x 96 azimuthal points with a 20 mm wall. The
mesh conforms to the measured non-planar layer geometry:

- ``i``: circumferential direction (azimuth, ring point order = print order),
- ``j``: radial direction (inner/outer wall offset from the centerline),
- ``k``: vertical printed-layer direction.

Node sheets follow the approved validation plan:

- sheet 0 is the FLAT build plate at ``z = 0`` (the first bead's bottom
  conforms to the table even though the toolpath waves),
- sheet ``k`` (k = 1..n_layers) is the per-azimuth 3D midpoint of centerline
  rings ``k-1`` and ``k`` — the top ring of the stack is measured data, no
  extrapolation.

The vertical printed layers use ``FRONT``/``BACK`` faces exactly like the
hollow-cylinder mesh so the existing inter-layer support mapping, activation
logic, strip partitioner, and boundary tagging are reused unchanged. All
coordinates are millimetres (framework convention, g = 9810 mm/s^2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

import numpy as np
from mpi4py import MPI
from tqdm import tqdm

from .hollow_cylinder import (
    HollowCylinderFaceType,
    HollowCylinderHexahedronCell,
)
from .mesh_core import (
    Face,
    FaceLocation,
    HexahedronVolumetricMesh,
    IntraLayerInterface,
    InterLayerInterface,
    Layer,
)

logger = logging.getLogger(__name__)
_TQDM_DISABLE = MPI.COMM_WORLD.size > 1 and MPI.COMM_WORLD.rank != 0


def _fit_axis_center_xy(points_xy: np.ndarray) -> np.ndarray:
    """Least-squares (Kasa) circle-center fit of the print axis.

    Args:
        points_xy: ``(N, 2)`` array of centerline points projected to the
            xy-plane.

    Returns:
        ``(2,)`` array with the fitted circle center ``(cx, cy)``.
    """
    A = np.column_stack(
        [2.0 * points_xy[:, 0], 2.0 * points_xy[:, 1], np.ones(len(points_xy))]
    )
    b = (points_xy**2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol[:2]


class NonPlanarCylinderMesh(HexahedronVolumetricMesh):
    """Conformal hex mesh of the non-planar collapse-print cylinder.

    Args:
        slicing_json_path: Path to the Rhino slicing export
            (``slicing.json``: closed centerline rings, mm units).
        n_layers: Number of printed layers to mesh (requires ``n_layers + 1``
            measured rings so the top sheet needs no extrapolation).
        layer_height: Nominal layer height h0 [mm]; sets the build-plate
            offset below the first ring's mean elevation.
        n_thickness: Cells through the wall thickness (the validation case
            uses 1; birth times are only defined for 1).
        generate_interfaces: Whether to build inter-/intra-layer interfaces.
        dirichlet_boundary_conditions: Optional ``{face_type: value}`` map
            (e.g. ``{"base": 0.0}``) applied after generation.
        neumann_boundary_conditions: Optional ``{face_type: value}`` map.

    Attributes:
        n_span: Azimuthal cell count (96 from the slicing export).
        n_thickness: Radial cell count.
        n_length: Vertical printed-layer count (= ``n_layers``).
        wall_thickness: Wall thickness [mm] from the slicing export.
        layer_height: Nominal layer height h0 [mm].
        axis_center_xy: Fitted axis center in original Rhino coordinates.
        axis_center_json: Axis center stored in the slicing export.
        z_plate: Build-plate elevation in original Rhino coordinates.
        sheet_centerline: ``(n_layers+1, n_span, 3)`` translated centerline
            points of every node sheet (sheet 0 = plate footprint).
        sheet_z: ``(n_layers+1, n_span)`` sheet elevations.
        azimuth_theta: ``(n_span,)`` azimuth angle of each ring station about
            the fitted axis (sheet 0 footprint).
        node_table: ``(n_layers+1, n_span, n_thickness+1)`` structured map
            ``(sheet k, azimuth i, radial j) -> global node id`` for the
            settlement extractor and front-arc selection.
        ring_path_length_mm: Per-layer 3D centerline path length, filled by
            :meth:`compute_birth_times`.
        layer_first_birth_s / layer_last_birth_s: Per-layer birth-time
            bounds, filled by :meth:`compute_birth_times`.
    """

    def __init__(
        self,
        slicing_json_path: Union[str, Path],
        n_layers: int = 11,
        layer_height: float = 9.0,
        n_thickness: int = 1,
        generate_interfaces: bool = True,
        dirichlet_boundary_conditions: Optional[
            Dict[Union[int, str, HollowCylinderFaceType], float]
        ] = None,
        neumann_boundary_conditions: Optional[
            Dict[Union[int, str, HollowCylinderFaceType], float]
        ] = None,
    ):
        super().__init__()
        self.slicing_json_path = Path(slicing_json_path)
        self.n_length = int(n_layers)
        self.layer_height = float(layer_height)
        self.n_thickness = int(n_thickness)

        self.dirichlet_boundary_conditions: Dict[Union[int, str], float] = {}
        self.neumann_boundary_conditions: Dict[Union[int, str], float] = {}

        self.ring_path_length_mm: Optional[np.ndarray] = None
        self.layer_first_birth_s: Optional[np.ndarray] = None
        self.layer_last_birth_s: Optional[np.ndarray] = None

        self._load_slicing_rings()
        self._build_sheets()
        self._generate_nodes()
        self._generate_layers_and_cells()
        if generate_interfaces:
            self._generate_interfaces()

        if dirichlet_boundary_conditions:
            self.assign_dirichlet_boundary_conditions(dirichlet_boundary_conditions)
        if neumann_boundary_conditions:
            self.assign_neumann_boundary_conditions(neumann_boundary_conditions)

    # ------------------------------------------------------------------
    # Geometry from measured slicing data
    # ------------------------------------------------------------------

    def _load_slicing_rings(self) -> None:
        """Load, validate, and axis-align the measured centerline rings."""
        with open(self.slicing_json_path) as f:
            data = json.load(f)

        if str(data.get("units", "")).lower() != "millimeter":
            raise ValueError(
                f"slicing export units must be millimeter, got {data.get('units')!r}."
            )
        if str(data.get("represents", "")).lower() != "centerline":
            raise ValueError(
                "slicing export must represent the bead centerline, got "
                f"{data.get('represents')!r}."
            )

        self.n_span = int(data["n_span"])
        self.wall_thickness = float(data["wall_thickness"])
        self.axis_center_json = np.asarray(data["axis_xy"], dtype=float)

        layers = sorted(data["layers"], key=lambda L: int(L["k"]))
        n_rings_needed = self.n_length + 1
        if len(layers) < n_rings_needed:
            raise ValueError(
                f"Need {n_rings_needed} measured rings for {self.n_length} layers "
                f"(top sheet uses ring {self.n_length}); export has {len(layers)}."
            )

        rings = []
        for k, entry in enumerate(layers[:n_rings_needed]):
            if int(entry["k"]) != k:
                raise ValueError(f"Ring index mismatch: expected k={k}, got {entry['k']}.")
            ring = np.asarray(entry["points"], dtype=float)
            if ring.shape != (self.n_span, 3):
                raise ValueError(
                    f"Ring {k} has shape {ring.shape}, expected ({self.n_span}, 3)."
                )
            rings.append(ring)
        rings = np.asarray(rings)  # (n_rings, n_span, 3)

        # Fit the print axis from the data (least squares over all used rings)
        # and translate it to the origin; the exported axis is kept for QA.
        self.axis_center_xy = _fit_axis_center_xy(
            rings[:, :, :2].reshape(-1, 2)
        )
        axis_delta = float(np.linalg.norm(self.axis_center_xy - self.axis_center_json))
        logger.info(
            "[NonPlanarCylinderMesh] Fitted axis (%.3f, %.3f) mm; "
            "|fitted - exported| = %.3f mm",
            self.axis_center_xy[0],
            self.axis_center_xy[1],
            axis_delta,
        )

        # Build plate at z = 0: the first bead's centerline sits h0/2 above
        # the table on average (the wave makes layer 1 thickness vary).
        self.z_plate = float(rings[0][:, 2].mean() - 0.5 * self.layer_height)

        rings[:, :, 0] -= self.axis_center_xy[0]
        rings[:, :, 1] -= self.axis_center_xy[1]
        rings[:, :, 2] -= self.z_plate
        self._rings = rings

    def _build_sheets(self) -> None:
        """Build node-sheet centerlines: flat plate + measured midpoints."""
        n_sheets = self.n_length + 1
        sheet_centerline = np.zeros((n_sheets, self.n_span, 3), dtype=float)

        # Sheet 0: flat plate footprint (ring-0 xy, z = 0).
        sheet_centerline[0, :, :2] = self._rings[0][:, :2]
        sheet_centerline[0, :, 2] = 0.0

        # Sheets 1..n_layers: per-azimuth 3D midpoint of rings (k-1, k).
        for k in range(1, n_sheets):
            sheet_centerline[k] = 0.5 * (self._rings[k - 1] + self._rings[k])

        self.sheet_centerline = sheet_centerline
        self.sheet_z = sheet_centerline[:, :, 2].copy()
        self.azimuth_theta = np.arctan2(
            sheet_centerline[0, :, 1], sheet_centerline[0, :, 0]
        )

        # Per-azimuth strict monotonicity in k: required for a valid
        # extrusion-like stack (verified true in the measured data).
        dz = np.diff(self.sheet_z, axis=0)
        if not (dz > 0.0).all():
            bad = np.argwhere(dz <= 0.0)
            raise ValueError(
                f"Sheet elevations are not strictly increasing in k at "
                f"(sheet, azimuth) pairs {bad[:5].tolist()}..."
            )

    def _generate_nodes(self) -> None:
        """Place nodes at +/- radial offsets from each sheet centerline."""
        n_sheets = self.n_length + 1
        offsets = np.linspace(
            -0.5 * self.wall_thickness, 0.5 * self.wall_thickness, self.n_thickness + 1
        )

        total_nodes = self.n_span * (self.n_thickness + 1) * n_sheets
        self.nodes = np.zeros((total_nodes, 3), dtype=float)
        self.node_table = np.zeros(
            (n_sheets, self.n_span, self.n_thickness + 1), dtype=int
        )

        with tqdm(
            total=total_nodes,
            desc="Generating non-planar cylinder nodes",
            unit="node",
            disable=_TQDM_DISABLE,
        ) as pbar:
            for k in range(n_sheets):
                centers = self.sheet_centerline[k]
                radial = centers[:, :2] / np.linalg.norm(
                    centers[:, :2], axis=1, keepdims=True
                )
                for j, offset in enumerate(offsets):
                    for i in range(self.n_span):
                        idx = self._node_index(i, j, k)
                        self.nodes[idx, :2] = centers[i, :2] + offset * radial[i]
                        self.nodes[idx, 2] = centers[i, 2]
                        self.node_table[k, i, j] = idx
                        pbar.update(1)

    # ------------------------------------------------------------------
    # Topology (same patterns as the hollow-cylinder mesh)
    # ------------------------------------------------------------------

    def _node_index(self, i: int, j: int, k: int) -> int:
        return (
            k * self.n_span * (self.n_thickness + 1)
            + j * self.n_span
            + (i % self.n_span)
        )

    def _generate_layers_and_cells(self) -> None:
        cell_id = 0
        with tqdm(
            total=self.n_length,
            desc="Generating non-planar cylinder layers",
            unit="layer",
            disable=_TQDM_DISABLE,
        ) as pbar_layer:
            for k in range(self.n_length):
                layer = Layer()
                for j in range(self.n_thickness):
                    for i in range(self.n_span):
                        i_next = (i + 1) % self.n_span
                        node_indices = np.array(
                            [
                                self._node_index(i, j, k),
                                self._node_index(i_next, j, k),
                                self._node_index(i, j + 1, k),
                                self._node_index(i_next, j + 1, k),
                                self._node_index(i, j, k + 1),
                                self._node_index(i_next, j, k + 1),
                                self._node_index(i, j + 1, k + 1),
                                self._node_index(i_next, j + 1, k + 1),
                            ],
                            dtype=int,
                        )
                        layer.add_cell(
                            HollowCylinderHexahedronCell(
                                node_indices,
                                cell_id=cell_id,
                                span_index=i,
                                thickness_index=j,
                                length_index=k,
                                n_span=self.n_span,
                                n_thickness=self.n_thickness,
                                n_length=self.n_length,
                            )
                        )
                        cell_id += 1
                self.add_layer(layer)
                pbar_layer.update(1)

    def _generate_interfaces(self) -> None:
        with tqdm(
            total=self.n_length - 1,
            desc="Non-planar cylinder inter-layer interfaces",
            unit="layer_pair",
            disable=_TQDM_DISABLE,
        ) as pbar:
            for k in range(self.n_length - 1):
                lower_layer = self.layers[k]
                upper_layer = self.layers[k + 1]
                for lower_cell, upper_cell in zip(lower_layer.cells, upper_layer.cells):
                    iface = InterLayerInterface(
                        interior_face_id=len(self.inter_layer_interfaces)
                    )
                    iface.add_face(lower_cell.get_face(FaceLocation.BACK))
                    iface.add_face(upper_cell.get_face(FaceLocation.FRONT))
                    iface.lower_layer = lower_layer
                    iface.upper_layer = upper_layer
                    self.add_inter_layer_interface(iface)
                pbar.update(1)

        with tqdm(
            total=self.n_length,
            desc="Non-planar cylinder intra-layer interfaces",
            unit="layer",
            disable=_TQDM_DISABLE,
        ) as pbar:
            for layer in self.layers:
                cells_2d = [
                    layer.cells[j * self.n_span : (j + 1) * self.n_span]
                    for j in range(self.n_thickness)
                ]

                for j in range(self.n_thickness):
                    for i in range(self.n_span):
                        cell_a = cells_2d[j][i]
                        cell_b = cells_2d[j][(i + 1) % self.n_span]
                        iface = IntraLayerInterface.create_between_cells(
                            cell_a,
                            cell_b,
                            FaceLocation.RIGHT,
                            FaceLocation.LEFT,
                        )
                        iface.layer = layer
                        self.add_intra_layer_interface(iface)

                for j in range(self.n_thickness - 1):
                    for i in range(self.n_span):
                        cell_a = cells_2d[j][i]
                        cell_b = cells_2d[j + 1][i]
                        iface = IntraLayerInterface.create_between_cells(
                            cell_a,
                            cell_b,
                            FaceLocation.TOP,
                            FaceLocation.BOTTOM,
                        )
                        iface.layer = layer
                        self.add_intra_layer_interface(iface)
                pbar.update(1)

    # ------------------------------------------------------------------
    # Birth times from the wavy 3D toolpath
    # ------------------------------------------------------------------

    def compute_birth_times(self, tcp_speed: float, mode: str = "layer") -> np.ndarray:
        """Assign birth times from cumulative 3D centroid path distance.

        Barrel-vault convention: walk the cells in deposition order (layer by
        layer, azimuth order = ring point order) and accumulate the 3D
        distance between successive cell centroids, so the wavy path length
        is included (~13.6 s ring period at 50 mm/s, matching the video).

        Args:
            tcp_speed: Tool center point speed [mm/s].
            mode: Unused; retained for interface compatibility.

        Returns:
            Array of birth times indexed by mesh cell order.

        Raises:
            ValueError: If ``tcp_speed`` is not strictly positive.
            NotImplementedError: If ``n_thickness != 1`` (deposition order
                through the wall is undefined).
        """
        del mode
        if tcp_speed <= 0.0:
            raise ValueError("tcp_speed must be positive.")
        if self.n_thickness != 1:
            raise NotImplementedError(
                "Birth times are only defined for n_thickness = 1 (one bead)."
            )

        nodes = self.nodes
        all_cells = self.get_all_cells()
        cell_index = {id(cell): idx for idx, cell in enumerate(all_cells)}

        birth_times = np.zeros(len(all_cells), dtype=float)
        ring_lengths = np.zeros(self.n_length, dtype=float)
        first_birth = np.zeros(self.n_length, dtype=float)
        last_birth = np.zeros(self.n_length, dtype=float)

        cumulative_distance = 0.0
        prev_centroid = None

        with tqdm(
            total=self.n_length,
            desc="Computing non-planar birth times",
            unit="layer",
            disable=_TQDM_DISABLE,
        ) as pbar:
            for layer_idx, layer in enumerate(self.layers):
                layer_start_distance = cumulative_distance
                for cell in layer.cells:
                    centroid = cell.compute_centroid(nodes)
                    if prev_centroid is not None:
                        cumulative_distance += np.linalg.norm(centroid - prev_centroid)

                    birth_time = cumulative_distance / tcp_speed
                    cell.birth_time = birth_time
                    birth_times[cell_index[id(cell)]] = birth_time
                    prev_centroid = centroid

                first_birth[layer_idx] = layer.cells[0].birth_time
                last_birth[layer_idx] = layer.cells[-1].birth_time
                ring_lengths[layer_idx] = cumulative_distance - layer_start_distance
                pbar.update(1)

        self.ring_path_length_mm = ring_lengths
        self.layer_first_birth_s = first_birth
        self.layer_last_birth_s = last_birth

        logger.info(
            "[NonPlanarCylinderMesh.compute_birth_times] Total path %.1f mm, "
            "total time %.2f s, ring periods %.2f-%.2f s",
            cumulative_distance,
            cumulative_distance / tcp_speed,
            ring_lengths.min() / tcp_speed,
            ring_lengths.max() / tcp_speed,
        )
        return birth_times

    # ------------------------------------------------------------------
    # Boundary conditions (same semantics as the hollow-cylinder mesh)
    # ------------------------------------------------------------------

    def assign_dirichlet_boundary_conditions(
        self,
        boundary_conditions: Dict[Union[int, str, HollowCylinderFaceType], float],
    ) -> None:
        self._assign_boundary_conditions(boundary_conditions, target="dirichlet")

    def assign_neumann_boundary_conditions(
        self,
        boundary_conditions: Dict[Union[int, str, HollowCylinderFaceType], float],
    ) -> None:
        self._assign_boundary_conditions(boundary_conditions, target="neumann")

    def _assign_boundary_conditions(
        self,
        boundary_conditions: Dict[Union[int, str, HollowCylinderFaceType], float],
        target: str,
    ) -> None:
        if not isinstance(boundary_conditions, dict):
            raise TypeError(
                "boundary_conditions must be a dict like {cell_id: value} or "
                "{face_type: value}."
            )
        if not boundary_conditions:
            return

        all_cells = self.get_all_cells()
        if not all_cells:
            return

        is_id_map = all(isinstance(key, int) for key in boundary_conditions.keys())
        if is_id_map:
            cell_by_id = {
                cell.cell_id: cell for cell in all_cells if cell.cell_id is not None
            }
            for cell_id, value in boundary_conditions.items():
                cell = cell_by_id.get(cell_id)
                if cell is None:
                    continue
                if target == "dirichlet":
                    self.dirichlet_boundary_conditions[cell_id] = value
                else:
                    self.neumann_boundary_conditions[cell_id] = value
                self._apply_boundary_condition_to_faces(cell, target, value)
            return

        face_type_map: Dict[str, float] = {}
        for key, value in boundary_conditions.items():
            if isinstance(key, HollowCylinderFaceType):
                type_key = key.value
            else:
                type_key = str(key).lower().replace("-", "_").replace(" ", "_")
            face_type_map[type_key] = value

        for cell in all_cells:
            boundary_faces = self._get_boundary_faces_for_cell(cell)
            for face in boundary_faces:
                face_type = getattr(face, "face_type", None)
                if face_type is None:
                    continue
                face_key = face_type.value if hasattr(face_type, "value") else str(face_type)
                if face_key not in face_type_map:
                    continue
                value = face_type_map[face_key]
                if target == "dirichlet":
                    self.dirichlet_boundary_conditions[face_key] = value
                    face.set_dirichlet(value)
                else:
                    self.neumann_boundary_conditions[face_key] = value
                    face.set_neumann(value)

    def _apply_boundary_condition_to_faces(
        self,
        cell: HollowCylinderHexahedronCell,
        target: str,
        value: float,
    ) -> None:
        for face in self._get_boundary_faces_for_cell(cell):
            if target == "dirichlet":
                face.set_dirichlet(value)
            else:
                face.set_neumann(value)

    def _get_boundary_faces_for_cell(
        self,
        cell: HollowCylinderHexahedronCell,
    ) -> List[Face]:
        j = getattr(cell, "thickness_index", None)
        k = getattr(cell, "length_index", None)
        if j is None or k is None:
            return []

        boundary_faces: List[Face] = []
        if j == 0:
            boundary_faces.append(cell.get_face(FaceLocation.BOTTOM))
        if j == self.n_thickness - 1:
            boundary_faces.append(cell.get_face(FaceLocation.TOP))
        if k == 0:
            boundary_faces.append(cell.get_face(FaceLocation.FRONT))
        if k == self.n_length - 1:
            boundary_faces.append(cell.get_face(FaceLocation.BACK))
        return boundary_faces

    def __repr__(self) -> str:
        return (
            "NonPlanarCylinderMesh(n_span="
            f"{self.n_span}, n_thickness={self.n_thickness}, "
            f"n_layers={self.n_length}, n_cells={self.n_cells})"
        )


__all__ = ["NonPlanarCylinderMesh"]
