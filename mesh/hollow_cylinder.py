# Author: Abdallah Kamhawi <Kamhawi@umich.edu>

"""Hollow-cylinder mesh classes built on the shared hexahedral core.

This module adds an add-only cylinder geometry implementation that reuses the
existing mesh core abstractions without modifying the original barrel-vault
workflow. The cylinder is structured as:

- ``i``: circumferential direction,
- ``j``: radial direction,
- ``k``: vertical printed-layer direction.

The vertical printed layers use ``FRONT``/``BACK`` faces so the existing
inter-layer support mapping and activation logic can be reused unchanged.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Union
import logging

import numpy as np
from mpi4py import MPI
from tqdm import tqdm

from .mesh_core import (
    Face,
    FaceLocation,
    HexahedronCell,
    HexahedronVolumetricMesh,
    IntraLayerInterface,
    InterLayerInterface,
    Layer,
)

logger = logging.getLogger(__name__)
_TQDM_DISABLE = MPI.COMM_WORLD.size > 1 and MPI.COMM_WORLD.rank != 0


class HollowCylinderFaceType(Enum):
    """Semantic face categories for the hollow-cylinder mesh."""

    INNER_WALL = "inner_wall"
    OUTER_WALL = "outer_wall"
    BASE = "base"
    TOP = "top"
    CIRCUMFERENTIAL = "circumferential"


class HollowCylinderFace(Face):
    """Quadrilateral face carrying hollow-cylinder boundary semantics."""

    def __init__(
        self,
        node_indices: np.ndarray,
        location: FaceLocation,
        face_type: HollowCylinderFaceType,
        parent_cell: Optional["HollowCylinderHexahedronCell"] = None,
        local_face_index: Optional[int] = None,
    ):
        super().__init__(node_indices, location, parent_cell, local_face_index)
        self.face_type = face_type

    def __repr__(self) -> str:
        return (
            "HollowCylinderFace(type="
            f"{self.face_type.value}, location={self.location.name}, "
            f"nodes={self.node_indices.tolist()})"
        )


class HollowCylinderHexahedronCell(HexahedronCell):
    """Hexahedral cell with hollow-cylinder indexing metadata."""

    def __init__(
        self,
        node_indices: np.ndarray,
        cell_id: Optional[int] = None,
        layer: Optional[Layer] = None,
        span_index: Optional[int] = None,
        thickness_index: Optional[int] = None,
        length_index: Optional[int] = None,
        n_span: Optional[int] = None,
        n_thickness: Optional[int] = None,
        n_length: Optional[int] = None,
    ):
        self.span_index = span_index
        self.thickness_index = thickness_index
        self.length_index = length_index
        self.n_span = n_span
        self.n_thickness = n_thickness
        self.n_length = n_length
        super().__init__(node_indices, cell_id=cell_id, layer=layer)

    def _create_faces(self) -> None:
        for location, local_indices in self.FACE_NODE_MAP.items():
            global_indices = self.node_indices[list(local_indices)]
            local_face_index = self.FACE_LOCAL_INDEX[location]
            face_type = self._map_face_type(location)
            self.faces[location] = HollowCylinderFace(
                global_indices,
                location,
                face_type,
                parent_cell=self,
                local_face_index=local_face_index,
            )

    def _map_face_type(self, location: FaceLocation) -> HollowCylinderFaceType:
        if location == FaceLocation.BOTTOM:
            return HollowCylinderFaceType.INNER_WALL
        if location == FaceLocation.TOP:
            return HollowCylinderFaceType.OUTER_WALL
        if location == FaceLocation.FRONT:
            return HollowCylinderFaceType.BASE
        if location == FaceLocation.BACK:
            return HollowCylinderFaceType.TOP
        return HollowCylinderFaceType.CIRCUMFERENTIAL

    def __repr__(self) -> str:
        return (
            "HollowCylinderHexahedronCell(id="
            f"{self.cell_id}, ijk=({self.span_index}, {self.thickness_index}, "
            f"{self.length_index}), nodes={self.node_indices.tolist()})"
        )


class HollowCylinderVolumetricMesh(HexahedronVolumetricMesh):
    """Structured hollow-cylinder mesh with vertical printed layers."""

    def __init__(
        self,
        heartline_radius: float,
        height: float,
        thickness: float,
        n_span: int,
        n_length: int,
        n_thickness: int,
        layer_height: Optional[float] = None,
        imperfection_amplitude: float = 1.0,
        generate_interfaces: bool = True,
        dirichlet_boundary_conditions: Optional[
            Dict[Union[int, str, HollowCylinderFaceType], float]
        ] = None,
        neumann_boundary_conditions: Optional[
            Dict[Union[int, str, HollowCylinderFaceType], float]
        ] = None,
    ):
        super().__init__()
        self.heartline_radius = float(heartline_radius)
        self.height = float(height)
        self.thickness = float(thickness)
        self.n_span = int(n_span)
        self.n_length = int(n_length)
        self.n_thickness = int(n_thickness)
        self.layer_height = (
            float(layer_height)
            if layer_height is not None
            else self.height / max(self.n_length, 1)
        )
        self.imperfection_amplitude = float(imperfection_amplitude)

        self.inner_radius = self.heartline_radius - 0.5 * self.thickness
        self.outer_radius = self.heartline_radius + 0.5 * self.thickness

        self.dirichlet_boundary_conditions: Dict[Union[int, str], float] = {}
        self.neumann_boundary_conditions: Dict[Union[int, str], float] = {}

        self._generate_nodes()
        self._generate_layers_and_cells()
        if generate_interfaces:
            self._generate_interfaces()

        if dirichlet_boundary_conditions:
            self.assign_dirichlet_boundary_conditions(dirichlet_boundary_conditions)
        if neumann_boundary_conditions:
            self.assign_neumann_boundary_conditions(neumann_boundary_conditions)

    def _generate_nodes(self) -> None:
        angles = np.linspace(0.0, 2.0 * np.pi, self.n_span, endpoint=False)
        radii = np.linspace(self.inner_radius, self.outer_radius, self.n_thickness + 1)
        z_coords = np.linspace(0.0, self.height, self.n_length + 1)

        total_nodes = self.n_span * (self.n_thickness + 1) * (self.n_length + 1)
        self.nodes = np.zeros((total_nodes, 3), dtype=float)

        idx = 0
        with tqdm(
            total=total_nodes,
            desc="Generating cylinder nodes",
            unit="node",
            disable=_TQDM_DISABLE,
        ) as pbar:
            for k, z_val in enumerate(z_coords):
                if self.height > 0.0:
                    imperf_scale = np.sin(np.pi * z_val / self.height)
                else:
                    imperf_scale = 0.0
                for j, radius in enumerate(radii):
                    for i, theta in enumerate(angles):
                        radius_eff = (
                            radius
                            + self.imperfection_amplitude * imperf_scale * np.cos(theta)
                        )
                        self.nodes[idx] = [
                            radius_eff * np.cos(theta),
                            radius_eff * np.sin(theta),
                            z_val,
                        ]
                        idx += 1
                        pbar.update(1)

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
            desc="Generating cylinder layers",
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

    def compute_birth_times(self, tcp_speed: float, mode: str = "layer") -> np.ndarray:
        del mode
        if tcp_speed <= 0.0:
            raise ValueError("tcp_speed must be positive.")

        layer_time = 2.0 * np.pi * self.heartline_radius / float(tcp_speed)
        segment_time = layer_time / max(self.n_span, 1)

        cells = self.get_all_cells()
        birth_times = np.zeros(len(cells), dtype=float)
        cell_index = {id(cell): idx for idx, cell in enumerate(cells)}

        with tqdm(
            total=len(cells),
            desc="Computing cylinder birth times",
            unit="cell",
            disable=_TQDM_DISABLE,
        ) as pbar:
            for cell in cells:
                if cell.length_index is None or cell.span_index is None:
                    continue
                layer_start = float(cell.length_index) * layer_time
                birth_time = layer_start + (float(cell.span_index) + 0.5) * segment_time
                cell.birth_time = birth_time
                birth_times[cell_index[id(cell)]] = birth_time
                pbar.update(1)

        return birth_times

    def _generate_interfaces(self) -> None:
        with tqdm(
            total=self.n_length - 1,
            desc="Cylinder inter-layer interfaces",
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
            desc="Cylinder intra-layer interfaces",
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
        i = getattr(cell, "span_index", None)
        j = getattr(cell, "thickness_index", None)
        k = getattr(cell, "length_index", None)
        if i is None or j is None or k is None:
            return []

        del i
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
            "HollowCylinderVolumetricMesh(heartline_radius="
            f"{self.heartline_radius}, height={self.height}, thickness={self.thickness}, "
            f"n_layers={self.n_length}, n_cells={self.n_cells})"
        )


__all__ = [
    "HollowCylinderFaceType",
    "HollowCylinderFace",
    "HollowCylinderHexahedronCell",
    "HollowCylinderVolumetricMesh",
]
