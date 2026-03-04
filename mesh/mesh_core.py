# Author: Abdallah Kamhawi <Kamhawi@umich.edu>
# Package Maintainer: Abdallah Kamhawi <Kamhawi@umich.edu>

"""Core mesh primitives for layered hexahedral meshes.

This module provides geometry-agnostic building blocks used by vault meshes:
faces, hexahedral cells, layer containers, and interface definitions that
connect adjacent cells within or across layers. The classes here focus on
topology and connectivity, leaving geometric generation to specialized mesh
classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Set

import numpy as np
from tqdm import tqdm
from mpi4py import MPI

logger = logging.getLogger(__name__)
_TQDM_DISABLE = MPI.COMM_WORLD.size > 1 and MPI.COMM_WORLD.rank != 0

class FaceLocation(Enum):
    """Face locations defined by local node indices.

    The enum values map to the local face index used by ``HexahedronCell``.
    The ordering of nodes for each face is encoded in
    ``HexahedronCell.FACE_NODE_MAP``.

    Members:
        BOTTOM: Face with nodes (0, 1, 5, 4).
        TOP: Face with nodes (2, 6, 7, 3).
        FRONT: Face with nodes (0, 2, 3, 1).
        BACK: Face with nodes (4, 5, 7, 6).
        LEFT: Face with nodes (0, 4, 6, 2).
        RIGHT: Face with nodes (1, 3, 7, 5).

    Attributes:
        name: Enum member name.
        value: Enum member value.

    Methods:
        __repr__: Return a string representation.
        __str__: Return the member name.
    """

    BOTTOM = 0  # nodes (0, 3, 2, 1)
    TOP = 1  # nodes (4, 5, 6, 7)
    FRONT = 2  # nodes (0, 1, 5, 4)
    BACK = 3  # nodes (2, 3, 7, 6)
    LEFT = 4  # nodes (0, 4, 7, 3)
    RIGHT = 5  # nodes (1, 2, 6, 5)


class FaceLabel(Enum):
    """Semantic labels for face boundary classification.

    Labels are used to mark faces as free, inter-layer, intra-layer, or
    subject to Dirichlet/Neumann boundary conditions.

    Members:
        FREE: Unconstrained boundary face.
        INTER_LAYER: Face belongs to an inter-layer interface.
        INTRA_LAYER: Face belongs to an intra-layer interface.
        DIRICHLET: Face has a Dirichlet boundary condition.
        NEUMANN: Face has a Neumann boundary condition.

    Attributes:
        name: Enum member name.
        value: Enum member value.

    Methods:
        __repr__: Return a string representation.
        __str__: Return the member name.
    """

    FREE = "free"
    INTER_LAYER = "inter_layer"
    INTRA_LAYER = "intra_layer"
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"


class Face:
    """A quadrilateral face defined by four node indices.

    Args:
        node_indices: Global node indices for the face (ordered).
        location: FaceLocation enum value for this face.
        parent_cell: Owning cell instance.
        local_face_index: Local face index within the parent cell (if known).

    Attributes:
        node_indices: Global node indices for the face (ordered).
        location: FaceLocation enum value for this face.
        parent_cell: Owning cell instance.
        local_face_index: Local face index within the parent cell (if known).
        label: Current FaceLabel used for boundary classification.
        dirichlet_value: Value applied when the face is Dirichlet constrained.
        neumann_value: Value applied when the face is Neumann constrained.

    Methods:
        __init__: Initialize a Face instance.
        compute_normal: Compute a unit normal vector for the face.
        compute_centroid: Compute the face centroid.
        compute_area: Compute the quadrilateral face area.
        set_label: Set the face label.
        set_dirichlet: Apply a Dirichlet boundary condition.
        set_neumann: Apply a Neumann boundary condition.
        __repr__: Return a string representation.
    """

    def __init__(
        self,
        node_indices: np.ndarray,
        location: FaceLocation,
        parent_cell: Optional[HexahedronCell] = None,
        local_face_index: Optional[int] = None,
    ):
        """Initialize a quadrilateral face.

        Args:
            node_indices: Ordered global node indices for the face.
            location: Local face location enum.
            parent_cell: Optional owning hexahedral cell.
            local_face_index: Optional local face index in parent cell.

        Returns:
            None.

        Raises:
            None.
        """
        self.node_indices = np.array(node_indices, dtype=int)
        self.location = location
        self.parent_cell = parent_cell
        self.local_face_index = local_face_index
        self.label = FaceLabel.FREE
        self.dirichlet_value: Optional[float] = None
        self.neumann_value: Optional[float] = None
        # Debug output for first few faces (only if parent_cell has cell_id 0)
        if (
            parent_cell is not None
            and hasattr(parent_cell, "cell_id")
            and parent_cell.cell_id == 0
        ):
            logger.debug(
                "[Face.__init__] Created %s face with nodes %s for cell %s",
                location.name,
                self.node_indices.tolist(),
                parent_cell.cell_id,
            )

    def compute_normal(self, nodes: np.ndarray) -> np.ndarray:
        """Compute a unit normal vector for the face.

        Args:
            nodes: Node coordinate array of shape (n_nodes, dim).

        Returns:
            Unit normal vector as a 1D array. If the face is degenerate, the
            unnormalized normal is returned.

        Raises:
            None.
        """
        p0, p1, _, p3 = nodes[self.node_indices]
        normal = np.cross(p1 - p0, p3 - p0)
        norm = np.linalg.norm(normal)
        return normal / norm if norm > 1e-12 else normal

    def compute_centroid(self, nodes: np.ndarray) -> np.ndarray:
        """Compute the centroid of the face.

        Args:
            nodes: Node coordinate array of shape (n_nodes, dim).

        Returns:
            Centroid coordinates as a 1D array.

        Raises:
            None.
        """
        return np.mean(nodes[self.node_indices], axis=0)

    def compute_area(self, nodes: np.ndarray) -> float:
        """Compute the quadrilateral face area.

        The area is computed using the cross product of the face diagonals.

        Args:
            nodes: Node coordinate array of shape (n_nodes, dim).

        Returns:
            Scalar area estimate of the quadrilateral face.

        Raises:
            None.
        """
        p0, p1, p2, p3 = nodes[self.node_indices]
        return 0.5 * np.linalg.norm(np.cross(p2 - p0, p3 - p1))

    def set_label(self, label: FaceLabel) -> None:
        """Set the face label.

        Args:
            label: FaceLabel to assign.

        Returns:
            None.

        Raises:
            None.
        """
        self.label = label

    def set_dirichlet(self, value: float) -> None:
        """Apply a Dirichlet boundary condition to the face.

        Args:
            value: Boundary value to assign.

        Returns:
            None.

        Raises:
            None.
        """
        self.dirichlet_value = value
        self.neumann_value = None
        self.label = FaceLabel.DIRICHLET

    def set_neumann(self, value: float) -> None:
        """Apply a Neumann boundary condition to the face.

        Args:
            value: Boundary value to assign.

        Returns:
            None.

        Raises:
            None.
        """
        self.neumann_value = value
        self.dirichlet_value = None
        self.label = FaceLabel.NEUMANN

    def __repr__(self) -> str:
        """Return developer-friendly face representation.

        Args:
            None.

        Returns:
            str: String representation with nodes/location/label.

        Raises:
            None.
        """
        return (
            f"Face(nodes={self.node_indices.tolist()}, "
            f"location={self.location.name}, label={self.label.value})"
        )


class HexahedronCell:
    """A hexahedral (8-node brick) cell composed of 6 quadrilateral faces.

    Args:
        node_indices: Ordered node indices for this cell (length 8).
        cell_id: Optional global cell identifier.
        layer: Optional parent Layer.

    Attributes:
        node_indices: Ordered node indices for the cell (global indices).
        cell_id: Optional global cell identifier.
        birth_time: Optional birth time assigned by mesh-specific logic.
        layer: Optional parent Layer.
        faces: Mapping of FaceLocation to Face objects.

    Notation mapping (ASCII):
        Omega^e : bulk cell e
        nodes^e : ordered node indices for cell e
        f^e_k : local face index k for cell e

    Methods:
        __init__: Initialize a HexahedronCell instance.
        _create_faces: Create Face instances for each FaceLocation.
        get_face: Retrieve a face by FaceLocation.
        compute_centroid: Compute the cell centroid.
        __repr__: Return a string representation.
    """

    FACE_NODE_MAP = {
        FaceLocation.BOTTOM: (0, 1, 5, 4),
        FaceLocation.TOP: (2, 6, 7, 3),
        FaceLocation.FRONT: (0, 2, 3, 1),
        FaceLocation.BACK: (4, 5, 7, 6),
        FaceLocation.LEFT: (0, 4, 6, 2),
        FaceLocation.RIGHT: (1, 3, 7, 5),
    }
    # Local face index mapping f^e_k (k = 0..5)
    FACE_LOCAL_INDEX = {
        FaceLocation.BOTTOM: 0,
        FaceLocation.TOP: 1,
        FaceLocation.FRONT: 2,
        FaceLocation.BACK: 3,
        FaceLocation.LEFT: 4,
        FaceLocation.RIGHT: 5,
    }

    def __init__(
        self,
        node_indices: np.ndarray,
        cell_id: Optional[int] = None,
        layer: Optional[Layer] = None,
    ):
        """Initialize a hexahedral cell and construct its faces.

        Args:
            node_indices: Ordered global node indices (length 8).
            cell_id: Optional global cell identifier.
            layer: Optional parent layer.

        Returns:
            None.

        Raises:
            None.
        """
        # nodes^e: ordered node indices for this cell (global indices)
        self.node_indices = np.array(node_indices, dtype=int)
        # e: global cell index (optional)
        self.cell_id = cell_id
        # birth time assigned by mesh-specific logic (optional)
        self.birth_time: Optional[float] = None
        self.layer = layer
        self.faces: Dict[FaceLocation, Face] = {}
        self._create_faces()
        # Debug output for first few cells
        if cell_id is not None and cell_id < 3:
            layer_id = (
                layer.layer_id
                if layer is not None and hasattr(layer, "layer_id")
                else None
            )
            logger.debug(
                "[HexahedronCell.__init__] Created cell %s in layer %s with %s nodes",
                cell_id,
                layer_id,
                len(self.node_indices),
            )

    def _create_faces(self) -> None:
        """Create face objects for each local face location.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        for location, local_indices in self.FACE_NODE_MAP.items():
            global_indices = self.node_indices[list(local_indices)]
            local_face_index = self.FACE_LOCAL_INDEX[location]
            self.faces[location] = Face(
                global_indices,
                location,
                parent_cell=self,
                local_face_index=local_face_index,
            )

    def get_face(self, location: FaceLocation) -> Face:
        """Return the face at a given location.

        Args:
            location: FaceLocation enum value.

        Returns:
            Face object corresponding to the location.

        Raises:
            KeyError: If the requested location is unavailable.
        """
        return self.faces[location]

    def compute_centroid(self, nodes: np.ndarray) -> np.ndarray:
        """Compute the centroid of the hexahedral cell.

        Args:
            nodes: Node coordinate array of shape (n_nodes, dim).

        Returns:
            Centroid coordinates as a 1D array.

        Raises:
            None.
        """
        return np.mean(nodes[self.node_indices], axis=0)

    def __repr__(self) -> str:
        """Return developer-friendly cell representation.

        Args:
            None.

        Returns:
            str: String representation with id/node ids.

        Raises:
            None.
        """
        return f"HexahedronCell(id={self.cell_id}, nodes={self.node_indices.tolist()})"


class Layer:
    """A collection of hexahedral cells belonging to one layer.

    Args:
        layer_id: Optional layer identifier.
        mesh: Optional parent mesh that owns this layer.

    Attributes:
        cells: List of HexahedronCell instances in the layer.
        layer_id: Optional layer identifier assigned by the mesh.
        mesh: Optional parent HexahedronVolumetricMesh.

    Properties:
        n_cells: Number of cells in the layer.

    Methods:
        __init__: Initialize a Layer instance.
        add_cell: Add a cell and set its layer reference.
        __repr__: Return a string representation.
    """

    def __init__(
        self, layer_id: Optional[int] = None, mesh: Optional[HexahedronVolumetricMesh] = None
    ):
        """Initialize a mesh layer container.

        Args:
            layer_id: Optional layer id.
            mesh: Optional parent mesh reference.

        Returns:
            None.

        Raises:
            None.
        """
        self.cells: List[HexahedronCell] = []
        self.layer_id = layer_id
        self.mesh = mesh
        if layer_id is not None and layer_id < 3:
            logger.debug("[Layer.__init__] Created layer %s", layer_id)

    @property
    def n_cells(self) -> int:
        """Return number of cells in the layer.

        Args:
            None.

        Returns:
            int: Number of cells.

        Raises:
            None.
        """
        return len(self.cells)

    def add_cell(self, cell: HexahedronCell) -> None:
        """Add cell to layer and set back-reference.

        Args:
            cell: Cell to add.

        Returns:
            None.

        Raises:
            None.
        """
        cell.layer = self
        self.cells.append(cell)

    def __repr__(self) -> str:
        """Return developer-friendly layer representation.

        Args:
            None.

        Returns:
            str: String representation with id and cell count.

        Raises:
            None.
        """
        return f"Layer(id={self.layer_id}, n_cells={self.n_cells})"


@dataclass(frozen=True)
class DirichletBoundaryCondition:
    """Dirichlet boundary condition value holder.

    Attributes:
        value: Boundary condition value (type is solver-specific).

    Methods:
        __init__: Initialize the value holder.
        __repr__: Return a string representation.
        __eq__: Compare value holders.
        __hash__: Hash value holders (enabled via frozen dataclass).
    """

    value: Any


@dataclass(frozen=True)
class NeumannBoundaryCondition:
    """Neumann (flux/traction) boundary condition value holder.

    Attributes:
        value: Boundary condition value (type is solver-specific).

    Methods:
        __init__: Initialize the value holder.
        __repr__: Return a string representation.
        __eq__: Compare value holders.
        __hash__: Hash value holders (enabled via frozen dataclass).
    """

    value: Any


class InteriorFace(ABC):
    """Interior face connecting adjacent cells.

    An InteriorFace aggregates two (or more) Face instances that represent a
    shared interface between cells. Subclasses define the semantics (intra- or
    inter-layer).

    Attributes:
        faces: List of Face instances that form the interface.
        interior_face_id: Optional global interface identifier.

    Properties:
        n_faces: Number of faces aggregated by the interface.

    Methods:
        __init__: Initialize an InteriorFace instance.
        add_face: Add a face to the interface.
        get_shared_nodes: Return node indices common to all faces.
        get_all_nodes: Return node indices in any of the faces.
        compute_centroid: Compute centroid from face centroids.
        compute_area: Compute mean face area.
        compute_normal: Compute average unit normal.
        get_connectivity: Build oriented connectivity info.
        get_interface_type: Return interface type identifier string.
        __repr__: Return a string representation.
    """

    def __init__(self, interior_face_id: Optional[int] = None):
        """Initialize an interior interface container.

        Args:
            interior_face_id: Optional interface id.

        Returns:
            None.

        Raises:
            None.
        """
        self.faces: List[Face] = []
        self.interior_face_id = interior_face_id
        # Debug output for first few interfaces
        if interior_face_id is not None and interior_face_id < 3:
            logger.debug(
                "[InteriorFace.__init__] Created interior face %s", interior_face_id
            )

    @property
    def n_faces(self) -> int:
        """Return number of faces aggregated in this interface.

        Args:
            None.

        Returns:
            int: Number of faces.

        Raises:
            None.
        """
        return len(self.faces)

    def add_face(self, face: Face) -> None:
        """Add a face to the interface.

        Args:
            face: Face to append.

        Returns:
            None.

        Raises:
            None.
        """
        self.faces.append(face)

    def get_shared_nodes(self) -> np.ndarray:
        """Return node ids shared by all faces in the interface.

        Args:
            None.

        Returns:
            np.ndarray: Sorted shared node ids.

        Raises:
            None.
        """
        if not self.faces:
            return np.array([], dtype=int)
        shared: Set[int] = set(self.faces[0].node_indices)
        for face in self.faces[1:]:
            shared &= set(face.node_indices)
        return np.array(sorted(shared))

    def get_all_nodes(self) -> np.ndarray:
        """Return union of node ids across all interface faces.

        Args:
            None.

        Returns:
            np.ndarray: Sorted node ids present in any face.

        Raises:
            None.
        """
        all_nodes: Set[int] = set()
        for face in self.faces:
            all_nodes.update(face.node_indices)
        return np.array(sorted(all_nodes))

    def compute_centroid(self, nodes: np.ndarray) -> np.ndarray:
        """Compute interface centroid from face centroids.

        Args:
            nodes: Node coordinate array.

        Returns:
            np.ndarray: Interface centroid coordinates.

        Raises:
            None.
        """
        centroids = [face.compute_centroid(nodes) for face in self.faces]
        return np.mean(centroids, axis=0)

    def compute_area(self, nodes: np.ndarray) -> float:
        """Compute mean interface area over aggregated faces.

        Args:
            nodes: Node coordinate array.

        Returns:
            float: Mean face area.

        Raises:
            None.
        """
        if not self.faces:
            return 0.0
        return np.mean([face.compute_area(nodes) for face in self.faces])

    def compute_normal(self, nodes: np.ndarray) -> np.ndarray:
        """Compute average unit normal vector of the interface.

        Args:
            nodes: Node coordinate array.

        Returns:
            np.ndarray: Unit normal (or non-normalized fallback if degenerate).

        Raises:
            None.
        """
        normals = [face.compute_normal(nodes) for face in self.faces]
        avg_normal = np.mean(normals, axis=0)
        norm = np.linalg.norm(avg_normal)
        return avg_normal / norm if norm > 1e-12 else avg_normal

    def get_connectivity(
        self, nodes: Optional[np.ndarray], cell_index: dict[int, int]
    ) -> Optional[dict]:
        """Return interior face connectivity and oriented normal.

        Args:
            nodes: Optional node coordinate array. If provided, the interface
                normal is oriented from the minus cell toward the plus cell.
            cell_index: Mapping from ``id(cell)`` to contiguous cell index.

        Returns:
            A dictionary with keys ``e_plus``, ``e_minus``, ``f_plus``,
            ``f_minus``, ``n_i`` (unit normal), and ``shared_nodes``. Returns
            ``None`` if fewer than two faces are present.

        Raises:
            None.
        """
        faces = [f for f in self.faces if f.parent_cell is not None]
        if len(faces) < 2:
            return None
        face_minus, face_plus = faces[0], faces[1]
        e_minus = cell_index.get(id(face_minus.parent_cell))
        e_plus = cell_index.get(id(face_plus.parent_cell))
        f_minus = face_minus.local_face_index
        f_plus = face_plus.local_face_index

        n_i = None
        if nodes is not None and e_minus is not None and e_plus is not None:
            c_minus = face_minus.parent_cell.compute_centroid(nodes)
            c_plus = face_plus.parent_cell.compute_centroid(nodes)
            n_i = self.compute_normal(nodes)
            if np.dot(n_i, c_plus - c_minus) < 0.0:
                n_i = -n_i

        return {
            "e_plus": e_plus,
            "e_minus": e_minus,
            "f_plus": f_plus,
            "f_minus": f_minus,
            "n_i": n_i,
            "shared_nodes": self.get_shared_nodes(),
        }

    @abstractmethod
    def get_interface_type(self) -> str:
        """Return interface type identifier.

        Args:
            None.

        Returns:
            str: Interface type string.

        Raises:
            None.
        """
        pass

    def __repr__(self) -> str:
        """Return developer-friendly interface representation.

        Args:
            None.

        Returns:
            str: String representation with id and face count.

        Raises:
            None.
        """
        return (
            f"{self.__class__.__name__}(id={self.interior_face_id}, n_faces={self.n_faces})"
        )


class IntraLayerInterface(InteriorFace):
    """Interior face connecting faces of adjacent cells within the same layer.

    Attributes:
        layer: Layer that owns both cells (if known).
        orientation: FaceLocation indicating the interface orientation.

    Inherited attributes:
        faces: List of Face instances forming the interface.
        interior_face_id: Optional global interface identifier.

    Inherited properties:
        n_faces: Number of faces aggregated by the interface.

    Inherited methods:
        get_shared_nodes: Return node indices common to all faces.
        get_all_nodes: Return node indices in any of the faces.
        compute_centroid: Compute centroid from face centroids.
        compute_area: Compute mean face area.
        compute_normal: Compute average unit normal.
        get_connectivity: Build oriented connectivity info.

    Methods:
        __init__: Initialize an IntraLayerInterface instance.
        get_interface_type: Return the interface type identifier.
        add_face: Add a face and mark it as intra-layer.
        create_between_cells: Build an interface between two cells.
        get_connected_cells: Return connected cells.
        __repr__: Return a string representation.
    """

    def __init__(
        self,
        interior_face_id: Optional[int] = None,
        layer: Optional[Layer] = None,
        orientation: Optional[FaceLocation] = None,
    ):
        """Initialize an intra-layer interface.

        Args:
            interior_face_id: Optional interface id.
            layer: Optional owning layer.
            orientation: Optional interface orientation.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(interior_face_id)
        self.layer = layer
        self.orientation = orientation

    def get_interface_type(self) -> str:
        """Return interface type identifier.

        Args:
            None.

        Returns:
            str: ``"intra_layer"``.

        Raises:
            None.
        """
        return "intra_layer"

    def add_face(self, face: Face) -> None:
        """Add face and mark it as intra-layer.

        Args:
            face: Face to append.

        Returns:
            None.

        Raises:
            None.
        """
        super().add_face(face)
        face.set_label(FaceLabel.INTRA_LAYER)

    @classmethod
    def create_between_cells(
        cls,
        cell1: HexahedronCell,
        cell2: HexahedronCell,
        face_location1: FaceLocation,
        face_location2: FaceLocation,
        interior_face_id: Optional[int] = None,
    ) -> "IntraLayerInterface":
        """Create an intra-layer interface between two cells.

        Args:
            cell1: First cell.
            cell2: Second cell.
            face_location1: FaceLocation on cell1.
            face_location2: FaceLocation on cell2.
            interior_face_id: Optional interface id.

        Returns:
            IntraLayerInterface instance connecting the two faces.

        Raises:
            None.
        """
        interface = cls(interior_face_id=interior_face_id, orientation=face_location1)
        interface.add_face(cell1.get_face(face_location1))
        interface.add_face(cell2.get_face(face_location2))
        if cell1.layer is not None and cell1.layer is cell2.layer:
            interface.layer = cell1.layer

        # Debug output for first few interfaces
        if interior_face_id is not None and interior_face_id < 3:
            layer_id = (
                cell1.layer.layer_id
                if cell1.layer is not None and hasattr(cell1.layer, "layer_id")
                else None
            )
            logger.debug(
                "[IntraLayerInterface.create_between_cells] Created interface %s in layer %s, orientation %s",
                interior_face_id,
                layer_id,
                face_location1.name,
            )

        return interface

    def get_connected_cells(self) -> List[HexahedronCell]:
        """Return cells connected by this interface.

        Args:
            None.

        Returns:
            List[HexahedronCell]: Connected cells with valid parent pointers.

        Raises:
            None.
        """
        return [f.parent_cell for f in self.faces if f.parent_cell is not None]

    def __repr__(self) -> str:
        """Return developer-friendly intra-layer interface representation.

        Args:
            None.

        Returns:
            str: String representation with id/layer/orientation.

        Raises:
            None.
        """
        orient = self.orientation.name if self.orientation else "None"
        layer_id = self.layer.layer_id if self.layer else None
        return (
            f"IntraLayerInterface(id={self.interior_face_id}, layer={layer_id}, orientation={orient})"
        )


class InterLayerInterface(InteriorFace):
    """Interior face connecting faces of cells in adjacent layers.

    Attributes:
        lower_layer: Layer below the interface.
        upper_layer: Layer above the interface.

    Inherited attributes:
        faces: List of Face instances forming the interface.
        interior_face_id: Optional global interface identifier.

    Inherited properties:
        n_faces: Number of faces aggregated by the interface.

    Inherited methods:
        get_shared_nodes: Return node indices common to all faces.
        get_all_nodes: Return node indices in any of the faces.
        compute_centroid: Compute centroid from face centroids.
        compute_area: Compute mean face area.
        compute_normal: Compute average unit normal.
        get_connectivity: Build oriented connectivity info.

    Methods:
        __init__: Initialize an InterLayerInterface instance.
        get_interface_type: Return the interface type identifier.
        add_face: Add a face and mark it as inter-layer.
        create_between_cells: Build an interface between two stacked cells.
        create_between_layers: Build interfaces by matching faces in layers.
        get_connected_cells: Return the lower and upper connected cells.
        __repr__: Return a string representation.
    """

    def __init__(
        self,
        interior_face_id: Optional[int] = None,
        lower_layer: Optional[Layer] = None,
        upper_layer: Optional[Layer] = None,
    ):
        """Initialize an inter-layer interface.

        Args:
            interior_face_id: Optional interface id.
            lower_layer: Optional lower layer reference.
            upper_layer: Optional upper layer reference.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(interior_face_id)
        self.lower_layer = lower_layer
        self.upper_layer = upper_layer

    def get_interface_type(self) -> str:
        """Return interface type identifier.

        Args:
            None.

        Returns:
            str: ``"inter_layer"``.

        Raises:
            None.
        """
        return "inter_layer"

    def add_face(self, face: Face) -> None:
        """Add face and mark it as inter-layer.

        Args:
            face: Face to append.

        Returns:
            None.

        Raises:
            None.
        """
        super().add_face(face)
        face.set_label(FaceLabel.INTER_LAYER)

    @classmethod
    def create_between_cells(
        cls,
        lower_cell: HexahedronCell,
        upper_cell: HexahedronCell,
        interior_face_id: Optional[int] = None,
    ) -> "InterLayerInterface":
        """Create an inter-layer interface between two stacked cells.

        Args:
            lower_cell: Cell from the lower layer.
            upper_cell: Cell from the upper layer.
            interior_face_id: Optional interface id.

        Returns:
            InterLayerInterface instance connecting the two faces.

        Raises:
            None.
        """
        interface = cls(interior_face_id=interior_face_id)
        interface.add_face(lower_cell.get_face(FaceLocation.TOP))
        interface.add_face(upper_cell.get_face(FaceLocation.BOTTOM))
        if lower_cell.layer:
            interface.lower_layer = lower_cell.layer
        if upper_cell.layer:
            interface.upper_layer = upper_cell.layer

        # Debug output for first few interfaces
        if interior_face_id is not None and interior_face_id < 3:
            lower_layer_id = lower_cell.layer.layer_id if lower_cell.layer else None
            upper_layer_id = upper_cell.layer.layer_id if upper_cell.layer else None
            logger.debug(
                "[InterLayerInterface.create_between_cells] Created interface %s between layers %s and %s",
                interior_face_id,
                lower_layer_id,
                upper_layer_id,
            )

        return interface

    @classmethod
    def create_between_layers(
        cls, lower_layer: Layer, upper_layer: Layer, nodes: np.ndarray
    ) -> List["InterLayerInterface"]:
        """Create inter-layer interfaces by matching faces between layers.

        Args:
            lower_layer: Layer below.
            upper_layer: Layer above.
            nodes: Node coordinate array used for debug output.

        Returns:
            List of InterLayerInterface instances for matching faces.

        Raises:
            None.
        """
        lower_layer_id = (
            lower_layer.layer_id if lower_layer.layer_id is not None else "unknown"
        )
        upper_layer_id = (
            upper_layer.layer_id if upper_layer.layer_id is not None else "unknown"
        )

        logger.debug(
            "[InterLayerInterface.create_between_layers] Creating interfaces between layers %s and %s",
            lower_layer_id,
            upper_layer_id,
        )
        logger.debug(
            "[InterLayerInterface.create_between_layers] Lower layer has %s cells, upper layer has %s cells",
            lower_layer.n_cells,
            upper_layer.n_cells,
        )

        interfaces = []
        matches_found = 0
        upper_face_map: Dict[frozenset[int], HexahedronCell] = {}
        duplicate_upper_keys = 0

        for upper_cell in upper_layer.cells:
            bottom_key = frozenset(
                upper_cell.get_face(FaceLocation.BOTTOM).node_indices.tolist()
            )
            if bottom_key in upper_face_map:
                duplicate_upper_keys += 1
            upper_face_map[bottom_key] = upper_cell

        if duplicate_upper_keys > 0:
            logger.warning(
                "[InterLayerInterface.create_between_layers] Duplicate upper-layer face keys detected: %s",
                duplicate_upper_keys,
            )

        with tqdm(
            total=lower_layer.n_cells,
            desc=f"Matching cells L{lower_layer_id}-L{upper_layer_id}",
            unit="cell",
            disable=_TQDM_DISABLE,
        ) as pbar:
            for lower_cell in lower_layer.cells:
                top_key = frozenset(
                    lower_cell.get_face(FaceLocation.TOP).node_indices.tolist()
                )
                upper_cell = upper_face_map.get(top_key)
                if upper_cell is not None:
                    interfaces.append(
                        cls.create_between_cells(lower_cell, upper_cell, len(interfaces))
                    )
                    matches_found += 1
                elif len(interfaces) == 0:
                    logger.warning(
                        "[InterLayerInterface.create_between_layers] No match found for lower_cell in layer %s",
                        lower_layer_id,
                    )
                pbar.update(1)

        logger.debug(
            "[InterLayerInterface.create_between_layers] Created %s interfaces (%s matches found)",
            len(interfaces),
            matches_found,
        )
        return interfaces

    def get_connected_cells(self) -> tuple[Optional[HexahedronCell], Optional[HexahedronCell]]:
        """Return lower and upper connected cells inferred from face locations.

        Args:
            None.

        Returns:
            tuple[Optional[HexahedronCell], Optional[HexahedronCell]]:
                ``(lower_cell, upper_cell)``.

        Raises:
            None.
        """
        lower, upper = None, None
        for face in self.faces:
            if face.parent_cell:
                if face.location in (FaceLocation.TOP, FaceLocation.BACK):
                    lower = face.parent_cell
                elif face.location in (FaceLocation.BOTTOM, FaceLocation.FRONT):
                    upper = face.parent_cell
        return lower, upper

    def __repr__(self) -> str:
        """Return developer-friendly inter-layer interface representation.

        Args:
            None.

        Returns:
            str: String representation with id and adjacent layer ids.

        Raises:
            None.
        """
        lower_id = self.lower_layer.layer_id if self.lower_layer else None
        upper_id = self.upper_layer.layer_id if self.upper_layer else None
        return (
            f"InterLayerInterface(id={self.interior_face_id}, layers=({lower_id}, {upper_id}))"
        )


class HexahedronVolumetricMesh:
    """Base class for volumetric meshes composed of hexahedral cells in layers.

    Attributes:
        nodes: Node coordinate array of shape (n_nodes, dim).
        layers: List of Layer instances.
        interfaces: List of InteriorFace instances (combined list).
        intra_layer_interfaces: List of interfaces within layers.
        inter_layer_interfaces: List of interfaces between layers.

    Properties:
        n_nodes: Number of mesh nodes.
        n_layers: Number of layers.
        n_cells: Number of cells across all layers.
        Gamma_c_i: Alias for the combined interface list.

    Methods:
        __init__: Initialize an empty volumetric mesh.
        add_layer: Add a layer to the mesh and assign its id.
        add_interface: Add an interface to the combined list.
        add_intra_layer_interface: Add an intra-layer interface.
        add_inter_layer_interface: Add an inter-layer interface.
        get_all_cells: Return a flat list of all cells.
        get_interface_connectivity_table: Build connectivity arrays and normals.
        __repr__: Return a string representation.
    """

    def __init__(self):
        """Initialize empty volumetric mesh containers.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        # X_j = (X_j, Y_j, Z_j), j = 1..N_n stored as array (N_n x d)
        self.nodes: Optional[np.ndarray] = None
        # Layers group cells; overall cell set is Omega^e, e = 1..N_e
        self.layers: List[Layer] = []
        # Interfaces Gamma_c^i, i = 1..N_i
        self.interfaces: List[InteriorFace] = []
        self.intra_layer_interfaces: List[InteriorFace] = []
        self.inter_layer_interfaces: List[InteriorFace] = []

    @property
    def n_nodes(self) -> int:
        """Return number of mesh nodes.

        Args:
            None.

        Returns:
            int: Number of nodes.

        Raises:
            None.
        """
        return len(self.nodes) if self.nodes is not None else 0

    @property
    def n_layers(self) -> int:
        """Return number of layers.

        Args:
            None.

        Returns:
            int: Number of layers.

        Raises:
            None.
        """
        return len(self.layers)

    @property
    def n_cells(self) -> int:
        """Return total number of cells across all layers.

        Args:
            None.

        Returns:
            int: Total cell count.

        Raises:
            None.
        """
        return sum(layer.n_cells for layer in self.layers)

    @property
    def Gamma_c_i(self) -> List[InteriorFace]:
        """Return combined interface list ``Gamma_c^i``.

        Args:
            None.

        Returns:
            List[InteriorFace]: Combined interface objects.

        Raises:
            None.
        """
        return self.interfaces

    def add_layer(self, layer: Layer) -> None:
        """Add layer to mesh and assign ``layer_id``.

        Args:
            layer: Layer to append.

        Returns:
            None.

        Raises:
            None.
        """
        layer.mesh = self
        layer.layer_id = len(self.layers)
        self.layers.append(layer)
        if len(self.layers) <= 3 or len(self.layers) % 10 == 0:
            logger.debug(
                "[HexahedronVolumetricMesh.add_layer] Added layer %s with %s cells (total layers: %s)",
                layer.layer_id,
                layer.n_cells,
                len(self.layers),
            )

    def add_interface(self, interface: InteriorFace) -> None:
        """Add interface to mesh and assign contiguous id.

        Args:
            interface: Interface to append.

        Returns:
            None.

        Raises:
            None.
        """
        interface.interior_face_id = len(self.interfaces)
        self.interfaces.append(interface)

    def add_intra_layer_interface(self, interface: InteriorFace) -> None:
        """Add intra-layer interface to both global and intra-layer lists.

        Args:
            interface: Intra-layer interface.

        Returns:
            None.

        Raises:
            None.
        """
        self.add_interface(interface)
        self.intra_layer_interfaces.append(interface)

    def add_inter_layer_interface(self, interface: InteriorFace) -> None:
        """Add inter-layer interface to both global and inter-layer lists.

        Args:
            interface: Inter-layer interface.

        Returns:
            None.

        Raises:
            None.
        """
        self.add_interface(interface)
        self.inter_layer_interfaces.append(interface)

    def get_all_cells(self) -> List[HexahedronCell]:
        """Return flattened list of all cells across layers.

        Args:
            None.

        Returns:
            List[HexahedronCell]: Flat cell list.

        Raises:
            None.
        """
        return [cell for layer in self.layers for cell in layer.cells]

    def get_interface_connectivity_table(self) -> dict:
        """Return interface connectivity table and normals.

        Args:
            None.

        Returns:
            Dictionary with keys ``e_plus``, ``e_minus``, ``f_plus``,
            ``f_minus``, ``n_i`` (unit normals), and ``shared_nodes``.

        Raises:
            None.
        """
        logger.debug(
            "[HexahedronVolumetricMesh.get_interface_connectivity_table] Building connectivity table for %s interfaces...",
            len(self.interfaces),
        )

        cell_index = {id(cell): i for i, cell in enumerate(self.get_all_cells())}
        logger.debug(
            "[HexahedronVolumetricMesh.get_interface_connectivity_table] Created cell index map for %s cells",
            len(cell_index),
        )

        e_plus = []
        e_minus = []
        f_plus = []
        f_minus = []
        n_i = []
        shared_nodes = []

        with tqdm(
            total=len(self.interfaces),
            desc="Processing interfaces",
            unit="interface",
            disable=_TQDM_DISABLE,
        ) as pbar:
            for iface in self.interfaces:
                info = iface.get_connectivity(self.nodes, cell_index)
                if info is None:
                    continue
                e_plus.append(info["e_plus"] if info["e_plus"] is not None else -1)
                e_minus.append(info["e_minus"] if info["e_minus"] is not None else -1)
                f_plus.append(info["f_plus"] if info["f_plus"] is not None else -1)
                f_minus.append(info["f_minus"] if info["f_minus"] is not None else -1)
                n_i.append(
                    info["n_i"]
                    if info["n_i"] is not None
                    else np.zeros(self.nodes.shape[1])
                )
                shared_nodes.append(info["shared_nodes"])
                pbar.update(1)

        result = {
            "e_plus": np.array(e_plus, dtype=int),
            "e_minus": np.array(e_minus, dtype=int),
            "f_plus": np.array(f_plus, dtype=int),
            "f_minus": np.array(f_minus, dtype=int),
            "n_i": np.vstack(n_i) if n_i else np.zeros((0, self.nodes.shape[1])),
            "shared_nodes": shared_nodes,
        }

        logger.debug(
            "[HexahedronVolumetricMesh.get_interface_connectivity_table] Connectivity table complete: %s valid interfaces",
            len(e_plus),
        )
        return result

    def __repr__(self) -> str:
        """Return developer-friendly volumetric mesh representation.

        Args:
            None.

        Returns:
            str: String representation with node/layer/cell counts.

        Raises:
            None.
        """
        return (
            f"HexahedronVolumetricMesh(n_nodes={self.n_nodes}, n_layers={self.n_layers}, n_cells={self.n_cells})"
        )


__all__ = [
    "Face",
    "FaceLabel",
    "FaceLocation",
    "HexahedronCell",
    "Layer",
    "InteriorFace",
    "IntraLayerInterface",
    "InterLayerInterface",
    "DirichletBoundaryCondition",
    "NeumannBoundaryCondition",
    "HexahedronVolumetricMesh",
]
