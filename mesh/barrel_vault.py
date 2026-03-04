# Author: Abdallah Kamhawi <Kamhawi@umich.edu>
# Package Maintainer: Abdallah Kamhawi <Kamhawi@umich.edu>

"""Barrel vault mesh classes built on layered hexahedral primitives.

This module defines barrel vault-specific faces, cell types, and a volumetric
mesh generator that discretizes a circular-arc vault by span, thickness, and
length.
"""

from __future__ import annotations
from enum import Enum
from typing import Dict, List, Optional, Union
import logging
import numpy as np
from tqdm import tqdm
from mpi4py import MPI

logger = logging.getLogger(__name__)
_TQDM_DISABLE = MPI.COMM_WORLD.size > 1 and MPI.COMM_WORLD.rank != 0
from .mesh_core import (
    Face,
    FaceLocation,
    HexahedronCell,
    HexahedronVolumetricMesh,
    IntraLayerInterface,
    InterLayerInterface,
    Layer,
)


class BarrelVaultFaceType(Enum):
    """Semantic face categories for a barrel vault cell.

    These labels are used for boundary condition assignment.

    Members:
        INTRADOS: Inner curved surface of the vault.
        EXTRADOS: Outer curved surface of the vault.
        CROWN_FACING: Lateral face oriented toward the crown.
        SPRINGING_FACING: Lateral face oriented toward the springing.
        UP_VAULT: Face at the positive length direction.
        DOWN_VAULT: Face at the negative length direction.

    Attributes:
        name: Enum member name.
        value: Enum member value.

    Methods:
        __repr__: Return a string representation.
        __str__: Return the member name.
    """

    INTRADOS = "intrados"
    EXTRADOS = "extrados"
    CROWN_FACING = "crown_facing"
    SPRINGING_FACING = "springing_facing"
    UP_VAULT = "up_vault"
    DOWN_VAULT = "down_vault"


class BarrelVaultFace(Face):
    """A quadrilateral face with barrel vault semantics.

    Args:
        node_indices: Global node indices for the face (ordered).
        location: FaceLocation enum value for this face.
        face_type: BarrelVaultFaceType semantic classification.
        parent_cell: Owning BarrelVaultHexahedronCell.
        local_face_index: Local face index within the parent cell (if known).

    Attributes:
        face_type: BarrelVaultFaceType semantic classification.

    Inherited attributes:
        node_indices: Global node indices for the face (ordered).
        location: FaceLocation enum value for this face.
        parent_cell: Owning cell instance.
        local_face_index: Local face index within the parent cell (if known).
        label: Current FaceLabel used for boundary classification.
        dirichlet_value: Value applied when the face is Dirichlet constrained.
        neumann_value: Value applied when the face is Neumann constrained.

    Inherited methods:
        compute_normal: Compute a unit normal vector for the face.
        compute_centroid: Compute the face centroid.
        compute_area: Compute the quadrilateral face area.
        set_label: Set the face label.
        set_dirichlet: Apply a Dirichlet boundary condition.
        set_neumann: Apply a Neumann boundary condition.

    Methods:
        __init__: Initialize a BarrelVaultFace instance.
        __repr__: Return a string representation.
    """

    def __init__(
        self,
        node_indices: np.ndarray,
        location: FaceLocation,
        face_type: BarrelVaultFaceType,
        parent_cell: Optional["BarrelVaultHexahedronCell"] = None,
        local_face_index: Optional[int] = None,
    ):
        """Initialize a barrel-vault face with semantic face typing.

        Args:
            node_indices: Ordered face node indices.
            location: Face location enum on the parent cell.
            face_type: Barrel-vault semantic face category.
            parent_cell: Optional owning cell.
            local_face_index: Optional local face index in parent cell.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(node_indices, location, parent_cell, local_face_index)
        self.face_type = face_type

    def __repr__(self) -> str:
        """Return developer-friendly representation of the face.

        Args:
            None.

        Returns:
            str: String representation with type/location/node ids.

        Raises:
            None.
        """
        return (
            "BarrelVaultFace(type="
            f"{self.face_type.value}, location={self.location.name}, "
            f"nodes={self.node_indices.tolist()})"
        )


class BarrelVaultCellType(Enum):
    """Cell categories along the barrel vault arch.

    Members:
        SPRINGER: Cells near the springing/supports.
        HAUNCH: Cells between springer and crown zones.
        CROWN: Cells near the crown (apex).

    Attributes:
        name: Enum member name.
        value: Enum member value.

    Methods:
        __repr__: Return a string representation.
        __str__: Return the member name.
    """

    SPRINGER = "springer"
    HAUNCH = "haunch"
    CROWN = "crown"


class BarrelVaultHexahedronCell(HexahedronCell):
    """Hexahedron cell with barrel vault face and cell classifications.

    The cell type is classified by position along the span:
    springer (near supports), haunch, or crown (near the apex).

    Args:
        node_indices: Ordered node indices for this cell (length 8).
        cell_id: Optional global cell identifier.
        layer: Optional parent Layer.
        span_index: Index along the span direction.
        thickness_index: Index through the thickness direction.
        length_index: Index along the length direction (layer index).
        n_span: Number of cells along the span.
        n_thickness: Number of cells through the thickness.
        n_length: Number of layers along the length.

    Attributes:
        span_index: Index along the span direction.
        thickness_index: Index through the thickness direction.
        length_index: Index along the length direction.
        n_span: Number of cells along the span.
        n_thickness: Number of cells through the thickness.
        n_length: Number of layers along the length.
        vault_cell_type: Classified BarrelVaultCellType or None.

    Inherited attributes:
        node_indices: Ordered node indices for the cell (global indices).
        cell_id: Optional global cell identifier.
        birth_time: Optional birth time assigned by mesh-specific logic.
        layer: Optional parent Layer.
        faces: Mapping of FaceLocation to Face objects.

    Inherited methods:
        get_face: Retrieve a face by FaceLocation.
        compute_centroid: Compute the cell centroid.

    Methods:
        __init__: Initialize a BarrelVaultHexahedronCell instance.
        _create_faces: Create BarrelVaultFace instances for each FaceLocation.
        _map_face_type: Map FaceLocation to BarrelVaultFaceType.
        _get_crown_facing_location: Pick lateral face closest to the crown.
        _classify_vault_cell: Classify cell as springer/haunch/crown.
        __repr__: Return a string representation.
    """

    CROWN_ZONE_LIMIT = 1.0 / 3.0
    HAUNCH_ZONE_LIMIT = 2.0 / 3.0

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
        """Initialize a barrel-vault hexahedron and classify its vault zone.

        Args:
            node_indices: Ordered cell node indices.
            cell_id: Optional global cell identifier.
            layer: Optional parent layer.
            span_index: Cell index along span.
            thickness_index: Cell index through thickness.
            length_index: Cell index along length (layer index).
            n_span: Number of span cells in mesh.
            n_thickness: Number of thickness cells in mesh.
            n_length: Number of length layers in mesh.

        Returns:
            None.

        Raises:
            None.
        """
        self.span_index = span_index
        self.thickness_index = thickness_index
        self.length_index = length_index
        self.n_span = n_span
        self.n_thickness = n_thickness
        self.n_length = n_length
        self.vault_cell_type: Optional[BarrelVaultCellType] = None
        super().__init__(node_indices, cell_id=cell_id, layer=layer)
        self.vault_cell_type = self._classify_vault_cell()

    def _create_faces(self) -> None:
        """Create typed barrel-vault faces for each local face location.

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
            face_type = self._map_face_type(location)
            self.faces[location] = BarrelVaultFace(
                global_indices,
                location,
                face_type,
                parent_cell=self,
                local_face_index=local_face_index,
            )

    def _map_face_type(self, location: FaceLocation) -> BarrelVaultFaceType:
        """Map local face location to barrel-vault semantic face type.

        Args:
            location: Face location enum.

        Returns:
            BarrelVaultFaceType: Semantic type for the face.

        Raises:
            None.
        """
        if location == FaceLocation.BOTTOM:
            return BarrelVaultFaceType.INTRADOS
        if location == FaceLocation.TOP:
            return BarrelVaultFaceType.EXTRADOS
        if location == FaceLocation.BACK:
            return BarrelVaultFaceType.UP_VAULT
        if location == FaceLocation.FRONT:
            return BarrelVaultFaceType.DOWN_VAULT
        if location in (FaceLocation.LEFT, FaceLocation.RIGHT):
            crown_location = self._get_crown_facing_location()
            if location == crown_location:
                return BarrelVaultFaceType.CROWN_FACING
            return BarrelVaultFaceType.SPRINGING_FACING
        return BarrelVaultFaceType.SPRINGING_FACING

    def _get_crown_facing_location(self) -> FaceLocation:
        """Return lateral face orientation that points toward the crown.

        Args:
            None.

        Returns:
            FaceLocation: ``LEFT`` or ``RIGHT`` crown-facing location.

        Raises:
            None.
        """
        if self.n_span is None or self.span_index is None or self.n_span <= 0:
            return FaceLocation.RIGHT
        crown_index = self.n_span / 2.0
        left_pos = float(self.span_index)
        right_pos = float(self.span_index + 1)
        left_dist = abs(left_pos - crown_index)
        right_dist = abs(right_pos - crown_index)
        if right_dist < left_dist:
            return FaceLocation.RIGHT
        if left_dist < right_dist:
            return FaceLocation.LEFT
        return FaceLocation.RIGHT

    def _classify_vault_cell(self) -> Optional[BarrelVaultCellType]:
        """Classify the cell zone as springer, haunch, or crown.

        Args:
            None.

        Returns:
            Optional[BarrelVaultCellType]: Classified cell type, or ``None``
            when span indexing metadata is unavailable.

        Raises:
            None.
        """
        if self.n_span is None or self.span_index is None or self.n_span <= 0:
            return None
        if self.span_index == 0 or self.span_index == self.n_span - 1:
            return BarrelVaultCellType.SPRINGER

        cell_center = float(self.span_index) + 0.5
        crown_index = self.n_span / 2.0
        denom = self.n_span / 2.0
        if denom <= 0:
            return None
        dist_norm = abs(cell_center - crown_index) / denom
        if dist_norm <= self.CROWN_ZONE_LIMIT:
            return BarrelVaultCellType.CROWN
        return BarrelVaultCellType.HAUNCH

    def __repr__(self) -> str:
        """Return developer-friendly representation of the cell.

        Args:
            None.

        Returns:
            str: String representation with id/type/node ids.

        Raises:
            None.
        """
        cell_type = self.vault_cell_type.value if self.vault_cell_type else None
        return (
            "BarrelVaultHexahedronCell(id="
            f"{self.cell_id}, type={cell_type}, "
            f"nodes={self.node_indices.tolist()})"
        )


class BarrelVaultVolumetricMesh(HexahedronVolumetricMesh):
    """Volumetric hexahedral mesh for a barrel vault geometry.

    Args:
        span: Total vault span (x-direction chord length).
        length: Vault length (y-direction).
        rise: Vertical rise of the vault from springing line to crown.
        thickness: Radial thickness of the vault.
        n_span: Number of cells along the span.
        n_length: Number of layers along the length.
        n_thickness: Number of cells through the thickness.
        generate_interfaces: If True, build interface connectivity.
        dirichlet_boundary_conditions: Optional mapping from cell id or
            BarrelVaultFaceType (or equivalent string) to Dirichlet values.
        neumann_boundary_conditions: Optional mapping from cell id or
            BarrelVaultFaceType (or equivalent string) to Neumann values.

    Notes:
        The mesh uses a circular arc in the x-z plane. Coordinates are
        generated with x across the span, y along the vault length, and z
        as vertical (radial) position.

    Attributes:
        span: Total vault span (x-direction chord length).
        length: Vault length (y-direction).
        rise: Vertical rise from springing line to crown.
        thickness: Radial thickness of the vault.
        n_span: Number of cells along the span.
        n_length: Number of layers along the length.
        n_thickness: Number of cells through the thickness.
        radius_inner: Inner radius of the circular arc.
        half_angle: Half-angle of the arc (radians).
        center_y: Arc center offset in z (used for coordinates).
        dirichlet_boundary_conditions: Mapping of assigned Dirichlet values.
        neumann_boundary_conditions: Mapping of assigned Neumann values.

    Inherited attributes:
        nodes: Node coordinate array of shape (n_nodes, dim).
        layers: List of Layer instances.
        interfaces: List of InteriorFace instances (combined list).
        intra_layer_interfaces: List of interfaces within layers.
        inter_layer_interfaces: List of interfaces between layers.

    Inherited properties:
        n_nodes: Number of mesh nodes.
        n_layers: Number of layers.
        n_cells: Number of cells across all layers.
        Gamma_c_i: Alias for the combined interface list.

    Inherited methods:
        add_layer: Add a layer to the mesh and assign its id.
        add_interface: Add an interface to the combined list.
        add_intra_layer_interface: Add an intra-layer interface.
        add_inter_layer_interface: Add an inter-layer interface.
        get_all_cells: Return a flat list of all cells.
        get_interface_connectivity_table: Build connectivity arrays and normals.

    Methods:
        __init__: Initialize the barrel vault mesh and generate nodes/cells.
        _generate_nodes: Create node coordinates for the vault mesh.
        _node_index: Map (i, j, k) grid indices to a flat node index.
        _generate_layers_and_cells: Build layers and cells with indices.
        compute_birth_times: Compute cell birth times from TCP speed.
        _generate_interfaces: Build inter- and intra-layer interfaces.
        assign_dirichlet_boundary_conditions: Apply Dirichlet boundary values.
        assign_intrados_dirichlet_up_to_layer: Apply Dirichlet BC on intrados
            faces from layer 0 up to a target layer (inclusive).
        assign_neumann_boundary_conditions: Apply Neumann boundary values.
        _assign_boundary_conditions: Internal BC dispatcher.
        _apply_boundary_condition_to_faces: Apply BC to a cell's boundary faces.
        _get_boundary_faces_for_cell: Return boundary faces for a cell.
        __repr__: Return a string representation.
    """

    def __init__(
        self,
        span: float,
        length: float,
        rise: float,
        thickness: float,
        n_span: int,
        n_length: int,
        n_thickness: int,
        generate_interfaces: bool = True,
        dirichlet_boundary_conditions: Optional[
            Dict[Union[int, str, BarrelVaultFaceType], float]
        ] = None,
        neumann_boundary_conditions: Optional[
            Dict[Union[int, str, BarrelVaultFaceType], float]
        ] = None,
    ):
        """Initialize the barrel-vault mesh and generate topology/connectivity.

        Args:
            span: Total vault span.
            length: Vault length.
            rise: Vault rise from springing to crown.
            thickness: Vault thickness.
            n_span: Number of cells along span.
            n_length: Number of cells/layers along length.
            n_thickness: Number of cells through thickness.
            generate_interfaces: Whether to precompute interface connectivity.
            dirichlet_boundary_conditions: Optional Dirichlet BC mapping.
            neumann_boundary_conditions: Optional Neumann BC mapping.

        Returns:
            None.

        Raises:
            None.
        """
        logger.info(
            "[BarrelVaultVolumetricMesh.__init__] Initializing barrel vault mesh..."
        )
        logger.debug(
            "[BarrelVaultVolumetricMesh.__init__] Parameters: span=%s, length=%s, rise=%s, thickness=%s",
            span,
            length,
            rise,
            thickness,
        )
        logger.debug(
            "[BarrelVaultVolumetricMesh.__init__] Discretization: n_span=%s, n_length=%s, n_thickness=%s",
            n_span,
            n_length,
            n_thickness,
        )

        super().__init__()
        self.span, self.length, self.rise, self.thickness = (
            span,
            length,
            rise,
            thickness,
        )
        self.n_span, self.n_length, self.n_thickness = n_span, n_length, n_thickness

        # Compute arc geometry
        half_span = span / 2
        self.radius_inner = (rise**2 + half_span**2) / (2 * rise)
        self.half_angle = np.arcsin(half_span / self.radius_inner)
        self.center_y = self.radius_inner - rise

        logger.debug(
            "[BarrelVaultVolumetricMesh.__init__] Computed geometry: radius_inner=%.4f, half_angle=%.2f°, center_y=%.4f",
            self.radius_inner,
            np.degrees(self.half_angle),
            self.center_y,
        )

        self._generate_nodes()
        self._generate_layers_and_cells()
        if generate_interfaces:
            self._generate_interfaces()

        self.dirichlet_boundary_conditions: Dict[Union[int, str], float] = {}
        self.neumann_boundary_conditions: Dict[Union[int, str], float] = {}
        if dirichlet_boundary_conditions:
            self.assign_dirichlet_boundary_conditions(dirichlet_boundary_conditions)
        if neumann_boundary_conditions:
            self.assign_neumann_boundary_conditions(neumann_boundary_conditions)

        logger.info(
            "[BarrelVaultVolumetricMesh.__init__] Initialization complete: %s nodes, %s layers, %s cells",
            self.n_nodes,
            self.n_layers,
            self.n_cells,
        )
        if generate_interfaces:
            logger.info(
                "[BarrelVaultVolumetricMesh.__init__] Generated %s interfaces (%s intra-layer, %s inter-layer)",
                len(self.interfaces),
                len(self.intra_layer_interfaces),
                len(self.inter_layer_interfaces),
            )

    def _generate_nodes(self) -> None:
        """Generate mesh node coordinates from arc and extrusion parameters.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        logger.debug("[BarrelVaultVolumetricMesh._generate_nodes] Generating nodes...")
        angles = np.linspace(-self.half_angle, self.half_angle, self.n_span + 1)
        y_coords = np.linspace(0, self.length, self.n_length + 1)
        radii = np.linspace(
            self.radius_inner, self.radius_inner + self.thickness, self.n_thickness + 1
        )

        n_i, n_j, n_k = self.n_span + 1, self.n_thickness + 1, self.n_length + 1
        total_nodes = n_k * n_j * n_i
        logger.debug(
            "[BarrelVaultVolumetricMesh._generate_nodes] Creating %s nodes (%s × %s × %s)",
            total_nodes,
            n_i,
            n_j,
            n_k,
        )

        self.nodes = np.zeros((total_nodes, 3))

        idx = 0
        with tqdm(
            total=total_nodes, desc="Generating nodes", unit="node", disable=_TQDM_DISABLE
        ) as pbar:
            for k in range(n_k):
                for j in range(n_j):
                    for i in range(n_i):
                        self.nodes[idx] = [
                            radii[j] * np.sin(angles[i]),
                            y_coords[k],
                            radii[j] * np.cos(angles[i]) - self.center_y,
                        ]
                        idx += 1
                        pbar.update(1)

        logger.debug(
            "[BarrelVaultVolumetricMesh._generate_nodes] Node generation complete: %s nodes created",
            len(self.nodes),
        )

    def _node_index(self, i: int, j: int, k: int) -> int:
        """Map structured indices ``(i, j, k)`` to flattened node index.

        Args:
            i: Span index.
            j: Thickness index.
            k: Length index.

        Returns:
            int: Flattened node index in ``self.nodes``.

        Raises:
            None.
        """
        return (
            k * (self.n_span + 1) * (self.n_thickness + 1) + j * (self.n_span + 1) + i
        )

    def _generate_layers_and_cells(self) -> None:
        """Generate layers and cells, assigning span/thickness/length indices.

        Cell ids are assigned in a serpentine pattern along the span that
        alternates per layer while the internal storage order remains
        consistent for connectivity logic.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        total_cells = self.n_length * self.n_thickness * self.n_span
        logger.debug(
            "[BarrelVaultVolumetricMesh._generate_layers_and_cells] Generating %s layers with %s total cells...",
            self.n_length,
            total_cells,
        )

        cell_id = 0
        with tqdm(
            total=self.n_length, desc="Generating layers", unit="layer", disable=_TQDM_DISABLE
        ) as pbar_layer:
            for k in range(self.n_length):
                layer = Layer()
                # Alternate cell_id ordering along +x / -x per layer while
                # keeping cell storage order consistent for connectivity logic.
                if k % 2 == 0:
                    i_order = range(self.n_span)
                else:
                    i_order = range(self.n_span - 1, -1, -1)
                id_map = {}
                for j in range(self.n_thickness):
                    for i in i_order:
                        id_map[(j, i)] = cell_id
                        cell_id += 1
                for j in range(self.n_thickness):
                    for i in range(self.n_span):
                        node_indices = np.array(
                            [
                                self._node_index(i, j, k),
                                self._node_index(i + 1, j, k),
                                self._node_index(i, j + 1, k),
                                self._node_index(i + 1, j + 1, k),
                                self._node_index(i, j, k + 1),
                                self._node_index(i + 1, j, k + 1),
                                self._node_index(i, j + 1, k + 1),
                                self._node_index(i + 1, j + 1, k + 1),
                            ]
                        )
                        layer.add_cell(
                            BarrelVaultHexahedronCell(
                                node_indices,
                                cell_id=id_map[(j, i)],
                                span_index=i,
                                thickness_index=j,
                                length_index=k,
                                n_span=self.n_span,
                                n_thickness=self.n_thickness,
                                n_length=self.n_length,
                            )
                        )
                self.add_layer(layer)
                pbar_layer.update(1)

        logger.debug(
            "[BarrelVaultVolumetricMesh._generate_layers_and_cells] Layer and cell generation complete: %s layers, %s cells",
            self.n_layers,
            self.n_cells,
        )

    def compute_birth_times(self, tcp_speed: float, mode: str = "layer") -> np.ndarray:
        """Assign birth time to each cell based on a zigzag scan path.

        The zigzag pattern matches the logic in ``main.py``:
            - Layer 0: cells sorted by centroid x (smallest to largest).
            - Layer 1: cells sorted by centroid x (largest to smallest).
            - And so on for subsequent layers.

        Args:
            tcp_speed: Tool center point speed (distance per unit time).
            mode: Unused; retained for backward compatibility.

        Returns:
            Array of birth times indexed by mesh cell order
            (``get_all_cells()`` / DOLFINx cell ordering).

        Raises:
            ValueError: If ``tcp_speed`` is not strictly positive.
        """
        logger.info(
            "[BarrelVaultVolumetricMesh.compute_birth_times] Computing birth times for %s cells...",
            self.n_cells,
        )
        logger.debug(
            "[BarrelVaultVolumetricMesh.compute_birth_times] TCP speed: %s",
            tcp_speed,
        )

        if tcp_speed <= 0:
            raise ValueError("tcp_speed must be positive.")

        nodes = self.nodes
        all_cells = self.get_all_cells()
        cell_index = {id(cell): idx for idx, cell in enumerate(all_cells)}

        birth_times = np.zeros(self.n_cells, dtype=float)
        cumulative_distance = 0.0
        prev_centroid = None

        with tqdm(
            total=self.n_layers, desc="Computing birth times", unit="layer", disable=_TQDM_DISABLE
        ) as pbar:
            for layer_idx, layer in enumerate(self.layers):
                cells_with_centroids = []
                for cell in layer.cells:
                    centroid = cell.compute_centroid(nodes)
                    cells_with_centroids.append((cell, centroid))

                reverse = (layer_idx % 2 == 1)
                cells_with_centroids.sort(key=lambda item: item[1][0], reverse=reverse)

                for cell, centroid in cells_with_centroids:
                    if prev_centroid is not None:
                        cumulative_distance += np.linalg.norm(centroid - prev_centroid)

                    birth_time = cumulative_distance / tcp_speed
                    cell.birth_time = birth_time
                    cell_idx = cell_index.get(id(cell))
                    if cell_idx is not None:
                        birth_times[cell_idx] = birth_time

                    prev_centroid = centroid

                pbar.update(1)

        logger.info(
            "[BarrelVaultVolumetricMesh.compute_birth_times] Birth time computation complete"
        )
        logger.debug(
            "[BarrelVaultVolumetricMesh.compute_birth_times] Total distance traveled: %.4f",
            cumulative_distance,
        )
        logger.debug(
            "[BarrelVaultVolumetricMesh.compute_birth_times] Birth time range: [%.4f, %.4f]",
            np.min(birth_times),
            np.max(birth_times),
        )

        return birth_times

    def _generate_interfaces(self) -> None:
        """Generate inter-layer and intra-layer interface objects.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        logger.info(
            "[BarrelVaultVolumetricMesh._generate_interfaces] Generating interfaces..."
        )

        # Inter-layer interfaces (BACK face of lower layer connects to FRONT face of upper layer)
        inter_layer_count = 0
        logger.debug(
            "[BarrelVaultVolumetricMesh._generate_interfaces] Creating inter-layer interfaces between %s layer pairs...",
            self.n_layers - 1,
        )
        with tqdm(
            total=self.n_layers - 1,
            desc="Inter-layer interfaces",
            unit="layer_pair",
            disable=_TQDM_DISABLE,
        ) as pbar:
            for idx in range(self.n_layers - 1):
                lower_layer, upper_layer = self.layers[idx], self.layers[idx + 1]
                for lower_cell, upper_cell in zip(lower_layer.cells, upper_layer.cells):
                    if lower_cell.layer is upper_cell.layer:
                        iface = IntraLayerInterface.create_between_cells(
                            lower_cell,
                            upper_cell,
                            FaceLocation.BACK,
                            FaceLocation.FRONT,
                        )
                        iface.layer = lower_cell.layer
                        self.add_intra_layer_interface(iface)
                    else:
                        iface = InterLayerInterface(
                            interior_face_id=len(self.inter_layer_interfaces)
                        )
                        iface.add_face(lower_cell.get_face(FaceLocation.BACK))
                        iface.add_face(upper_cell.get_face(FaceLocation.FRONT))
                        iface.lower_layer, iface.upper_layer = lower_layer, upper_layer
                        self.add_inter_layer_interface(iface)
                        inter_layer_count += 1
                pbar.update(1)

        logger.debug(
            "[BarrelVaultVolumetricMesh._generate_interfaces] Created %s inter-layer interfaces",
            inter_layer_count,
        )

        # Intra-layer interfaces
        intra_layer_count = 0
        logger.debug(
            "[BarrelVaultVolumetricMesh._generate_interfaces] Creating intra-layer interfaces..."
        )
        with tqdm(
            total=self.n_layers, desc="Intra-layer interfaces", unit="layer", disable=_TQDM_DISABLE
        ) as pbar:
            for layer in self.layers:
                cells_2d = [
                    layer.cells[j * self.n_span : (j + 1) * self.n_span]
                    for j in range(self.n_thickness)
                ]
                # Right-Left connections
                for j in range(self.n_thickness):
                    for i in range(self.n_span - 1):
                        cell_a = cells_2d[j][i]
                        cell_b = cells_2d[j][i + 1]
                        if cell_a.layer is cell_b.layer:
                            iface = IntraLayerInterface.create_between_cells(
                                cell_a, cell_b, FaceLocation.RIGHT, FaceLocation.LEFT
                            )
                            iface.layer = layer
                            self.add_intra_layer_interface(iface)
                            intra_layer_count += 1
                        else:
                            iface = InterLayerInterface(
                                interior_face_id=len(self.inter_layer_interfaces)
                            )
                            iface.add_face(cell_a.get_face(FaceLocation.RIGHT))
                            iface.add_face(cell_b.get_face(FaceLocation.LEFT))
                            self.add_inter_layer_interface(iface)
                # Top-Bottom connections
                for j in range(self.n_thickness - 1):
                    for i in range(self.n_span):
                        cell_a = cells_2d[j][i]
                        cell_b = cells_2d[j + 1][i]
                        if cell_a.layer is cell_b.layer:
                            iface = IntraLayerInterface.create_between_cells(
                                cell_a, cell_b, FaceLocation.TOP, FaceLocation.BOTTOM
                            )
                            iface.layer = layer
                            self.add_intra_layer_interface(iface)
                            intra_layer_count += 1
                        else:
                            iface = InterLayerInterface(
                                interior_face_id=len(self.inter_layer_interfaces)
                            )
                            iface.add_face(cell_a.get_face(FaceLocation.TOP))
                            iface.add_face(cell_b.get_face(FaceLocation.BOTTOM))
                            self.add_inter_layer_interface(iface)
                pbar.update(1)

        logger.debug(
            "[BarrelVaultVolumetricMesh._generate_interfaces] Created %s intra-layer interfaces",
            intra_layer_count,
        )
        logger.info(
            "[BarrelVaultVolumetricMesh._generate_interfaces] Interface generation complete: %s total interfaces",
            len(self.interfaces),
        )

    def assign_dirichlet_boundary_conditions(
        self,
        boundary_conditions: Dict[Union[int, str, BarrelVaultFaceType], float],
    ) -> None:
        """Assign Dirichlet boundary conditions to boundary faces.

        The mapping keys can be:
            - cell ids (ints)
            - BarrelVaultFaceType values
            - strings matching face types (case-insensitive, hyphens/spaces
              treated as underscores)

        Args:
            boundary_conditions: Mapping from cell id or face type to value.

        Returns:
            None.

        Raises:
            TypeError: If ``boundary_conditions`` is not a dictionary.
        """
        self._assign_boundary_conditions(boundary_conditions, target="dirichlet")

    def assign_intrados_dirichlet_up_to_layer(
        self,
        layer_number: int,
        value: float = 0.0,
    ) -> None:
        """Set Dirichlet BC on intrados faces up to and including a layer index.

        The ``layer_number`` uses the mesh internal layer indexing (0-based).
        For each layer ``k`` with ``0 <= k <= layer_number``, all cells on the
        intrados boundary (``thickness_index == 0``) get a Dirichlet value on
        their BOTTOM face.

        Args:
            layer_number: Maximum layer index to constrain (inclusive).
            value: Dirichlet value to apply (default 0.0).

        Returns:
            None.

        Raises:
            TypeError: If ``layer_number`` is not an integer.
            ValueError: If ``layer_number`` is outside valid mesh range.
        """
        if not isinstance(layer_number, int):
            raise TypeError("layer_number must be an integer (0-based index).")
        if self.n_layers == 0:
            return
        if layer_number < 0:
            raise ValueError("layer_number must be >= 0.")
        if layer_number >= self.n_layers:
            raise ValueError(
                f"layer_number={layer_number} out of range for {self.n_layers} layers "
                f"(valid 0..{self.n_layers - 1})."
            )

        for layer in self.layers:
            if layer.layer_id is None or layer.layer_id > layer_number:
                continue
            for cell in layer.cells:
                if getattr(cell, "thickness_index", None) != 0:
                    continue
                cell.get_face(FaceLocation.BOTTOM).set_dirichlet(value)

        bc_key = f"intrados_upto_layer_{layer_number}"
        self.dirichlet_boundary_conditions[bc_key] = value

    def assign_neumann_boundary_conditions(
        self,
        boundary_conditions: Dict[Union[int, str, BarrelVaultFaceType], float],
    ) -> None:
        """Assign Neumann boundary conditions to boundary faces.

        The mapping keys can be:
            - cell ids (ints)
            - BarrelVaultFaceType values
            - strings matching face types (case-insensitive, hyphens/spaces
              treated as underscores)

        Args:
            boundary_conditions: Mapping from cell id or face type to value.

        Returns:
            None.

        Raises:
            TypeError: If ``boundary_conditions`` is not a dictionary.
        """
        self._assign_boundary_conditions(boundary_conditions, target="neumann")

    def _assign_boundary_conditions(
        self,
        boundary_conditions: Dict[Union[int, str, BarrelVaultFaceType], float],
        target: str,
    ) -> None:
        """Apply boundary conditions to faces by cell id or face type.

        Args:
            boundary_conditions: Mapping from cell id or face type to value.
            target: "dirichlet" or "neumann".

        Returns:
            None.

        Raises:
            TypeError: If ``boundary_conditions`` is not a dictionary.
        """
        if not isinstance(boundary_conditions, dict):
            raise TypeError(
                "boundary_conditions must be a dict like {cell_id: value} or "
                "{face_type: value} (e.g., {'intrados': 0.0})."
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
                    self._apply_boundary_condition_to_faces(cell, target, value)
                else:
                    self.neumann_boundary_conditions[cell_id] = value
                    self._apply_boundary_condition_to_faces(cell, target, value)
            return

        face_type_map: Dict[str, float] = {}
        for key, value in boundary_conditions.items():
            if isinstance(key, BarrelVaultFaceType):
                type_key = key.value
            else:
                type_key = str(key).lower().replace("-", "_").replace(" ", "_")
            face_type_map[type_key] = value

        for cell in all_cells:
            boundary_faces = self._get_boundary_faces_for_cell(cell)
            if not boundary_faces:
                continue
            for face in boundary_faces:
                face_type = getattr(face, "face_type", None)
                if face_type is None:
                    continue
                face_key = (
                    face_type.value if hasattr(face_type, "value") else str(face_type)
                )
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
        cell: BarrelVaultHexahedronCell,
        target: str,
        value: float,
    ) -> None:
        """Apply a boundary condition value to all boundary faces of a cell.

        Args:
            cell: Target cell.
            target: Boundary type (``"dirichlet"`` or ``"neumann"``).
            value: Boundary value.

        Returns:
            None.

        Raises:
            None.
        """
        boundary_faces = self._get_boundary_faces_for_cell(cell)
        for face in boundary_faces:
            if target == "dirichlet":
                face.set_dirichlet(value)
            else:
                face.set_neumann(value)

    def _get_boundary_faces_for_cell(
        self,
        cell: BarrelVaultHexahedronCell,
    ) -> List[Face]:
        """Return boundary faces of a cell from structured index location.

        Args:
            cell: Target cell.

        Returns:
            List[Face]: Boundary faces for the given cell.

        Raises:
            None.
        """
        i = getattr(cell, "span_index", None)
        j = getattr(cell, "thickness_index", None)
        k = getattr(cell, "length_index", None)
        if i is None or j is None or k is None:
            return []
        boundary_faces = []
        if j == 0:
            boundary_faces.append(cell.get_face(FaceLocation.BOTTOM))
        if j == self.n_thickness - 1:
            boundary_faces.append(cell.get_face(FaceLocation.TOP))
        if i == 0:
            boundary_faces.append(cell.get_face(FaceLocation.LEFT))
        if i == self.n_span - 1:
            boundary_faces.append(cell.get_face(FaceLocation.RIGHT))
        if k == 0:
            boundary_faces.append(cell.get_face(FaceLocation.FRONT))
        if k == self.n_length - 1:
            boundary_faces.append(cell.get_face(FaceLocation.BACK))
        return boundary_faces

    def __repr__(self) -> str:
        """Return developer-friendly representation of the volumetric mesh.

        Args:
            None.

        Returns:
            str: String representation with geometric/discretization summary.

        Raises:
            None.
        """
        return (
            f"BarrelVaultVolumetricMesh(span={self.span}, length={self.length}, "
            f"rise={self.rise}, thickness={self.thickness}, "
            f"n_layers={self.n_layers}, n_cells={self.n_cells})"
        )
