# Author: Abdallah Kamhawi <Kamhawi@umich.edu>
# Package Maintainer: Abdallah Kamhawi <Kamhawi@umich.edu>

"""Mesh subpackage for structural 3DCP geometries.

The subpackage exposes core mesh primitives (faces, cells, layers, interfaces)
as well as geometry-specific mesh generation utilities for barrel vaults and
hollow cylinders. Most public classes are re-exported here for convenient
imports.
"""

from .mesh_core import (
    Face,
    FaceLabel,
    FaceLocation,
    HexahedronCell,
    Layer,
    InteriorFace,
    HexahedronVolumetricMesh,
    IntraLayerInterface,
    InterLayerInterface,
    DirichletBoundaryCondition,
    NeumannBoundaryCondition,
)
from .barrel_vault import (
    BarrelVaultCellType,
    BarrelVaultFace,
    BarrelVaultFaceType,
    BarrelVaultHexahedronCell,
    BarrelVaultVolumetricMesh,
)
from .hollow_cylinder import (
    HollowCylinderFaceType,
    HollowCylinderFace,
    HollowCylinderHexahedronCell,
    HollowCylinderVolumetricMesh,
)

from .dolfinx_mapping import (
    compute_cell_permutation,
    reorder_cell_data,
    tag_interfaces,
    tag_boundary_faces,
    build_custom_node_lookup,
)
from .dolfinx_setup import (
    configure_streaming_stdio,
    build_partitioned_dolfinx_mesh,
    tag_interfaces_and_boundaries,
)
from .mesh_quality import evaluate_mesh_quality, MeshQualityReport

__all__ = [
    "Face",
    "FaceLabel",
    "FaceLocation",
    "HexahedronCell",
    "Layer",
    "InteriorFace",
    "HexahedronVolumetricMesh",
    "IntraLayerInterface",
    "InterLayerInterface",
    "DirichletBoundaryCondition",
    "NeumannBoundaryCondition",
    "BarrelVaultVolumetricMesh",
    "BarrelVaultFace",
    "BarrelVaultFaceType",
    "BarrelVaultHexahedronCell",
    "BarrelVaultCellType",
    "HollowCylinderVolumetricMesh",
    "HollowCylinderFace",
    "HollowCylinderFaceType",
    "HollowCylinderHexahedronCell",
    "compute_cell_permutation",
    "reorder_cell_data",
    "tag_interfaces",
    "tag_boundary_faces",
    "build_custom_node_lookup",
    "configure_streaming_stdio",
    "build_partitioned_dolfinx_mesh",
    "tag_interfaces_and_boundaries",
    "evaluate_mesh_quality",
    "MeshQualityReport",
]
