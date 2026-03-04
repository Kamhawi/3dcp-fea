# Author: Abdallah Kamhawi <Kamhawi@umich.edu>
# Package Maintainer: Abdallah Kamhawi <Kamhawi@umich.edu>

"""Utilities for preparing DOLFINx mesh objects and facet tags.

This module isolates orchestration setup steps used by `main.py`:
- stdio streaming setup for MPI runs,
- custom-mesh -> DOLFINx conversion with partitioning,
- interface/boundary facet tagging and interior facet classification.
"""

import sys
import time

import numpy as np
from basix.ufl import element
from dolfinx.mesh import GhostMode, create_cell_partitioner, create_mesh, meshtags

from .dolfinx_mapping import build_custom_node_lookup, tag_boundary_faces, tag_interfaces
from .mesh_core import FaceLabel


def configure_streaming_stdio():
    """Force line-buffered streams for low-latency MPI logging.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(line_buffering=True, write_through=True)
    except Exception:
        pass


def build_partitioned_dolfinx_mesh(vault_mesh, comm):
    """Create a distributed DOLFINx hexahedral mesh from the custom mesh.

    Args:
        vault_mesh: Custom hexahedral mesh object.
        comm: MPI communicator.

    Returns:
        tuple:
            ``(msh, cells_lst, cells_layers)`` where ``cells_layers`` stores
            original layer ids per custom cell.

    Raises:
        None.
    """
    cells_lst = vault_mesh.get_all_cells()

    all_cells = np.ascontiguousarray(
        np.array([cell.node_indices for cell in cells_lst]), dtype=np.int64
    )
    x = np.ascontiguousarray(vault_mesh.nodes, dtype=np.float64)

    n_total_cells = len(all_cells)
    base_chunk = n_total_cells // comm.size
    remainder = n_total_cells % comm.size

    # Deterministic block partitioning before handing cells to DOLFINx
    # partitioner keeps the setup reproducible across runs.
    if comm.rank < remainder:
        start_idx = comm.rank * (base_chunk + 1)
        end_idx = start_idx + base_chunk + 1
    else:
        start_idx = comm.rank * base_chunk + remainder
        end_idx = start_idx + base_chunk

    cells = np.ascontiguousarray(all_cells[start_idx:end_idx, :])
    cells.flags.writeable = False
    x.flags.writeable = False

    cells_layers = np.array([cell.layer.layer_id for cell in cells_lst], dtype=np.float64)
    coord_elem = element("Lagrange", "hexahedron", 1, shape=(3,))

    if comm.rank == 0:
        print("Creating DOLFINx mesh and partitioning across cores...", flush=True)
    t0 = time.time()

    # ``GhostMode.shared_facet`` is required for interior-facet integrals and
    # cohesive/interface terms that couple neighboring partitions.
    part = create_cell_partitioner(GhostMode.shared_facet)
    msh = create_mesh(comm=comm, cells=cells, x=x, e=coord_elem, partitioner=part)

    if comm.rank == 0:
        print(f"  Mesh created and partitioned in {time.time() - t0:.2f}s", flush=True)
        print(f"  Global cells: {msh.topology.index_map(3).size_global}", flush=True)

    return msh, cells_lst, cells_layers


def tag_interfaces_and_boundaries(msh, vault_mesh):
    """Build interior/boundary meshtags from custom interface definitions.

    Tag convention for interior facets:
        0: generic interior
        1: inter-layer cohesive interface
        2: intra-layer bonded interface

    Args:
        msh: Distributed DOLFINx mesh.
        vault_mesh: Custom barrel-vault mesh with interface/boundary labels.

    Returns:
        tuple:
            ``(interlayer_facet_indices, intralayer_facet_indices,
            interior_facet_tags, dirichlet_tags)``.

    Raises:
        None.
    """
    custom_lookup, custom_lookup_collisions = build_custom_node_lookup(vault_mesh, decimals=10)

    interlayer_facet_indices, _ = tag_interfaces(
        msh,
        vault_mesh,
        "inter_layer",
        custom_lookup=custom_lookup,
        custom_lookup_collisions=custom_lookup_collisions,
    )
    intralayer_facet_indices, _ = tag_interfaces(
        msh,
        vault_mesh,
        "intra_layer",
        custom_lookup=custom_lookup,
        custom_lookup_collisions=custom_lookup_collisions,
    )

    fdim = msh.topology.dim - 1
    tdim = msh.topology.dim
    msh.topology.create_entities(fdim)
    msh.topology.create_connectivity(fdim, tdim)
    f_to_c = msh.topology.connectivity(fdim, tdim)
    num_facets_local = msh.topology.index_map(fdim).size_local

    all_interior_facets = np.array(
        [f for f in range(num_facets_local) if len(f_to_c.links(f)) == 2], dtype=np.int32
    )
    all_interior_tags = np.zeros_like(all_interior_facets, dtype=np.int32)

    facet_pos = np.full(num_facets_local, -1, dtype=np.int32)
    facet_pos[all_interior_facets] = np.arange(all_interior_facets.size, dtype=np.int32)

    valid_inter = facet_pos[np.asarray(interlayer_facet_indices, dtype=np.int64)]
    valid_inter = valid_inter[valid_inter >= 0]
    all_interior_tags[valid_inter] = 1

    valid_intra = facet_pos[np.asarray(intralayer_facet_indices, dtype=np.int64)]
    valid_intra = valid_intra[valid_intra >= 0]
    # Preserve cohesive precedence: facets already tagged as inter-layer (1)
    # are not overwritten by intra-layer tag (2).
    free_intra = valid_intra[all_interior_tags[valid_intra] != 1]
    all_interior_tags[free_intra] = 2

    interior_facet_tags = meshtags(msh, fdim, all_interior_facets, all_interior_tags)

    _, dirichlet_tags = tag_boundary_faces(
        msh,
        vault_mesh,
        FaceLabel.DIRICHLET,
        custom_lookup=custom_lookup,
        custom_lookup_collisions=custom_lookup_collisions,
    )

    return interlayer_facet_indices, intralayer_facet_indices, interior_facet_tags, dirichlet_tags
