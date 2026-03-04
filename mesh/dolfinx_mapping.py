# Author: Abdallah Kamhawi <Kamhawi@umich.edu>
# Package Maintainer: Abdallah Kamhawi <Kamhawi@umich.edu>

"""Mapping and facet-tag transfer utilities between custom mesh and DOLFINx.

This module is responsible for:
- building cell permutations between custom mesh ordering and DOLFINx ordering,
- mapping custom interface/boundary definitions to DOLFINx facet ids,
- constructing robust coordinate/node-id lookup keys tolerant to small
  floating-point differences.

Parallel Notes:
    MPI collectives (for example ``allreduce`` and ``gather``) are intentionally
    kept at synchronization points where global tagging diagnostics are needed.
"""

import numpy as np
from dolfinx.mesh import meshtags, exterior_facet_indices
from mpi4py import MPI
import sys

try:
    from dolfinx.mesh import entities_to_geometry as _entities_to_geometry
except Exception:
    _entities_to_geometry = None


def compute_cell_permutation(dolfinx_mesh, custom_mesh):
    """Compute custom-cell permutation aligned with DOLFINx local+ghost ordering.

    Args:
        dolfinx_mesh: Distributed DOLFINx mesh.
        custom_mesh: Source custom mesh object with ``get_all_cells()``.

    Returns:
        np.ndarray: Permutation array ``perm`` where ``perm[i]`` is the custom
        cell index corresponding to DOLFINx cell ``i``.

    Raises:
        None: fatal mismatches are handled via ``comm.Abort(1)``.
    """
    original_cells = custom_mesh.get_all_cells()
    original_nodes = custom_mesh.nodes

    tdim = dolfinx_mesh.topology.dim
    dolfinx_mesh.topology.create_connectivity(tdim, 0)

    # Include ghost cells in the permutation mapping
    map_c = dolfinx_mesh.topology.index_map(tdim)
    num_cells = map_c.size_local + map_c.num_ghosts

    def node_coords_key(coords):
        """Create hashable key from node coordinates."""
        # Reduced decimals from 6 to 4 to tolerate parallel floating-point drift
        rounded = np.round(coords, decimals=4)
        sorted_coords = rounded[np.lexsort(rounded.T[::-1])]
        return tuple(sorted_coords.flatten())

    # Build lookup: node coords key -> original cell index
    original_lookup = {}
    for idx, cell in enumerate(original_cells):
        coords = original_nodes[cell.node_indices]
        key = node_coords_key(coords)
        original_lookup[key] = idx

    # Match DOLFINx cells to original cells
    perm = np.zeros(num_cells, dtype=np.int64)
    unmatched = []

    for i in range(num_cells):
        dolfinx_node_indices = dolfinx_mesh.geometry.dofmap[i]
        coords = dolfinx_mesh.geometry.x[dolfinx_node_indices]
        key = node_coords_key(coords)

        if key in original_lookup:
            perm[i] = original_lookup[key]
        else:
            unmatched.append(i)
            perm[i] = -1

    if unmatched:
        print(f"[Rank {dolfinx_mesh.comm.rank}] ERROR: Could not match {len(unmatched)} DOLFINx cells.", flush=True)
        dolfinx_mesh.comm.Abort(1)

    unique_matches = len(np.unique(perm))
    if unique_matches != num_cells:
        print(f"[Rank {dolfinx_mesh.comm.rank}] ERROR: Permutation not 1-to-1 (Found {unique_matches} unique for {num_cells} cells)", flush=True)
        dolfinx_mesh.comm.Abort(1)

    return perm


def reorder_cell_data(data, perm):
    """Reorder cell-wise data from custom ordering to DOLFINx ordering.

    Args:
        data: Array-like cell data in custom ordering.
        perm: Permutation from ``compute_cell_permutation``.

    Returns:
        np.ndarray: Reordered data in DOLFINx local+ghost ordering.

    Raises:
        None.
    """
    return data[perm]


def _coords_key(coords, decimals=4):
    """Create order-independent hashable key from vertex coordinates.

    Args:
        coords: Coordinate array.
        decimals: Rounding precision used for key stability.

    Returns:
        tuple: Hashable coordinate key.

    Raises:
        None.
    """
    rounded = np.round(np.asarray(coords, dtype=float), decimals=decimals)
    sorted_coords = rounded[np.lexsort(rounded.T[::-1])]
    return tuple(sorted_coords.flatten())


def _node_ids_key(node_ids):
    """Create order-independent hashable key from node ids.

    Args:
        node_ids: Iterable of node ids.

    Returns:
        tuple: Sorted node-id key.

    Raises:
        None.
    """
    nodes = np.asarray(node_ids, dtype=np.int64)
    return tuple(np.sort(nodes).tolist())


def _point_key(point, decimals=10):
    """Build hashable rounded coordinate key for one point.

    Args:
        point: Point coordinates.
        decimals: Rounding precision used for the key.

    Returns:
        tuple: Hashable point key.

    Raises:
        None.
    """
    p = np.round(np.asarray(point, dtype=float), decimals=decimals)
    return tuple(p.tolist())


def _build_custom_node_lookup(custom_mesh, decimals=10):
    """Build map from rounded coordinates to custom node ids.

    Args:
        custom_mesh: Custom mesh with ``nodes`` coordinates.
        decimals: Rounding precision used in key generation.

    Returns:
        tuple[dict, int]:
            ``(lookup, collisions)`` where collisions counts repeated keys.

    Raises:
        None.
    """
    lookup = {}
    collisions = 0
    for node_id, xyz in enumerate(np.asarray(custom_mesh.nodes, dtype=float)):
        key = _point_key(xyz, decimals=decimals)
        prev = lookup.get(key)
        if prev is None:
            lookup[key] = int(node_id)
        elif prev != int(node_id):
            collisions += 1
    return lookup, collisions


def build_custom_node_lookup(custom_mesh, decimals=10):
    """Build reusable custom-node lookup table for tagging workflows.

    Args:
        custom_mesh: Custom mesh with ``nodes`` coordinates.
        decimals: Rounding precision used in key generation.

    Returns:
        tuple[dict, int]: ``(lookup, collisions)``.

    Raises:
        None.
    """
    return _build_custom_node_lookup(custom_mesh, decimals=decimals)


def _build_vertex_to_custom_ids(dolfinx_mesh, custom_lookup, decimals=10):
    """Build topology-vertex to custom-node-id map on a local rank.

    Args:
        dolfinx_mesh: Distributed DOLFINx mesh.
        custom_lookup: Mapping from rounded coordinates to custom node ids.
        decimals: Rounding precision used for coordinate keys.

    Returns:
        tuple[np.ndarray, int, str]:
            ``(v_to_custom, unmapped_count, mapping_method)``.

    Raises:
        None.

    DOLFINx topology vertex ids are not guaranteed to index geometry.x directly,
    so we use entities_to_geometry(0D entities) when available.
    """
    v_map = dolfinx_mesh.topology.index_map(0)
    num_vertices = v_map.size_local + v_map.num_ghosts
    vertex_entities = np.arange(num_vertices, dtype=np.int32)

    mapping_method = "entities_to_geometry"
    geom_dofs = None

    if _entities_to_geometry is not None:
        try:
            tdim = dolfinx_mesh.topology.dim
            # Some builds require both directions to be created beforehand.
            dolfinx_mesh.topology.create_connectivity(tdim, 0)
            dolfinx_mesh.topology.create_connectivity(0, tdim)
            geom_dofs = np.asarray(
                _entities_to_geometry(dolfinx_mesh, 0, vertex_entities, False),
                dtype=np.int64,
            ).reshape(-1)
        except Exception:
            geom_dofs = None

    if geom_dofs is None:
        # Fallback: reconstruct topology-vertex -> geometry-dof map from each cell.
        # This is robust for linear hexahedra and avoids relying on 0->tdim APIs.
        mapping_method = "cell_dofmap"
        tdim = dolfinx_mesh.topology.dim
        dolfinx_mesh.topology.create_connectivity(tdim, 0)
        c_to_v = dolfinx_mesh.topology.connectivity(tdim, 0)
        c_map = dolfinx_mesh.topology.index_map(tdim)
        num_cells = c_map.size_local + c_map.num_ghosts
        geom_dofmap = np.asarray(dolfinx_mesh.geometry.dofmap, dtype=np.int64)

        v_to_g = np.full(num_vertices, -1, dtype=np.int64)
        fallback_conflicts = 0
        for c in range(num_cells):
            verts = np.asarray(c_to_v.links(c), dtype=np.int64)
            gdofs = np.asarray(geom_dofmap[c], dtype=np.int64)
            if verts.size != gdofs.size:
                continue
            for lv in range(verts.size):
                v = int(verts[lv])
                g = int(gdofs[lv])
                prev = int(v_to_g[v])
                if prev < 0:
                    v_to_g[v] = g
                elif prev != g:
                    fallback_conflicts += 1

        if np.any(v_to_g < 0):
            # Last-resort fallback: direct index (can be incorrect on some builds).
            mapping_method = "direct_index_fallback"
            geom_dofs = vertex_entities.astype(np.int64)
        else:
            geom_dofs = v_to_g
            if fallback_conflicts > 0:
                mapping_method = f"{mapping_method}_with_conflicts({fallback_conflicts})"

    coords = dolfinx_mesh.geometry.x[geom_dofs]
    v_to_custom = np.full(num_vertices, -1, dtype=np.int64)
    unmapped = 0
    for v, xyz in enumerate(coords):
        nid = custom_lookup.get(_point_key(xyz, decimals=decimals))
        if nid is None:
            unmapped += 1
        else:
            v_to_custom[v] = int(nid)

    return v_to_custom, int(unmapped), mapping_method


def _sample_keys(keys, n=3):
    """Return up to ``n`` sample keys for debug output.

    Args:
        keys: Iterable/set of keys.
        n: Maximum number of keys to return.

    Returns:
        list: Sample key list.

    Raises:
        None.
    """
    if not keys:
        return []
    out = list(keys)
    out = out[: min(len(out), n)]
    return out


def tag_interfaces(
    dolfinx_mesh,
    custom_mesh,
    interface_type="inter_layer",
    custom_lookup=None,
    custom_lookup_collisions=None,
):
    """Tag interface facets using custom mesh interface definitions.

    Args:
        dolfinx_mesh: Distributed DOLFINx mesh.
        custom_mesh: Custom mesh with interface lists.
        interface_type: Either ``"inter_layer"`` or ``"intra_layer"``.
        custom_lookup: Optional precomputed coordinate-to-node lookup.
        custom_lookup_collisions: Optional collision count for diagnostics.

    Returns:
        tuple[np.ndarray, dolfinx.mesh.MeshTags]:
            Owned facet indices and corresponding tag object.

    Raises:
        ValueError: If ``interface_type`` is unsupported.
        None: fatal mapping failures call ``comm.Abort(1)``.
    """
    if interface_type == "inter_layer":
        interfaces = custom_mesh.inter_layer_interfaces
    elif interface_type == "intra_layer":
        interfaces = custom_mesh.intra_layer_interfaces
    else:
        raise ValueError(f"Unknown interface_type: {interface_type}")

    interface_facet_nodes = [iface.get_shared_nodes() for iface in interfaces]
    interface_keys = set()
    invalid_shared_nodes = 0
    for shared_nodes in interface_facet_nodes:
        if len(shared_nodes) != 4:
            invalid_shared_nodes += 1
            continue
        interface_keys.add(_node_ids_key(shared_nodes))
    if custom_lookup is None:
        custom_lookup, custom_lookup_collisions = _build_custom_node_lookup(
            custom_mesh, decimals=10
        )
    elif custom_lookup_collisions is None:
        custom_lookup_collisions = 0

    fdim = dolfinx_mesh.topology.dim - 1
    dolfinx_mesh.topology.create_entities(fdim)
    dolfinx_mesh.topology.create_connectivity(fdim, 0)

    comm = dolfinx_mesh.comm
    rank = comm.rank

    target_nodes_flat = np.asarray(interface_facet_nodes, dtype=np.int64).reshape(-1)
    if target_nodes_flat.size > 0:
        target_min = int(target_nodes_flat.min())
        target_max = int(target_nodes_flat.max())
    else:
        target_min = -1
        target_max = -1

    facet_map = dolfinx_mesh.topology.index_map(fdim)
    num_facets_owned = facet_map.size_local
    num_facets = num_facets_owned + facet_map.num_ghosts
    f_to_v = dolfinx_mesh.topology.connectivity(fdim, 0)
    v_map = dolfinx_mesh.topology.index_map(0)
    num_vertices = v_map.size_local + v_map.num_ghosts
    v_global = v_map.local_to_global(np.arange(num_vertices, dtype=np.int32))
    local_vmin = int(v_global.min()) if v_global.size > 0 else sys.maxsize
    local_vmax = int(v_global.max()) if v_global.size > 0 else -sys.maxsize
    # Global min/max diagnostics require collectives so rank-0 reports full
    # distributed ranges instead of rank-local slices.
    global_vmin = comm.allreduce(local_vmin, op=MPI.MIN)
    global_vmax = comm.allreduce(local_vmax, op=MPI.MAX)
    v_to_custom, unmapped_vertices_local, vertex_map_method = _build_vertex_to_custom_ids(
        dolfinx_mesh, custom_lookup, decimals=10
    )
    unmapped_vertices_global = comm.allreduce(int(unmapped_vertices_local), op=MPI.SUM)

    facet_lookup = {}
    collisions = {}
    unmapped_facets_local = 0
    for f in range(num_facets):
        verts = f_to_v.links(f)
        mapped_ids = v_to_custom[np.asarray(verts, dtype=np.int64)]
        if np.any(mapped_ids < 0):
            unmapped_facets_local += 1
            continue
        key = _node_ids_key(mapped_ids.astype(np.int64))
        prev = facet_lookup.get(key)
        if prev is None:
            facet_lookup[key] = f
        else:
            collisions.setdefault(key, [prev]).append(f)

    if collisions:
        print(f"[Rank {rank}] ERROR: {interface_type} non-unique facet keys.", flush=True)
        comm.Abort(1)

    local_found_keys = set(facet_lookup.keys()) & interface_keys
    facet_indices_owned = []
    for key in local_found_keys:
        f = facet_lookup[key]
        if f < num_facets_owned:
            facet_indices_owned.append(f)
    facet_indices = np.array(sorted(set(facet_indices_owned)), dtype=np.int32)

    # Collect global counts for consistency checks across partitions.
    global_found = comm.allreduce(int(facet_indices.size), op=MPI.SUM)
    rank_counts = comm.gather(int(facet_indices.size), root=0)
    unmapped_facets_global = comm.allreduce(int(unmapped_facets_local), op=MPI.SUM)
    expected = len(interface_keys)
    if rank == 0:
        if invalid_shared_nodes > 0:
            print(
                f"  WARNING: {interface_type}: skipped {invalid_shared_nodes} interfaces "
                "with shared-node count != 4.",
                flush=True,
            )
        print(
            f"  DEBUG {interface_type}: target facets={expected}, "
            f"custom node-id range=[{target_min}, {target_max}], "
            f"dolfinx vertex global-id range=[{global_vmin}, {global_vmax}]",
            flush=True,
        )
        if rank_counts is not None:
            print(
                f"  DEBUG {interface_type}: owned matches per rank={rank_counts}",
                flush=True,
            )
        print(
            f"  DEBUG {interface_type}: custom lookup collisions={custom_lookup_collisions}, "
            f"unmapped dolfinx facets={unmapped_facets_global}, "
            f"unmapped dolfinx vertices={unmapped_vertices_global}, "
            f"vertex_map={vertex_map_method}",
            flush=True,
        )
        print(
            f"  DEBUG {interface_type}: sample target keys={_sample_keys(interface_keys)}",
            flush=True,
        )
        print(
            f"  DEBUG {interface_type}: sample dolfinx keys={_sample_keys(facet_lookup.keys())}",
            flush=True,
        )

    if global_found != expected and rank == 0:
        print(f"  WARNING: {interface_type}: global facet-key mismatch (expected {expected}, found {global_found}).", flush=True)

    tags = meshtags(
        dolfinx_mesh,
        fdim,
        facet_indices,
        np.ones_like(facet_indices, dtype=np.int32)
    )

    return facet_indices, tags


def tag_boundary_faces(
    dolfinx_mesh,
    custom_mesh,
    label,
    custom_lookup=None,
    custom_lookup_collisions=None,
):
    """Tag exterior facets that match a target custom boundary label.

    Args:
        dolfinx_mesh: Distributed DOLFINx mesh.
        custom_mesh: Custom mesh with boundary-labeled faces.
        label: Target custom face label.
        custom_lookup: Optional precomputed coordinate-to-node lookup.
        custom_lookup_collisions: Optional collision count for diagnostics.

    Returns:
        tuple[np.ndarray, dolfinx.mesh.MeshTags]:
            Owned exterior facet indices and corresponding tag object.

    Raises:
        None: fatal mapping failures call ``comm.Abort(1)``.
    """
    target_keys = set()
    for cell in custom_mesh.get_all_cells():
        for _, face in cell.faces.items():
            if face.label == label:
                target_keys.add(_node_ids_key(face.node_indices))
    if custom_lookup is None:
        custom_lookup, custom_lookup_collisions = _build_custom_node_lookup(
            custom_mesh, decimals=10
        )
    elif custom_lookup_collisions is None:
        custom_lookup_collisions = 0

    if len(target_keys) == 0:
        print(f"ERROR: No boundary faces found in custom mesh for label {label.name}.", flush=True)
        dolfinx_mesh.comm.Abort(1)

    fdim = dolfinx_mesh.topology.dim - 1
    dolfinx_mesh.topology.create_entities(fdim)
    dolfinx_mesh.topology.create_connectivity(fdim, dolfinx_mesh.topology.dim)
    dolfinx_mesh.topology.create_connectivity(fdim, 0)
    ext_facets = exterior_facet_indices(dolfinx_mesh.topology)
    comm = dolfinx_mesh.comm
    rank = comm.rank

    f_to_v = dolfinx_mesh.topology.connectivity(fdim, 0)
    v_map = dolfinx_mesh.topology.index_map(0)
    num_vertices = v_map.size_local + v_map.num_ghosts
    v_global = v_map.local_to_global(np.arange(num_vertices, dtype=np.int32))
    local_vmin = int(v_global.min()) if v_global.size > 0 else sys.maxsize
    local_vmax = int(v_global.max()) if v_global.size > 0 else -sys.maxsize
    # Collectives are required here so printed ranges reflect all MPI ranks.
    global_vmin = comm.allreduce(local_vmin, op=MPI.MIN)
    global_vmax = comm.allreduce(local_vmax, op=MPI.MAX)
    v_to_custom, unmapped_vertices_local, vertex_map_method = _build_vertex_to_custom_ids(
        dolfinx_mesh, custom_lookup, decimals=10
    )
    unmapped_vertices_global = comm.allreduce(int(unmapped_vertices_local), op=MPI.SUM)

    target_nodes = np.asarray(list(target_keys), dtype=np.int64)
    if target_nodes.size > 0:
        target_min = int(target_nodes.min())
        target_max = int(target_nodes.max())
    else:
        target_min = -1
        target_max = -1

    ext_lookup = {}
    collisions = {}
    unmapped_facets_local = 0
    for f in ext_facets:
        verts = f_to_v.links(f)
        mapped_ids = v_to_custom[np.asarray(verts, dtype=np.int64)]
        if np.any(mapped_ids < 0):
            unmapped_facets_local += 1
            continue
        key = _node_ids_key(mapped_ids.astype(np.int64))
        prev = ext_lookup.get(key)
        if prev is None:
            ext_lookup[key] = f
        else:
            collisions.setdefault(key, [prev]).append(f)

    if collisions:
        print(f"[Rank {rank}] ERROR: Boundary tagging for {label.name} has collisions.", flush=True)
        comm.Abort(1)
    
    facet_indices = [
        facet_id for key, facet_id in ext_lookup.items() if key in target_keys
    ]
    facet_indices = np.array(sorted(set(facet_indices)), dtype=np.int32)
    # Global reduction verifies boundary-key matching completeness.
    global_found = comm.allreduce(int(facet_indices.size), op=MPI.SUM)
    rank_counts = comm.gather(int(facet_indices.size), root=0)
    unmapped_facets_global = comm.allreduce(int(unmapped_facets_local), op=MPI.SUM)
    expected = len(target_keys)
    if rank == 0:
        print(
            f"  DEBUG boundary {label.name}: target facets={expected}, "
            f"custom node-id range=[{target_min}, {target_max}], "
            f"dolfinx vertex global-id range=[{global_vmin}, {global_vmax}]",
            flush=True,
        )
        if rank_counts is not None:
            print(
                f"  DEBUG boundary {label.name}: owned matches per rank={rank_counts}",
                flush=True,
            )
        print(
            f"  DEBUG boundary {label.name}: custom lookup collisions={custom_lookup_collisions}, "
            f"unmapped dolfinx facets={unmapped_facets_global}, "
            f"unmapped dolfinx vertices={unmapped_vertices_global}, "
            f"vertex_map={vertex_map_method}",
            flush=True,
        )
        print(
            f"  DEBUG boundary {label.name}: sample target keys={_sample_keys(target_keys)}",
            flush=True,
        )
        print(
            f"  DEBUG boundary {label.name}: sample dolfinx keys={_sample_keys(ext_lookup.keys())}",
            flush=True,
        )

    if global_found != expected and rank == 0:
        print(f"  WARNING: Boundary tagging for {label.name}: global facet-key mismatch (expected {expected}, found {global_found}).", flush=True)

    tags = meshtags(
        dolfinx_mesh,
        fdim,
        facet_indices,
        np.ones_like(facet_indices, dtype=np.int32)
    )

    return facet_indices, tags
