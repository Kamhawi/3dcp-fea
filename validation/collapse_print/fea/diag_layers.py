# Author: Abdallah Kamhawi <Kamhawi@umich.edu>
"""Per-layer diagnostic: compression, interface jumps, strain, and load path.

Loads a saved displacement checkpoint and reports, per printed layer, why it
does or does not deform: within-cell compression (thickness change), the DG
displacement jump across each inter-layer interface (interpenetration), the
vertical strain, and the von Mises / mean stress (is the layer actually
carrying load, or starved / locked).

Usage:
    python -m validation.collapse_print.fea.diag_layers <ckpt_dir> [config.yaml]
"""
from __future__ import annotations
import sys
import numpy as np
from validation.collapse_print.fea.setup import (
    build_validation_state,
    LAYER_TOP_Z_DOF_POSITIONS,
)

BOT_Z = np.array([2, 5, 8, 11])      # bottom-sheet z-dofs (local nodes 0,1,2,3)
TOP_Z = LAYER_TOP_Z_DOF_POSITIONS    # top-sheet z-dofs (local nodes 4,5,6,7)


def main():
    ckpt_dir = sys.argv[1]
    cfg = sys.argv[2] if len(sys.argv) > 2 else None
    state = build_validation_state(element="DG", config_path=cfg)
    msh = state.msh
    nuo = msh.topology.index_map(3).size_local
    n_layers = int(state.cfg["geometry"]["n_layers"])
    lid = state.cells_layers_dolfinx[:nuo].astype(int)
    span = state.span_indices_dolfinx[:nuo].astype(int)
    u = np.load(f"{ckpt_dir}/checkpoint_latest.npz")["u"]
    state.u.x.array[: len(u)] = u
    state.u.x.scatter_forward()

    c2d = state.cell_to_dofs[:nuo]
    top = -u[c2d[:, TOP_Z]].reshape(nuo, -1).mean(1)   # settlement of top sheet
    bot = -u[c2d[:, BOT_Z]].reshape(nuo, -1).mean(1)   # settlement of bottom sheet
    h0 = state.h0_mm

    # Recompute total strain (DG0 projection) and stress to see the load path.
    import ufl
    from dolfinx import fem
    from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
    from petsc4py import PETSc
    from solver.kinematics import epsilon

    state.materials.update_properties(float(np.load(f"{ckpt_dir}/checkpoint_latest.npz").get("t", 0)) if False else 146.0)
    dx = ufl.Measure("dx", domain=msh)
    tt = ufl.TrialFunction(state.materials.V_DG0_tensor)
    vv = ufl.TestFunction(state.materials.V_DG0_tensor)
    A = assemble_matrix(fem.form(ufl.inner(tt, vv) * dx)); A.assemble()
    invd = A.getDiagonal(); invd.array[:] = 1.0 / np.maximum(invd.array, 1e-30)
    b = create_vector(fem.form(ufl.inner(epsilon(state.u), vv) * dx))
    with b.localForm() as bl: bl.set(0.0)
    assemble_vector(b, fem.form(ufl.inner(epsilon(state.u), vv) * dx))
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    state.materials.strain.x.petsc_vec.pointwiseMult(invd, b)
    state.materials.strain.x.scatter_forward()
    eps = state.materials.strain.x.array[: nuo * 9].reshape(nuo, 3, 3)
    eps_zz = eps[:, 2, 2]
    E = state.materials.E.x.array[:nuo]

    print(f"checkpoint={ckpt_dir}  n_layers={n_layers}")
    print(f"{'L':>3} {'compress_mm':>11} {'compr_%h0':>9} {'eps_zz_%':>9} "
          f"{'jump_below_mm':>13} {'E_MPa':>8}")
    prev_top = {}
    for L in range(n_layers):
        m = lid == L
        if m.sum() == 0:
            continue
        comp = float(np.median(top[m] - bot[m]))
        ezz = float(np.median(eps_zz[m])) * 100.0
        # interface jump at this layer's BOTTOM: layer-L bottom settlement minus
        # layer-(L-1) top settlement, matched per azimuth. Positive => layer L's
        # bottom sits BELOW the layer below's top => interpenetration.
        if L == 0:
            jb = float(np.median(bot[m]))   # vs the fixed plate (=0)
        else:
            mb = lid == (L - 1)
            # match by azimuth (span index)
            jbs = []
            for i in np.unique(span[m]):
                a = bot[m & (span == i)]
                c = top[mb & (span == i)]
                if a.size and c.size:
                    jbs.append(np.median(a) - np.median(c))
            jb = float(np.median(jbs)) if jbs else float("nan")
        print(f"L{L+1:>2} {comp:11.4f} {100*comp/h0:9.2f} {ezz:9.3f} "
              f"{jb:13.4f} {float(np.median(E[m])):8.4f}")


if __name__ == "__main__":
    main()
