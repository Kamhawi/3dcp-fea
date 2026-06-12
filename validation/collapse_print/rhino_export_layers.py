"""RhinoPython exporter: non-planar cylinder layer curves -> conformal-ready JSON.

Run inside Rhino (Tools > PythonScript > Run, or the Rhino 8 Script Editor).
Select the per-layer toolpath curves (one closed curve per deposited layer; order
does not matter, they are sorted bottom-to-top by height) and run.

Each curve is resampled at the SAME N azimuths about the cylinder axis, so vertex
i lies at the same angular position on every layer. That common-seam, equal-count
sampling is what lets the FEA build a structured hex mesh whose inter-layer facets
are conformal (top ring of layer k == bottom ring of layer k+1), which the
cohesive interface law requires. Output schema (consumed by the mesh generator):

  { "units": "...", "represents": "centerline", "wall_thickness": <mm>,
    "n_span": N, "axis_xy": [cx, cy],
    "layers": [ {"k": 0, "points": [[x,y,z], ...]}, ... ] }   # bottom -> top

Assumptions: the cylinder axis is roughly global +Z, and each ring is star-shaped
about that axis (one curve point per azimuth) -- true for a cylinder whose
non-planarity is in z, not in plan. If your rings are inner+outer pairs per layer
rather than centerlines, export one set and tell me; or set REPRESENTS below.
"""

import rhinoscriptsyntax as rs
import math
import json

# ----------------------------- settings -----------------------------
N = 96                 # circumferential samples per layer (mesh n_span)
WALL_THICKNESS = 20.0  # bead width [doc units]; used if these are centerlines
REPRESENTS = "centerline"   # "centerline" or "interface_boundary"
DENSE = 3000           # dense samples per curve used to locate the azimuths
# --------------------------------------------------------------------


def mean_z(cid):
    ps = rs.DivideCurve(cid, 100)
    return sum(p[2] for p in ps) / len(ps)


def main():
    crv_ids = rs.GetObjects("Select the layer curves (any order)",
                            rs.filter.curve, preselect=True)
    if not crv_ids:
        print("No curves selected.")
        return

    crv_ids = sorted(crv_ids, key=mean_z)          # bottom -> top

    # axis center = centroid of all curve points (axis assumed ~ global Z)
    allpts = []
    for cid in crv_ids:
        allpts += rs.DivideCurve(cid, 200)
    cx = sum(p[0] for p in allpts) / len(allpts)
    cy = sum(p[1] for p in allpts) / len(allpts)

    def circdist(a, b):                            # signed-free circular distance
        return abs(((a - b + math.pi) % (2.0 * math.pi)) - math.pi)

    layers = []
    for k, cid in enumerate(crv_ids):
        dense = rs.DivideCurve(cid, DENSE)
        az = [math.atan2(p[1] - cy, p[0] - cx) for p in dense]
        pts = []
        for n in range(N):
            target = -math.pi + 2.0 * math.pi * n / N
            best = min(range(len(dense)), key=lambda i: circdist(az[i], target))
            p = dense[best]
            pts.append([round(p[0], 5), round(p[1], 5), round(p[2], 5)])
        layers.append({"k": k, "points": pts})

    data = {
        "units": rs.UnitSystemName(),
        "represents": REPRESENTS,
        "wall_thickness": WALL_THICKNESS,
        "n_span": N,
        "axis_xy": [round(cx, 5), round(cy, 5)],
        "layers": layers,
    }

    path = rs.SaveFileName("Save layer curves", "JSON (*.json)|*.json||")
    if not path:
        print("Cancelled.")
        return
    with open(path, "w") as f:
        json.dump(data, f)
    print("Wrote %d layers x %d points/layer -> %s" % (len(layers), N, path))


main()
