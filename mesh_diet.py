#!/usr/bin/env python3
"""mesh-diet: make a raw image-to-3D mesh fit for a game.

Image-to-3D models (Hi3D, TRELLIS, Hunyuan3D, TripoSR ...) hand back meshes with
hundreds of thousands to millions of triangles, floating crumbs, an origin
somewhere in the middle and no idea of size. mesh-diet cleans the crumbs off,
turns it upright, scales it to metres, puts the pivot at its feet, decimates it
to a triangle budget, builds LODs, and tells you how much shape that cost.
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import trimesh

__version__ = "0.3.0"


@dataclass
class Result:
    lods: list    # trimesh.Trimesh, LOD0 first
    report: dict


# ---------- measuring ----------

def components(m):
    """Face index arrays, one per connected part. (trimesh's split() would also patch holes.)"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(m.faces)
    adj = m.face_adjacency
    g = coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(n, n))
    count, label = connected_components(g, directed=False)
    order = np.argsort(label, kind="stable")
    return np.split(order, np.cumsum(np.bincount(label, minlength=count))[:-1])


def _welded(m):
    """A copy with vertices merged by position only: UV and normal seams are not holes."""
    g = m.copy()
    g.merge_vertices(merge_tex=True, merge_norm=True)
    return g


def loose_parts(m):
    """Face index arrays, one per part that shares no vertex position with the rest. Edge-connected pieces (see
    components) also split at non-manifold edges: a decimated box gave 7 of them, single slivers glued to the body
    by a vertex. A loose part is what floats free."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    g = _welded(m)
    if not len(g.faces):
        return []
    e, n = g.edges_unique, len(g.vertices)
    count, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)), directed=False)
    flab = lab[g.faces[:, 0]]
    order = np.argsort(flab, kind="stable")
    return [p for p in np.split(order, np.cumsum(np.bincount(flab, minlength=count))[:-1]) if len(p)]


def stats(m):
    raw, m = m, _welded(m)
    edges = m.edges_sorted
    _, counts = np.unique(edges, axis=0, return_counts=True) if len(edges) else (None, np.array([]))
    return {
        "faces": int(len(raw.faces)),
        "vertices": int(len(raw.vertices)),
        "size_m": [round(float(v), 4) for v in m.extents],
        "watertight": bool(m.is_watertight),
        "components": len(loose_parts(m)),
        "boundary_edges": int((counts == 1).sum()),   # edges of holes
        "non_manifold_edges": int((counts > 2).sum()),
    }


def _to_surface(points, mesh, k=16):
    """Exact distance from each point to the surface of mesh. A first guess uses the k triangles with the nearest
    centres; then every triangle that could still be closer (its centre within guess + its own radius) is measured.
    The guess alone missed big triangles whose centre is far (a decimated mesh has them): it reported 7.06 for a
    point lying 0.01 from the surface."""
    from scipy.spatial import cKDTree
    tris = mesh.triangles
    centres = tris.mean(axis=1)
    radius = np.linalg.norm(tris - centres[:, None, :], axis=2).max(axis=1)

    def nearest(pts, ids):
        """min distance from each point to the triangles ids (one row of ids per point)."""
        p = np.repeat(pts, ids.shape[1], axis=0)
        d = np.linalg.norm(trimesh.triangles.closest_point(tris[ids.ravel()], p) - p, axis=1)
        return d.reshape(len(pts), ids.shape[1]).min(axis=1)

    # the few biggest triangles are measured against every point; the bound below then only has to cover the rest
    # (one big triangle used to make every point "unsure" and the check ran per point in Python: 19 min of tests)
    n_big = min(64, len(tris) - 1)
    big = np.argsort(radius)[len(tris) - n_big:] if n_big > 0 else np.array([], int)
    small = np.setdiff1d(np.arange(len(tris)), big)
    best = np.full(len(points), np.inf)
    for s in range(0, len(points), 4096):
        if len(big):
            best[s:s + 4096] = nearest(points[s:s + 4096], np.tile(big, (len(points[s:s + 4096]), 1)))
    tree, rmax = cKDTree(centres[small]), float(radius[small].max())
    todo = np.arange(len(points))
    while len(todo):
        kk = min(k, len(small))
        dist, idx = tree.query(points[todo], k=kk)
        dist, idx = dist.reshape(len(todo), kk), idx.reshape(len(todo), kk)
        best[todo] = np.minimum(best[todo], nearest(points[todo], small[idx]))
        if kk == len(small):
            break                                        # every small triangle was measured: exact
        # a small triangle not among these kk is at least (kk-th centre distance - the biggest small radius) away
        todo = todo[dist[:, -1] - rmax < best[todo]]
        k *= 4
    return best


def shape_error(a, b, n=20000):
    """Symmetric sampled point-to-surface distance between a and b, in metres and in % of a's size."""
    pa, _ = trimesh.sample.sample_surface(a, n, seed=1)
    pb, _ = trimesh.sample.sample_surface(b, n, seed=2)
    d = np.concatenate([_to_surface(pa, b), _to_surface(pb, a)])
    size = float(np.linalg.norm(a.extents)) or 1.0
    return {
        "max_m": round(float(d.max()), 6),
        "mean_m": round(float(d.mean()), 6),
        "max_percent_of_size": round(100 * float(d.max()) / size, 3),
    }


# ---------- steps ----------

def clean(m, min_component=0.002):
    m = _welded(m)
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    parts = components(m)
    if len(parts) <= 1:
        return m, 0
    keep = [p for p in parts if len(p) >= min_component * len(m.faces)]
    if not keep:
        keep = [max(parts, key=len)]
    mask = np.zeros(len(m.faces), dtype=bool)
    mask[np.concatenate(keep)] = True
    m.update_faces(mask)
    m.remove_unreferenced_vertices()
    return m, len(parts) - len(keep)


def orient(m, up="y"):
    """Turn the mesh so its up axis is +Y (the glTF convention)."""
    if up == "z":
        m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    elif up == "-z":
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    elif up == "-y":
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    return m


def placement(m, height=None, size=None, pivot="bottom"):
    """4x4 matrix: scale to metres, then move the pivot to the origin."""
    k = 1.0
    if height:
        k = height / m.extents[1]
    elif size:
        k = size / m.extents.max()
    lo, hi = m.bounds * k
    t = np.zeros(3)
    if pivot == "bottom":
        t = [-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2]
    elif pivot == "center":
        t = -(lo + hi) / 2
    mat = np.eye(4)
    mat[:3, :3] *= k
    mat[:3, 3] = t
    return mat


def decimate(m, faces, tolerance=0.03, tries=6):
    """Quadric decimation to about `faces` triangles (never adds any)."""
    import fast_simplification
    n = len(m.faces)
    if faces >= n:
        return m.copy()
    keep = faces / n
    best = None
    for _ in range(tries):
        v, f = fast_simplification.simplify(
            np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32),
            target_reduction=float(min(max(1 - keep, 0.0), 0.9999)))
        got = len(f)
        if best is None or abs(got - faces) < abs(len(best[1]) - faces):
            best = (v, f)
        if abs(got - faces) <= tolerance * faces or got == 0:
            break
        keep *= faces / max(got, 1)   # the simplifier misses its target; correct and retry
    out = trimesh.Trimesh(best[0].astype(np.float64), best[1], process=True)
    out.fix_normals()
    return out


# ---------- the whole diet ----------

def diet(mesh, faces=20000, lods=1, lod_ratio=0.5, height=None, size=None, up="y",
         pivot="bottom", min_component=0.002):
    before = stats(mesh)
    # the simplifier keeps geometry only: say so instead of dropping the texture without a word
    texture_lost = getattr(mesh.visual, "kind", None) == "texture" or getattr(mesh.visual, "uv", None) is not None
    m, removed = clean(mesh, min_component)
    orient(m, up)
    chain, targets = [], []
    budget = faces
    for i in range(max(1, lods)):
        targets.append(int(round(budget)))
        chain.append(decimate(chain[-1] if chain else m, targets[-1]))
        budget *= lod_ratio
    # measure the size on LOD0, so the mesh people use is exactly as tall as asked - on its main body: a kept loose
    # part 2 m above the head used to shrink a 1.8 m figure to 0.82 m without a word (25.10 regrade)
    parts = loose_parts(chain[0])
    body = chain[0] if len(parts) <= 1 else chain[0].submesh([max(parts, key=len)], append=True)
    if height and body.extents[1] <= 0:
        raise ValueError(f"the mesh has no height along the up axis ({up}); use --size, or set --up")
    mat = placement(body, height=height, size=size, pivot=pivot)
    if len(parts) > 1:
        lo, hi = body.bounds
        out_lo, out_hi = chain[0].bounds
        reach = float(np.max(np.concatenate([lo - out_lo, out_hi - hi])) / max(body.extents.max(), 1e-12))
    else:
        reach = 0.0
    for x in [m] + chain:
        x.apply_transform(mat)
    err = shape_error(m, chain[0])
    warnings = []
    for i, (x, t) in enumerate(zip(chain, targets)):
        if len(x.faces) > 1.05 * t and len(x.faces) < len(m.faces):
            warnings.append(f"LOD{i}: the simplifier stopped at {len(x.faces)} faces, budget {t}. "
                            "Usually fans of very many triangles around one vertex (poles, cones).")
    if texture_lost:
        warnings.append("the input has a texture / UVs; the output is geometry only (the simplifier does not carry "
                        "UVs). Re-unwrap and bake the texture onto LOD0 in your DCC tool, or keep the original "
                        "for close-ups.")
    if len(parts) > 1:
        warnings.append(f"{len(parts)} parts left; the size was measured on the main body"
                        + (f", and the others reach {reach * 100:.0f}% of its size beyond it" if reach > 0.05 else "")
                        + ". Raise --min-component (e.g. 0.05) to drop small loose parts.")
    if err["max_percent_of_size"] > 2.0:
        warnings.append(f"shape moved by up to {err['max_percent_of_size']}% of the size; check it by eye "
                        "or raise --faces.")
    report = {
        "before": before,
        "removed_components": removed,
        "after": stats(chain[0]),
        "shape_error": err,
        "texture_lost": bool(texture_lost),
        "lods": [{"lod": i, "faces": int(len(x.faces)), "vertices": int(len(x.vertices)), "target": t}
                 for i, (x, t) in enumerate(zip(chain, targets))],
        "warnings": warnings,
    }
    return Result(chain, report)


# ---------- CLI ----------

def _lod_path(out, i):
    if i == 0:
        return out
    stem, ext = os.path.splitext(out)
    return f"{stem}_LOD{i}{ext}"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mesh-diet", description=__doc__.splitlines()[0])
    ap.add_argument("input", help="mesh file: .glb .gltf .obj .ply .stl .off ...")
    ap.add_argument("-o", "--output", help="output (default: <input>.diet.glb); LODs go next to it as _LOD1, _LOD2 ...")
    ap.add_argument("-f", "--faces", type=int, default=20000, help="triangle budget for LOD0 (default 20000)")
    ap.add_argument("--lods", type=int, default=1, help="how many LODs, LOD0 included (default 1)")
    ap.add_argument("--lod-ratio", type=float, default=0.5, help="each LOD keeps this share of the previous (default 0.5)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--height", type=float, help="scale so the mesh is this tall, in metres")
    g.add_argument("--size", type=float, help="scale so the longest side is this long, in metres")
    ap.add_argument("--up", choices=["y", "-y", "z", "-z"], default="y", help="the input's up axis (default y)")
    ap.add_argument("--pivot", choices=["bottom", "center", "none"], default="bottom",
                    help="where the origin goes (default bottom: centred, standing on y=0)")
    ap.add_argument("--min-component", type=float, default=0.002,
                    help="drop loose parts with fewer than this share of all faces (default 0.002)")
    ap.add_argument("--report", help="write the JSON report here (it is also printed)")
    ap.add_argument("--version", action="version", version=__version__)
    a = ap.parse_args(argv)
    for flag, ok, need in (("--faces", a.faces >= 1, "at least 1"), ("--lods", a.lods >= 1, "at least 1"),
                           ("--lod-ratio", 0 < a.lod_ratio < 1, "between 0 and 1"),
                           ("--height", a.height is None or a.height > 0, "above 0"),
                           ("--size", a.size is None or a.size > 0, "above 0"),
                           ("--min-component", 0 <= a.min_component < 1, "from 0 to below 1")):
        if not ok:
            print(f"mesh-diet: {flag} must be {need}", file=sys.stderr)
            return 2

    try:
        mesh = trimesh.load(a.input, force="mesh")
    except Exception as e:  # trimesh raises many kinds on a bad file
        print(f"mesh-diet: cannot read {a.input}: {e}", file=sys.stderr)
        return 2
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        print(f"mesh-diet: no triangles in {a.input}", file=sys.stderr)
        return 2

    try:
        res = diet(mesh, faces=a.faces, lods=a.lods, lod_ratio=a.lod_ratio, height=a.height,
                   size=a.size, up=a.up, pivot=a.pivot, min_component=a.min_component)
    except ValueError as e:
        print(f"mesh-diet: {e}", file=sys.stderr)
        return 2
    out = a.output or os.path.splitext(a.input)[0] + ".diet.glb"
    for i, m in enumerate(res.lods):
        p = _lod_path(out, i)
        m.export(p)
        res.report["lods"][i]["file"] = os.path.basename(p)
    res.report["input"] = os.path.basename(a.input)
    text = json.dumps(res.report, indent=2)
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            f.write(text)
    b, af, e = res.report["before"], res.report["after"], res.report["shape_error"]
    print(f"faces      {b['faces']:,} -> {af['faces']:,}   (" +
          ", ".join(f"LOD{x['lod']} {x['faces']:,}" for x in res.report["lods"]) + ")")
    print(f"size       {' x '.join(str(v) for v in af['size_m'])} m   pivot {a.pivot}")
    print(f"cleaned    {res.report['removed_components']} loose part(s) removed, "
          f"{af['components']} left, watertight {af['watertight']}, hole edges {af['boundary_edges']}")
    print(f"shape cost max {e['max_m']} m ({e['max_percent_of_size']}% of size), mean {e['mean_m']} m")
    print(f"wrote      " + ", ".join(x["file"] for x in res.report["lods"]))
    for w in res.report["warnings"]:
        print(f"WARNING    {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
