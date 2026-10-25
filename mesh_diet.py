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

__version__ = "0.5.2"


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

    PAIRS = 1_500_000                  # point-triangle pairs per batch: memory stays ~100 MB whatever the shape

    def nearest(pts, ids):
        """min distance from each point to the triangles ids (one row of ids per point), in bounded batches (a
        capsule's long side triangles once asked for 78.7 million pairs at once: MemoryError, regrade 4)."""
        out = np.empty(len(pts))
        step = max(1, PAIRS // max(1, ids.shape[1]))
        for s in range(0, len(pts), step):
            pp, ii = pts[s:s + step], ids[s:s + step]
            p = np.repeat(pp, ii.shape[1], axis=0)
            d = np.linalg.norm(trimesh.triangles.closest_point(tris[ii.ravel()], p) - p, axis=1)
            out[s:s + step] = d.reshape(len(pp), ii.shape[1]).min(axis=1)
        return out

    # the big triangles are measured against every point; the bound below then only has to cover the rest (one big
    # triangle used to make every point "unsure": 19 min of tests). «Big» = the 64 largest and every triangle over
    # 3x the median radius: a capsule has 128 long side triangles, and the 65th kept the bound from ever shrinking.
    order = np.argsort(radius)
    over = np.flatnonzero(radius > 3 * np.median(radius))
    big = np.union1d(order[max(0, len(tris) - 64):], over)
    if len(big) >= len(tris):
        big = order[max(0, len(tris) - 64):]
    big = big[:4096] if len(big) > 4096 else big
    small = np.setdiff1d(np.arange(len(tris)), big)
    best = np.full(len(points), np.inf)
    if len(big):
        best = nearest(points, np.broadcast_to(big, (len(points), len(big))))
    if not len(small):
        return best
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
    pa, fa = trimesh.sample.sample_surface(a, n, seed=1)
    pb, _ = trimesh.sample.sample_surface(b, n, seed=2)
    da = _to_surface(pa, b)
    d = np.concatenate([da, _to_surface(pb, a)])
    size = float(np.linalg.norm(a.extents)) or 1.0
    out = {
        "max_m": round(float(d.max()), 6),
        "mean_m": round(float(d.mean()), 6),
        "max_percent_of_size": round(100 * float(d.max()) / size, 3),
        "worst_part_percent": 0.0,
    }
    # the same error as a share of each loose part's own size: 400 crates each crushed to a bipyramid moved 69% of a
    # crate but 1.25% of the scene, and nothing warned (regrade 5). Small parts get extra samples of their own.
    parts = loose_parts(a)
    if 1 < len(parts) <= MAX_PART_CHECK:
        label = np.empty(len(a.faces), dtype=int)
        for i, p in enumerate(parts):
            label[p] = i
        per = np.bincount(label[fa], minlength=len(parts))
        worst, worst_faces = 0.0, 0
        for i, p in enumerate(parts):
            sub = a.submesh([p], append=True)
            diag = float(np.linalg.norm(sub.extents))
            if diag <= 0:
                continue
            dd = da[label[fa] == i]
            if per[i] < 32:
                extra, _ = trimesh.sample.sample_surface(sub, 32 - per[i], seed=3 + i)
                dd = np.concatenate([dd, _to_surface(extra, b)])
            share = 100 * float(dd.max()) / diag
            if share > worst:
                worst, worst_faces = share, len(p)
        out["worst_part_percent"] = round(worst, 3)
        out["worst_part_faces"] = int(worst_faces)
    return out


# ---------- steps ----------

def clean(m, min_component=0.002):
    """(welded clean mesh, parts removed, % of faces removed, mask of the ORIGINAL faces kept). Welding keeps the
    face order, so the mask carries over: a mesh that needs no decimation is written from the original vertices,
    UV seams intact (regrade 4: a crate's 24 UV vertices became 8 with «texture_lost: false»)."""
    m = _welded(m)
    good = np.zeros(len(m.faces), dtype=bool)
    nd = np.flatnonzero(m.nondegenerate_faces())
    sub = m.submesh([nd], append=True) if len(nd) else m.submesh([[]], append=True)
    uq = nd[sub.unique_faces()] if len(nd) else nd
    good[uq] = True
    m.update_faces(good)
    m.remove_unreferenced_vertices()
    alive = np.flatnonzero(good)                       # m's face i is original face alive[i]
    # the README's «loose part» (shares no vertex): edge-components split a body at a non-manifold edge, and a
    # character's 5 parts were counted as 60, «56 removed, 3 left» (regrade 5)
    parts = loose_parts(m)
    if len(parts) <= 1:
        return m, 0, 0.0, good
    keep = [p for p in parts if len(p) >= min_component * len(m.faces)]
    kept_faces = sum(len(p) for p in keep)
    # «small loose parts» are crumbs only while they are a small share of the model: 1000 equal cubes are each 0.1 %
    # of the faces, and 999 of them were deleted with «shape cost 0 %» (regrade 3). Past half the model, keep all.
    if kept_faces < 0.5 * len(m.faces):
        return m, 0, 0.0, good
    mask = np.zeros(len(m.faces), dtype=bool)
    mask[np.concatenate(keep)] = True
    share = 100.0 * (1 - mask.mean())
    orig = np.zeros_like(good)
    orig[alive[mask]] = True
    m.update_faces(mask)
    m.remove_unreferenced_vertices()
    return m, len(parts) - len(keep), round(share, 3), orig


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


MIN_LOD_FACES = 4
MAX_PART_CHECK = 4096       # loose parts measured one by one in the shape check; past this only the whole is measured
PART_WARN_PERCENT = 10.0    # a part that moved more than this share of its own size is named
OUT_EXT = {".glb", ".gltf", ".obj", ".ply", ".stl", ".off"}


def measured_parts(m, parts, touch=0.02, big=0.25):
    """The face lists of the parts that make up the model for measuring: the largest, every part whose box touches
    the group's (within `touch` of the main body's size, repeated until nothing joins), and any part at least `big`
    of the largest's faces (two equal halves both count)."""
    if len(parts) <= 1:
        return list(parts)
    main = max(range(len(parts)), key=lambda i: len(parts[i]))
    boxes = [m.submesh([p], append=True).bounds for p in parts]
    eps = touch * float(np.max(boxes[main][1] - boxes[main][0]))
    # «a quarter of its size» is its size: the longest side of its box, not its face count — a 12-face 3 m pole next
    # to a 5000-face ball is the bigger thing (regrade 3: the README said size, the code counted faces)
    reach = [float(np.max(b[1] - b[0])) for b in boxes]
    keep = {main} | {i for i in range(len(parts)) if reach[i] >= big * reach[main]}
    grew = True
    while grew:
        grew = False
        for i, b in enumerate(boxes):
            if i in keep:
                continue
            if any(np.all(b[0] <= boxes[j][1] + eps) and np.all(boxes[j][0] - eps <= b[1]) for j in keep):
                keep.add(i)
                grew = True
    return [parts[i] for i in sorted(keep)]


# ---------- the whole diet ----------

def diet(mesh, faces=20000, lods=1, lod_ratio=0.5, height=None, size=None, up="y",
         pivot="bottom", min_component=0.002, source_had=None):
    """source_had: {"texture": bool, "colours": bool} of the FILE, when the caller merged a scene into `mesh` —
    merging parts with different materials drops every visual before diet() could see one (regrade 5)."""
    before = stats(mesh)
    # the simplifier keeps geometry only: say so instead of dropping the texture without a word
    has_texture = getattr(mesh.visual, "kind", None) == "texture" or getattr(mesh.visual, "uv", None) is not None
    # vertex colours are what image-to-3D output carries; they went without a word (regrade 3)
    has_colours = getattr(mesh.visual, "kind", None) == "vertex"
    src = source_had or {}
    had_texture = has_texture or bool(src.get("texture"))
    had_colours = has_colours or bool(src.get("colours"))
    lost_on_load = (had_texture and not has_texture) or (had_colours and not has_colours)
    m, removed, removed_share, keep_orig = clean(mesh, min_component)
    if len(m.faces) == 0:
        raise ValueError("nothing is left after cleaning: every triangle was degenerate (zero area)")
    orient(m, up)
    raw = mesh.copy()                                 # the original vertices, UV seams and colours intact
    raw.update_faces(keep_orig)
    raw.remove_unreferenced_vertices()
    orient(raw, up)
    chain, targets = [], []
    budget = faces
    warnings = []
    for i in range(max(1, lods)):
        t = int(round(budget))
        if i and (t < MIN_LOD_FACES or len(chain[-1].faces) <= MIN_LOD_FACES):
            # a LOD of 0-2 faces is an empty file (regrade 2: LOD7-9 were written with nothing in them)
            warnings.append(f"LODs stopped at LOD{i - 1}: the next would have under {MIN_LOD_FACES} faces "
                            f"(asked for {lods}). Raise --faces or lower --lods.")
            break
        targets.append(t)
        if t >= len(m.faces) and (not chain or len(chain[-1].faces) >= len(m.faces)):
            chain.append(raw.copy())                  # within budget: nothing to simplify, nothing welded away
        else:
            chain.append(decimate(m if not chain or len(chain[-1].faces) >= len(m.faces) else chain[-1], t))
        if i and len(chain[-1].faces) >= len(chain[-2].faces) and t >= len(chain[-2].faces):
            # a budget above the mesh made identical LOD files without a word (regrade 4)
            warnings.append(f"LOD{i} is no smaller than LOD{i - 1} ({len(chain[-1].faces)} faces): the budget is "
                            "above what this mesh has; lower --faces to get real LODs.")
        if len(chain[-1].faces) == 0:            # --faces 1 on a small mesh decimated to nothing (regrade 3)
            if i == 0:
                raise ValueError(f"--faces {faces} leaves no triangles for this mesh; ask for more")
            chain.pop()
            targets.pop()
            break
        budget *= lod_ratio
    decimated = len(chain[0].faces) < len(m.faces)
    # nothing decimated = nothing lost (regrade 3: a false warning) — unless merging the scene already lost it
    texture_lost = had_texture and (decimated or lost_on_load)
    colours_lost = had_colours and (decimated or lost_on_load)
    # measure the size on LOD0, so the mesh people use is exactly as tall as asked. Not on every loose part: a stray
    # shard 2 m above the head shrank a 1.8 m figure to 0.82 m (regrade 1). Not on the main body alone either: a
    # table whose legs are separate pieces came out 11.25 m tall with its legs under the floor (regrade 2). So: the
    # main body and every part that touches it (or touches a part that does), and any part a quarter its size.
    parts = loose_parts(chain[0])
    group = measured_parts(chain[0], parts)
    body = chain[0] if len(group) == len(parts) else chain[0].submesh(group, append=True)
    if height and body.extents[1] <= 0:
        raise ValueError(f"the mesh has no height along the up axis ({up}); use --size, or set --up")
    mat = placement(body, height=height, size=size, pivot=pivot)
    if len(group) < len(parts):
        lo, hi = body.bounds
        out_lo, out_hi = chain[0].bounds
        reach = float(np.max(np.concatenate([lo - out_lo, out_hi - hi])) / max(body.extents.max(), 1e-12))
    else:
        reach = 0.0
    for x in [m] + [c for c in chain if c is not m]:
        x.apply_transform(mat)
    err = shape_error(m, chain[0])
    for i, (x, t) in enumerate(zip(chain, targets)):
        # also when it could not reduce at all (regrade 5: a 256-face cylinder asked for 40 stayed 256, no word)
        if len(x.faces) > 1.05 * t and t < len(m.faces):
            warnings.append(f"LOD{i}: the simplifier stopped at {len(x.faces)} faces, budget {t}. "
                            "Usually fans of very many triangles around one vertex (poles, cones, cylinder caps), "
                            "or a mesh already too coarse to lose more without falling apart.")
    if err.get("worst_part_percent", 0) > PART_WARN_PERCENT:
        warnings.append(f"a separate part ({err['worst_part_faces']} faces) changed shape by up to "
                        f"{err['worst_part_percent']:g}% of its own size, though the whole moved less. Small parts "
                        "(eyes, bolts, crates) lose most; raise --faces or simplify them separately.")
    if texture_lost:
        warnings.append("the input has a texture / UVs; the output is geometry only (the simplifier does not carry "
                        "UVs). Re-unwrap and bake the texture onto LOD0 in your DCC tool, or keep the original "
                        "for close-ups.")
    if colours_lost:
        warnings.append("the input has vertex colours; the output is geometry only (the simplifier drops them). "
                        "Re-project the colours onto LOD0 in your DCC tool, or keep the original for close-ups.")
    if removed:
        warnings.append(f"{removed} small loose part(s) removed, {removed_share:g}% of the faces (--min-component); "
                        "set --min-component 0 to keep every part.")
    if len(group) < len(parts):
        warnings.append(f"{len(parts) - len(group)} of {len(parts)} parts float apart from the model and were left out "
                        "of the size" + (f"; they reach {reach * 100:.0f}% of its size beyond it" if reach > 0.05 else "")
                        + ". If they are stray bits, raise --min-component to drop them.")
    if err["max_percent_of_size"] > 2.0:
        warnings.append(f"shape moved by up to {err['max_percent_of_size']}% of the size; check it by eye "
                        "or raise --faces.")
    report = {
        "before": before,
        "removed_components": removed,
        "removed_faces_percent": removed_share,
        "after": stats(chain[0]),
        "shape_error": err,
        "texture_lost": bool(texture_lost),
        "colours_lost": bool(colours_lost),
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
    ap.add_argument("--report", help="write the full JSON report here (the console shows a summary)")
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

    out = a.output or os.path.splitext(a.input)[0] + ".diet.glb"
    # checked before the work, not after it (regrade 3: -o x.fbx crashed once every LOD was computed)
    if os.path.splitext(out)[1].lower() not in OUT_EXT:
        print(f"mesh-diet: cannot write {os.path.splitext(out)[1] or 'a file with no extension'}; "
              f"use one of {', '.join(sorted(OUT_EXT))}", file=sys.stderr)
        return 2
    try:
        loaded = trimesh.load(a.input)
        geoms = list(loaded.geometry.values()) if isinstance(loaded, trimesh.Scene) else [loaded]
        kinds = [getattr(getattr(g, "visual", None), "kind", None) for g in geoms]
        source_had = {"texture": "texture" in kinds or any(getattr(getattr(g, "visual", None), "uv", None) is not None
                                                           for g in geoms),
                      "colours": "vertex" in kinds}
        mesh = trimesh.load(a.input, force="mesh")
    except Exception as e:  # trimesh raises many kinds on a bad file
        print(f"mesh-diet: cannot read {a.input}: {e}", file=sys.stderr)
        return 2
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        print(f"mesh-diet: no triangles in {a.input}", file=sys.stderr)
        return 2

    try:
        res = diet(mesh, faces=a.faces, lods=a.lods, lod_ratio=a.lod_ratio, height=a.height,
                   size=a.size, up=a.up, pivot=a.pivot, min_component=a.min_component, source_had=source_had)
    except ValueError as e:
        print(f"mesh-diet: {e}", file=sys.stderr)
        return 2
    for p in (out, a.report):                    # a missing folder used to crash after the LODs were written
        if p and os.path.dirname(os.path.abspath(p)):
            try:
                os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
            except OSError as e:
                print(f"mesh-diet: cannot make the folder for {p}: {e.strerror}", file=sys.stderr)
                return 2
    for i, m in enumerate(res.lods):
        p = _lod_path(out, i)
        if p.lower().endswith(".gltf"):
            # one self-contained file per LOD: separate buffers were all named gltf_buffer_0.bin, so every LOD
            # overwrote the last one's and LOD0 no longer loaded (regrade 4)
            from trimesh.exchange.gltf import export_gltf
            files = export_gltf(trimesh.Scene(m), embed_buffers=True)
            with open(p, "wb") as fh:
                fh.write(files["model.gltf"])
        else:
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
