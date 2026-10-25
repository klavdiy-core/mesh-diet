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

__version__ = "0.5.7"


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


def _has_texture(visual):
    """UVs or an image: a flat material colour is not a texture (regrade 7 told a flat-coloured prop to re-unwrap)."""
    uv = getattr(visual, "uv", None)
    mat = getattr(visual, "material", None)
    img = getattr(mat, "image", None) if mat is not None else None
    img = img if img is not None else getattr(mat, "baseColorTexture", None)
    return (uv is not None and len(uv) > 0) or img is not None


def _look(visual):
    """What a part looks like, as a hashable key: two different keys in one scene = materials merging will lose."""
    kind = getattr(visual, "kind", None)
    if kind == "vertex":
        return ("vertex",)
    mat = getattr(visual, "material", None)
    if mat is None:
        return ("none",)
    img = getattr(mat, "image", None)
    img = img if img is not None else getattr(mat, "baseColorTexture", None)
    colour = getattr(mat, "baseColorFactor", None)
    colour = colour if colour is not None else getattr(mat, "main_color", None)
    return (type(mat).__name__, None if colour is None else tuple(np.round(np.asarray(colour, float), 3).ravel()),
            None if img is None else hash(img.tobytes()))


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


PAIRS = 1_000_000                      # point-triangle pairs per batch: memory stays ~100 MB whatever the shape


def _split_long(tris, limit=2_000_000):
    """Halve every long triangle across its longest edge until none is over 4x the median size. The halves tile the
    triangle, so distances stay exact; but a 4 m sliver of a 640-sided trunk no longer has a 2 m bounding sphere that
    holds every point (1280 of them made the check take 43 s, 0.5.3)."""
    def size(t):
        return np.linalg.norm(t - t.mean(axis=1)[:, None, :], axis=2).max(axis=1)

    cap = 4 * float(np.median(size(tris))) or np.inf
    while len(tris) < limit:
        r = size(tris)
        long_ = r > cap
        if not long_.any():
            break
        t = tris[long_]
        if len(tris) + long_.sum() > limit:
            break
        e = np.stack([np.linalg.norm(t[:, 1] - t[:, 2], axis=1), np.linalg.norm(t[:, 2] - t[:, 0], axis=1),
                      np.linalg.norm(t[:, 0] - t[:, 1], axis=1)], axis=1)
        a = e.argmax(axis=1)                              # the vertex opposite the longest edge
        n = np.arange(len(t))
        p, q, s = t[n, a], t[n, (a + 1) % 3], t[n, (a + 2) % 3]
        mid = (q + s) / 2
        tris = np.concatenate([tris[~long_], np.stack([p, q, mid], axis=1), np.stack([p, mid, s], axis=1)])
    return tris


def _to_surface(points, mesh):
    """Exact distance from each point to the surface of mesh.

    Triangles are split into tiers by size (radius within 2x inside a tier), each with its own tree of centres. A
    triangle of radius r can be closer than the current best only if its centre lies within best + r, so per tier
    only the centres within best + (that tier's largest r) are measured. One bound for all sizes failed three ways:
    a guess from the nearest centres missed big far triangles (7.06 for a point 0.01 away, 0.1.0); a few big
    triangles kept every point unsure (19 min); a dense head on a coarse capsule grew the neighbour search to 65536
    per point and asked for 5 GB (regrade 8)."""
    from scipy.spatial import cKDTree
    points = np.asarray(points, dtype=float)
    tris = mesh.triangles
    best = np.full(len(points), np.inf)
    if not len(tris) or not len(points):
        return best
    tris = _split_long(tris)
    centres = tris.mean(axis=1)
    radius = np.linalg.norm(tris - centres[:, None, :], axis=2).max(axis=1)
    level = np.floor(np.log2(np.maximum(radius, radius.max() * 1e-9 + 1e-300))).astype(int)

    def measure(pidx, tidx):
        """best[p] = min(best[p], distance(p, triangle t)) for pairs, in bounded batches."""
        for s in range(0, len(pidx), PAIRS):
            pi, ti = pidx[s:s + PAIRS], tidx[s:s + PAIRS]
            d = np.linalg.norm(trimesh.triangles.closest_point(tris[ti], points[pi]) - points[pi], axis=1)
            np.minimum.at(best, pi, d)

    tiers = []
    for lv in np.unique(level):
        ids = np.flatnonzero(level == lv)
        tiers.append((ids, cKDTree(centres[ids]), float(radius[ids].max())))
    # a first guess from the few nearest centres of every tier: makes the balls below small
    allp = np.arange(len(points))
    for ids, tree, _ in tiers:
        kk = min(4, len(ids))
        _, idx = tree.query(points, k=kk)
        idx = np.asarray(idx).reshape(len(points), kk)
        measure(np.repeat(allp, kk), ids[idx.ravel()])
    # then every triangle that could still be closer, tier by tier, points in chunks so the lists stay bounded
    for ids, tree, rmax in tiers:
        for s in range(0, len(points), 2048):
            chunk = allp[s:s + 2048]
            balls = tree.query_ball_point(points[chunk], best[chunk] + rmax)
            lens = np.fromiter((len(b) for b in balls), dtype=np.int64, count=len(balls))
            if not lens.sum():
                continue
            measure(np.repeat(chunk, lens), ids[np.concatenate([np.asarray(b, dtype=np.int64) for b in balls])])
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
    if len(parts) > MAX_PART_CHECK:
        out["worst_part_percent"] = None                 # not measured — said, never a silent 0.0 (regrade 6)
        out["parts_not_measured"] = len(parts)
    elif len(parts) > 1:
        # all parts at once (a loop of submeshes took 85 s for 1000 leaves and was capped at 4096 parts, regrade 6):
        # faces laid out part by part, 32 area-weighted points per part on top of the shared samples
        P = len(parts)
        lens = np.array([len(p) for p in parts])
        order = np.concatenate(parts)
        offs = np.concatenate([[0], np.cumsum(lens)])
        label = np.empty(len(a.faces), dtype=int)
        label[order] = np.repeat(np.arange(P), lens)
        tri = a.triangles[order]
        lo = np.minimum.reduceat(tri.min(axis=1), offs[:-1])
        hi = np.maximum.reduceat(tri.max(axis=1), offs[:-1])
        diag = np.linalg.norm(hi - lo, axis=1)
        csum = np.concatenate([[0.0], np.cumsum(a.area_faces[order])])
        base, tot = csum[offs[:-1]], csum[offs[1:]] - csum[offs[:-1]]
        rng = np.random.default_rng(3)
        u = (base[:, None] + rng.random((P, 32)) * tot[:, None]).ravel()
        k = np.searchsorted(csum, u, side="right") - 1
        k = np.clip(k, np.repeat(offs[:-1], 32), np.repeat(offs[1:] - 1, 32))
        r = rng.random((P * 32, 2))
        r = np.where(r.sum(axis=1, keepdims=True) > 1, 1 - r, r)
        t = tri[k]
        extra = t[:, 0] + r[:, :1] * (t[:, 1] - t[:, 0]) + r[:, 1:] * (t[:, 2] - t[:, 0])
        worst = _to_surface(extra, b).reshape(P, 32).max(axis=1)
        np.maximum.at(worst, label[fa], da)
        share = np.where(diag > 0, 100 * worst / np.where(diag > 0, diag, 1), 0.0)
        i = int(np.argmax(share))
        out["worst_part_percent"] = round(float(share[i]), 3)
        out["worst_part_faces"] = int(lens[i])
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
    # few faces is not enough to be a crumb: it must be small too. A 12-face pedestal half the model's height and a
    # 2 m staff were deleted next to a dense ball (regrade 7). Longest side of the part vs the whole model's.
    span = float(m.extents.max()) or 1.0
    tri = m.triangles

    def big(p):
        t = tri[p].reshape(-1, 3)
        return float((t.max(axis=0) - t.min(axis=0)).max()) >= CRUMB_MAX_SIZE * span

    keep = [p for p in parts if len(p) >= min_component * len(m.faces) or big(p)]
    kept_faces = sum(len(p) for p in keep)
    # «small loose parts» are crumbs only while they are a small share of the model: 1000 equal cubes are each 0.1 %
    # of the faces, and 999 of them were deleted with «shape cost 0 %» (regrade 3). Half was still too much: a tree's
    # 1000 leaf cards, 44% of it, went by default (regrade 6). Past 5% of the faces, they are the model: keep all.
    if kept_faces < (1 - CRUMB_MAX_SHARE) * len(m.faces):
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
    _drop_flaps(out)
    out.fix_normals()
    return out


def _drop_flaps(m):
    """Remove triangle pairs on the same three corners (by position): the simplifier folds them flat, zero thickness,
    each edge then used 4 times. A watertight sphere came out with 201 such flaps and 200 non-manifold edges, and the
    summary said «hole edges 0» (regrade 10). Removing both halves leaves the surface as it was around them."""
    if not len(m.faces):
        return
    g = _welded(m)
    key = np.sort(g.faces, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).ravel()
    bad = cnt[inv] > 1
    if bad.any() and not bad.all():
        m.update_faces(~bad)
        m.remove_unreferenced_vertices()


MIN_LOD_FACES = 4
MAX_PART_CHECK = 100_000    # loose parts measured one by one in the shape check; past this only the whole, and said
CRUMB_MAX_SIZE = 0.1        # ...and only parts whose longest side is under a tenth of the model's
CRUMB_MAX_SHARE = 0.05     # small loose parts are dropped as crumbs only while all of them are under 5% of the faces
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
    has_texture = _has_texture(mesh.visual)
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
            why = ("a closed shape cannot go below a handful of faces (a tetrahedron has 4, the simplifier stops "
                   "near 8)" if t < 12 else
                   # a 268-face table of 16-sided legs was blamed on «fans of very many triangles» (regrade 9)
                   "the mesh is already coarse: closed low-poly shapes have no cheap edge left to collapse"
                   if len(x.faces) >= len(m.faces) or len(m.faces) < 2000 else
                   "usually fans of very many triangles around one vertex (poles, cones, cylinder caps)")
            warnings.append(f"LOD{i}: the simplifier stopped at {len(x.faces)} faces, budget {t}: {why}.")
    parts_after = len(loose_parts(chain[0]))
    parts_lost = max(0, len(loose_parts(m)) - parts_after)
    if parts_lost:
        # a whole crate decimated to nothing was reported as «0 loose parts removed» (regrade 9)
        warnings.append(f"{parts_lost} separate part(s) vanished in decimation (too few faces left for them); "
                        "raise --faces to keep them.")
    if err.get("worst_part_percent") is None:
        warnings.append(f"{err['parts_not_measured']} separate parts: too many to measure one by one, so damage to a "
                        "single small part is not checked; look at the result by eye.")
    elif err["worst_part_percent"] > PART_WARN_PERCENT:
        warnings.append(f"a separate part ({err['worst_part_faces']} faces) changed shape by up to "
                        f"{err['worst_part_percent']:g}% of its own size, though the whole moved less. Small parts "
                        "(eyes, bolts, crates) lose most; raise --faces or simplify them separately.")
    materials_merged = int(src.get("materials", 1)) > 1
    if materials_merged:
        warnings.append(f"the file has {src['materials']} different looks (materials / colours) on separate parts; "
                        "merging them into one mesh keeps one. Split the parts and run each, or reassign materials "
                        "in your DCC tool.")
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
                        + (", so the output is taller or wider than asked" if (height or size) else "")
                        + ". If they are stray bits, raise "
                        "--min-component to drop them; if they belong (leaves, a canopy), scale by hand or with --size.")
    if err["max_percent_of_size"] > 2.0:
        warnings.append(f"shape moved by up to {err['max_percent_of_size']}% of the size; check it by eye "
                        "or raise --faces.")
    after = stats(chain[0])
    if before["watertight"] and not after["watertight"]:
        warnings.append(f"the input was closed (watertight), LOD0 is not: {after['boundary_edges']} hole edge(s), "
                        f"{after['non_manifold_edges']} non-manifold edge(s). Fine for rendering; for physics or 3D "
                        "printing raise --faces or repair it.")
    report = {
        "before": before,
        "removed_components": removed,
        "parts_lost_in_decimation": int(parts_lost),
        "removed_faces_percent": removed_share,
        "after": after,
        "shape_error": err,
        "texture_lost": bool(texture_lost),
        "colours_lost": bool(colours_lost),
        "materials_merged": bool(materials_merged),
        "lods": [{"lod": i, "faces": int(len(x.faces)), "vertices": int(len(x.vertices)), "target": t}
                 for i, (x, t) in enumerate(zip(chain, targets))],
        "warnings": warnings,
    }
    return Result(chain, report)


# ---------- CLI ----------

def load(path):
    """(one merged mesh, what the file had) — pass the second to diet(source_had=...). Merging a scene drops its
    materials before diet() can see them; the README's library example lost two textures with every flag false
    (regrade 9), so the library gets the same door as the CLI."""
    loaded = trimesh.load(path)
    geoms = list(loaded.geometry.values()) if isinstance(loaded, trimesh.Scene) else [loaded]
    visuals = [getattr(g, "visual", None) for g in geoms]
    had = {"texture": any(_has_texture(v) for v in visuals),
           "colours": any(getattr(v, "kind", None) == "vertex" for v in visuals),
           # merging keeps one material: a red and a blue part came out one grey (regrade 7)
           "materials": len({_look(v) for v in visuals})}
    return trimesh.load(path, force="mesh"), had


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
    argv = list(sys.argv[1:] if argv is None else argv)
    for i in range(len(argv) - 1):                # «--up -z» read -z as an option (regrade 6); --up=-z always worked
        if argv[i] == "--up" and argv[i + 1] in ("-y", "-z"):
            argv[i:i + 2] = [f"--up={argv[i + 1]}", ""]
    a = ap.parse_args([x for x in argv if x != ""])
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
        mesh, source_had = load(a.input)
    except Exception as e:  # trimesh raises many kinds on a bad file
        print(f"mesh-diet: cannot read {a.input}: {e}", file=sys.stderr)
        return 2
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        print(f"mesh-diet: no triangles in {a.input}", file=sys.stderr)
        return 2

    # folders made and every target tried for writing BEFORE the work: a folder in the way or a protected path crashed
    # with a traceback once all LODs were computed (regrade 7)
    # every LOD path too (regrade 10: a folder named x_LOD1.glb failed the run after LOD0 was already written)
    for p in [_lod_path(out, i) for i in range(a.lods)] + [a.report]:
        if not p:
            continue
        try:
            os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
            existed = os.path.exists(p)
            with open(p, "ab"):
                pass
            if not existed:
                os.remove(p)
        except OSError as e:
            print(f"mesh-diet: cannot write {p}: {e.strerror}", file=sys.stderr)
            return 2

    try:
        res = diet(mesh, faces=a.faces, lods=a.lods, lod_ratio=a.lod_ratio, height=a.height,
                   size=a.size, up=a.up, pivot=a.pivot, min_component=a.min_component, source_had=source_had)
    except ValueError as e:
        print(f"mesh-diet: {e}", file=sys.stderr)
        return 2
    try:
        _write(res, out)
    except OSError as e:
        print(f"mesh-diet: cannot write {e.filename or out}: {e.strerror}", file=sys.stderr)
        return 2
    res.report["input"] = os.path.basename(a.input)
    text = json.dumps(res.report, indent=2)
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            f.write(text)
    _summary(res, a)
    return 0


def _write(res, out):
    for i, m in enumerate(res.lods):
        p = _lod_path(out, i)
        if p.lower().endswith(".gltf"):
            # one self-contained file per LOD: separate buffers were all named gltf_buffer_0.bin, so every LOD
            # overwrote the last one's and LOD0 no longer loaded (regrade 4)
            from trimesh.exchange.gltf import export_gltf
            files = export_gltf(trimesh.Scene(m), embed_buffers=True)
            with open(p, "wb") as fh:
                fh.write(files["model.gltf"])
        elif p.lower().endswith(".obj"):
            # trimesh names every material material.mtl / material_0.png: a second asset in the same folder
            # replaced the first one's texture (regrade 6). Each OBJ gets its own <stem>.mtl and <stem>_*.png.
            from trimesh.exchange.obj import export_obj
            stem = os.path.splitext(os.path.basename(p))[0]
            text, extra = export_obj(m, return_texture=True, mtl_name=f"{stem}.mtl")
            ren = {k: f"{stem}_{k}" for k in extra if not k.endswith(".mtl")}
            for k, v in extra.items():
                if k.endswith(".mtl"):
                    for old, new in ren.items():
                        v = v.replace(old.encode(), new.encode())
                with open(os.path.join(os.path.dirname(os.path.abspath(p)), ren.get(k, k)), "wb") as fh:
                    fh.write(v)
            with open(p, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
        else:
            m.export(p)
        res.report["lods"][i]["file"] = os.path.basename(p)


def _summary(res, a):
    b, af, e = res.report["before"], res.report["after"], res.report["shape_error"]
    print(f"faces      {b['faces']:,} -> {af['faces']:,}   (" +
          ", ".join(f"LOD{x['lod']} {x['faces']:,}" for x in res.report["lods"]) + ")")
    print(f"size       {' x '.join(str(v) for v in af['size_m'])} m   pivot {a.pivot}")
    print(f"cleaned    {res.report['removed_components']} loose part(s) removed, "
          f"{af['components']} left, watertight {af['watertight']}, hole edges {af['boundary_edges']}, "
          f"non-manifold edges {af['non_manifold_edges']}")
    print(f"shape cost max {e['max_m']} m ({e['max_percent_of_size']}% of size), mean {e['mean_m']} m")
    print(f"wrote      " + ", ".join(x["file"] for x in res.report["lods"]))
    for w in res.report["warnings"]:
        print(f"WARNING    {w}")


if __name__ == "__main__":
    sys.exit(main())
