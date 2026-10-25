"""Before / after gif for the README: a raw image-to-3D-like blob (dense, crumbs around it, off-centre, no size)
next to what mesh-diet makes of it (a budget, crumbs gone, 1.8 m tall, feet on the floor). Made from our own
generated mesh, no outside model.

    python examples/make_demo_gif.py [out.gif]
"""
import os
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import mesh_diet  # noqa: E402

W, H = 360, 400
BG, FLOOR, INK, DIM = (24, 26, 32), (60, 64, 74), (236, 236, 240), (150, 154, 166)


def blob():
    rng = np.random.default_rng(7)
    m = trimesh.creation.icosphere(subdivisions=6)                      # 81,920 faces
    v = m.vertices
    bumps = 1 + 0.10 * np.sin(5 * v[:, 0]) * np.cos(4 * v[:, 1]) + 0.05 * np.sin(9 * v[:, 2])
    m.vertices = v * bumps[:, None] * [1.0, 1.35, 0.9]                   # a figure-ish egg
    parts = [m]
    for _ in range(30):                                                  # crumbs, as image-to-3D leaves them
        c = trimesh.creation.icosphere(subdivisions=1, radius=rng.uniform(0.02, 0.06))
        c.apply_translation(rng.normal(0, 1.1, 3) + [0, 0.3, 0])
        parts.append(c)
    out = trimesh.util.concatenate(parts)
    out.apply_translation([1.7, -0.6, 0.9])                              # off-centre
    out.apply_scale(2.3)                                                 # no real size
    return out


def draw(mesh, yaw, wire, label, sub):
    """Flat-shaded, painter's order. Each panel fits its own model; the floor (y = 0) and the origin are drawn where
    they really are, so «floating off-centre» and «feet on the floor» can be seen."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    R = trimesh.transformations.rotation_matrix(np.radians(14), [1, 0, 0])[:3, :3] @ \
        trimesh.transformations.rotation_matrix(np.radians(yaw), [0, 1, 0])[:3, :3]
    lo, hi = mesh.bounds
    lo, hi = np.minimum(lo, 0), np.maximum(hi, 0)                        # keep the origin in view
    span = max(hi[1] - lo[1], np.linalg.norm([hi[0] - lo[0], hi[2] - lo[2]]))
    scale = (H * 0.66) / span
    top = 80
    cy = top + (hi[1] - 0) * scale                                       # screen row of y = 0
    floor_y = cy
    d.line([(20, floor_y), (W - 20, floor_y)], fill=FLOOR, width=2)
    d.line([(W / 2 - 6, floor_y), (W / 2 + 6, floor_y)], fill=(230, 120, 90), width=3)   # the origin
    d.line([(W / 2, floor_y - 6), (W / 2, floor_y + 6)], fill=(230, 120, 90), width=3)
    v = mesh.vertices @ R.T
    tri = v[mesh.faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.array([-0.4, 0.7, 0.6]) / np.linalg.norm([-0.4, 0.7, 0.6])
    shade = 0.25 + 0.75 * np.clip(n @ light, 0, 1)
    vis = n[:, 2] > 0
    order = np.argsort(tri[:, :, 2].mean(1))
    for i in order:
        if not vis[i]:
            continue
        g = shade[i]
        col = (int(90 + 140 * g), int(150 + 80 * g), int(200 + 40 * g))
        pts = [(W / 2 + x * scale, floor_y - y * scale) for x, y, _ in tri[i]]
        d.polygon(pts, fill=col, outline=(70, 110, 150) if wire else None)
    f = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 18) if os.name == "nt" else ImageFont.load_default()
    fs = ImageFont.truetype(r"C:\Windows\Fonts\segoeui.ttf", 14) if os.name == "nt" else ImageFont.load_default()
    d.text((16, 12), label, font=f, fill=INK)
    for k, line in enumerate(sub):
        d.text((16, 38 + 18 * k), line, font=fs, fill=DIM)
    return img


def main(out):
    raw = blob()
    res = mesh_diet.diet(raw, faces=2000, height=1.8)
    lod0 = res.lods[0]
    before_sub = [f"{len(raw.faces):,} faces, {res.report['removed_components']} crumbs",
                  f"{raw.extents[1]:.2f} m tall, floating off-centre"]
    after_sub = [f"{len(lod0.faces):,} faces, crumbs gone",
                 f"{lod0.extents[1]:.2f} m tall, feet on the floor"]
    frames = []
    for k in range(24):
        yaw = k * 15
        left = draw(raw, yaw, False, "before", before_sub + ["+ = origin, line = floor"])
        right = draw(lod0, yaw, True, "after mesh-diet", after_sub)
        frame = Image.new("RGB", (W * 2 + 8, H), (12, 13, 16))
        frame.paste(left, (0, 0))
        frame.paste(right, (W + 8, 0))
        frames.append(frame.convert("P", palette=Image.ADAPTIVE, colors=64))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=110, loop=0, optimize=True)
    print("wrote", out, os.path.getsize(out) // 1024, "KB;", res.report["shape_error"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "demo.gif"))
