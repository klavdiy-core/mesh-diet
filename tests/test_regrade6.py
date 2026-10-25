"""The sixth skeptic re-grade (7/10, 25.10): each test red on 0.5.2."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import trimesh
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import mesh_diet as md  # noqa: E402


def _tex_sphere(colour):
    s = trimesh.creation.uv_sphere(count=[16, 16])
    s.visual = trimesh.visual.TextureVisuals(uv=np.random.default_rng(0).random((len(s.vertices), 2)),
                                             image=Image.new("RGB", (8, 8), colour))
    return s


def _crates(n, step=1.5):
    side = int(np.ceil(np.sqrt(n)))
    out = []
    for k in range(n):
        b = trimesh.creation.box(extents=[0.5] * 3)
        b.apply_translation([(k % side) * step, 0, (k // side) * step])
        out.append(b)
    return trimesh.util.concatenate(out)


class Regrade6(unittest.TestCase):
    def test_two_textured_objs_in_one_folder_keep_their_own_texture(self):
        with tempfile.TemporaryDirectory() as d:
            for name, colour in (("red", (255, 0, 0)), ("blue", (0, 0, 255))):
                src = os.path.join(d, f"{name}_src.glb")
                _tex_sphere(colour).export(src)
                self.assertEqual(md.main([src, "-o", os.path.join(d, "o", f"{name}.obj"), "--faces", "50000"]), 0)
            red = trimesh.load(os.path.join(d, "o", "red.obj"), force="mesh")
            px = np.array(red.visual.material.image.convert("RGB"))[0, 0]
            self.assertEqual(tuple(px), (255, 0, 0))

    def test_default_cleaning_does_not_delete_a_tree_canopy(self):
        trunk = trimesh.creation.cylinder(0.2, 4, sections=640)          # 2560 faces: leaves are 44% of the model
        leaves = []
        rng = np.random.default_rng(2)
        for _ in range(1000):
            q = trimesh.Trimesh(vertices=[[0, 0, 0], [0.1, 0, 0], [0.1, 0.1, 0], [0, 0.1, 0]], faces=[[0, 1, 2], [0, 2, 3]])
            q.apply_translation(rng.uniform([-1, -1, 1], [1, 1, 2.5]))
            leaves.append(q)
        res = md.diet(trimesh.util.concatenate([trunk] + leaves), faces=50000)
        self.assertEqual(res.report["removed_components"], 0)

    def test_many_parts_are_still_measured_one_by_one(self):
        res = md.diet(_crates(5000), faces=30000)
        self.assertGreater(res.report["shape_error"]["worst_part_percent"], 20)

    def test_up_minus_z_with_a_space(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "c.glb")
            trimesh.creation.cone(1, 2).export(src)
            self.assertEqual(md.main([src, "-o", os.path.join(d, "o.glb"), "--up", "-z"]), 0)


if __name__ == "__main__":
    unittest.main()
