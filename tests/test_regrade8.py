"""The eighth skeptic re-grade (6/10, 25.10): red on 0.5.4."""
import os
import sys
import time
import unittest

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import mesh_diet as md  # noqa: E402


class Regrade8(unittest.TestCase):
    def test_a_dense_head_on_a_coarse_capsule_finishes_quickly(self):
        body = trimesh.creation.capsule(height=1, radius=0.25, count=[128, 128])
        head = trimesh.creation.icosphere(subdivisions=6, radius=0.15)       # 81920 tiny triangles
        head.apply_translation([0, 0, 0.9])
        t = time.time()
        res = md.diet(trimesh.util.concatenate([body, head]), faces=8000, up="z")
        self.assertLess(time.time() - t, 90)
        self.assertLess(res.report["shape_error"]["max_percent_of_size"], 5)

    def test_distance_is_exact_against_brute_force(self):
        rng = np.random.default_rng(5)
        body = trimesh.creation.capsule(height=1, radius=0.25, count=[32, 32])
        head = trimesh.creation.icosphere(subdivisions=4, radius=0.15)
        head.apply_translation([0, 0, 0.9])
        m = trimesh.util.concatenate([body, head])
        pts = rng.uniform(-1, 1.5, (300, 3))
        fast = md._to_surface(pts, m)
        tris = m.triangles
        brute = np.array([np.linalg.norm(trimesh.triangles.closest_point(tris, np.repeat(p[None], len(tris), 0)) - p,
                                         axis=1).min() for p in pts])
        self.assertLess(float(np.abs(fast - brute).max()), 1e-9)


class Regrade9(unittest.TestCase):
    """The ninth re-grade (7.5/10): red on 0.5.5."""

    def test_the_library_load_reports_merged_materials_like_the_cli(self):
        import tempfile
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            s = trimesh.creation.icosphere(3)
            s.visual = trimesh.visual.TextureVisuals(uv=np.random.default_rng(0).random((len(s.vertices), 2)),
                                                     image=Image.new("RGB", (8, 8), (200, 0, 0)))
            box = trimesh.creation.box()
            box.apply_translation([3, 0, 0])
            src = os.path.join(d, "s.glb")
            trimesh.Scene([s, box]).export(src)
            mesh, had = md.load(src)
            res = md.diet(mesh, faces=100000, source_had=had)
            self.assertTrue(res.report["texture_lost"])
            self.assertTrue(res.report["materials_merged"])

    def test_a_part_decimated_away_is_counted(self):
        boxes = []
        for k in range(500):
            b = trimesh.creation.box(extents=[0.5] * 3)
            b.apply_translation([(k % 23) * 0.6, 0, (k // 23) * 0.6])
            boxes.append(b)
        res = md.diet(trimesh.util.concatenate(boxes), faces=2500)   # 1000 = 2 faces a box: all flat flaps, a plain error since 0.5.9
        lost = res.report["parts_lost_in_decimation"]
        self.assertEqual(lost, 500 - res.report["after"]["components"])
        if lost:
            self.assertTrue(any("vanished" in w for w in res.report["warnings"]))


if __name__ == "__main__":
    unittest.main()
