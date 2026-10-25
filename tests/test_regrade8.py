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


if __name__ == "__main__":
    unittest.main()
