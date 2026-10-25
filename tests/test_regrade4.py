"""The fourth skeptic re-grade (6/10, 25.10): each test red on 0.5.0."""
import os
import sys
import tempfile
import time
import unittest

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import mesh_diet as md  # noqa: E402


class Regrade4(unittest.TestCase):
    def test_a_capsule_does_not_run_out_of_memory(self):
        m = trimesh.creation.capsule(height=1.4, radius=0.25, count=[64, 64])      # 128 long side triangles
        t = time.time()
        res = md.diet(m, faces=4000)
        self.assertLess(time.time() - t, 120)
        self.assertLess(res.report["shape_error"]["max_percent_of_size"], 5)

    def test_gltf_lods_each_load_on_their_own(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "s.glb")
            trimesh.creation.icosphere(subdivisions=4).export(src)
            out = os.path.join(d, "o", "m.gltf")
            self.assertEqual(md.main([src, "-o", out, "--faces", "1500", "--lods", "3"]), 0)
            faces = [len(trimesh.load(md._lod_path(out, i), force="mesh").faces) for i in range(3)]
            self.assertEqual(faces, sorted(faces, reverse=True))
            self.assertGreater(faces[0], faces[2])

    def test_a_textured_mesh_within_budget_keeps_its_uv_seams(self):
        m = trimesh.creation.box()
        m = trimesh.Trimesh(vertices=m.vertices[m.faces].reshape(-1, 3), faces=np.arange(36).reshape(-1, 3), process=False)
        m.visual = trimesh.visual.TextureVisuals(uv=np.random.default_rng(0).random((36, 2)))
        res = md.diet(m, faces=20000)
        self.assertFalse(res.report["texture_lost"])
        self.assertEqual(len(res.lods[0].visual.uv), 36)                       # not welded to 8

    def test_lods_that_do_not_shrink_are_said(self):
        res = md.diet(trimesh.creation.icosphere(subdivisions=3), faces=10 ** 7, lods=3)
        self.assertTrue(any("no smaller than" in w for w in res.report["warnings"]))


if __name__ == "__main__":
    unittest.main()
