"""The twelfth skeptic re-grade (6/10, 25.10): red on 0.5.8."""
import os
import sys
import subprocess
import tempfile
import time
import unittest

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
import mesh_diet as md  # noqa: E402


class Regrade12(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt" and os.path.isdir(r"C:\Windows\System32"), "Windows only")
    def test_a_protected_folder_is_a_plain_error_not_a_hang(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "s.glb")
            trimesh.creation.box().export(src)
            r = subprocess.run([sys.executable, os.path.join(REPO, "mesh_diet.py"), src, "-o",
                                r"C:\Windows\System32\mesh_diet_test_x.glb"], capture_output=True, text=True,
                               timeout=60)
            self.assertEqual(r.returncode, 2, r.stderr)

    def test_thousands_of_floating_leaves_are_measured_quickly(self):
        trunk = trimesh.creation.cylinder(0.2, 4, sections=64)
        rng = np.random.default_rng(2)
        # a dense canopy: cards overlap each other, so the touching group grows wave after wave
        v = rng.uniform([-0.6, 1.6, -0.6], [0.6, 2.8, 0.6], (8000, 1, 3)) + \
            np.array([[0, 0, 0], [0.12, 0, 0], [0, 0.12, 0]])
        leaves = trimesh.Trimesh(v.reshape(-1, 3), np.arange(24000).reshape(-1, 3), process=False)
        m = trimesh.util.concatenate([trunk, leaves])
        parts = md.loose_parts(m)
        t = time.time()
        md.measured_parts(m, parts)
        self.assertLess(time.time() - t, 10)

    def test_a_vine_of_cards_touching_one_by_one_is_measured_quickly(self):
        # the worst case for growing the touching group: each card touches only the next one, listed backwards
        trunk = trimesh.creation.cylinder(0.2, 4, sections=64)                 # along z, top at z = 2
        n = 1500
        base = np.array([[0, 0, 0], [0.12, 0, 0], [0, 0, 0.12]])
        v = np.array([[0.0, 0.0, 2.0 + 0.1 * k] for k in range(n)])[::-1, None, :] + base
        vine = trimesh.Trimesh(v.reshape(-1, 3), np.arange(3 * n).reshape(-1, 3), process=False)
        m = trimesh.util.concatenate([trunk, vine])
        parts = md.loose_parts(m)
        t = time.time()
        got = md.measured_parts(m, parts)
        self.assertLess(time.time() - t, 20)
        self.assertEqual(len(got), len(parts))                                # every card reached through the chain

    def test_texture_lost_only_in_lower_lods_is_said(self):
        s = trimesh.creation.icosphere(subdivisions=4)
        s.visual = trimesh.visual.TextureVisuals(uv=np.random.default_rng(0).random((len(s.vertices), 2)))
        res = md.diet(s, faces=100000, lods=2, lod_ratio=0.01)                # LOD0 whole (5120), LOD1 1000
        self.assertTrue(res.report["texture_lost"])
        self.assertTrue(any("LOD1 and lower" in w for w in res.report["warnings"]), res.report["warnings"])

    def test_all_flaps_are_not_reported_as_a_mesh(self):
        boxes = []
        for k in range(100):
            b = trimesh.creation.box(extents=[0.5] * 3)
            b.apply_translation([k * 0.8, 0, 0])
            boxes.append(b)
        try:
            res = md.diet(trimesh.util.concatenate(boxes), faces=200)
        except ValueError:
            return                                                             # a plain error is fine
        key = np.sort(md._welded(res.lods[0]).faces, axis=1)
        self.assertEqual(len(np.unique(key, axis=0)), len(key))

    def test_infinite_height_is_refused_cleanly(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "s.glb")
            trimesh.creation.box().export(src)
            self.assertEqual(md.main([src, "-o", os.path.join(d, "o.glb"), "--height", "inf"]), 2)


if __name__ == "__main__":
    unittest.main()
