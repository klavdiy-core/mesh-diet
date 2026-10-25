"""The tenth skeptic re-grade (7/10, 25.10): red on 0.5.6."""
import os
import sys
import tempfile
import unittest

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import mesh_diet as md  # noqa: E402


class Regrade10(unittest.TestCase):
    def test_decimation_leaves_no_folded_flaps(self):
        m = trimesh.creation.icosphere(subdivisions=7)
        m.vertices *= 1 + 0.02 * np.random.default_rng(3).standard_normal((len(m.vertices), 1))
        out = md.decimate(m, 20000)
        key = np.sort(md._welded(out).faces, axis=1)
        self.assertEqual(len(np.unique(key, axis=0)), len(key))

    def test_a_closed_input_that_comes_out_open_is_said(self):
        m = trimesh.creation.box()
        res = md.diet(m, faces=20000)
        self.assertFalse(any("watertight" in w for w in res.report["warnings"]))   # negative control

    def test_lod_paths_are_checked_before_the_work(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "s.glb")
            trimesh.creation.icosphere(4).export(src)
            os.mkdir(os.path.join(d, "x_LOD1.glb"))                            # a folder where LOD1 must go
            self.assertEqual(md.main([src, "-o", os.path.join(d, "x.glb"), "--lods", "2", "--faces", "500"]), 2)
            self.assertFalse(os.path.exists(os.path.join(d, "x.glb")))          # nothing half-written


if __name__ == "__main__":
    unittest.main()
