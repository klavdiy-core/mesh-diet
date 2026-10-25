"""The eleventh skeptic re-grade (7/10, 25.10): red on 0.5.7."""
import os
import sys
import tempfile
import unittest

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import mesh_diet as md  # noqa: E402


def _crates(n):
    side = int(np.ceil(np.sqrt(n)))
    out = []
    for k in range(n):
        b = trimesh.creation.box(extents=[0.5] * 3)
        b.apply_translation([(k % side) * 0.8, 0, (k // side) * 0.8])
        out.append(b)
    return trimesh.util.concatenate(out)


class Regrade11(unittest.TestCase):
    def test_an_explicit_min_component_removes_a_floater_cloud(self):
        body = trimesh.creation.icosphere(subdivisions=5)
        fl = []
        rng = np.random.default_rng(1)
        for _ in range(200):
            f = trimesh.creation.icosphere(subdivisions=2, radius=0.03)
            f.apply_translation(rng.uniform(-2, 2, 3) * [1, 1, 1] + [0, 2.5, 0])
            fl.append(f)
        m = trimesh.util.concatenate([body] + fl)                              # floaters ~ 13% of the faces
        self.assertEqual(md.diet(m, faces=20000).report["removed_components"], 0)   # default: they may be the model
        res = md.diet(m, faces=20000, min_component=0.01, strict_min_component=True)
        self.assertEqual(res.report["removed_components"], 200)

    def test_many_equal_parts_keep_their_parts_and_the_budget(self):
        res = md.diet(_crates(2000), faces=12000)
        self.assertEqual(res.report["parts_lost_in_decimation"], 0)
        self.assertGreater(res.report["lods"][0]["faces"], 0.85 * 12000)

    def test_a_far_undershot_budget_is_said(self):
        res = md.diet(_crates(2000), faces=5000)
        f = res.report["lods"][0]["faces"]
        if f < 0.9 * 5000:
            self.assertTrue(any("under the budget" in w for w in res.report["warnings"]), res.report["warnings"])

    def test_a_scene_given_to_diet_is_a_plain_error(self):
        with self.assertRaises(ValueError):
            md.diet(trimesh.Scene([trimesh.creation.box()]), faces=100)

    def test_a_failed_run_leaves_no_new_folders(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "quad.obj")
            trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], [[0, 1, 2], [0, 2, 3]]).export(src)
            self.assertEqual(md.main([src, "-o", os.path.join(d, "new", "sub", "q.glb"), "--height", "1"]), 2)
            self.assertFalse(os.path.exists(os.path.join(d, "new")))

    def test_a_folder_in_the_way_is_named_as_a_folder(self):
        import io
        from contextlib import redirect_stderr
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "s.glb")
            trimesh.creation.box().export(src)
            os.mkdir(os.path.join(d, "o.glb"))
            err = io.StringIO()
            with redirect_stderr(err):
                self.assertEqual(md.main([src, "-o", os.path.join(d, "o.glb")]), 2)
            self.assertIn("is a folder", err.getvalue())


if __name__ == "__main__":
    unittest.main()
