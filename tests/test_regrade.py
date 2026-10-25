"""What the regrade skeptic found (6.5/10, 25.10). Each test is red on 0.2.0."""
import contextlib
import io
import os
import sys
import tempfile
import unittest

import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mesh_diet as md  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def sphere_and_far_crumb():
    body = trimesh.creation.icosphere(subdivisions=5)                 # 20480 faces, 2 m tall
    crumb = trimesh.creation.icosphere(subdivisions=2)                # 320 faces: 1.6 %, above the 0.2 % default
    crumb.apply_scale(0.1)
    crumb.apply_translation([0, 3, 0])                                 # floating 2 m above the head
    return trimesh.util.concatenate([body, crumb])


class Height(unittest.TestCase):
    def test_height_is_measured_on_the_main_body_and_a_kept_part_is_said(self):
        r = md.diet(sphere_and_far_crumb(), faces=4000, height=1.8)
        parts = md.loose_parts(r.lods[0])
        main = max(parts, key=len)
        body = r.lods[0].submesh([main], append=True)
        self.assertAlmostEqual(body.extents[1], 1.8, places=2)       # not 0.82: the crumb no longer shrinks it
        self.assertTrue(any("part" in w and "--min-component" in w for w in r.report["warnings"]),
                        r.report["warnings"])


    def test_slivers_left_by_the_simplifier_are_not_loose_parts(self):
        # measured: a subdivided box decimated to 300 faces has 7 edge-connected pieces - single triangles on
        # non-manifold edges that share vertices with the body. They are not loose; no warning, size on the whole.
        box = trimesh.creation.box(extents=[1, 2, 1]).subdivide().subdivide().subdivide()
        r = md.diet(box, faces=300, height=1.8)
        self.assertFalse(any("parts left" in w for w in r.report["warnings"]), r.report["warnings"])
        self.assertEqual(r.report["after"]["components"], 1)
        self.assertAlmostEqual(r.lods[0].extents[1], 1.8, places=3)


class Inputs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "s.glb")
        trimesh.creation.icosphere(subdivisions=3).export(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, *args):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = md.main([*args])
        return code, err.getvalue()

    def test_bad_numbers_are_errors_not_tracebacks(self):
        for args in (["--faces", "0"], ["--lods", "0"], ["--lod-ratio", "1.5"], ["--height", "-1"]):
            code, err = self.run_main(self.path, *args)
            self.assertEqual(code, 2, args)
            self.assertIn(args[0], err)

    def test_a_flat_mesh_with_a_height_is_refused_plainly(self):
        flat = os.path.join(self.tmp.name, "flat.glb")
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]], float)
        trimesh.Trimesh(v, [[0, 1, 2], [1, 3, 2]]).export(flat)          # lies in y = 0: no height to scale
        code, err = self.run_main(flat, "--height", "1.8")
        self.assertEqual(code, 2)
        self.assertIn("no height", err)

    def test_networkx_is_a_listed_dependency(self):
        with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as f:
            self.assertIn("networkx", f.read())                       # fix_normals needs it on some paths


if __name__ == "__main__":
    unittest.main()
