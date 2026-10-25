"""The report must not lie (the tracker's verdict, 4/10): the shape error looked only at the 16 triangles with the
nearest centres, so a big triangle after decimation could be missed and the error inflated; and textures/UVs were
dropped without a word. Red on 0.1.0."""
import os
import sys
import unittest

import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mesh_diet as md  # noqa: E402


class ShapeError(unittest.TestCase):
    def test_a_big_triangle_whose_centre_is_far_still_counts(self):
        big = [[-100, -100, 0], [100, -100, 0], [0, 100, 0]]           # its centre is far below the point
        verts, faces = [*big], [[0, 1, 2]]
        for i in range(20):                                             # small triangles 5 units above the point
            x = i * 0.3
            base = len(verts)
            verts += [[x, 95, 5], [x + 0.2, 95, 5], [x, 95.2, 5]]
            faces.append([base, base + 1, base + 2])
        mesh = trimesh.Trimesh(np.array(verts, float), np.array(faces), process=False)
        d = md._to_surface(np.array([[0.0, 90.0, 0.01]]), mesh)
        self.assertAlmostEqual(float(d[0]), 0.01, places=4)            # it lies right on the big triangle

    def test_identical_meshes_have_zero_error(self):
        m = trimesh.creation.icosphere(subdivisions=3)
        self.assertLess(md.shape_error(m, m.copy())["max_m"], 1e-6)


class Textures(unittest.TestCase):
    def test_a_textured_input_says_its_texture_is_not_kept(self):
        m = trimesh.creation.icosphere(subdivisions=3)
        uv = np.random.default_rng(0).random((len(m.vertices), 2))
        m.visual = trimesh.visual.TextureVisuals(uv=uv, material=trimesh.visual.material.PBRMaterial())
        r = md.diet(m, faces=200)
        self.assertTrue(r.report["texture_lost"])
        self.assertTrue(any("texture" in w.lower() or "uv" in w.lower() for w in r.report["warnings"]),
                        r.report["warnings"])

    def test_an_untextured_input_does_not_warn(self):
        r = md.diet(trimesh.creation.icosphere(subdivisions=3), faces=200)
        self.assertFalse(r.report["texture_lost"])
        self.assertFalse(any("texture" in w.lower() for w in r.report["warnings"]))


if __name__ == "__main__":
    unittest.main()
