import json
import os
import sys
import tempfile
import unittest

import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mesh_diet  # noqa: E402


def heavy_sphere(radius=0.37, center=(5.0, -2.0, 3.0)):
    """A 327k-face sphere with a slightly bumpy surface, off-centre and at a random scale.

    Even triangles, like the FlexiCubes / marching-cubes meshes image-to-3D models extract.
    """
    m = trimesh.creation.icosphere(subdivisions=7, radius=radius)
    n = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    bump = 1 + 0.03 * np.sin(5 * n[:, 0]) * np.cos(4 * n[:, 1])
    m.vertices = n * radius * bump[:, None]
    m.apply_translation(center)
    return m


def surface_gap(a, b, n=300000):
    """Symmetric max distance between two surfaces, as a fraction of a's bbox diagonal.

    Point clouds this dense keep the sampling error near 0.2%; independent of mesh_diet's own measure.
    """
    from scipy.spatial import cKDTree
    pa, _ = trimesh.sample.sample_surface(a, n, seed=1)
    pb, _ = trimesh.sample.sample_surface(b, n, seed=2)
    d1 = cKDTree(pb).query(pa)[0].max()
    d2 = cKDTree(pa).query(pb)[0].max()
    return max(d1, d2) / np.linalg.norm(a.extents)


class Diet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = heavy_sphere()
        assert len(cls.src.faces) > 190_000, len(cls.src.faces)

    def test_budget_within_5_percent_and_shape_kept(self):
        out = mesh_diet.diet(self.src, faces=5000)
        lod0 = out.lods[0]
        self.assertLessEqual(abs(len(lod0.faces) - 5000), 250, len(lod0.faces))
        # compare shapes in the same frame: undo our move by fitting both to unit size
        a = self.src.copy(); a.apply_translation(-a.bounds.mean(0)); a.apply_scale(1 / a.extents.max())
        b = lod0.copy(); b.apply_translation(-b.bounds.mean(0)); b.apply_scale(1 / b.extents.max())
        self.assertLess(surface_gap(a, b), 0.01)
        self.assertTrue(lod0.is_watertight)

    def test_scale_to_height_and_pivot_at_feet(self):
        out = mesh_diet.diet(self.src, faces=3000, height=1.8, up="y", pivot="bottom")
        m = out.lods[0]
        lo, hi = m.bounds
        self.assertAlmostEqual(hi[1] - lo[1], 1.8, places=4)
        self.assertAlmostEqual(lo[1], 0.0, places=5)
        self.assertAlmostEqual((lo[0] + hi[0]) / 2, 0.0, places=5)
        self.assertAlmostEqual((lo[2] + hi[2]) / 2, 0.0, places=5)

    def test_z_up_input_is_turned_y_up(self):
        box = trimesh.creation.box(extents=(1.0, 2.0, 4.0))  # tallest along z
        out = mesh_diet.diet(box, faces=1000, up="z", height=4.0)
        ext = out.lods[0].extents
        self.assertAlmostEqual(ext[1], 4.0, places=4)   # z became y
        self.assertAlmostEqual(ext[0], 1.0, places=4)

    def test_lods_shrink_by_ratio(self):
        out = mesh_diet.diet(self.src, faces=8000, lods=3, lod_ratio=0.5)
        counts = [len(m.faces) for m in out.lods]
        self.assertEqual(len(counts), 3)
        for want, got in zip((8000, 4000, 2000), counts):
            self.assertLessEqual(abs(got - want), 0.05 * want, counts)

    def test_floating_junk_is_removed(self):
        junk = trimesh.creation.icosphere(subdivisions=1, radius=0.01)
        junk.apply_translation((9, 9, 9))
        out = mesh_diet.diet(trimesh.util.concatenate([self.src, junk]), faces=4000)
        self.assertEqual(out.report["after"]["components"], 1)
        self.assertEqual(out.report["removed_components"], 1)

    def test_open_mesh_with_holes(self):
        # real scans and image-to-3D outputs are often open; nothing may crash or quietly patch them
        m = heavy_sphere()
        cut = m.triangles_center[:, 1] > m.bounds[1][1] - 0.05   # slice the cap off
        m.update_faces(~cut)
        m.remove_unreferenced_vertices()
        self.assertFalse(m.is_watertight)
        out = mesh_diet.diet(m, faces=4000)
        self.assertGreater(out.report["after"]["boundary_edges"], 0)
        self.assertFalse(out.report["after"]["watertight"])
        self.assertEqual(out.report["after"]["components"], 1)

    def test_small_mesh_is_not_inflated(self):
        box = trimesh.creation.box()
        out = mesh_diet.diet(box, faces=5000)
        self.assertEqual(len(out.lods[0].faces), 12)

    def test_pole_fans_are_warned_about_not_hidden(self):
        uv = trimesh.creation.uv_sphere(radius=1.0, count=[317, 317])  # 317 triangles meet at each pole
        out = mesh_diet.diet(uv, faces=5000)
        self.assertTrue(any("stopped at" in w for w in out.report["warnings"]), out.report["warnings"])

    def test_report(self):
        out = mesh_diet.diet(self.src, faces=5000, height=1.0)
        r = out.report
        self.assertEqual(r["before"]["faces"], len(self.src.faces))
        self.assertTrue(r["before"]["watertight"])
        self.assertEqual(r["after"]["faces"], len(out.lods[0].faces))
        self.assertIn("boundary_edges", r["after"])
        self.assertLess(r["shape_error"]["max_percent_of_size"], 1.0)
        self.assertEqual(len(r["lods"]), 1)
        self.assertEqual(r["warnings"], [])


class Cli(unittest.TestCase):
    def test_writes_glb_lods_and_report(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "raw.glb")
            heavy_sphere().export(src)
            out = os.path.join(d, "prop.glb")
            rep = os.path.join(d, "report.json")
            code = mesh_diet.main([src, "-o", out, "--faces", "4000", "--lods", "2",
                                   "--height", "0.5", "--report", rep])
            self.assertEqual(code, 0)
            for p in (out, os.path.join(d, "prop_LOD1.glb"), rep):
                self.assertTrue(os.path.exists(p), p)
            back = trimesh.load(out, force="mesh")
            self.assertAlmostEqual(back.extents[1], 0.5, places=3)
            with open(rep) as f:
                self.assertEqual(json.load(f)["lods"][1]["file"], "prop_LOD1.glb")


if __name__ == "__main__":
    unittest.main()
