"""The fifth skeptic re-grade (6/10, 25.10): each test red on 0.5.1."""
import os
import sys
import tempfile
import unittest

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import mesh_diet as md  # noqa: E402


def _textured_sphere():
    s = trimesh.creation.icosphere(subdivisions=3)
    img = __import__("PIL.Image", fromlist=["Image"]).new("RGB", (8, 8), (200, 30, 30))
    s.visual = trimesh.visual.TextureVisuals(uv=np.random.default_rng(1).random((len(s.vertices), 2)), image=img)
    return s


class Regrade5(unittest.TestCase):
    def test_a_scene_with_one_textured_part_says_the_texture_is_lost(self):
        with tempfile.TemporaryDirectory() as d:
            src, rep = os.path.join(d, "s.glb"), os.path.join(d, "r.json")
            box = trimesh.creation.box()
            box.apply_translation([3, 0, 0])
            trimesh.Scene([_textured_sphere(), box]).export(src)
            self.assertEqual(md.main([src, "-o", os.path.join(d, "o.glb"), "--faces", "500", "--report", rep]), 0)
            import json
            r = json.load(open(rep, encoding="utf-8"))
            self.assertTrue(r["texture_lost"])
            self.assertTrue(any("texture" in w for w in r["warnings"]))

    def test_a_budget_the_simplifier_cannot_reach_at_all_is_said(self):
        res = md.diet(trimesh.creation.cylinder(0.3, 0.7, sections=64), faces=40)
        self.assertTrue(any("simplifier stopped" in w for w in res.report["warnings"]), res.report["warnings"])

    def test_no_smaller_lod_does_not_blame_a_budget_below_the_mesh(self):
        res = md.diet(trimesh.creation.cylinder(0.3, 0.7, sections=64), faces=100, lods=3)
        self.assertFalse(any("above what this mesh has" in w for w in res.report["warnings"]), res.report["warnings"])

    def test_crushed_small_parts_warn_even_when_the_scene_error_is_small(self):
        crates = []
        for i in range(20):
            for j in range(20):
                b = trimesh.creation.box(extents=[0.5, 0.5, 0.5])
                b.apply_translation([i * 1.5, 0, j * 1.5])
                crates.append(b)
        res = md.diet(trimesh.util.concatenate(crates), faces=2000)
        self.assertGreater(res.report["shape_error"]["worst_part_percent"], 20)
        self.assertTrue(any("part" in w and "shape" in w for w in res.report["warnings"]), res.report["warnings"])

    def test_removed_parts_never_exceed_the_parts_there_are(self):
        # each crumb = two tiny boxes touching at one corner: one loose part (shares a vertex), two edge-components
        body = trimesh.creation.icosphere(subdivisions=5)
        crumbs = []
        for k in range(5):
            for off in (0.0, 0.01):
                c = trimesh.creation.box(extents=[0.01] * 3)
                c.apply_translation([2 + k + off, off, off])
                crumbs.append(c)
        m = trimesh.util.concatenate([body] + crumbs)
        res = md.diet(m, faces=20000)
        self.assertEqual(res.report["removed_components"], 5)
        self.assertEqual(res.report["before"]["components"], 6)


if __name__ == "__main__":
    unittest.main()
