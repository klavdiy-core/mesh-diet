"""The seventh skeptic re-grade (6.5/10, 25.10): each test red on 0.5.3."""
import json
import os
import sys
import tempfile
import unittest

import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import mesh_diet as md  # noqa: E402


def _coloured(m, rgba):
    m.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(baseColorFactor=rgba))
    return m


class Regrade7(unittest.TestCase):
    def test_a_big_low_poly_pedestal_is_not_a_crumb(self):
        ball = trimesh.creation.icosphere(subdivisions=5)                      # 20480 faces
        ball.apply_translation([0, 1.5, 0])
        pedestal = trimesh.creation.box(extents=[1.2, 1.0, 1.2])               # 12 faces, not touching
        res = md.diet(trimesh.util.concatenate([ball, pedestal]), faces=5000)
        self.assertEqual(res.report["removed_components"], 0)

    def test_tiny_crumbs_still_go(self):
        ball = trimesh.creation.icosphere(subdivisions=5)
        crumb = trimesh.creation.box(extents=[0.02] * 3)
        crumb.apply_translation([3, 0, 0])
        res = md.diet(trimesh.util.concatenate([ball, crumb]), faces=5000)
        self.assertEqual(res.report["removed_components"], 1)

    def test_two_flat_colour_materials_merged_are_reported(self):
        with tempfile.TemporaryDirectory() as d:
            src, rep = os.path.join(d, "s.glb"), os.path.join(d, "r.json")
            box = trimesh.creation.box()
            box.apply_translation([3, 0, 0])
            trimesh.Scene([_coloured(trimesh.creation.icosphere(3), [1, 0, 0, 1]),
                           _coloured(box, [0, 0, 1, 1])]).export(src)
            self.assertEqual(md.main([src, "-o", os.path.join(d, "o.glb"), "--faces", "99999", "--report", rep]), 0)
            r = json.load(open(rep, encoding="utf-8"))
            self.assertTrue(r["materials_merged"])
            self.assertTrue(any("materials" in w for w in r["warnings"]))
            self.assertFalse(any("UVs" in w for w in r["warnings"]))         # no texture, no UVs: don't say so

    def test_an_unwritable_output_is_a_clean_error_before_the_work(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "s.glb")
            trimesh.creation.icosphere(3).export(src)
            out = os.path.join(d, "o.glb")
            os.mkdir(out)                                                      # a FOLDER where the file must go
            self.assertEqual(md.main([src, "-o", out]), 2)


if __name__ == "__main__":
    unittest.main()
