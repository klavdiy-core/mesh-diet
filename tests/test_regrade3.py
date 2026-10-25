"""The third skeptic re-grade (6/10, 25.10): each test red on 0.4.0."""
import os
import sys
import tempfile
import unittest

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import mesh_diet as md  # noqa: E402


def grid_of_cubes(n=1000):
    cubes = []
    for i in range(n):
        c = trimesh.creation.box((0.5, 0.5, 0.5))
        c.apply_translation(((i % 40) * 1.0, 0, (i // 40) * 1.0))
        cubes.append(c)
    return trimesh.util.concatenate(cubes)


class Regrade3(unittest.TestCase):
    def test_a_model_of_many_equal_parts_is_not_thrown_away(self):
        res = md.diet(grid_of_cubes(), faces=20000)
        self.assertGreater(len(md.loose_parts(res.lods[0])), 900)       # not 999 of 1000 deleted
        self.assertEqual(res.report["removed_components"], 0)

    def test_dropped_parts_are_said_with_their_share_of_the_faces(self):
        body = trimesh.creation.icosphere(subdivisions=5)
        crumbs = [trimesh.creation.box((0.01,) * 3) for _ in range(3)]
        for k, c in enumerate(crumbs):
            c.apply_translation((3 + k, 0, 0))
        res = md.diet(trimesh.util.concatenate([body] + crumbs), faces=5000)
        self.assertEqual(res.report["removed_components"], 3)
        self.assertIn("removed_faces_percent", res.report)

    def test_vertex_colours_lost_are_said(self):
        m = trimesh.creation.icosphere(subdivisions=5)
        m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=np.tile([200, 40, 40, 255], (len(m.vertices), 1)))
        res = md.diet(m, faces=2000)
        self.assertTrue(res.report["colours_lost"])
        self.assertTrue(any("colour" in w for w in res.report["warnings"]))

    def test_no_texture_warning_when_nothing_was_decimated(self):
        m = trimesh.creation.box()
        m.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(m.vertices), 2)))
        res = md.diet(m, faces=20000)
        self.assertFalse(res.report["texture_lost"])

    def test_a_tiny_budget_or_degenerate_input_is_a_clean_error(self):
        with tempfile.TemporaryDirectory() as d:
            small = os.path.join(d, "small.glb")
            trimesh.creation.icosphere(subdivisions=1).export(small)
            self.assertEqual(md.main([small, "--faces", "1", "-o", os.path.join(d, "o.glb")]), 2)
            degen = os.path.join(d, "degen.ply")
            trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [2, 0, 0]], faces=[[0, 1, 2]], process=False).export(degen)
            self.assertEqual(md.main([degen, "-o", os.path.join(d, "o2.glb")]), 2)

    def test_an_unsupported_output_extension_is_refused_before_the_work(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "s.glb")
            trimesh.creation.icosphere(subdivisions=2).export(src)
            self.assertEqual(md.main([src, "-o", os.path.join(d, "x.fbx")]), 2)
            self.assertEqual(md.main([src, "-o", os.path.join(d, "noext")]), 2)

    def test_a_big_separate_part_counts_by_size_not_by_face_count(self):
        ball = trimesh.creation.icosphere(subdivisions=4)                # 1 m wide, many faces
        pole = trimesh.creation.box((0.05, 3.0, 0.05))                    # 3 m tall, 12 faces
        pole.apply_translation((1.5, 1.5, 0))                             # apart from the ball
        res = md.diet(trimesh.util.concatenate([ball, pole]), faces=4000, height=1.8, min_component=0.0)
        lo, hi = res.lods[0].bounds
        self.assertAlmostEqual(hi[1] - lo[1], 1.8, places=2)              # the whole asset is 1.8 m


if __name__ == "__main__":
    unittest.main()
