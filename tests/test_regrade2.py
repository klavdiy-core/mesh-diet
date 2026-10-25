"""The second skeptic re-grade (5/10, 25.10): each test red on 0.3.0."""
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import mesh_diet as md  # noqa: E402


def table():
    """A 0.75 m table: a 5 cm top and four separate 0.7 m legs touching it (a model built from pieces)."""
    top = trimesh.creation.box((1.2, 0.05, 0.8))
    top.apply_translation((0, 0.725, 0))
    parts = [top]
    for x in (-0.55, 0.55):
        for z in (-0.35, 0.35):
            leg = trimesh.creation.box((0.05, 0.7, 0.05))
            leg.apply_translation((x, 0.35, z))
            parts.append(leg)
    return trimesh.util.concatenate([p.subdivide().subdivide() for p in parts])


class Regrade2(unittest.TestCase):
    def test_a_model_built_from_touching_pieces_is_measured_whole(self):
        res = md.diet(table(), faces=5000, height=0.75)
        lo, hi = res.lods[0].bounds
        self.assertAlmostEqual(hi[1] - lo[1], 0.75, places=2)
        self.assertAlmostEqual(lo[1], 0.0, places=3)                  # standing on the floor, legs included

    def test_a_far_floating_shard_is_still_left_out_of_the_height(self):
        body = trimesh.creation.icosphere(subdivisions=4, radius=0.9)
        shard = trimesh.creation.icosphere(subdivisions=2, radius=0.05)
        shard.apply_translation((0, 3.0, 0))                          # far above, not touching
        res = md.diet(trimesh.util.concatenate([body, shard]), faces=3000, height=1.8, min_component=0.0)
        body_out = max(md.loose_parts(res.lods[0]), key=len)
        sub = res.lods[0].submesh([body_out], append=True)
        self.assertAlmostEqual(sub.extents[1], 1.8, places=2)

    def test_tiny_lods_are_not_written_empty(self):
        res = md.diet(trimesh.creation.icosphere(subdivisions=3), faces=100, lods=10)
        self.assertTrue(all(len(x.faces) >= 4 for x in res.lods))
        self.assertLess(len(res.lods), 10)
        self.assertTrue(any("LOD" in w and "stopped" in w.lower() for w in res.report["warnings"]))

    def test_missing_output_folders_are_made(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "s.glb")
            trimesh.creation.icosphere(subdivisions=3).export(src)
            out = os.path.join(d, "new folder", "out", "x.glb")
            rep = os.path.join(d, "reports", "r.json")
            code = md.main([src, "-o", out, "--faces", "200", "--report", rep])
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(out) and os.path.exists(rep))

    def test_help_does_not_promise_a_printed_report(self):
        r = subprocess.run([sys.executable, os.path.join(os.path.dirname(HERE), "mesh_diet.py"), "--help"],
                           capture_output=True, text=True)
        self.assertNotIn("also printed", r.stdout)

    def test_pillow_is_a_requirement(self):
        req = open(os.path.join(os.path.dirname(HERE), "requirements.txt"), encoding="utf-8").read().lower()
        self.assertIn("pillow", req)


if __name__ == "__main__":
    unittest.main()
