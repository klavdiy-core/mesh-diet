# mesh-diet

Make a raw image-to-3D mesh fit for a game.

> RU: меши из image→3D (Hi3D, TRELLIS, Hunyuan3D…) тяжёлые и сырые: миллион треугольников, крошки вокруг, центр где попало, размера нет. mesh-diet чистит, ставит на ноги, задаёт размер в метрах, режет до бюджета треугольников, делает LOD и честно говорит, сколько формы это стоило.

## What hurts

Image-to-3D got good. Hi3D v1.0 refines geometry up to a 1536³ grid and, by its own docs, simplifies toward **1,000,000–2,000,000 faces**, centres the mean vertex and scales to a unit radius. That is right for a model viewer. A game prop wants 2k–30k triangles, a pivot at its feet, a real size, and LODs. Between the two sits an evening in Blender, per asset.

## What it does

1. **Cleans**: welds vertices by position (UV seams are not holes), drops degenerate and duplicate faces, removes loose crumbs below a share of the faces (default 0.2%).
2. **Turns it upright**: `--up z` turns a Z-up mesh into glTF's Y-up.
3. **Decimates to a budget** with quadric edge collapse ([fast-simplification](https://github.com/pyvista/fast-simplification)), within ~3% of the target, never adding triangles.
4. **Builds LODs**: each one a share of the previous (`--lods 3 --lod-ratio 0.5` → 20k, 10k, 5k).
5. **Sizes and places it**: `--height 1.8` (metres, measured on LOD0 over the whole model, so the asset is exactly as tall as asked) or `--size`; pivot at the bottom centre, at the centre, or left alone.
6. **Reports what it cost**: faces and vertices before and after, size, watertight or not, hole edges, non-manifold edges, parts removed, and the **shape error**: sampled point-to-surface distance between the cleaned original and LOD0, max and mean, in metres and as % of the size. It warns when the shape moved more than 2%, or when the simplifier could not reach the budget.

Output is `.glb` (or anything trimesh writes: `.obj`, `.ply`, `.stl`, `.off`).

## Install

```sh
git clone https://github.com/klavdiy-core/mesh-diet
cd mesh-diet
pip install -r requirements.txt     # trimesh, fast-simplification, scipy, numpy, networkx, pillow (OBJ textures)
```

CPU only, no GPU, no network.

## Use

```sh
python mesh_diet.py refined.glb                                   # 20k faces, pivot at feet -> refined.diet.glb
python mesh_diet.py refined.glb -o chair.glb --faces 8000 --height 0.9
python mesh_diet.py refined.glb -o hero.glb --faces 30000 --lods 4 --height 1.8 --report hero.json
python mesh_diet.py scan.obj -o rock.glb --up z --size 2.5 --pivot center
python mesh_diet.py upside.obj --up -z                            # -z / -y work with a space too (0.5.3)
```

LODs land next to the output: `hero.glb` (LOD0), `hero_LOD1.glb`, `hero_LOD2.glb`… The `_LODn` suffix is the naming Unity and Unreal recognise on import.

Example: [DamagedHelmet](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/DamagedHelmet) from the Khronos sample assets, cut to a third:

```
$ python mesh_diet.py DamagedHelmet.glb -o helmet.glb --faces 5000 --lods 3 --height 0.3
faces      15,452 -> 4,997   (LOD0 4,997, LOD1 2,490, LOD2 1,237)
size       0.3145 x 0.3 x 0.3334 m   pivot bottom
cleaned    0 loose part(s) removed, 55 left, watertight False, hole edges 715, non-manifold edges 0
shape cost max 0.011082 m (2.022% of size), mean 0.000299 m
wrote      helmet.glb, helmet_LOD1.glb, helmet_LOD2.glb
WARNING    a separate part (116 faces) changed shape by up to 13.589% of its own size, though the whole moved less. Small parts (eyes, bolts, crates) lose most; raise --faces or simplify them separately.
WARNING    the input has a texture / UVs; the output is geometry only (the simplifier does not carry UVs). Re-unwrap and bake the texture onto LOD0 in your DCC tool, or keep the original for close-ups.
WARNING    shape moved by up to 2.022% of the size; check it by eye or raise --faces.
```

(0.5.7, re-run on the real file; its loose parts are panels a tenth of the helmet or more, so none is a crumb.) The mean error is a third of a millimetre on a 30 cm helmet; the max is where thin panels lost their edge, and one small 116-face piece lost 14% of its own shape. That is what the warnings are for.

As a library:

```python
import trimesh, mesh_diet
mesh, had = mesh_diet.load("refined.glb")        # had = textures / colours / materials the file carried
res = mesh_diet.diet(mesh, faces=10000, lods=3, height=1.2, source_had=had)
res.lods[0].export("prop.glb")
print(res.report["shape_error"], res.report["warnings"])
```

## Limits (said plainly)

- **Geometry only.** Textures and UVs do not survive decimation here; Hi3D's output has none anyway. A textured input now says so in the report (`texture_lost: true` and a warning) instead of losing the texture silently. Bake the texture onto the decimated mesh afterwards.
- The shape error is exact (0.2.0): 0.1.0 measured each point only against the 16 triangles with the nearest centres and missed big triangles whose centre is far, which a decimated mesh has; it once reported 7.06 for a point 0.01 from the surface. Checked against brute force on 500 points: 0 difference.
- One mesh per file: a multi-object scene is merged into one mesh.
- `--height` and `--size` are measured on the model (0.4.0): the largest part, every part that touches it or touches
  a part that does (a table with separate legs is measured whole), and any part at least a quarter of its size. Only
  a part floating apart from all of them is left out, and a warning says how far it reaches — then the output is
  taller or wider than asked (a tree whose leaves float around the top: asked 4 m, 5.48 m overall). Scale such a
  model by hand or with `--size`. 0.3.0 measured the
  largest part alone, and a 0.75 m table with separate legs came out 11.25 m tall with the legs under the floor;
  before 0.3.0 a stray shard above the head shrank the figure. A part is loose when it shares no vertex with the
  rest: slivers the simplifier leaves on a sharp edge are not counted.
- LODs stop when the next one would have under 4 faces (a warning says so); 0.3.0 wrote empty LOD files.
- Loose parts below `--min-component` are deleted, and the report says how many and what share of the faces they were.
  A crumb must also be small: a part whose longest side is a tenth of the model's or more stays, however few faces it
  has (0.5.4: 0.5.3 deleted a 12-face pedestal and a 2 m staff next to a dense ball).
  Only crumbs, though: if they add up to 5% of the faces or more, they are the model and none is deleted (0.5.3:
  0.5.2 allowed up to half and deleted a tree's 1000 leaf cards, 44% of it, by default; 0.4.0 deleted 999 of 1000
  equal cubes). For a hand-made model with many small bolts set `--min-component 0`.
- "A quarter of its size" means the longest side of its box (0.5.0), not its face count: a 12-face pole 3 m tall
  next to a 5000-face ball is part of the model.
- Vertex colours (what most image-to-3D tools output) are not carried either: `colours_lost: true` and a warning.
  Neither warning appears when nothing was decimated.
- Output formats: .glb .gltf .obj .ply .stl .off, checked before any work; a .gltf is written self-contained (buffers embedded), so LODs never share buffer files; an .obj gets its own `<name>.mtl` and `<name>_*.png` (0.5.3: every OBJ used to
  write `material.mtl`, and a second asset in the folder took the first one's texture). A mesh within its budget is written from the original vertices, UV seams intact. A budget that leaves no triangles, or an
  input of only zero-area triangles, is a plain error.
- The shape error is also measured per loose part, against that part's own size (`worst_part_percent`), and a part
  that moved more than 10% of itself is named (0.5.2): 400 crates each crushed to a bipyramid moved 69% of a crate but
  1.25% of the scene, and 0.5.1 said nothing. Every part gets 32 sample points of its own. Past 100,000 parts only
  the whole is measured, and the report says so (`worst_part_percent: null` and a warning); 0.5.2 stopped at 4096
  and reported 0.0.
- A scene file with several materials is merged into one mesh, and merging drops the materials: that is reported as
  `texture_lost` / `colours_lost` even when nothing was decimated (0.5.2; 0.5.1 said false). Parts with different
  flat colours or materials are reported as `materials_merged: true` with a warning (0.5.4: a red and a blue part
  came out one grey with nothing said). A texture means UVs or an image; a flat colour is not called one.
- Every output path, each LOD's included (0.5.7), is tried for writing before the work starts: a protected folder or
  a folder in the way is a plain error, not a traceback or half the files after the LODs are computed (0.5.4).
- Folded flaps are dropped after decimation (0.5.7): the simplifier can leave pairs of triangles on the same three
  corners, zero thickness; a closed 1.3M-face sphere came out with 200 non-manifold edges. It now stays closed, and
  if a closed input still comes out open a warning says so. The summary line shows non-manifold edges too.
- Loose parts are counted one way everywhere (shares no vertex with the rest); 0.5.1 removed by edge-connectivity and
  could report «56 removed, 3 left» for a 5-part character.
- LOD1 and lower are placed with LOD0's transform, so they do not pop sideways on a switch; a decimated LOD can sit off
  the floor by about 1.5% of the height (27 mm on a 1.8 m blob's LOD2) or stray beside LOD0's outline.
- The shape check is exact and its memory is bounded (0.5.5): triangles are grouped by size, each group searched with
  its own bound, long slivers halved. 0.5.4 asked for 5 GB and crashed on a dense sculpted head on a coarse capsule
  body; that case now takes about 8 s. Still slow on a dense ring of long slivers: a 640-sided 4 m trunk with 1000
  leaves takes 34 s for the check, 44 s in all.
- Quadric decimation stalls on fans where hundreds of triangles meet at one vertex (UV-sphere poles, cones). The report says so instead of pretending.
- Tested on synthetic meshes built like image-to-3D output (even triangles, bumps, crumbs, holes) and on Khronos sample assets. Not yet on Hi3D's own output: it needs a 96 GB GPU. If you run it on one, an issue with the report is welcome.

## Test

```sh
python -m unittest discover -s tests
```

A 327k-face bumpy sphere, off-centre and at a random scale: LOD0 must land within 5% of a 5k budget with the shape within 1% (checked by an independent dense point-cloud distance), LODs must halve, the height must be exact and the feet on y = 0, Z-up must become Y-up, a floating crumb must go, an open mesh must stay open and not crash, a 12-face box must not grow, and a UV sphere must raise the "simplifier stopped" warning.

## License

MIT
