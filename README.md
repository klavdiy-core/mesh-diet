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
```

LODs land next to the output: `hero.glb` (LOD0), `hero_LOD1.glb`, `hero_LOD2.glb`… The `_LODn` suffix is the naming Unity and Unreal recognise on import.

Example: [DamagedHelmet](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/DamagedHelmet) from the Khronos sample assets, cut to a third:

```
$ python mesh_diet.py DamagedHelmet.glb -o helmet.glb --faces 5000 --lods 3 --height 0.3
faces      15,452 -> 4,999   (LOD0 4,999, LOD1 2,499, LOD2 1,249)
size       0.3145 x 0.3 x 0.3334 m   pivot bottom
cleaned    8 loose part(s) removed, 55 left, watertight False, hole edges 707
shape cost max 0.019612 m (3.579% of size), mean 0.000462 m
wrote      helmet.glb, helmet_LOD1.glb, helmet_LOD2.glb
WARNING    shape moved by up to 3.579% of the size; check it by eye or raise --faces.
```

The mean error is half a millimetre on a 30 cm helmet; the max is where thin panels lost their edge. That is what the warning is for.

As a library:

```python
import trimesh, mesh_diet
res = mesh_diet.diet(trimesh.load("refined.glb", force="mesh"), faces=10000, lods=3, height=1.2)
res.lods[0].export("prop.glb")
print(res.report["shape_error"], res.report["warnings"])
```

## Limits (said plainly)

- **Geometry only.** Textures and UVs do not survive decimation here; Hi3D's output has none anyway. A textured input now says so in the report (`texture_lost: true` and a warning) instead of losing the texture silently. Bake the texture onto the decimated mesh afterwards.
- The shape error is exact (0.2.0): 0.1.0 measured each point only against the 16 triangles with the nearest centres and missed big triangles whose centre is far, which a decimated mesh has; it once reported 7.06 for a point 0.01 from the surface. Checked against brute force on 500 points: 0 difference.
- One mesh per file: a multi-object scene is merged into one mesh.
- `--height` and `--size` are measured on the model (0.4.0): the largest part, every part that touches it or touches
  a part that does (a table with separate legs is measured whole), and any part at least a quarter of its size. Only
  a part floating apart from all of them is left out, and a warning says how far it reaches. 0.3.0 measured the
  largest part alone, and a 0.75 m table with separate legs came out 11.25 m tall with the legs under the floor;
  before 0.3.0 a stray shard above the head shrank the figure. A part is loose when it shares no vertex with the
  rest: slivers the simplifier leaves on a sharp edge are not counted.
- LODs stop when the next one would have under 4 faces (a warning says so); 0.3.0 wrote empty LOD files.
- Loose parts below `--min-component` are deleted, and the report says how many and what share of the faces they were.
  Only crumbs, though: if they add up to more than half the model (1000 equal cubes are each 0.1% of the faces), none
  is deleted (0.5.0; 0.4.0 deleted 999 of them and reported 0% shape cost). For a hand-made model with many small
  bolts set `--min-component 0`.
- "A quarter of its size" means the longest side of its box (0.5.0), not its face count: a 12-face pole 3 m tall
  next to a 5000-face ball is part of the model.
- Vertex colours (what most image-to-3D tools output) are not carried either: `colours_lost: true` and a warning.
  Neither warning appears when nothing was decimated.
- Output formats: .glb .gltf .obj .ply .stl .off, checked before any work; a .gltf is written self-contained (buffers embedded), so LODs never share buffer files. A mesh within its budget is written from the original vertices, UV seams intact. A budget that leaves no triangles, or an
  input of only zero-area triangles, is a plain error.
- Quadric decimation stalls on fans where hundreds of triangles meet at one vertex (UV-sphere poles, cones). The report says so instead of pretending.
- Tested on synthetic meshes built like image-to-3D output (even triangles, bumps, crumbs, holes) and on Khronos sample assets. Not yet on Hi3D's own output: it needs a 96 GB GPU. If you run it on one, an issue with the report is welcome.

## Test

```sh
python -m unittest discover -s tests
```

A 327k-face bumpy sphere, off-centre and at a random scale: LOD0 must land within 5% of a 5k budget with the shape within 1% (checked by an independent dense point-cloud distance), LODs must halve, the height must be exact and the feet on y = 0, Z-up must become Y-up, a floating crumb must go, an open mesh must stay open and not crash, a 12-face box must not grow, and a UV sphere must raise the "simplifier stopped" warning.

## License

MIT
