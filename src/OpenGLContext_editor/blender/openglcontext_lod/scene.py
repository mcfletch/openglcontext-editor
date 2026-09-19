"""Making things in Blender: materials, boxes, chains of detail, the gallery.

The shapes and the places come from :mod:`gallery` and :mod:`msftlod`, which
hold no Blender; this is the part that calls ``bpy``. Keeping the two apart is
what lets the arithmetic be checked without opening Blender, and it keeps this
module down to what Blender is actually for.

Everything here works headless, so the gallery is built by a script rather than
by hand:

    blender --background --python -m ... -- --output gallery.glb
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Sequence

import bpy

from . import gallery as layout
from . import msftlod
from .content import MaterialMaps

__all__ = [
    'MaterialMaps',
    'box_mesh',
    'build_gallery',
    'clear',
    'decimation_chain',
    'lod_chain',
    'mark_level',
    'pbr_material',
    'place',
]

#: Blender's Principled BSDF calls clearcoat 'Coat' from 4.0 on; the exporter
#: writes those inputs as ``KHR_materials_clearcoat``.
_COAT_WEIGHT = 'Coat Weight'
_COAT_ROUGHNESS = 'Coat Roughness'

#: How far apart a box's UVs repeat by default, in metres.
DEFAULT_TILE = 2.0


def clear() -> None:
    """Empty the file: objects, meshes, materials, images, lights, cameras.

    A build starts from nothing so that running it twice gives the same world
    rather than two of it.
    """
    for collection in (bpy.data.objects, bpy.data.meshes, bpy.data.materials,
                       bpy.data.images, bpy.data.lights, bpy.data.cameras,
                       bpy.data.collections, bpy.data.node_groups):
        for item in list(collection):
            collection.remove(item, do_unlink=True)


def _image(path: str, srgb: bool) -> bpy.types.Image:
    """Load ``path`` once, in the colour space its contents are in.

    Kept by path rather than by name. Every material in a content directory
    calls its base colour ``color.jpg``, so a cache on the leaf name hands the
    floor's parquet to the walls, the ceiling and the plinths -- and the world
    still builds, exports and renders, wearing one texture throughout.
    """
    path = os.path.abspath(path)
    for existing in bpy.data.images:
        if existing.filepath and os.path.abspath(
                bpy.path.abspath(existing.filepath)) == path:
            return existing
    image = bpy.data.images.load(path)
    image.name = '%s_%s' % (os.path.basename(os.path.dirname(path)),
                            os.path.basename(path))
    image.colorspace_settings.name = 'sRGB' if srgb else 'Non-Color'
    return image


def pbr_material(name: str, maps: MaterialMaps) -> bpy.types.Material:
    """A Principled BSDF wired from ``maps``, ready for the glTF exporter.

    The exporter reads the Principled inputs, so what it writes is metallic /
    roughness with the coat as ``KHR_materials_clearcoat`` -- no node layout of
    our own for it to guess at.
    """
    existing = bpy.data.materials.get(name)
    if existing is not None:
        return existing
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    principled = tree.nodes['Principled BSDF']
    principled.inputs['Base Color'].default_value = maps.base_colour
    principled.inputs['Roughness'].default_value = maps.roughness_value
    principled.inputs['Metallic'].default_value = maps.metallic
    if _COAT_WEIGHT in principled.inputs:
        principled.inputs[_COAT_WEIGHT].default_value = maps.coat
        principled.inputs[_COAT_ROUGHNESS].default_value = maps.coat_roughness

    if maps.colour:
        texture = _texture_node(tree, maps.colour, srgb=True, row=0)
        tree.links.new(texture.outputs['Color'], principled.inputs['Base Color'])
    if maps.roughness:
        texture = _texture_node(tree, maps.roughness, srgb=False, row=1)
        tree.links.new(texture.outputs['Color'], principled.inputs['Roughness'])
    if maps.normal:
        texture = _texture_node(tree, maps.normal, srgb=False, row=2)
        mapper = tree.nodes.new('ShaderNodeNormalMap')
        mapper.location = (-260, -560)
        tree.links.new(texture.outputs['Color'], mapper.inputs['Color'])
        tree.links.new(mapper.outputs['Normal'], principled.inputs['Normal'])
    return material


def _texture_node(tree: bpy.types.NodeTree, path: str, srgb: bool,
                  row: int) -> bpy.types.Node:
    node = tree.nodes.new('ShaderNodeTexImage')
    node.image = _image(path, srgb)
    node.location = (-560, -280 * row)
    return node


def box_mesh(name: str, size: Sequence[float],
             tile: float = DEFAULT_TILE) -> bpy.types.Mesh:
    """A rectangular box, its faces unwrapped at a fixed number of metres.

    A gallery floor is forty-eight metres long, so the box's own zero-to-one
    unwrap would stretch one tile of parquet over the whole of it. Each face is
    laid out in metres instead and divided by ``tile``, which makes the texture
    the same size everywhere in the room whatever the wall it is on.
    """
    x, y, z = (value / 2.0 for value in size)
    corners = [(-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z),
               (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)]
    # Each face gets its own four vertices so the seams are sharp and the two
    # in-plane axes can be chosen per face.
    faces = [
        ((0, 3, 2, 1), (0, 1)),      # down    -- x, y
        ((4, 5, 6, 7), (0, 1)),      # up      -- x, y
        ((0, 1, 5, 4), (0, 2)),      # front   -- x, z
        ((2, 3, 7, 6), (0, 2)),      # back    -- x, z
        ((1, 2, 6, 5), (1, 2)),      # right   -- y, z
        ((3, 0, 4, 7), (1, 2)),      # left    -- y, z
    ]
    vertices: list[tuple[float, float, float]] = []
    polygons: list[tuple[int, ...]] = []
    uvs: list[tuple[float, float]] = []
    for indices, (across, up) in faces:
        first = len(vertices)
        for index in indices:
            point = corners[index]
            vertices.append(point)
            uvs.append((point[across] / tile, point[up] / tile))
        polygons.append(tuple(range(first, first + 4)))

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], polygons)
    layer = mesh.uv_layers.new(name='UVMap')
    for loop, value in zip(mesh.loops, uvs, strict=True):
        layer.data[loop.index].uv = value
    mesh.update()
    return mesh


def place(mesh: bpy.types.Mesh, name: str,
          location: Sequence[float] = (0.0, 0.0, 0.0),
          turn: float = 0.0,
          collection: bpy.types.Collection | None = None) -> bpy.types.Object:
    """An object over ``mesh`` at ``location``, turned ``turn`` about Z.

    Objects made this way share their mesh, which is what the glTF exporter
    needs to write one mesh and many nodes -- and what lets a renderer draw
    them as one instanced batch.
    """
    obj = bpy.data.objects.new(name, mesh)
    obj.location = tuple(location)
    obj.rotation_euler = (0.0, 0.0, turn)
    (collection or bpy.context.scene.collection).objects.link(obj)
    return obj


def mark_level(obj: bpy.types.Object, group: str, level: int,
               coverage: float | None = None) -> None:
    """Say which chain ``obj`` belongs to and how fine it is.

    These are the custom properties the export extension reads, and they are
    the author's own word: an object carrying them is a level whatever it is
    called.
    """
    obj[msftlod.GROUP_PROPERTY] = group
    obj[msftlod.LEVEL_PROPERTY] = level
    if coverage is not None:
        obj[msftlod.COVERAGE_PROPERTY] = float(coverage)


def decimation_chain(obj: bpy.types.Object, levels: int,
                     ratio: float = 0.5) -> list[bpy.types.Mesh]:
    """``levels`` meshes from ``obj``, each ``ratio`` of the triangles before.

    Blender's own Decimate modifier in collapse mode, which every Blender has
    and which interpolates the UVs, so the levels keep the material they came
    with. The first mesh returned is the original, untouched.

    The ratio is measured against the original rather than the level before,
    because that is what the modifier does: asking for a half twice is a
    quarter of the first mesh, not a quarter of the second.
    """
    if levels < 1:
        raise ValueError('a chain needs at least one level')
    made = [obj.data]
    modifier = obj.modifiers.new('LevelOfDetail', 'DECIMATE')
    modifier.decimate_type = 'COLLAPSE'
    modifier.use_collapse_triangulate = True
    try:
        for level in range(1, levels):
            modifier.ratio = ratio ** level
            depsgraph = bpy.context.evaluated_depsgraph_get()
            reduced = bpy.data.meshes.new_from_object(
                obj.evaluated_get(depsgraph), depsgraph=depsgraph)
            reduced.name = '%s_LOD%d' % (obj.data.name, level)
            made.append(reduced)
    finally:
        obj.modifiers.remove(modifier)
    return made


def lod_chain(meshes: Sequence[bpy.types.Mesh], group: str,
              location: Sequence[float] = (0.0, 0.0, 0.0),
              turn: float = 0.0,
              coverage: Sequence[float] | None = None,
              collection: bpy.types.Collection | None = None,
              coarse_collection: bpy.types.Collection | None = None,
              ) -> list[bpy.types.Object]:
    """One object per mesh, all at one place, marked as one chain.

    The finest goes in ``collection`` and the rest in ``coarse_collection``, so
    an author can put the alternatives out of sight without taking them out of
    the file -- they are exported either way and the export takes them out of
    the scene itself.
    """
    thresholds = list(coverage) if coverage else [None] * len(meshes)
    if len(thresholds) != len(meshes):
        raise ValueError('%d levels want %d coverage values, got %d'
                         % (len(meshes), len(meshes), len(thresholds)))
    made = []
    for level, (mesh, threshold) in enumerate(zip(meshes, thresholds,
                                              strict=True)):
        into = collection if level == 0 else (coarse_collection or collection)
        obj = place(mesh, '%s_LOD%d' % (group, level), location, turn, into)
        mark_level(obj, group, level, threshold)
        made.append(obj)
    return made


def _collection(name: str,
                parent: bpy.types.Collection | None = None
                ) -> bpy.types.Collection:
    made = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(made)
    return made


def import_mesh(path: str) -> bpy.types.Object:
    """The one mesh object in the glTF at ``path``, joined at the origin.

    A Poly Haven model is a single mesh under a transform; the transform is
    taken into the mesh so a level of it can stand where it is put rather than
    where the file left it.
    """
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    added = [obj for obj in bpy.data.objects if obj not in before]
    meshes = [obj for obj in added if obj.type == 'MESH']
    if len(meshes) != 1:
        raise ValueError('%s holds %d meshes; expected exactly one'
                         % (path, len(meshes)))
    obj = meshes[0]
    # Take the import's own transform into the vertices, then unparent, so the
    # mesh is the thing that is placed and every copy of it lands square.
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for other in added:
        if other is not obj:
            bpy.data.objects.remove(other, do_unlink=True)
    obj.select_set(False)
    return obj


def build_gallery(plan: layout.Gallery, bust: str,
                  materials: dict[str, MaterialMaps],
                  levels: int = 6, ratio: float = 0.5,
                  coverage: Sequence[float] | None = None) -> dict[str, int]:
    """Build the whole world; return what was made, by kind.

    ``materials`` is keyed by the names :class:`~gallery.Gallery` uses for its
    surfaces. ``coverage`` is the thresholds the chain switches at, one per
    level; without it the levels halve.
    """
    clear()
    made = {name: pbr_material(name, maps) for name, maps in materials.items()}

    room = _collection('Room')
    stands = _collection('Plinths')
    busts = _collection('Busts')
    coarse = _collection('Busts_LOD')

    counted = {'slabs': 0, 'plinths': 0, 'busts': 0, 'levels': 0, 'lights': 0}

    # Two slabs of the same size in the same material are the same mesh, which
    # is what lets thirty identical beams cost one mesh and thirty nodes -- the
    # same sharing the busts rely on, and the same instanced draw at the end.
    shapes: dict[tuple, bpy.types.Mesh] = {}

    def slab_mesh(name: str, slab: layout.Slab) -> bpy.types.Mesh:
        maps = materials[slab.material]
        key = (slab.size, slab.material, maps.tile)
        mesh = shapes.get(key)
        if mesh is None:
            mesh = shapes[key] = box_mesh(name, slab.size, tile=maps.tile)
            mesh.materials.append(made[slab.material])
        return mesh

    for slab in plan.room():
        place(slab_mesh(slab.name, slab), slab.name, slab.centre,
              collection=room)
        counted['slabs'] += 1
    for slab in plan.beams():
        place(slab_mesh('Beam', slab), slab.name, slab.centre, collection=room)
        counted['slabs'] += 1

    for slab in plan.plinths():
        place(slab_mesh('Plinth', slab), slab.name, slab.centre,
              collection=stands)
        counted['plinths'] += 1

    original = import_mesh(bust)
    chain = decimation_chain(original, levels, ratio)
    bpy.data.objects.remove(original, do_unlink=True)
    thresholds = list(coverage) if coverage else msftlod.coverage_series(levels)

    for placement in plan.busts():
        lod_chain(chain, placement.group, placement.position, placement.turn,
                  coverage=thresholds, collection=busts,
                  coarse_collection=coarse)
        counted['busts'] += 1
        counted['levels'] += len(chain)

    for lamp in plan.lights():
        light = bpy.data.lights.new(lamp.name, type='POINT')
        light.energy = lamp.power
        light.color = lamp.colour
        light.shadow_soft_size = 0.15
        obj = bpy.data.objects.new(lamp.name, light)
        obj.location = lamp.position
        bpy.context.scene.collection.objects.link(obj)
        counted['lights'] += 1

    for number, view in enumerate(plan.cameras()):
        camera = bpy.data.cameras.new(view.name)
        camera.lens_unit = 'FOV'
        # The thresholds in the file are a share of the window's *height*, so
        # the camera's vertical angle is the one that has to be what they
        # assume. With the sensor left on automatic Blender fits the wider axis
        # instead, and the exported ``yfov`` comes out narrower than asked for.
        camera.sensor_fit = 'VERTICAL'
        camera.angle_y = math.radians(layout.FIELD_OF_VIEW)
        obj = bpy.data.objects.new(view.name, camera)
        obj.location = view.position
        # A Blender camera looks down its own -Z; stood upright and turned
        # about to look along -Y, which is down the hall from the end it
        # starts at.
        obj.rotation_euler = (math.pi / 2, 0.0, math.pi + view.turn)
        bpy.context.scene.collection.objects.link(obj)
        if number == 0:
            bpy.context.scene.camera = obj
        counted['cameras'] = counted.get('cameras', 0) + 1

    _hide(coarse)
    return counted


def _hide(collection: bpy.types.Collection) -> None:
    """Put a collection out of the author's way without taking it out of the
    file: hidden objects are still exported, and the levels have to be."""
    layer = bpy.context.view_layer.layer_collection.children.get(collection.name)
    if layer is not None:
        layer.hide_viewport = True


def export(path: str, **settings: object) -> str:
    """Write the scene as a glB, with ``MSFT_lod`` on every chain in it.

    The add-on's export extension does the LODs, so this is Blender's ordinary
    exporter with the settings the world needs: cameras and lights kept, and
    no limit to what is visible, because the coarse levels are hidden.
    """
    arguments: dict[str, object] = dict(
        filepath=path,
        export_format='GLB',
        export_apply=False,
        export_cameras=True,
        export_lights=True,
        export_yup=True,
        export_extras=False,
        use_visible=False,
        use_renderable=False,
        use_selection=False,
        export_texture_dir='',
    )
    arguments.update(settings)
    bpy.ops.export_scene.gltf(**arguments)       # type: ignore[arg-type]
    return path


def register_extension() -> None:
    """Make the export extension available when running without the add-on.

    A headless build imports this module directly rather than installing an
    add-on, and the glTF exporter looks its extensions up among the *enabled
    add-ons* -- so the package has to be registered as one for the LODs to be
    written.
    """
    from . import ops

    ops.register()


def _names(objects: Iterable[bpy.types.Object]) -> list[str]:
    return sorted(obj.name for obj in objects)
