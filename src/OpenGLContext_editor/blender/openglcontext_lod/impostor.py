"""Baking an octahedral impostor: one picture of a model per direction.

The coarsest level of a chain is still a mesh, and past a certain distance a
mesh is the wrong thing entirely -- a bust twenty pixels tall spends five
hundred triangles on a silhouette a picture would draw exactly. So the last
level is not a coarser model, it is a *photograph* of the model from every
direction, folded onto one square texture, and a card that shows whichever one
matches where the viewer is standing.

    from openglcontext_lod import impostor

    baked = impostor.bake_atlas(bust, '/tmp/bust_impostor.png', grid=8)
    level = impostor.impostor_object(baked, 'bust_LOD6')

The fold is :mod:`octahedral`, which this shares with the renderer that reads
the atlas back. How large to make the atlas is a question about how small the
model will be on screen when the impostor takes over: at a switching threshold
of three per cent of a 720-line window the model is twenty-odd pixels tall, and
a 256-pixel atlas at eight views a side gives thirty-two pixels a view, which
covers it. Twice as many views is four times the texture for angles a distant
object does not resolve.

The views are rendered **orthographically**. A perspective view has a
foreshortening that belongs to the distance it was taken from, and an impostor
is drawn at every distance.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from typing import Any

import bpy
import numpy as np
from mathutils import Vector

from . import impostorspec, octahedral

__all__ = [
    'BakedImpostor',
    'bake_atlas',
    'bounding_sphere',
    'impostor_material',
    'impostor_mesh',
    'impostor_object',
]


#: How much room to leave around the model in its tile, as a multiple of its
#: radius. A little, so a silhouette touching the tile's edge cannot bleed into
#: the view next door under a bilinear filter.
DEFAULT_MARGIN = 1.06


class BakedImpostor:
    """What a bake produced: the atlas, and what is needed to draw it."""

    def __init__(self, path: str, grid: int, image: int, hemi: bool,
                 radius: float, centre: Sequence[float]):
        self.path = path
        self.grid = grid
        self.image = image
        self.hemi = hemi
        #: The radius the card has to be to cover the model it stands for.
        self.radius = float(radius)
        self.centre = tuple(float(v) for v in centre)

    def __repr__(self) -> str:
        return ('<BakedImpostor %s %dx%d views of %d px, radius %.3f>'
                % (os.path.basename(self.path), self.grid, self.grid,
                   self.image // self.grid, self.radius))


def bounding_sphere(obj: bpy.types.Object) -> tuple:
    """``(centre, radius)`` of ``obj`` in world space, from its bounds.

    The corners of the bounding box rather than the vertices: it is what the
    card has to cover from any angle, and a box's corner is the furthest any
    vertex can be.
    """
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    centre = sum(corners, Vector((0.0, 0.0, 0.0))) / len(corners)
    radius = max((corner - centre).length for corner in corners)
    return centre, radius


def bake_atlas(obj: bpy.types.Object, path: str, grid: int = 8,
               image: int = 256, hemi: bool = True,
               engine: str | None = None, world_strength: float = 1.0,
               margin: float = DEFAULT_MARGIN,
               samples: int = 16) -> BakedImpostor:
    """Render ``grid`` x ``grid`` views of ``obj`` into one atlas at ``path``.

    The scene's render settings, camera and world are put back afterwards, so
    this can be run on the file somebody is working in.

    Everything but ``obj`` is hidden for the duration: an impostor is a picture
    of one model, and a neighbour leaning into the shot is baked into it for
    good. ``engine`` defaults to the best of :data:`impostorspec.ENGINES` this
    Blender offers.
    """
    if engine is None:
        offered = bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items
        engine = impostorspec.render_engine(item.identifier for item in offered)
    tile = octahedral.tile_size(image, grid)
    centre, radius = bounding_sphere(obj)
    scene = bpy.context.scene
    with _scene_for_baking(scene, obj, tile, engine, world_strength, samples):
        camera = _camera_looking_at(scene, radius, margin)
        atlas = np.zeros((image, image, 4), dtype='f')
        directions = octahedral.view_directions(grid, hemi)
        with tempfile.TemporaryDirectory() as into:
            for index, direction in enumerate(directions):
                row, column = divmod(index, grid)
                _aim(camera, centre, radius, direction)
                pixels = _render_tile(scene, into, index, tile)
                x, y = octahedral.tile_origin(row, column, image, grid)
                atlas[y:y + tile, x:x + tile] = pixels
        _write_png(atlas, path)
    return BakedImpostor(path, grid, image, hemi, radius * margin, centre)


class _scene_for_baking:
    """Render settings for a bake, and the scene put back after it."""

    def __init__(self, scene: Any, subject: Any, tile: int, engine: str,
                 world_strength: float, samples: int):
        self.scene = scene
        self.subject = subject
        self.tile = tile
        self.engine = engine
        self.world_strength = world_strength
        self.samples = samples
        self.held: dict = {}

    def __enter__(self) -> _scene_for_baking:
        scene, render = self.scene, self.scene.render
        self.held = {
            'engine': render.engine, 'x': render.resolution_x,
            'y': render.resolution_y, 'percent': render.resolution_percentage,
            'transparent': render.film_transparent,
            'format': render.image_settings.file_format,
            'colour': render.image_settings.color_mode,
            'filepath': render.filepath, 'camera': scene.camera,
            'world': scene.world,
            'view_transform': scene.view_settings.view_transform,
            'look': scene.view_settings.look,
            'exposure': scene.view_settings.exposure,
            'gamma': scene.view_settings.gamma,
            'hidden': [(o, o.hide_render) for o in scene.objects],
        }
        # No film curve on a bake. Blender views a render through AgX by
        # default, which is right for a picture somebody looks at and wrong for
        # one a renderer reads back: the impostor takes over from a mesh drawn
        # from the same colours, and a tone-mapped card is a visibly paler,
        # flatter version of the model at the moment it appears.
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.look = 'None'
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
        render.engine = self.engine
        render.resolution_x = render.resolution_y = self.tile
        render.resolution_percentage = 100
        render.film_transparent = True
        render.image_settings.file_format = 'PNG'
        render.image_settings.color_mode = 'RGBA'
        if hasattr(scene, 'cycles'):
            scene.cycles.samples = self.samples
        if hasattr(scene, 'eevee'):
            scene.eevee.taa_render_samples = self.samples
        # An even white surround, so the bake carries the model's own colours
        # rather than one scene's lighting. A card lit from the left is lit
        # from the left wherever it is later put.
        scene.world = bpy.data.worlds.new('ImpostorBake')
        scene.world.use_nodes = True
        background = scene.world.node_tree.nodes['Background']
        background.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
        background.inputs['Strength'].default_value = self.world_strength
        for obj in scene.objects:
            obj.hide_render = obj is not self.subject
        return self

    def __exit__(self, *exception: Any) -> None:
        scene, render = self.scene, self.scene.render
        held = self.held
        baked, scene.world = scene.world, held['world']
        if baked is not None and baked.users == 0:
            bpy.data.worlds.remove(baked)
        render.engine = held['engine']
        render.resolution_x, render.resolution_y = held['x'], held['y']
        render.resolution_percentage = held['percent']
        render.film_transparent = held['transparent']
        render.image_settings.file_format = held['format']
        render.image_settings.color_mode = held['colour']
        render.filepath = held['filepath']
        scene.view_settings.view_transform = held['view_transform']
        scene.view_settings.look = held['look']
        scene.view_settings.exposure = held['exposure']
        scene.view_settings.gamma = held['gamma']
        for obj, hidden in held['hidden']:
            obj.hide_render = hidden
        camera = scene.camera
        scene.camera = held['camera']
        if camera is not None and camera.name.startswith('ImpostorCamera'):
            data = camera.data
            bpy.data.objects.remove(camera, do_unlink=True)
            bpy.data.cameras.remove(data)


def _camera_looking_at(scene: Any, radius: float, margin: float) -> Any:
    """An orthographic camera wide enough to hold the model from any angle."""
    data = bpy.data.cameras.new('ImpostorCamera')
    data.type = 'ORTHO'
    data.ortho_scale = 2.0 * radius * margin
    # Room in front and behind for the model at any angle, and no more: a
    # clip range wider than it needs to be spends depth precision.
    data.clip_start = max(1e-4, radius * margin)
    data.clip_end = radius * margin * 4.0
    camera = bpy.data.objects.new('ImpostorCamera', data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    return camera


def _aim(camera: Any, centre: Any, radius: float, direction: Sequence) -> None:
    """Stand the camera off along ``direction`` and look back at the model."""
    towards = Vector(direction).normalized()
    camera.location = Vector(centre) + towards * (radius * 2.5)
    # The camera looks down its own -Z, so +Z has to point back the way it came.
    camera.rotation_mode = 'QUATERNION'
    camera.rotation_quaternion = towards.to_track_quat('Z', 'Y')


def _render_tile(scene: Any, into: str, index: int, tile: int) -> np.ndarray:
    """Render one view and return it as ``tile`` x ``tile`` RGBA, top row first."""
    path = os.path.join(into, 'view%04d.png' % (index,))
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(path)
    try:
        flat = np.zeros(len(image.pixels), dtype='f')
        image.pixels.foreach_get(flat)
        # Blender hands pixels back bottom row first, the way GL stores them;
        # the atlas is written top row first, the way an image file is read.
        return flat.reshape((tile, tile, 4))[::-1]
    finally:
        bpy.data.images.remove(image)


def _write_png(atlas: np.ndarray, path: str) -> None:
    """Save a top-row-first RGBA float array as a PNG."""
    height, width = atlas.shape[:2]
    image = bpy.data.images.new('ImpostorAtlas', width=width, height=height,
                                alpha=True, float_buffer=False)
    try:
        image.pixels.foreach_set(atlas[::-1].reshape(-1))
        image.filepath_raw = path
        image.file_format = 'PNG'
        image.save()
    finally:
        bpy.data.images.remove(image)


def impostor_material(baked: BakedImpostor, name: str) -> bpy.types.Material:
    """A material that draws ``baked``: the atlas, unlit, cut out by its alpha.

    **Unlit** because the lighting is already in the picture -- baking it again
    under the scene's own lights would light it twice. **Cut out** rather than
    blended because an impostor is opaque where the model was, and an alpha
    blend would have to be sorted against everything else in the world for the
    sake of an edge a few pixels long.
    """
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.blend_method = 'CLIP' if hasattr(material, 'blend_method') else \
        material.blend_method
    tree = material.node_tree
    principled = tree.nodes['Principled BSDF']
    texture = tree.nodes.new('ShaderNodeTexImage')
    texture.image = bpy.data.images.load(baked.path)
    texture.interpolation = 'Linear'
    texture.location = (-420, 0)
    tree.links.new(texture.outputs['Color'], principled.inputs['Base Color'])
    tree.links.new(texture.outputs['Alpha'], principled.inputs['Alpha'])
    principled.inputs['Roughness'].default_value = 1.0
    principled.inputs['Metallic'].default_value = 0.0
    # What the glTF exporter writes as KHR_materials_unlit.
    for node in tree.nodes:
        if node.type == 'OUTPUT_MATERIAL':
            output = node
            break
    else:                                   # pragma: no cover - always present
        return material
    background = tree.nodes.new('ShaderNodeBackground')
    background.location = (-120, -220)
    tree.links.new(texture.outputs['Color'], background.inputs['Color'])
    material['gltf_unlit'] = True
    tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    return material


def impostor_mesh(baked: BakedImpostor, name: str) -> bpy.types.Mesh:
    """A square card of the right size, wearing ``baked``, as a mesh.

    A mesh rather than an object, so it is a **level like any other**: the chain
    builder places one object per level at the same spot, and copies of a model
    share their level meshes, which is what lets a field of impostors collapse
    into one instanced draw.

    The card is built around the model's own centre rather than around its
    origin -- a bust's origin is between its feet and its middle is at chest
    height, and a card centred on the origin would show the model hanging off
    the top of it. That offset is baked into the vertices, which means the
    subject must have been at the origin when it was baked; ``import_mesh``
    puts it there.
    """
    reach = baked.radius
    cx, cy, cz = baked.centre
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(
        [(cx - reach, cy - reach, cz), (cx + reach, cy - reach, cz),
         (cx + reach, cy + reach, cz), (cx - reach, cy + reach, cz)],
        [], [(0, 1, 2, 3)])
    layer = mesh.uv_layers.new(name='UVMap')
    for loop, uv in zip(mesh.loops, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0),
                                     (0.0, 1.0)], strict=True):
        layer.data[loop.index].uv = uv
    mesh.update()
    material = impostor_material(baked, name)
    impostorspec.mark(material, baked.grid, baked.hemi)
    mesh.materials.append(material)
    return mesh


def impostor_object(baked: BakedImpostor, name: str,
                    collection: bpy.types.Collection | None = None
                    ) -> bpy.types.Object:
    """A square card of the right size, wearing ``baked``.

    The card is in the XY plane and centred on the model's own centre; the
    renderer turns it to the viewer and picks the tile. Its size is the
    model's bounding sphere, which is what makes the picture line up with the
    mesh it takes over from.
    """
    obj = bpy.data.objects.new(name, impostor_mesh(baked, name))
    (collection or bpy.context.scene.collection).objects.link(obj)
    return obj

