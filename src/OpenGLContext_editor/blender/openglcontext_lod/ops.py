"""The operations: make a chain, check one, build the gallery, export it.

Everything here is a thin shell. What a chain *is* lives in :mod:`msftlod`,
where the gallery goes lives in :mod:`gallery`, and how Blender is made to do
it lives in :mod:`scene`; these are the buttons.
"""

from __future__ import annotations

import os
from typing import ClassVar

import bpy
from bpy.props import BoolProperty, FloatProperty, IntProperty, StringProperty
from bpy.types import Operator, Panel

from . import budget, msftlod, scene
from . import content as content_module
from . import gallery as layout

__all__ = ['classes', 'register', 'unregister']


class OBJECT_OT_make_lod_chain(Operator):
    """Make coarser levels of the selected mesh and mark them as one chain"""

    bl_idname = 'object.make_lod_chain'
    bl_label = 'Make LOD chain'
    bl_options: ClassVar[set[str]] = {'REGISTER', 'UNDO'}

    levels: IntProperty(
        name='Levels', default=6, min=2, max=12,
        description='How many levels the chain has, counting the original',
    )
    ratio: FloatProperty(
        name='Ratio', default=0.5, min=0.01, max=0.99,
        description='What share of the triangles each level keeps of the one '
                    'before it',
    )
    max_triangles: IntProperty(
        name='Finest level at most', default=0, min=0, soft_max=200_000,
        description='Triangle budget for the finest level: a denser mesh is '
                    'decimated down to it before the chain starts. 0 leaves '
                    'the mesh as it came',
    )
    hide_coarse: BoolProperty(
        name='Hide the coarse levels', default=True,
        description='Put the alternatives out of the viewport. They are still '
                    'exported: the export takes them out of the scene itself',
    )

    @classmethod
    def poll(cls, context):
        return context.active_object is not None \
            and context.active_object.type == 'MESH'

    def execute(self, context):
        original = context.active_object
        group = msftlod.named_level(original.name)
        group = group[0] if group else original.name
        try:
            meshes = scene.decimation_chain(original, self.levels, self.ratio,
                                            self.max_triangles or None)
        except (RuntimeError, ValueError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}

        coarse = bpy.data.collections.get('%s_LOD' % (group,))
        if coarse is None:
            coarse = bpy.data.collections.new('%s_LOD' % (group,))
            context.scene.collection.children.link(coarse)

        scene.mark_level(original, group, 0)
        for level, mesh in enumerate(meshes[1:], start=1):
            scene.place(mesh, '%s_LOD%d' % (group, level),
                        original.location, original.rotation_euler.z, coarse)
            scene.mark_level(coarse.objects['%s_LOD%d' % (group, level)],
                             group, level)
        if self.hide_coarse:
            found = context.view_layer.layer_collection.children.get(coarse.name)
            if found is not None:
                found.hide_viewport = True

        self.report({'INFO'}, '%s: %s' % (group, budget.describe(
            [scene.triangle_count(mesh) for mesh in meshes])))
        return {'FINISHED'}


class OBJECT_OT_check_lod_chains(Operator):
    """Say what the chains in this scene will export as, without exporting"""

    bl_idname = 'object.check_lod_chains'
    bl_label = 'Check LOD chains'
    bl_options: ClassVar[set[str]] = {'REGISTER'}

    def execute(self, context):
        levels = []
        for index, obj in enumerate(context.scene.objects):
            try:
                one = msftlod.level_of(obj.name, obj, node=index)
            except ValueError as error:
                self.report({'ERROR'}, str(error))
                return {'CANCELLED'}
            if one is not None:
                levels.append(one)
        if not levels:
            self.report({'WARNING'}, 'no object in this scene is a level')
            return {'CANCELLED'}
        try:
            plan = msftlod.plan(levels)
        except ValueError as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        self.report({'INFO'}, '%d chain(s), %d level(s) in all'
                    % (len(plan.ids), len(levels)))
        return {'FINISHED'}


class EXPORT_SCENE_OT_lod_glb(Operator):
    """Export the scene as a glB, with every chain written as MSFT_lod"""

    bl_idname = 'export_scene.lod_glb'
    bl_label = 'Export world (glTF + MSFT_lod)'
    bl_options: ClassVar[set[str]] = {'REGISTER'}

    filepath: StringProperty(subtype='FILE_PATH')
    filename_ext = '.glb'
    filter_glob: StringProperty(default='*.glb', options={'HIDDEN'})

    def invoke(self, context, event):  # noqa: ARG002 Operator.invoke signature
        if not self.filepath:
            self.filepath = os.path.splitext(
                bpy.data.filepath or 'world')[0] + '.glb'
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):  # noqa: ARG002 Operator.execute signature
        scene.export(self.filepath)
        self.report({'INFO'}, 'wrote %s' % (self.filepath,))
        return {'FINISHED'}


class SCENE_OT_build_bust_gallery(Operator):
    """Build the bust gallery demo world in this file"""

    bl_idname = 'scene.build_bust_gallery'
    bl_label = 'Build the bust gallery'
    bl_options: ClassVar[set[str]] = {'REGISTER', 'UNDO'}

    content: StringProperty(
        name='Content', subtype='DIR_PATH',
        description='The directory holding the bust and the CC0 materials',
    )
    bays: IntProperty(name='Bays', default=30, min=1, max=200)

    def execute(self, context):  # noqa: ARG002 Operator.execute signature
        if not self.content or not os.path.isdir(self.content):
            self.report({'ERROR'}, 'name the directory the content is in')
            return {'CANCELLED'}
        plan = layout.Gallery(bays=self.bays)
        try:
            counted = scene.build_gallery(
                plan, *content_module.gallery_content(self.content))
        except (OSError, ValueError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        self.report({'INFO'}, layout.describe(plan) + ' (%(busts)d busts, '
                    '%(levels)d level objects)' % counted)
        return {'FINISHED'}


class VIEW3D_PT_lod(Panel):
    """Levels of detail, in the 3D view's sidebar"""

    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'LOD'
    bl_label = 'Levels of Detail'

    def draw(self, context):
        column = self.layout.column(align=True)
        column.operator(OBJECT_OT_make_lod_chain.bl_idname, icon='MOD_DECIM')
        column.operator(OBJECT_OT_check_lod_chains.bl_idname, icon='CHECKMARK')
        column.separator()
        column.operator(EXPORT_SCENE_OT_lod_glb.bl_idname, icon='EXPORT')

        obj = context.active_object
        if obj is None:
            return
        box = self.layout.box()
        group = obj.get(msftlod.GROUP_PROPERTY)
        if group is None:
            box.label(text='%s is not a level' % (obj.name,), icon='DOT')
            return
        box.label(text='chain: %s' % (group,), icon='MOD_DECIM')
        box.label(text='level: %s' % (obj.get(msftlod.LEVEL_PROPERTY, 0),))
        coverage = obj.get(msftlod.COVERAGE_PROPERTY)
        box.label(text='takes over at: %s'
                  % ('%.3f' % coverage if coverage is not None else 'halving'))


def _menu(self, context):  # noqa: ARG001 Blender menu draw callback signature
    self.layout.operator(EXPORT_SCENE_OT_lod_glb.bl_idname,
                         text='glTF 2.0 with MSFT_lod (.glb)')


def _object_menu(self, context):  # noqa: ARG001 Blender menu draw callback signature
    self.layout.separator()
    self.layout.operator(OBJECT_OT_make_lod_chain.bl_idname)


classes = (
    OBJECT_OT_make_lod_chain,
    OBJECT_OT_check_lod_chains,
    EXPORT_SCENE_OT_lod_glb,
    SCENE_OT_build_bust_gallery,
    VIEW3D_PT_lod,
)


def register() -> None:
    for one in classes:
        bpy.utils.register_class(one)
    bpy.types.TOPBAR_MT_file_export.append(_menu)
    bpy.types.VIEW3D_MT_object.append(_object_menu)


def unregister() -> None:
    bpy.types.VIEW3D_MT_object.remove(_object_menu)
    bpy.types.TOPBAR_MT_file_export.remove(_menu)
    for one in reversed(classes):
        bpy.utils.unregister_class(one)
