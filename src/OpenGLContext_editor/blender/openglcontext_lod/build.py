"""Build the bust gallery and export it, inside a headless Blender.

Run as Blender's own script, which is how the world is made from a command
rather than by hand:

    blender --background --factory-startup -P build.py -- \\
        --content ~/.cache/openglcontext/gallery --output gallery.glb

The add-on is *enabled* rather than merely imported, because Blender's glTF
exporter looks for export extensions among the enabled add-ons -- importing the
module is not enough to have the LODs written, and an export that quietly
omitted them would look like a working one.

``--blend`` also saves the Blender file, which is what an author opens to see
how the world is put together.
"""

from __future__ import annotations

import argparse
import os
import sys


def _bootstrap() -> None:
    """Make ``openglcontext_lod`` importable when run as a loose script."""
    if __package__:
        return
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    if parent not in sys.path:
        sys.path.insert(0, parent)


def arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='blender -b -P build.py --',
        description='Build the bust gallery world and export it as glB.')
    parser.add_argument('--content', required=True,
                        help='the directory holding the bust and the CC0 '
                             'materials')
    parser.add_argument('--output', required=True, help='the glB to write')
    parser.add_argument('--blend', default=None,
                        help='also save the Blender file here')
    parser.add_argument('--bays', type=int, default=30,
                        help='how many plinths there are down the hall')
    parser.add_argument('--levels', type=int, default=6,
                        help='how many levels each bust is given')
    parser.add_argument('--max-triangles', type=int, default=0,
                        help='triangle budget for each bust\'s finest level; '
                             '0 leaves the model as it came')
    parser.add_argument('--ratio', type=float, default=0.5,
                        help='what share of the triangles a level keeps of '
                             'the one before it')
    parser.add_argument('--coverage', default=None,
                        help='the thresholds the levels switch at, comma '
                             'separated, finest first; measured figures where '
                             'you have them. Without it the levels halve')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _bootstrap()
    import bpy
    from openglcontext_lod import content, gallery, scene

    if argv is None:
        argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    options = arguments(argv)

    bpy.ops.preferences.addon_enable(module='openglcontext_lod')
    if 'openglcontext_lod' not in bpy.context.preferences.addons.keys():
        print('FAILED: the add-on would not enable, so no LODs would be '
              'written', file=sys.stderr)
        return 1

    plan = gallery.Gallery(bays=options.bays)
    coverage = ([float(value) for value in options.coverage.split(',')]
                if options.coverage else None)
    counted = scene.build_gallery(
        plan, *content.gallery_content(options.content),
        levels=options.levels, ratio=options.ratio, coverage=coverage,
        max_triangles=options.max_triangles or None)
    print('BUILT: %s' % (gallery.describe(plan),))
    print('BUILT: %(slabs)d slabs, %(plinths)d plinths, %(busts)d busts, '
          '%(levels)d level objects, %(lights)d lights' % counted)

    output = os.path.abspath(options.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    scene.export(output)
    print('WROTE: %s (%d bytes)' % (output, os.path.getsize(output)))

    if options.blend:
        blend = os.path.abspath(options.blend)
        os.makedirs(os.path.dirname(blend), exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=blend)
        print('WROTE: %s' % (blend,))
    return 0


if __name__ == '__main__':
    sys.exit(main())
