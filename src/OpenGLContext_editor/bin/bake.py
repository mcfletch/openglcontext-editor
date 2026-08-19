"""``oglc-bake`` -- bake a world into a streamable 3D Tiles dataset.

With no arguments it bakes the shipped procedural world
(:mod:`OpenGLContext_editor.world.procedural`) into a directory and prints where
it put it::

    oglc-bake --output /tmp/world
    oglc-view /tmp/world/tileset.json

``--view`` runs the second line for you. Everything else on the command line
tunes the world's size, its tile detail, and how deep the tree refines.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import date
from typing import Any

from OpenGLContext_editor.bake.driver import bake_summary, bake_world
from OpenGLContext_editor.bake.manifest import (
    WorldManifest,
    carried,
    write_manifest,
)
from OpenGLContext_editor.world.procedural import ProceduralWorld

#: Where a bake goes when the command line does not say.
DEFAULT_OUTPUT = os.path.join('.', 'baked-world')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='oglc-bake', description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', '-o', default=DEFAULT_OUTPUT,
                        help='directory to write the tileset and its tiles into '
                             '(default: %(default)s)')
    parser.add_argument('--extent', type=float, default=4096.0,
                        help='the world is this many metres across (default: '
                             '%(default)s)')
    parser.add_argument('--depth', type=int, default=4,
                        help='how many times the tree subdivides; each level is '
                             'four times the tiles and twice the ground detail '
                             '(default: %(default)s)')
    parser.add_argument('--resolution', type=int, default=33,
                        help='ground samples across each tile (default: %(default)s)')
    parser.add_argument('--tree-density', type=float, default=None,
                        help='trees per square metre (default: the world\'s own)')
    parser.add_argument('--max-instances', type=int, default=None,
                        help='cap on instances written into any one tile')
    parser.add_argument('--ground', choices=('field', 'tiles'), default='field',
                        help="how the landscape is carried: one splat terrain "
                             "beside the tileset, or meshed into the tiles")
    parser.add_argument('--forest', choices=('field', 'tiles'), default='field',
                        help="how the trees are carried: one table beside the "
                             "tileset, or instanced into the tiles")
    parser.add_argument('--name', default=None,
                        help='what this world is called, for anything offering '
                             'a choice of them (default: the output '
                             "directory's own name)")
    parser.add_argument('--seed', type=int, default=11,
                        help='the world is the same every bake for a given seed '
                             '(default: %(default)s)')
    parser.add_argument('--force', action='store_true',
                        help='overwrite the output directory if it already holds '
                             'a baked world')
    parser.add_argument('--view', action='store_true',
                        help='open the baked world in oglc-view when it is done')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='say nothing but the tileset path')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    options = build_parser().parse_args(argv)
    _prepare_output(options.output, force=options.force)

    world = ProceduralWorld(extent=options.extent, resolution=options.resolution,
                            seed=options.seed, ground=options.ground,
                            forest=options.forest)
    if options.tree_density is not None:
        world.tree_density = options.tree_density

    progress = None if options.quiet else _printing_progress()
    # No `bounds`: the region to partition is whatever the layers cover, height
    # included. The world's own `bounds()` is a footprint, and a region with no
    # vertical extent would prune every cell the ground does not pass exactly
    # through.
    result = bake_world(world.layers(), options.output, depth=options.depth,
                        credits=world.credits(),
                        max_instances=options.max_instances,
                        progress=progress)
    manifest = describe(world, options, result)
    write_manifest(options.output, manifest)
    if options.quiet:
        print(result.tileset)
    else:
        print('\r' + ' ' * 40, end='\r')
        print(bake_summary(result))
        print('  world:       %s' % manifest.summary())
        print('\nView it with:\n  oglc-view %s' % result.tileset)
    if options.view:
        return _view(result.tileset)
    return 0


def describe(world: ProceduralWorld, options: Any, result: Any) -> WorldManifest:
    """What this bake produced, as the manifest written beside it.

    The name defaults to the output directory's own, tidied: a world baked into
    ``ashdown-forest`` is *Ashdown Forest* until somebody says otherwise, which
    beats making every world "Untitled" and beats asking for a name before one
    can be baked at all.
    """
    road = world.circuit()
    return WorldManifest(
        name=options.name or world_name(options.output),
        tileset=os.path.basename(result.tileset),
        seed=options.seed, extent=float(options.extent),
        road_length=float(road.length), structures=carried(road),
        closed=True, baked=date.today().isoformat())


def world_name(directory: str) -> str:
    """A readable name for a world baked into that directory."""
    stem = os.path.basename(os.path.abspath(directory))
    return stem.replace('-', ' ').replace('_', ' ').strip().title() or 'Untitled'


def _prepare_output(directory: str, force: bool) -> None:
    """Make the output directory, refusing to bake over a world unasked."""
    tileset = os.path.join(directory, 'tileset.json')
    if os.path.exists(tileset) and not force:
        raise SystemExit(
            "%s already holds a baked world; pass --force to replace it" % directory)
    os.makedirs(directory, exist_ok=True)


def _printing_progress() -> Callable[[int, int], None]:
    """A progress callback that keeps to one line."""
    def report(done: int, total: int) -> None:
        sys.stdout.write('\r  baking %d/%d nodes' % (done, total))
        sys.stdout.flush()
    return report


def _view(tileset: str) -> int:
    """Hand the baked world to the engine's viewer."""
    viewer = shutil.which('oglc-view')
    if viewer is None:
        print("oglc-view is not on PATH; install OpenGLContext's scripts to use "
              "--view", file=sys.stderr)
        return 1
    return subprocess.call([viewer, tileset])


if __name__ == '__main__':                       # pragma: no cover - module entry
    raise SystemExit(main())
