"""Build the bust gallery world (``oglce-gallery``).

The demo world for level of detail: a hall of marble busts on plinths, each one
a six-level chain declared with ``MSFT_lod``, in a room with a polished parquet
floor, white plaster walls, dark beams across the top of them and the sky above
-- one generated panorama, which is the backdrop, the reflections and the light
that is not the sun.

One command covers the whole of it -- fetch the CC0 art, build the world in
Blender, export the glB the viewer opens:

    oglce-gallery --output gallery/gallery.glb

The art is CC0 and is fetched rather than committed: the bust from Poly Haven,
the surfaces from ambientCG. What was fetched and under what terms is written
to ``CREDITS.txt`` beside it, which is what the content pack's ``copyright``
is built from.

``--content-only`` stops after the fetch, and ``--blend`` also saves the
Blender file -- the one an author opens to see how the world is put together,
and the one the add-on's own operators were used to make.

Blender does the export. The levels are cut by Blender's Decimate modifier
and written by the add-on in
``OpenGLContext_editor/blender/openglcontext_lod``, so what this produces is
what an author gets from *File > Export > glTF 2.0* with the add-on installed,
and not a second path that might disagree with it.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from OpenGLContext import atomicfiles, userpaths
from OpenGLContext.loaders import cc0

from OpenGLContext_editor import blender
from OpenGLContext_editor.assets import polyhaven
from OpenGLContext_editor.blender.openglcontext_lod import content as recipes
from OpenGLContext_editor.blender.openglcontext_lod import gallery as layout
from OpenGLContext_editor.blender.openglcontext_lod import sky

__all__ = ['assemble', 'build', 'content_dir', 'main', 'roof']


def content_dir(directory: str | None = None) -> str:
    """Where the gallery's art is kept: per user, not in shared temp."""
    if directory:
        return os.path.abspath(directory)
    return os.path.join(userpaths.appdatadirectory(), 'OpenGLContext',
                        'gallery')


def assemble(into: str | None = None, resolution: str = '1k') -> str:
    """Fetch the bust and the surfaces into one directory; return it.

    Whatever is already there is left alone, so a second build costs nothing
    and works offline.
    """

    directory = content_dir(into)
    os.makedirs(directory, exist_ok=True)
    notices = []

    bust = polyhaven.fetch(recipes.BUST, directory, resolution=resolution)
    notices.append(bust.credit)

    for recipe in recipes.MATERIALS.values():
        target = os.path.join(directory, 'materials', recipe.asset)
        if os.path.isdir(target) and os.listdir(target):
            continue
        os.makedirs(target, exist_ok=True)
        for kind, path in cc0.material(recipe.asset,
                                       resolution.upper()).items():
            leaf = {'color': 'color.jpg', 'roughness': 'roughness.jpg',
                    'normal': 'normal.jpg'}.get(kind)
            if leaf:
                atomicfiles.copy_file(path, os.path.join(target, leaf))

    for recipe in recipes.MATERIALS.values():
        notices.append('%s -- CC0, https://ambientcg.com/view?id=%s'
                       % (recipe.asset, recipe.asset))
    _write_credits(directory, notices)
    return directory


def _write_credits(directory: str, notices: Sequence[str]) -> None:
    """What the world is made of and under what terms.

    A pack that cannot state its terms is a pack that would be redistributed
    unattributed, so this is written from what was actually fetched rather than
    kept as a list somebody has to remember to update.
    """
    seen: list[str] = []
    for notice in notices:
        if notice and notice not in seen:
            seen.append(notice)
    atomicfiles.write_text(os.path.join(directory, 'CREDITS.txt'),
                           'The bust gallery is built from public-domain art.\n\n'
                           + '\n'.join(seen) + '\n')


def build(content: str, output: str, bays: int = 30, levels: int = 6,
          ratio: float = 0.5, coverage: Sequence[float] | None = None,
          blend: str | None = None, blender_binary: str | None = None,
          impostor: int = 0, impostor_image: int = 256,
          timeout: float = 3600.0) -> str:
    """The glB Blender wrote from the content directory, with its sky.

    Blender exits 0 after a traceback in ``--background``, so the build is
    done only when ``build.py`` reports ``WROTE:`` for ``output``; a file an
    earlier run left there is removed first, so it cannot pass for this one.
    """
    if os.path.exists(output):
        os.remove(output)
    script = os.path.join(blender.addon_directory(), 'build.py')
    argv = ['-P', script, '--',
            '--content', content, '--output', os.path.abspath(output),
            '--bays', str(bays), '--levels', str(levels), '--ratio', str(ratio),
            '--impostor', str(impostor),
            '--impostor-image', str(impostor_image)]
    if coverage:
        argv += ['--coverage', ','.join('%g' % value for value in coverage)]
    if blend:
        argv += ['--blend', os.path.abspath(blend)]
    done = blender.run(argv, blender=blender_binary, check=False,
                       timeout=timeout)
    wrote = False
    for line in done.stdout.splitlines():
        if line.startswith(('BUILT:', 'WROTE:', 'FAILED:')):
            print(line)
        if line.startswith('WROTE: %s ' % (os.path.abspath(output),)):
            wrote = True
    if done.returncode or not wrote or not os.path.exists(output):
        sys.stderr.write(done.stdout[-4000:])
        sys.stderr.write(done.stderr[-4000:])
        raise SystemExit('Blender did not write %s' % (output,))
    roof(output)
    return output


def roof(path: str, plan: layout.Gallery | None = None) -> str:
    """Put the sky in the world Blender wrote; the path again.

    Blender's exporter writes materials, meshes and lights, and has nothing to
    say about a document-level environment -- so the panorama is added here,
    where the extension is a few lines of JSON rather than an export hook. What
    it adds is an ordinary image, texture and sampler plus
    ``OMI_environment_sky``, so a reader that never heard of the extension
    opens the same world with no sky in it.
    """
    plan = plan or layout.Gallery()
    with open(path, 'rb') as handle:
        blob = handle.read()
    panorama = sky.panorama(sun_azimuth=plan.sun_azimuth(),
                            sun_elevation=plan.sun_elevation())
    written = sky.with_sky(blob, panorama)
    atomicfiles.write_bytes(path, written)
    print('SKY: %dx%d panorama, %+.1f KB'
          % (panorama.width, panorama.height,
             (len(written) - len(blob)) / 1024.0))
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='oglce-gallery', description=__doc__.split('\n\n')[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', default='gallery/gallery.glb',
                        help='the glB to write (default: %(default)s)')
    parser.add_argument('--content', default=None,
                        help='where the art is kept (default: the per-user '
                             'app-data directory)')
    parser.add_argument('--content-only', action='store_true',
                        help='fetch the art and stop')
    parser.add_argument('--blend', default=None,
                        help='also save the Blender file here')
    parser.add_argument('--bays', type=int, default=30,
                        help='plinths down the hall (default: %(default)s)')
    parser.add_argument('--levels', type=int, default=6,
                        help='levels per bust (default: %(default)s)')
    parser.add_argument('--ratio', type=float, default=0.5,
                        help='triangles a level keeps of the one before it')
    parser.add_argument('--coverage', default=None,
                        help='measured switching thresholds, comma separated, '
                             'finest first')
    parser.add_argument('--impostor', type=int, default=0, metavar='VIEWS',
                        help='make the coarsest level an octahedral impostor '
                             'with this many baked views a side (8 is a good '
                             'start); 0 leaves the chain all meshes')
    parser.add_argument('--impostor-image', type=int, default=256,
                        help='pixels across the impostor atlas (default: '
                             '%(default)s)')
    parser.add_argument('--blender', default=None,
                        help='which Blender to use (default: $BLENDER, or one '
                             'on the path)')
    options = parser.parse_args(argv)

    content = assemble(options.content)
    print('CONTENT: %s' % (content,))
    if options.content_only:
        return 0

    try:
        print('BLENDER: %s' % (blender.version(options.blender),))
    except blender.BlenderMissing as error:
        parser.error(str(error))

    coverage = ([float(value) for value in options.coverage.split(',')]
                if options.coverage else None)
    build(content, options.output, bays=options.bays, levels=options.levels,
          ratio=options.ratio, coverage=coverage, blend=options.blend,
          blender_binary=options.blender, impostor=options.impostor,
          impostor_image=options.impostor_image)
    print(layout.describe(layout.Gallery(bays=options.bays)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
