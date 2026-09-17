"""``oglc-bake-plants`` -- published plant models into ground cover.

    oglc-bake-plants --out assets fern_02 shrub_04=0.15 grass_medium_01=2.4

Each argument names an asset to bake, optionally with how densely it should
grow, in plants per square metre. What comes out is one ``.glb`` per asset
holding every variant of the plant at two levels of detail, a billboard card for
each, a ``cover.json`` naming the species, and a ``CREDITS.txt``.

``cover.json`` is what a world bake or a demo reads to know what it may grow;
see :mod:`OpenGLContext_editor.assets.plants`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence

from OpenGLContext_editor.assets import plants, polyhaven


def _wanted(argument: str) -> tuple[str, float | None]:
    """``slug`` or ``slug=density``."""
    if '=' not in argument:
        return argument, None
    slug, _, density = argument.partition('=')
    try:
        return slug, float(density)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "%r: a density is plants per square metre, as a number"
            % (argument,)) from None


def build_arg_parser() -> argparse.ArgumentParser:
    """The command's options."""
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('assets', nargs='+', type=_wanted, metavar='SLUG[=N]',
                        help='Poly Haven asset ids, with an optional density '
                             'in plants per square metre')
    parser.add_argument('--out', default='cover-assets',
                        help='where the baked assets are written')
    parser.add_argument('--downloads', default=None,
                        help='where fetched sources are cached (default: the '
                             'shared per-user cache, so nothing is downloaded '
                             'twice)')
    parser.add_argument('--resolution', default='1k',
                        help='texture resolution to fetch (1k, 2k, 4k)')
    parser.add_argument('--near', type=int, default=plants.NEAR_TRIANGLES,
                        help='triangle budget for the close-up geometry')
    parser.add_argument('--far', type=int, default=plants.FAR_TRIANGLES,
                        help='triangle budget for the geometry beyond that')
    parser.add_argument('--per-asset', type=int, default=None, metavar='N',
                        help='keep at most N plants from each file, fullest '
                             'first (a published grass can be seventeen tufts)')
    parser.add_argument('--patchiness', type=float, default=0.0, metavar='P',
                        help='0 spreads a plant evenly, 1 gathers it into beds '
                             'with bare ground between')
    parser.add_argument('--patch-metres', type=float, default=None,
                        metavar='M', help='roughly how far across one bed is')
    parser.add_argument('--canopy', type=float, nargs=2, default=None,
                        metavar=('DIM', 'BRIGHT'),
                        help='the band of canopy light this plant grows in, '
                             '0 under a closed canopy to 1 in the open')
    parser.add_argument('--density', type=float, default=0.35,
                        help='density for any asset that names none')
    parser.add_argument('--no-cards', action='store_true',
                        help='skip the billboards, which need a GL context')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Fetch and bake every named asset; write what grew."""
    options = build_arg_parser().parse_args(
        sys.argv[1:] if argv is None else list(argv))
    grown = []
    for slug, density in options.assets:
        try:
            source = polyhaven.fetch(slug, options.downloads,
                                     resolution=options.resolution).source()
        except LookupError as error:
            sys.stderr.write('%s: %s\n' % (slug, error))
            return 1
        species = plants.bake(
            source, options.out, rungs=(options.near, options.far),
            keep=options.per_asset, card=not options.no_cards,
            density=options.density if density is None else density,
            patchiness=options.patchiness,
            patch_metres=options.patch_metres,
            canopy=None if options.canopy is None else tuple(options.canopy))
        grown.extend(species)
        print('%-22s %d variant%s, %s' % (
            slug, len(species), '' if len(species) == 1 else 's',
            ', '.join('%s %.2fm' % (one.name, one.height) for one in species)))
    manifest = _write(options.out, grown)
    print('%d species -> %s' % (len(grown), manifest))
    return 0


def _write(directory: str, grown: list) -> str:
    """Add ``grown`` to the directory's ``cover.json``, by name.

    Merged rather than replaced, because plants do not all want the same
    settings: a shrub of a hundred and fifty thousand triangles needs a budget
    a grass tuft does not, so a set is built up over several runs into one
    directory. A species baked again replaces the entry it had.
    """
    manifest = os.path.join(directory, 'cover.json')
    named = {}
    if os.path.exists(manifest):
        with open(manifest, encoding='utf-8') as handle:
            for entry in json.load(handle).get('species', ()):
                named[entry['name']] = entry
    for one in grown:
        named[one.name] = one.to_json()
    with open(manifest, 'w', encoding='utf-8') as handle:
        json.dump({'species': [named[key] for key in sorted(named)]}, handle,
                  indent=1)
    return manifest


if __name__ == '__main__':
    raise SystemExit(main())
