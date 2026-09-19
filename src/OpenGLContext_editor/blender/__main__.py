"""Put the Blender add-on where Blender can find it.

    python -m OpenGLContext_editor.blender --package        # write the zip
    python -m OpenGLContext_editor.blender --install        # copy it in place

``--package`` is the one an author wants: a zip carrying the add-on and nothing
else, which Blender installs through *Get Extensions > Install from Disk* or
``blender --command extension install-file``. ``--install`` copies it straight
into the user add-on directory, which saves a step while working on it.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from OpenGLContext_editor import blender


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='python -m OpenGLContext_editor.blender',
        description=__doc__.splitlines()[0])
    parser.add_argument('--package', action='store_true',
                        help='write the add-on as a zip Blender installs')
    parser.add_argument('--install', action='store_true',
                        help="copy it into this machine's Blender")
    parser.add_argument('--into', default=None,
                        help='where to write the zip, or which add-on '
                             'directory to install into')
    options = parser.parse_args(argv)
    if not (options.package or options.install):
        parser.error('nothing to do: pass --package or --install')
    if options.package:
        print(blender.package(options.into))
    if options.install:
        try:
            print(blender.install(into=options.into))
        except blender.BlenderMissing as error:
            parser.error('%s -- or pass --into an add-on directory' % (error,))
    return 0


if __name__ == '__main__':
    sys.exit(main())
