"""``oglc-bake`` -- the command that bakes the shipped circuit world, moved.

The world it baked is a *racing circuit* in hill country, and its options are
that world's options, so it now lives with the game's tools as
``glisteel-bake`` in `glisteel-editor
<https://github.com/mcfletch/glisteel-editor>`_::

    pip install glisteel-editor
    glisteel-bake --output /tmp/world

The baker itself did not move.
:func:`OpenGLContext_editor.bake.driver.bake_world` takes any layers at all, and
is what both that command and the editor's own *File -> Bake a world* call.

This name is kept for one release cycle so existing scripts say where to go
rather than failing as an unknown command.  It cannot run the bake itself:
openglcontext-editor does not depend on glisteel-editor, and a world-authoring
toolkit that required a particular game would be the wrong way round.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence

#: What to install, and what to type.  Named here so the notice and the
#: replacement cannot drift.
PACKAGE = 'glisteel-editor'
REPLACEMENT = 'glisteel-bake'


def main(argv: Sequence[str] | None = None) -> int:
    """Say where the command went, and fail, so a script notices.

    The options are handed back in the notice, because the replacement takes
    the same ones: the line that failed is the line to run.
    """
    arguments = sys.argv[1:] if argv is None else argv
    sys.stderr.write(
        "oglc-bake has moved to %s, which bakes the same circuit world with the "
        "same options: pip install %s, then %s%s\n"
        % (REPLACEMENT, PACKAGE, REPLACEMENT,
           ' ' + ' '.join(arguments) if arguments else ''))
    sys.stderr.flush()
    return 1


if __name__ == '__main__':                       # pragma: no cover - module entry
    raise SystemExit(main())
