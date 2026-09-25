"""Authoring levels of detail in Blender, and the add-on that exports them.

:mod:`openglcontext_lod` beside this module is a Blender add-on. It is a
directory Blender can consume as it stands: nothing in it imports this package,
and the only things it needs are Blender's own Python and the glTF exporter
Blender ships. That is deliberate -- an author installs it into Blender and
never installs this toolkit at all.

What it adds to Blender:

* Levels of detail on any object - one operator makes a chain from the
  selected mesh with Blender's own Decimate modifier, names the levels and
  records which chain they belong to.
* ``MSFT_lod`` on export - a glTF export extension turns those levels into
  the vendor extension OpenGLContext reads, so a normal *File > Export > glTF*
  produces a file that switches levels in a viewer and draws the finest in one
  that has never heard of the extension.
* The bust gallery - the demo world, built in Blender from this toolkit's
  own content, as a worked example of the above.

This module is the part outside Blender: where the add-on is, how to put it in
a Blender installation, and how to drive Blender from a script.

    from OpenGLContext_editor import blender

    blender.addon_directory()          # the directory Blender is given
    blender.install()                  # copy it into the user's Blender
    blender.run(['--python', script])  # drive a headless Blender

``docs/blender.md`` in the openglcontext-editor repository is the authoring
guide.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Sequence

try:
    import tomllib
except ImportError:                       # Python 3.10; tomli is the same reader
    import tomli as tomllib  # type: ignore[no-redef]

__all__ = [
    'ADDON',
    'BlenderMissing',
    'addon_directory',
    'addon_version',
    'executable',
    'install',
    'package',
    'run',
    'user_addon_directory',
    'version',
]

#: The add-on's module name, which is also its directory name. Blender imports
#: an add-on by the name of the directory it is installed under, so the two
#: cannot drift apart.
ADDON = 'openglcontext_lod'

#: The executable names searched for on ``PATH`` when neither a Blender nor
#: ``$BLENDER`` is given.
_SEARCH = ('blender',)


class BlenderMissing(EnvironmentError):
    """No Blender executable was found to run."""


def addon_directory() -> str:
    """The add-on, in this checkout."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), ADDON)


def executable(blender: str | None = None) -> str:
    """The Blender to drive: ``blender``, ``$BLENDER``, or one on the path."""
    for candidate in (blender, os.environ.get('BLENDER')) + _SEARCH:
        if candidate and shutil.which(candidate):
            return shutil.which(candidate)          # type: ignore[return-value]
        if candidate and os.path.exists(candidate):
            return candidate
    raise BlenderMissing(
        'no Blender found: pass one, set $BLENDER, or put `blender` on the path'
    )


def version(blender: str | None = None) -> str:
    """What ``blender --version`` says, first line only."""
    output = subprocess.run([executable(blender), '--version'],
                            capture_output=True, text=True, check=True)
    return output.stdout.strip().splitlines()[0]


def run(arguments: Sequence[str], blender: str | None = None,
        background: bool = True, check: bool = True,
        timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run Blender with ``arguments``, headless by default.

    Blender exits 0 on a Python traceback in ``--background``, so a caller that
    needs to know whether its script worked should have the script say so
    rather than trust the return code.
    """
    argv = [executable(blender)]
    if background:
        argv += ['--background', '--factory-startup']
    argv += list(arguments)
    return subprocess.run(argv, capture_output=True, text=True, check=check,
                          timeout=timeout)


def user_addon_directory(blender_version: str) -> str:
    """Where a user's own add-ons live for ``blender_version`` (``'4.5'``)."""
    if sys.platform == 'win32':
        base = os.path.join(os.environ.get('APPDATA', ''), 'Blender Foundation',
                            'Blender')
    elif sys.platform == 'darwin':
        base = os.path.expanduser('~/Library/Application Support/Blender')
    else:
        base = os.path.join(
            os.environ.get('XDG_CONFIG_HOME', os.path.expanduser('~/.config')),
            'blender')
    return os.path.join(base, blender_version, 'scripts', 'addons')


def install(blender_version: str | None = None,
            into: str | None = None) -> str:
    """Copy the add-on into a Blender installation; return where it landed.

    ``into`` names a directory directly; otherwise the user add-on directory
    for ``blender_version`` -- which defaults to what the Blender on the path
    reports. An existing copy is replaced.
    """
    if into is None:
        if blender_version is None:
            blender_version = '.'.join(version().split()[1].split('.')[:2])
        into = user_addon_directory(blender_version)
    target = os.path.join(into, ADDON)
    os.makedirs(into, exist_ok=True)
    if os.path.exists(target):
        shutil.rmtree(target)
    shutil.copytree(addon_directory(), target,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    return target


def addon_version() -> str:
    """What the add-on calls itself, read from its own manifest.

    From the manifest rather than from anything of this package's: the add-on
    is versioned as the thing Blender installs, and a zip named for a release
    of the toolkit around it would be naming the wrong thing.
    """
    manifest = os.path.join(addon_directory(), 'blender_manifest.toml')
    with open(manifest, 'rb') as handle:
        return str(tomllib.load(handle)['version'])


def package(into: str | None = None) -> str:
    """Write the add-on as a zip Blender installs on its own; return its path.

    This is how somebody who has never heard of this toolkit gets it: Blender's
    *Install from Disk*, or ``blender --command extension install-file``, over a
    file that carries the add-on and nothing else. ``MSFT_lod`` is a Khronos
    vendor extension rather than anything of ours, and an author exporting to
    some other engine should not have to install a renderer to write one.

    The zip holds the add-on directory at its root, which is the shape both
    Blender's legacy installer and its extension installer read.
    """
    into = os.path.abspath(into or os.getcwd())
    os.makedirs(into, exist_ok=True)
    path = os.path.join(into, '%s-%s.zip' % (ADDON, addon_version()))
    source = addon_directory()
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for root, directories, files in os.walk(source):
            directories[:] = [d for d in directories if d != '__pycache__']
            for leaf in sorted(files):
                if leaf.endswith(('.pyc', '.pyo')):
                    continue
                whole = os.path.join(root, leaf)
                archive.write(whole, os.path.join(
                    ADDON, os.path.relpath(whole, source)))
    return path
