"""The distribution installs, and resolves against the engine it authors for.

Two things rest on this. A baker is only reachable from the demo and the editor
app if installing this package installs OpenGLContext alongside it, and "which
build produced this tileset" is only answerable while the metadata version and
the module version are the same number.
"""

import importlib.metadata
import re

import OpenGLContext_editor

DISTRIBUTION = "OpenGLContext-editor"

# PEP 440, restricted to the release/pre-release forms this project uses.
VERSION = re.compile(r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?$")


def test_module_version_is_pep440() -> None:
    assert VERSION.match(OpenGLContext_editor.__version__), (
        f"not a PEP 440 version: {OpenGLContext_editor.__version__!r}"
    )


def test_installed_metadata_reports_the_module_version() -> None:
    """The distribution's version is read from the module, not restated."""
    assert importlib.metadata.version(DISTRIBUTION) == OpenGLContext_editor.__version__


def test_the_engine_is_a_declared_dependency() -> None:
    """A bake runs against OpenGLContext, so installing this must install it."""
    requires = importlib.metadata.requires(DISTRIBUTION) or []
    names = {re.split(r"[\s<>=!;\[(]", r, maxsplit=1)[0].lower() for r in requires}
    assert "openglcontext" in names


def test_the_engine_imports() -> None:
    import OpenGLContext

    assert OpenGLContext.__version__
