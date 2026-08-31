"""Naming a world after the directory it was baked into.

The manifest format itself is the engine's
(:mod:`OpenGLContext.loaders.tiles3d.manifest`); what is here is the one thing
the baker decides about it.
"""
import pytest

from OpenGLContext_editor.bake.manifest import world_name


class TestNamingAWorldAfterItsDirectory:
    """A world baked into ``ashdown-forest`` is Ashdown Forest until somebody
    says otherwise, which beats calling every world Untitled and beats refusing
    to bake one without a name."""

    def test_a_hyphenated_directory_becomes_a_title(self):
        assert world_name('/tmp/ashdown-forest') == 'Ashdown Forest'

    def test_underscores_do_the_same(self):
        assert world_name('/tmp/my_track') == 'My Track'

    def test_a_trailing_separator_is_not_the_name(self):
        assert world_name('/tmp/my_track/') == 'My Track'

    def test_a_relative_directory_is_resolved(self):
        assert world_name('./baked-world') == 'Baked World'

    def test_a_directory_that_names_nothing_is_untitled(self):
        assert world_name('/') == 'Untitled'


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
