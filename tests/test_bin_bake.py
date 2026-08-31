"""``oglc-bake`` says where the command went.

A name that has moved is worth more than an unknown command for one release:
a script that calls it gets told what to install and what to type, and a
non-zero status so it stops rather than carrying on over a world that was
never baked.
"""
from OpenGLContext_editor.bin import bake


class TestTheNoticeItLeaves:
    def test_it_fails_rather_than_pretending_to_bake(self, capsys) -> None:
        assert bake.main(['--output', '/tmp/world']) == 1
        assert not capsys.readouterr().out

    def test_it_names_what_to_install_and_what_to_type(self, capsys) -> None:
        bake.main([])
        said = capsys.readouterr().err
        assert bake.PACKAGE in said
        assert bake.REPLACEMENT in said

    def test_it_hands_back_the_arguments_it_was_given(self, capsys) -> None:
        """The replacement takes the same options, so the line can be reused."""
        bake.main(['--output', '/tmp/world', '--seed', '3'])
        assert '--output /tmp/world --seed 3' in capsys.readouterr().err
