"""The settings a world is described by, declared by the world itself.

A recipe or a command line that bakes a ``ProceduralWorld`` offers its
parameters; the world declares which, of what kind and what each means, so a
parameter added to the world is offered wherever the table is read.
"""
import dataclasses
import typing

from OpenGLContext_editor.world.procedural import ProceduralWorld, WorldSetting


def _fields():
    hints = typing.get_type_hints(ProceduralWorld)
    return {one.name: (one, hints[one.name]) for one in dataclasses.fields(ProceduralWorld)}


class TestTheTable:
    def test_every_setting_is_a_field_of_the_world(self) -> None:
        fields = _fields()
        for setting in ProceduralWorld.SETTINGS:
            assert setting.name in fields, setting.name

    def test_each_is_of_its_field_s_kind(self) -> None:
        fields = _fields()
        for setting in ProceduralWorld.SETTINGS:
            _field, hint = fields[setting.name]
            if setting.choices is not None:
                assert set(setting.choices) == set(typing.get_args(hint)) or \
                    all(isinstance(one, str) for one in setting.choices)
            else:
                assert hint is setting.kind, setting.name

    def test_each_says_what_it_is(self) -> None:
        for setting in ProceduralWorld.SETTINGS:
            assert isinstance(setting, WorldSetting)
            assert setting.help and not setting.help.endswith('.')

    def test_the_names_are_unique(self) -> None:
        names = [one.name for one in ProceduralWorld.SETTINGS]
        assert len(names) == len(set(names))
