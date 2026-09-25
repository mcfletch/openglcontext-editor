"""The example world's species: what it plants, and that a caller's changes stay
the caller's.

A species is a scenegraph node, and a node's fields can be set. The example
world's species are made afresh for each caller, so one that retunes a tree
for its own world does not retune every world after it.
"""
from OpenGLContext_editor.world import species


class TestTheExampleWorldsSpecies:
    def test_four_kinds_of_tree(self) -> None:
        assert [one.name for one in species.shipped_trees()] == [
            'fir', 'noel', 'maple0', 'maple2']

    def test_a_change_to_one_is_not_a_change_to_the_next(self) -> None:
        mine = species.shipped_trees()
        mine[0].cardWidth = 9.0
        assert species.shipped_trees()[0].cardWidth == 0.50

    def test_nor_to_the_default_cover(self) -> None:
        mine = species.default_cover()
        mine.density = 99.0
        assert species.default_cover().density == 1.6

    def test_varied_is_how_to_make_one_s_own(self) -> None:
        taller = species.default_cover().varied(height=1.2)
        assert taller.height == 1.2
        assert species.default_cover().height == 0.5
