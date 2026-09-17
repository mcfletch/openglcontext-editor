"""A world's water level is the one the world was given.

``water_level`` raises the sea until the valleys flood, which is what turns hill
country into an archipelago and puts the circuit on causeways between the
islands. Everything that has to know where the water is asks the world: the
forest grows above it, the boulders lie above it, the shore is painted at it,
the circuit is held clear of it, and the fill carried over drowned ground is
labelled a causeway.

A world that raised the sea and left those reading a default would be one where
the whole of the flooded ground still carries trees, and where the road runs
along the sea floor with nothing built to carry it.
"""
import numpy as np
import pytest

from OpenGLContext_editor.world.procedural import (
    CAUSEWAY_FREEBOARD,
    ProceduralWorld,
)
from OpenGLContext_editor.world.structures import Op

#: Where Tidewater's sea sits, in metres: high enough that most of the shipped
#: landscape is under it, which is what makes the question worth asking.
FLOODED = 48.0


@pytest.fixture(scope='module')
def flooded() -> ProceduralWorld:
    """The shipped landscape with its valleys under the sea."""
    return ProceduralWorld(seed=23, water_level=FLOODED, maximum_bank=0.02,
                           variety=0.7)


class TestTheForest:
    def test_no_tree_stands_under_the_sea(self, flooded) -> None:
        heights = flooded.scatter().positions[:, 1]
        assert len(heights), 'the world grew no trees at all'
        assert float(heights.min()) > FLOODED

    def test_and_the_treeline_starts_at_the_waterline(self, flooded) -> None:
        assert flooded.treeline()[0] > FLOODED

    def test_a_world_at_sea_level_keeps_the_forest_it_had(self) -> None:
        """The default is a world whose water is at zero, and raising the sea
        is the change: nothing here moves the ordinary world's treeline."""
        assert ProceduralWorld().treeline()[0] == pytest.approx(1.0)


class TestTheGround:
    def test_no_boulder_lies_under_the_sea(self, flooded) -> None:
        lying = flooded.rocks().positions
        assert len(lying), 'the world laid no boulders at all'
        assert float(np.asarray(lying)[:, 1].min()) > FLOODED

    def test_the_shore_is_painted_at_the_waterline(self, flooded) -> None:
        """The soft layers stop at the shore, which is what keeps the ground
        cover out of the sea: nothing grows on what is left below it."""
        lowest = min(float(rule.height[0])
                     for rule in flooded.field_terrain().rules
                     if rule.height is not None and rule.height[0] > -1.0e8)
        assert lowest > FLOODED


class TestTheCircuit:
    def test_the_road_is_held_clear_of_the_sea(self, flooded) -> None:
        heights = flooded.circuit().points[:, 1]
        assert float(heights.min()) == pytest.approx(
            FLOODED + CAUSEWAY_FREEBOARD, abs=1e-6)

    def test_the_drowned_ground_it_crosses_is_carried_on_causeways(
            self, flooded) -> None:
        """Fill standing on the sea bed is a causeway, and a world that
        measured against the wrong waterline built none of them."""
        circuit = flooded.circuit()
        natural = np.asarray(flooded.natural()(circuit.points[:, 0],
                                               circuit.points[:, 2]), 'd')
        drowned = natural < FLOODED
        assert drowned.any(), 'the sea flooded none of the ground'
        kinds = circuit.ops
        carried = np.array([kind is not Op.DIRT for kind in kinds])
        assert carried[drowned].mean() > 0.9, (
            '%.0f%% of the road over drowned ground is laid on the sea bed'
            % (100.0 * (1.0 - carried[drowned].mean()),))


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
