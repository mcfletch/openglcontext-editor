"""The ground a world is built on: a base, and the edits made to it.

Pure arithmetic over ``(x, z)``: no GL, no tiles, no files.
"""
import json

import numpy as np
import pytest
from OpenGLContext.loaders.tiles3d.procedural import terrain_height

from OpenGLContext_editor.world.height import (
    HeightEdit,
    HeightSource,
    ProceduralBase,
    edit_from_json,
    register_edit,
)


def _grid(half=400.0, steps=17):
    axis = np.linspace(-half, half, steps)
    return np.meshgrid(axis, axis, indexing='ij')


class Lift(HeightEdit):
    """A square of ground raised by a fixed amount, for the tests to compose."""

    KIND = 'test-lift'

    def __init__(self, centre=(0.0, 0.0), radius=100.0, amount=10.0):
        self.centre = (float(centre[0]), float(centre[1]))
        self.radius = float(radius)
        self.amount = float(amount)

    def bounds(self):
        x, z = self.centre
        return (x - self.radius, z - self.radius,
                x + self.radius, z + self.radius)

    def delta(self, x, z, height):
        return np.full(np.shape(x), self.amount)

    def to_json(self):
        return {'kind': self.KIND, 'centre': list(self.centre),
                'radius': self.radius, 'amount': self.amount}

    @classmethod
    def from_json(cls, document):
        return cls(centre=document['centre'], radius=document['radius'],
                   amount=document['amount'])


register_edit(Lift)


class TestTheBase:
    def test_the_procedural_base_is_the_shipped_landscape(self) -> None:
        x, z = _grid()
        source = HeightSource(base=ProceduralBase(relief=1.0))
        assert np.allclose(source.height_fn()(x, z), terrain_height(x, z))

    def test_relief_scales_the_whole_landscape(self) -> None:
        x, z = _grid()
        source = HeightSource(base=ProceduralBase(relief=0.5))
        assert np.allclose(source.height_fn()(x, z),
                           terrain_height(x, z) * 0.5)

    def test_the_answer_is_the_shape_of_the_question(self) -> None:
        source = HeightSource(base=ProceduralBase())
        x, z = _grid(steps=5)
        assert source.height_fn()(x, z).shape == x.shape

    def test_a_single_point_answers_too(self) -> None:
        source = HeightSource(base=ProceduralBase())
        value = source.height_fn()(np.asarray([0.0]), np.asarray([0.0]))
        assert np.shape(value) == (1,)


class TestAnEditOnTop:
    def _source(self, *edits):
        return HeightSource(base=ProceduralBase(relief=1.0), edits=list(edits))

    def test_an_edit_raises_its_own_region(self) -> None:
        x, z = _grid()
        raised = self._source(Lift(centre=(0.0, 0.0), radius=100.0, amount=25.0))
        inside = (np.abs(x) <= 100.0) & (np.abs(z) <= 100.0)
        difference = raised.height_fn()(x, z) - terrain_height(x, z)
        assert np.allclose(difference[inside], 25.0)

    def test_it_leaves_everything_else_exactly_as_it_was(self) -> None:
        x, z = _grid()
        raised = self._source(Lift(centre=(0.0, 0.0), radius=100.0, amount=25.0))
        inside = (np.abs(x) <= 100.0) & (np.abs(z) <= 100.0)
        difference = raised.height_fn()(x, z) - terrain_height(x, z)
        assert np.all(difference[~inside] == 0.0)

    def test_edits_compose_in_the_order_they_were_made(self) -> None:
        x, z = _grid()
        source = self._source(Lift(radius=100.0, amount=10.0),
                              Lift(radius=50.0, amount=4.0))
        difference = source.height_fn()(x, z) - terrain_height(x, z)
        near = (np.abs(x) <= 50.0) & (np.abs(z) <= 50.0)
        ring = ((np.abs(x) <= 100.0) & (np.abs(z) <= 100.0)) & ~near
        assert np.allclose(difference[near], 14.0)
        assert np.allclose(difference[ring], 10.0)

    def test_an_edit_that_reaches_nothing_costs_nothing(self) -> None:
        """Each edit carries its rectangle, so a bake skips the ones it misses."""
        x, z = _grid(half=100.0)
        asked = []

        class Counting(Lift):
            def delta(self, x, z, height):
                asked.append(len(np.ravel(x)))
                return super().delta(x, z, height)

        source = self._source(Counting(centre=(10000.0, 0.0), radius=10.0))
        source.height_fn()(x, z)
        assert asked == []

    def test_an_edit_reads_the_ground_under_it(self) -> None:
        """So a stroke can lift a hill rather than replace it."""
        seen = {}

        class Reading(Lift):
            def delta(self, x, z, height):
                seen['under'] = np.asarray(height).copy()
                return np.zeros(np.shape(x))

        x, z = _grid(half=50.0, steps=5)
        self._source(Reading(radius=1000.0)).height_fn()(x, z)
        assert np.allclose(np.sort(seen['under']),
                           np.sort(terrain_height(x, z).ravel()))


class TestTheFile:
    def test_a_source_round_trips_through_json(self) -> None:
        x, z = _grid()
        source = HeightSource(base=ProceduralBase(relief=0.4),
                              edits=[Lift(centre=(30.0, -20.0), radius=80.0,
                                          amount=12.0)])
        again = HeightSource.from_json(json.loads(json.dumps(source.to_json())))
        assert np.allclose(again.height_fn()(x, z), source.height_fn()(x, z))

    def test_an_empty_stack_round_trips(self) -> None:
        source = HeightSource(base=ProceduralBase())
        again = HeightSource.from_json(source.to_json())
        assert again.edits == []

    def test_an_edit_kind_nobody_knows_is_refused(self) -> None:
        """Rather than dropped: a designer's work is not silently discarded."""
        with pytest.raises(ValueError):
            edit_from_json({'kind': 'a-kind-from-the-future'})

    def test_a_base_kind_nobody_knows_is_refused(self) -> None:
        with pytest.raises(ValueError):
            HeightSource.from_json({'base': {'kind': 'martian'}})

    def test_the_default_source_is_the_shipped_landscape(self) -> None:
        """A project written before there was a source block reads as this."""
        source = HeightSource.from_json({})
        assert isinstance(source.base, ProceduralBase)
        assert source.edits == []


class TestTheWorldItMakes:
    """A world takes a source and builds its ground from it."""

    def _world(self, **named):
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        return ProceduralWorld(extent=512.0, road=False, tree_density=0.0,
                               **named)

    def test_a_world_with_no_source_is_its_own_relief(self) -> None:
        x, z = _grid(half=200.0, steps=9)
        world = self._world(relief=0.5)
        assert np.allclose(world.natural()(x, z), terrain_height(x, z) * 0.5)

    def test_a_source_decides_the_ground(self) -> None:
        x, z = _grid(half=200.0, steps=9)
        world = self._world(source=HeightSource(base=ProceduralBase(relief=0.25)))
        assert np.allclose(world.natural()(x, z), terrain_height(x, z) * 0.25)

    def test_an_edit_reaches_the_world(self) -> None:
        x, z = _grid(half=200.0, steps=9)
        world = self._world(source=HeightSource(
            base=ProceduralBase(relief=1.0),
            edits=[Lift(centre=(0.0, 0.0), radius=1000.0, amount=7.0)]))
        assert np.allclose(world.natural()(x, z), terrain_height(x, z) + 7.0)

    def test_the_treeline_still_follows_the_relief_it_was_given(self) -> None:
        world = self._world(relief=0.5)
        assert world.treeline()[1] == pytest.approx(
            self._world(relief=1.0).treeline()[1] * 0.5)
