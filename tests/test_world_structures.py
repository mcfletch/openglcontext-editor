"""Choosing what to build where a road leaves the ground.

An alignment is settled for grade and for the speed it is driven at, and doing
that puts it above the land in some places and below it in others. Small
departures are earthworks -- a cutting or an embankment, which the ground itself
absorbs. Large ones are not: a road ninety metres above a valley floor is a
viaduct and a road ninety metres inside a mountain is a tunnel, and building
either as a heap of soil would bury the landscape it crosses.

What is tested here is the choice: a height profile in, a sequence of operations
out. No geometry and no GL -- the meshes are :mod:`OpenGLContext.scenegraph.roadworks`
and are tested there.
"""
import numpy as np
import pytest

from OpenGLContext_editor.world.structures import (
    APPROACH_LIMIT,
    CUTTING_LIMIT,
    EMBANKMENT_LIMIT,
    MINIMUM_SPAN,
    MINIMUM_TUNNEL,
    Op,
    Structure,
    choose_structures,
)

SPACING = 10.0


def _line(departures, spacing=SPACING):
    """An alignment running east at ``spacing``, and the ground under it.

    ``departures`` is how far the road is above the ground at each point; the
    ground itself is level, because what the choice reads is the difference.
    """
    departures = np.asarray(departures, dtype='d')
    x = np.arange(len(departures)) * spacing
    line = np.stack([x, departures, np.zeros(len(departures))], axis=-1)
    return line, np.zeros(len(departures))


def _kinds(structures):
    return [s.kind for s in structures]


def _flat(count, value=0.0):
    return [value] * count


class TestARoadOnTheGround:
    def test_a_road_on_the_ground_is_all_dirt(self) -> None:
        line, natural = _line(_flat(40))
        assert _kinds(choose_structures(line, natural)) == [Op.DIRT]

    def test_the_one_run_covers_every_point(self) -> None:
        line, natural = _line(_flat(40))
        only = choose_structures(line, natural)[0]
        assert (only.first, only.last) == (0, 39)

    def test_a_shallow_cutting_is_still_dirt(self) -> None:
        line, natural = _line(_flat(40, -CUTTING_LIMIT * 0.5))
        assert _kinds(choose_structures(line, natural)) == [Op.DIRT]

    def test_a_shallow_embankment_is_still_dirt(self) -> None:
        line, natural = _line(_flat(40, EMBANKMENT_LIMIT * 0.5))
        assert _kinds(choose_structures(line, natural)) == [Op.DIRT]

    def test_every_point_belongs_to_exactly_one_operation(self) -> None:
        line, natural = _line([0.0] * 10 + [-40.0] * 20 + [0.0] * 10)
        covered = np.zeros(40, dtype=int)
        for structure in choose_structures(line, natural):
            covered[structure.first:structure.last + 1] += 1
        assert list(np.unique(covered)) == [1]


class TestAMountainInTheWay:
    def _through(self, depth=-40.0, points=40):
        return _line(_flat(10) + _flat(points, depth) + _flat(10))

    def test_a_deep_run_below_the_ground_is_a_tunnel(self) -> None:
        line, natural = self._through()
        assert Op.TUNNEL in _kinds(choose_structures(line, natural))

    def test_it_is_bored_where_the_road_is_buried(self) -> None:
        line, natural = self._through()
        tunnel = next(s for s in choose_structures(line, natural)
                      if s.kind is Op.TUNNEL)
        assert tunnel.first <= 10 and tunnel.last >= 49

    def test_the_road_arrives_and_leaves_on_dirt(self) -> None:
        line, natural = self._through()
        assert _kinds(choose_structures(line, natural)) == [
            Op.DIRT, Op.TUNNEL, Op.DIRT]

    def test_a_short_dip_below_the_ground_is_a_cutting(self) -> None:
        """Shorter than a tunnel is worth: the machine digs it."""
        short = int(MINIMUM_TUNNEL / SPACING) - 2
        line, natural = _line(_flat(10) + _flat(short, -40.0) + _flat(10))
        assert _kinds(choose_structures(line, natural)) == [Op.DIRT]

    def test_a_long_shallow_dip_is_a_cutting(self) -> None:
        line, natural = _line(_flat(10) + _flat(40, -CUTTING_LIMIT * 0.9)
                              + _flat(10))
        assert _kinds(choose_structures(line, natural)) == [Op.DIRT]

    def test_two_mountains_are_two_tunnels(self) -> None:
        line, natural = _line(_flat(10) + _flat(20, -40.0) + _flat(30)
                              + _flat(20, -40.0) + _flat(10))
        assert _kinds(choose_structures(line, natural)) == [
            Op.DIRT, Op.TUNNEL, Op.DIRT, Op.TUNNEL, Op.DIRT]


class TestAValleyToCross:
    def _over(self, height=40.0, points=40):
        return _line(_flat(10) + _flat(points, height) + _flat(10))

    def test_a_high_run_above_the_ground_is_a_bridge(self) -> None:
        line, natural = self._over()
        assert Op.BRIDGE in _kinds(choose_structures(line, natural))

    def test_the_road_arrives_and_leaves_on_dirt(self) -> None:
        line, natural = self._over()
        assert _kinds(choose_structures(line, natural)) == [
            Op.DIRT, Op.BRIDGE, Op.DIRT]

    def test_a_short_hop_is_filled_rather_than_spanned(self) -> None:
        short = int(MINIMUM_SPAN / SPACING) - 2
        line, natural = _line(_flat(10) + _flat(short, 40.0) + _flat(10))
        assert _kinds(choose_structures(line, natural)) == [Op.DIRT]

    def test_a_long_low_rise_is_an_embankment(self) -> None:
        line, natural = _line(_flat(10) + _flat(40, EMBANKMENT_LIMIT * 0.9)
                              + _flat(10))
        assert _kinds(choose_structures(line, natural)) == [Op.DIRT]


class TestReachingTheGround:
    """A deck lands on an abutment and a bore opens at a portal, so a structure
    runs out to where the road meets the land rather than stopping at the
    threshold that chose it."""

    def test_a_bridge_reaches_out_towards_the_ground(self) -> None:
        ramp = list(np.linspace(0.0, 40.0, 20))
        line, natural = _line(ramp + _flat(20, 40.0) + ramp[::-1])
        bridge = next(s for s in choose_structures(line, natural)
                      if s.kind is Op.BRIDGE)
        # The threshold is crossed part way up the ramp; the abutment is lower.
        crossing = int(np.searchsorted(ramp, EMBANKMENT_LIMIT))
        assert bridge.first < crossing

    def test_it_does_not_reach_further_than_the_approach_limit(self) -> None:
        ramp = list(np.linspace(0.0, 40.0, 200))
        line, natural = _line(ramp + _flat(20, 40.0) + ramp[::-1])
        bridge = next(s for s in choose_structures(line, natural)
                      if s.kind is Op.BRIDGE)
        crossing = int(np.searchsorted(ramp, EMBANKMENT_LIMIT))
        assert (crossing - bridge.first) * SPACING <= APPROACH_LIMIT + SPACING

    def test_a_tunnel_reaches_out_towards_its_portals(self) -> None:
        ramp = list(np.linspace(0.0, -40.0, 20))
        line, natural = _line(ramp + _flat(20, -40.0) + ramp[::-1])
        tunnel = next(s for s in choose_structures(line, natural)
                      if s.kind is Op.TUNNEL)
        crossing = int(np.searchsorted(-np.asarray(ramp), CUTTING_LIMIT))
        assert tunnel.first < crossing

    def test_reaching_out_never_swallows_the_whole_road(self) -> None:
        ramp = list(np.linspace(0.0, 40.0, 20))
        line, natural = _line(ramp + _flat(20, 40.0) + ramp[::-1])
        kinds = _kinds(choose_structures(line, natural))
        assert kinds[0] is Op.DIRT and kinds[-1] is Op.DIRT


class TestACausewayAcrossWater:
    def test_fill_over_a_drowned_valley_is_a_causeway(self) -> None:
        """The ground drops below the waterline and the road is carried over on
        fill: not a bridge, because the fill is shallow, but not plain dirt
        either -- the world has to know the road is on an embankment in water."""
        natural = np.array(_flat(10, 0.0) + _flat(30, -6.0) + _flat(10, 0.0))
        heights = np.maximum(natural, -2.0) + 1.0
        x = np.arange(len(natural)) * SPACING
        line = np.stack([x, heights, np.zeros(len(natural))], axis=-1)
        found = choose_structures(line, natural, waterline=-2.0)
        assert Op.CAUSEWAY in _kinds(found)

    def test_dry_ground_gets_no_causeway(self) -> None:
        line, natural = _line(_flat(40, 3.0))
        assert Op.CAUSEWAY not in _kinds(
            choose_structures(line, natural, waterline=-100.0))

    def test_without_a_waterline_there_is_no_causeway(self) -> None:
        natural = np.array(_flat(10, 0.0) + _flat(30, -6.0) + _flat(10, 0.0))
        heights = np.maximum(natural, -2.0) + 1.0
        x = np.arange(len(natural)) * SPACING
        line = np.stack([x, heights, np.zeros(len(natural))], axis=-1)
        assert Op.CAUSEWAY not in _kinds(choose_structures(line, natural))

    def test_a_bridge_over_water_is_still_a_bridge(self) -> None:
        natural = np.array(_flat(10, 0.0) + _flat(30, -60.0) + _flat(10, 0.0))
        heights = np.zeros(len(natural)) + 1.0
        x = np.arange(len(natural)) * SPACING
        line = np.stack([x, heights, np.zeros(len(natural))], axis=-1)
        assert Op.BRIDGE in _kinds(
            choose_structures(line, natural, waterline=-2.0))


class TestACircuit:
    def test_a_structure_across_the_start_line_is_one_structure(self) -> None:
        """A lap's array begins somewhere arbitrary, and a viaduct that happens
        to straddle that point is not two viaducts."""
        line, natural = _line(_flat(20, 40.0) + _flat(30, 0.0)
                              + _flat(20, 40.0))
        found = choose_structures(line, natural, closed=True)
        assert _kinds(found).count(Op.BRIDGE) == 1

    def test_the_wrapped_structure_knows_it_wraps(self) -> None:
        line, natural = _line(_flat(20, 40.0) + _flat(30, 0.0)
                              + _flat(20, 40.0))
        bridge = next(s for s in choose_structures(line, natural, closed=True)
                      if s.kind is Op.BRIDGE)
        assert bridge.wraps

    def test_an_open_road_never_wraps(self) -> None:
        line, natural = _line(_flat(20, 40.0) + _flat(30, 0.0)
                              + _flat(20, 40.0))
        assert not any(s.wraps for s in choose_structures(line, natural))

    def test_a_lap_entirely_on_a_viaduct_is_one_ring(self) -> None:
        line, natural = _line(_flat(60, 40.0))
        found = choose_structures(line, natural, closed=True)
        assert _kinds(found) == [Op.BRIDGE]


class TestWhatAStructureTellsYou:
    def test_it_knows_the_points_it_covers(self) -> None:
        line, natural = _line(_flat(10) + _flat(40, -40.0) + _flat(10))
        tunnel = next(s for s in choose_structures(line, natural)
                      if s.kind is Op.TUNNEL)
        assert len(tunnel.indices(60)) == tunnel.last - tunnel.first + 1

    def test_a_wrapped_structure_lists_its_points_in_order(self) -> None:
        line, natural = _line(_flat(20, 40.0) + _flat(30, 0.0)
                              + _flat(20, 40.0))
        bridge = next(s for s in choose_structures(line, natural, closed=True)
                      if s.kind is Op.BRIDGE)
        indices = bridge.indices(70)
        assert indices[0] > indices[-1] or bridge.first <= bridge.last
        assert len(set(indices.tolist())) == len(indices)

    def test_it_knows_how_long_it_is(self) -> None:
        line, natural = _line(_flat(10) + _flat(40, -40.0) + _flat(10))
        tunnel = next(s for s in choose_structures(line, natural)
                      if s.kind is Op.TUNNEL)
        assert tunnel.length(line) == pytest.approx(
            (tunnel.last - tunnel.first) * SPACING)

    def test_it_reads_as_its_kind(self) -> None:
        assert 'tunnel' in repr(Structure(Op.TUNNEL, 3, 9)).lower()


class TestOverridingTheChoice:
    def test_a_designer_can_force_a_stretch_to_be_a_bridge(self) -> None:
        line, natural = _line(_flat(40, 2.0))
        found = choose_structures(line, natural,
                                  overrides=[(10, 25, Op.BRIDGE)])
        assert _kinds(found) == [Op.DIRT, Op.BRIDGE, Op.DIRT]

    def test_an_override_wins_over_what_the_ground_says(self) -> None:
        line, natural = _line(_flat(10) + _flat(40, -40.0) + _flat(10))
        found = choose_structures(line, natural,
                                  overrides=[(0, 59, Op.DIRT)])
        assert _kinds(found) == [Op.DIRT]

    def test_an_override_may_span_the_whole_road(self) -> None:
        line, natural = _line(_flat(40))
        found = choose_structures(line, natural,
                                  overrides=[(0, 39, Op.TUNNEL)])
        assert _kinds(found) == [Op.TUNNEL]


class TestARealAlignment:
    """The shipped circuit, which is what all of this exists for."""

    @pytest.fixture(scope='class')
    def circuit(self):
        from OpenGLContext.loaders.tiles3d.procedural import terrain_height
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        world = ProceduralWorld(structures=False)
        path = world.circuit()
        natural = np.asarray(terrain_height(path.points[:, 0], path.points[:, 2]),
                             dtype='d')
        return path.points, natural

    def test_the_mountains_it_crosses_become_tunnels(self, circuit) -> None:
        line, natural = circuit
        found = choose_structures(line, natural, closed=True)
        assert _kinds(found).count(Op.TUNNEL) >= 2

    def test_the_valleys_it_crosses_become_bridges(self, circuit) -> None:
        line, natural = circuit
        found = choose_structures(line, natural, closed=True)
        assert _kinds(found).count(Op.BRIDGE) >= 2

    def test_no_earthwork_is_left_taller_than_a_building(self, circuit) -> None:
        """The point of the exercise: what is left for the machines to move is
        a cutting or an embankment, not a mountain."""
        line, natural = circuit
        departure = line[:, 1] - natural
        remaining = np.ones(len(line), dtype=bool)
        for structure in choose_structures(line, natural, closed=True):
            if structure.kind in (Op.BRIDGE, Op.TUNNEL):
                remaining[structure.indices(len(line))] = False
        worst = float(np.abs(departure[remaining]).max())
        assert worst <= max(CUTTING_LIMIT, EMBANKMENT_LIMIT) + 1.0


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
