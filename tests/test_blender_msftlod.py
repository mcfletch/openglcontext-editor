"""What a set of levels in a scene becomes in the exported document.

``MSFT_lod`` is written in terms of node *indices*, which exist only once the
glTF exporter has laid the document out -- so the add-on's job at export time is
to turn "these objects are levels of one another" into "node 7 names nodes 8, 9
and 10 as its coarser selves, and those three are in no scene". That mapping is
arithmetic over indices and carries no Blender in it, so it is tested here
rather than through an export.

The cases the spec pins down: the node carrying the extension is the *finest*
level, ``ids`` runs in decreasing detail, and ``MSFT_screencoverage`` has one
value per level including the finest.
"""

import pytest

from OpenGLContext_editor.blender.openglcontext_lod import msftlod


def level(node, group='bust', index=0, coverage=None):
    return msftlod.Level(node=node, group=group, level=index, coverage=coverage)


class TestHowAnObjectSaysWhichLevelItIs:
    def test_a_suffix_names_a_level(self):
        assert msftlod.named_level('Bust_LOD2') == ('Bust', 2)

    def test_the_finest_is_zero(self):
        assert msftlod.named_level('Bust_LOD0') == ('Bust', 0)

    def test_a_dot_or_a_dash_reads_the_same_way(self):
        assert msftlod.named_level('Bust.LOD3') == ('Bust', 3)
        assert msftlod.named_level('Bust-LOD3') == ('Bust', 3)

    def test_case_does_not_matter(self):
        assert msftlod.named_level('Bust_lod1') == ('Bust', 1)

    def test_an_ordinary_name_is_not_a_level(self):
        assert msftlod.named_level('Plinth') is None

    def test_blenders_own_duplicate_suffix_is_not_a_level(self):
        """``Bust.001`` is a copy, not a level of detail."""
        assert msftlod.named_level('Bust.001') is None


class TestTheCoverageSeries:
    def test_one_value_per_level(self):
        assert len(msftlod.coverage_series(4)) == 4

    def test_it_decreases(self):
        series = msftlod.coverage_series(5)
        assert series == sorted(series, reverse=True)

    def test_the_finest_takes_over_at_a_half(self):
        assert msftlod.coverage_series(3)[0] == 0.5

    def test_the_coarsest_never_stops_drawing(self):
        """A threshold nobody measured must not be why a model disappears."""
        assert msftlod.coverage_series(6)[-1] == 0.0

    def test_a_single_level_covers_everything(self):
        assert msftlod.coverage_series(1) == [0.0]


class TestThePlanAChainBecomes:
    def test_the_finest_node_carries_the_extension(self):
        made = msftlod.plan([level(7, index=0), level(8, index=1), level(9, index=2)])

        assert list(made.ids) == [7]

    def test_the_coarser_levels_are_named_in_decreasing_detail(self):
        made = msftlod.plan([level(9, index=2), level(7, index=0), level(8, index=1)])

        assert made.ids[7] == [8, 9]

    def test_every_level_but_the_finest_is_taken_out_of_the_scene(self):
        made = msftlod.plan([level(7, index=0), level(8, index=1), level(9, index=2)])

        assert made.detach == frozenset([8, 9])

    def test_coverage_has_one_value_per_level(self):
        made = msftlod.plan([level(7, index=0), level(8, index=1), level(9, index=2)])

        assert len(made.coverage[7]) == 3

    def test_what_the_author_stated_is_what_is_written(self):
        made = msftlod.plan([level(7, index=0, coverage=0.7),
                             level(8, index=1, coverage=0.2),
                             level(9, index=2, coverage=0.05)])

        assert made.coverage[7] == [0.7, 0.2, 0.05]

    def test_a_group_of_one_is_not_a_chain(self):
        made = msftlod.plan([level(7, index=0)])

        assert made.ids == {}
        assert made.detach == frozenset()

    def test_two_chains_are_kept_apart(self):
        made = msftlod.plan([level(1, 'bust', 0), level(2, 'bust', 1),
                             level(3, 'urn', 0), level(4, 'urn', 1)])

        assert made.ids == {1: [2], 3: [4]}

    def test_the_level_numbers_order_rather_than_index(self):
        """An author who deleted LOD1 still has a chain, not a hole."""
        made = msftlod.plan([level(1, index=0), level(2, index=2), level(3, index=5)])

        assert made.ids[1] == [2, 3]

    def test_two_objects_claiming_one_level_is_refused(self):
        with pytest.raises(ValueError):
            msftlod.plan([level(1, index=0), level(2, index=0)])

    def test_coverage_that_rises_with_distance_is_refused(self):
        with pytest.raises(ValueError):
            msftlod.plan([level(1, index=0, coverage=0.1),
                          level(2, index=1, coverage=0.4)])

    def test_a_chain_where_only_some_levels_state_coverage_is_refused(self):
        """Half a series is not a series; guessing the rest would be silent."""
        with pytest.raises(ValueError):
            msftlod.plan([level(1, index=0, coverage=0.5), level(2, index=1)])


class TestApplyingThePlanToADocument:
    def document(self):
        return {
            'nodes': [{'name': 'Bust', 'mesh': 0}, {'name': 'Bust_LOD1', 'mesh': 1}],
            'scenes': [{'nodes': [0, 1]}],
        }

    def test_the_extension_lands_on_the_finest_node(self):
        made = msftlod.plan([level(0, index=0), level(1, index=1)])
        document = self.document()

        msftlod.apply(document, made)

        assert document['nodes'][0]['extensions']['MSFT_lod'] == {'ids': [1]}

    def test_the_coverage_lands_in_extras(self):
        made = msftlod.plan([level(0, index=0), level(1, index=1)])
        document = self.document()

        msftlod.apply(document, made)

        assert document['nodes'][0]['extras']['MSFT_screencoverage'] == [0.5, 0.0]

    def test_the_coarse_level_leaves_the_scene(self):
        made = msftlod.plan([level(0, index=0), level(1, index=1)])
        document = self.document()

        msftlod.apply(document, made)

        assert document['scenes'][0]['nodes'] == [0]

    def test_the_coarse_level_leaves_its_parents_children(self):
        made = msftlod.plan([level(1, index=0), level(2, index=1)])
        document = {'nodes': [{'children': [1, 2]}, {}, {}], 'scenes': [{'nodes': [0]}]}

        msftlod.apply(document, made)

        assert document['nodes'][0]['children'] == [1]

    def test_a_node_left_with_no_children_has_none_rather_than_an_empty_list(self):
        """glTF gives ``children`` a minimum of one entry."""
        made = msftlod.plan([level(1, index=0), level(2, index=1)])
        document = {'nodes': [{'children': [2]}, {}, {}], 'scenes': [{'nodes': [0, 1]}]}

        msftlod.apply(document, made)

        assert 'children' not in document['nodes'][0]

    def test_the_document_declares_the_extension(self):
        made = msftlod.plan([level(0, index=0), level(1, index=1)])
        document = self.document()

        msftlod.apply(document, made)

        assert document['extensionsUsed'] == ['MSFT_lod']

    def test_it_is_declared_once(self):
        made = msftlod.plan([level(0, index=0), level(1, index=1)])
        document = self.document()
        document['extensionsUsed'] = ['MSFT_lod']

        msftlod.apply(document, made)

        assert document['extensionsUsed'] == ['MSFT_lod']

    def test_it_is_not_required(self):
        """A reader that ignores it draws the finest level, which is correct."""
        made = msftlod.plan([level(0, index=0), level(1, index=1)])
        document = self.document()

        msftlod.apply(document, made)

        assert 'extensionsRequired' not in document

    def test_nothing_is_written_for_an_empty_plan(self):
        document = self.document()

        msftlod.apply(document, msftlod.plan([]))

        assert document == self.document()


class TestWhatAnObjectSaysAboutItself:
    """How a level is read off a Blender object, without a Blender object.

    Anything answering ``get`` will do, which is what an object is.
    """

    def test_the_properties_are_believed(self):
        one = msftlod.level_of('whatever', {'lod_group': 'bust', 'lod_level': 2}, 4)

        assert (one.group, one.level, one.node) == ('bust', 2, 4)

    def test_the_name_answers_when_the_properties_do_not(self):
        one = msftlod.level_of('Bust_LOD3', {}, 4)

        assert (one.group, one.level) == ('Bust', 3)

    def test_the_properties_win_over_the_name(self):
        one = msftlod.level_of('Bust_LOD3', {'lod_group': 'urn', 'lod_level': 1}, 0)

        assert (one.group, one.level) == ('urn', 1)

    def test_a_group_with_no_level_is_the_finest(self):
        one = msftlod.level_of('Bust', {'lod_group': 'bust'}, 0)

        assert one.level == 0

    def test_a_coverage_is_carried_through(self):
        one = msftlod.level_of('Bust', {'lod_group': 'bust', 'lod_coverage': 0.25}, 0)

        assert one.coverage == 0.25

    def test_an_object_that_says_nothing_is_not_a_level(self):
        assert msftlod.level_of('Plinth', {}, 0) is None

    def test_a_level_number_that_is_not_one_is_refused(self):
        with pytest.raises(ValueError):
            msftlod.level_of('Bust', {'lod_group': 'bust', 'lod_level': 'near'}, 0)


class TestWritingIntoTheExportersOwnObjects:
    """The glTF exporter hands over objects rather than a document.

    Same rules, same module -- a second copy of them somewhere else is how the
    two drift apart.
    """

    def document(self):
        from types import SimpleNamespace as N
        return N(
            nodes=[N(name='Bust', extensions=None, extras=None, children=[1]),
                   N(name='Bust_LOD1', extensions=None, extras=None, children=None)],
            scenes=[N(nodes=[0, 1])],
            extensions_used=[],
        )

    def test_the_extension_lands_on_the_finest_node(self):
        gltf = self.document()

        msftlod.write_into(gltf, msftlod.plan([level(0, index=0), level(1, index=1)]))

        assert gltf.nodes[0].extensions['MSFT_lod'] == {'ids': [1]}

    def test_the_coverage_lands_in_extras(self):
        gltf = self.document()

        msftlod.write_into(gltf, msftlod.plan([level(0, index=0), level(1, index=1)]))

        assert gltf.nodes[0].extras['MSFT_screencoverage'] == [0.5, 0.0]

    def test_the_alternative_leaves_the_scene_and_its_parent(self):
        gltf = self.document()

        msftlod.write_into(gltf, msftlod.plan([level(0, index=0), level(1, index=1)]))

        assert gltf.scenes[0].nodes == [0]
        assert gltf.nodes[0].children is None

    def test_the_document_declares_the_extension(self):
        gltf = self.document()

        msftlod.write_into(gltf, msftlod.plan([level(0, index=0), level(1, index=1)]))

        assert gltf.extensions_used == ['MSFT_lod']

    def test_the_declaration_is_kept_where_the_exporter_prunes_it(self):
        """The exporter drops any declaration it was not told to keep."""
        gltf = self.document()
        keep = []

        msftlod.write_into(gltf, msftlod.plan([level(0, index=0), level(1, index=1)]),
                           keep_declared=keep)

        assert keep == ['MSFT_lod']
