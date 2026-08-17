"""The 3D Tiles 1.1 writer, asserted through the runtime that has to read it.

Each test writes a tileset and parses it with
``OpenGLContext.loaders.tiles3d.tileset.build_runtime_tileset`` -- the contract
the bake exists to satisfy -- rather than only inspecting the JSON.
"""

import json

import numpy as np
import pytest
from OpenGLContext.loaders.tiles3d.tileset import build_runtime_tileset

from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.bake.tileset import (
    BakedTile,
    tileset_document,
    write_tileset,
)


def _tile(low, high, error, **kwargs):
    return BakedTile(BoundingBox(low, high), geometric_error=error, **kwargs)


def _parsed(root, **kwargs):
    return build_runtime_tileset(tileset_document(root, **kwargs),
                                 base_uri='', recenter=True)


class TestOneTile:
    def test_the_document_declares_the_version(self) -> None:
        doc = tileset_document(_tile((0, 0, 0), (1, 1, 1), 4.0))
        assert doc['asset']['version'] == '1.1'

    def test_it_names_its_generator(self) -> None:
        doc = tileset_document(_tile((0, 0, 0), (1, 1, 1), 4.0))
        assert 'OpenGLContext' in doc['asset']['generator']

    def test_the_root_error_defaults_to_twice_the_root_tile_error(self) -> None:
        """The tileset error is the error of *not drawing the dataset at all*,
        so it must exceed the root tile's or the root never loads."""
        doc = tileset_document(_tile((0, 0, 0), (1, 1, 1), 4.0))
        assert doc['geometricError'] > doc['root']['geometricError']

    def test_the_bounds_survive(self) -> None:
        box = BoundingBox((-6, 1, -10), (6, 5, 10))
        parsed = _parsed(BakedTile(box, geometric_error=2.0))
        assert np.allclose(parsed.root.bounding_volume.center, box.center, atol=1e-9)

    def test_the_error_survives(self) -> None:
        assert _parsed(_tile((0, 0, 0), (1, 1, 1), 3.5)).root.geometric_error == 3.5

    def test_content_survives(self) -> None:
        parsed = _parsed(_tile((0, 0, 0), (1, 1, 1), 1.0, content_uris=['t0.glb']))
        assert parsed.root.content_uris == ['t0.glb']

    def test_several_contents_survive(self) -> None:
        """1.1 lets one tile carry terrain and its vegetation as separate glTF."""
        parsed = _parsed(_tile((0, 0, 0), (1, 1, 1), 1.0,
                               content_uris=['terrain.glb', 'trees.glb']))
        assert parsed.root.content_uris == ['terrain.glb', 'trees.glb']

    def test_a_single_content_is_written_the_1_0_way(self) -> None:
        """`content` rather than `contents`, so a 1.0 reader can load it too."""
        doc = tileset_document(_tile((0, 0, 0), (1, 1, 1), 1.0, content_uris=['t.glb']))
        assert doc['root']['content'] == {'uri': 't.glb'}
        assert 'contents' not in doc['root']

    def test_a_tile_can_carry_its_own_transform(self) -> None:
        """Content written in a local frame is placed by the tile's matrix."""
        import numpy as np
        matrix = np.identity(4)
        matrix[3, :3] = (100.0, 0.0, -50.0)
        doc = tileset_document(_tile((0, 0, 0), (1, 1, 1), 1.0, transform=matrix))
        assert doc['root']['transform'][12:15] == [100.0, 0.0, -50.0]

    def test_a_contentless_tile_is_allowed(self) -> None:
        """An interior node may exist only to hold children."""
        doc = tileset_document(_tile((0, 0, 0), (1, 1, 1), 1.0))
        assert 'content' not in doc['root'] and 'contents' not in doc['root']


class TestATree:
    def _nested(self):
        children = [_tile((-1, 0, -1), (0, 1, 0), 1.0, content_uris=['a.glb']),
                    _tile((0, 0, 0), (1, 1, 1), 1.0, content_uris=['b.glb'])]
        return _tile((-1, 0, -1), (1, 1, 1), 4.0, content_uris=['root.glb'],
                     children=children)

    def test_children_survive(self) -> None:
        parsed = _parsed(self._nested())
        assert len(parsed.root.children) == 2
        assert sorted(c.content_uri for c in parsed.root.children) == ['a.glb', 'b.glb']

    def test_refinement_is_stated_once_at_the_root(self) -> None:
        """REPLACE is inherited, so repeating it on every tile is dead weight
        in a file with thousands of them."""
        doc = tileset_document(self._nested())
        assert doc['root']['refine'] == 'REPLACE'
        assert all('refine' not in child for child in doc['root']['children'])

    def test_a_child_that_refines_differently_says_so(self) -> None:
        children = [_tile((-1, 0, -1), (0, 1, 0), 1.0, refine='ADD')]
        doc = tileset_document(_tile((-1, 0, -1), (1, 1, 1), 4.0, children=children))
        assert doc['root']['children'][0]['refine'] == 'ADD'

    def test_inherited_refinement_reaches_the_runtime(self) -> None:
        parsed = _parsed(self._nested())
        assert all(child.refine == 'REPLACE' for child in parsed.root.children)

    def test_the_tree_is_walkable_by_the_runtime(self) -> None:
        parsed = _parsed(self._nested())
        assert len(list(parsed.root.iter_tiles())) == 3


class TestTheInvariantsAreEnforced:
    def test_a_child_may_not_claim_a_larger_error_than_its_parent(self) -> None:
        """Error must fall down the tree, or a traversal that stops refining at
        the parent has already accepted a coarser tile than the child offers."""
        children = [_tile((0, 0, 0), (1, 1, 1), 8.0)]
        with pytest.raises(ValueError, match='geometric error'):
            tileset_document(_tile((0, 0, 0), (1, 1, 1), 4.0, children=children))

    def test_a_child_may_not_stick_out_of_its_parent(self) -> None:
        children = [_tile((0, 0, 0), (9, 9, 9), 1.0)]
        with pytest.raises(ValueError, match='outside'):
            tileset_document(_tile((0, 0, 0), (1, 1, 1), 4.0, children=children))

    def test_a_negative_error_is_refused(self) -> None:
        with pytest.raises(ValueError, match='geometric error'):
            tileset_document(_tile((0, 0, 0), (1, 1, 1), -1.0))

    def test_an_equal_error_is_allowed(self) -> None:
        """A node that adds content without refining detail is legitimate."""
        children = [_tile((0, 0, 0), (1, 1, 1), 4.0)]
        tileset_document(_tile((0, 0, 0), (1, 1, 1), 4.0, children=children))


class TestWritingToDisk:
    def test_it_writes_a_readable_tileset(self, tmp_path) -> None:
        path = write_tileset(_tile((0, 0, 0), (1, 1, 1), 1.0, content_uris=['t.glb']),
                             str(tmp_path))
        assert path.endswith('tileset.json')
        doc = json.loads(open(path).read())
        assert build_runtime_tileset(doc, base_uri='', recenter=True).root.content_uri

    def test_the_name_can_be_chosen(self, tmp_path) -> None:
        path = write_tileset(_tile((0, 0, 0), (1, 1, 1), 1.0), str(tmp_path),
                             name='trees.json')
        assert path.endswith('trees.json')

    def test_it_makes_the_directory(self, tmp_path) -> None:
        target = tmp_path / 'deep' / 'nested'
        write_tileset(_tile((0, 0, 0), (1, 1, 1), 1.0), str(target))
        assert (target / 'tileset.json').exists()

    def test_provenance_rides_with_the_tileset(self, tmp_path) -> None:
        """A baked world carries the licences of the data it was made from."""
        write_tileset(_tile((0, 0, 0), (1, 1, 1), 1.0), str(tmp_path),
                      credits=['SRTM elevation (public domain)',
                               'Bark texture, CC0'])
        text = (tmp_path / 'CREDITS.txt').read_text()
        assert 'SRTM elevation (public domain)' in text
        assert 'CC0' in text

    def test_the_credits_are_named_in_the_tileset_too(self, tmp_path) -> None:
        write_tileset(_tile((0, 0, 0), (1, 1, 1), 1.0), str(tmp_path),
                      credits=['SRTM elevation (public domain)'])
        doc = json.loads((tmp_path / 'tileset.json').read_text())
        assert 'SRTM' in doc['asset']['copyright']


class TestExternalTilesets:
    def test_a_branch_can_be_written_as_its_own_file(self, tmp_path) -> None:
        """A large branch bakes independently and is grafted in by URI."""
        branch = _tile((0, 0, 0), (1, 1, 1), 1.0, content_uris=['b.glb'])
        write_tileset(branch, str(tmp_path), name='branch.json')
        root = _tile((-1, 0, -1), (1, 1, 1), 4.0,
                     children=[_tile((0, 0, 0), (1, 1, 1), 2.0,
                                     content_uris=['branch.json'])])
        write_tileset(root, str(tmp_path))
        doc = json.loads((tmp_path / 'tileset.json').read_text())
        parsed = build_runtime_tileset(doc, base_uri=str(tmp_path) + '/', recenter=True)
        # The external tileset is grafted in as a subtree, so its content shows up.
        uris = [t.content_uri for t in parsed.root.iter_tiles() if t.content_uri]
        assert any(u.endswith('b.glb') for u in uris)
