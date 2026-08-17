"""``oglc-bake`` from the outside: what the command line does and refuses."""

import os

import pytest

from OpenGLContext_editor.bin import bake


def _run(tmp_path, *arguments):
    return bake.main(['--output', str(tmp_path / 'world'), '--extent', '512',
                      '--depth', '1', '--resolution', '9', *arguments])


class TestBakingFromTheCommandLine:
    def test_it_writes_a_world_and_reports_success(self, tmp_path, capsys) -> None:
        assert _run(tmp_path) == 0
        assert (tmp_path / 'world' / 'tileset.json').exists()
        assert 'tiles' in capsys.readouterr().out

    def test_quiet_prints_only_the_path(self, tmp_path, capsys) -> None:
        _run(tmp_path, '--quiet')
        printed = capsys.readouterr().out.strip()
        assert printed.endswith('tileset.json')
        assert '\n' not in printed

    def test_it_names_the_viewer_command(self, tmp_path, capsys) -> None:
        _run(tmp_path)
        assert 'oglc-view' in capsys.readouterr().out

    def test_the_world_it_writes_is_credited(self, tmp_path) -> None:
        _run(tmp_path, '--quiet')
        assert 'OpenGLContext' in (tmp_path / 'world' / 'CREDITS.txt').read_text()

    def test_the_seed_decides_the_world(self, tmp_path) -> None:
        _run(tmp_path, '--quiet', '--seed', '3')
        first = (tmp_path / 'world' / 'tileset.json').read_text()
        _run(tmp_path, '--quiet', '--seed', '3', '--force')
        assert (tmp_path / 'world' / 'tileset.json').read_text() == first

    def test_tree_density_reaches_the_world(self, tmp_path) -> None:
        import json
        _run(tmp_path, '--quiet')
        dense = json.load(open(tmp_path / 'world' / 'tileset.json'))
        _run(tmp_path, '--quiet', '--force', '--tree-density', '0.0002')
        sparse = json.load(open(tmp_path / 'world' / 'tileset.json'))
        assert sparse['extras']['vegetation']['count'] \
            < dense['extras']['vegetation']['count']

    def test_a_world_with_no_trees_at_all_still_bakes(self, tmp_path) -> None:
        import json
        assert _run(tmp_path, '--quiet', '--tree-density', '0.0') == 0
        document = json.load(open(tmp_path / 'world' / 'tileset.json'))
        assert document['extras']['vegetation']['count'] == 0

    def test_the_road_surface_is_written_once_beside_the_tileset(
            self, tmp_path) -> None:
        """Embedded in every tile it would outweigh all the geometry."""
        _run(tmp_path, '--quiet')
        world = tmp_path / 'world'
        surface = world / 'road-surface.png'
        assert surface.exists()
        tiles = [entry for entry in os.listdir(world) if entry.endswith('.glb')]
        biggest = max(os.path.getsize(world / entry) for entry in tiles)
        assert biggest < 4 * os.path.getsize(surface)
        assert _bytes_of_tiles(tmp_path, '--quiet', '--force') \
            < len(tiles) * os.path.getsize(surface)

    def test_the_trees_are_written_once_beside_it_too(self, tmp_path) -> None:
        _run(tmp_path, '--quiet')
        world = tmp_path / 'world'
        assert (world / 'trees.npz').exists()
        assert (world / 'trees' / 'fir.npz').exists()

    def test_a_tile_forest_puts_them_in_the_tiles_instead(self, tmp_path) -> None:
        import json
        _run(tmp_path, '--quiet', '--forest', 'tiles')
        document = json.load(open(tmp_path / 'world' / 'tileset.json'))
        assert 'vegetation' not in document['extras']
        assert not (tmp_path / 'world' / 'trees.npz').exists()

    def test_a_tiled_ground_puts_the_landscape_in_the_tiles(self, tmp_path) -> None:
        import json
        _run(tmp_path, '--quiet', '--ground', 'tiles')
        document = json.load(open(tmp_path / 'world' / 'tileset.json'))
        assert 'terrain' not in document['extras']

    def test_the_trees_it_used_are_credited(self, tmp_path) -> None:
        """They are CC-BY, and the attribution travels with the world."""
        _run(tmp_path, '--quiet')
        assert 'CC-BY' in (tmp_path / 'world' / 'CREDITS.txt').read_text()

    def test_an_instance_cap_reaches_the_world(self, tmp_path) -> None:
        assert _run(tmp_path, '--quiet', '--max-instances', '4') == 0


def _bytes_of_tiles(tmp_path, *arguments):
    _run(tmp_path, *arguments)
    world = tmp_path / 'world'
    return sum(os.path.getsize(world / entry) for entry in os.listdir(world)
               if entry.endswith('.glb'))


class TestWhatItRefuses:
    def test_it_will_not_bake_over_a_world_unasked(self, tmp_path) -> None:
        _run(tmp_path, '--quiet')
        with pytest.raises(SystemExit, match='--force'):
            _run(tmp_path, '--quiet')

    def test_force_replaces_it(self, tmp_path) -> None:
        _run(tmp_path, '--quiet')
        assert _run(tmp_path, '--quiet', '--force') == 0


class TestOpeningTheResult:
    def test_view_hands_the_tileset_to_the_viewer(self, tmp_path, monkeypatch) -> None:
        called = {}

        def fake_call(command):
            called['command'] = command
            return 0

        monkeypatch.setattr(bake.shutil, 'which', lambda name: '/usr/bin/' + name)
        monkeypatch.setattr(bake.subprocess, 'call', fake_call)
        assert _run(tmp_path, '--quiet', '--view') == 0
        assert called['command'][0].endswith('oglc-view')
        assert called['command'][1].endswith('tileset.json')

    def test_a_missing_viewer_is_reported(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.setattr(bake.shutil, 'which', lambda name: None)
        assert _run(tmp_path, '--quiet', '--view') == 1
        assert 'oglc-view' in capsys.readouterr().err


class TestTheHelp:
    def test_every_option_says_what_it_does(self) -> None:
        parser = bake.build_parser()
        for action in parser._actions:
            assert action.help, action.dest
