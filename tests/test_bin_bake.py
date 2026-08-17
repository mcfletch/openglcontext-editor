"""``oglc-bake`` from the outside: what the command line does and refuses."""

import json
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
        _run(tmp_path, '--quiet', '--tree-density', '0.0')
        document = json.loads((tmp_path / 'world' / 'tileset.json').read_text())
        # With no trees the tiles hold ground alone, so each is smaller.
        sizes = [os.path.getsize(tmp_path / 'world' / entry)
                 for entry in os.listdir(tmp_path / 'world') if entry.endswith('.glb')]
        assert document['root']['content']
        assert max(sizes) < 60 * 1024

    def test_an_instance_cap_reaches_the_world(self, tmp_path) -> None:
        assert _run(tmp_path, '--quiet', '--max-instances', '4') == 0


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
