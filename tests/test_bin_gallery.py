"""``oglce-gallery``'s build step: a Blender run is judged by what it wrote.

Blender exits 0 after a Python traceback in ``--background``, so the return
code says nothing about whether the build script finished. The build is taken
as done only when the script says it wrote the file, and a file an earlier run
left behind is never mistaken for this run's. Blender itself is stood in for
here; ``test_blender_export`` drives the real one.
"""
import subprocess

import pytest

from OpenGLContext_editor import blender
from OpenGLContext_editor.bin import gallery


def _blender(stdout, writes=None):
    def run(arguments, **_options):
        if writes is not None:
            with open(writes, 'wb') as handle:
                handle.write(b'glTF from this run')
        return subprocess.CompletedProcess(arguments, 0, stdout, '')
    return run


class TestABuildThatFailed:
    def test_a_file_left_by_an_earlier_run_is_not_this_runs(self, tmp_path,
                                                          monkeypatch):
        output = tmp_path / 'gallery.glb'
        output.write_bytes(b'glTF from last week')
        monkeypatch.setattr(blender, 'run', _blender(
            'Traceback (most recent call last):\n  ...\nKeyError: bust\n'))
        roofed = []
        monkeypatch.setattr(gallery, 'roof', roofed.append)

        with pytest.raises(SystemExit):
            gallery.build(str(tmp_path), str(output))

        assert roofed == []
        assert not output.exists()

    def test_a_file_written_without_the_script_saying_so_is_refused(
            self, tmp_path, monkeypatch):
        output = tmp_path / 'gallery.glb'
        monkeypatch.setattr(blender, 'run', _blender('BUILT: half a hall\n',
                                                     writes=output))
        monkeypatch.setattr(gallery, 'roof', lambda path: path)

        with pytest.raises(SystemExit):
            gallery.build(str(tmp_path), str(output))


class TestABuildThatWorked:
    def test_it_is_roofed_and_returned(self, tmp_path, monkeypatch):
        output = tmp_path / 'gallery.glb'
        monkeypatch.setattr(blender, 'run', _blender(
            'BUILT: a hall\nWROTE: %s (18 bytes)\n' % (output,), writes=output))
        roofed = []
        monkeypatch.setattr(gallery, 'roof', roofed.append)

        assert gallery.build(str(tmp_path), str(output)) == str(output)
        assert roofed == [str(output)]
