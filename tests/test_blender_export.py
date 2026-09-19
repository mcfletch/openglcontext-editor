"""The add-on, in a real Blender, writing a real file.

The rules a chain is written by are tested without Blender in
``test_blender_msftlod.py``; what cannot be tested that way is whether Blender's
own glTF exporter actually calls us, whether what we set survives its
post-processing, and whether a mesh shared between objects survives as one mesh.
Each of those has failed quietly in a way no unit test could see -- an exporter
that swallows a hook's exception and logs it produces a file that is wrong and an
export that looked like it worked -- so this runs the thing.

Blender is a separate application rather than a dependency: these cases run
where one is on the path (or ``$BLENDER`` names it) and are skipped where none
is. The gate that has to have run them is a machine with Blender installed; see
``docs/blender.md``.
"""

import json
import os
import struct
import subprocess
import sys
import textwrap

import pytest

from OpenGLContext_editor import blender

pytestmark = pytest.mark.blender


@pytest.fixture(scope='module')
def have_blender():
    try:
        return blender.executable()
    except blender.BlenderMissing as error:
        pytest.skip(str(error))


def run_in_blender(script: str, tmp_path, have_blender) -> dict:
    """Run ``script`` inside Blender; return the JSON it printed after MARK.

    Blender exits 0 on a Python traceback under ``--background``, so the script
    says what happened rather than the return code being trusted.
    """
    path = tmp_path / 'inside.py'
    path.write_text(textwrap.dedent(script))
    done = subprocess.run(
        [have_blender, '--background', '--factory-startup', '-P', str(path)],
        capture_output=True, text=True, timeout=600)
    if 'MARK ' not in done.stdout:
        sys.stderr.write(done.stdout[-4000:] + done.stderr[-4000:])
        raise AssertionError('the script in Blender printed no result')
    return json.loads(done.stdout.split('MARK ', 1)[1].splitlines()[0])


def read_glb(path: str) -> dict:
    with open(path, 'rb') as handle:
        handle.read(12)
        length, _kind = struct.unpack('<II', handle.read(8))
        return json.loads(handle.read(length))


BUILD = '''
    import json, sys
    sys.path.insert(0, %(addon)r)
    import bpy
    bpy.ops.preferences.addon_enable(module='openglcontext_lod')
    from openglcontext_lod import scene
    scene.clear()
    fine = scene.box_mesh('Block', (1.0, 1.0, 1.0))
    coarse = scene.box_mesh('Block_LOD1', (1.0, 1.0, 1.0))
    where = bpy.context.scene.collection
    %(body)s
    scene.export(%(out)r)
    print('MARK ' + json.dumps({'enabled':
        'openglcontext_lod' in bpy.context.preferences.addons.keys()}))
'''


def build(tmp_path, have_blender, body: str) -> dict:
    out = str(tmp_path / 'world.glb')
    said = run_in_blender(
        BUILD % {'addon': blender.addon_directory().rsplit(os.sep, 1)[0],
                 'body': textwrap.indent(textwrap.dedent(body), '    ').strip(),
                 'out': out},
        tmp_path, have_blender)
    assert said['enabled'], 'the add-on did not enable, so no levels are written'
    return read_glb(out)


TWO_CHAINS = '''
for number, x in enumerate((0.0, 3.0)):
    scene.lod_chain([fine, coarse], 'block_%d' % number, (x, 0, 0),
                    coverage=[0.5, 0.0], collection=where)
'''


@pytest.fixture(scope='module')
def document(tmp_path_factory, have_blender):
    """Two chains over two shared meshes, exported once for the whole class."""
    return build(tmp_path_factory.mktemp('two'), have_blender, TWO_CHAINS)


class TestAChainThroughBlendersOwnExporter:
    def test_the_extension_is_declared(self, document):
        """The exporter drops a declaration it was not told to keep."""
        assert 'MSFT_lod' in document['extensionsUsed']

    def test_it_is_not_required(self, document):
        assert 'MSFT_lod' not in (document.get('extensionsRequired') or [])

    def test_every_chain_is_written(self, document):
        carrying = [node for node in document['nodes']
                    if 'MSFT_lod' in (node.get('extensions') or {})]

        assert len(carrying) == 2

    def test_the_finest_node_names_the_coarser_one(self, document):
        carrying = next(node for node in document['nodes']
                        if 'MSFT_lod' in (node.get('extensions') or {}))

        assert len(carrying['extensions']['MSFT_lod']['ids']) == 1

    def test_the_coverage_survives_the_exporter(self, document):
        carrying = next(node for node in document['nodes']
                        if 'MSFT_lod' in (node.get('extensions') or {}))

        assert carrying['extras']['MSFT_screencoverage'] == [0.5, 0.0]

    def test_the_alternatives_are_in_no_scene(self, document):
        named = set()
        for node in document['nodes']:
            named.update((node.get('extensions') or {})
                         .get('MSFT_lod', {}).get('ids', []))

        assert not (named & set(document['scenes'][0]['nodes']))

    def test_the_alternatives_are_still_in_the_file(self, document):
        """Out of the scene is not out of the document."""
        assert len(document['nodes']) == 4

    def test_a_shared_mesh_is_written_once(self, document):
        """Four objects over two meshes: what lets a renderer batch them."""
        assert len(document['meshes']) == 2


UNMARKED = '''
scene.place(fine, 'Plinth', (0, 0, 0), collection=where)
'''


class TestASceneWithNoChainsInIt:
    def test_nothing_is_declared(self, tmp_path, have_blender):
        document = build(tmp_path, have_blender, UNMARKED)

        assert 'MSFT_lod' not in (document.get('extensionsUsed') or [])

    def test_the_object_is_still_exported(self, tmp_path, have_blender):
        document = build(tmp_path, have_blender, UNMARKED)

        assert len(document['nodes']) == 1


NAMED_ONLY = '''
a = scene.place(fine, 'Urn_LOD0', (0, 0, 0), collection=where)
b = scene.place(coarse, 'Urn_LOD1', (0, 0, 0), collection=where)
'''


class TestAChainMarkedOnlyByName:
    """A chain brought in from elsewhere needs nothing done to it."""

    def test_the_suffix_is_enough(self, tmp_path, have_blender):
        document = build(tmp_path, have_blender, NAMED_ONLY)

        carrying = [node for node in document['nodes']
                    if 'MSFT_lod' in (node.get('extensions') or {})]

        assert len(carrying) == 1


PLACED = '''
scene.lod_chain([fine, coarse], 'block', (2.0, 0.0, 3.0), turn=0.7,
                coverage=[0.5, 0.0], collection=where)
'''


class TestWhereTheLevelsAreWritten:
    def test_both_levels_state_the_same_placement(self, tmp_path, have_blender):
        """An alternative stands in place of the node that names it, so the
        two agree; a reader that applied both would draw it twice as far out."""
        document = build(tmp_path, have_blender, PLACED)
        places = [(node.get('translation'), node.get('rotation'))
                  for node in document['nodes']]

        assert len(set(map(str, places))) == 1


BUDGETED = '''
bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32)
dense = bpy.context.active_object
meshes = scene.decimation_chain(dense, 4, 0.5, max_triangles=500)
counts = [scene.triangle_count(m) for m in meshes]
print('MARK ' + json.dumps({'enabled': True, 'counts': counts,
                            'source': scene.triangle_count(dense.data)}))
'''


@pytest.fixture(scope='module')
def counted(tmp_path_factory, have_blender):
    """A dense sphere cut to a 500-triangle budget, measured inside Blender."""
    tmp_path = tmp_path_factory.mktemp('budget')
    return run_in_blender(BUILD % {
        'addon': blender.addon_directory().rsplit(os.sep, 1)[0],
        'body': textwrap.indent(textwrap.dedent(BUDGETED), '    ').strip(),
        'out': str(tmp_path / 'unused.glb'),
    }, tmp_path, have_blender)


class TestATriangleBudget:
    """Blender's Decimate takes a ratio and reports a read-only face count, so
    a budget in triangles has to be turned into ratios. Whether the arithmetic
    lands where it says it does is a question only Blender can answer."""

    def test_the_source_really_is_denser_than_the_budget(self, counted):
        assert counted['source'] > 500

    def test_the_finest_level_is_brought_down_to_the_budget(self, counted):
        """Within what a collapse can land on -- it removes whole edges."""
        assert counted['counts'][0] == pytest.approx(500, rel=0.1)

    def test_the_chain_halves_from_there(self, counted):
        first, second = counted['counts'][0], counted['counts'][1]

        assert second == pytest.approx(first / 2, rel=0.15)

    def test_every_level_is_coarser_than_the_one_before(self, counted):
        assert counted['counts'] == sorted(counted['counts'], reverse=True)


class TestTheAddOnInstallsOnItsOwn:
    """`MSFT_lod` is a Khronos extension rather than anything of ours, so the
    add-on has to reach Blender without this toolkit: a zip, and Blender's own
    installer."""

    def test_blender_installs_the_zip_and_finds_the_export_hook(
            self, tmp_path, have_blender):
        zipped = blender.package(str(tmp_path))
        script = tmp_path / 'install.py'
        script.write_text(textwrap.dedent('''
            import bpy, json
            bpy.ops.extensions.package_install_files(
                filepath=%r, repo='user_default', enable_on_install=True)
            import sys
            named = [k for k in bpy.context.preferences.addons.keys()
                     if 'openglcontext_lod' in k]
            module = sys.modules.get(named[0]) if named else None
            print('MARK ' + json.dumps({
                'enabled': bool(named),
                'hook': hasattr(module, 'glTF2ExportUserExtension'),
                'operator': hasattr(bpy.ops.object, 'make_lod_chain'),
            }))
        ''') % (zipped,))
        done = subprocess.run(
            [have_blender, '--background', '--factory-startup', '-P',
             str(script)], capture_output=True, text=True, timeout=600)
        said = json.loads(done.stdout.split('MARK ', 1)[1].splitlines()[0])

        assert said == {'enabled': True, 'hook': True, 'operator': True}

    def test_the_zip_carries_the_add_on_and_nothing_else(self, tmp_path):
        import zipfile

        held = zipfile.ZipFile(blender.package(str(tmp_path))).namelist()

        assert all(name.startswith('openglcontext_lod/') for name in held)
        assert 'openglcontext_lod/blender_manifest.toml' in held
        assert not [name for name in held if '__pycache__' in name]
