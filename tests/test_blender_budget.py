"""How many triangles each level of a chain is allowed.

An author thinks in budgets -- "the finest level is not to exceed twenty
thousand" -- and Blender's Decimate modifier thinks in ratios: ``ratio`` is the
only target it takes, and its ``face_count`` is read-only and merely reports
what came out. Turning one into the other is arithmetic over triangle counts
and carries no Blender, so it is here.

The ratios are all measured against the *original* mesh, because that is what
the modifier does: one modifier on the source object, asked for a different
ratio per level.
"""
import pytest

from OpenGLContext_editor.blender.openglcontext_lod import budget


class TestWhatEachLevelIsAllowed:
    def test_the_finest_is_the_mesh_as_it_came(self):
        assert budget.targets_for(10_000, levels=3)[0] == 10_000

    def test_each_level_halves_by_default(self):
        assert budget.targets_for(8_000, levels=4) == [8_000, 4_000, 2_000, 1_000]

    def test_another_ratio_is_honoured(self):
        assert budget.targets_for(1_000, levels=3, ratio=0.1) == [1_000, 100, 10]

    def test_one_level_is_just_the_mesh(self):
        assert budget.targets_for(5_000, levels=1) == [5_000]

    def test_a_chain_always_decreases(self):
        made = budget.targets_for(17_456, levels=6)

        assert made == sorted(made, reverse=True)

    def test_no_level_is_too_small_to_be_a_mesh(self):
        """A level of two triangles is not a coarser model, it is a fold."""
        made = budget.targets_for(100, levels=10)

        assert min(made) >= budget.FLOOR


class TestCappingTheFinestLevel:
    def test_a_mesh_over_the_cap_is_brought_down_to_it(self):
        """'The top LOD should not be more than twenty thousand.'"""
        assert budget.targets_for(500_000, levels=4, max_triangles=20_000)[0] \
            == 20_000

    def test_the_rest_of_the_chain_follows_from_the_cap(self):
        made = budget.targets_for(500_000, levels=3, max_triangles=20_000)

        assert made == [20_000, 10_000, 5_000]

    def test_a_mesh_already_under_the_cap_is_left_alone(self):
        assert budget.targets_for(9_000, levels=2, max_triangles=20_000) \
            == [9_000, 4_500]

    def test_a_cap_of_none_caps_nothing(self):
        assert budget.targets_for(500_000, levels=1, max_triangles=None) \
            == [500_000]


class TestTheRatiosTheModifierIsGiven:
    def test_an_uncapped_finest_level_is_not_decimated_at_all(self):
        assert budget.ratios_for(10_000, levels=3)[0] == 1.0

    def test_a_capped_finest_level_is(self):
        assert budget.ratios_for(100_000, levels=2, max_triangles=20_000)[0] \
            == pytest.approx(0.2)

    def test_every_ratio_is_against_the_original(self):
        """One modifier on the source object, asked for a ratio per level."""
        ratios = budget.ratios_for(1_000, levels=3)

        assert ratios == pytest.approx([1.0, 0.5, 0.25])

    def test_no_ratio_exceeds_one(self):
        for ratio in budget.ratios_for(10, levels=5):
            assert 0.0 < ratio <= 1.0

    def test_they_decrease(self):
        ratios = budget.ratios_for(17_456, levels=6, max_triangles=8_000)

        assert ratios == sorted(ratios, reverse=True)

    def test_a_mesh_smaller_than_the_floor_still_gives_usable_ratios(self):
        for ratio in budget.ratios_for(3, levels=4):
            assert 0.0 < ratio <= 1.0


class TestSayingWhatItWillDo:
    def test_it_reads_as_a_chain(self):
        said = budget.describe(budget.targets_for(17_456, levels=6))

        assert '17,456' in said and '546' in said

    def test_a_single_level_says_so(self):
        assert 'one level' in budget.describe([1_000])


class TestRefusals:
    def test_no_levels_at_all(self):
        with pytest.raises(ValueError):
            budget.targets_for(1_000, levels=0)

    def test_a_ratio_of_one_would_never_coarsen(self):
        with pytest.raises(ValueError):
            budget.targets_for(1_000, levels=3, ratio=1.0)

    def test_a_ratio_over_one_would_grow_the_mesh(self):
        with pytest.raises(ValueError):
            budget.targets_for(1_000, levels=3, ratio=1.5)

    def test_an_empty_mesh(self):
        with pytest.raises(ValueError):
            budget.targets_for(0, levels=3)
