from dataclasses import replace
from random import Random

import pytest

from molhallulens.config import DEFAULT_HALLUCINATION_CONFIG
from molhallulens.modules.error_injection import UnifiedHallucinationInjector
from molhallulens.modules.error_planning import UnifiedHallucinationPlanner


def test_maximum_mode_uses_capacity_without_sampling_or_artificial_cap():
    config = replace(
        DEFAULT_HALLUCINATION_CONFIG,
        edit_count_mode="maximum",
        max_edit_count=2,
    )
    random_source = Random(7)
    state = random_source.getstate()
    assert config.requested_edit_count(random_source, maximum_available=8) == 8
    assert random_source.getstate() == state
    with pytest.raises(ValueError, match="requires maximum_available"):
        config.requested_edit_count(random_source)
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            config.requested_edit_count(random_source, maximum_available=invalid)


def test_every_origin_uses_its_exact_maximum(all_references, fragment_pool):
    config = replace(DEFAULT_HALLUCINATION_CONFIG, edit_count_mode="maximum")
    planner = UnifiedHallucinationPlanner(fragment_pool, config)
    injector = UnifiedHallucinationInjector(config)
    assert len(all_references) == 150
    for reference in all_references:
        maximum = planner.maximum_edit_count(reference)
        plan = planner.plan(reference, variant_index=0)
        assert plan.requested_edit_count == len(plan.mutations) == maximum
        # Injection retains the existing propagation and edge-audit contracts.
        injector.apply(reference.state_dag, plan)


def test_cli_max_edits_selects_maximum_config(monkeypatch):
    import molhallulens.generate_dataset as generator

    class ObservedConfig(Exception):
        pass

    def observe(**kwargs):
        assert kwargs["config"].edit_count_mode == "maximum"
        assert kwargs["variants_per_origin"] == 1
        assert kwargs["config"].emit_matched_negative is True
        raise ObservedConfig

    monkeypatch.setattr("sys.argv", ["generate_dataset", "--max-edits"])
    monkeypatch.setattr(generator, "generate_dataset", observe)
    with pytest.raises(ObservedConfig):
        generator.main()
