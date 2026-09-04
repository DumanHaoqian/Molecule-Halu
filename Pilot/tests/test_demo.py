from __future__ import annotations

import json

import pytest

from molhallulens.config import DEFAULT_HALLUCINATION_CONFIG
from molhallulens.demo import (
    _inline_diff_html,
    _local_outputs,
    _text_outputs,
    build_demo,
    complete_text_run,
    maximum_root_edit_count,
    prepare_local_run,
)
from molhallulens.modules.text_realization import PoeStepTextAgent
from molhallulens.modules.text_realization import FORMAL_MARKER


def _marked_head(step: dict) -> str:
    prefix = f"Step {step['step_index']} [{step['step_name']}]: "
    if step["rewrite_mode"] == "derivation_rewrite":
        claims = "; ".join(
            f"{item['node_id']}={item['after_text']}"
            for item in step["affected_node_claims"]
        )
        return f"[[HALLU:rewrite.01]]Updated claims: {claims}.[[/HALLU]]"
    head = step["original_step_text"].split(FORMAL_MARKER, 1)[0]
    body = head[len(prefix) :]
    for occurrence in sorted(
        step["required_hallucination_occurrences"],
        key=lambda item: item["original_span"][0],
        reverse=True,
    ):
        start, end = occurrence["original_span"]
        marker = (
            f"[[HALLU:{occurrence['occurrence_id']}]]"
            f"{occurrence['after_text']}[[/HALLU]]"
        )
        body = body[:start] + marker + body[end:]
    return body


def test_gradio_demo_runs_real_a_to_f_modules_without_live_poe(tmp_path):
    local = prepare_local_run("mol_edit.add_v2.0003", 0)
    assert len(local.reference.state_dag.values) == 22
    assert len(local.reference.state_dag.schema.edges) == 20
    assert len(_local_outputs(local)) == 15

    def fake_poe(system_prompt, user_prompt, bot_name, temperature):
        del system_prompt, bot_name, temperature
        payload = json.loads(user_prompt.split("\nINPUT:\n", 1)[1])
        return json.dumps(
            {
                "steps": [
                    {
                        "step_index": step["step_index"],
                        "rewritten_natural_language": (
                            step["original_step_text"].split(FORMAL_MARKER, 1)[0][
                                len(
                                    f"Step {step['step_index']} "
                                    f"[{step['step_name']}]: "
                                ) :
                            ]
                            if step["rewrite_mode"] == "copy"
                            else _marked_head(step)
                        ),
                    }
                    for step in payload["steps"]
                ]
            }
        )

    agent = PoeStepTextAgent(
        transport=fake_poe,
        environment={},
        cache_directory=tmp_path,
    )
    text_run = complete_text_run(local, agent=agent)
    outputs = _text_outputs(text_run)
    assert len(outputs) == 5
    assert '<mark class="diff-before"' in outputs[1]
    assert '<mark class="diff-after"' in outputs[1]
    assert len(outputs[2]) == len(text_run.annotated.spans)
    assert text_run.annotated.spans

    app = build_demo()
    config = app.get_config_file()
    assert len(config["dependencies"]) == 3
    edit_count_components = [
        item
        for item in config["components"]
        if item["type"] == "slider"
        and item["props"].get("label") == "修改 root node 数量"
    ]
    assert len(edit_count_components) == 1
    assert edit_count_components[0]["props"]["minimum"] == 1
    assert edit_count_components[0]["props"]["maximum"] == 4


def test_demo_user_can_choose_every_supported_root_edit_count():
    minimum = DEFAULT_HALLUCINATION_CONFIG.min_edit_count
    maximum = maximum_root_edit_count("mol_edit.add_v2.0003")
    assert maximum == 4

    for edit_count in range(minimum, maximum + 1):
        local = prepare_local_run("mol_edit.add_v2.0003", 0, edit_count)
        assert local.plan.requested_edit_count == edit_count
        assert len(local.plan.mutations) == edit_count
        assert _local_outputs(local)[0]["edit_count"] == edit_count

    with pytest.raises(ValueError, match="between 1 and 4"):
        prepare_local_run("mol_edit.add_v2.0003", 0, minimum - 1)
    with pytest.raises(ValueError, match="between 1 and 4"):
        prepare_local_run("mol_edit.add_v2.0003", 0, maximum + 1)

    assert maximum_root_edit_count("mol_edit.delete_v2.0016") == 3
    assert maximum_root_edit_count("mol_edit.substitute_v2.0000") == 3


def test_inline_diff_highlights_both_original_and_modified_values():
    original, modified = _inline_diff_html("The product has 43 heavy atoms.", "The product has 44 heavy atoms.")

    assert '<mark class="diff-before"' in original
    assert ">43</mark>" in original
    assert '<mark class="diff-after"' in modified
    assert ">44</mark>" in modified
