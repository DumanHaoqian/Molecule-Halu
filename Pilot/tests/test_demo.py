from __future__ import annotations

import json

from molhallulens.demo import (
    _inline_diff_html,
    _local_outputs,
    _text_outputs,
    build_demo,
    complete_text_run,
    prepare_local_run,
)
from molhallulens.modules.text_realization import PoeStepTextAgent
from molhallulens.modules.text_realization import FORMAL_MARKER


def _marked_head(step: dict) -> str:
    prefix = f"Step {step['step_index']} [{step['step_name']}]: "
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
    return prefix + body


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
                        "rewritten_step_text": (
                            (
                                step["original_step_text"].split(FORMAL_MARKER, 1)[0]
                                if not step["required_hallucination_occurrences"]
                                else _marked_head(step)
                            )
                            + FORMAL_MARKER
                            + step["modified_formal_ab"]
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
    assert len(config["dependencies"]) == 2


def test_inline_diff_highlights_both_original_and_modified_values():
    original, modified = _inline_diff_html("The product has 43 heavy atoms.", "The product has 44 heavy atoms.")

    assert '<mark class="diff-before"' in original
    assert ">43</mark>" in original
    assert '<mark class="diff-after"' in modified
    assert ">44</mark>" in modified
