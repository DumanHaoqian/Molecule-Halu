from __future__ import annotations

from conftest import structured_fixture_transport

import json

import pytest
import molhallulens.demo as demo_module
from conftest import preserve_enumerations

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
            f"{item['node_id']}="
            f"[[HALLU:{item['node_id']}.01]]{item['after_text']}[[/HALLU]]"
            for item in step["affected_node_claims"]
        )
        return preserve_enumerations(f"Updated claims: {claims}.", step)
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


def test_gradio_demo_runs_real_a_to_f_modules_without_live_poe(tmp_path, monkeypatch):
    # Small real tokenizer fixture; no external vocabulary download in tests.
    import tiktoken
    encoding = tiktoken.Encoding(
        name="test_bytes", pat_str=r"(?s).",
        mergeable_ranks={bytes([i]): i for i in range(256)}, special_tokens={},
    )
    monkeypatch.setattr(demo_module, "_demo_encoding", lambda: encoding)
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
        transport=structured_fixture_transport(fake_poe),
        environment={},
        cache_directory=tmp_path,
    )
    text_run = complete_text_run(local, agent=agent)
    outputs = _text_outputs(text_run)
    assert len(outputs) == 5
    assert '<mark class="diff-before"' in outputs[1]
    assert '<mark class="diff-after"' in outputs[1]
    assert '原始分子（编辑前）' in outputs[1]
    assert '<svg' in outputs[1]
    assert outputs[1].index('source-molecule') < outputs[1].index('diff-legend')
    assert '整段粗粒度 span' not in outputs[0]
    assert len(outputs[2]) == len(text_run.annotated.spans)
    assert text_run.annotated.spans

    app = build_demo()
    config = app.get_config_file()
    assert len(config["dependencies"]) == 2
    buttons = [c for c in config["components"] if c["type"] == "button"]
    assert len(buttons) == 1
    assert "运行 A–E" in buttons[0]["props"]["value"]
    assert "幻觉 token 占比" in outputs[0]
    coverage = outputs[3]["demo_token_coverage"]
    assert 0 < coverage["hallucination_tokens"] <= coverage["total_tokens"]
    assert coverage["scope"] == "serialized.text"
    calls = []
    def fake_complete(local_run):
        calls.append(local_run)
        return text_run
    monkeypatch.setattr(demo_module, "complete_text_run", fake_complete)
    frames = list(demo_module._run_all_ui(
        local.reference.anonymous_sample_id, 0, local.plan.requested_edit_count,
    ))
    assert len(calls) == 1
    assert len(frames) == 3
    assert all(len(frame) == 19 for frame in frames)
    assert frames[0][15:] == ("", [], None, None)
    assert "幻觉 token 占比" in frames[-1][14]
    edit_count_components = [
        item
        for item in config["components"]
        if item["type"] == "slider"
        and item["props"].get("label") == "修改 root node 数量"
    ]
    assert len(edit_count_components) == 1
    assert edit_count_components[0]["props"]["minimum"] == 1
    assert edit_count_components[0]["props"]["maximum"] == 4


def test_token_coverage_uses_full_tokens_unicode_and_deduplicates():
    class Encoding:
        name = "fixture"
        def encode_ordinary(self, text):
            assert text == "hello中!"
            return [0, 1, 2, 3]
        def decode_single_token_bytes(self, token):
            return [b"hello", "中".encode()[:1], "中".encode()[1:], b"!"][token]
    # Two distinct spans inside 'hello' count once; 中 spans two byte tokens.
    result = demo_module._token_coverage("hello中!", [(1, 2), (3, 4), (5, 6)], Encoding())
    assert result["total_tokens"] == 4
    assert result["hallucination_tokens"] == 3
    assert result["percentage"] == 75.0
    assert demo_module._token_coverage("hello中!", [], Encoding())["percentage"] == 0.0
    with pytest.raises(ValueError, match="invalid character span"):
        demo_module._token_coverage("hello中!", [(0, 99)], Encoding())


def test_combined_run_clears_e_on_failure(monkeypatch):
    monkeypatch.setattr(demo_module, "_demo_encoding", lambda: None)
    def fail(local):
        raise RuntimeError("simulated Poe failure")
    monkeypatch.setattr(demo_module, "complete_text_run", fail)
    iterator = demo_module._run_all_ui("mol_edit.add_v2.0003", 0, 1)
    next(iterator)
    next(iterator)
    failure = next(iterator)
    assert "simulated Poe failure" in failure[14]
    assert failure[15:] == ("", [], None, None)
    with pytest.raises(demo_module.gr.Error):
        next(iterator)


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
