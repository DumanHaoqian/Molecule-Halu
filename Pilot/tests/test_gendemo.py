from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import tiktoken

from scripts import gendemo
from molhallulens.modules.error_planning import UnifiedHallucinationPlanner
from molhallulens.modules.error_injection import UnifiedHallucinationInjector
from molhallulens.modules.text_realization import PoeStepTextAgent


@pytest.fixture
def saved_viewer(monkeypatch):
    if not gendemo.DEFAULT_RECORDS.exists():
        pytest.skip("Saved maximum-edit release is a local generated artifact")

    def forbidden(*args, **kwargs):
        raise AssertionError("Saved viewer must not plan, inject or call Poe")

    monkeypatch.setattr(PoeStepTextAgent, "__init__", forbidden)
    monkeypatch.setattr(UnifiedHallucinationPlanner, "plan", forbidden)
    monkeypatch.setattr(UnifiedHallucinationInjector, "apply", forbidden)
    encoding = tiktoken.Encoding(
        name="test_bytes", pat_str=r"(?s).",
        mergeable_ranks={bytes([i]): i for i in range(256)}, special_tokens={},
    )
    monkeypatch.setattr(gendemo, "_demo_encoding", lambda: encoding)
    return gendemo.SavedDemo()


def test_all_saved_pairs_replay_without_regeneration(saved_viewer):
    assert len(saved_viewer.pairs) == len(saved_viewer.references) == 150
    for pair_id, pair in saved_viewer.pairs.items():
        h = pair["H"]
        before = json.dumps(pair, sort_keys=True)
        reference = saved_viewer.references[h["origin_id"]]
        original_values = dict(reference.state_dag.values)
        graph, changes, changed = gendemo.replay_graph(reference, h)
        assert len(changes) == len(changed)
        for event in h["mutation_events"]:
            for node in event["target_node_ids"]:
                assert graph.values[node].normalized_value == event["after"]
        for event in h["propagation_events"]:
            assert graph.values[event["target_node_id"]].normalized_value == event["after"]
        output = saved_viewer.outputs(pair_id)
        assert len(output) == 18
        assert "本次 Poe 请求 **0**" in output[10]
        assert h["text_realization"]["bot_name"] in output[10]
        assert "幻觉 token 占比" in output[10]
        assert "<svg" in output[11]
        assert output[13].count('class="saved-hallu"') == len(h["hallucination_spans"])
        assert output[13].count('class="saved-control"') == len(pair["N"]["control_spans"])
        assert len(output[14]) == len(h["hallucination_spans"])
        assert output[16]["serialized"] == h["serialized"]
        assert output[17]["serialized"] == pair["N"]["serialized"]
        assert json.dumps(pair, sort_keys=True) == before
        assert dict(reference.state_dag.values) == original_values


def test_saved_ui_only_offers_successful_pairs(saved_viewer, monkeypatch):
    monkeypatch.setattr(gendemo, "SavedDemo", lambda *args: saved_viewer)
    app = gendemo.build_demo()
    try:
        dropdown = next(c for c in app.config["components"] if c["type"] == "dropdown")
        assert {choice[1] for choice in dropdown["props"]["choices"]} == set(saved_viewer.pairs)
        events = {event['api_name']: event for event in app.config['dependencies']}
        assert len(events) == 3
        assert 'load_saved_pair' not in events
        assert len(events['select_saved_pair']['outputs']) == 21
        for name in ('previous_saved_pair', 'next_saved_pair'):
            assert len(events[name]['outputs']) == 4
            assert events[name]['outputs'][0] == dropdown['id']
        buttons = {c['props']['value']: c for c in app.config['components'] if c['type'] == 'button'}
        assert buttons['← 上一个']['props']['interactive'] is False
        assert buttons['下一个 →']['props']['interactive'] is True
        assert '加载已保存的 A–E' not in buttons

        components = {c['id']: c for c in app.config['components']}
        root_ids = [child['id'] for child in app.config['layout']['children']]
        counter_id = events['select_saved_pair']['outputs'][-1]
        counter_index = root_ids.index(counter_id)
        assert components[root_ids[counter_index + 1]]['props']['value'] == '## 原始 vs H · 文字差异'
        comparison_id = events['select_saved_pair']['outputs'][12]
        assert root_ids[counter_index + 3] == comparison_id
        assert components[comparison_id]['type'] == 'html'
        assert not any(c['type'] == 'tabitem' and c['props'].get('label') == '原始 vs H · 文字差异'
                       for c in components.values())

        # Exercise the actual wired callbacks: next, previous, arbitrary jump,
        # last boundary, and full A–E rendering for the selected record.
        callbacks = {fn.api_name: fn.fn for fn in app.fns.values()}
        ids = tuple(saved_viewer.pairs)
        result = callbacks['next_saved_pair'](ids[0])
        assert result == (ids[1], {'__type__': 'update', 'interactive': True},
                          {'__type__': 'update', 'interactive': True}, '第 2 / 150 题')
        assert callbacks['previous_saved_pair'](result[0])[0] == ids[0]
        assert callbacks['next_saved_pair'](ids[70])[0] == ids[71]
        assert callbacks['next_saved_pair'](ids[-1])[0] == ids[-1]
        rendered = callbacks['select_saved_pair'](ids[-1])
        assert rendered[12] == saved_viewer.outputs(ids[-1])[12]
        assert rendered[16]['record_id'] == saved_viewer.pairs[ids[-1]]['H']['record_id']
        assert rendered[-2]['interactive'] is False
        assert rendered[-1] == '第 150 / 150 题'
    finally:
        app.close()


@pytest.mark.parametrize('current,offset,expected', [
    ('a', -1, ('a', False, True, '第 1 / 3 题')),
    ('a', 1, ('b', True, True, '第 2 / 3 题')),
    ('b', -1, ('a', False, True, '第 1 / 3 题')),
    ('b', 1, ('c', True, False, '第 3 / 3 题')),
    ('c', 1, ('c', True, False, '第 3 / 3 题')),
])
def test_navigation_order_and_boundaries(current, offset, expected):
    assert gendemo._navigate_pair(('a', 'b', 'c'), current, offset) == expected


def test_single_saved_pair_disables_both_directions():
    for offset in (-1, 0, 1):
        assert gendemo._navigate_pair(('only',), 'only', offset) == ('only', False, False, '第 1 / 1 题')


def test_saved_loader_rejects_incomplete_and_corrupt_pairs(saved_viewer, tmp_path):
    pair = deepcopy(next(iter(saved_viewer.pairs.values())))
    target = tmp_path / "bad.jsonl"
    # Test fixtures are output artifacts, not modifications to a real release.
    target.write_text(json.dumps(pair["H"]) + "\n")
    with pytest.raises(ValueError, match="Incomplete"):
        gendemo.load_pairs(target)
    target.write_text((json.dumps(pair["H"]) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        gendemo.load_pairs(target)
    pair["H"]["serialized"]["text"] += "corruption"
    with pytest.raises(ValueError, match="SHA256"):
        gendemo.validate_pair(pair["H"], pair["N"])
    pair = deepcopy(next(iter(saved_viewer.pairs.values())))
    pair["H"]["hallucination_spans"][0]["span"] = [0, 1]
    with pytest.raises(ValueError, match="offset/text"):
        gendemo.validate_pair(pair["H"], pair["N"])


def test_saved_graph_rejects_source_drift(saved_viewer):
    h = deepcopy(next(iter(saved_viewer.pairs.values()))["H"])
    h["mutation_events"][0]["before"] = "not the saved truth"
    with pytest.raises(ValueError, match="no longer matches"):
        gendemo.replay_graph(saved_viewer.references[h["origin_id"]], h)


def test_saved_highlight_escapes_html():
    text = "<script>"
    span = {"span": [0, len(text)], "text": text, "node_id": "atom",
            "pair_occurrence_id": 'node.01" onclick="alert(1)'}
    marked = gendemo._marked_html(text, [span])
    assert "<script>" not in marked
    assert "&lt;script&gt;" in marked
    assert 'onclick="alert' not in marked


def test_saved_script_runs_from_another_working_directory(tmp_path):
    result = subprocess.run([sys.executable, str(Path(gendemo.__file__)), "--help"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0
    assert "--records" in result.stdout and "--port" in result.stdout
