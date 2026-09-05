from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import pytest

from scripts.merge_recovered_pairs import merge, validate_pair
from molhallulens.config import DEFAULT_HALLUCINATION_CONFIG
from molhallulens.modules.annotation import UnifiedHallucinationAnnotator
from molhallulens.modules.error_injection import UnifiedHallucinationInjector
from molhallulens.modules.error_planning import UnifiedHallucinationPlanner
from molhallulens.modules.release import UnifiedRecordBuilder
from molhallulens.modules.text_realization import DeterministicTextRenderer, MatchedNegativeTextBuilder


@pytest.fixture
def recovered_pair(all_references, fragment_pool):
    reference = all_references[0]
    config = replace(DEFAULT_HALLUCINATION_CONFIG, edit_count_mode='maximum')
    injected = UnifiedHallucinationInjector(config).apply(
        reference.state_dag, UnifiedHallucinationPlanner(fragment_pool, config).plan(reference, variant_index=0))
    rendered = DeterministicTextRenderer().render(reference, injected)
    annotator = UnifiedHallucinationAnnotator()
    positive = annotator.annotate(rendered, injected)
    pair = MatchedNegativeTextBuilder().build(reference, injected, rendered)
    negative = annotator.annotate_negative(pair.negative, positive)
    h, n = UnifiedRecordBuilder().build_pair(reference, injected, pair, positive, negative)
    return h.data, n.data


def test_recovery_checks_spans_pairing_and_prose(recovered_pair):
    h, n = recovered_pair
    validate_pair(h, n)
    bad = deepcopy(n)
    bad['control_spans'][0]['text'] += 'x'
    with pytest.raises(ValueError, match='round-trip'):
        validate_pair(h, bad)
    bad = deepcopy(n)
    # A grammatically innocuous suffix outside a span is still forbidden.
    bad['step_texts'][-1] += ' '
    bad['detector_input']['reasoning_chain'] += ' '
    text = bad['serialized']['text'].replace('\n\n<FINAL_ANSWER>', ' \n\n<FINAL_ANSWER>')
    bad['serialized'] = {'text': text, 'sha256': hashlib.sha256(text.encode()).hexdigest()}
    for span in bad['control_spans']:
        if span['component'] == 'final_answer':
            for key in ('serialized_span', 'serialized_context_span'):
                span[key] = [v + 1 for v in span[key]]
    with pytest.raises(ValueError, match='substitution invariant'):
        validate_pair(h, bad)


def test_merge_refuses_incomplete_dataset_without_writing(recovered_pair, tmp_path, project_root):
    source, output = tmp_path / 'one.jsonl', tmp_path / 'merged.jsonl'
    source.write_text(''.join(json.dumps(r) + '\n' for r in recovered_pair))
    with pytest.raises(ValueError, match='incomplete origin coverage'):
        merge([source], output, project_root / 'Dataset')
    assert not output.exists()


def test_merge_preserves_inputs_and_corrects_only_known_metadata(
    recovered_pair, tmp_path, project_root, monkeypatch, all_references, fragment_pool,
):
    import scripts.merge_recovered_pairs as merger
    monkeypatch.setattr(merger.ChemCoTMolEditAdapter, 'load', lambda self, path: all_references[:1])
    monkeypatch.setattr(merger, 'build_reference_dag', lambda reference: reference)
    monkeypatch.setattr(merger.FragmentPool, 'from_reference_artifacts', lambda refs: fragment_pool)
    h, n = deepcopy(recovered_pair)
    h['text_realization']['protocol_version'] = 'poe_segments_v15'
    h['text_realization']['step_execution'] = [{'protocol': 'poe_segments_v18', 'response_mode': 'poe_selected_local_draft'}]
    source, output = tmp_path / 'one.jsonl', tmp_path / 'merged.jsonl'
    source.write_text(''.join(json.dumps(r) + '\n' for r in (h, n)))
    original = source.read_bytes()
    report = merge([source], output, project_root / 'Dataset')
    assert report['origins'] == 1 and report['records'] == 2
    assert source.read_bytes() == original
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[0]['serialized'] == h['serialized']
    assert rows[0]['text_realization']['protocol_version'] == 'poe_segments_v18'
    assert rows[1] == json.loads(json.dumps(n))
    assert report['protocol_metadata_corrections'] == [h['record_id']]
    with pytest.raises(ValueError, match='overwrite'):
        merge([source], output, project_root / 'Dataset')
