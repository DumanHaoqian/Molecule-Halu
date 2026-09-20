"""CPU-only checks of request preparation, checkpointing and molecular scoring."""
from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import pytest

from scripts import run_chemLLM_outcome as runner


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    metrics = runner.metrics_module()
    monkeypatch.setattr(runner, "metrics_module", lambda: metrics)
    (tmp_path / "ChemCoTBench-V2").symlink_to(runner.ROOT / "ChemCoTBench-V2", target_is_directory=True)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    raw_dir = tmp_path / "Pilot/Dataset/raw_benchmark_data/mol_edit"
    raw_dir.mkdir(parents=True)
    records = []
    for subtask in ("add", "delete", "substitute"):
        source = "[CH3:1][CH3:2]"
        raw = {"anonymous_sample_id": subtask, "indexed_smiles": source,
               "instruction": "Make ethanol", "gt_smiles": "CCO"}
        (raw_dir / f"{subtask}_pilot_origin.json").write_text(json.dumps([raw]))
        for label in ("H", "N"):
            records.append({
                "pair_id": subtask, "origin_id": subtask, "subtask": subtask,
                "variant_label": label, "record_id": f"{subtask}:{label}", "edit_count": 2,
                "detector_input": {"indexed_smiles": source, "instruction": raw["instruction"],
                                   "final_answer": "CCO", "reasoning_chain": f"  {subtask} {label}: keep CC intact.\n"},
            })
    dataset = tmp_path / "records.jsonl"
    write_rows(dataset, records)
    args = SimpleNamespace(output=tmp_path / "experiment", dataset=dataset,
                           model=tmp_path / "model", model_name="ChemDFM-R-14B", pairs_per_subtask=1,
                           batch_size=8, tensor_parallel_size=2, limit=None, allow_partial=False)
    return args, records


def test_prepare_preserves_dataset_reasoning_without_hashes(experiment):
    args, records = experiment
    before = args.dataset.read_bytes()
    runner.prepare(args)
    requests = runner.read_jsonl(args.output / "requests.jsonl")
    assert len(requests) == 15
    by_record = {(r["pair_id"], r["variant_label"]): r for r in records}
    for request in requests:
        group = request["group"]
        if group in "BC":
            record = by_record[request["pair_id"], "H" if group == "B" else "N"]
            assert request["messages"][1]["content"].endswith(record["detector_input"]["reasoning_chain"])
        if group == "E":
            source = request["reasoning_source"]
            donor = by_record[source["pair_id"], "N"]
            assert source["origin_id"] != request["origin_id"]
            assert source["record_id"] == donor["record_id"]
            assert source["subtask"] == donor["subtask"]
            prompt = request["messages"][1]["content"]
            assert prompt.endswith(donor["detector_input"]["reasoning_chain"])
            c_request = next(q for q in requests if q["pair_id"] == request["pair_id"] and q["group"] == "C")
            own = by_record[request["pair_id"], "N"]["detector_input"]["reasoning_chain"]
            assert prompt == c_request["messages"][1]["content"].removesuffix(own) + donor["detector_input"]["reasoning_chain"]
            assert request["messages"][0]["content"] == runner.FOLLOW_REASONING
        if group in "BCE":
            assert request["messages"][0]["content"] == runner.FOLLOW_REASONING
            assert "Intermediate claims may be wrong" not in request["messages"][1]["content"]
            assert "independently determine the answer" not in request["messages"][1]["content"]
        elif group == "A":
            assert request["messages"][0]["content"] == runner.DIRECT
        else:
            assert request["messages"][0]["content"] == runner.COT
        assert request["assistant_prefix"] == ("<think>\n" if group == "D" else "<think>\n</think>\n<answer>\n")
    for filename in ("requests.jsonl", "ground_truth.jsonl", "manifest.json"):
        assert "sha256" not in (args.output / filename).read_text()
    assert args.dataset.read_bytes() == before
    with pytest.raises(RuntimeError, match="Manifest already exists"):
        runner.prepare(args)


def test_donors_are_deterministic_same_subtask_and_different_origin():
    pairs = {}
    for task in ("add", "delete", "substitute"):
        for origin in ("first", "second"):
            for variant in (0, 1):
                key = f"{task}-{origin}-{variant}"
                pairs[key] = {"N": {"origin_id": f"{task}-{origin}", "subtask": task}}
    donors = runner.select_reasoning_donors(pairs)
    assert donors == runner.select_reasoning_donors(dict(reversed(list(pairs.items()))))
    for target, source in donors.items():
        assert pairs[target]["N"]["origin_id"] != pairs[source]["N"]["origin_id"]
        assert pairs[target]["N"]["subtask"] == pairs[source]["N"]["subtask"]


def test_single_origin_cannot_supply_e_reasoning(experiment):
    args, records = experiment
    write_rows(args.dataset, records[:2])
    with pytest.raises(ValueError, match="different origin"):
        runner.prepare(args)
    assert not (args.output / "manifest.json").exists()


def test_interrupt_resume_and_summarize(experiment, monkeypatch):
    args, _ = experiment
    runner.prepare(args)
    requests = runner.read_jsonl(args.output / "requests.jsonl")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _: "test")
    tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **kw: json.dumps(messages),
        encode=lambda text, **kw: [ord(c) for c in text], eos_token_id=2,
        convert_tokens_to_ids=lambda token: 3,
        get_vocab=lambda: {}, all_special_ids=[2],
    )
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer)))
    calls = []
    class Engine:
        def __init__(self, **kwargs):
            pass

        def generate(self, prompts, params, **kwargs):
            assert all(isinstance(p, dict) and "prompt_token_ids" in p for p in prompts)
            calls.append(prompts)
            texts = ["".join(chr(i) for i in p["prompt_token_ids"]) for p in prompts]
            return [SimpleNamespace(prompt_token_ids=[1, 2], outputs=[SimpleNamespace(
                text=("" if prompt.endswith("<answer>\n") else "Done.</think><answer>")
                     + "[CH3:1][CH2:2][OH:3]</answer>", token_ids=[3],
                finish_reason="abort" if len(calls) == 1 and i == 1 else "stop", stop_reason=None,
            )]) for i, prompt in enumerate(texts)]
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=Engine, SamplingParams=lambda **kw: kw))
    with pytest.raises(RuntimeError, match="interruption"):
        runner.run(args)
    saved = runner.read_jsonl(args.output / "predictions.jsonl")
    assert len(saved) == 7
    assert requests[1]["request_id"] not in {r["request_id"] for r in saved}
    runner.run(args)
    assert len(calls[1]) == 8
    assert len(runner.read_jsonl(args.output / "predictions.jsonl")) == 15
    runner.run(args)
    assert len(calls) == 2
    runner.summarize(args)
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["complete"]
    for group in "ABCDE":
        assert summary["groups"][group]["primary_accuracy"] == 1
        assert summary["groups"][group]["atom_mapped_outputs"] == 3
    for comparison in ("C_vs_B", "E_vs_A", "C_vs_E", "B_vs_E"):
        assert summary["paired_primary_comparisons"][comparison]["both_correct"] == 3
    changed = deepcopy(requests)
    changed[0]["messages"][1]["content"] += " changed"
    write_rows(args.output / "requests.jsonl", changed)
    with pytest.raises(ValueError, match="request"):
        runner.run(args)


def test_empty_partial_summary(experiment):
    args, _ = experiment
    runner.prepare(args)
    args.allow_partial = True
    runner.summarize(args)
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["n_completed"] == 0
    assert not summary["complete"]


@pytest.mark.parametrize("name,folder", [
    ("Chem-R-8B", "chem_r8b_outcome_abcde_follow_reasoning"),
    ("ChemDFM-R-14B", "chemdfm_r14b_outcome_abcde_follow_reasoning"),
])
def test_model_selection_and_output_directory(name, folder):
    for command in ("prepare", "run", "summarize"):
        args = runner.parse_args([command, "--model", name])
        assert args.model_name == name
        assert args.model == runner.MODELS[name]
        assert args.output.name == folder


def test_model_path_override(tmp_path):
    args = runner.parse_args(["prepare", "--model", "Chem-R-8B", "--model-path", str(tmp_path)])
    assert args.model == tmp_path
    assert args.model_name == "Chem-R-8B"


def test_four_gpu_configuration(experiment):
    args, _ = experiment
    parsed = runner.parse_args(["prepare", "--tensor-parallel-size", "4"])
    args.tensor_parallel_size = parsed.tensor_parallel_size
    runner.prepare(args)
    manifest, _ = runner.load_experiment(args.output)
    assert manifest["engine"]["tensor_parallel_size"] == 4


def test_stop_tokens_do_not_include_unknown_token(tmp_path):
    tokenizer = SimpleNamespace(eos_token_id=128009, all_special_ids=[128001, 128009],
                                get_vocab=lambda: {"<|end_of_text|>": 128001, "<|eot_id|>": 128009})
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [128001, 128009]}))
    assert runner.stop_token_ids(tokenizer, tmp_path) == [128001, 128009]


def test_old_correction_protocol_cannot_resume(experiment):
    args, _ = experiment
    runner.prepare(args)
    path = args.output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["protocol_version"] = "outcome_abcde_saved_reasoning"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fresh output directory"):
        runner.load_experiment(args.output)


def test_wrong_model_cannot_resume_or_summarize(experiment):
    args, _ = experiment
    runner.prepare(args)
    args.model_name = "Chem-R-8B"
    for command in (runner.run, runner.summarize):
        with pytest.raises(ValueError, match="Selected model differs"):
            command(args)


@pytest.mark.parametrize("name", ["Chem-R-8B", "ChemDFM-R-14B"])
def test_local_model_chat_template_and_stops(name):
    path = runner.MODELS[name]
    if not path.exists():
        pytest.skip("Local model is not available")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    for group in "ABCDE":
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": runner.COT if group == "D" else runner.FOLLOW_REASONING if group in "BCE" else runner.DIRECT},
             {"role": "user", "content": "Return ethanol SMILES."}],
            tokenize=False, add_generation_prompt=True,
        ) + runner.PREFIXES[group]
        assert prompt.endswith(runner.PREFIXES[group])
        assert len(tokenizer.encode(prompt, add_special_tokens=False)) > 0
    stops = runner.stop_token_ids(tokenizer, path)
    assert tokenizer.eos_token_id in stops
    assert set(stops) <= set(tokenizer.all_special_ids)
