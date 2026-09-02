"""T036 content-addressed Poe cache and frozen replay tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from molhallulens.infrastructure.providers.poe.client import (
    POE_MODEL_ID,
    PoeAttemptStatus,
    PoeClientConfig,
    PoeClientProvenance,
    PoeClientResult,
    PoeToolExecution,
    PoeTransport,
    PoeTransportAttempt,
)
from molhallulens.infrastructure.providers.poe.response_cache import (
    CACHE_SCHEMA,
    CacheMode,
    PoeCacheContext,
    PoeResponseCache,
    PoeResponseCacheError,
    canonical_json_bytes,
)
from molhallulens.infrastructure.providers.poe.schemas import (
    FROZEN_GLOBAL_SEED,
    ProposalConstraints,
    ProposalManifestIdentity,
    ProposalRequest,
    ProposalResponse,
    derive_proposal_seed,
)

FIXED_TIME = "2026-08-30T03:00:00Z"


def _request() -> ProposalRequest:
    values: dict[str, Any] = {
        "request_id": "request:substitute:0216",
        "origin_id": "mol_edit.substitute_v2.0216",
        "operator_id": "mol_edit.substitute.incoming_fragment_bucket_swap",
        "propagation": "FULL_CF",
        "candidate_source_mode": "HYBRID",
        "target_root": "add_fragment",
        "constraints": ProposalConstraints(
            same_attachment_element=True,
            match_heavy_count=True,
            match_ring_count=True,
        ),
        "global_seed": FROZEN_GLOBAL_SEED,
        "dataset_version": "pilot_v1",
        "variant_index": 0,
        "split": "train",
        "manifest_identity": ProposalManifestIdentity(
            dataset_version="pilot_v1",
            split_seed=8347206628578381721,
            manifest_sha256="a" * 64,
            source_origin_audit_sha256="b" * 64,
            source_split_report_sha256="c" * 64,
        ),
    }
    values["derived_seed"] = derive_proposal_seed(
        global_seed=values["global_seed"],
        dataset_version=values["dataset_version"],
        origin_id=values["origin_id"],
        operator_id=values["operator_id"],
        policy=values["propagation"],
        variant_index=values["variant_index"],
    )
    return ProposalRequest.model_validate(values, strict=True)


def _response(request: ProposalRequest) -> ProposalResponse:
    return ProposalResponse.model_validate(
        {
            "proposal_version": "1.0",
            "request_id": request.request_id,
            "candidates": tuple(
                {
                    "candidate_id": f"candidate-{index}",
                    "root_field": request.target_root,
                    "replacement": {
                        "smiles": smiles,
                        "attachment_atom": 0,
                    },
                    "bond_edits": (),
                    "minimal_surface_realization": f"candidate fragment {smiles}",
                    "plausibility_reason": "A matched local chemical near-neighbor.",
                }
                for index, smiles in enumerate(("N", "O", "S"), start=1)
            ),
            "abstain_reason": None,
        },
        strict=True,
    )


def _result(request: ProposalRequest) -> PoeClientResult:
    execution = PoeToolExecution(
        turn=1,
        sequence=1,
        tool_call_id="tool-call-1",
        tool="analyze_smiles",
        arguments_json='{"smiles":"CCO"}',
        result_json=(
            '{"cache_key":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",'
            '"canonical_smiles":"CCO"}'
        ),
    )
    attempt = PoeTransportAttempt(
        transport=PoeTransport.RESPONSES,
        status=PoeAttemptStatus.SUCCEEDED,
        request_id=request.request_id,
        requested_model_id=POE_MODEL_ID,
        response_model=POE_MODEL_ID,
        response_ids=("resp-1", "resp-2"),
        query_ids=("query-1", "query-2"),
        x_request_ids=("x-request-1", "x-request-2"),
        turns=2,
        tool_executions=(execution,),
    )
    provenance = PoeClientProvenance(
        provider="poe",
        request_id=request.request_id,
        origin_id=request.origin_id,
        operator_id=request.operator_id,
        propagation=request.propagation,
        candidate_source_mode=request.candidate_source_mode,
        target_root=request.target_root,
        constraints_json=(
            '{"match_heavy_count":true,"match_ring_count":true,'
            '"not_equivalent_to_reference":true,"same_attachment_element":true,'
            '"sanitized":true}'
        ),
        requested_model_id=POE_MODEL_ID,
        selected_transport=PoeTransport.RESPONSES,
        attempts=(attempt,),
    )
    return PoeClientResult(
        request=request,
        response=_response(request),
        provenance=provenance,
    )


def _context(**overrides: object) -> PoeCacheContext:
    values: dict[str, object] = {
        "source_record_sha256": "1" * 64,
        "model_catalog_identity": "2" * 64,
        "operator_version": "substitution-v1",
        "attempt_index": 0,
        "tool_result_identities": ("d" * 64,),
    }
    values.update(overrides)
    return PoeCacheContext(**values)  # type: ignore[arg-type]


@dataclass
class _Producer:
    result: PoeClientResult
    calls: int = 0

    def propose(self, request: object) -> PoeClientResult:
        assert request == self.result.request
        self.calls += 1
        return self.result


def test_same_normalized_contract_hits_one_cache_entry(tmp_path: Path) -> None:
    request = _request()
    producer = _Producer(_result(request))
    cache = PoeResponseCache(tmp_path, clock=lambda: FIXED_TIME)
    first = cache.load_or_produce(
        request,
        context=_context(),
        producer=producer,
        config=PoeClientConfig(),
        usage={
            "input_tokens": 101,
            "output_tokens": 17,
            "total_tokens": 118,
            "cost_points": 9,
        },
    )
    # A catalog wrapper with the same verified entry identity and mapping order
    # differences must resolve to the same canonical key.
    second_context = _context(
        model_catalog_identity={
            "catalog_fetched_at": "2026-08-30T02:30:00Z",
            "catalog_entry_sha256": "2" * 64,
        }
    )
    second = cache.load_or_produce(
        request,
        context=second_context,
        producer=producer,
        config=PoeClientConfig(),
    )

    assert producer.calls == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.cache_key == second.cache_key
    assert second.result == first.result
    assert tuple((tmp_path / "proposals").glob("*.json")) == (first.path,)

    raw = first.path.read_bytes()
    record = json.loads(raw)
    assert raw == canonical_json_bytes(record) + b"\n"
    assert record["cache_schema"] == CACHE_SCHEMA
    assert record["schema_version"] == "1.0"
    assert record["created_at"] == FIXED_TIME
    assert record["key_material"]["requested_model_id"] == POE_MODEL_ID
    assert record["payload"]["request_id"] == request.request_id
    assert record["payload"]["response_ids"] == ["resp-1", "resp-2"]
    assert record["payload"]["query_ids"] == ["query-1", "query-2"]
    assert record["payload"]["x_request_ids"] == [
        "x-request-1",
        "x-request-2",
    ]
    assert record["payload"]["transport"] == "responses"
    assert record["payload"]["usage"]["cost_points"] == 9
    assert len(record["payload"]["tool_transcript"]) == 1


def test_catalog_schema_tool_result_and_attempt_identities_partition_cache(
    tmp_path: Path,
) -> None:
    request = _request()
    producer = _Producer(_result(request))
    cache = PoeResponseCache(tmp_path, clock=lambda: FIXED_TIME)
    contexts = (
        _context(),
        _context(model_catalog_identity="3" * 64),
        _context(tool_result_identities=("e" * 64,)),
        _context(proposal_schema_identity="f" * 64),
        _context(attempt_index=1),
    )

    keys = {
        cache.load_or_produce(
            request,
            context=context,
            producer=producer,
        ).cache_key
        for context in contexts
    }

    assert len(keys) == len(contexts)
    assert producer.calls == len(contexts)


def test_frozen_and_release_replay_never_call_producer(tmp_path: Path) -> None:
    request = _request()
    producer = _Producer(_result(request))
    PoeResponseCache(tmp_path, clock=lambda: FIXED_TIME).load_or_produce(
        request,
        context=_context(),
        producer=producer,
    )
    assert producer.calls == 1

    replay_producer = _Producer(_result(request))
    frozen = PoeResponseCache(tmp_path, mode=CacheMode.FROZEN_REPLAY)
    replayed = frozen.load_or_produce(
        request,
        context=_context(),
        producer=replay_producer,
    )
    assert replayed.cache_hit is True
    assert replayed.result.response == _response(request)
    assert replay_producer.calls == 0

    release = PoeResponseCache(tmp_path, release_mode=True)
    with pytest.raises(PoeResponseCacheError) as captured:
        release.load_or_produce(
            request,
            context=_context(attempt_index=9),
            producer=replay_producer,
        )
    assert captured.value.code == "CACHE_MISS_FROZEN"
    assert replay_producer.calls == 0


def test_corrupt_or_mismatched_replay_fails_closed_before_producer(
    tmp_path: Path,
) -> None:
    request = _request()
    initial = PoeResponseCache(tmp_path, clock=lambda: FIXED_TIME).load_or_produce(
        request,
        context=_context(),
        producer=_Producer(_result(request)),
    )
    initial.path.write_text("{}\n", encoding="utf-8")
    never = _Producer(_result(request))

    with pytest.raises(PoeResponseCacheError) as captured:
        PoeResponseCache(tmp_path, mode=CacheMode.FROZEN_REPLAY).load_or_produce(
            request,
            context=_context(),
            producer=never,
        )

    assert captured.value.code == "CACHE_ENTRY_CORRUPT"
    assert never.calls == 0


def test_raw_headers_and_credentials_are_rejected_without_persistence(
    tmp_path: Path,
) -> None:
    request = _request()
    producer = _Producer(_result(request))
    cache = PoeResponseCache(tmp_path, clock=lambda: FIXED_TIME)

    with pytest.raises(PoeResponseCacheError) as captured:
        cache.load_or_produce(
            request,
            context=_context(),
            producer=producer,
            raw_request={
                "headers": {
                    "Authorization": "Bearer poe-test-secret-never-persist",
                }
            },
        )
    assert captured.value.code == "CACHE_SECRET_MATERIAL_REJECTED"
    assert not tuple(tmp_path.rglob("*.json"))

    with pytest.raises(PoeResponseCacheError) as accepted_error:
        cache.store_accepted_artifact(
            {"record_id": "accepted-1"},
            provenance={"api_key": "poe-test-secret-never-persist"},
        )
    assert accepted_error.value.code == "CACHE_SECRET_MATERIAL_REJECTED"
    assert not tuple(tmp_path.rglob("*.json"))


def test_accepted_artifact_is_named_and_verified_by_its_content_identity(
    tmp_path: Path,
) -> None:
    cache = PoeResponseCache(tmp_path, clock=lambda: FIXED_TIME)
    artifact = {
        "record_id": "mol-edit-accepted-1",
        "candidate": {"smiles": "CCN", "root": "add_fragment"},
        "reject_codes": [],
    }
    stored = cache.store_accepted_artifact(
        artifact,
        provenance={"validator_version": "1.0"},
        request_id="request:substitute:0216",
        response_ids=("resp-2",),
        transport="responses",
        usage={"total_tokens": 118, "cost_points": 9},
    )

    assert stored.artifact_sha256 is not None
    assert stored.content_identity == stored.artifact_sha256
    assert stored.path.name == f"{stored.artifact_sha256}.json"
    assert cache.load_accepted_artifact(stored.artifact_sha256) == artifact
    record = json.loads(stored.path.read_bytes())
    assert record["artifact_sha256"] == stored.artifact_sha256
    assert record["payload"]["artifact_sha256"] == stored.artifact_sha256

    repeated = cache.store_accepted_artifact(
        {
            "reject_codes": [],
            "candidate": {"root": "add_fragment", "smiles": "CCN"},
            "record_id": "mol-edit-accepted-1",
        },
        provenance={"validator_version": "1.0"},
        request_id="request:substitute:0216",
        response_ids=("resp-2",),
        transport="responses",
        usage={"cost_points": 9, "total_tokens": 118},
    )
    assert repeated.cache_hit is True
    assert repeated.artifact_sha256 == stored.artifact_sha256


def test_tool_run_and_render_namespaces_are_immutable(tmp_path: Path) -> None:
    cache = PoeResponseCache(tmp_path, clock=lambda: FIXED_TIME)
    tool = cache.store_tool_run(
        key_material={"tool": "analyze_smiles", "arguments": {"smiles": "CCO"}},
        result={"canonical_smiles": "CCO", "atom_count": 3},
        provenance={"tool_result_version": "1.0"},
    )
    render = cache.store_render(
        key_material={"trace_id": "trace-1", "renderer_version": "1.0"},
        render={"text": "A deterministic rendering."},
    )
    assert tool.path.parent.name == "tool_runs"
    assert render.path.parent.name == "renders"
    assert cache.load_tool_run(
        key_material={"arguments": {"smiles": "CCO"}, "tool": "analyze_smiles"}
    ) == {"canonical_smiles": "CCO", "atom_count": 3}
    assert cache.load_render(
        key_material={"renderer_version": "1.0", "trace_id": "trace-1"}
    ) == {"text": "A deterministic rendering."}

    with pytest.raises(PoeResponseCacheError) as captured:
        cache.store_tool_run(
            key_material={
                "arguments": {"smiles": "CCO"},
                "tool": "analyze_smiles",
            },
            result={"canonical_smiles": "CCN", "atom_count": 3},
            provenance={"tool_result_version": "1.0"},
        )
    assert captured.value.code == "CACHE_IMMUTABILITY_VIOLATION"


def test_frozen_cache_rejects_all_artifact_writes(tmp_path: Path) -> None:
    cache = PoeResponseCache(tmp_path, mode=CacheMode.FROZEN_REPLAY)
    with pytest.raises(PoeResponseCacheError) as captured:
        cache.store_accepted_artifact({"record_id": "not-written"})
    assert captured.value.code == "CACHE_FROZEN_WRITE_REJECTED"
    assert not tuple(tmp_path.rglob("*.json"))
