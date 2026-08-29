"""Build multi-axis character annotations from graph mutations and mentions.

Only exact ``(target_kind, state_or_edge_id)`` joins are allowed.  Pure
omissions are retained in a separate unlocalized ledger and never projected
onto a neighboring literal, claim boundary, or final-answer token.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from molhallulens.annotation.char_spans import MentionSpan
from molhallulens.domain import (
    CausalRole,
    CharAnnotation,
    EditErrorSubtype,
    EvidenceRelation,
    GraphDelta,
    HallucinationType,
    MutationEvent,
    MutationTargetKind,
    SegmentKind,
)
from molhallulens.rendering.trace_ast import RenderedExample

CHAR_ANNOTATION_BUILDER_VERSION = "char_annotation_builder_v1"


class CharAnnotationBuildError(RuntimeError):
    """Structured fail-closed error without rendered or molecular payloads."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        event_id: str | None = None,
        evidence: Mapping[str, object] | None = None,
    ) -> None:
        if type(code) is not str or not code:
            raise ValueError("error code must be non-empty text")
        if type(detail) is not str or not detail:
            raise ValueError("error detail must be non-empty text")
        if event_id is not None and (type(event_id) is not str or not event_id):
            raise ValueError("event_id must be non-empty text or None")
        if evidence is not None and not isinstance(evidence, Mapping):
            raise TypeError("error evidence must be a mapping or None")
        self.code = code
        self.detail = detail
        self.event_id = event_id
        self.evidence = MappingProxyType(dict(evidence or {}))
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class UnlocalizedOmission:
    """A pure omission intentionally excluded from character/token projection."""

    event_id: str
    target_kind: MutationTargetKind
    state_or_edge_id: str
    suppressed_mention_ids: tuple[str, ...] = ()
    reason: str = "PURE_OMISSION_HAS_NO_LITERAL_SPAN"

    def __post_init__(self) -> None:
        for value, name in (
            (self.event_id, "event_id"),
            (self.state_or_edge_id, "state_or_edge_id"),
            (self.reason, "reason"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty text")
        if type(self.target_kind) is not MutationTargetKind:
            raise TypeError("target_kind must be MutationTargetKind")
        mentions = tuple(self.suppressed_mention_ids)
        if any(type(item) is not str or not item for item in mentions):
            raise TypeError("suppressed_mention_ids must contain non-empty strings")
        if len(mentions) != len(set(mentions)):
            raise ValueError("suppressed omission mention IDs must be unique")
        object.__setattr__(self, "suppressed_mention_ids", mentions)


@dataclass(frozen=True, slots=True)
class EventAnnotationLink:
    """Audit join from one mutation event to its emitted annotation spans."""

    event_id: str
    span_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.event_id) is not str or not self.event_id:
            raise ValueError("event_id must be non-empty text")
        spans = tuple(self.span_ids)
        if any(type(item) is not str or not item for item in spans):
            raise TypeError("span_ids must contain non-empty strings")
        if len(spans) != len(set(spans)):
            raise ValueError("event annotation span IDs must be unique")
        object.__setattr__(self, "span_ids", spans)


@dataclass(frozen=True, slots=True)
class CharAnnotationBuildResult:
    annotations: tuple[CharAnnotation, ...]
    event_links: tuple[EventAnnotationLink, ...]
    unlocalized_omissions: tuple[UnlocalizedOmission, ...] = ()
    builder_version: str = CHAR_ANNOTATION_BUILDER_VERSION

    def __post_init__(self) -> None:
        annotations = tuple(self.annotations)
        links = tuple(self.event_links)
        omissions = tuple(self.unlocalized_omissions)
        if any(type(item) is not CharAnnotation for item in annotations):
            raise TypeError("annotations must contain CharAnnotation values")
        if any(type(item) is not EventAnnotationLink for item in links):
            raise TypeError("event_links must contain EventAnnotationLink values")
        if any(type(item) is not UnlocalizedOmission for item in omissions):
            raise TypeError(
                "unlocalized_omissions must contain UnlocalizedOmission values"
            )
        if (
            tuple(
                sorted(
                    annotations,
                    key=lambda item: (
                        item.literal_span.start,
                        item.literal_span.end,
                        item.span_id,
                    ),
                )
            )
            != annotations
        ):
            raise ValueError("annotations must use rendered occurrence order")
        annotation_ids = tuple(item.span_id for item in annotations)
        linked_ids = tuple(span_id for link in links for span_id in link.span_ids)
        if len(annotation_ids) != len(set(annotation_ids)):
            raise ValueError("annotation span IDs must be unique")
        if sorted(annotation_ids) != sorted(linked_ids):
            raise ValueError("event links must exactly cover emitted annotations")
        link_ids = tuple(link.event_id for link in links)
        omission_ids = tuple(item.event_id for item in omissions)
        if len(link_ids) != len(set(link_ids)) or len(omission_ids) != len(
            set(omission_ids)
        ):
            raise ValueError("event and omission IDs must be unique")
        if set(link_ids).intersection(omission_ids):
            raise ValueError("an event cannot be localized and pure omission")
        if self.builder_version != CHAR_ANNOTATION_BUILDER_VERSION:
            raise ValueError("unsupported char annotation builder version")
        object.__setattr__(self, "annotations", annotations)
        object.__setattr__(self, "event_links", links)
        object.__setattr__(self, "unlocalized_omissions", omissions)

    @property
    def has_unlocalized_omissions(self) -> bool:
        return bool(self.unlocalized_omissions)


def is_pure_omission(event: MutationEvent) -> bool:
    if type(event) is not MutationEvent:
        raise TypeError("event must be MutationEvent")
    return event.hallucination_types == frozenset({HallucinationType.OMISSION})


def derive_evidence_relations(
    event: MutationEvent,
    *,
    additional: Iterable[EvidenceRelation] = (),
) -> frozenset[EvidenceRelation]:
    """Derive auditable evidence axes without inspecting detector text.

    Every GraphDelta event differs from the verified reference state.  Other
    relations follow only from explicit semantic/subtype axes or caller-supplied
    audited relations; no source/instruction relation is guessed from wording.
    """

    if type(event) is not MutationEvent:
        raise TypeError("event must be MutationEvent")
    if isinstance(additional, (str, bytes)):
        raise TypeError("additional evidence relations must be a collection")
    extras = frozenset(additional)
    if any(type(item) is not EvidenceRelation for item in extras):
        raise TypeError("additional values must be EvidenceRelation members")
    relations = {EvidenceRelation.CONTRADICTS_REFERENCE_STATE, *extras}
    if HallucinationType.UNSUPPORTED in event.hallucination_types:
        relations.add(EvidenceRelation.UNSUPPORTED_BY_EVIDENCE)
    if HallucinationType.CONSTRAINT_VIOLATION in event.hallucination_types:
        relations.add(EvidenceRelation.CONTRADICTS_INSTRUCTION)
    if EditErrorSubtype.INTERNAL_INCONSISTENCY in event.edit_subtypes:
        relations.add(EvidenceRelation.INTERNAL_INCONSISTENCY)
    return frozenset(relations)


def _additional_relations(
    graph_delta: GraphDelta,
    values: Mapping[str, Iterable[EvidenceRelation]] | None,
) -> Mapping[str, frozenset[EvidenceRelation]]:
    if values is None:
        return MappingProxyType({})
    if not isinstance(values, Mapping):
        raise TypeError("additional_evidence_relations must be a mapping or None")
    event_ids = {event.event_id for event in graph_delta.events}
    unknown = tuple(sorted(set(values) - event_ids))
    if unknown:
        raise CharAnnotationBuildError(
            "UNKNOWN_EVIDENCE_EVENT",
            "additional evidence references an unknown mutation event",
            evidence={"event_ids": unknown},
        )
    normalized: dict[str, frozenset[EvidenceRelation]] = {}
    for event_id, raw_relations in values.items():
        if type(event_id) is not str or not event_id:
            raise TypeError("evidence event IDs must be non-empty strings")
        if isinstance(raw_relations, (str, bytes)):
            raise TypeError("evidence relations must be non-string collections")
        relations = frozenset(raw_relations)
        if any(type(item) is not EvidenceRelation for item in relations):
            raise TypeError("evidence relations must contain EvidenceRelation values")
        normalized[event_id] = relations
    return MappingProxyType(normalized)


def _mentions_by_target(
    rendered: RenderedExample,
) -> Mapping[tuple[MutationTargetKind, str], tuple[MentionSpan, ...]]:
    grouped: defaultdict[tuple[MutationTargetKind, str], list[MentionSpan]] = (
        defaultdict(list)
    )
    for mention in rendered.mentions:
        grouped[(mention.target_kind, mention.state_or_edge_id)].append(mention)
    return MappingProxyType(
        {
            target: tuple(
                sorted(
                    mentions,
                    key=lambda item: (
                        item.literal_span.start,
                        item.literal_span.end,
                        item.mention_id,
                    ),
                )
            )
            for target, mentions in grouped.items()
        }
    )


def _annotation_span_id(mention: MentionSpan) -> str:
    return f"char:{mention.mention_id}"


def _validate_mention_component(event: MutationEvent, mention: MentionSpan) -> None:
    propagated_final_answer = (
        event.target_kind is MutationTargetKind.NODE
        and event.node_or_edge_id == "final_answer"
        and event.causal_role
        in {CausalRole.PROPAGATED_FALSE, CausalRole.PROPAGATED_CONDITIONAL}
    )
    expected = (
        SegmentKind.FINAL_ANSWER
        if event.causal_role is CausalRole.TERMINAL or propagated_final_answer
        else SegmentKind.REASONING
    )
    if mention.component is not expected:
        raise CharAnnotationBuildError(
            "CAUSAL_COMPONENT_MISMATCH",
            "mutation mention appears in the wrong detector component",
            event_id=event.event_id,
            evidence={
                "causal_role": event.causal_role.value,
                "expected_component": expected.value,
                "actual_component": mention.component.value,
            },
        )


@dataclass(frozen=True, slots=True)
class CharAnnotationBuilder:
    """Project exact rendered mutation mentions into frozen multi-axis labels."""

    builder_version: str = CHAR_ANNOTATION_BUILDER_VERSION

    def __post_init__(self) -> None:
        if self.builder_version != CHAR_ANNOTATION_BUILDER_VERSION:
            raise ValueError("char annotation builder version is frozen")

    def build(
        self,
        graph_delta: GraphDelta,
        rendered: RenderedExample,
        *,
        additional_evidence_relations: Mapping[str, Iterable[EvidenceRelation]]
        | None = None,
    ) -> CharAnnotationBuildResult:
        if type(graph_delta) is not GraphDelta:
            raise TypeError("graph_delta must be GraphDelta")
        if type(rendered) is not RenderedExample:
            raise TypeError("rendered must be RenderedExample")
        extras = _additional_relations(graph_delta, additional_evidence_relations)
        mentions_by_target = _mentions_by_target(rendered)
        events = tuple(graph_delta.events)
        if not events:
            return CharAnnotationBuildResult(annotations=(), event_links=())

        root_event = graph_delta.root_events[0]
        root_target = (root_event.target_kind, root_event.node_or_edge_id)
        root_mentions = mentions_by_target.get(root_target, ())
        root_is_omission = is_pure_omission(root_event)
        if root_is_omission and any(
            event.event_id != root_event.event_id and not is_pure_omission(event)
            for event in events
        ):
            raise CharAnnotationBuildError(
                "UNLOCALIZED_ROOT_HAS_PROPAGATION",
                "a pure-omission root cannot anchor localized propagated spans",
                event_id=root_event.event_id,
            )
        if not root_is_omission and not root_mentions:
            raise CharAnnotationBuildError(
                "ROOT_MENTION_MISSING",
                "independent mutation root has no exact rendered mention",
                event_id=root_event.event_id,
                evidence={"target": root_event.node_or_edge_id},
            )
        canonical_root_span_id = (
            None if root_is_omission else _annotation_span_id(root_mentions[0])
        )

        annotations_with_events: list[tuple[CharAnnotation, str]] = []
        omissions: list[UnlocalizedOmission] = []
        for event in events:
            target = (event.target_kind, event.node_or_edge_id)
            mentions = mentions_by_target.get(target, ())
            if is_pure_omission(event):
                omissions.append(
                    UnlocalizedOmission(
                        event_id=event.event_id,
                        target_kind=event.target_kind,
                        state_or_edge_id=event.node_or_edge_id,
                        suppressed_mention_ids=tuple(
                            mention.mention_id for mention in mentions
                        ),
                    )
                )
                continue
            if not mentions:
                raise CharAnnotationBuildError(
                    "MUTATION_MENTION_MISSING",
                    "adjudicated non-omission mutation has no exact rendered mention",
                    event_id=event.event_id,
                    evidence={
                        "target_kind": event.target_kind.value,
                        "target": event.node_or_edge_id,
                    },
                )
            evidence_relations = derive_evidence_relations(
                event,
                additional=extras.get(event.event_id, frozenset()),
            )
            for mention in mentions:
                _validate_mention_component(event, mention)
                span_id = _annotation_span_id(mention)
                root_span_id = (
                    span_id
                    if event.causal_role in {CausalRole.ROOT, CausalRole.TERMINAL}
                    else canonical_root_span_id
                )
                if root_span_id is None:
                    raise CharAnnotationBuildError(
                        "ROOT_SPAN_UNAVAILABLE",
                        "propagated mutation cannot link to an unlocalized root",
                        event_id=event.event_id,
                    )
                annotations_with_events.append(
                    (
                        CharAnnotation(
                            span_id=span_id,
                            component=mention.component,
                            step_index=mention.step_index,
                            state_or_edge_id=event.node_or_edge_id,
                            literal_span=mention.literal_span,
                            claim_span=mention.claim_span,
                            semantic_types=event.hallucination_types,
                            edit_subtypes=event.edit_subtypes,
                            evidence_relations=evidence_relations,
                            causal_role=event.causal_role,
                            root_span_id=root_span_id,
                        ),
                        event.event_id,
                    )
                )

        ordered = tuple(
            sorted(
                annotations_with_events,
                key=lambda item: (
                    item[0].literal_span.start,
                    item[0].literal_span.end,
                    item[0].span_id,
                ),
            )
        )
        spans_by_event: defaultdict[str, list[str]] = defaultdict(list)
        for annotation, event_id in ordered:
            spans_by_event[event_id].append(annotation.span_id)
        links = tuple(
            EventAnnotationLink(
                event_id=event.event_id,
                span_ids=tuple(spans_by_event[event.event_id]),
            )
            for event in events
            if event.event_id in spans_by_event
        )
        return CharAnnotationBuildResult(
            annotations=tuple(annotation for annotation, _ in ordered),
            event_links=links,
            unlocalized_omissions=tuple(omissions),
            builder_version=self.builder_version,
        )


def build_char_annotations(
    graph_delta: GraphDelta,
    rendered: RenderedExample,
    *,
    additional_evidence_relations: Mapping[str, Iterable[EvidenceRelation]]
    | None = None,
) -> CharAnnotationBuildResult:
    return CharAnnotationBuilder().build(
        graph_delta,
        rendered,
        additional_evidence_relations=additional_evidence_relations,
    )


__all__ = [
    "CHAR_ANNOTATION_BUILDER_VERSION",
    "CharAnnotationBuildError",
    "CharAnnotationBuildResult",
    "CharAnnotationBuilder",
    "EventAnnotationLink",
    "UnlocalizedOmission",
    "build_char_annotations",
    "derive_evidence_relations",
    "is_pure_omission",
]
