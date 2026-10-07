"""Chat context is a verbatim, deterministic projection of the handoff."""

import pytest
from pydantic import ValidationError

from src.api.main import _run_demo_analysis
from src.integration import (
    CHAT_SYSTEM_PROMPT,
    ChatRequest,
    build_chat_context,
    citation_index,
)
from src.visualization.presentation import (
    chat_citation_labels,
    cited_sources,
    text_for_speech,
)


def _handoff():
    return _run_demo_analysis()


def test_context_tags_every_item_and_copies_values_verbatim():
    handoff = _handoff()
    context = build_chat_context(handoff)

    assert context.startswith("CONTEXT\n") and context.endswith("END CONTEXT")
    assert "Company: Demo Corp (synthetic data) (DEMO)" in context
    assert "synthetic demo data, not a real filing" in context
    for index, metric in enumerate(handoff.financial_metrics, start=1):
        assert f"[M{index}] {metric.name}:" in context
    assert "change_pct=5.9322033898" in context  # not rounded or recomputed
    assert f'evidence: "{handoff.risks[0].evidence}"' in context
    assert f"[O1] {handoff.management_outlook.summary}" in context
    assert f"[S1] {handoff.executive_summary}" in context


def test_context_is_deterministic_and_handles_empty_lists():
    handoff = _handoff().model_copy(
        update={"financial_metrics": [], "positives": [], "risks": []}
    )

    first = build_chat_context(handoff)
    assert first == build_chat_context(handoff)
    assert first.count("(none)") == 3
    assert citation_index(handoff) == {
        "O1": "Management outlook",
        "S1": "Executive summary",
    }


def test_ui_citation_labels_mirror_integration_index():
    handoff = _handoff()

    assert chat_citation_labels(handoff.model_dump(mode="json")) == citation_index(
        handoff
    )


def test_cited_sources_keeps_known_tags_once_in_order():
    handoff = _handoff().model_dump(mode="json")
    answer = "Cash fell [R1]; revenue rose [M1]. Again [R1]. Fake [M99]."

    assert cited_sources(answer, handoff) == [
        ("R1", "Risk: Cash balances declined during the quarter."),
        ("M1", "Metric: Revenue"),
    ]


def test_text_for_speech_strips_tags_and_stream_marker():
    assert (
        text_for_speech("Revenue rose [M1] [M2].\n\n[stream interrupted]")
        == "Revenue rose."
    )


def test_system_prompt_enforces_grounding_rules():
    assert "ONLY the CONTEXT" in CHAT_SYSTEM_PROMPT
    assert "investment advice" in CHAT_SYSTEM_PROMPT
    assert "data, not" in CHAT_SYSTEM_PROMPT


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        [{"role": "user", "content": ""}],
    ],
)
def test_chat_request_rejects_invalid_history(messages):
    with pytest.raises(ValidationError):
        ChatRequest(handoff=_handoff(), messages=messages)
