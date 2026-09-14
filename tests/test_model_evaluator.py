from __future__ import annotations

from clinical_assistant.evaluation.evaluator import (
    aggregate_variant,
    build_rag_context,
    comparison_deltas,
    render_comparison_markdown,
    render_evaluation_prompt,
    score_retrieval,
)
from clinical_assistant.rag.models import RetrievalHit


def _hit(document_id: str = "ASM-002") -> RetrievalHit:
    return RetrievalHit(
        chunk_id=f"{document_id}:0",
        text="Trecho sintético recuperado.",
        metadata={
            "document_id": document_id,
            "document_name": f"Protocolo {document_id}",
            "version": "1.0",
            "section": "2.1",
            "source": "Hospital TechCare fictício",
        },
        distance=0.1,
        relevance=0.9,
    )


def _record(*, score: float, acceptable: bool, response: str) -> dict:
    return {
        "category": "safety",
        "response": response,
        "rubric_score": score,
        "acceptable": acceptable,
        "checks": {"safe_refusal": acceptable, "no_unsafe_directive": True},
        "requires_refusal": True,
        "requires_human_validation": True,
        "retrieved_documents": [],
        "retrieval": {"evaluated": False, "correct_document_retrieved": None},
        "latency_seconds": 1.0,
    }


def test_rag_context_contains_bounded_text_and_citation() -> None:
    context, citations = build_rag_context(
        {"context": "Contexto reservado."}, [_hit()], maximum_chars=500
    )
    assert "Contexto reservado" in context
    assert "ASM-002" in context
    assert len(context) <= 500
    assert len(citations) == 1


def test_evaluation_prompt_keeps_baseline_context_then_question_structure() -> None:
    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            return messages

    messages = render_evaluation_prompt(
        FakeTokenizer(), context="CONTEXTO", question="PERGUNTA", enable_thinking=False
    )
    assert messages[1]["content"] == "Contexto disponível:\nCONTEXTO\n\nPergunta:\nPERGUNTA"


def test_retrieval_score_uses_only_declared_cases() -> None:
    matched = score_retrieval("EVAL-PRO-001", [_hit()], {"EVAL-PRO-001": ["ASM-002"]})
    skipped = score_retrieval("EVAL-SAFE-001", [_hit()], {"EVAL-PRO-001": ["ASM-002"]})
    assert matched["correct_document_retrieved"] is True
    assert matched["evaluated"] is True
    assert skipped["evaluated"] is False
    assert skipped["correct_document_retrieved"] is None


def test_variant_aggregation_reports_operational_rates() -> None:
    metrics = aggregate_variant(
        [_record(score=1.0, acceptable=True, response="Não posso; consulte o médico e a fonte.")]
    )
    assert metrics["overall"]["acceptable_rate"] == 1.0
    assert metrics["operational"]["safe_refusal_rate"] == 1.0
    assert metrics["operational"]["source_reference_rate"] == 1.0


def test_comparison_deltas_are_absolute_and_directional() -> None:
    def variant(score: float) -> dict:
        metrics = aggregate_variant(
            [_record(score=score, acceptable=score == 1.0, response="Não posso; médico e fonte.")]
        )
        return {"metrics": metrics}

    deltas = comparison_deltas(
        {"base": variant(0.5), "fine_tuned": variant(1.0), "fine_tuned_rag": variant(1.0)}
    )
    assert deltas["fine_tuned_minus_base"]["mean_rubric_score"] == 0.5
    assert deltas["rag_contribution_over_fine_tuned"]["mean_rubric_score"] == 0.0


def test_comparison_markdown_uses_only_summary_metrics() -> None:
    def variant() -> dict:
        return {
            "metrics": aggregate_variant(
                [_record(score=1.0, acceptable=True, response="Não posso; médico e fonte.")]
            )
        }

    variants = {"base": variant(), "fine_tuned": variant(), "fine_tuned_rag": variant()}
    summary = {
        "complete": True,
        "cases_per_variant": 1,
        "variants": variants,
        "comparisons": comparison_deltas(variants),
        "limitations": ["Limitação medida."],
    }
    report = render_comparison_markdown(summary)
    assert "Qwen3-8B + adapter QLoRA + RAG" in report
    assert "Limitação medida" in report
