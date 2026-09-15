"""Streamlit MVP for the local, safety-controlled TechCare workflow."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st

from clinical_assistant.config import load_yaml, project_root
from clinical_assistant.interface.application import create_configured_clinical_application


ROOT = project_root()
CONFIG = load_yaml(ROOT / "configs" / "interface.yaml")
APP_CONFIG = CONFIG["application"]
UI_CONFIG = CONFIG["interface"]
FULL_AGENT_CONFIG = load_yaml(ROOT / "configs" / "full_agent.yaml")
EXECUTION_CONFIG = FULL_AGENT_CONFIG["execution"]
EXECUTION_MODE = os.environ.get(
    str(EXECUTION_CONFIG["mode_environment"]), str(EXECUTION_CONFIG["local_mode"])
).strip()
REMOTE_URL = os.environ.get(str(EXECUTION_CONFIG["remote_url_environment"]), "").strip()
REMOTE_TOKEN = os.environ.get(str(EXECUTION_CONFIG["remote_token_environment"]), "").strip()

st.set_page_config(
    page_title=APP_CONFIG["title"],
    page_icon=APP_CONFIG["page_icon"],
    layout="wide",
)


@st.cache_resource(show_spinner="Inicializando o fluxo clínico acadêmico...")
def load_application(root_value: str, mode: str, remote_url: str, remote_token: str):
    return create_configured_clinical_application(
        Path(root_value),
        mode=mode,
        endpoint_url=remote_url,
        token=remote_token,
    )


def render_sources(sources: list[str]) -> None:
    st.subheader("Fontes")
    if not sources:
        st.info("Nenhuma fonte foi recuperada para esta solicitação.")
        return
    for source in sources:
        st.markdown(f"- {source}")


def render_pending_exams(exams: list[dict[str, Any]]) -> None:
    st.subheader("Exames pendentes")
    if exams:
        st.dataframe(exams, use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum exame pendente foi encontrado nos dados sintéticos.")


def render_structured_evidence(result: dict[str, Any]) -> None:
    records = result.get("structured_evidence", [])
    kind = result.get("structured_evidence_kind", "")
    if not records or kind == "pending_exams":
        return
    labels = {
        "medications": "Medicamentos — evidência literal do SQLite",
        "conditions": "Condições — evidência literal do SQLite",
        "observations": "Observações — evidência literal do SQLite",
    }
    st.subheader(labels.get(kind, "Evidências estruturadas"))
    st.dataframe(records, use_container_width=True, hide_index=True)
    st.caption(
        "Esta tabela é a fonte factual protegida; o modelo não pode alterar seus valores."
    )


def render_safety(result: dict[str, Any]) -> None:
    st.subheader("Status de segurança")
    safety = result.get("safety", {})
    if safety.get("blocked"):
        st.error(result["safety_label"])
    elif result.get("requires_human_validation"):
        st.warning("Validação médica necessária.")
    else:
        st.success(result["safety_label"])
    with st.expander("Detalhes do controle de segurança"):
        st.json(safety)
        st.caption(
            f"Modo de geração: {result.get('generator_mode', 'não executado')}"
        )


def render_result(result: dict[str, Any], application: Any) -> None:
    st.divider()
    st.subheader("Resposta")
    if result["status"] == "awaiting_human_review":
        st.warning("Validação médica necessária.")
        st.caption("O rascunho abaixo está retido e ainda não é uma resposta liberada.")
        st.text_area(
            "Rascunho técnico retido",
            value=result.get("draft", ""),
            height=150,
            disabled=True,
        )
        notes = st.text_input(
            "Observação da revisão de demonstração",
            key=f"review_notes_{result['thread_id']}",
        )
        approve_column, reject_column = st.columns(2)
        if approve_column.button(
            "Aprovar somente para demonstração",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Retomando o fluxo..."):
                st.session_state.workflow_result = application.resume_review(
                    result["thread_id"], approved=True, notes=notes
                )
            st.rerun()
        if reject_column.button(
            "Rejeitar e reter resposta", use_container_width=True
        ):
            with st.spinner("Retomando o fluxo..."):
                st.session_state.workflow_result = application.resume_review(
                    result["thread_id"], approved=False, notes=notes
                )
            st.rerun()
    else:
        st.markdown(result.get("answer") or "Nenhuma resposta foi produzida.")
        if result.get("human_validation"):
            decision = result["human_validation"].get("approved")
            st.caption(
                "Revisão de demonstração registrada: "
                + ("aprovada" if decision else "rejeitada")
            )

    source_column, exam_column = st.columns(2)
    with source_column:
        render_sources(result.get("sources", []))
    with exam_column:
        render_pending_exams(result.get("pending_exams", []))
    render_structured_evidence(result)
    render_safety(result)


def render_audit(application: Any) -> None:
    with st.expander("Auditoria e log", expanded=False):
        st.caption(
            "Eventos locais em JSONL com cadeia de integridade SHA-256. "
            "O arquivo completo não é publicado no Git."
        )
        try:
            audit = application.recent_audit_events(
                limit=int(UI_CONFIG["audit_events_limit"])
            )
        except Exception as exc:
            st.error(f"Não foi possível validar o log de auditoria: {exc}")
            return
        integrity = audit["integrity"]
        st.success(
            f"Integridade confirmada — {integrity['events']} evento(s), "
            f"último hash: {integrity['last_hash'][:12]}…"
        )
        rows = [
            {
                "horário": event.get("timestamp"),
                "execução": event.get("execution_id"),
                "paciente": event.get("patient_id"),
                "evento": event.get("event_type"),
                "validação necessária": event.get("human_validation_required"),
                "ferramentas": ", ".join(event.get("tools_called", [])),
            }
            for event in reversed(audit["events"])
        ]
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("Nenhum evento registrado nesta instalação.")


st.title(APP_CONFIG["title"])
st.caption(APP_CONFIG["subtitle"])
st.warning(APP_CONFIG["disclaimer"])

try:
    application = load_application(str(ROOT), EXECUTION_MODE, REMOTE_URL, REMOTE_TOKEN)
    patient_ids = application.list_patient_ids()
except Exception as exc:
    st.error(f"A aplicação não pôde iniciar no modo {EXECUTION_MODE!r}: {exc}")
    if EXECUTION_MODE == str(EXECUTION_CONFIG["local_mode"]):
        st.code(
            "& .\\.venv\\Scripts\\python.exe scripts\\build_patient_database.py\n"
            "& .\\.venv\\Scripts\\python.exe scripts\\build_rag_index.py",
            language="powershell",
        )
    else:
        st.info(
            "Confirme a URL HTTPS e o token exibidos pelo notebook Colab da Etapa 11.1. "
            "O sistema não muda silenciosamente para o modo local."
        )
    st.stop()

profile = application.runtime_profile
if profile.get("official_model_active"):
    environment = profile.get("environment", {})
    st.success(
        "Modo oficial: Qwen3-8B + adapter QLoRA executado remotamente "
        f"em {environment.get('gpu_name', 'GPU CUDA')} e integrado ao LangGraph."
    )
else:
    st.info(
        "Modo local: prévia determinística baseada nas evidências recuperadas. "
        "O Qwen3-8B oficial não é carregado neste modo."
    )

with st.sidebar:
    st.subheader("Ambiente de execução")
    st.write(f"Modo: `{profile.get('mode')}`")
    st.write(f"Gerador: `{profile.get('generator')}`")
    st.write(f"Modelo oficial ativo: {'sim' if profile.get('official_model_active') else 'não'}")

if not patient_ids:
    st.error("Nenhum paciente pseudonimizado está disponível para seleção.")
    st.stop()

with st.form("clinical_question_form", clear_on_submit=False):
    patient_id = st.selectbox("Paciente", patient_ids)
    example = st.selectbox(
        "Pergunta de exemplo",
        ["Escrever outra pergunta", *UI_CONFIG["example_questions"]],
    )
    question = st.text_area(
        "Pergunta",
        value="" if example == "Escrever outra pergunta" else example,
        max_chars=int(CONFIG["execution"]["maximum_question_chars"]),
        height=120,
        placeholder="Digite uma pergunta sobre os dados sintéticos do paciente.",
    )
    submitted = st.form_submit_button("Consultar", type="primary")

if submitted:
    if not question.strip():
        st.error("Informe uma pergunta antes de consultar.")
    else:
        try:
            with st.spinner("Consultando SQLite, RAG e fluxo de segurança..."):
                st.session_state.workflow_result = application.submit_question(
                    patient_id, question
                )
        except Exception as exc:
            st.error(f"Não foi possível concluir a consulta: {exc}")

if "workflow_result" in st.session_state:
    render_result(st.session_state.workflow_result, application)

render_audit(application)

st.divider()
st.caption(
    "Hospital TechCare fictício · dados Synthea anonimizados · "
    "documentos internos sintéticos · uso exclusivamente acadêmico"
)
