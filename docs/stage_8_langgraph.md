# ETAPA 8 — LangChain e LangGraph

## Objetivo e correção de sequência

O roteiro original associa o fechamento da ETAPA 7 à chain do LangChain e a
ETAPA 8 ao LangGraph. Como a entrega anterior concluiu SQLite, repository e
tools, esta etapa primeiro conecta essas ferramentas ao LangChain e então
implementa o StateGraph. Não inclui Streamlit, avaliação comparativa nem a
auditoria persistente e os guardrails completos da ETAPA 9.

## Dependências e ambiente

As dependências locais estão isoladas em `requirements/agent-local.txt`:

- `langchain==0.3.26`;
- `langchain-community==0.3.26`;
- `langgraph==0.4.8`;
- a pilha RAG local em CPU da ETAPA 6.

O fluxo não carrega o Qwen3-8B na GTX 1650. A geração local usa
`deterministic_evidence_preview`, identificada no estado e no relatório como
uma prévia técnica, nunca como o modelo oficial. A interface `ResponseGenerator`
permite injetar a LLM posteriormente sem reescrever os nós.

## Ferramentas LangChain

As operações permitidas são:

- `get_patient`;
- `get_patient_conditions`;
- `get_patient_medications`;
- `get_pending_exams`;
- `get_patient_observations`;
- `search_internal_protocol`;
- `search_clinical_guideline`.

A última ferramenta informa explicitamente que nenhuma diretriz clínica oficial
revisada foi fornecida. Não existe ferramenta de SQL arbitrário. O futuro
`save_audit_log` pertence à ETAPA 9.

## Grafo implementado

```text
START
  -> validate_input
  -> load_patient
  -> check_pending_exams
  -> retrieve_protocols
  -> build_context
  -> generate_response
  -> safety_check
       | informativo ---------> finalize_response -> END
       | ação clínica --------> human_review (interrupt)
                                  -> finalize_response -> END
```

O estado mantém `patient_id`, `question`, `patient_data`, `pending_exams`,
`retrieved_documents`, `context`, `llm_response`, `safety_result`,
`requires_human_validation`, `human_validation`, `final_response` e
`execution_id`, além de citações, limitações, ferramentas selecionadas e trace.

O nó foi chamado `finalize_response` porque o LangGraph 0.4.8 impede que um nó
tenha o mesmo nome da chave de estado `final_response`.

## Human-in-the-loop

Perguntas que envolvem tratamento, medicamento, dose, prescrição, diagnóstico,
procedimento ou alteração de conduta seguem uma aresta condicional real para
`human_review`. O nó chama `interrupt()`, grava o checkpoint em memória e só
continua após receber `Command(resume=...)` com aprovação ou rejeição.

A aprovação apenas demonstra a retomada do fluxo e não afirma revisão médica
profissional. Uma rejeição retém a resposta. Persistência durável de checkpoints
e identificação profissional do revisor ainda não fazem parte desta etapa.

## Validação e demonstração

```powershell
python scripts\validate_agent_workflow.py
python scripts\run_agent_workflow.py PAC004 "Quais exames estão pendentes?"
python scripts\run_agent_workflow.py PAC004 "Devo alterar o medicamento?" --review reject
python -m pytest
```

O primeiro exemplo percorre o ramo informativo. O segundo interrompe o grafo,
registra a decisão humana de demonstração e retém a orientação.

Resultado medido nesta execução: validação estrutural `ok`, 9 nós, 1 ponto de
decisão condicional e human-in-the-loop habilitado. O ramo informativo do
PAC004 recuperou 1 pendência explícita e 1 protocolo sintético citado. A
pergunta `Devo alterar o medicamento?` ativou os gatilhos `medicamento` e
`alterar`, interrompeu o grafo, retomou com rejeição e reteve a resposta. A
suíte completa terminou com 80 testes aprovados em 164,30 segundos no Windows.

O LangGraph emitiu um `LangChainPendingDeprecationWarning` interno sobre o
valor padrão do serializador de checkpoint. É um aviso da dependência e não uma
falha; não houve teste reprovado.

## Limitações

- O corpus RAG continua inteiramente sintético.
- O prontuário é Synthea anonimizado e não representa uma pessoa real.
- Nenhuma diretriz oficial foi carregada.
- A prévia determinística não substitui a inferência do Qwen3-8B.
- O checkpointer em memória não é adequado para produção.
- Segurança completa, auditoria persistente e testes adversariais são ETAPA 9.
