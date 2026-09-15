# ETAPA 11 — Interface Streamlit

## Status

**Concluída.** A interface, o controller, o validador e os testes locais foram
executados com sucesso.

## Objetivo

Disponibilizar um MVP local e demonstrável sobre as camadas já validadas de
SQLite, RAG, LangChain, LangGraph, segurança e auditoria. A interface não
carrega o Qwen3-8B na GTX 1650 de 4 GB.

## Interface entregue

- seletor com pacientes pseudonimizados;
- campo de pergunta e botão `Consultar`;
- resposta ou rascunho retido;
- fontes recuperadas;
- exames pendentes;
- status de segurança;
- aviso claro `Validação médica necessária.`;
- aprovação/rejeição de demonstração para retomar o LangGraph;
- área simples com integridade e eventos recentes do log.

O controller em `src/clinical_assistant/interface/application.py` permanece
independente do Streamlit e converte o estado do grafo em um modelo de
visualização estável. A instância do grafo é mantida pelo cache de recursos do
Streamlit, enquanto o `thread_id` fica no estado da sessão para permitir a
retomada após `interrupt()`.

## Ambiente e segurança

O modo local é `deterministic_evidence_preview`. Ele produz uma prévia técnica
baseada somente nos dados recuperados e informa que o Qwen3-8B não foi
executado. Isso evita substituir silenciosamente o modelo oficial e respeita a
limitação da GPU local.

Consultas clínicas continuam passando pelos guardrails. Solicitações de
prescrição, dose, diagnóstico ou alteração de conduta são bloqueadas ou
interrompidas para revisão. A aprovação da interface demonstra o mecanismo e
não representa validação profissional real.

## Execução

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements\app-local.txt
& .\.venv\Scripts\python.exe -m pip install -e .
& .\.venv\Scripts\python.exe scripts\validate_streamlit_app.py
& .\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Se o banco ou o índice local estiverem ausentes, reconstrua-os antes:

```powershell
& .\.venv\Scripts\python.exe scripts\build_patient_database.py
& .\.venv\Scripts\python.exe scripts\build_rag_index.py
```

## Limitações

- todos os pacientes e protocolos são sintéticos;
- o gerador local é uma prévia determinística, não a LLM oficial;
- o checkpointer do LangGraph permanece em memória;
- a decisão humana é apenas uma demonstração e não autentica um profissional;
- o log local pode conter perguntas e pseudônimos e, por isso, não é versionado;
- a aplicação não tem autenticação, autorização ou garantias de produção.

## Validação medida

O validador retornou `ok: true`, confirmou Streamlit 1.46.1, 108 pacientes no
seletor, todas as oito seções obrigatórias e os artefatos completos das ETAPAS
6 a 10. A página respondeu com HTTP 200 e o teste do Streamlit não encontrou
exceções. Uma consulta informativa exibiu resposta, fontes, pendências e
auditoria; uma consulta sobre alteração de medicamento interrompeu o grafo e
foi retomada com rejeição, mantendo a resposta retida.

A suíte completa terminou com 116 testes aprovados em 40,25 segundos e um
aviso de depreciação futura do LangGraph, sem falha funcional. Nenhuma GPU foi
utilizada.
