# Rastreabilidade dos requisitos da Fase 3

Fonte: `docs/references/8IADT - Fase 3 - Tech challenge.pdf`.

| Requisito do PDF | Implementação | Evidência principal |
|---|---|---|
| Fine-tuning de LLM | QLoRA 4-bit do Qwen3-8B | `scripts/train_qlora.py`, `notebooks/05_finetuning.ipynb`, `docs/stage_5_qlora.md`, [adapter publicado](https://github.com/mo1sess/tech-challenge-fase3/releases/tag/stage5-qlora-adapter-v1) |
| Protocolos, FAQ, laudos, receitas e procedimentos | Catálogo sintético versionado e identificado | `data/synthetic/hospital/`, `src/clinical_assistant/synthetic/` |
| Preprocessing, anonimização e curadoria | Limpeza, mascaramento, anonimização relacional e deduplicação | `src/clinical_assistant/preprocessing/`, `docs/dataset.md` |
| Pipeline LangChain com LLM customizada | Context builder e gerador remoto Qwen | `src/clinical_assistant/chains/context_chain.py`, `src/clinical_assistant/finetuning/remote.py` |
| Consultas em dados estruturados | SQLite somente leitura e ferramentas controladas | `src/clinical_assistant/database/`, `src/clinical_assistant/tools/` |
| Contexto atualizado do paciente | SQLite + RAG montados antes da geração | `src/clinical_assistant/graph/workflow.py`, `docs/architecture.md` |
| Fluxos LangGraph | StateGraph com seleção de ferramentas e revisão humana | `src/clinical_assistant/graph/`, `docs/stage_8_langgraph.md` |
| Limites contra sugestões impróprias | Guardrails de entrada/saída e interrupção para revisão | `src/clinical_assistant/safety/guardrails.py`, `docs/stage_9_safety_audit.md` |
| Logging e auditoria | JSONL append-only encadeado por SHA-256 | `src/clinical_assistant/audit/audit_logger.py`, `outputs/safety/stage9_validation.json` |
| Explainability e fontes | Citações RAG, origem SQLite e evidência factual estruturada | `src/clinical_assistant/graph/response_consistency.py`, `app/streamlit_app.py` |
| Projeto modularizado em Python | Pacotes separados por aquisição, dados, RAG, grafo, segurança e interface | `src/clinical_assistant/`, `pyproject.toml` |
| Instruções completas no README | Clone novo, dois modos e solução de problemas | `README.md`, `docs/evaluator_guide.md` |
| Dataset anonimizado ou sintético | Exemplos sintéticos versionados e Synthea anonimizado reproduzível | `data/synthetic/hospital/`, `data/processed/preprocessing_manifest.json` |
| Relatório técnico detalhado | Relatório consolidado e relatórios por etapa | `docs/technical_report.md`, `docs/stage_*.md` |
| Diagrama do fluxo LangChain | Diagrama Mermaid renderizável no GitHub | `docs/architecture.md` |
| Avaliação e análise de resultados | Comparação base, QLoRA e QLoRA + RAG | `docs/evaluation.md`, `outputs/evaluation/evaluation-20260915T010249Z/` |
| Vídeo de até 15 minutos | Roteiro completo; gravação externa pendente | `docs/video_demo.md` |

## Situação de entrega

Os requisitos técnicos possuem implementação e evidências versionadas. Antes da
entrega final, ainda devem ser anexados ou publicados:

1. a validação automatizada ponta a ponta gerada por
   `scripts/run_remote_agent_validation.py`;
2. o vídeo de demonstração com até 15 minutos.

O adapter QLoRA já está publicado como Release separado e verificável, com os
hashes do ZIP e do arquivo `adapter_model.safetensors` documentados.
