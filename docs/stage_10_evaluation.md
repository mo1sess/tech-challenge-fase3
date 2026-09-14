# ETAPA 10 - Avaliação comparativa

## Status

**Pipeline implementado e pré-flight local aprovado. Resultados das variantes
fine-tuned e fine-tuned + RAG ainda não executados.**

O pré-flight real de 14/09/2026 retornou `ok: true`: baseline, treinamento,
RAG, 24 casos reservados, hashes normalizados e adapter oficial foram
confirmados. A suíte local completa encerrou com 110 testes aprovados em 196,99
segundos e um aviso de depreciação futura do LangGraph, sem falha funcional.

Nenhum número comparativo é apresentado nesta fase de preparação. O baseline
oficial preservado tem 24 casos, nota lexical média 0,7604 e taxa aceitável
0,25; esses valores vêm da execução real da ETAPA 4.

## Arquivos principais

- `configs/model_evaluation.yaml`: contrato das três variantes.
- `src/clinical_assistant/evaluation/evaluator.py`: inferência, métricas, deltas
  e relatório reproduzível.
- `src/clinical_assistant/evaluation/validation.py`: pré-flight e validação das
  evidências importadas.
- `notebooks/06_model_evaluation.ipynb`: execução Colab com T4.
- `scripts/run_model_evaluation.py`: runner GPU independente do notebook.
- `scripts/validate_model_evaluation.py`: validação local sem carregar a LLM.
- `requirements/evaluation-gpu.txt`: dependências exclusivas do runtime remoto.

## Execução planejada

O notebook:

1. confirma Python 3.12 e CUDA;
2. clona o repositório e instala dependências fixadas;
3. executa o pré-flight;
4. reconstrói ChromaDB com embeddings em CPU;
5. recebe o ZIP real da ETAPA 5;
6. valida o hash do adapter;
7. executa smoke test de 2 casos por variante;
8. executa 24 casos por variante ajustada, sem `--limit`;
9. valida a execução completa;
10. gera e baixa um ZIP de evidências.

## Evidências esperadas

```text
outputs/evaluation/evaluation-AAAAMMDDTHHMMSSZ/
  fine_tuned_responses.jsonl
  fine_tuned_rag_responses.jsonl
  summary.json
  comparison.md
```

O ZIP somente será incorporado ao repositório depois de passar pelo validador
local. O resumo precisa declarar `complete: true`, 24 casos por variante e
`run_type: official_full_model_comparison`.

## Limitações do ambiente

- A GTX 1650 com 4 GB está bloqueada para esta execução.
- A avaliação requer Google Colab/Kaggle com pelo menos 14 GB de VRAM.
- A variante base usa a evidência preservada da ETAPA 4; ela não é executada
  novamente sem necessidade.
- RAG usa somente documentos internos sintéticos, não uma diretriz oficial.
- Resultados automáticos não substituem revisão profissional cega.

## Próximos passos desta etapa

Executar o notebook no Colab, baixar o ZIP e validar os resultados localmente.
Somente então atualizar este documento com números reais e concluir a ETAPA 10.
