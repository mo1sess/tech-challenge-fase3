# ETAPA 10 - Avaliação comparativa

## Status

**Concluída. A comparação oficial foi executada em GPU remota e as evidências
importadas passaram no validador local.**

O pré-flight real de 14/09/2026 retornou `ok: true`: baseline, treinamento,
RAG, 24 casos reservados, hashes normalizados e adapter oficial foram
confirmados. A suíte local completa encerrou com 110 testes aprovados em 196,99
segundos e um aviso de depreciação futura do LangGraph, sem falha funcional.

A execução oficial de 14/09/2026 usou uma Tesla T4, Python 3.12.13, PyTorch
2.11.0+cu128 e 6,20 GB de pico de memória GPU. Foram avaliados 24 casos por
variante, totalizando 48 novas gerações, além do baseline preservado.

| Variante | Nota média | Taxa aceitável | Fonte | Recusa segura | Validação humana | Latência média |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B base | 0,7604 | 0,2500 | 0,6667 | 0,9091 | 0,5000 | 13,54 s |
| Qwen3-8B + QLoRA | 0,7382 | 0,2083 | 0,6250 | 0,8636 | 0,0000 | 7,78 s |
| Qwen3-8B + QLoRA + RAG | 0,7139 | 0,1667 | 0,5417 | 0,8182 | 0,3000 | 9,76 s |

O RAG recuperou o documento esperado em um dos quatro casos com expectativa
objetiva, taxa de 0,25. Na rubrica automática, o QLoRA ficou 0,0222 abaixo do
baseline e QLoRA + RAG ficou 0,0465 abaixo. A latência média caiu nas variantes
ajustadas, mas isso não compensa a ausência de ganho lexical.

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

## Execução realizada

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

## Evidências preservadas

```text
outputs/evaluation/evaluation-20260915T010249Z/
  fine_tuned_responses.jsonl
  fine_tuned_rag_responses.jsonl
  summary.json
  comparison.md
```

O ZIP foi incorporado depois de passar pelo validador local. O resultado
declarou `complete: true`, 24 casos por variante e
`run_type: official_full_model_comparison`; foram confirmadas 24 respostas
fine-tuned, 24 respostas fine-tuned + RAG e nenhum erro.

## Limitações do ambiente

- GPUs abaixo do requisito mínimo são bloqueadas para esta execução.
- A avaliação requer Google Colab/Kaggle com pelo menos 14 GB de VRAM.
- A variante base usa a evidência preservada da ETAPA 4; ela não é executada
  novamente sem necessidade.
- RAG usa somente documentos internos sintéticos, não uma diretriz oficial.
- Resultados automáticos não substituem revisão profissional cega.

## Conclusão

O experimento não demonstrou ganho do fine-tuning ou do RAG na rubrica lexical.
O resultado negativo é preservado sem seleção retrospectiva de casos. A análise
é limitada pela amostra de 24 casos, pelos 94 exemplos de treinamento, pelos
cinco protocolos sintéticos e pela ausência de revisão profissional cega.
