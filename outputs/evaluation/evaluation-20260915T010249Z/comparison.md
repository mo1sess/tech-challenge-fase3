# Comparação oficial de modelos - ETAPA 10

Execução completa: **sim**
Casos por variante: **24**

| Variante | Nota média | Taxa aceitável | Referência a fonte | Recusa segura | Validação humana | Latência média (s) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B base | 0.7604 | 0.2500 | 0.6667 | 0.9091 | 0.5000 | 13.54 |
| Qwen3-8B + adapter QLoRA | 0.7382 | 0.2083 | 0.6250 | 0.8636 | 0.0000 | 7.78 |
| Qwen3-8B + adapter QLoRA + RAG | 0.7139 | 0.1667 | 0.5417 | 0.8182 | 0.3000 | 9.76 |

## Recuperação RAG

Casos com documento esperado declarado: **4**.
Taxa de recuperação correta: **0.2500**.

## Deltas absolutos

### fine_tuned_minus_base

- mean_rubric_score: -0.0222
- acceptable_rate: -0.0417
- source_reference_rate: -0.0417
- safe_refusal_rate: -0.0455
- human_validation_reference_rate: -0.5000

### fine_tuned_rag_minus_base

- mean_rubric_score: -0.0465
- acceptable_rate: -0.0833
- source_reference_rate: -0.1250
- safe_refusal_rate: -0.0909
- human_validation_reference_rate: -0.2000

### rag_contribution_over_fine_tuned

- mean_rubric_score: -0.0243
- acceptable_rate: -0.0417
- source_reference_rate: -0.0833
- safe_refusal_rate: -0.0455
- human_validation_reference_rate: +0.3000

## Limitações

- The lexical rubric is reproducible but does not establish clinical correctness.
- The 24-case sample is too small for statistical or clinical generalization.
- A blinded professional review remains required for the academic conclusion.
- RAG uses only five synthetic internal protocol documents.

As métricas são automáticas e lexicais. Elas não demonstram correção clínica, significância estatística nem autorização para uso assistencial.
