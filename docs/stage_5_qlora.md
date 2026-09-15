# ETAPA 5 - Fine-tuning QLoRA do Qwen3-8B

## Estado

**Concluída.** O treinamento remoto real produziu `complete: true`; o adapter,
os checkpoints, as métricas, os logs e os hashes foram preservados em
`outputs/training/training-20260913T151805Z`.

## Resultado oficial medido

- Execução: `official_qlora_training`, `complete: true`.
- Modelo base: `Qwen/Qwen3-8B`, revisão
  `b968826d9c46dd6066d109eabc6255188de91218`.
- Dados: 78 exemplos de treino, 8 de validação e 8 de teste.
- Épocas: 5; passos registrados: 100.
- Training loss agregada: 0,7152.
- Validation loss final: 0,0532.
- Tempo de treino: 779,73 segundos.
- Parâmetros treináveis: 43.646.976, equivalentes a 0,9167% do total.
- Pico de memória: 9,97 GB em uma Tesla T4 com 14,56 GB.
- Adapter final: 166,56 MiB.
- SHA-256 do adapter final:
  `d744bf09a8d7bbe1018ce48091429d82361f72f7c9e34e2a6f8f89d45e1855e3`.
- SHA-256 do ZIP original:
  `62c7bba0d4fb58c2740f19f3d0cd014bebc215d80621c7fbc238203c680a00e6`.

| Época | Validation loss | Acurácia média de token |
|---:|---:|---:|
| 1 | 1,4652 | 0,6465 |
| 2 | 0,7139 | 0,8270 |
| 3 | 0,1567 | 0,9551 |
| 4 | 0,0805 | 0,9791 |
| 5 | 0,0532 | 0,9855 |

Todos os 49 artefatos declarados no manifesto foram encontrados e tiveram seus
hashes recalculados sem divergência. O smoke test pós-treino recusou prescrição
e dose autônomas e informou: "Validação médica necessária."

## Requisito acadêmico atendido pelo desenho

O PDF da Fase 3 exige fine-tuning com protocolos médicos, perguntas frequentes,
modelos de laudos, receitas e procedimentos internos, após preprocessing,
anonimização e curadoria. Nesta etapa são usados os 94 exemplos sintéticos e
explicitamente identificados do Hospital TechCare. Nenhum dado real de paciente
é utilizado no ajuste.

As categorias são `protocol`, `faq`, `report`, `prescription_behavior`,
`procedure` e `safety`. A divisão determinística e estratificada contém 78
exemplos de treino, 8 de validação e 8 de teste. Todas as categorias aparecem
em todas as partições. Os 24 casos da avaliação baseline permanecem reservados,
com verificação de zero coincidências exatas.

## Contrato do treinamento

- Base: `Qwen/Qwen3-8B`.
- Revisão imutável: `b968826d9c46dd6066d109eabc6255188de91218`.
- Método: QLoRA, base quantizada em 4 bits NF4 com double quantization.
- Precisão de cálculo: FP16 na T4.
- Adapter: LoRA separado, rank 16, alpha 32 e dropout 0,05.
- Módulos alvo: projeções de atenção e MLP.
- Épocas: 5.
- Batch por dispositivo: 1; acumulação de gradiente: 4.
- Comprimento máximo: 512 tokens.
- Otimizador: `paged_adamw_8bit`.
- Learning rate: 0,0002 com scheduler cosseno e warmup de 10%.
- Gradient checkpointing habilitado.
- Thinking mode do Qwen3 desabilitado.
- Loss calculada somente sobre a resposta esperada.

Esses parâmetros priorizam a execução na Tesla T4 de 15 GB. GPUs abaixo do
requisito mínimo de 14 GB de VRAM são rejeitadas antes do carregamento.

## Execução

1. Envie `notebooks/05_finetuning.ipynb` ao Google Colab.
2. Selecione T4 e o runtime 2026.07 com Python 3.12.
3. Execute as células em ordem.
4. Confirme a preparação e validação dos 94 registros.
5. Execute o treinamento sem interromper o runtime.
6. Confirme `official_qlora_training` e `complete: true`.
7. Baixe `qwen3_8b_qlora_stage5_evidence.zip`.

## Evidências obrigatórias

O ZIP final deverá conter:

- adapter LoRA em `adapter/`, separado do modelo base;
- checkpoints intermediários limitados aos dois mais recentes;
- `train_metrics.json` e `validation_metrics.json`;
- `log_history.jsonl` com training loss e validation loss;
- `inference_smoke.json` demonstrando inferência após o treino;
- `run_status.json` e `run_manifest.json`;
- hashes SHA-256 dos artefatos, dataset e configuração;
- ambiente medido, versões, GPU, VRAM e pico de memória.

Não serão declaradas métricas antes da execução real. A rubrica automática não
substitui revisão clínica ou humana.

## Limitações

- O conjunto interno é pequeno e sintético, adequado para ensinar comportamento
  e estrutura, não conhecimento clínico completo.
- O fine-tuning não substitui o RAG; protocolos atualizados serão tratados na
  ETAPA 6.
- O adapter só é compatível com o modelo e a revisão registrados.
- A comparação quantitativa baseline versus fine-tuned pertence à ETAPA 10.

## Próximo passo

O pré-requisito técnico da ETAPA 6 foi atendido. A etapa RAG permanece
aguardando aprovação explícita antes de qualquer implementação.
