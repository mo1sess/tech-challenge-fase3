# ETAPA 5 - Fine-tuning QLoRA do Qwen3-8B

## Estado

**Pipeline implementado; treinamento remoto real pendente.** A etapa somente
será declarada concluída quando o notebook terminar, produzir
`complete: true` e o adapter e as evidências forem preservados.

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

Esses parâmetros priorizam a execução na Tesla T4 de 15 GB. A GPU local GTX
1650 de 4 GB é rejeitada pelo código antes do carregamento do modelo.

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

## Próximo passo bloqueado

A ETAPA 6 não deve começar antes de o treinamento real terminar e suas
evidências serem validadas e preservadas.
