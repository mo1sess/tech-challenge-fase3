# ETAPA 4 - Baseline do Qwen3-8B

## Estado

**Concluída.** A execução oficial remota terminou os 24 casos e as evidências
foram preservadas em `outputs/baseline/baseline-20260913T143007Z`.

## Resultado oficial medido

- Data UTC: `2026-09-13T14:35:32.346759+00:00`.
- Execução: `official_full_baseline`, `complete: true`.
- Casos: 24, sendo seis por categoria.
- Nota média da rubrica lexical: 0,7604.
- Taxa aceitável: 0,25.
- Tempo total: 324,98 segundos; média de 13,54 segundos por caso.
- Pico de memória da GPU: 5,89 GB.
- Hardware: Tesla T4 com 14,56 GB de VRAM.
- Ambiente: Python 3.12.13, PyTorch 2.11.0+cu128 e Transformers 4.53.2.
- SHA-256 das respostas:
  `f86d028201009ab59948423170ff4b5bb87653d70929146e20f34a8093295eec`.

Por categoria, as notas médias foram 0,7639 em clínica, 0,6750 em paciente,
0,7528 em protocolo e 0,8500 em segurança. A taxa aceitável foi respectivamente
0,3333, 0,1667, 0 e 0,5. Esses valores formam a referência para comparação com
o modelo ajustado e não comprovam correção clínica.

## Contrato do experimento

- Modelo: `Qwen/Qwen3-8B`.
- Revisão imutável: `b968826d9c46dd6066d109eabc6255188de91218`.
- Adapter: nenhum.
- Quantização: bitsandbytes 4-bit NF4 com double quantization.
- Compute: BF16 quando suportado; FP16 em T4.
- Thinking mode: desativado.
- Amostragem: temperatura 0,7; top-p 0,8; top-k 20; seed por caso.
- Saída máxima: 256 tokens.
- Hardware: CUDA com no mínimo 12 GB de VRAM.

A mesma configuração de geração deverá ser reutilizada nas comparações com
o adapter e com RAG.

## Conjunto de avaliação

São 24 perguntas em português, seis por categoria: clínica, protocolo,
paciente e segurança. A validação local encontrou 24 IDs e perguntas únicas e
zero coincidências exatas contra os arquivos de treino disponíveis.

As métricas automáticas utilizam uma rubrica lexical transparente. Elas medem
aderência ao comportamento esperado, recusa segura, menção de validação humana
e transparência sobre fontes. Elas não comprovam correção clínica; a avaliação
final exigirá revisão humana cega.

## Execução no Colab

1. Abra `notebooks/04_baseline_evaluation.ipynb` no Google Colab.
2. Selecione um runtime com GPU.
3. Execute as células em ordem.
4. Confira se `nvidia-smi` mostra pelo menos 12 GB de VRAM.
5. Execute primeiro o smoke test de dois casos.
6. Execute a célula oficial sem `--limit`.
7. Baixe o ZIP de evidências gerado pela última célula.

Cada execução salva respostas, latência, tokens, pico de memória, GPU, CUDA,
versões das bibliotecas, hashes e métricas em `outputs/baseline`.

## Limitações

- O baseline não possui acesso ao banco nem RAG; perguntas de paciente e
  protocolo verificam se o modelo evita inventar dados.
- Amostragem em GPU pode ter pequenas variações mesmo com seed.
- A quantização faz parte do contrato e deve permanecer igual na comparação.
- A GTX 1650 local é bloqueada explicitamente pelo requisito de VRAM.
- Smoke tests não podem ser apresentados como baseline completo.

## Próximo passo

O pré-requisito técnico para a ETAPA 5 (QLoRA) foi atendido. A etapa permanece
aguardando aprovação explícita antes de qualquer implementação ou treinamento.
