# Arquitetura até a ETAPA 6

As ETAPAS 0 e 1 contêm fundação, configuração, aquisição, inventário e
validação estrutural. A ETAPA 2 acrescenta um fluxo local e reproduzível:

```text
data/raw (imutável)
  -> parsers MedQuAD/PubMedQA
  -> limpeza + anonimização + deduplicação
  -> split por hash de conteúdo
  -> data/processed/training/*.jsonl

data/raw/synthea
  -> mapa relacional ID -> PACnnn
  -> remoção de identificadores diretos
  -> data/processed/synthea_anonymized/*.csv
```

O hash do conteúdo determina a partição, portanto uma repetição idêntica não
pode cair em dois conjuntos. Um validador independente recalcula hashes, confere
o esquema JSONL e verifica a integridade referencial dos CSVs anonimizados.

A ETAPA 3 acrescenta uma fonte interna inteiramente ficticia e versionada. Um
catalogo curado em codigo e a configuracao `configs/synthetic_data.yaml`
alimentam um gerador deterministico. Ele produz seis arquivos JSONL com aviso
academico explicito e um manifesto com contagens e hashes. Um validador
independente confere esquema, procedencia, unicidade e controles de seguranca.

O desenho futuro preserva camadas separadas para dados estruturados, RAG,
LangChain, LangGraph, segurança, auditoria, avaliação e interface.

O modelo oficial permanece `Qwen/Qwen3-8B`. Seu treinamento QLoRA é restrito a
Google Colab ou Kaggle; a GTX 1650 local é bloqueada pelo requisito mínimo de
VRAM.

A ETAPA 4 mantém um runner de baseline separado do futuro treinamento. Ele
carrega a revisão fixada do Qwen3-8B em 4 bits, executa o conjunto reservado,
grava respostas e telemetria e calcula uma rubrica lexical auditável. A camada
local valida os dados e o código; somente o runner remoto importa a pilha CUDA.

A ETAPA 5 acrescenta uma camada `finetuning` independente. Os 94 registros
sintéticos internos são validados e divididos de forma determinística e
estratificada em treino, validação e teste, sem reutilizar as 24 perguntas de
avaliação. O runner remoto carrega a revisão imutável do Qwen3-8B em NF4,
treina apenas parâmetros LoRA e preserva o adapter separado do modelo base.

```text
data/synthetic/hospital/*.jsonl
  -> validação de procedência e aviso acadêmico
  -> split estratificado 78 / 8 / 8
  -> prompt-completion sem thinking mode
  -> Qwen3-8B 4-bit NF4 + LoRA
  -> adapter + checkpoints + losses + métricas + hashes
```

O notebook apenas orquestra a execução. A preparação, o treino e a inferência
permanecem em `src/clinical_assistant/finetuning`. Nenhum resultado de treino é
declarado antes da execução remota produzir evidências reais.

A ETAPA 6 introduz RAG sem acoplar LangChain ou a LLM. Os 15 exemplos de
treinamento de protocolos são agrupados por documento, versão, seção e
orientação, resultando em 5 documentos lógicos. Isso impede que paráfrases
idênticas artificialmente dominem o ranking.

```text
data/synthetic/hospital/protocols.jsonl
  -> validação de procedência e aviso sintético
  -> deduplicação em ASM-001 ... ASM-005
  -> chunking determinístico + metadata de citação
  -> embedding multilíngue em CPU
  -> ChromaDB persistente
  -> top-k + limiar de relevância + citação
```

Os vetores são informados explicitamente ao Chroma, mantendo o modelo de
embedding desacoplado do banco. O manifesto registra hashes, revisão imutável do
modelo, dimensão e contagens. A coleção pode ser reconstruída integralmente a
partir dos arquivos versionados.
