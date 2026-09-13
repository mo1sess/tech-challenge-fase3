# Arquitetura até a ETAPA 3

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

O desenho futuro preserva camadas separadas para fine-tuning, dados estruturados,
RAG, LangChain, LangGraph, segurança, auditoria, avaliação e interface.

O modelo oficial permanece `Qwen/Qwen3-8B`. Seu treinamento QLoRA será remoto
em Google Colab ou Kaggle e não é implementado nesta etapa.
