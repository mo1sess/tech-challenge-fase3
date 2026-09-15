# Arquitetura até a ETAPA 11

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

O desenho preserva camadas separadas para dados estruturados, RAG, LangChain,
LangGraph, segurança, auditoria, avaliação e interface.

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

A ETAPA 7 acrescenta uma camada estruturada independente do RAG e da LLM:

```text
data/processed/synthea_anonymized/*.csv
  + data/synthetic/hospital/pending_exams.jsonl
  -> importação determinística + integridade referencial
  -> data/database/techcare.db
  -> PatientRepository (SQL fixo, parametrizado e somente leitura)
  -> PatientService (origem + aviso de segurança)
  -> PatientTools (operações nomeadas e limitadas)
```

O banco não contém nome, documento, endereço ou coordenadas. O agente futuro
não receberá uma ferramenta de SQL genérico: somente consultas predefinidas de
paciente, condições, medicamentos, observações e exames pendentes. O manifesto
versionado permite confirmar hash, contagens e versão do esquema.

A ETAPA 8 conecta essas fontes por ferramentas LangChain e um StateGraph. Cada
nó recebe somente a parcela de estado necessária. O fluxo lê o paciente e as
pendências no SQLite, recupera no máximo os chunks configurados do Chroma,
monta um prompt limitado e gera uma prévia determinística injetável.

Após a geração, uma aresta condicional separa consultas informativas das que
podem alterar conduta. O segundo ramo chama `interrupt()` no nó de revisão
humana e exige retomada explícita antes da resposta final. O checkpointer desta
etapa continua volátil e não é confundido com o log de auditoria.

A ETAPA 9 envolve o grafo com duas barreiras determinísticas e uma trilha de
auditoria independente:

```text
pergunta
  -> validação estrutural
  -> guardrail de entrada --bloqueio--> resposta segura
  -> SQLite + RAG -> geração
  -> guardrail de saída --bloqueio--> resposta segura
  -> [revisão humana quando exigida]
  -> resposta final
  -> evento JSONL append-only + cadeia SHA-256
```

O bloqueio de entrada ocorre antes das ferramentas de paciente e retrieval. O
evento final registra ID da execução, horário, paciente pseudonimizado,
pergunta, ferramentas, documentos recuperados, fontes, modo de geração,
resposta, resultado de segurança e resultado da revisão humana. O conteúdo
completo dos documentos não é duplicado no log; ficam apenas identificadores e
relevância. A cadeia detecta remoção, reordenação ou alteração retroativa de
eventos, mas não substitui armazenamento regulatório com controle de acesso.

A ETAPA 10 mantém a avaliação desacoplada do agente operacional. O baseline
oficial da ETAPA 4 não é reescrito. Em GPU remota, o mesmo modelo base e o
adapter verificado por SHA-256 são carregados uma única vez para avaliar as
duas variantes restantes.

```text
24 casos reservados ───────────────> baseline oficial salvo
          │
          ├── Qwen3-8B + QLoRA ───> respostas fine_tuned
          │
          └── retrieval CPU (5 protocolos)
                -> contexto + fontes
                -> Qwen3-8B + QLoRA
                -> respostas fine_tuned_rag

3 conjuntos de respostas
  -> mesma rubrica lexical
  -> métricas operacionais e retrieval
  -> deltas absolutos
  -> summary.json + comparison.md
```

O runner bloqueia CPU e GPUs com menos de 14 GB, impede substituição do modelo,
verifica a revisão, o adapter e os hashes do conjunto reservado. O ChromaDB e
os embeddings são reconstruídos no Colab, mas permanecem em CPU para preservar
a VRAM da Tesla T4 para a LLM.

A ETAPA 11 adiciona somente uma camada de apresentação sobre o mesmo fluxo
protegido. O controller da aplicação não depende do Streamlit e preserva o
`thread_id` necessário para interromper e retomar o LangGraph. A interface
mantém esse identificador no estado da sessão e nunca oferece SQL arbitrário.

```text
Streamlit
  -> ClinicalApplication
       -> PatientRepository somente leitura -> SQLite
       -> LangGraph
            -> ferramentas LangChain -> SQLite + ChromaDB
            -> guardrails
            -> human_review (interrupt/resume)
            -> auditoria JSONL encadeada
  <- resposta + fontes + pendências + segurança + revisão
```

O modo local é rotulado `deterministic_evidence_preview` e não carrega o
Qwen3-8B na GTX 1650. A inferência oficial permanece comprovada pelas
evidências remotas das ETAPAS 4, 5 e 10; a interface demonstra a integração e
os controles sem substituir silenciosamente o modelo oficial.

## ETAPA 11.1 — LLM customizada no fluxo operacional

O ponto de extensão `ResponseGenerator` passa a aceitar também
`RemoteQwenResponseGenerator`, implementado como um `RunnableLambda` do
LangChain. O contexto mínimo continua sendo montado localmente a partir do
SQLite e do RAG. Somente esse contexto sintético é enviado por HTTPS ao serviço
GPU.

O serviço remoto carrega a revisão fixada do Qwen3-8B em NF4 e o adapter QLoRA
final. Antes de servir, confere o SHA-256 do adapter contra o manifesto da ETAPA
5 e exige GPU com pelo menos 14 GB. A API usa bearer token e limita o tamanho do
prompt. O cliente confirma modelo, revisão e hash; qualquer divergência encerra
a inicialização sem fallback silencioso.

```text
Streamlit local -> LangGraph -> SQLite + RAG -> ContextBuilderChain
                -> LangChain RemoteQwenResponseGenerator
                -> HTTPS -> Qwen3-8B + QLoRA (Colab/T4)
                -> safety -> human review -> audit -> resposta + fontes
```
