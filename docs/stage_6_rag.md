# ETAPA 6 — RAG local

## Objetivo e limite

Esta etapa implementa ingestão, chunking, embeddings, ChromaDB, retrieval e
citações. Ela não integra o Qwen3-8B, LangChain, LangGraph, prontuários SQLite ou
Streamlit. O isolamento permite testar a recuperação antes de expor o conteúdo
a um modelo gerador.

O requisito acadêmico pede contextualização e indicação das fontes utilizadas.
Por isso, cada trecho recuperado mantém nome do documento, versão, seção e
procedência. A base atual contém somente material sintético claramente marcado;
nenhuma orientação é apresentada como protocolo clínico real.

## Dados e deduplicação

Fonte: `data/synthetic/hospital/protocols.jsonl`.

- 15 registros de treinamento;
- 5 protocolos lógicos, ASM-001 a ASM-005;
- 3 paráfrases por protocolo, consolidadas na ingestão;
- 5 chunks com a configuração de 1.200 caracteres e overlap de 150;
- aviso obrigatório `DOCUMENTO SINTÉTICO PARA FINS ACADÊMICOS`.

A identidade de cada chunk é o SHA-256 do documento, versão, seção, posição e
texto. A mesma entrada gera os mesmos IDs.

## Embeddings e armazenamento

O modelo oficial de embeddings desta etapa é
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, revisão
`e8f8c211226b894fcb81acc59f3b34ba3efd5f42`. Ele suporta português e é
executado explicitamente em CPU com vetores normalizados. Isso não depende de
GPU nem altera o ambiente CUDA usado pelo QLoRA no Colab.

O ChromaDB usa cliente persistente e distância cosseno. Os embeddings são
calculados fora do Chroma e fornecidos explicitamente na indexação e consulta.
Assim, modelo e vector store permanecem testáveis separadamente.

A versão `1.0.15` foi fixada porque corrige a incompatibilidade de telemetria
presente na `1.0.13`, sem mudar a API de persistência utilizada pelo projeto.
O retrieval retorna até 3 chunks, mas descarta relevância inferior a `0,45`.
Na calibração técnica, as cinco perguntas-alvo recuperaram o ASM correto no
primeiro lugar (relevância de 0,486 a 0,670), enquanto três perguntas fora do
escopo ficaram entre 0,000 e 0,165.

## Execução no Windows

```powershell
Set-Location '.\tech-challenge-fase3'
& .\.venv\Scripts\python.exe -m pip install -r requirements\rag-local.txt
& .\.venv\Scripts\python.exe scripts\build_rag_index.py
& .\.venv\Scripts\python.exe scripts\validate_rag_index.py
& .\.venv\Scripts\python.exe scripts\query_rag.py "Como verificar exames pendentes?"
& .\.venv\Scripts\python.exe -m pytest
```

Na primeira construção, o modelo de embeddings é baixado. As execuções
seguintes reutilizam o cache local. O banco fica em `data/vectorstore/chroma`.

## Evidências e critérios de aceite

O manifesto `outputs/rag/index_manifest.json` registra SHA-256 da origem e dos
chunks, modelo/revisão, dimensão dos embeddings e total indexado. O validador
independente exige:

- manifesto completo da ETAPA 6;
- metadados de citação em todos os chunks;
- rótulo sintético e aviso acadêmico em todos os documentos;
- hashes e contagens coerentes;
- cinco documentos lógicos sem IDs de chunk duplicados;
- contagem da coleção Chroma igual à dos chunks.

Execução local medida em 2026-09-13:

- Python 3.12.14 em Windows;
- ChromaDB 1.0.15;
- Sentence Transformers 4.1.0;
- Transformers 4.53.2 e PyTorch CPU 2.14.0;
- modelo de embeddings com 384 dimensões, executado em CPU;
- 15 registros de origem, 5 documentos lógicos e 5 chunks indexados;
- SHA-256 da origem:
  `a634d4427ea8636ab7703a7e84525a2343c3aad095a056c69da645ebaa7ef6b6`;
- SHA-256 dos chunks:
  `9a43a06bca72cbc972f850905519bff3237be74c25524f8b198443941c2f121d`;
- validação `ok: true`, sem erros;
- consulta de exames pendentes: ASM-002 em primeiro lugar, relevância 0,5615;
- 57 testes aprovados em 68,23 segundos;
- `pip check`: nenhuma dependência quebrada;
- `gpu_used: false` registrado no manifesto.

## Limitações

- A base possui somente cinco protocolos internos fictícios.
- Similaridade vetorial indica proximidade textual/semântica, não validade
  clínica.
- O limiar inicial precisa ser calibrado com um conjunto maior e revisão humana.
- Antes de uso clínico seria obrigatório ingerir documentos oficiais revisados,
  com versão, licença, vigência e governança registradas.
- As próximas etapas ainda precisam integrar pacientes, safety, auditoria e a
  LLM fine-tuned.
