# Catálogo inicial de dados

| Dataset | Classificação | Aquisição nesta etapa | Uso futuro |
|---|---|---|---|
| MedQuAD | público/real | arquivo oficial fixado por commit | QA clínica auxiliar |
| PubMedQA PQA-L | público/real | arquivo oficial fixado por commit | QA biomédica/evidência |
| Synthea CSV | público/sintético | amostra oficial; hash registrado | prontuários sintéticos |
| Protocolos locais | local/público ou sintético | importação controlada | conhecimento e exemplos internos |

PQA-A e PQA-U não são baixados por padrão porque o repositório oficial os
distribui separadamente. Protocolos clínicos precisam ter origem, versão e
licença/proveniência documentadas antes da curadoria.

## Inventário medido em 2026-09-12

- MedQuAD: 11.274 arquivos XML e 47.441 elementos `QAPair`.
- PubMedQA PQA-L: 1.000 registros indexados por PMID.
- Synthea: 108 pacientes em 18 CSVs; entre as tabelas estão 5.571 encontros,
  3.517 condições, 68.648 observações, 3.850 medicamentos e 15.884
  procedimentos.
- Protocolos clínicos locais: 0 arquivos; nenhum foi fornecido nesta etapa.

As quantidades acima vieram da execução dos validadores sobre os arquivos
baixados. O ZIP `latest` do Synthea pode mudar; o manifesto registra o hash exato
da amostra observada.

## Resultado da curadoria da ETAPA 2

| Partição | Total | MedQuAD | PubMedQA |
|---|---:|---:|---:|
| Treino | 13.885 | 13.066 | 819 |
| Validação | 1.752 | 1.657 | 95 |
| Teste | 1.721 | 1.635 | 86 |

Dos 48.441 exemplos lidos, 17.358 foram aceitos. Foram rejeitados 31.035
exemplos com resposta vazia ou curta — comportamento esperado em parte do
MedQuAD por restrições de copyright na fonte — e removidas 48 duplicatas. A
curadoria é estrutural e não equivale a revisão clínica do conteúdo.

O Synthea processado mantém as 18 tabelas e as contagens de linhas da origem.
Os IDs dos 108 pacientes foram pseudonimizados, identificadores diretos foram
removidos de `patients.csv`, campos livres foram mascarados e a validação
encontrou zero referências de paciente desconhecidas.

## Dados internos sintéticos da ETAPA 3

| Arquivo | Categoria | Registros |
|---|---|---:|
| `protocols.jsonl` | protocolo interno sintético | 15 |
| `medical_faq.jsonl` | FAQ institucional sintética | 20 |
| `reports.jsonl` | modelo de laudo/registro | 12 |
| `prescriptions.jsonl` | comportamento seguro para receitas | 12 |
| `procedures.jsonl` | procedimento interno sintético | 15 |
| `safety.jsonl` | exemplo adversarial e resposta segura | 20 |

Total medido: 94 registros, 94 IDs únicos e 94 hashes de conteúdo únicos.
Todos possuem aviso acadêmico, fonte fictícia, versão, seção e indicador
de validação humana. Os protocolos `ASM-001` a `ASM-005` descrevem fluxos
operacionais e de segurança; não constituem diretriz clínica real.

## Conjunto reservado para baseline

| Categoria | Perguntas |
|---|---:|
| Clínica | 6 |
| Protocolo | 6 |
| Paciente | 6 |
| Segurança | 6 |

As 24 perguntas são versionadas em `data/evaluation`. A verificação encontrou
24 IDs e textos únicos e zero coincidências exatas com as instruções de treino
disponíveis. O conjunto não deve ser incorporado ao fine-tuning.

## Dataset específico da ETAPA 5

| Categoria | Treino | Validação | Teste |
|---|---:|---:|---:|
| FAQ | 16 | 2 | 2 |
| Comportamento para receitas | 10 | 1 | 1 |
| Procedimento | 13 | 1 | 1 |
| Protocolo | 13 | 1 | 1 |
| Modelo de laudo/registro | 10 | 1 | 1 |
| Segurança | 16 | 2 | 2 |
| **Total** | **78** | **8** | **8** |

A divisão é estratificada por categoria e ordenada por hash com seed 42. Os 94
hashes permanecem únicos e não há compartilhamento entre as partições. A
validação também confirmou zero coincidências exatas com as 24 perguntas
reservadas.

O QLoRA desta etapa usa somente o conjunto interno sintético, pois o objetivo é
ensinar comportamento, estrutura institucional e limites de atuação. MedQuAD e
PubMedQA permanecem como dados públicos auxiliares já preparados na ETAPA 2;
incluí-los por padrão faria os poucos exemplos internos serem dominados pelo
volume público, majoritariamente em inglês.

## Corpus RAG da ETAPA 6

Os 15 registros de `protocols.jsonl` contêm três formulações de pergunta para
cada orientação. A ingestão agrupa essas paráfrases em 5 documentos lógicos,
ASM-001 a ASM-005, e gera 5 chunks na configuração atual.

Metadados obrigatórios de cada chunk: `document_id`, `document_name`, `version`,
`section`, `source`, `document_type`, `synthetic`, `notice`, `chunk_index` e
`chunk_count`. O texto e os metadados ficam em
`data/processed/rag/protocol_chunks.jsonl`; a coleção Chroma é reconstruível e
fica fora do Git.

Este corpus é inteiramente fictício e operacional. Ele permite testar ingestão,
retrieval e explainability, mas não substitui PCDT, diretriz pública revisada ou
validação médica. Nenhum protocolo clínico oficial foi fornecido até esta etapa.

## Banco estruturado da ETAPA 7

O SQLite importa as tabelas anonimizadas `patients`, `encounters`, `conditions`,
`observations`, `medications`, `procedures` e `allergies`, preservando as
contagens do manifesto da ETAPA 2. A tabela adicional `pending_exams` possui
quatro solicitações fictícias vinculadas a pacientes pseudonimizados e marcadas
como sintéticas. Ela não é produzida por inferência de dados ausentes.

O arquivo do banco é reconstruível e não fica no Git. Seu manifesto versionado
registra hash SHA-256, contagens e confirmação de que identificadores diretos
não foram importados.
