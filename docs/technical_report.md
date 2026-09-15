# Relatório técnico - Assistente Clínico TechCare

## Objetivo e escopo

O projeto implementa um protótipo acadêmico de assistente clínico com dados
sintéticos. O sistema integra uma LLM customizada, dados estruturados de
pacientes, recuperação de protocolos, fluxos seguros, revisão humana e
auditoria. Não é um dispositivo médico e não deve apoiar decisões clínicas
reais.

## Dados e preparação

O pipeline usa MedQuAD, PubMedQA, dados sintéticos do Synthea e um catálogo
interno fictício do Hospital TechCare. A preparação aplica limpeza,
normalização, mascaramento de identificadores, anonimização relacional,
deduplicação e divisão determinística entre treino, validação e teste.

Os exemplos internos incluem protocolos, perguntas frequentes, laudos,
comportamento seguro sobre receitas, procedimentos e casos de segurança. Cada
registro traz procedência, versão, seção, hash e indicação explícita de conteúdo
sintético. Os detalhes e limitações estão em `docs/dataset.md`.

## Fine-tuning

O modelo oficial é `Qwen/Qwen3-8B`, fixado na revisão
`b968826d9c46dd6066d109eabc6255188de91218`. O treinamento usa QLoRA com
quantização NF4 em 4 bits e mantém os pesos do modelo base imutáveis. Foram
utilizados 78 exemplos de treino, 8 de validação e 8 de teste, sem coincidência
exata com os 24 casos reservados da avaliação.

A execução medida em Tesla T4 completou 5 épocas e 100 passos em 779,73
segundos. O training loss agregado foi 0,7152, o validation loss final foi
0,0532 e o pico de memória foi 9,97 GB. Foram treinados 43.646.976 parâmetros,
0,9167% do total. O conjunto pequeno implica risco de memorização e as métricas
de loss não comprovam qualidade clínica. Consulte `docs/stage_5_qlora.md`.

## Assistente e arquitetura

O LangChain encapsula ferramentas controladas para SQLite e RAG e constrói um
contexto mínimo antes da geração. O LangGraph coordena validação de entrada,
seleção de ferramentas, recuperação de dados, geração, verificação de saída,
revisão humana e auditoria.

O banco SQLite é somente leitura durante consultas e não aceita SQL arbitrário.
O RAG usa cinco protocolos internos sintéticos, embeddings multilíngues em CPU
e ChromaDB persistente. As respostas apresentam fontes do prontuário e dos
documentos recuperados. O fluxo completo está em `docs/architecture.md`.

## Integração da LLM

No modo oficial, o contexto produzido localmente é enviado por HTTPS a um
serviço que executa o Qwen3-8B com o adapter QLoRA em GPU. O cliente valida o
modelo, a revisão e o SHA-256 do adapter. O serviço usa token temporário e não
há fallback silencioso para outro modelo.

No modo local, a aplicação executa a mesma orquestração sem carregar a LLM e
identifica a saída como prévia determinística. Essa separação permite avaliar
as camadas locais sem representar que uma LLM foi executada.

## Segurança, validação e explainability

Guardrails determinísticos bloqueiam instruções para ignorar regras, prescrever
doses, alterar medicamentos, afirmar diagnósticos definitivos, fabricar dados
ou ocultar fontes. Solicitações sensíveis são retidas para revisão humana.

Consultas factuais sobre medicamentos, condições, observações e exames usam a
política `structured_patient_evidence_v1`: a LLM permanece integrada, mas os
campos exibidos são reconstruídos dos registros literais do SQLite. Marcadores
não preenchidos são descartados. Cada execução gera evento JSONL encadeado por
SHA-256 para detectar alterações retroativas.

## Avaliação

A comparação oficial utilizou os mesmos 24 casos e a mesma configuração em
três variantes. O modelo base obteve nota lexical média 0,7604; o QLoRA,
0,7382; e o QLoRA com RAG, 0,7139. A recuperação RAG encontrou o documento
esperado em 25% dos quatro casos com expectativa objetiva.

Os resultados não demonstram superioridade do fine-tuning nem correção clínica.
A rubrica é lexical, a amostra é pequena e os protocolos são sintéticos. A
conclusão acadêmica exige revisão profissional cega. Consulte
`docs/evaluation.md` e as evidências em
`outputs/evaluation/evaluation-20260915T010249Z/`.

## Interface e reprodução

O Streamlit apresenta seletor de paciente, pergunta, resposta, fontes,
evidência factual, exames pendentes, status de segurança, revisão humana e
auditoria. O procedimento completo para um clone novo e para o modo oficial
está em `docs/evaluator_guide.md`.

## Limitações

- pacientes e documentos internos são sintéticos;
- o corpus RAG possui somente cinco documentos lógicos;
- o adapter não é armazenado no histórico Git e está distribuído no
  [Release oficial da ETAPA 5](https://github.com/mo1sess/tech-challenge-fase3/releases/tag/stage5-qlora-adapter-v1);
  o ZIP oficial possui SHA-256
  `8330716861f243ac22df56bb553a6cbe5067d1025f971289264f25abc6fb668e`;
- o serviço Colab e seu túnel HTTPS são temporários;
- não há autenticação, autorização ou garantias de disponibilidade de produção;
- a demonstração não autoriza diagnóstico, prescrição ou decisão clínica.

## Conclusão

O projeto atende tecnicamente ao pipeline de fine-tuning, integração LangChain,
fluxos LangGraph, dados estruturados, segurança, auditoria e indicação de
fontes. A evidência medida também registra resultados negativos do ajuste sem
ocultá-los. O adapter já está publicado como artefato verificável. Para
finalizar a entrega, ainda devem ser preservadas a validação remota automatizada
e o vídeo de até 15 minutos exigido pelo PDF.
