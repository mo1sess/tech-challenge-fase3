# Assistente Clínico TechCare — Tech Challenge Fase 3

Fundação reproduzível de um protótipo acadêmico de apoio ao acompanhamento de
pacientes com asma. O repositório está deliberadamente limitado às **ETAPAS 0
a 5: fundação, dados, baseline e preparação do fine-tuning QLoRA**.

> **Aviso:** Este sistema é um protótipo acadêmico e não deve ser utilizado para
> diagnóstico, prescrição ou tomada autônoma de decisões clínicas.

## Estado do projeto

- Implementado: estrutura, ambiente local, aquisição e validação dos dados;
  limpeza e normalização textual; mascaramento de identificadores; anonimização
  relacional do Synthea; curadoria e deduplicação; divisão determinística em
  treino/validação/teste; dados internos sintéticos identificados; manifestos e
  testes.
- Concluído em GPU remota: baseline reproduzível do Qwen3-8B com 24 casos
  separados, execução oficial completa e evidências preservadas em
  `outputs/baseline/baseline-20260913T143007Z`.
- Implementado, aguardando execução remota: pipeline QLoRA, dataset
  estratificado, validação, adapter separado, métricas, logs e notebook Colab.
- Não implementado: SQLite, RAG, LangChain, LangGraph, guardrails em tempo de
  execução, auditoria, comparação de modelos e Streamlit.
- Modelo oficial: `Qwen/Qwen3-8B`, sem substituição silenciosa.

## Ambiente escolhido

- Windows 11 para desenvolvimento local.
- Python 3.12 (`>=3.12,<3.13`).
- GPU local GTX 1650 4 GB somente para testes técnicos leves; não executar o
  treinamento QLoRA do Qwen3-8B nela.
- Fine-tuning em Linux com GPU no Google Colab; Kaggle como alternativa.

As dependências estão separadas em `requirements/local.txt`,
`requirements/dev.txt`, `requirements/gpu-colab-kaggle.txt` e
`requirements/future-app.txt`. Somente o grupo local/dev é necessário agora.

## Preparação no Windows

```powershell
Set-Location 'C:\Users\msiqu\OneDrive\Documentos\tech-challenge-fase3'
.\scripts\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
```

Se o launcher da Microsoft Store listar Python 3.12 mas não conseguir executá-lo,
instale a distribuição oficial do Python ou informe explicitamente o executável:

```powershell
.\scripts\setup_windows.ps1 -PythonExecutable 'C:\caminho\para\python.exe'
```

## Aquisição dos dados

```powershell
python scripts\acquire_data.py --dataset all
python scripts\validate_data.py
```

Para importar protocolos fornecidos posteriormente, coloque PDF/TXT/MD em
`data/inbox/protocols` e execute:

```powershell
python scripts\import_protocols.py
```

Os dados brutos não são versionados. Cada aquisição gera um manifesto em
`data/raw/_manifests` com fonte, revisão, horário, SHA-256 e contagens medidas.
O relatório consolidado é gravado em `outputs/data_validation.json`.

Inventário medido nesta execução: MedQuAD com 47.441 pares em 11.274 XMLs;
PubMedQA PQA-L com 1.000 registros; Synthea com 108 pacientes em 18 CSVs; e
zero protocolos clínicos locais, pois nenhum foi fornecido.

## Preprocessing e anonimização

```powershell
python scripts\preprocess_data.py
python scripts\validate_processed.py
```

Os arquivos de treinamento são gravados em `data/processed/training` e as 18
tabelas anonimizadas do Synthea em `data/processed/synthea_anonymized`. Os dados
processados não são versionados; somente o manifesto auditável
`data/processed/preprocessing_manifest.json` permanece no Git.

Resultado medido: 48.441 exemplos lidos, 17.358 aceitos, 31.035 rejeitados por
resposta vazia/curta e 48 duplicatas removidas. A divisão resultou em 13.885
exemplos de treino, 1.752 de validação e 1.721 de teste, sem hashes
compartilhados. Os 108 pacientes do Synthea foram renomeados de forma
determinística (`PAC001`, ...), os campos identificadores diretos foram
removidos e nenhuma referência de paciente ficou órfã.

## Dados internos sintéticos

```powershell
python scripts\generate_synthetic_data.py
python scripts\validate_synthetic_data.py
```

A ETAPA 3 gera 94 exemplos do Hospital TechCare: 15 protocolos, 20 FAQs, 12
modelos de laudo, 12 exemplos de comportamento seguro para receitas, 15
procedimentos e 20 casos de safety. Cada registro possui ID, categoria,
procedência, versão, seção, hash, indicador de revisão humana e o aviso
`DOCUMENTO SINTÉTICO PARA FINS ACADÊMICOS`.

O material ensina comportamento institucional, rastreabilidade e limites de
atuação. Ele não contém dose, prescrição real, diagnóstico novo ou recomendação
clínica sem fonte.

## Baseline remoto do Qwen3-8B

O baseline usa o modelo e a revisão fixados em `configs/baseline.yaml`, sem
adapter, com quantização 4-bit NF4. O conjunto separado possui 24 perguntas,
seis em cada categoria: clínica, protocolo, paciente e segurança.

Validação local:

```powershell
python scripts\validate_evaluation_data.py
```

A execução oficial deve ser feita em GPU remota pelo notebook
`notebooks/04_baseline_evaluation.ipynb`. Primeiro execute o smoke test de dois
casos e depois a execução completa, sem `--limit`. As evidências são gravadas
em `outputs/baseline`.

Resultado medido: 24 casos concluídos em uma Tesla T4, nota média de rubrica
lexical 0,7604, taxa aceitável 0,25, tempo total de 324,98 segundos e pico de
5,89 GB de VRAM. A rubrica não comprova correção clínica.

## Fine-tuning QLoRA

Preparação e validação local do dataset da ETAPA 5:

```powershell
python scripts\prepare_finetuning_data.py
python scripts\validate_finetuning_data.py
```

O dataset usa os 94 exemplos sintéticos internos e produz uma divisão
estratificada com 78 registros de treino, 8 de validação e 8 de teste. Os 24
casos da avaliação permanecem reservados, com zero coincidências exatas.

O treinamento real deve ser executado em GPU remota pelo notebook
`notebooks/05_finetuning.ipynb`. O pipeline usa QLoRA 4-bit NF4, mantém o
Qwen3-8B imutável e salva o adapter LoRA separadamente, além de losses, métricas,
logs, hashes e um teste de inferência. Consulte `docs/stage_5_qlora.md`.

## Testes

```powershell
python -m pytest
```

Os markers são `unit`, `integration`, `network` e `gpu`. Os 48 testes locais não
usam rede nem GPU; a execução pesada permanece separada.

## Fontes

- MedQuAD: https://github.com/abachaa/MedQuAD
- PubMedQA: https://github.com/pubmedqa/pubmedqa
- Synthea: https://synthetichealth.github.io/downloads.html

Consulte `docs/dataset.md` e os arquivos de licença/readme preservados dentro
dos downloads antes de redistribuir os datasets.

## Reprodutibilidade e segurança

- Caminhos em código são relativos à raiz; não há caminhos pessoais embutidos.
- Tokens ficam somente em `.env`, nunca no Git.
- ZIPs são extraídos com proteção contra path traversal.
- Protocolos importados são marcados como não revisados; material sintético não
  pode ser apresentado como protocolo real.
- Resultados e quantidades só são documentados quando medidos por execução.

## Próxima etapa (bloqueada)

A implementação local da ETAPA 5 está pronta, mas o fine-tuning ainda precisa
ser realmente executado no Colab. A ETAPA 6 (RAG) permanece bloqueada até o
treinamento produzir `complete: true`, o adapter e as evidências serem
preservados e houver nova aprovação explícita.
