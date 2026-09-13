# Assistente Clínico TechCare — Tech Challenge Fase 3

Fundação reproduzível de um protótipo acadêmico de apoio ao acompanhamento de
pacientes com asma. O repositório está deliberadamente limitado às **ETAPAS 0
a 3: fundação, aquisição, preparação e dados internos sintéticos**.

> **Aviso:** Este sistema é um protótipo acadêmico e não deve ser utilizado para
> diagnóstico, prescrição ou tomada autônoma de decisões clínicas.

## Estado do projeto

- Implementado: estrutura, ambiente local, aquisição e validação dos dados;
  limpeza e normalização textual; mascaramento de identificadores; anonimização
  relacional do Synthea; curadoria e deduplicação; divisão determinística em
  treino/validação/teste; dados internos sintéticos identificados; manifestos e
  testes.
- Não implementado: baseline, fine-tuning, SQLite, RAG, LangChain, LangGraph,
  guardrails em tempo de execução, auditoria, avaliação de modelo e Streamlit.
- Modelo oficial futuro: `Qwen/Qwen3-8B`, sem substituição silenciosa.

## Ambiente escolhido

- Windows 11 para desenvolvimento local.
- Python 3.12 (`>=3.12,<3.13`).
- GPU local GTX 1650 4 GB somente para testes técnicos leves; não executar o
  treinamento QLoRA do Qwen3-8B nela.
- Fine-tuning futuro em Linux com GPU no Google Colab; Kaggle como alternativa.

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

## Testes

```powershell
python -m pytest
```

Os markers são `unit`, `integration`, `network` e `gpu`. Os 31 testes das ETAPAS
0–3 são locais e não usam rede nem GPU.

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

## Próxima etapa (bloqueada até aprovação)

A ETAPA 4 (baseline do `Qwen/Qwen3-8B`) permanece bloqueada. Nenhum modelo foi
baixado ou executado nesta entrega. Consulte `docs/stage_3_report.md` para os
resultados e limitações.
