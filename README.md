# Assistente Clínico TechCare — Tech Challenge Fase 3

Fundação reproduzível de um protótipo acadêmico de apoio ao acompanhamento de
pacientes com asma. O repositório está deliberadamente limitado às **ETAPAS 0 e
1: fundação e aquisição de dados**.

> **Aviso:** Este sistema é um protótipo acadêmico e não deve ser utilizado para
> diagnóstico, prescrição ou tomada autônoma de decisões clínicas.

## Estado do projeto

- Implementado: estrutura, ambiente local, configuração, aquisição de MedQuAD,
  PubMedQA PQA-L e Synthea CSV, importação controlada de protocolos, manifestos,
  validações estruturais e testes.
- Não implementado: preprocessing, anonimização, curadoria, SQLite, baseline,
  fine-tuning, RAG, LangChain, LangGraph, guardrails, auditoria, avaliação e
  Streamlit.
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

## Testes

```powershell
python -m pytest
```

Os markers são `unit`, `integration`, `network` e `gpu`. A suíte desta etapa é
local, pequena e não usa rede nem GPU.

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

A ETAPA 2 deverá implementar limpeza, anonimização, normalização, curadoria e
divisão train/validation/test com prevenção de data leakage. Ela não faz parte
desta entrega.
