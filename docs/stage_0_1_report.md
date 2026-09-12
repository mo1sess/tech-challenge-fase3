# Relatorio de execucao — ETAPAS 0 e 1

Data da execucao: 2026-09-12  
Raiz: `C:\Users\msiqu\OneDrive\Documentos\tech-challenge-fase3`

## Escopo concluido

- Repositorio Git e estrutura modular inicial.
- Ambiente Python 3.12.14 e dependencias locais/dev.
- Configuracoes de aplicacao, datasets e contrato de treinamento futuro.
- Aquisicao reproduzivel de MedQuAD e PubMedQA por commit.
- Aquisicao da amostra CSV mais recente do Synthea, identificada por hash.
- Importador controlado de protocolos locais.
- Manifestos, validacao estrutural e testes locais.

Nao foram implementados preprocessing, anonimização, fine-tuning, RAG,
LangChain, LangGraph, banco de dados, guardrails ou Streamlit.

## Inventario medido

| Fonte | Contagem primaria | Arquivos/tamanho observado |
|---|---:|---:|
| MedQuAD | 47.441 pares QA | 11.274 XMLs; conjunto extraido com 36.735.138 bytes |
| PubMedQA PQA-L | 1.000 registros | conjunto extraido com 2.603.745 bytes |
| Synthea CSV | 108 pacientes | 18 CSVs; 62.862.715 bytes |
| Protocolos locais | 0 importados | nenhum protocolo clinico foi fornecido |

Contagens Synthea relevantes: 5.571 encontros, 3.517 condicoes, 68.648
observacoes, 3.850 medicamentos, 15.884 procedimentos e 105 alergias.

## Proveniencia

- MedQuAD commit `577bd37b96c02d1833b2c9eed2de9f96964e96cb`; ZIP SHA-256
  `161d948f8ce8f82accca1f5769a512dff2a01af2b3004d48f2b3da2758a38a09`.
- PubMedQA commit `1cbae8e92f72f20c8d3747cbb3bf5bc53554d997`; ZIP SHA-256
  `887301d2e53eb4dc4a6138cf20b7c29035ea651f93080a3766ca23410d554bce`.
- Synthea sample CSV `latest`; ZIP SHA-256
  `d61417b551e5b0997c33851b339c157421751f0ea68c18ea686ceb1850907c35`.

Os manifestos JSON em `data/raw/_manifests` sao a fonte canonica para essas
medicoes.

## Comandos principais executados

```powershell
.\scripts\setup_windows.ps1 -PythonExecutable '<python-3.12>'
.\.venv\Scripts\python.exe scripts\acquire_data.py --dataset all
.\.venv\Scripts\python.exe scripts\validate_data.py
.\.venv\Scripts\python.exe -m pytest --cov=clinical_assistant --cov-report=term-missing
```

A resolucao conjunta de `requirements/gpu-colab-kaggle.txt` e
`requirements/future-app.txt` tambem foi verificada com `pip --dry-run`; nenhum
conflito de dependencias foi encontrado. Esses pacotes nao foram instalados.

## Resultado dos testes

- 12 testes aprovados em 35,16 segundos na raiz final.
- Cobertura medida: 47% para o codigo das ETAPAS 0 e 1.
- Validacao consolidada: `ok: true` para MedQuAD, PubMedQA, Synthea e o estado
  vazio permitido da caixa de protocolos.

## Problemas encontrados e resolucao

1. O launcher do Windows lista Python 3.12 da Microsoft Store, mas sua execucao
   falhou com acesso negado. O ambiente foi criado com outro executavel Python
   3.12.14 disponivel; o setup aceita `-PythonExecutable` para tratar esse caso.
2. A primeira execucao unitaria detectou um hash esperado incorreto na propria
   fixture de teste; o valor foi corrigido e a suite repetida.
3. O diretorio temporario global do pytest apresentou ACL inacessivel depois de
   execucoes interrompidas. `--basetemp=.pytest-tmp` tornou os testes isolados e
   reproduziveis dentro do projeto.
4. Java 23 nao e a versao LTS recomendada pelo Synthea. Foi usada a amostra CSV
   oficial, sem compilar o gerador.
5. Nenhum protocolo clinico foi anexado. O PDF academico foi preservado em
   `docs/references`, sem ser classificado como protocolo medico.

## Proximos passos aguardando aprovacao

A ETAPA 2 podera implementar limpeza, anonimizacao, normalizacao, curadoria,
deduplicacao e divisao train/validation/test com verificacao de data leakage.
Nenhum desses passos foi iniciado.

