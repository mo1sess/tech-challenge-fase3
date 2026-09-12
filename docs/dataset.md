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
