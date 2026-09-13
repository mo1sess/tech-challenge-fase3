# Relatório de execução — ETAPA 2

Data da execução: 2026-09-12  
Raiz prevista: `C:\Users\msiqu\OneDrive\Documentos\tech-challenge-fase3`

## Escopo concluído

- Normalização Unicode, espaços, entidades HTML e caracteres de controle.
- Mascaramento de e-mail, CPF, telefone e rótulos textuais de nome/endereço.
- Conversão de MedQuAD e PubMedQA para um esquema JSONL comum com procedência.
- Rejeição de exemplos sem pergunta/resposta útil e deduplicação por SHA-256.
- Split determinístico 80/10/10 pelo hash do conteúdo.
- Pseudonimização determinística e relacional de pacientes do Synthea.
- Remoção de identificadores diretos e coordenadas de `patients.csv`.
- Validação independente de esquema, hashes, leakage e relações entre CSVs.

Não foram implementados fine-tuning, RAG, SQLite, LangChain, LangGraph,
guardrails, avaliação de modelo ou Streamlit.

## Dados processados

| Métrica | Resultado medido |
|---|---:|
| Exemplos brutos MedQuAD | 47.441 |
| Exemplos brutos PubMedQA | 1.000 |
| Exemplos aceitos | 17.358 |
| Rejeitados por saída vazia/curta | 31.035 |
| Duplicatas removidas | 48 |
| Treino | 13.885 |
| Validação | 1.752 |
| Teste | 1.721 |
| Hashes compartilhados entre splits | 0 |

| Partição | MedQuAD | PubMedQA |
|---|---:|---:|
| Treino | 13.066 | 819 |
| Validação | 1.657 | 95 |
| Teste | 1.635 | 86 |

O Synthea manteve 108 pacientes e 18 tabelas. A validação encontrou zero
referências de paciente desconhecidas.

## Comandos executados

```powershell
.\.venv\Scripts\python.exe scripts\preprocess_data.py
.\.venv\Scripts\python.exe scripts\validate_processed.py
.\.venv\Scripts\python.exe -m pytest
```

## Resultado dos testes

- 27 testes aprovados em 48,55 segundos no Python 3.12.14 para Windows.
- Validação consolidada da ETAPA 2: `ok: true`.
- Sobreposição treino/validação/teste: 0/0/0.
- Referências desconhecidas no Synthea anonimizado: 0.
- Nenhum teste usou GPU ou executou fine-tuning.

## Arquivos gerados

- `data/processed/training/train.jsonl`
- `data/processed/training/validation.jsonl`
- `data/processed/training/test.jsonl`
- `data/processed/synthea_anonymized/*.csv` (18 tabelas)
- `data/processed/preprocessing_manifest.json`
- `outputs/preprocessing_validation.json`

Os JSONL e CSV processados são regeneráveis e permanecem fora do Git. O
manifesto de preprocessing é versionado para auditoria.

## Limitações

1. A curadoria é estrutural; não houve validação por profissional de saúde.
2. MedQuAD e PubMedQA são predominantemente em inglês.
3. Nenhum protocolo clínico local foi fornecido.
4. Synthea é sintético e não representa prevalência ou trajetória clínica real.
5. Mascaramento baseado em padrões não substitui revisão formal de privacidade.
6. A GTX 1650 de 4 GB não é adequada ao fine-tuning do Qwen3-8B; nenhum treino
   foi tentado localmente.

## Próximo passo bloqueado

A ETAPA 3 só deve começar após aprovação explícita. O modelo oficial continua
`Qwen/Qwen3-8B`, e qualquer futuro treinamento QLoRA deverá ocorrer no Google
Colab ou Kaggle com dependências GPU separadas do ambiente local.
