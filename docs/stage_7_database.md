# ETAPA 7 — SQLite e ferramentas controladas de paciente

## Objetivo

Esta etapa transforma sete tabelas anonimizadas do Synthea em um banco SQLite
local e reproduzível. O banco oferece contexto estruturado para etapas futuras,
sem acoplar LangChain, LangGraph, Streamlit ou o modelo Qwen3-8B.

## Dados importados

As tabelas `patients`, `encounters`, `conditions`, `observations`,
`medications`, `procedures` e `allergies` vêm de
`data/processed/synthea_anonymized`. Somente identificadores pseudonimizados no
formato `PACnnn` são aceitos. Campos diretamente identificadores não fazem parte
do esquema.

A tabela `pending_exams` contém quatro registros fictícios, claramente
identificados pelo aviso `DOCUMENTO SINTÉTICO PARA FINS ACADÊMICOS`. Uma
pendência existe apenas quando há uma solicitação explícita no arquivo de seed;
o sistema não deduz pendência pela ausência de um resultado.

## Construção e validação

```powershell
python scripts\build_patient_database.py
python scripts\validate_patient_database.py
```

O banco gerado fica em `data/database/techcare.db` e não é versionado. O
manifesto auditável em `outputs/database/database_manifest.json` registra
contagens, hashes, tamanho, versão do esquema e controles de segurança.

Resultado medido nesta execução: banco de 35.520.512 bytes, SHA-256
`4ca933a1c07c12c9b31100d69a79002acc1e38ec841bdc02339a5c5765a8f8ba`,
integridade SQLite `ok`, zero erro de chave estrangeira e zero identificador
direto importado. Foram carregados 108 pacientes, 5.571 encontros, 3.517
condições, 68.648 observações, 3.850 medicamentos, 15.884 procedimentos, 105
alergias e 4 exames pendentes fictícios. A suíte completa terminou com 69
testes aprovados em 18,49 segundos no Windows.

## Consultas de demonstração

```powershell
python scripts\query_patient.py PAC004 patient
python scripts\query_patient.py PAC004 summary
python scripts\query_patient.py PAC004 conditions
python scripts\query_patient.py PAC004 medications
python scripts\query_patient.py PAC004 observations
python scripts\query_patient.py PAC004 pending-exams
```

O repositório usa consultas fixas e parametrizadas em modo somente leitura. Ele
não expõe um método para executar SQL arbitrário. O serviço acrescenta origem,
marcação de dado sintético e aviso de que a saída não pode ser usada para
diagnóstico, prescrição ou decisão clínica autônoma.

## Limitações

- Todos os prontuários são sintéticos e não representam pessoas reais.
- As quatro pendências de exame são exemplos acadêmicos explícitos.
- Não há interpretação clínica, recomendação de tratamento ou prescrição.
- A correção clínica exige avaliação profissional independente.
- A integração do SQLite com agente e RAG pertence às próximas etapas.
