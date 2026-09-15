# ETAPA 11.1 — Integração da LLM customizada

## Status

**Implementação e preflight locais concluídos; integração remota confirmada por
consulta manual.** A validação automatizada ponta a ponta ainda deve ser
preservada. Esta etapa fecha a lacuna entre o fluxo operacional da ETAPA 11 e
o Qwen3-8B ajustado e avaliado nas ETAPAS 5 e 10.

## Atendimento ao requisito acadêmico

O assistente completo segue o fluxo:

```text
Streamlit
  -> LangGraph
  -> ferramentas LangChain (SQLite + RAG)
  -> ContextBuilderChain
  -> RemoteQwenResponseGenerator (LangChain Runnable)
  -> Qwen3-8B + adapter QLoRA em GPU
  -> trava de evidência factual estruturada
  -> safety check
  -> human-in-the-loop quando necessário
  -> auditoria e resposta com fontes
```

O modo `qwen_remote` é explícito. Se a URL, o token, a identidade do modelo, a
revisão fixada ou o SHA-256 do adapter estiverem incorretos, a aplicação não
inicia nesse modo. Não existe fallback silencioso para outro modelo.

## Consistência factual

Uma consulta remota real mostrou que o adapter podia transformar
`Meperidine Hydrochloride 50 MG Oral Tablet` em outro nome e associar um motivo
inexistente. Outra geração retornou os marcadores `[CÓDIGO]` e `[DATA]`. Esses
comportamentos são compatíveis com a limitação já medida na ETAPA 10: o
fine-tuning não superou o modelo base na rubrica automática.

A política `structured_patient_evidence_v1` agora protege consultas factuais
sobre medicamentos, condições, observações e exames pendentes:

- o Qwen remoto permanece no fluxo e sua saída bruta fica no estado interno;
- a resposta liberada é reconstruída com valores literais do SQLite;
- a interface mostra os registros estruturados usados;
- placeholders não preenchidos nunca são liberados;
- a decisão da política e o tipo de evidência são registrados na auditoria.

Essa trava garante fidelidade dos campos estruturados consultados, mas não
transforma texto generativo livre em informação clinicamente validada.

## Separação de ambientes

### Windows local

- Streamlit;
- SQLite com prontuários sintéticos;
- ChromaDB e retrieval;
- ferramentas e contexto LangChain;
- orquestração LangGraph;
- guardrails, revisão humana e auditoria;
- cliente HTTPS do gerador remoto.

### Google Colab com Tesla T4

- Qwen/Qwen3-8B na revisão fixada;
- quantização NF4 em 4 bits;
- adapter QLoRA final da ETAPA 5;
- serviço FastAPI protegido por bearer token;
- túnel HTTPS temporário usado somente durante a demonstração.

O preflight bloqueia ambientes sem GPU compatível e exige pelo menos 14 GB de
VRAM para a execução oficial.

## Integridade do modelo

O serviço verifica `adapter_model.safetensors` contra o SHA-256 registrado no
manifesto oficial da ETAPA 5:

```text
d744bf09a8d7bbe1018ce48091429d82361f72f7c9e34e2a6f8f89d45e1855e3
```

O cliente valida em cada inicialização:

- modelo `Qwen/Qwen3-8B`;
- revisão `b968826d9c46dd6066d109eabc6255188de91218`;
- SHA-256 do adapter;
- disponibilidade do serviço.

## Execução

1. Abra `notebooks/07_full_agent_colab.ipynb` no Google Colab.
2. Selecione Python 3.12 e GPU Tesla T4.
3. Baixe o adapter no
   [Release oficial da ETAPA 5](https://github.com/mo1sess/tech-challenge-fase3/releases/tag/stage5-qlora-adapter-v1)
   e coloque `qwen3_8b_qlora_adapter_only.zip` no Google Drive.
4. Execute as células em ordem e copie a URL HTTPS e o token impressos.
5. No PowerShell local, defina:

```powershell
$env:TECHCARE_EXECUTION_MODE = "qwen_remote"
$env:TECHCARE_REMOTE_URL = "URL_HTTPS_FORNECIDA_PELO_COLAB"
$env:TECHCARE_REMOTE_TOKEN = "TOKEN_FORNECIDO_PELO_COLAB"
& .\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

6. Em outro PowerShell, com as mesmas variáveis, preserve a validação oficial:

```powershell
& .\.venv\Scripts\python.exe scripts\run_remote_agent_validation.py
```

O script executa consultas de condições, exames, protocolo, revisão humana e
uma tentativa adversarial. As evidências são gravadas em
`outputs/app/remote/remote-agent-*/` sem armazenar o token.

## Validação local

```powershell
& .\.venv\Scripts\python.exe scripts\validate_full_agent.py
& .\.venv\Scripts\python.exe -m pytest
```

O preflight local confirma o código, notebook, modelo, revisão e hash. Ele não
declara que o Qwen foi executado: esse estado permanece
`pending_remote_execution` até a execução real no Colab.

O preflight retornou `ok: true`. A suíte completa terminou com 130 testes
aprovados em 26,46 segundos e um aviso de depreciação futura do LangGraph, sem
falha funcional. Um teste automatizado do Streamlit confirmou o modo local,
consulta, resposta sem duplicação de fontes e ausência de exceções. Nenhuma GPU
foi utilizada nesta validação local.

## Limitações

- todos os pacientes e protocolos são sintéticos;
- a URL temporária existe somente enquanto o Colab e o túnel estiverem ativos;
- o token não deve ser publicado, salvo em notebook público ou versionado;
- a integração não comprova correção clínica;
- a execução oficial requer GPU remota e download do modelo base;
- métricas anteriores mostram que o fine-tuning e o RAG não superaram o
  baseline na rubrica lexical utilizada.
