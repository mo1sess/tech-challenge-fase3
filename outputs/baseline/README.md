# Evidências do baseline

Cada execução remota cria uma pasta `baseline-<data-hora>` contendo:

- `responses.jsonl`: pergunta, resposta, tokens, latência e rubrica por caso;
- `summary.json`: modelo/revisão, GPU, versões, quantização, memória e métricas.

Uma execução com `--limit` é marcada como `smoke_test` e não vale como resultado
oficial. Somente a execução completa dos 24 casos possui
`run_type: official_full_baseline` e `complete: true`.
