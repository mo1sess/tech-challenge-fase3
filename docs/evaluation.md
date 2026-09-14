# Metodologia de avaliação

## Objetivo

Comparar exatamente três variantes no mesmo conjunto reservado de 24 casos:

- `base`: Qwen3-8B sem adapter, medido na ETAPA 4;
- `fine_tuned`: Qwen3-8B com o adapter QLoRA real da ETAPA 5;
- `fine_tuned_rag`: a mesma variante ajustada com contexto recuperado dos cinco
  protocolos internos sintéticos da ETAPA 6.

## Controle experimental

O modelo base permanece `Qwen/Qwen3-8B` na revisão
`b968826d9c46dd6066d109eabc6255188de91218`. O runner não permite substituição.
A geração reutiliza seed 42, thinking desativado, temperatura 0,7, top-p 0,8,
top-k 20 e no máximo 256 tokens. Cada variante recebe os mesmos casos e a
mesma seed por posição. O system prompt, os rótulos e a ordem
`contexto -> pergunta` são idênticos ao baseline; na terceira variante muda
somente o contexto acrescentado pelo RAG.

Os hashes do dataset são normalizados apenas quanto a `CRLF/LF`, permitindo
comparar o checkout Windows com o arquivo usado no Colab sem aceitar mudança de
conteúdo. O `adapter_model.safetensors` precisa corresponder ao SHA-256 salvo no
manifesto oficial da ETAPA 5.

## Métricas automáticas

- nota média da rubrica lexical;
- taxa de respostas que atendem a todos os checks;
- checks por categoria e requisito;
- taxa de referência a fonte;
- taxa de recusa segura nos casos que exigem recusa;
- taxa de referência à validação humana nos casos aplicáveis;
- latência média e pico de memória GPU;
- taxa de retrieval não vazio;
- recuperação correta do documento nos quatro casos com expectativa objetiva.

Os documentos esperados do retrieval são declarados separadamente em
`configs/model_evaluation.yaml`; nenhuma pergunta reservada é alterada após o
baseline.

## Interpretação

Os deltas são diferenças absolutas, não significância estatística. A rubrica
lexical confirma presença de comportamentos observáveis, mas não julga correção
clínica, qualidade integral, factualidade médica ou utilidade assistencial. Os
24 casos e os cinco documentos sintéticos são insuficientes para generalização.

Uma revisão profissional cega continua necessária para qualquer conclusão
clínica. O software permanece restrito a demonstração acadêmica.
