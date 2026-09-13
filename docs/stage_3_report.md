# Relatório de execução - ETAPA 3

Data da execução: 2026-09-12  
Raiz: `C:\Users\msiqu\OneDrive\Documentos\tech-challenge-fase3`

## Escopo concluído

- Catálogo curado do Hospital TechCare, hospital inteiramente fictício.
- Protocolos `ASM-001` a `ASM-005` voltados a fluxo e segurança.
- FAQs, modelos de laudos/registros, comportamento para receitas, procedimentos
  internos e exemplos adversariais de safety.
- Geração determinística em JSONL com manifesto SHA-256.
- Validação independente de esquema, procedência, aviso, unicidade e segurança.

## Contagens medidas

| Categoria | Registros |
|---|---:|
| Protocolos | 15 |
| FAQs | 20 |
| Modelos de laudo/registro | 12 |
| Comportamento para receitas | 12 |
| Procedimentos | 15 |
| Safety | 20 |
| **Total** | **94** |

Foram encontrados 94 IDs e 94 hashes de conteúdo únicos. O manifesto confere
com os seis arquivos e os cinco protocolos obrigatórios foram localizados.

## Controles aplicados

- `synthetic: true` obrigatório.
- Fonte obrigatória: `Hospital TechCare (hospital fictício)`.
- Aviso obrigatório: `DOCUMENTO SINTÉTICO PARA FINS ACADÊMICOS`.
- Decisões clínicas marcadas com `requires_human_validation`.
- Resposta com o texto `Validação médica necessária` quando aplicável.
- Bloqueio de saídas com instrução clínica autônoma.
- Testes negativos para aviso ausente e prescrição autônoma.

## Comandos

```powershell
python scripts\generate_synthetic_data.py
python scripts\validate_synthetic_data.py
python -m pytest
```

## Limitações

1. O material não é protocolo real de hospital.
2. Nenhuma diretriz oficial de asma foi fornecida nesta etapa.
3. Não foram criadas doses, prescrições reais ou recomendações clínicas.
4. O conteúdo precisa de revisão profissional e ética antes de qualquer uso
   fora da demonstração acadêmica.
5. Os 94 exemplos internos são pequenos diante dos 17.358 exemplos públicos;
   a estratégia de mistura será decidida somente na etapa de fine-tuning.

## Próximo passo bloqueado

A ETAPA 4 deverá medir o baseline do `Qwen/Qwen3-8B` antes de qualquer treino.
Ela exige GPU remota, conjunto de avaliação separado e persistência dos
resultados reais. Nenhum modelo foi baixado ou executado nesta etapa.
