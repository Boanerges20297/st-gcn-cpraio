# Análise de Sazonalidade de CVLI

Escopo: contagem total de ocorrências ao longo do tempo, filtradas por crimes do tipo CVLI (homicídios, latrocínios, tentativas).

Fonte dos dados: `data/raw/ocorrencia_policial_operacional.json`

## Parâmetros e filtro
- Palavras-chave usadas para filtrar CVLI: `homicid`, `latroc`, `tentativa de homicid`, `intervenção policial letal` (busca por substring, sem diferenciação de maiúsculas/minúsculas)
- Agregações calculadas: diário / mensal / dia da semana / autocorrelação (lags 0..60)
- Período analisado: 2022-01-01 até 2026-12-31 (1826 dias)
- Total de ocorrências CVLI encontradas: 140

## Principais resultados
- Média diária de CVLI: 0.077
- Dia com pico de CVLI: 2025-02-03 com 2 ocorrências

## Totais mensais
Consulte `outputs/cvli_monthly_summary.csv` para os totais mensais detalhados.

## Totais por dia da semana
Consulte `outputs/cvli_dow_summary.csv` para os totais por dia da semana.

## Sazonalidade por bairro
Foram gerados os arquivos `outputs/cvli_by_bairro_monthly.csv` (contagem por mês e bairro) e `outputs/cvli_bairro_stats.csv` (média, desvio, CV por bairro).

Top 5 bairros por média mensal de CVLI:
- ZONA RURAL: média mensal=2.55, pico mensal=6, CV=0.69
- FLORES: média mensal=2.00, pico mensal=2, CV=0.00
- CENTRO: média mensal=1.50, pico mensal=3, CV=0.47
-  BOA ESPERANÇA : média mensal=1.00, pico mensal=1, CV=0.00
- 02 DE AGOSTO: média mensal=1.00, pico mensal=1, CV=0.00

## Autocorrelação
Consulte `outputs/cvli_autocorr.csv` para a função de autocorrelação (lags até 60 dias).

## Interpretação prática e próximos passos
- Se houver padrões mensais ou por dia da semana relevantes, adicionar indicadores sazonais (dummies de mês, dia da semana) aos modelos.
- Se a autocorrelação em certos lags for alta, aumentar a janela temporal ou adicionar alvos defasados como features.
- Se as contagens de CVLI forem esparsas, considerar agregações por bairro/mês ou usar modelos de contagem (Poisson/Negativa Binomial) para estabilizar o sinal.
