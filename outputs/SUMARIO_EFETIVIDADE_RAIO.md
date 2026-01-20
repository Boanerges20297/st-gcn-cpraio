ANALISE DE EFETIVIDADE: OPERACOES RAIO EM 2025/2026
CAUCAIA vs CEARA (ESTADO COMPLETO)

================================================================================
ESCALA DA ANALISE
================================================================================

CAUCAIA (Municipio):
  * Operacoes: 315 total | 24 por periodo (media)
  * Periodo: Jan-Dez 2025
  * Periodos analisados: 7
  * Cobertura: 1 municipio, 76 bairros com operacoes

CEARA (Estado):
  * Operacoes: 9.060 total | 697 por periodo (media) [28x MAIS]
  * Periodo: Jan 2025 - Jan 2026
  * Periodos analisados: 7
  * Cobertura: 162 cidades, 43 bairros com dados de criticidade


================================================================================
METRICAS DE EFETIVIDADE
================================================================================

                              CAUCAIA       CEARA      Diferenca
================================================================================
Reducao >= 15%                1/7 (14.3%)   1/7(14.3%)  = (IGUAL)
Qualquer reducao              2/7 (28.6%)   3/7(42.9%)  +14.3pp
Dose-resposta (r)             -0.2468       -0.1619     +0.0849 (MELHOR)
Slope (%/operacao)            -2.10         -0.038      +1.97 (MELHOR)

Esperado para eficacia:       r < -0.3 (e slope negativo forte)
Resultado:                    FALHA         FALHA       (IGUAL OUTCOME)


================================================================================
INTERPRETACAO ESTRATEGICA
================================================================================

[1] EFETIVIDADE BAIXA EM AMBOS OS NIVEIS
  - Apenas 14.3% dos periodos mostram reducao >=15% (esperado: >30%)
  - Qualquer reducao em apenas 28.6% (Caucaia) a 42.9% (CE)
  - Esperado: >60-70% para padrao reativo

[2] CORRELACAO DOSE-RESPOSTA NEGATIVA (REATIVA, NAO PREVENTIVA)
  - Caucaia: r = -0.2468 (correlacao moderada negativa)
  - Ceara: r = -0.1619 (correlacao fraca negativa)
  - Ambos LONGE de r < -0.3 necessario para eficacia
  - MAIS operacoes NÃO se correlaciona com MENOS crimes

[3] OPERACOES PARECEM SER REATIVAS, NAO PREVENTIVAS
  - Correlacao negativa sugere: vao onde ha MAIS crime
  - Nao ha efeito preventivo mensuravel apos
  - Padrao: Operacoes aumentam -> Depois vem resposta policial

[4] ESCALA ESTADUAL MELHORA LEVEMENTE (mas ainda insuficiente)
  - Ceara tem 42.9% de periodos com alguma reducao vs 28.6% em Caucaia
  - Possivel: efeito de media (alguns locais melhoram, outros nao)
  - Mas correlacao dose-resposta melhora (r mais proximo de 0)
  - Sugestao: Ou crime muito descentralizado, ou operacoes nao sao data-driven


================================================================================
CONCLUSAO FINAL
================================================================================

EFETIVIDADE OPERACIONAL: BAIXA EM AMBOS OS CENARIOS

Tanto no nivel municipal (Caucaia) quanto estadual (Ceara), as operacoes RAIO
NAO demonstram correlacao significativa com reducao de crimes.

Possveis causas:
  1. Operacoes sao REATIVAS (vao onde crime ja ocorreu)
  2. Defasagem temporal (crimes agora -> efeito em +30 dias?)
  3. Crimes altamente descentralizados (nao concentrados em poucos locais)
  4. Estrategia operacional nao e data-driven

Recomendacoes para proximos passos:
  * Investigar se operacoes tem INTELIGENCIA alem de volume de crimes
  * Testar com defasagens maiores (60, 90 dias)
  * Analisar tipo de crime (alguns tipos podem ter melhor resposta)
  * Considerar abordagem inteligencia-driven vs volume-driven
  * Verificar se dados de crime sao completos (subnotificacao?)


================================================================================
METODOLOGIA
================================================================================

Comparacao: Periodo-a-periodo (30 dias cada)
Metrica: Variacao percentual de criticidade/crimes
Dados: RAIO 2025-2026 vs Modelo ST-GCN de criticidade
Validacao: Correlacao Pearson e analise de slope (dose-resposta)

================================================================================
