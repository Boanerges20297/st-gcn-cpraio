# 📊 Sazonalidade de CVLI — Índice de Documentação

Esta pasta contém análise completa de padrões sazonais para **Crimes Violentos Letais Intencionais (CVLI)** no Ceará.

---

## 🎯 Comece por aqui

### [**RESUMO_ANALISE_SAZONALIDADE_CVLI.md**](RESUMO_ANALISE_SAZONALIDADE_CVLI.md)
- ✅ **Leia primeiro**: resumo executivo com principais achados
- Comparação sem filtro vs. com filtro de volume (≥10 CVLI)
- Padrões de mês, hora, dia da semana
- Top bairros com melhor consistência sazonal

---

## 📈 Relatórios Detalhados

### [**cvli_seasonality_patterns.md**](cvli_seasonality_patterns.md)
- Top 20 bairros com maior volume de CVLI
- Distribuição mensal/semanal/horária para cada bairro
- Índices padronizados (0-100 escala)

### [**cvli_seasonality_analysis_cold.md**](cvli_seasonality_analysis_cold.md)
- Análise fria (sem filtros) — inclui todos os 122 bairros
- 4 questões respondidas: mês, hora, dia, bairros com sazonalidade forte
- ⚠️ Inclui bairros com <10 CVLI (padrões menos robustos)

### [**cvli_seasonality_analysis_cold_min10cvli.md**](cvli_seasonality_analysis_cold_min10cvli.md)
- Análise fria **filtrada** — apenas 85 bairros com ≥10 CVLI
- Mesmas 4 questões, com maior confiança estatística
- ✅ **Recomendado para decisões operacionais**

### [**cvli_verification_analysis.md**](cvli_verification_analysis.md)
- Auditoria de dados: confirmação de filtro CVLI + validação de bairros
- Diagnóstico de 182 bairros com `bairro='nan'` (cidades sem subdivisão geográfica)

---

## 📊 Dados Tabulares

### [**cvli_bairros_volume_analysis_min10.csv**](cvli_bairros_volume_analysis_min10.csv)
```csv
cidade,bairro,consistency,total_cvli
Fortaleza,AUTRAN NUNES,1.0,7
Fortaleza,ITAPERI,1.0,8
Fortaleza,ENGENHEIRO LUCIANO CAVALCANTE,1.0,7
...
```
- Ranking de bairros por **consistência sazonal** (score 0-1)
- Volume total de CVLI por bairro
- Filtrado para ≥10 CVLI

---

## 📁 Estrutura de Dados (em `outputs/`)

| Arquivo | Descrição |
|---------|-----------|
| `sazonalidade_bairro_cidade_monthly.csv` | Contagens mensais (raw) |
| `sazonalidade_bairro_cidade_weekday.csv` | Contagens por dia/semana (raw) |
| `sazonalidade_bairro_cidade_hourly.csv` | Contagens por hora (raw) |
| `sazonalidade_bairro_cidade_monthly_index.csv` | Índices normalizados (mensal) |
| `sazonalidade_bairro_cidade_weekday_index.csv` | Índices normalizados (dia/semana) |
| `sazonalidade_bairro_cidade_hourly_index.csv` | Índices normalizados (hora) |

---

## 🔍 Metodologia Resumida

**Filtro Principal**: `tipo = 'CVLI'` (case-insensitive)  
**Total de CVLI**: 12.339 registros (16,7% de 73.998 totais)  
**Escala Geográfica**: Cidade + Bairro (Fortaleza + 181 municípios do Ceará)  
**Períodos**: Mensal, semanal, horário (0-23h)

**Índice Padronizado**: `(mean_count_para_período / mean_geral) × 100`
- 100 = média geral
- >100 = período com incidência acima da média
- <100 = período com incidência abaixo da média

**Consistência**: `1.0 / (1.0 + CV)` onde CV = coef. de variação mensal
- 1.0 = padrão perfeitamente previsível
- 0.0 = padrão completamente aleatório

---

## 🎯 Recomendações Operacionais

1. **Calendário**: Intensificar patrulhas em **Março/Abril** (picos mensais)
2. **Turno**: Aumentar efetivos **entre 19h–21h** (pico noturno)
3. **Fim de semana**: Reforço em **sábados e domingos** (especialmente domingo)
4. **Policiamento Preditivo**: Priorizar bairros com alta consistência:
   - Messejana (0.844, 31 CVLI)
   - Jangurussu (0.821, 40 CVLI)
   - Granja Lisboa (0.802, 32 CVLI)
   - Barra do Ceará (0.795, 39 CVLI)
   - Edson Queiroz (0.789, 16 CVLI)

---

**Última atualização:** 21 de janeiro de 2026  
**Script principal**: `scripts/29_analise_fria_min10cvli.py`

