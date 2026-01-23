# ANÁLISE 1: VIABILIDADE ST-GCN CVLI-ONLY

## 📊 Resumo Executivo

**Dataset:** View_Ocorrencias_2022_ENRIQUECIDO.csv (dados com lat/long IBGE)
**Período:** 2022-01-10 a 2026-01-18 (1470 dias)
**Eventos CVLI:** 313 homicídios + latrocínios
**Cobertura geográfica:** 94 bairros (Fortaleza + RMF + Interior)
**Dimensões do tensor:** T=1470 × N=94 → 138,180 células

## 📈 Métricas de Qualidade

### Esparsidade
- **Células não-vazias:** 310/138,180 (0.22%)
- **Esparsidade:** 99.78%
- **Avaliação:** ⚠️ CRÍTICO

### Sinal Temporal
- **Intensidade média (todas as células):** 0.002265 eventos/dia/bairro
- **Intensidade média (apenas células com evento):** 1.0097
- **Avaliação:** ✅ FORTE

### Variabilidade Temporal
- **Dias com eventos:** 277/1470 (18.8%)
- **Coeficiente de Variação:** 0.342
- **Autocorrelação (Lag-1):** 0.017
- **Avaliação:** ✅ PREVISÍVEL

### Sinal Temporal
- **Intensidade média (todas as células):** 0.002265 eventos/dia/bairro
- **Intensidade média (apenas células com evento):** 1.0097
- **Avaliação:** ✅ FORTE

### Variabilidade Temporal
- **Dias com eventos:** 277/1470 (18.8%)
- **Coeficiente de Variação:** 0.342
- **Autocorrelação (Lag-1):** 0.017
- **Avaliação:** ✅ PREVISÍVEL

### Cobertura Espacial
- **Bairros com eventos:** 94/94 (100.0%)
- **Avaliação:** ✅ EXCELENTE

## 🎯 Viabilidade ST-GCN

### Scoring (0-100)
| Aspecto | Score | Peso | Contribuição |
|---------|-------|------|--------------|
| Esparsidade | 0.0 | 25% | 0.0 |
| Sinal (intensidade) | 100.0 | 35% | 35.0 |
| Variabilidade | 92.1 | 20% | 18.4 |
| Cobertura | 100.0 | 20% | 20.0 |
| **GERAL** | **73.4** | 100% | **73.4** |

### Recomendação
**🟢 VIÁVEL** - Score 73.4/100
ST-GCN pode ser implementado com performance aceitável.

## 📋 Próximos Passos

1. Comparar com Análise 2 (CVLI + Prisões) para avaliar impacto de features adicionais
2. Se score < 60: Enriquecer com dados de operações policiais correlacionadas
3. Considerar agregação temporal (dias → semanas) se esparsidade muito alta
4. Validação cruzada com período holdout (teste em últimos 30 dias)

---
**Data de geração:** 2026-01-22 23:57:41
**Arquivo tensor:** tensor_cvli_only.npy (1470 × 94)
