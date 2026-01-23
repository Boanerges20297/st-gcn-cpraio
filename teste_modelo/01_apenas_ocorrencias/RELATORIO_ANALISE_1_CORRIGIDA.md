# ANÁLISE 1 (CORRIGIDA): VIABILIDADE ST-GCN CVLI-ONLY

## 📊 Resumo Executivo

**Dataset:** outputs/cvli_with_bairro.csv filtrado (tipo='cvli')
**Período:** 2022-01-01 a 2026-01-11 (1472 dias)
**Eventos CVLI:** 3,180 
**Cobertura geográfica:** 121 bairros normalizados (Fortaleza + RMF + Interior)
**Dimensões do tensor:** T=1472 × N=121 → 178,112 células

## 📈 Métricas de Qualidade

### Esparsidade
- **Células não-vazias:** 2,959/178,112 (1.66%)
- **Esparsidade:** 98.34%
- **Avaliação:** ⚠️ CRÍTICO

### Sinal Temporal
- **Intensidade média (todas as células):** 0.017854 eventos/dia/bairro
- **Intensidade média (apenas células com evento):** 1.0747
- **Avaliação:** ✅ FORTE

### Variabilidade Temporal
- **Dias com eventos:** 1269/1472 (86.2%)
- **Coeficiente de Variação:** 0.601
- **Autocorrelação (Lag-1):** 0.064
- **Avaliação:** ✅ PREVISÍVEL

### Cobertura Espacial
- **Bairros com eventos:** 121/121 (100.0%)
- **Avaliação:** ✅ EXCELENTE

## 🎯 Viabilidade ST-GCN

### Scoring (0-100)
| Aspecto | Score | Peso | Contribuição |
|---------|-------|------|--------------|
| Esparsidade | 0.0 | 25% | 0.0 |
| Sinal (intensidade) | 100.0 | 35% | 35.0 |
| Variabilidade | 94.9 | 20% | 19.0 |
| Cobertura | 100.0 | 20% | 20.0 |
| **GERAL** | **74.0** | 100% | **74.0** |

### Recomendação
**🟡 VIÁVEL** - Score 74.0/100
ST-GCN pode funcionar com performance aceitável.

## 📋 Próximos Passos

1. Comparar com Análise 2 (CVLI + Contexto Operacional)
2. Se score >= 60: Proceder com implementação
3. Validação cruzada com holdout (últimos 30 dias)
4. Tuning de hyperparâmetros do ST-GCN

---
**Data de geração:** 2026-01-23 00:02:09
**Arquivo tensor:** tensor_cvli_only_CORRIGIDO.npy (1472 × 121)
**Fonte:** outputs/cvli_with_bairro.csv (tipo='cvli')
