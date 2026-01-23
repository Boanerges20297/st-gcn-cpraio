# RELATÓRIO COMPARATIVO: ST-GCN PARA PREVISÃO DE CVLI

## 🎯 Resumo Executivo

**Objetivo:** Determinar a abordagem mais viável para implementar ST-GCN (Spatio-Temporal Graph Convolutional Networks) na previsão de CVLI (Crimes Violentos Letais Intencionais).

**Análises Realizadas:**
1. **Análise 1:** CVLI-only (simples contagem de eventos CVLI)
2. **Análise 2:** CVLI + Contexto Operacional (CVLI + Prisões + Apreensões)

**Período Analisado:** 2022-01-01 a 2026-01-11  
**Cobertura Geográfica:** 121 bairros normalizados (Fortaleza + RMF + Interior)  
**Dados:** 12.339 eventos CVLI + contexto operacional

---

## 📊 Resultados Detalhados

### Análise 1: CVLI-Only

**Configuração:**
- Dataset: outputs/cvli_with_bairro.csv (tipo='cvli')
- Features: 1 (apenas contagem de CVLI)
- Tensor: 1472 dias × 121 bairros = 178,112 células
- Eventos: 3,180 CVLI

**Metrics:**
| Métrica | Valor | Avaliação |
|---------|-------|-----------|
| Esparsidade | 98.34% | ⚠️ Alta |
| Sinal Médio | 1.0747 | ✅ Forte |
| Variabilidade (CV) | 0.601 | ✅ Previsível |
| Cobertura Espacial | 100% | ✅ Excelente |

**Score Final: 74.0/100**

**Vantagens:**
- ✅ Dataset simples e direto
- ✅ Menos overhead computacional
- ✅ Mais fácil de interpretabilidade
- ✅ Sinal forte e previsível (CV=0.601)

**Desvantagens:**
- ❌ Apenas 1 feature (menos contexto)
- ❌ Sem informações de operações policiais correlacionadas
- ❌ Menor potencial preditivo

---

### Análise 2: CVLI + Contexto Operacional

**Configuração:**
- Dataset: outputs/cvli_with_bairro.csv + operacional_modelo.csv
- Features: 3 (CVLI, Prisões, Apreensões)
- Tensor: 1472 dias × 121 bairros × 3 features = 534,336 células
- Eventos:
  - CVLI: 3,180
  - Prisões: 3,073
  - Apreensões: 15,209

**Feature Scores:**
| Feature | Score | Eventos |
|---------|-------|---------|
| CVLI | 74.0/100 | 3,180 |
| Prisões | 73.9/100 | 3,073 |
| Apreensões | 73.4/100 | 15,209 |

**Correlações:**
| Relação | r |  Significância |
|---------|---|---------------|
| CVLI ↔ Prisões | 0.004 | Fraca |
| CVLI ↔ Apreensões | 0.018 | Muito Fraca |
| Prisões ↔ Apreensões | 0.186 | Moderada |

**Score Final: 73.8/100**

**Vantagens:**
- ✅ Múltiplas features (contexto rico)
- ✅ Informações operacionais complementares
- ✅ Potencial para capturar padrões mais complexos
- ✅ Score similar ao CVLI-only

**Desvantagens:**
- ❌ Correlações fracas entre CVLI e contexto
- ❌ Maior complexidade computacional
- ❌ Mais difícil de interpretabilidade
- ❌ Features parcialmente correlacionadas (não independentes)

---

## 🏆 Recomendação Final

### 🟡 AMBAS

**Justificativa:**

Scores praticamente iguais. Escolha depende de complexidade aceitável.

**Score Comparison:**
```
Análise 1 (CVLI-only):      74.0/100 ██████████████ 
Análise 2 (CVLI+Contexto):  73.8/100 ██████████████
Diferença:                  -0.2 (-0.3%)
```

### Implementação Recomendada

**ABORDAGEM: HÍBRIDA (Recomendado para Produção)**

**Fase 1 - MVP:** Implementar com Análise 1 (CVLI-only)
- Rápido para prototipagem
- Score adequado (74.0/100)
- Base para validação

**Fase 2 - Enriquecimento:** Migrar para Análise 2 (CVLI+Contexto)
- Após validação do MVP
- Score similar (73.8/100) com mais contexto
- Melhor para produção

**Decisão Final:**
- **Para MVP:** ANÁLISE 1 (mais simples)
- **Para Produção:** ANÁLISE 2 (mais robusto)


---

## 📋 Próximos Passos

1. **Implementação ST-GCN**
   - Usar framework: PyTorch com ST-GCN customizado
   - Configuração: 2-3 camadas spatio-temporais
   - Otimizador: Adam com learning rate adaptativo

2. **Validação**
   - Cross-validation temporal (respeitando ordem dos dias)
   - Holdout do período 2026-01-01 a 2026-01-11
   - Métricas: MAE, RMSE, precisão em eventos raros

3. **Baseline Comparativo**
   - AR (AutoRegressivo)
   - ARIMA
   - Prophet
   - FCN (Fully Connected Network)

4. **Deploy**
   - API REST para predições
   - Dashboard com mapas dos bairros
   - Alertas para anomalias

---

## 📑 Artefatos Gerados

### Análise 1 (CVLI-only)
- **Tensor:** teste_modelo/01_apenas_ocorrencias/tensor_cvli_only_CORRIGIDO.npy (1472 × 121)
- **Relatório:** teste_modelo/01_apenas_ocorrencias/RELATORIO_ANALISE_1_CORRIGIDA.md
- **Metadados:** teste_modelo/01_apenas_ocorrencias/metadata_analise_1_CORRIGIDA.json

### Análise 2 (CVLI+Contexto)
- **Tensor:** teste_modelo/02_ocorrencias_prisoes/tensor_cvli_prisoes_CORRIGIDO.npy (1472 × 121 × 3)
- **Relatório:** teste_modelo/02_ocorrencias_prisoes/RELATORIO_ANALISE_2_CORRIGIDA.md
- **Metadados:** teste_modelo/02_ocorrencias_prisoes/metadata_analise_2_CORRIGIDA.json

---

## 📞 Contato & Dúvidas

Para questões sobre:
- **Dados:** Verificar LIMPEZA_22JAN2026.md e README.md
- **Metodologia:** Consultar scripts em teste_modelo/
- **ST-GCN:** Referência em src/models/

---

**Relatório Gerado:** 2026-01-23 00:05:04  
**Status:** ✅ ANÁLISES CONCLUÍDAS - PRONTO PARA IMPLEMENTAÇÃO
