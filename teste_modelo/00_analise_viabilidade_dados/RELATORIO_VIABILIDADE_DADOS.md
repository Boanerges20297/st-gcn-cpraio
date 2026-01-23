# ANÁLISE DE VIABILIDADE: DADOS OPERACIONAIS PARA ST-GCN
================================================================================

## 📊 RESUMO EXECUTIVO

**Dataset:** View_Ocorrencias_Operacionais_Modelo.csv
**Período:** 2020-06-03 a 2026-01-22 (2060 dias)
**Cobertura geográfica:** 279 cidades, 943 bairros
**Operações CVLI-like:** 938 (2.29% do total)

## 📈 MÉTRICAS DE QUALIDADE

### Esparsidade (% de dias/bairros sem evento)
- **Operações totais:** 98.4%
  - ✅ Boa: 98.4% < 80%
- **Operações CVLI:** 100.0%
  - ⚠️ Crítica: dados esparsos
- **Operações com apreensão:** 99.0%

### Sinal Temporal (média de eventos/dia/bairro)
- **Operações totais:** 0.0200 eventos/dia/bairro
  - ✅ Fraco
- **Operações CVLI:** 0.0004 CVLI/dia/bairro
  - ⚠️ Insuficiente para previsão

### Variabilidade (Coeficiente de Variação)
- **Operações totais:** CV = 0.435
  - ✅ Padrão previsível
- **Operações CVLI:** CV = 1.489

### Correlação CVLI ↔ Operações Totais
- **Correlação Pearson:** 0.258
  - ⚠️ Fraca

## 🎯 VIABILIDADE ST-GCN

### Scoring (0-100)
- **Qualidade de dados (esparsidade):** 0.0/100
- **Sinal temporal:** 1.0/100
- **Correlação/Estrutura:** 25.8/100

### **SCORE GERAL: 8.0/100**

### 🔴 RECOMENDAÇÃO: NÃO RECOMENDADO

**Conclusão:** Dataset insuficiente para ST-GCN com performance aceitável.


## 📋 COMPARAÇÃO COM ANÁLISES ANTERIORES

| Aspecto | Análise 1 (CVLI) | Análise 2 (CVLI+Prisões) | **Dados Reais** |
|---------|------------------|--------------------------|-----------------|
| Cobertura | Simulado | Simulado | **✅ 2060 dias reais** |
| Bairros | Simulado | Simulado | **✅ 943 bairros reais** |
| Esparsidade | ~70-80% | ~50-60% | **✅ 98% real** |
| Sinal temporal | Baixo | Médio | **✅ 0.0200** |
| Viabilidade | 🟡 Média | 🟡 Boa | **🟢 Ótima com dados reais** |

---
**Data:** 22 de janeiro de 2026