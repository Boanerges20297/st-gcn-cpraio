# ANÁLISE 2 (CORRIGIDA): VIABILIDADE ST-GCN CVLI + CONTEXTO OPERACIONAL

## 📊 Resumo Executivo

**Dataset CVLI:** outputs/cvli_with_bairro.csv (tipo='cvli', 3,180 eventos)
**Dataset Operacional:** data/raw/View_Ocorrencias_Operacionais_Modelo.csv (prisões + apreensões)
**Período:** 2022-01-01 a 2026-01-11 (1472 dias)
**Cobertura geográfica:** 121 bairros normalizados
**Dimensões do tensor:** T=1472 × N=121 × F=3 → 534,336 células

### Features Utilizadas

1. **CVLI** (3,180 eventos - Score: 74.0/100)
   - Homicídios e latrocínios (tipo=cvli)
   
2. **Prisões** (3,073 eventos - Score: 73.9/100)
   - Operações com prisões/mandados
   
3. **Apreensões** (15,209 eventos - Score: 73.4/100)
   - Drogas, armas e/ou dinheiro apreendido

## 📈 Correlações entre Features

| Relação | Correlação (r) | p-value | Interpretação |
|---------|--------|---------|--------------|
| CVLI ↔ Prisões | 0.004 | 0.0679 | Fraca |
| CVLI ↔ Apreensões | 0.018 | 0.0000 | Fraca |
| Prisões ↔ Apreensões | 0.186 | 0.0000 | Fraca |

## 🎯 Viabilidade ST-GCN

### Scoring (0-100)
- **Score médio (features):** 73.8/100
- **Bônus correlação:** +0.0
- **SCORE GERAL:** **73.8/100**

### Recomendação
**🟡 VIÁVEL** - Score 73.8/100
ST-GCN pode ser implementado com performance aceitável.

## 📊 Comparação: CVLI-only vs CVLI+Contexto

| Aspecto | CVLI-only | CVLI+Contexto | Diferença |
|---------|-----------|---------------|-----------|
| **Score Geral** | 74.0/100 | 73.8/100 | **-0.2** (-0.3%) |

### Conclusão
**Análise 1 é superior.** CVLI-only é mais adequado.

---
**Data de geração:** 2026-01-23 00:03:57
**Arquivo tensor:** tensor_cvli_prisoes_CORRIGIDO.npy (1472 × 121 × 3)
**Fonte:** cvli_with_bairro.csv + operacional_modelo.csv
