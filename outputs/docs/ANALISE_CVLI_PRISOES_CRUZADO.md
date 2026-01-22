# Análise Cruzada: Sazonalidade CVLI × Prisões

## Objetivo
Identificar correlações, divergências e padrões de aumento/diminuição
entre CVLIs e prisões por período temporal (mensal, semanal, horário).

## 1. Correlações entre CVLI e Prisões (por Cidade)

### Cidades com CORRELAÇÃO FORTE positiva (CVLI ↔ Prisões)

- **JARDIM**: corr=0.972, p=0.0011, CVLI_trend=0.0, Pris_trend=0.0

### Cidades com CORRELAÇÃO MODERADA (0.3 - 0.7)

- IPUEIRAS: corr=0.663
- REDENCAO: corr=0.593
- OCARA: corr=0.556
- MARCO: corr=0.534
- TAMBORIL: corr=0.530
- PORANGA: corr=0.518
- JAGUARUANA: corr=0.510
- SANTA QUITERIA: corr=0.488
- IBIAPINA: corr=0.458
- URUBURETAMA: corr=0.447

### Cidades com CORRELAÇÃO FRACA ou NEGATIVA (<0.3)

Total de cidades com baixa correlação: 83
- NOVA RUSSAS: corr=0.274
- ITAREMA: corr=0.244
- IPU: corr=0.211
- QUIXERAMOBIM: corr=0.184
- BREJO SANTO: corr=0.182

**CSV Salvo:** outputs\docs\cvli_prisoes_correlacao_por_cidade.csv

## 2. Padrões de Divergência Forte (Aumento/Diminuição Oposta)

### Encontrados 154 padrões de divergência

**JIJOCA DE JERICOACOARA** (Mês 5 → 6)
- CVLI: 1 → 7 (+600.0%)
- Prisões: 3 → 2 (-33.3%)
- Padrão: **CVLI↑ Pris↓**

**MONSENHOR TABOSA** (Mês 1 → 2)
- CVLI: 1 → 6 (+500.0%)
- Prisões: 3 → 2 (-33.3%)
- Padrão: **CVLI↑ Pris↓**

**CEDRO** (Mês 6 → 7)
- CVLI: 1 → 5 (+400.0%)
- Prisões: 2 → 1 (-50.0%)
- Padrão: **CVLI↑ Pris↓**

**CHOROZINHO** (Mês 2 → 3)
- CVLI: 1 → 5 (+400.0%)
- Prisões: 3 → 2 (-33.3%)
- Padrão: **CVLI↑ Pris↓**

**MONSENHOR TABOSA** (Mês 4 → 5)
- CVLI: 1 → 5 (+400.0%)
- Prisões: 6 → 3 (-50.0%)
- Padrão: **CVLI↑ Pris↓**

**SENADOR POMPEU** (Mês 9 → 11)
- CVLI: 1 → 5 (+400.0%)
- Prisões: 3 → 2 (-33.3%)
- Padrão: **CVLI↑ Pris↓**

**ARACATI** (Mês 8 → 9)
- CVLI: 3 → 14 (+366.7%)
- Prisões: 6 → 4 (-33.3%)
- Padrão: **CVLI↑ Pris↓**

**CHOROZINHO** (Mês 4 → 5)
- CVLI: 1 → 4 (+300.0%)
- Prisões: 5 → 3 (-40.0%)
- Padrão: **CVLI↑ Pris↓**

**BELA CRUZ** (Mês 5 → 6)
- CVLI: 1 → 4 (+300.0%)
- Prisões: 6 → 2 (-66.7%)
- Padrão: **CVLI↑ Pris↓**

**ARATUBA** (Mês 6 → 7)
- CVLI: 1 → 4 (+300.0%)
- Prisões: 3 → 1 (-66.7%)
- Padrão: **CVLI↑ Pris↓**

**JAGUARIBE** (Mês 6 → 7)
- CVLI: 1 → 4 (+300.0%)
- Prisões: 2 → 1 (-50.0%)
- Padrão: **CVLI↑ Pris↓**

**CAPISTRANO** (Mês 1 → 2)
- CVLI: 1 → 4 (+300.0%)
- Prisões: 3 → 2 (-33.3%)
- Padrão: **CVLI↑ Pris↓**

**CARIRE** (Mês 9 → 10)
- CVLI: 2 → 8 (+300.0%)
- Prisões: 3 → 1 (-66.7%)
- Padrão: **CVLI↑ Pris↓**

**PINDORETAMA** (Mês 6 → 7)
- CVLI: 1 → 4 (+300.0%)
- Prisões: 3 → 1 (-66.7%)
- Padrão: **CVLI↑ Pris↓**

**VARJOTA** (Mês 9 → 10)
- CVLI: 2 → 8 (+300.0%)
- Prisões: 3 → 1 (-66.7%)
- Padrão: **CVLI↑ Pris↓**

**CSV Salvo:** outputs\docs\cvli_prisoes_divergencias_forte.csv

---
**Análise gerada em:** 22 de janeiro de 2026