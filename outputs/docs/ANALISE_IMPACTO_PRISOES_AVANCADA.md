# Análise de Impacto: Prisões vs CVLI

**Objetivo**: Quantificar efetividade de operações policiais
sobre redução/controle de Crimes Violentos Letais Intencionais

---

## 1. Efetividade Geral por Cidade

### 🟢 MUITO ALTA EFETIVIDADE (Prisões ↓↓ CVLI)

*(Mais prisões resulta em substancial redução de crimes)*

- **MORRINHOS**: corr=-0.591, Total: 23 prisões → 12 CVLI (10 períodos)
- **OROS**: corr=-0.577, Total: 6 prisões → 5 CVLI (4 períodos)

### 🟡 ALTA EFETIVIDADE (Prisões ↓ CVLI)

*(Padrão claro: aumento de operações → queda de crimes)*

- FORTIM: corr=-0.500, 16 prisões, 12 CVLI
- INDEPENDENCIA: corr=-0.447, 11 prisões, 10 CVLI
- PARAMBU: corr=-0.423, 21 prisões, 11 CVLI
- SENADOR POMPEU: corr=-0.358, 26 prisões, 17 CVLI
- BELA CRUZ: corr=-0.353, 78 prisões, 20 CVLI
- ICO: corr=-0.332, 139 prisões, 34 CVLI
- VARJOTA: corr=-0.304, 52 prisões, 48 CVLI
- GENERAL SAMPAIO: corr=-0.293, 9 prisões, 11 CVLI
- GRANJA: corr=-0.292, 107 prisões, 30 CVLI
- MASSAPE: corr=-0.285, 109 prisões, 23 CVLI

### ⚪ NEUTRA (sem padrão claro)

**62 cidades** com correlação entre -0.2 e 0.2

### 🔴 INEFICAZ (Prisões ↑↑ CVLI - SEM EFEITO ou PIORADO)

*(ALERTA: Aumento de operações NÃO reduz crimes - possível retalho, reorganização ou falta de integração)*

- **JAGUARUANA**: corr=0.510, 69 prisões vs 17 CVLI (↑↑↑)
- **PORANGA**: corr=0.518, 11 prisões vs 10 CVLI (↑↑↑)
- **TAMBORIL**: corr=0.530, 80 prisões vs 27 CVLI (↑↑↑)
- **MARCO**: corr=0.534, 76 prisões vs 19 CVLI (↑↑↑)
- **OCARA**: corr=0.556, 36 prisões vs 11 CVLI (↑↑↑)
- **REDENCAO**: corr=0.593, 15 prisões vs 8 CVLI (↑↑↑)
- **IPUEIRAS**: corr=0.663, 27 prisões vs 6 CVLI (↑↑↑)
- **JARDIM**: corr=0.972, 26 prisões vs 7 CVLI (↑↑↑)

**CSV**: outputs\docs\efetividade_prisoes_por_cidade.csv

---

## 2. Padrões de Impacto Detectados

### Operações com Resultado Positivo: 94 casos

*(Período: aumento de prisões → queda subsequente de CVLI)*

**PACATUBA** (Mês 7 → 8)
- Prisões: 7 → 10 (+3)
- CVLI: 7 → 6 (↓-14.3%)

**MARACANAU** (Mês 2 → 3)
- Prisões: 25 → 29 (+4)
- CVLI: 20 → 17 (↓-15.0%)

**PACATUBA** (Mês 5 → 6)
- Prisões: 11 → 12 (+1)
- CVLI: 6 → 5 (↓-16.7%)

**CAUCAIA** (Mês 8 → 9)
- Prisões: 19 → 33 (+14)
- CVLI: 23 → 19 (↓-17.4%)

**CASCAVEL** (Mês 5 → 6)
- Prisões: 20 → 22 (+2)
- CVLI: 5 → 4 (↓-20.0%)

**AMONTADA** (Mês 1 → 2)
- Prisões: 3 → 7 (+4)
- CVLI: 5 → 4 (↓-20.0%)

**ITAREMA** (Mês 6 → 7)
- Prisões: 4 → 7 (+3)
- CVLI: 4 → 3 (↓-25.0%)

**AQUIRAZ** (Mês 10 → 11)
- Prisões: 3 → 5 (+2)
- CVLI: 4 → 3 (↓-25.0%)

**SOBRAL** (Mês 4 → 5)
- Prisões: 27 → 30 (+3)
- CVLI: 4 → 3 (↓-25.0%)

**SAO GONCALO DO AMARANTE** (Mês 10 → 11)
- Prisões: 8 → 19 (+11)
- CVLI: 4 → 3 (↓-25.0%)

**ITAPIPOCA** (Mês 8 → 9)
- Prisões: 8 → 10 (+2)
- CVLI: 7 → 5 (↓-28.6%)

**PACAJUS** (Mês 2 → 3)
- Prisões: 9 → 13 (+4)
- CVLI: 3 → 2 (↓-33.3%)

**CANINDE** (Mês 7 → 8)
- Prisões: 3 → 4 (+1)
- CVLI: 3 → 2 (↓-33.3%)

**CRATEUS** (Mês 7 → 8)
- Prisões: 4 → 6 (+2)
- CVLI: 3 → 2 (↓-33.3%)

**ICO** (Mês 5 → 6)
- Prisões: 4 → 8 (+4)
- CVLI: 3 → 2 (↓-33.3%)


### Operações SEM Efeito (ou Contraproducentes): 116 casos

*(ALERTA: Aumento de prisões mas CVLI também aumentou)*

**VICOSA DO CEARA** (Mês 5 → 6)
- Prisões: 3 → 4 (+1)
- CVLI: 1 → 9 (↑800.0%) ⚠️

**GUAIUBA** (Mês 3 → 4)
- Prisões: 2 → 3 (+1)
- CVLI: 1 → 7 (↑600.0%) ⚠️

**PACATUBA** (Mês 8 → 9)
- Prisões: 10 → 14 (+4)
- CVLI: 1 → 7 (↑600.0%) ⚠️

**QUIXADA** (Mês 8 → 9)
- Prisões: 7 → 17 (+10)
- CVLI: 1 → 7 (↑600.0%) ⚠️

**TAMBORIL** (Mês 3 → 4)
- Prisões: 2 → 10 (+8)
- CVLI: 1 → 6 (↑500.0%) ⚠️

**CASCAVEL** (Mês 2 → 3)
- Prisões: 5 → 8 (+3)
- CVLI: 1 → 6 (↑500.0%) ⚠️

**TAUA** (Mês 2 → 3)
- Prisões: 2 → 3 (+1)
- CVLI: 1 → 6 (↑500.0%) ⚠️

**QUIXADA** (Mês 3 → 4)
- Prisões: 12 → 22 (+10)
- CVLI: 1 → 5 (↑400.0%) ⚠️

**BEBERIBE** (Mês 6 → 7)
- Prisões: 10 → 20 (+10)
- CVLI: 1 → 5 (↑400.0%) ⚠️

**HORIZONTE** (Mês 9 → 10)
- Prisões: 7 → 17 (+10)
- CVLI: 1 → 5 (↑400.0%) ⚠️

**CSV**: outputs\docs\impacto_prisoes_padroes.csv

---

## 3. Recomendações Operacionais

1. **Cidades com ALTA efetividade**: Manter/expandir operações RAIO (estratégia funcionando)
2. **Cidades com BAIXA efetividade**: Revisar tática operacional (possível retalho, desorganização)
3. **Cidades com padrão NEUTRO**: Integrar com outras inteligências (drogas, inteligência, fações)
4. **Correlações NEGATIVAS (pior caso)**: Investigar possível aumento de retaliatória/conflitos

---
**Análise gerada em:** 22 de janeiro de 2026