# 📊 Análise Cruzada CVLI × Prisões — Índice Consolidado

## 🎯 Visão Geral

Este conjunto de análises cruza dados de **Sazonalidade de CVLI** com **Operações Policiais (Prisões)** para identificar:
- **Correlações**: Quando crimes e operações fluem juntos
- **Divergências**: Quando aumento de prisões não reduz crimes
- **Efetividade**: Cidades onde operações são mais/menos eficazes
- **Padrões de Impacto**: Períodos com operações bem-sucedidas

---

## 📁 Documentos Gerados

### 1. **ANALISE_CVLI_PRISOES_CRUZADO.md** 
Análise inicial de correlações e divergências
- Correlações por cidade (Pearson, Spearman)
- Padrões de divergência (aumento de prisões mas CVLI ↑)
- 154 casos de divergência forte detectados

### 2. **ANALISE_IMPACTO_PRISOES_AVANCADA.md** ⭐ **LEIA ESTE**
Análise aprofundada de efetividade operacional
- **Cidades com ALTA efetividade** (Prisões ↓ CVLI):
  - MORRINHOS: -0.591 correlação (23 prisões → 12 CVLI)
  - OROS: -0.577 correlação
  - FORTIM, INDEPENDÊNCIA, PARAMBU, etc.
- **Cidades com BAIXA efetividade** (Prisões ↑ CVLI):
  - JARDIM: +0.972 correlação (⚠️ ALERTA)
  - IPUEIRAS: +0.663 correlação
  - JAGUARUANA, PORANGA, TAMBORIL, etc.
- **210 padrões de impacto** (operações com resultado)
  - 94 operações com resultado POSITIVO
  - 116 operações SEM EFEITO ou contraproducentes

---

## 📊 CSVs de Dados Tabulares

### Efetividade por Cidade
- **efetividade_prisoes_por_cidade.csv**
  - 127 cidades analisadas
  - Colunas: cidade, correlação, categorias, totais CVLI/prisões

### Padrões de Impacto
- **impacto_prisoes_padroes.csv**
  - 210 padrões detectados
  - Colunas: cidade, mês, antes/depois (prisões, CVLI), tipo de impacto

### Correlações (Primeira Análise)
- **cvli_prisoes_correlacao_por_cidade.csv**
  - Detalhes de correlação Pearson/Spearman
  - Tendências e médias

### Divergências (Primeira Análise)
- **cvli_prisoes_divergencias_forte.csv**
  - Casos de padrões opostos (prisões ↑ mas CVLI ↓ ou vice-versa)

---

## 🔍 Principais Achados

### ✅ Operações EFICAZES (correlação negativa)
- **MORRINHOS**: -0.591 corr. → Reduz CVLI com prisões
- **OROS**: -0.577 corr. → Estratégia funcionando
- **FORTIM**: -0.500 corr. → Operações têm impacto

**Recomendação**: Expandir modelo operacional destas cidades

### ⚠️ Operações INEFICAZES (correlação positiva)
- **JARDIM**: +0.972 corr. → Operações NÃO reduzem crimes
- **IPUEIRAS**: +0.663 corr. → Possível retalho ou falta de integração
- **JAGUARUANA**: +0.510 corr. → Investigar causa

**Recomendação**: Revisar tática, integrar com inteligência, analisar possível retalho

### 📈 Padrões de Sucesso (94 casos)
- **PACATUBA**: Mês 7→8: +3 prisões, -14.3% CVLI ✓
- **CAUCAIA**: Mês 8→9: +14 prisões, -17.4% CVLI ✓
- **SOBRAL**: Mês 4→5: +3 prisões, -25% CVLI ✓

### 📉 Padrões de Falha (116 casos)
- Operações intensas mas CVLI continua subindo
- Possíveis causas:
  - Ausência de inteligência preventiva
  - Retalho/conflito com facções (pioram situação)
  - Falta de coordenação com polícia civil/federal
  - Inadequação de prisões (pequenos tráficos) vs. homicídios

---

## 🎯 Recomendações Operacionais

### Curto Prazo (0-3 meses)
1. **Cidades ALERTA** (JARDIM, IPUEIRAS, etc.): Revisar operações, investigar desorganização
2. **Cidades SUCESSO** (MORRINHOS, OROS, etc.): Replicar modelo em outras regiões
3. **Integração**: Trazer inteligência de drogas + gangues para operações de CVLI

### Médio Prazo (3-6 meses)
1. Criar "manual" de operações eficazes baseado em MORRINHOS/OROS
2. Treinar equipes em cidades de correlação neutra para passar para positiva
3. Monitorar evolução mensal de efetividade por região

### Longo Prazo (6+ meses)
1. Construir modelo preditivo: dados de operações → predição de CVLI futuro
2. Integrar com ST-GCN: usar operações histórias como **feature exógena**
3. Feedback loop: modelo → predição → recomendação operacional → resultado

---

## 📌 Correlações Explicadas

- **r > 0.7**: Forte correlação positiva (prisões ↑ E CVLI ↑)
- **0.3 < r < 0.7**: Correlação moderada (alguma relação)
- **-0.3 < r < 0.3**: Fraca/neutra (sem padrão claro)
- **-0.7 < r < -0.3**: Correlação negativa moderada (prisões ↓ CVLI ↓) ✓ DESEJÁVEL
- **r < -0.7**: Forte correlação negativa (operações MUY EFICAZES) ✓✓ IDEAL

---

## 📋 Próximas Análises Sugeridas

1. **Análise por Facção**: Prisões de PCC/CV vs CVLI naquele território
2. **Análise Temporal com Lag**: Detectar delay entre operação → redução de crime (ex: 2 meses depois)
3. **Análise de Bairro (Fortaleza)**: Granularidade detalhada nas operações RAIO
4. **Análise de Tipo de Crime**: Prisões por tráfico vs prisões por outra natureza
5. **Integração com ST-GCN**: Features de operações passadas como input do modelo

---

**Análise Executada**: 22 de janeiro de 2026  
**Próximo Checkpoint**: Resumo Phase-4 ST-GCN Training

