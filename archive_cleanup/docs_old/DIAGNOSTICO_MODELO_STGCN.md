# Diagnóstico: Capacidades e Limitações do Modelo ST-GCN

**Data:** Janeiro 2026  
**Status:** Transparência adicionada ao dashboard  
**Questão Central:** "Se o modelo espaço-temporal não faz nada diferente do scikit-learn, apenas gasta mais CPU, qual é o valor real?"

---

## 1. O QUE FOI ENTREGUE

### Transparência Implementada ✅
- Campo `explicacao_modelo` adicionado a cada recomendação no JSON da API
- Exibição visual no dashboard: seção **"🔍 Por que desta forma?"** em cada card
- Breakdown de fatores: histórico (90d) + tendência (%) + previsão do modelo

**Exemplo de saída:**
```
📊 Histórico: 8 homicídios em 90 dias
📈 Tendência: +15% vs período anterior
🤖 Modelo prevém risco de 32% para próximos dias
```

---

## 2. O PROBLEMA LEVANTADO (USER FEEDBACK)

**Citação do usuário:**
> "Padrões históricos eu faço até na mão os cálculos... quero aquilo que não consigo ver.  
> Padrão na sazonalidade, eventos críticos influenciando áreas e mudando estratégias."

**Interpretação:**
- ✅ O usuário VÊ dados históricos → não é valor agregado
- ❌ O usuário NÃO VÊ padrões complexos que exigem poder computacional
- ❌ O usuário suspeita que ST-GCN está fazendo o mesmo que EMA (média móvel exponencial)

---

## 3. ANÁLISE TÉCNICA: O QUE O MODELO DEVERIA FAZER

### Se o Modelo é Verdadeiramente Spatio-Temporal:

**1. Captura de Sazonalidade**
- Detecção automática de ciclos (horários, dias da semana, meses)
- Anomalias quando padrão quebra (ex: "quinta à noite geralmente é crítica, mas essa semana foi silenciosa")
- Output: "⏰ Padrão semanal não-linear detectado: picos Qua-Qui; anomalia em Jan/2026"

**2. Propagação Espacial**
- Crime em ponto A aumenta probabilidade em zona B próxima (efeito de vizinhança)
- Predição influenciada por geograficamente próximos, não apenas por histórico local
- Output: "🗺️ Influência de zona adjacente (CAPITAL): +8% risco por correlação espacial"

**3. Integração de Contexto Externo**
- Operações policiais reduzem risco local (feedback automático)
- Eventos externos (festas, eventos religiosos) aumentam atividade esperada
- Mudanças de regime (captura de liderança) alteram padrão estruturalmente
- Output: "🚓 Operação recente em zona: modelo ajusta predição -12%"

**4. Identificação de Mudanças de Padrão**
- Segmentação de períodos diferentes (pré/pós-evento)
- Detecção de inflexão (quando estratégia passada já não funciona)
- Output: "⚠️ Mudança de padrão detectada em dez/2025: estrutura antiga não aplica"

---

## 4. REALIDADE ATUAL: O QUE O MODELO FAZ

Baseado na análise de outputs (risco baixo em quase tudo), o modelo **provavelmente está fazendo**:

### ✅ O que funciona:
1. **Regressão temporal simples** → prediz com base em valores passados
2. **Agregação por região** → diferencia CAPITAL de INTERIOR
3. **Suavização de ruído** → reduz outliers isolados
4. **Correlação básica** → homicídios passados → risco futuro

### ❌ O que NÃO está acontecendo:
1. **Sazonalidade complexa** → não distingue "quiet Sábado" de "active Sábado após operação"
2. **Efeito de vizinhança** → cada bairro é ilhado, não há difusão espacial
3. **Contexto exógeno** → operações, eventos, mudanças políticas não alimentam o modelo
4. **Changepoint detection** → quando padrão quebra, modelo não percebe; continua usando histórico velho

### Indicador crítico de limitação:
**Scores baixos e pouca variação** → modelo está suavizando tudo para média  
= Equivalente a: `risco_previsto = (histórico_90d / max_histórico) * alfa_suavização`  
= **Não adiciona valor vs. especialista humano**

---

## 5. COMO VALIDAR SE O MODELO REALMENTE FUNCIONA

### Teste A: Sazonalidade
**Pergunta ao especialista:**
- "Existe padrão semanal forte? (ex: domingos são sempre quietos)"
- "Existe padrão sazonal? (ex: julho/agosto crime concentrado)"

**Se modelo realmente funciona:**
- API retorna: "⏰ Padrão semanal: picos Qua-Qui; Sábado 30% mais baixo"
- Hoje é sábado → Risco automaticamente reduzido
- Resultado: Recomendações mudam por-dia-da-semana mesmo com crimes constantes

**Se modelo está limitado:**
- API retorna: "Risco 32%" (sempre o mesmo, dia da semana não importa)
- Resultado: Recomendações idênticas seg/ter/qua/qui/sex/sab (apenas variam se história muda)

### Teste B: Impacto Espacial
**Pergunta ao especialista:**
- "Crime no SG afeta risco do Bairro X próximo?"
- "Existem zonas 'contagiadas' (maior risco quando vizinhos estão ativos)?"

**Se modelo funciona:**
- Previsão para Bairro A depende: A-histórico (60%) + proximidade de A (40%)
- Mesmo crime/dia, Risco-Bairro-A varia se B ao lado tem pico

**Se modelo está limitado:**
- Previsão para Bairro A: isolado, depende apenas de A-histórico
- Vizinhos não afetam predição

### Teste C: Evidência de Contexto
**Pergunta ao especialista:**
- "Depois de operação policial, risco cai? Quanto tempo demora a voltar?"
- "Existe evento externo que historicamente causa pico (final de período de salário, datas comemorativas)?"

**Se modelo funciona:**
- Flag: "Operação em zona" → modelo prevê redução automática por N dias
- Previsão incorpora calendário (Carnaval = risco +25%)

**Se modelo está limitado:**
- Operação não afeta modelo
- Datas especiais não geram padrão

---

## 6. RECOMENDAÇÃO IMEDIATA

### Para o Usuário:

Você tem **razão em questionar o modelo**. As evidências apontam:

1. **Scores baixos e monótonos** sugerem suavização excessiva (comportamento de EMA)
2. **Ausência de variação por sazonalidade** indica que modelo não captura ciclos
3. **Recomendações mudam principalmente por histórico** (não por padrões descobertos)

### Próximas Ações:

**Opção 1: Diagnosticar o Modelo (1-2 horas)**
- Extrair features que ST-GCN realmente usa (verificar `config.FEATURE_LIST`)
- Comparar output com regressão linear simples: `risco = a*crime_lag1 + b*crime_lag7 + c*crime_lag30`
- Se diferença < 5% → modelo NÃO está agregando valor → considerar substituição

**Opção 2: Enriquecer o Modelo (4-6 horas)**
- Adicionar features exógenas (operações, eventos, dia-da-semana-one-hot, sazonalidade)
- Retreinar ST-GCN com contexto
- Validar: novos scores devem ter maior variação e correlação com especialista

**Opção 3: Abordagem Híbrida (2-3 horas)**
- Manter ST-GCN como scoring base
- Adicionar regras expertise: "Se sazonalidade sugere pico, multiplica por 1.3"
- Adicionar detecção de anomalias: "Se padrão quebra historicamente, reduz confiança"
- Resultado: Combina rigor computacional com conhecimento humano

---

## 7. IMPLEMENTAÇÃO IMEDIATA: DIAGNÓSTICO

Vou criar um script que:
1. Extrai features do modelo (quais variáveis ele usa)
2. Compara ST-GCN vs regressão linear simples
3. Gera relatório: "Modelo agrega X% acima da baseline"
4. Valida: quando model diverge do especialista, quem acerta?

**Arquivo:** `scripts/diagnosticar_modelo_stgcn.py`

---

## 8. CHANGELOG

- **[2026-01-XX] Transparência**: Campo `explicacao_modelo` adicionado a API e dashboard
- **[2026-01-XX] Diagnóstico**: Este documento criado para estruturar validação de modelo
- **Próximo**: Script de validação automática

---

**Conclusão:**
Você não está errado em ser cético. A transparência foi adicionada, mas a **questão real é**: o ST-GCN está realmente capturando padrões espaço-temporais, ou está apenas fazendo suavização + correlação temporal básica?

Vou ajudá-lo a responder isso com dados.
