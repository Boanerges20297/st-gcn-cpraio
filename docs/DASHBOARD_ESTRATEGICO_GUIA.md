# Dashboard Estratégico Descritivo - Guia de Uso

## 📋 O QUE É

Um **dashboard separado** do SIGERAIO que fornece:
1. **Análise descritiva** clara da situação em Fortaleza
2. **Recomendações estratégicas** geradas por IA (Gemini)
3. **Indicadores operacionais** para decisão rápida

**Objetivo**: Um gestor olha, entende a situação, e sabe EXATAMENTE onde aplicar policiamento.

---

## 🎯 COMO ACESSAR

Quando o servidor estiver rodando:

```
http://localhost:5000/dashboard-estrategico
```

---

## 📊 O QUE VÊ NO DASHBOARD

### Seção 1: Situação Geral
```
📊 Números consolidados
├─ Total de Crimes: X.XXX
├─ Em Fortaleza: X.XXX
├─ Roubos (CVP): X.XXX
└─ Homicídios (CVLI): X.XXX
```

### Seção 2: Facções em Atuação
```
👥 Distribuição por facção
├─ CV: XXXX crimes
├─ TCP: XXX crimes
└─ [outras]: XX crimes
```

### Seção 3: Bairros Críticos
```
🚨 Ranking de risco (próximos 15 dias)
├─ 1. DE LOURDES → 33.30% (🔴 CRÍTICO)
├─ 2. AUTRAN NUNES → 32.49% (🔴 CRÍTICO)
├─ 3. VICENTE PINZÓN → 31.91% (🟠 ALTO)
└─ ... (todos os 138 bairros listados)
```

### Seção 4: Análise com IA
```
🤖 Análise Estratégica
└─ [BOTÃO] ⚡ Gerar Análise
   ↓
   [Gemini processa dados]
   ↓
   [Parecer tático recomendado]
```

---

## 🤖 COMO FUNCIONA A ANÁLISE COM IA

### O Fluxo:

1. **Você clica em "Gerar Análise"**
   - Estado: loading com spinner
   - Mensagem: "Analisando dados e gerando recomendações..."

2. **Backend coleta dados agregados**
   - Crimes históricos (parquet)
   - Distribuição por facção
   - Predições por bairro
   - Top 10 áreas críticas

3. **Gemini recebe contexto + prompt estratégico**
   ```
   "Você é analista de SP. Aqui estão os dados.
    Recomende onde aplicar policiamento."
   ```

4. **IA gera parecer EXECUTIVO**
   - Diagnóstico rápido
   - Hotspots prioritários (nomes dos bairros)
   - Tipo de crime predominante + facção
   - Recomendações operacionais claras
   - Métrica de sucesso

5. **Resultado aparece no dashboard**
   - Timestamp de quando foi gerado
   - Texto formatado e legível

### Exemplo de Resultado Esperado:

```
✓ Análise gerada com sucesso às 14:35

DIAGNÓSTICO EXECUTIVO
Fortaleza experimenta concentração crítica de crimes de roubo (CVP) em 
três zonas específicas, com atuação predominante da facção CV. O padrão 
prevê escalation de 15% nos próximos dias.

HOTSPOTS PRIORITÁRIOS
1. DE LOURDES (33.3% risco) - Zona de roubo a pedestres
2. AUTRAN NUNES (32.5% risco) - Roubo a residências
3. VICENTE PINZÓN (31.9% risco) - Roubo a comerciantes

TIPOLOGIA DO CRIME
Predominância de CVP (79% dos crimes): roubos patrimoniais. 
Homicídios (21%) concentrados em 4 bairros. Facção CV controla 99.9% 
da territorialidade em Fortaleza.

RECOMENDAÇÕES OPERACIONAIS
→ Reforço imediato: DE LOURDES, AUTRAN NUNES, VICENTE PINZÓN
→ Estratégia para CVP: bloqueios nas vias de fuga (Barra do Ceará, Praia)
→ Prioridade 1: Patrulhamento comunitário em DE LOURDES
→ Prioridade 2: Operação concentrada em AUTRAN NUNES
→ Prioridade 3: Monitoramento VICENTE PINZÓN

MÉTRICA DE SUCESSO
Redução de 15% em CVP nos hotspots em 7 dias. 
KPI: Crimes por bairro vs. predição inicial.
```

---

## 🔧 INTEGRAÇÃO COM SIGERAIO

### Arquitetura (Sem Saturação)

```
SIGERAIO (portal principal)
│
├─ Página 1: Mapa Tático (geoponto, crimes)
├─ Página 2: Mapa Territorial (facções)
│
└─ [NOVO] Menu: Dashboard Estratégico
           ↓
           Abre em nova aba: /dashboard-estrategico
           ├─ Análise descritiva (não é mapa)
           ├─ Números claros
           └─ Recomendações IA
```

**Vantagem**: Sem poluir SIGERAIO. Gestor escolhe quando quer:
- **Mapa tático** → Operação em tempo real
- **Mapa territorial** → Análise de dominância
- **Dashboard estratégico** → Decisão de alocação de recursos

---

## 📱 DESIGN UX

### Cores e Hierarquia
- 🔴 **CRÍTICO** (risco > 32%): Vermelho sangue
- 🟠 **ALTO** (31-32%): Laranja
- 🟡 **MÉDIO** (30-31%): Amarelo
- 🟢 **BAIXO** (< 25%): Verde

### Responsividade
- Desktop: 2 colunas (dados + facções lado a lado)
- Tablet/Mobile: 1 coluna (fluido)

### Interatividade
- Botão "Gerar Análise": visual feedback (disabled ao processar)
- Resultados scroll within box (max 600px altura)
- Timestamps mostram quando foi atualizado

---

## 🛠️ TECHNICAL STACK

| Componente | Tecnologia |
|-----------|-----------|
| **Frontend** | HTML5 + CSS3 + Vanilla JS |
| **Backend** | Flask (Python) |
| **Dados** | Parquet (histórico) + CSV (predições) |
| **IA** | Google Gemini API |
| **Status** | ✅ Pronto para usar |

---

## 📝 ROTAS DISPONÍVEIS

### 1. Dashboard UI
```
GET /dashboard-estrategico
```
Retorna a página HTML do dashboard

### 2. API de Dados
```
GET /api/strategic_insights
```
Retorna JSON com:
```json
{
  "sucesso": true,
  "data": {
    "total_crimes": 83295,
    "crimes_capital": 55252,
    "crime_types": {"CVP": 69046, "CVLI": 14249},
    "facctions": {"CV": 55xxx, "TCP": xxx},
    "top_bairros": [
      {"local_oficial": "DE LOURDES", "risco_previsto": 0.333},
      ...
    ]
  }
}
```

### 3. API de Análise IA
```
POST /api/ai_analysis
```
Retorna JSON com:
```json
{
  "sucesso": true,
  "analise": "Parecer completo gerado por Gemini...",
  "timestamp": "2026-01-16T14:35:22.123456"
}
```

---

## 🚀 COMO USAR NA PRÁTICA

### Cenário 1: Gestor quer saber onde reforçar
```
1. Abrir: http://localhost:5000/dashboard-estrategico
2. Ver top bairros (cores indicam criticidade)
3. Clicar: "⚡ Gerar Análise"
4. Ler parecer
5. Executar recomendações
```

### Cenário 2: Diretor quer justificar alocação orçamentária
```
1. Dashboard mostra números (crimes, facções)
2. Análise IA justifica "Por que aqui"
3. Métricas de sucesso definem KPIs
```

### Cenário 3: Operações em tempo real
```
1. Manter SIGERAIO (mapas táticos) abertos
2. Usar Dashboard quando precisa de direcionamento estratégico
3. "Hoje vamos intensificar em X,Y,Z" (baseado em recomendação)
```

---

## ⚙️ CONFIGURAÇÃO

### Variáveis de Ambiente Necessárias
```
GEMINI_KEY_1=sua_chave_aqui
```

(Ou usar pool de chaves: GEMINI_KEY_1, GEMINI_KEY_2, GEMINI_KEY_3)

### Arquivos Necessários
```
✓ outputs/reports/pred_capital_bairros.csv
✓ data/processed/base_consolidada.parquet
✓ src/gemini_client.py (existente)
```

---

## 📈 MÉTRICAS E MONITORAMENTO

### O que monitorar após usar o dashboard

1. **Efetividade das recomendações**
   - Risco previsto vs. risco realizado
   - Redução de crimes nos bairros recomendados

2. **Tempo de decisão**
   - Antes: X horas (sem dashboard)
   - Depois: Minutos (com dashboard)

3. **Cobertura operacional**
   - % de bairros críticos com atuação
   - Taxa de crime vs. predição

---

## 🔐 SEGURANÇA

- ✅ Dashboard só acessa dados agregados (sem PII)
- ✅ Gemini não recebe dados de criminosos (apenas estatísticas)
- ✅ Timestamps para auditoria
- ✅ Cache local no navegador (histórico de análises)

---

## 🐛 TROUBLESHOOTING

### Problema: "Dados não disponíveis"
**Solução**: Rodar ETL primeiro
```bash
python src/etl.py
```

### Problema: "Erro de conexão com Gemini"
**Solução**: Verificar GEMINI_KEY no .env
```bash
echo "GEMINI_KEY_1=sua_chave" >> .env
```

### Problema: "Análise muito lenta"
**Solução**: Gemini pode estar saturado. Esperar 30s.

---

## 📞 SUPORTE

**Integração com SIGERAIO?**
- Adicionar link em menu superior
- URL: `/dashboard-estrategico`
- Descrição: "Análise Estratégica com IA"

**Customizações?**
- Cores dos alertas: editar CSS em `dashboard_estrategico.html`
- Prompt da IA: editar `/api/ai_analysis` em `app.py`
- Dados inclusos: gerenciar em `get_strategic_insights()`

---

✅ **Status**: Pronto para produção
🎯 **Objetivo alcançado**: Gestor vê dados, clica, recebe recomendação IA, aplica policiamento
