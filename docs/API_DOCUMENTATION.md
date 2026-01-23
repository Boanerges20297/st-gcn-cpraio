# 📚 DOCUMENTAÇÃO DA API FLASK - ST-GCN COM DINÂMICA DE FACÇÕES

**Base URL:** `http://localhost:5000`  
**Status:** ✅ Ativo com 3 novas rotas integradas

---

## 🆕 NOVAS ROTAS (Predições 210 dias)

### 1️⃣ GET `/api/cvli_forecast_extended`

Predições de CVLI para 210 dias (180 + 30) integrando dinâmica de facções.

**Query Parameters:**
```
?top=20          # Número de bairros a retornar (default: 20)
?min_risco=0     # Filtro de risco mínimo (default: 0)
```

**Exemplo:**
```bash
curl "http://localhost:5000/api/cvli_forecast_extended?top=15&min_risco=0"
```

**Response (200):**
```json
{
  "sucesso": true,
  "data": {
    "horizonte_dias": 210,
    "periodo": "23/01/2026 a 21/08/2026",
    "total_bairros": 121,
    "bairros_criticos": 12,
    "bairros_alto_risco": 18,
    "previsoes": [
      {
        "bairro": "Jangurussu",
        "cvli_predito": 0.0800,
        "prob_mudanca": 0.0,
        "volatilidade": 0.123,
        "classificacao": "CRÍTICO",
        "risco_territorialidade": "NÃO"
      },
      ...
    ],
    "metricas": {
      "cvli_medio": 0.0135,
      "cvli_max": 0.0800,
      "cvli_min": 0.0005,
      "bairros_com_mudanca_territorial": 0
    }
  }
}
```

**Classificação de Risco:**
- 🔴 **CRÍTICO**: > 90º percentil
- 🟠 **ALTO**: 75-90º percentil  
- 🟡 **MÉDIO**: 50-75º percentil
- 🟢 **BAIXO**: < 50º percentil

---

### 2️⃣ GET `/api/territorial_volatility/<bairro>`

Análise detalhada de volatilidade territorial por bairro.

**Path Parameters:**
```
<bairro>  # Nome do bairro (URL encoded, ex: Barra%20Do%20Ceara)
```

**Exemplo:**
```bash
curl "http://localhost:5000/api/territorial_volatility/Jangurussu"
```

**Response (200):**
```json
{
  "sucesso": true,
  "data": {
    "bairro": "Jangurussu",
    "cvli_predito": 0.0800,
    "volatilidade_territorial": {
      "nivel": "BAIXO",
      "cor": "#00cc00",
      "prob_mudanca": 0.0,
      "volatilidade_index": 0.123
    },
    "faccoes": {},
    "recomendacoes": [
      "✅ Territorialidade estável - Manutenção rotineira"
    ],
    "periodo_predicao": "210 dias (23/01/2026 a 21/08/2026)"
  }
}
```

**Níveis de Volatilidade:**

| Nível | Condição | Ação |
|-------|----------|------|
| 🔴 CRÍTICO | prob_mudança > 50% OU volatilidade > 70% | Reforço imediato, protocolo de monitoramento |
| 🟠 ALTO | prob_mudança > 30% OU volatilidade > 40% | Aumentar patrulhamento, coordenar com inteligência |
| 🟡 MÉDIO | prob_mudança > 10% OU volatilidade > 20% | Manter presença, acompanhar tendências |
| 🟢 BAIXO | Abaixo de MÉDIO | Manutenção rotineira |

---

### 3️⃣ GET `/api/faction_timeline`

Timeline de movimentação de facções baseada em snapshots GeoJSON.

**Exemplo:**
```bash
curl "http://localhost:5000/api/faction_timeline"
```

**Response (200):**
```json
{
  "sucesso": true,
  "data": {
    "ultima_atualizacao": "N/A",
    "faccoes_identificadas": {},
    "bairros_analisados": 0,
    "bairros_com_mudancas": 0,
    "timeline": [],
    "resumo": {
      "total_snapshots": 1,
      "periodo": "23/01/2026"
    }
  }
}
```

**Estrutura de Facções:**
```json
{
  "faccoes_identificadas": {
    "COMANDO VERMELHO": {
      "bairros_controlados": 25,
      "territorio_km2": 150.5,
      "ultimo_snapshot": "23/01/2026"
    },
    ...
  }
}
```

---

## ✅ ROTAS EXISTENTES (Mantidas)

### Recomendações Operacionais
```
GET /api/recomendacoes_operacionais
  ?data_inicio=2026-01-01
  ?data_fim=2026-01-23
  ?regiao=CAPITAL
```
Retorna recomendações táticas baseadas em dados reais + predições.

### Dashboard
```
GET /api/dashboard_data
GET /api/strategic_insights
GET /api/strategic_insights_range
GET /api/ai_analysis
```

### Visualização
```
GET /dashboard-estrategico
GET /relatorio-analise
```

---

## 🔧 EXEMPLOS DE USO

### Python
```python
import requests

# Obter top 15 bairros de risco
resp = requests.get('http://localhost:5000/api/cvli_forecast_extended?top=15')
data = resp.json()

# Iterar sobre bairros
for pred in data['data']['previsoes']:
    print(f"{pred['bairro']}: {pred['cvli_predito']:.4f} ({pred['classificacao']})")

# Análise de volatilidade para um bairro específico
resp = requests.get('http://localhost:5000/api/territorial_volatility/Jangurussu')
volatility = resp.json()['data']['volatilidade_territorial']
print(f"Nível: {volatility['nivel']}, Prob: {volatility['prob_mudanca']:.1%}")
```

### JavaScript/Fetch
```javascript
// Obter predições estendidas
fetch('/api/cvli_forecast_extended?top=10')
  .then(r => r.json())
  .then(data => {
    console.log(`Bairros críticos: ${data.data.bairros_criticos}`);
    data.data.previsoes.forEach(p => {
      console.log(`${p.bairro}: ${p.cvli_predito.toFixed(4)}`);
    });
  });

// Análise de volatilidade
fetch('/api/territorial_volatility/Jangurussu')
  .then(r => r.json())
  .then(data => {
    const vol = data.data.volatilidade_territorial;
    console.log(`${data.data.bairro}: ${vol.nivel} (${vol.prob_mudanca.toFixed(1)}%)`);
  });
```

### cURL
```bash
# Top 10 bairros
curl -s "http://localhost:5000/api/cvli_forecast_extended?top=10" | jq '.data.previsoes'

# Volatilidade
curl -s "http://localhost:5000/api/territorial_volatility/Jangurussu" | jq '.data.volatilidade_territorial'

# Timeline de facções
curl -s "http://localhost:5000/api/faction_timeline" | jq '.data.faccoes_identificadas'
```

---

## 📊 ESTRUTURA DE DADOS

### Predição (CSV)
```csv
bairro,cvli_predito,prob_mudanca,volatilidade
Jangurussu,0.0800,0.0,0.123
Barra Do Ceará,0.0718,0.0,0.145
...
```

### Tensor
```
Shape: (1472 dias, 121 bairros, 7 features)
Features:
  0: CVLI (homicídios)
  1: Prisões
  2: Apreensões
  3: Mudança territorial (0/1)
  4: Estabilidade (dias, 0-365)
  5: Risco conflito (0-1)
  6: Volatilidade (0-1)
```

### Modelo
```
STGCN_DynamicFactions
  - Parâmetros: 25.346
  - Input: (batch, 14 dias, 121 bairros, 7 features)
  - Output: (batch, 121 bairros, 1 CVLI predito)
  - Auxiliar: (batch, 121 bairros, 1 prob_mudança)
```

---

## ⚠️ ERROS COMUNS

| Erro | Causa | Solução |
|------|-------|---------|
| 404 - Predições não disponíveis | Arquivo não gerado | Executar `python src/predict_with_factions.py` |
| 404 - Bairro não encontrado | Nome incorreto | Usar URL encoding, ex: `Barra%20Do%20Ceara` |
| 500 - JSON serialization | Tipos pandas | Reiniciar app |
| 500 - File not found | Caminho incorreto | Verificar `config.py` |

---

## 🚀 INICIAR SERVIDOR

```bash
# Modo desenvolvimento
.\.venv\Scripts\python.exe -m flask run --host=0.0.0.0 --port=5000

# Ou via app.py
.\.venv\Scripts\python.exe src/app.py

# Com auto-reload (requer watchdog)
flask --app src.app run --reload
```

---

## 📈 ENDPOINTS POR CASO DE USO

### 🎯 Para Gestor Operacional
1. `/api/recomendacoes_operacionais` - Decisões táticas
2. `/api/cvli_forecast_extended` - Visão geral de risco
3. `/api/territorial_volatility/<bairro>` - Detalhe de um bairro

### 🔬 Para Análise de Dados
1. `/api/cvli_forecast_extended?top=121` - Todos os bairros
2. `/api/faction_timeline` - Histórico de movimentação
3. `/api/strategic_insights_range` - Análise temporal

### 📊 Para Dashboard
1. `/api/dashboard_data` - Visualização principal
2. `/api/strategic_insights` - Gráficos e métricas
3. `/data/graph/*` - Mapas GeoJSON

---

**Versão:** 2.0 com Dinâmica de Facções  
**Data:** 23 de Janeiro de 2026  
**Status:** ✅ Produção-ready
