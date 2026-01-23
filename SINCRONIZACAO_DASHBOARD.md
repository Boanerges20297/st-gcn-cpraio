# 🔄 SINCRONIZAÇÃO DO DASHBOARD COM NOVO MODELO

**Data:** 23 de Janeiro de 2026  
**Versão:** 2.0 com Dinâmica de Facções  
**Status:** ✅ Sincronização Completa

---

## 📊 Como o Novo Modelo Reflete no Dashboard

### Antes (Arquivo Consolidado)
```
Dashboard → Busca arquivo consolidado.parquet
           → Se não existe → "Dados Indisponíveis"
```

### Depois (Data Adapter + Novas Rotas)
```
Dashboard → /api/dashboard_sync
           → DataAdapter carrega predições + tensor + facções
           → Retorna dados estruturados
           → ✅ "Dados Disponíveis"
```

---

## 🎯 Dados Síncronos com o Dashboard

### 1. **GET `/api/dashboard_sync`** - Dashboard Principal

Fornece tudo que o dashboard precisa em uma única requisição.

**Response:**
```json
{
  "sucesso": true,
  "data": {
    "top_15_bairros": [
      {
        "bairro": "Jangurussu",
        "cvli_predito": 0.0800,
        "prob_mudanca": 0.0,
        "volatilidade": 0.123
      },
      ...
    ],
    "metricas_globais": {
      "total_bairros": 121,
      "bairros_criticos": 12,
      "cvli_medio": 0.0135,
      "periodo": "210 dias (23/01/2026 a 21/08/2026)"
    },
    "por_regiao": {
      "CAPITAL": {
        "cvli_medio": 0.014,
        "cvli_max": 0.08,
        "bairros_criticos": 5,
        "volatilidade_media": 0.12
      },
      "PERIFERIA": {
        "cvli_medio": 0.013,
        "cvli_max": 0.072,
        "bairros_criticos": 7,
        "volatilidade_media": 0.13
      }
    },
    "timeline_ultimos_30_dias": [
      {
        "data": "2022-01-01",
        "cvli_medio": 0.0179,
        "mudancas_territoriais": 0.0,
        "volatilidade": 0.15
      },
      ...
    ]
  }
}
```

**Uso no Dashboard:**
```javascript
// Buscar dados
fetch('/api/dashboard_sync')
  .then(r => r.json())
  .then(data => {
    // Top bairros
    data.data.top_15_bairros.forEach(b => {
      console.log(`${b.bairro}: ${b.cvli_predito}`);
    });
    
    // Métricas
    document.getElementById('total-bairros').textContent = 
      data.data.metricas_globais.total_bairros;
    
    // Regiões
    Object.entries(data.data.por_regiao).forEach(([regiao, dados]) => {
      console.log(`${regiao}: ${dados.cvli_medio.toFixed(4)}`);
    });
    
    // Timeline (para gráfico)
    plotarTimeline(data.data.timeline_ultimos_30_dias);
  });
```

---

### 2. **GET `/api/bairro_detalhes/<bairro>`** - Detalhe de um Bairro

Clique em um bairro no mapa → Abre painel lateral com detalhes.

**Exemplo:**
```bash
GET /api/bairro_detalhes/Jangurussu
```

**Response:**
```json
{
  "sucesso": true,
  "data": {
    "bairro": "Jangurussu",
    "cvli_predito": 0.0800,
    "score_risco": 100.0,
    "prob_mudanca": 0.0,
    "volatilidade": 0.123,
    "recomendacao": "🔴 CRÍTICO - Reforço policial imediato necessário",
    "cor_risco": "#ff0000",
    "risco_territorial": "NORMAL",
    "volatilidade_status": "NORMAL",
    "horizonte": "210 dias (23/01/2026 a 21/08/2026)"
  }
}
```

**Uso no Dashboard (painel lateral):**
```javascript
// Ao clicar em um bairro
function mostrarDetalhes(bairro) {
  fetch(`/api/bairro_detalhes/${encodeURIComponent(bairro)}`)
    .then(r => r.json())
    .then(data => {
      if (data.sucesso) {
        document.getElementById('painel-bairro').innerHTML = `
          <h2>${data.data.bairro}</h2>
          <div style="background: ${data.data.cor_risco}; padding: 10px;">
            Score: ${data.data.score_risco.toFixed(1)}/100
            ${data.data.recomendacao}
          </div>
          <p>CVLI Predito: ${data.data.cvli_predito.toFixed(4)}</p>
          <p>Risco Territorial: ${data.data.risco_territorial}</p>
          <p>Volatilidade: ${data.data.volatilidade_status}</p>
        `;
      }
    });
}
```

---

## 🔧 Mudanças na Estrutura de Dados

### Antes
```
Arquivo: base_consolidada.parquet
Colunas: [id, data, local, crimes, ...]
Problema: Arquivo não era regenerado após novo modelo
```

### Depois
```
Arquivo: predicoes_cvli.csv (novo modelo)
Colunas: [bairro, cvli_predito, prob_mudanca, volatilidade]

Tensor: tensor_cvli_prisoes_faccoes.npy
Shape: (1472 dias, 121 bairros, 7 features)
Features: [CVLI, Prisões, Apreensões, Mudança, Estabilidade, Conflito, Volatilidade]

Adapter: Sincroniza tudo automaticamente via DataAdapter
```

---

## 📈 Mapa Visual - Cores de Risco

```
Score de Risco (0-100)
├─ 75-100 (🔴) → CRÍTICO (Reforço imediato)
├─ 50-75  (🟠) → ALTO    (Vigilância reforçada)
├─ 25-50  (🟡) → MÉDIO   (Monitoramento)
└─ 0-25   (🟢) → BAIXO   (Rotina)

Volatilidade Territorial
├─ > 70%    → CRÍTICA (Protocolo de monitoramento)
├─ 40-70%   → ALTA    (Aumentar patrulhamento)
├─ 20-40%   → MÉDIA   (Manter vigilância)
└─ < 20%    → BAIXA   (Rotina)

Risco Territorial
├─ Prob Mudança > 30% → ALTO RISCO
└─ Prob Mudança ≤ 30% → NORMAL
```

---

## 🚀 Checklist de Atualização do Dashboard

- [ ] **Remover** busca por `config.CONSOLIDATED_FILE`
- [ ] **Adicionar** chamada para `/api/dashboard_sync` ao carregar página
- [ ] **Usar** `top_15_bairros` para ranking visual
- [ ] **Usar** `metricas_globais` para números principais
- [ ] **Usar** `por_regiao` para análise regional
- [ ] **Usar** `timeline_ultimos_30_dias` para gráfico de série temporal
- [ ] **Implementar** clique em bairro → `/api/bairro_detalhes/<bairro>`
- [ ] **Colorir** mapa com base em `score_risco` (0-100)
- [ ] **Exibir** indicador de `risco_territorial` com badge
- [ ] **Atualizar** legenda: "Predições 210 dias (ST-GCN + Facções)"

---

## 📋 Rotas Alteradas/Novas

### ✅ Novas (Sincronização)
```
GET /api/dashboard_sync              ← Dashboard principal
GET /api/bairro_detalhes/<bairro>    ← Detalhe de bairro
```

### ✅ Mantidas (Compatibilidade)
```
GET /api/cvli_forecast_extended      ← Predições estendidas
GET /api/territorial_volatility/<b>  ← Volatilidade
GET /api/faction_timeline            ← Timeline de facções
GET /api/recomendacoes_operacionais  ← Recomendações táticas
```

---

## 🧪 Teste Rápido

```bash
# 1. Iniciar servidor
.\.venv\Scripts\python.exe src/app.py

# 2. Em outro terminal, testar
curl "http://localhost:5000/api/dashboard_sync" | jq '.data.metricas_globais'

curl "http://localhost:5000/api/bairro_detalhes/Jangurussu" | jq '.data.recomendacao'
```

---

## 📝 Anotações Técnicas

### DataAdapter (src/data_adapter.py)
- Carrega predições + tensor + facções automaticamente
- Sincroniza ao iniciar app (`init_adapter()`)
- Cache em memória (rápido)
- Pode ser atualizado executando `python src/predict_with_factions.py`

### Período de Dados
- **Histórico:** 2022-01-01 a 2026-01-23 (1472 dias)
- **Predição:** 2026-01-23 a 2026-08-21 (210 dias)
- **Período Atual:** 210 dias à frente

### Dimensões
- **121 bairros** analisados
- **7 features** por bairro-dia (crime + facções)
- **12 bairros críticos** identificados
- **18 bairros de alto risco**

---

## ✅ Status de Integração

| Componente | Status | Nota |
|-----------|--------|------|
| Predições | ✅ | Carregadas em `/api/dashboard_sync` |
| Tensor | ✅ | Sincronizado para timeline |
| Facções | ✅ | Análise disponível em `/api/faction_timeline` |
| Dashboard | ⚠️ | **Aguarda atualização HTML** |
| APIs | ✅ | Todas as 5 rotas funcionando |

---

**Próximo Passo:** Atualizar `templates/dashboard_estrategico.html` para usar as novas rotas.
