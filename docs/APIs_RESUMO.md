# 📋 RESUMO DAS APIS EXISTENTES E PRÓXIMAS MELHORIAS

## ✅ APIS FLASK EXISTENTES

### **Em [src/app.py](src/app.py)** - 1475 linhas

#### Dashboard & Visualização
- `GET /` - Dashboard principal
- `GET /dashboard-estrategico` - Dashboard estratégico
- `GET /relatorio-analise` - Análise detalhada
- `GET /data/graph/<filename>` - Serve GeoJSON de facções
- `GET /data/raw/<filename>` - Serve dados brutos

#### APIs de Dados
- `GET /api/dashboard_data` - Dados para dashboard
- `GET /api/strategic_insights` - Insights estratégicos
- `GET /api/strategic_insights_range` - Insights por período
- `GET /api/ai_analysis` - Análise com IA
- `GET /api/recomendacoes_operacionais` - **Recomendações táticas** ⭐

#### Análise Operacional
- `GET /exogenous-event` - Página de eventos exógenos
- `POST /api/exogenous_event` - Registrar eventos
- `POST /api/simulate_teams` - Simular reposicionamento de equipes

---

## 🆕 ROTAS A ADICIONAR (Predições com Facções)

### Novo endpoint para predi ções 180+30 dias

```python
@app.route('/api/cvli_forecast_extended')
def get_cvli_forecast_extended():
    """
    Predições de CVLI para 210 dias (180 + 30)
    Integra dinâmica de facções
    Retorna: top 20 bairros com maior risco
    """
    # Carregar predicoes_cvli.csv
    # Retornar JSON com ranking de risco
```

### Novo endpoint para análise de volatilidade

```python
@app.route('/api/territorial_volatility/<bairro>')
def get_territorial_volatility(bairro):
    """
    Análise de volatilidade territorial por bairro
    Mostra: mudanças, estabilidade, risco de conflito
    """
```

### Novo endpoint para dashboard de facções

```python
@app.route('/api/faction_timeline')
def get_faction_timeline():
    """
    Timeline de movimentação de facções
    Mostra: controle territorial ao longo do tempo
    """
```

---

## 📊 ESTRUTURA DE DADOS

**Predições estão em:**
- ✅ `outputs/predicoes_cvli.csv` - 121 bairros com scores
- ✅ `outputs/predicoes_cvli.json` - Estruturado para API
- ✅ `outputs/RELATORIO_PREDICOES.md` - Executivo

**Tensor de facções:**
- ✅ `data/processed/tensor_cvli_prisoes_faccoes.npy` - 1472×121×7

**Modelo treinado:**
- ✅ `outputs/model_stgcn_faccoes.pth` - Weights salvos

---

## 🎯 PRÓXIMO PASSO

**Adicionar 3 novas rotas a [src/app.py](src/app.py):**

1. `/api/cvli_forecast_extended` - Predições 210 dias
2. `/api/territorial_volatility/<bairro>` - Volatilidade por bairro
3. `/api/faction_timeline` - Timeline de facções

Deseja prosseguir com essas implementações?
