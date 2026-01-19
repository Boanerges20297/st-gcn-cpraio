# RELATÓRIO DE VIABILIDADE: TREINO DO MODELO COM NOVOS PARÂMETROS
**Data:** 19 de Janeiro de 2026  
**Status:** ✅ VIÁVEL COM IMPLEMENTAÇÕES  

---

## 📋 SUMÁRIO EXECUTIVO

O modelo ST-GCN CPRAIO pode ser **treinado com os novos parâmetros solicitados**, porém requer:
- ✅ Normalização de dados do JSON operacional (possível)
- ✅ Priorização de CVLI (implementação simples)
- ✅ Integração de drogas/armas como features (requer engenharia)
- ✅ Correlação temporal com prisões (requer dataset estruturado)
- ⚠️ Validação de eficácia (depende de dados de 2025)

**Cronograma estimado:** 15-20 dias úteis

---

## 🔍 ANÁLISE DETALHADA

### 1. ESTADO ATUAL DO MODELO

#### Arquitetura Implementada:
```
STGCN_Cpraio (src/model.py):
├── LSTM: Processamento temporal (14 dias)
├── GCN: Aprendizado espacial + facções
└── Head: Predição linear (próximos 15 dias)
```

#### Dados Utilizados Atualmente:
- **X_base**: Contagem diária de crimes por local
- **Edge Index**: Topologia física (vizinhos geométricos) + lógica (facções)
- **Features Exógenas**: Precipitação (INMET), temperatura, feriados

#### Treinamento Atual:
- Regiões: CAPITAL, RMF, INTERIOR
- Window size: 14 dias
- Target window: 15 dias
- Épocas: 200 com early stopping

---

### 2. DADOS DISPONÍVEIS

#### ✅ Existentes e Utilizáveis:

| Dados | Fonte | Status | Volume |
|-------|-------|--------|--------|
| **Ocorrências Gerais** | base_consolidada_orcrim_v3.parquet | ✅ Integrado | ~9.000 registros (jan/2025) |
| **Territorios Faccionados** | data/raw/inteligencia/*.geojson | ✅ Disponível | CV, GDE, PCC (3 facções mapeadas) |
| **JSON Operacional** | ocorrencia_policial_operacional.json | ⚠️ Precisa normalização | 9.069 registros (janela) |
| **Ocorrências de Tropa** | ocorrencias_tropa.json | ⚠️ Precisa parsing | ~500 registros (eventos críticos) |

#### ⚠️ Campos Relevantes no JSON Operacional:

```json
{
  "Natureza": "TRÁFICO DE DROGAS",           // Pode ser CVLI?
  "total_drogas_cache": "345.00",            // ✅ Já existe!
  "total_armas_cache": "1",                  // ✅ Já existe!
  "area_faccao": "CV",                       // ✅ Facção identificada
  "Data": "2025-01-15",                      // ✅ Série temporal
  "lat_long": "-3.7668038,-38.584197"        // ✅ Georreferência
}
```

**Achado crítico:** Os campos `total_drogas_cache` e `total_armas_cache` **JÁ EXISTEM** no dataset!

---

### 3. REQUISITOS vs VIABILIDADE

#### ✅ Requisito 1: Ocorrências Gerais (CIops)
**Status:** VIÁVEL  
**Implementação:** Já integrado via `base_consolidada_orcrim_v3.parquet`  
**Ação:** Nenhuma (já funciona)

---

#### ✅ Requisito 2: Territórios Faccionados (Georreferência)
**Status:** VIÁVEL  
**Dados:** `data/raw/inteligencia/*.geojson`  
- CV (Comando Vermelho): ✅ Mapeado
- GDE (Guardiões do Estado): ✅ Mapeado
- PCC (Primeiro Comando da Capital): ✅ Mapeado

**Implementação Necessária:**
1. Integrar GeoJSON de facções à topologia do grafo
2. Criar camada "lógica" baseada em territórios (já existe em `graph_builder.py`)
3. Ponderar edges por overlapping territorial

**Ação:** Melhorar `graph_builder.py` (2-3 dias)

---

#### ⚠️ Requisito 3: Prisões Equipes Raio (Normalização)
**Status:** PARCIALMENTE VIÁVEL  
**Dados:** `ocorrencias_tropa.json`  
**Desafios:**
- Formato não estruturado (texto narrativo)
- Falta latitude/longitude em muitos registros
- Dados fragmentados em 2025

**O que precisa ser feito:**
1. **Parsing NLP/Regex** das narrativas para extrair:
   - Local da prisão
   - Datas/horas
   - Quantidade de presos
   - Armas/drogas apreendidas

2. **Normalização de coordenadas:**
   - Alguns registros têm DMS (latitude: -5°15'53.4"S)
   - Converter para decimal

3. **Vinculação a ocorrências:**
   - Lincar com `ocorrencia_policial_operacional.json` via Data+Local

**Esforço:** 5-7 dias

**Código necessário:**
```python
def parse_tropa_narrative(narrative):
    """Extrai local, datas, armas, drogas de texto narrativo"""
    # Regex patterns
    # Conversão DMS -> decimal
    # Busca por palavras-chave
    pass

def normalize_tropa_coordinates(lat_str):
    """Converte DMS para coordenadas decimais"""
    pass

def link_tropa_to_operational(df_tropa, df_operational):
    """Vincula prisões às ocorrências operacionais"""
    pass
```

---

#### ✅ Requisito 4: Correlações Ocorrência → Território → Impacto de Prisões
**Status:** VIÁVEL  
**Implementação:**

1. **Fase 1: Feature Engineering**
   - Adicionar `is_cvli` como feature prioritária (weight 3x)
   - Adicionar `has_drugs_1kg` (True se drogas ≥ 1000g)
   - Adicionar `has_weapons_and_drugs` (True se ambos)
   - Adicionar `arrested_count` por território/dia

2. **Fase 2: Ponderação da Topologia**
   ```python
   # Para cada edge:
   edge_weight = base_weight
   if crime_is_cvli: edge_weight *= 3.0
   if has_large_drug_seizure: edge_weight *= 2.0
   if arrest_in_territory: edge_weight *= 1.5
   ```

3. **Fase 3: Modelo Ajustado**
   ```python
   # No STGCN_Cpraio.forward():
   # Usar edge_weight na GCN
   # Adicionar term de regularização para "arrest_impact"
   ```

**Ação:** Modificar `model.py`, `graph_builder.py`, `trainer.py` (3-4 dias)

---

#### 🚨 Requisito 5: CVLI com Prioridade Suprema
**Status:** VIÁVEL  
**Desafio:** Identificar CVLI no dataset

**Possíveis nomes em `Natureza`:**
```
✅ Encontrados no JSON:
- "HOMICÍDIO"
- "TENTATIVA DE HOMICIDIO"
- "MORTE DECORRENTE..."
- "MORTE POR INTERVENÇÃO POLICIAL"
- "ESTUPRO"
- "ROUBO" (crimes violentos)
- "LESÃO CORPORAL"
```

**Implementação:**
```python
CVLI_KEYWORDS = [
    'HOMICÍDIO', 'MORTE', 'ESTUPRO', 'ROUBO DE VEÍCULO', 'LESÃO CORPORAL'
]

def is_cvli(natureza_str):
    return any(kw in natureza_str.upper() for kw in CVLI_KEYWORDS)
```

**Ação:** Criar dicionário de CVLI em `config.py` (1 dia)

---

#### ✅ Requisito 6: Drogas ≥ 1kg Influenciam Território
**Status:** ✅ IMPLEMENTÁVEL  
**Campo já existe:** `total_drogas_cache`

```python
# No feature engineering:
df['large_drug_seizure'] = df['total_drogas_cache'] >= 1000  # em gramas

# No graph_builder:
# Aumentar peso de edges para territórios com apreensões grandes
```

**Ação:** 1-2 dias

---

#### ✅ Requisito 7: Armas + Drogas Influenciam Territorio
**Status:** ✅ IMPLEMENTÁVEL  
**Campo já existe:** `total_armas_cache`

```python
# No feature engineering:
df['weapons_and_drugs'] = (df['total_armas_cache'] > 0) & (df['total_drogas_cache'] > 0)

# No graph_builder:
# Criar edges especiais entre territórios com essa combinação
```

**Ação:** 1-2 dias

---

## 📊 ESTRUTURA DE DADOS NECESSÁRIA

### Schema Consolidado Recomendado:

```python
df_unified = pd.DataFrame({
    'id': str,                      # Unique ID
    'data': datetime,               # Data da ocorrência
    'municipio': str,               # Município
    'bairro': str,                  # Bairro
    'lat': float,                   # Latitude
    'long': float,                  # Longitude
    
    # Tipo de crime
    'natureza': str,                # Descrição da ocorrência
    'is_cvli': bool,                # ✅ PRIORIDADE
    'categoria_crime': str,         # Classificação
    
    # Armas e Drogas
    'total_armas': int,             # Quantidade de armas
    'total_drogas_g': float,        # Drogas em gramas
    'has_large_seizure': bool,      # >= 1000g
    'has_weapons_drugs': bool,      # Ambos presentes
    'dinheiro_apreendido': float,   # R$ confiscados
    
    # Facções
    'area_faccao': str,             # CV, GDE, PCC, etc
    'territorio_id': str,           # FK para geometria
    
    # Prisões/Ações
    'num_presos': int,              # Quantidade de detidos
    'equipe_origem': str,           # Base/Raio de origem
    'base_raio': str,               # RAIO-XX ou unidade específica
    'fonte': str,                   # 'operacional' ou 'tropa'
})
```

---

## 🛠️ PLANO DE IMPLEMENTAÇÃO

### Fase 1: Normalização de Dados (5-7 dias)

**Task 1.1: Limpeza e Estruturação do JSON Operacional**
```
Entrada: ocorrencia_policial_operacional.json
Saída: df_operational.parquet

- Extrair corretamente lat/long do campo "lat_long"
- Mapear "area_faccao" para territórios
- Identificar CVLI por "Natureza"
- Normalizar unidades (drogas em gramas)
```

**Task 1.2: Parsing ocorrencias_tropa.json**
```
Entrada: ocorrencias_tropa.json
Saída: df_prisoes.parquet

- Extrair narrativa estruturada
- Converter DMS → decimal
- Vincular a data/local
- Contar presos e apreensões
```

**Task 1.3: Integração com Territorios**
```
Entrada: GeoJSON + dados
Saída: df_unified com territorio_id

- Spatial join lat/long → geometria GeoJSON
- Validar cobertura territorial
```

### Fase 2: Feature Engineering (3-4 dias)

**Task 2.1: Criar Features Compostas**
```python
# No data_loader.py
- is_cvli: detectar crimes violentos
- has_large_seizure: drogas >= 1kg
- has_weapons_drugs: arma + droga conjuntamente
- arrest_impact: normalizar presos por área
```

**Task 2.2: Implementar Ponderações**
```python
# No graph_builder.py
- Aumentar pesos de edges para CVLI (3x)
- Aumentar para apreensões grandes (2x)
- Aumentar para arma+droga (2x)
- Considerar recency das prisões
```

### Fase 3: Modificações do Modelo (3-4 dias)

**Task 3.1: Estender Tensor de Features**
```python
# Antes: X_base (num_days, num_nodes, 1) [apenas contagens]
# Depois: X_extended (num_days, num_nodes, K) onde K inclui:
#   - total_crimes
#   - cvli_count
#   - drug_seizures_total
#   - weapons_count
#   - arrest_count
#   - arrest_drug_value
#   - territory_stability_score
```

**Task 3.2: Atualizar Graph Builder**
```python
# Usar edge_weights na GCN
gcn_with_weights(x, edge_index, edge_weight)
```

**Task 3.3: Retraining**
```python
# Usar novo dataset estruturado
# Aumentar épocas (200 → 250)
# Aplicar weights balanceados
```

### Fase 4: Validação (3-5 dias)

**Task 4.1: Teste Preditivo 2025**
```
- Treinar com dados Jan-Ago 2025
- Prever Set-Out-Nov 2025
- Comparar com real ocorrências
- Calcular RMSE, MAE, R²
```

**Task 4.2: Análise de Impacto**
```
- Correlação entre prisões → redução de crimes
- Efeito de CVLI nas predições
- Validação de territórios faccionados
```

---

## 📈 ESTIMATIVAS E CRONOGRAMA

| Fase | Atividade | Dias | Responsável | Prioridade |
|------|-----------|------|-------------|-----------|
| 1.1  | JSON Operational | 2 | Data Engineer | 🔴 CRÍTICA |
| 1.2  | Tropa Parsing | 3 | Data Engineer | 🟡 ALTA |
| 1.3  | Territorial Integration | 2 | Data Engineer | 🟡 ALTA |
| 2.1  | Feature Engineering | 2 | Data Scientist | 🔴 CRÍTICA |
| 2.2  | Ponderações | 1 | ML Engineer | 🟡 ALTA |
| 3.1  | Extend Tensor | 1 | ML Engineer | 🔴 CRÍTICA |
| 3.2  | Graph Update | 1 | ML Engineer | 🟡 ALTA |
| 3.3  | Retraining | 2 | ML Engineer | 🔴 CRÍTICA |
| 4.1  | Teste Preditivo | 2 | Data Scientist | 🔴 CRÍTICA |
| 4.2  | Impact Analysis | 1 | Analyst | 🟡 ALTA |
| **TOTAL** | | **17-20 dias** | — | — |

---

## 🎯 ROADMAP RECOMENDADO

### Sprint 1 (5-7 dias): Dados
```
✅ Data Preparation
  ├─ ocorrencia_policial_operacional.json → parquet
  ├─ ocorrencias_tropa.json → parquet
  └─ GeoJSON integration
  
Output: df_unified.parquet (pronto para treinamento)
```

### Sprint 2 (3-4 dias): Features
```
✅ Feature Engineering
  ├─ CVLI classification
  ├─ Drug seizure detection
  ├─ Weapon+Drug correlation
  └─ Arrest impact scoring
  
Output: Tensor X_extended (num_days, num_nodes, K)
```

### Sprint 3 (3-4 dias): Model
```
✅ Model Adaptation
  ├─ GCN com edge weights
  ├─ Tensor extension
  └─ Retraining loop
  
Output: model_capital_v2.pth (novo modelo)
```

### Sprint 4 (2-3 dias): Validation
```
✅ Backtesting & Analysis
  ├─ 2025 validation split
  ├─ Correlation analysis
  └─ Impact report
  
Output: RELATORIO_VALIDACAO.md
```

---

## ⚠️ RISCOS E MITIGAÇÕES

| Risco | Impacto | Probabilidade | Mitigação |
|-------|---------|---------------|-----------|
| **Ocorrencias_tropa.json mal estruturado** | 🔴 CRÍTICO | 60% | Parsing robusto com fallbacks; manual review amostra |
| **Falta de histórico de prisões em 2025** | 🟡 MÉDIO | 40% | Usar dados operacionais como proxy; validar com CI/CIS |
| **Geolocalização imprecisa** | 🟡 MÉDIO | 30% | Usar spatial tolerance; validação manual de outliers |
| **CVLI definição ambígua** | 🟡 MÉDIO | 50% | Usar decreto/definição oficial CBTU; validar com especialista |
| **Overfitting em CVLI** | 🟡 MÉDIO | 35% | L2 regularization; cross-validation estratificada |

---

## ✅ CHECKLIST PRÉ-IMPLEMENTAÇÃO

- [ ] **Validação de CVLI:** Confirmar lista de naturezas com delegado/especialista
- [ ] **Validação de Territorios:** Confirmar GeoJSON facções com inteligência
- [ ] **Acesso a Dados:** Verificar acesso aos datasets mencionados
- [ ] **Capacidade Computacional:** Validar GPUs/CPU para retraining (250 épocas)
- [ ] **Dados de Teste:** Reservar dados reais de Jan-Ago 2025 para validação
- [ ] **Aprovação de Stakeholders:** Alinhar com CPRAIO sobre métricas de sucesso

---

## 📋 CONCLUSÃO

### ✅ O que é VIÁVEL:

1. **Integrar drogas/armas** como features (campos já existem)
2. **Priorizar CVLI** com ponderação 3x (fácil de implementar)
3. **Correlacionar com prisões** (requer normalização de dados)
4. **Mapear territorios faccionados** (GeoJSON disponível)
5. **Validar eficácia** com dados reais 2025 (backtest possível)

### ⚠️ O que requer esforço:

1. **Parsing de texto narrativo** (ocorrencias_tropa.json)
2. **Conversão de coordenadas DMS** → decimal
3. **Estruturação de novo tensor** com K features
4. **Novo ciclo de treinamento** (250 épocas)

### 🎯 Recomendação Final:

**PROSSEGUIR COM IMPLEMENTAÇÃO** em 4 sprints paralelos.  
Cronograma: **17-20 dias úteis** para modelo funcional.  
Validação: **2-3 dias adicionais** para backtesting completo.

**Total: ~25 dias para modelo em produção.**

---

## 📞 PRÓXIMOS PASSOS

1. **Aprovação deste plano** com CPRAIO/Gestão
2. **Alocação de recursos** (Data Engineers + ML Engineers)
3. **Kick-off Sprint 1** com entrega de `df_unified.parquet`
4. **Daily standups** (Sprint meetings 2x semana)
5. **Entrega final** com relatório de impacto e modelos treinados

---

**Relatório preparado por:** AI Assistant  
**Versão:** 1.0  
**Data:** 2026-01-19
