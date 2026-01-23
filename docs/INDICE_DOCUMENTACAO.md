# 📚 ÍNDICE COMPLETO - ST-GCN COM DINÂMICA DE FACÇÕES

**Versão:** 2.0 | **Data:** 23/01/2026 | **Status:** ✅ Produção

---

## 🗂️ DOCUMENTAÇÃO

### 1. 📖 Documentos de Início Rápido

#### [CONCLUSAO_FINAL.md](CONCLUSAO_FINAL.md)
- Sumário do que foi entregue
- Como usar agora
- Próximas ações
- **Leia primeiro se tiver pouco tempo**

#### [RESUMO_VISUAL.md](RESUMO_VISUAL.md)
- Visualização da arquitetura
- Arquivos gerados
- Performance
- Checklist de validação
- **Excelente para stakeholders**

### 2. 🚀 Documentos de Deployment

#### [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- Instalação completa
- Cenários de uso
- Docker (opcional)
- Monitoramento
- CI/CD pipeline
- **Essencial para DevOps**

#### [IMPLANTACAO_COMPLETA_FACCOES.md](IMPLANTACAO_COMPLETA_FACCOES.md)
- Implantação 360°
- Arquivos gerados
- Considerações de produção
- Troubleshooting
- **Referência técnica completa**

### 3. 🔬 Documentos Técnicos

#### [PRODUCAO_COM_FACCOES_SUMARIO.md](PRODUCAO_COM_FACCOES_SUMARIO.md)
- Pipeline de produção (6 estágios)
- Dimensões do tensor (7D)
- Arquitetura do modelo
- Loss function dinâmica
- **Para engenheiros**

#### [data/processed/ADAPTACAO_MODELO_FACCOES.md](data/processed/ADAPTACAO_MODELO_FACCOES.md)
- Arquitetura neural detalhada
- Flow do treinamento
- Benefícios da adaptação
- **Para pesquisadores**

#### [data/processed/RELATORIO_DINAMICA_FACCOES.md](data/processed/RELATORIO_DINAMICA_FACCOES.md)
- Análise de movimentação de facções
- Cronologia de mudanças
- Bairros com maior volatilidade
- **Para inteligência operacional**

---

## 💻 SCRIPTS

### Pipeline ETL

#### [src/data/etl_producao_v2.py](src/data/etl_producao_v2.py)
```python
# 7 stages: Load → Normalize → Enrich → Tensor → Save → Integration → Report
# Execução: python src/data/etl_producao_v2.py
# Outputs: tensor_*.npy, *.csv, metadata
```

#### [src/data/integrate_production_tensors.py](src/data/integrate_production_tensors.py)
```python
# Converte numpy arrays para PyTorch datasets
# Execução: python src/data/integrate_production_tensors.py
# Output: dataset_producao_v2.pt (2.1 MB)
```

#### [src/data/analyze_faction_movements.py](src/data/analyze_faction_movements.py)
```python
# Processa GeoJSONs de facções e cria features dinâmicas
# Execução: python src/data/analyze_faction_movements.py
# Outputs: tensor com 7D, análise de movimentação
```

### Modelo & Treinamento

#### [src/model_faction_adapter.py](src/model_faction_adapter.py)
```python
# Define STGCN_DynamicFactions e DynamicFactionLoss
# Classes: STGCN_DynamicFactions, DynamicFactionLoss
# Parâmetros: 25.346
```

#### [src/train_with_factions.py](src/train_with_factions.py)
```python
# Treinador com dinâmica de facções
# Execução: python src/train_with_factions.py
# Output: model_stgcn_faccoes.pth
```

#### [src/predict_with_factions.py](src/predict_with_factions.py)
```python
# Preditor e gerador de relatórios
# Execução: python src/predict_with_factions.py
# Outputs: CSV, JSON, Markdown reports
```

### Validação

#### [validate_deployment.py](validate_deployment.py)
```python
# Verifica se tudo está funcionando
# Execução: python validate_deployment.py
# Checks: arquivos, dependências, modelo, dados
```

---

## 📊 DADOS

### Entrada (Raw)

```
outputs/cvli_with_bairro.csv
└─ 12.339 eventos CVLI (tipo='cvli')
   └─ Campos: id, data, bairro_assigned, latitude, longitude, tipo

data/raw/View_Ocorrencias_Operacionais_Modelo.csv
└─ 34.270 registros operacionais
   └─ Prisões, Apreensões, Drogas, Armas
```

### Processado (Intermediate)

```
data/processed/
├─ tensor_cvli_univariado.npy (1472×121)
├─ tensor_multivariado.npy (1472×121×3)
├─ tensor_prisoes.npy (1472×121)
├─ tensor_apreensoes.npy (1472×121)
├─ cvli_producao.csv (3.180 registros)
├─ operacional_producao.csv (29.286 registros)
└─ metadata_producao_v2.json
```

### Principal (Output)

```
data/processed/
└─ tensor_cvli_prisoes_faccoes.npy ⭐ (1472×121×7)

data/tensors/
└─ dataset_producao_v2.pt (2.1 MB)

outputs/
├─ model_stgcn_faccoes.pth ⭐ (100 KB)
├─ predicoes_cvli.csv ⭐ (121 bairros)
├─ predicoes_cvli.json ⭐ (Estruturado)
└─ RELATORIO_PREDICOES.md ⭐ (Executivo)
```

---

## 🗺️ ESTRUTURA DE PASTAS

```
projeto-stgcn-cpraio/
│
├─ data/
│  ├─ raw/                          # Dados brutos
│  ├─ processed/                    # ⭐ Tensores processados
│  ├─ graph/faccoes_DD_MM_YYYY/    # ⭐ Snapshots de facções
│  └─ tensors/dataset_producao_v2.pt
│
├─ src/
│  ├─ data/
│  │  ├─ etl_producao_v2.py        # ETL
│  │  ├─ integrate_production_tensors.py
│  │  └─ analyze_faction_movements.py
│  ├─ model_faction_adapter.py      # ⭐ Modelo
│  ├─ train_with_factions.py        # Treino
│  ├─ predict_with_factions.py      # ⭐ Predição
│  ├─ config.py
│  └─ [outros módulos]
│
├─ outputs/
│  ├─ model_stgcn_faccoes.pth       # ⭐ Modelo
│  ├─ predicoes_cvli.*              # ⭐ Resultados
│  └─ [relatórios e análises]
│
├─ IMPLANTACAO_COMPLETA_FACCOES.md
├─ DEPLOYMENT_GUIDE.md
├─ PRODUCAO_COM_FACCOES_SUMARIO.md
├─ RESUMO_VISUAL.md
├─ CONCLUSAO_FINAL.md
├─ validate_deployment.py
└─ requirements.txt
```

---

## 📋 COMO NAVEGAR

### Se você quer...

#### ✅ Começar rápido
1. Leia [CONCLUSAO_FINAL.md](CONCLUSAO_FINAL.md)
2. Execute: `python src/predict_with_factions.py`
3. Verifique: `outputs/RELATORIO_PREDICOES.md`

#### ✅ Fazer deploy em produção
1. Consulte [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
2. Siga o checklist de deployment
3. Configure CI/CD pipeline

#### ✅ Entender a arquitetura
1. Estude [PRODUCAO_COM_FACCOES_SUMARIO.md](PRODUCAO_COM_FACCOES_SUMARIO.md)
2. Revise [data/processed/ADAPTACAO_MODELO_FACCOES.md](data/processed/ADAPTACAO_MODELO_FACCOES.md)
3. Examine `src/model_faction_adapter.py`

#### ✅ Retreinar o modelo
1. Atualize snapshot de facções em `data/graph/faccoes_DD_MM_YYYY/`
2. Execute: `python src/data/analyze_faction_movements.py`
3. Execute: `python src/train_with_factions.py`
4. Execute: `python src/predict_with_factions.py`

#### ✅ Troubleshooting
1. Consulte [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md#-troubleshooting)
2. Verifique [IMPLANTACAO_COMPLETA_FACCOES.md](IMPLANTACAO_COMPLETA_FACCOES.md#-considerações-importantes)
3. Execute: `python validate_deployment.py`

---

## 🎯 ROADMAP

### Immediate (Done ✅)
- [x] ETL pipeline V2
- [x] Integração de tensores
- [x] Análise de facções
- [x] Modelo adaptado
- [x] Treinamento
- [x] Predições
- [x] Documentação

### Next 30 days
- [ ] API deployment
- [ ] Dashboard integration
- [ ] Alert system setup
- [ ] Performance monitoring

### Next 90 days
- [ ] New faction snapshots
- [ ] Model retraining
- [ ] Ensemble methods
- [ ] Feature expansion

---

## 🔗 REFERÊNCIAS CRUZADAS

### Por Componente

**ETL:**
- Script: `src/data/etl_producao_v2.py`
- Documentação: `IMPLANTACAO_COMPLETA_FACCOES.md#estágio-1`
- Report: `data/processed/ETL_PRODUCAO_V2_RELATORIO.md`

**Facções:**
- Script: `src/data/analyze_faction_movements.py`
- Documentação: `PRODUCAO_COM_FACCOES_SUMARIO.md#estágio-3`
- Report: `data/processed/RELATORIO_DINAMICA_FACCOES.md`

**Modelo:**
- Script: `src/model_faction_adapter.py`
- Documentação: `data/processed/ADAPTACAO_MODELO_FACCOES.md`
- Arquitetura: `PRODUCAO_COM_FACCOES_SUMARIO.md#arquitetura-do-modelo`

**Predição:**
- Script: `src/predict_with_factions.py`
- Output: `outputs/RELATORIO_PREDICOES.md`
- Como usar: `DEPLOYMENT_GUIDE.md#fazer-uma-predição-rápida`

---

## 📞 SUPORTE

| Tópico | Arquivo | Seção |
|--------|---------|-------|
| Instalação | DEPLOYMENT_GUIDE.md | Installation |
| Uso | CONCLUSAO_FINAL.md | Como usar agora |
| API | DEPLOYMENT_GUIDE.md | Integração em API |
| Monitoramento | DEPLOYMENT_GUIDE.md | Monitoramento |
| Troubleshooting | DEPLOYMENT_GUIDE.md | Troubleshooting |
| Arquitetura | ADAPTACAO_MODELO_FACCOES.md | Completo |
| Performance | RESUMO_VISUAL.md | Performance |

---

## ✅ Checklist de Leitura

- [ ] Leia CONCLUSAO_FINAL.md (5 min)
- [ ] Revise RESUMO_VISUAL.md (10 min)
- [ ] Estude PRODUCAO_COM_FACCOES_SUMARIO.md (20 min)
- [ ] Consulte DEPLOYMENT_GUIDE.md (para seu use case)
- [ ] Examine scripts em src/ (30 min)

---

**Pronto para explorar? Comece por [CONCLUSAO_FINAL.md](CONCLUSAO_FINAL.md)** 🚀

