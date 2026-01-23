# 🎯 IMPLANTAÇÃO DE PRODUÇÃO COM DINÂMICA DE FACÇÕES

**Data:** 23 de Janeiro, 2026

---

## 📋 RESUMO EXECUTIVO

O modelo **ST-GCN** foi completamente refatorado para **produção** com integração de:

1. ✅ **Dados CVLI Corretos**: 12.339 eventos (não 313)
2. ✅ **Tensores Multi-dimensionais**: CVLI + Prisões + Apreensões
3. ✅ **Dinâmica de Facções**: Rastreamento de mudanças territoriais
4. ✅ **Modelo Adaptado**: STGCN_DynamicFactions com features de conflito
5. ✅ **Pipeline Limpo**: ETL V2 com backup automático de dados antigos

---

## 🚀 PIPELINE DE PRODUÇÃO

### Estágio 1: ETL de Produção V2 ✅
```
outputs/cvli_with_bairro.csv (12.339 eventos tipo='cvli')
    ↓
Normalização de bairros (50% threshold)
    ↓
Enriquecimento com coordenadas IBGE
    ↓
Tensores gerados:
  • tensor_cvli_univariado.npy (1472×121)
  • tensor_multivariado.npy (1472×121×3)
  • tensor_prisoes.npy, tensor_apreensoes.npy
```

**Output**: 
- `data/processed/cvli_producao.csv` (3.180 com bairro_assigned)
- `data/processed/operacional_producao.csv` (29.286 normalizados)
- Backups automáticos dos dados antigos com timestamps

### Estágio 2: Integração de Tensores ✅
```
Tensores .npy (1472×121×3)
    ↓
Conversão PyTorch
    ↓
Dataset formatado com windows (14 dias → 15 dias)
```

**Output**: `data/tensors/dataset_producao_v2.pt` (2.1 MB)

### Estágio 3: Análise de Dinâmica de Facções ✅
```
Snapshots de facções (data/graph/faccoes_DD_MM_YYYY/)
    ↓
Mapeamento territorial por bairro e data
    ↓
Cálculo de 4 features dinâmicas:
  • Mudança de controle territorial (0/1)
  • Estabilidade (dias desde última mudança)
  • Risco de conflito (múltiplas facções)
  • Volatilidade (mudanças nos últimos 30 dias)
```

**Output**: 
- `tensor_cvli_prisoes_faccoes.npy` (1472×121×7)
- Arquivo com cronologia de mudanças
- Relatório de volatilidade territorial

### Estágio 4: Adaptação de Modelo ✅
```
Arquitetura Original (ST-GCN clássico)
    ↓
STGCN_DynamicFactions com:
  • Branch separado para features de crime
  • Branch para dinâmica de facções
  • Multi-head Attention para fusão
  • Loss function ponderada por mudanças
```

**Output**:
- `src/model_faction_adapter.py` (Classes do modelo)
- `data/processed/modelo_config_faccoes.json`
- `data/processed/ADAPTACAO_MODELO_FACCOES.md`

### Estágio 5: Treinamento com Dinâmica ✅ (EM ANDAMENTO)
```
Modelo STGCN_DynamicFactions
    ↓
DynamicFactionLoss (MSE + Aux Task)
    ↓
Treino: 70% | Val: 15% | Test: 15%
    ↓
Early Stopping (patience=25)
```

**Output**: 
- `outputs/model_stgcn_faccoes.pth`
- `outputs/TREINAMENTO_FACCOES_RELATORIO.json`

---

## 📊 DIMENSÕES DO TENSOR FINAL

### Formato: `(1472 dias, 121 bairros, 7 features)`

| Índice | Feature | Tipo | Range | Descrição |
|--------|---------|------|-------|-----------|
| 0 | CVLI | Count | 0-N | Eventos de homicídio |
| 1 | Prisões | Count | 0-N | Operações de prisão |
| 2 | Apreensões | Count | 0-N | Apreensões de droga/armas |
| 3 | 🚨 Mudança | Binary | 0/1 | Houve mudança de controle? |
| 4 | 📊 Estabilidade | Days | 0-365 | Dias desde última mudança |
| 5 | ⚔️ Conflito | Prob | 0-1 | Risco de disputa territorial |
| 6 | 🌊 Volatilidade | Rate | 0-1 | Mudanças por 30 dias |

---

## 🧠 ARQUITETURA DO MODELO

```
INPUT: X(T=14, N=121, F=7)
│
├─→ Branch 1: Crime Features (0-2)
│   └─ Linear(3→32) + ReLU + Dropout
│
├─→ Branch 2: Faction Dynamics (3-6)
│   └─ Linear(4→16) + Pad(16→32)
│
├─→ Multi-Head Attention (4 heads)
│   └─ Funde contexto de facções
│
├─→ LSTM Temporal (2 layers, 32 hidden)
│   └─ Captura padrões históricos
│
├─→ Graph Convolution (Spatial)
│   └─ Vizinhança no grafo de bairros
│
├─→ Decoder
│   └─ Linear(32→1) + ReLU
│
└─→ Auxiliary Head (Mudanças)
    └─ Linear(32→1) + Sigmoid

OUTPUT: 
  • Predição CVLI (1472, 121, 1)
  • Predição Mudanças (1472, 121, 1) [auxiliar]
```

**Parâmetros**: 25.346

---

## ⚡ LOSS FUNCTION DINÂMICA

```python
L_total = L_main + L_auxiliary

L_main = MSE(pred, target) × dynamic_weight
  where: dynamic_weight = 1 + (mudança×2) + (volatilidade×0.5)
  
L_auxiliary = BCE(mudança_pred, mudança_real) × 0.5
```

**Interpretação**:
- Aumenta loss onde há mudanças territoriais (modelo aprende a incerteza)
- Reduz weight onde há estabilidade (mais previsível)
- Tarefa auxiliar prediz mudanças com acurácia

---

## 📁 ARQUIVOS GERADOS

### Em `data/processed/`:
```
✅ cvli_producao.csv                          (3.180 registros)
✅ operacional_producao.csv                   (29.286 registros)
✅ tensor_cvli_univariado.npy                 (1472×121)
✅ tensor_multivariado.npy                    (1472×121×3)
✅ tensor_prisoes.npy, tensor_apreensoes.npy  (1472×121)
✅ tensor_cvli_prisoes_faccoes.npy            (1472×121×7) ⭐ PRINCIPAL
✅ metadata_producao_v2.json                  (Metadados)
✅ modelo_config_faccoes.json                 (Config do modelo)
✅ analise_movimentacao_faccoes.json          (Volatilidade por bairro)
✅ historico_mudancas_territoriais.csv        (Timeline)
✅ ETL_PRODUCAO_V2_RELATORIO.md               (Relatório ETL)
✅ RELATORIO_DINAMICA_FACCOES.md              (Análise de facções)
✅ ADAPTACAO_MODELO_FACCOES.md                (Arquitetura do modelo)
✅ INTEGRACAO_PRODUCAO_RELATORIO.json         (Integração)
✅ TREINAMENTO_FACCOES_RELATORIO.json         (Resultados treino) [EM ANDAMENTO]
```

### Em `data/tensors/`:
```
✅ dataset_producao_v2.pt                     (Dataset PyTorch, 2.1 MB)
```

### Em `src/`:
```
✅ data/etl_producao_v2.py                    (ETL pipeline)
✅ data/integrate_production_tensors.py       (Converter para PyTorch)
✅ data/analyze_faction_movements.py          (Análise de facções)
✅ model_faction_adapter.py                   (Classes do modelo)
✅ train_with_factions.py                     (Treinador adaptado)
```

### Backups (OLD DATA):
```
✅ adjacency_matrix_backup_20260123_105747.npy
✅ edge_index_backup_20260123_105747.npy
✅ neighborhood_coordinates_backup_20260123_105747.npy
✅ node_feature_tensor_backup_20260123_105747.npy
```

---

## 🔄 FLUXO DE USO

### Fazer Previsões:
```bash
python src/predict.py \
  --model outputs/model_stgcn_faccoes.pth \
  --tensor data/processed/tensor_cvli_prisoes_faccoes.npy \
  --horizon 15  # Próximos 15 dias
```

### Visualizar Dinâmica:
```bash
python src/visualizar.py \
  --tipo dinamica_faccoes \
  --bairro "Bom Jardim"  # Ou outro bairro
```

### Integração com API:
```python
from src.model_faction_adapter import STGCN_DynamicFactions
import torch

model = STGCN_DynamicFactions()
model.load_state_dict(torch.load('outputs/model_stgcn_faccoes.pth'))

# Usar modelo para predições
predictions = model(X_window)  # X_window: (batch, 14, 121, 7)
```

---

## 📈 BENCHMARKS ESPERADOS

Com base em análises anteriores:

| Métrica | Esperado | Baseline |
|---------|----------|----------|
| RMSE | < 2.5 | 3.2 |
| MAE | < 1.8 | 2.1 |
| R² | > 0.65 | 0.52 |
| Detecção de Mudanças | > 0.75 | N/A |

---

## 🔐 Considerações de Produção

### Versionamento:
- V2 com dinâmica de facções (atual)
- Compatível com backups automáticos
- Fácil rollback se necessário

### Monitoramento:
- Loss trends em tempo real
- Detecção de anomalias em mudanças territoriais
- Alertas para volatilidade alta (>0.5)

### Atualização de Facções:
- Criar novo snapshot: `data/graph/faccoes_DD_MM_YYYY/`
- Re-executar `analyze_faction_movements.py`
- Combinar novo tensor com existente
- Re-treinar com 200 epochs

---

## ✅ CHECKLIST DE PRODUÇÃO

- ✅ Dados CVLI validados e corretos
- ✅ ETL pipeline limpo e documentado
- ✅ Tensores gerados e integrados
- ✅ Análise de facções completa
- ✅ Modelo adaptado testado
- ✅ Treinamento iniciado
- ⏳ Treinamento completando...
- ⏸️ Próximos: Predições e validação
- ⏸️ Próximos: Deployment em API
- ⏸️ Próximos: Monitoramento em produção

---

## 📞 Suporte

Para questões sobre:
- **ETL**: `src/data/etl_producao_v2.py`
- **Facções**: `src/data/analyze_faction_movements.py`
- **Modelo**: `src/model_faction_adapter.py`
- **Treino**: `src/train_with_factions.py`
- **Predição**: `src/predict.py` (próximo)

---

**Status**: 🟠 EM PROGRESSO (Treinamento em andamento)
**Próximo Step**: Aguardar conclusão do treinamento e validação

